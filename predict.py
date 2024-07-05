import asyncio
import contextlib
import inspect
import json
import multiprocessing as mp
import os
import re
import subprocess
import time
from typing import AsyncIterator, Optional

import cog
from cog import BasePredictor, AsyncConcatenateIterator, Input

if mp.current_process().name != "MainProcess":
    import httpx
    import numpy as np
    import pytriton.utils.distribution
    from transformers import AutoTokenizer

    from sse import receive_sse
    from triton_config_generator import generate_configs, load_yaml_config
    from utils import (
        StreamingTokenStopSequenceHandler,
        maybe_download_tarball_with_pget,
    )

    TRITONSERVER_DIST_DIR = (
        pytriton.utils.distribution.get_root_module_path() / "tritonserver"
    )
    TRITONSERVER_BACKEND_DIR = os.getenv(
        "TRITONSERVER_BACKEND_DIR", str(TRITONSERVER_DIST_DIR / "backends")
    )

TRITON_START_TIMEOUT_MINUTES = 5
TRITON_TIMEOUT = 120


class UserError(Exception):
    pass


class TritonError(Exception):
    pass


def format_prompt(
    prompt: str, prompt_template: str, system_prompt: Optional[str]
) -> str:
    if not prompt_template:
        prompt_template = "{prompt}"
    if prompt and "{prompt}" not in prompt_template:
        raise UserError(
            "E1003 BadPromptTemplate: You have submitted both a prompt and a prompt template that doesn't include '{prompt}'."
            "Your prompt would not be used. "
            "If don't want to use formatting, use your full prompt for the prompt argument and set prompt_template to '{prompt}'."
        )
    try:
        return prompt_template.format(
            system_prompt=system_prompt or "",
            prompt=prompt,
        )
    except (ValueError, KeyError, IndexError) as e:
        # sometimes people put the prompt in prompt_template
        if len(prompt_template) > len(prompt):
            raise UserError(
                "E1004 PromptTemplateError: Prompt template must be a valid python format spec. "
                "Did you submit your prompt as `prompt_template` instead of `prompt`? "
                'If you want finer control over templating, set prompt_template to `"{prompt}"` to disable formatting. '
                "You can't put JSON in prompt_template, because braces will be parsed as a python format string. "
                f"Detail: {repr(e)}"
            )
        # most common case is "unmatched '{' in format spec",
        # but IndexError/KeyError and other formatting errors can happen
        # str(KeyError) is only the missing key which can be confusing
        raise UserError(
            f"E1004 PromptTemplateError: Prompt template must be a valid python format spec: {repr(e)}"
        )


@contextlib.asynccontextmanager
async def wrap_httpx_error(
    req: contextlib._AsyncGeneratorContextManager,
) -> AsyncIterator["httpx.Response"]:
    try:
        async with req as resp:
            yield resp
    except httpx.ReadTimeout:
        raise TritonError(
            f"E2101 TritonTimeout: Triton timed out after {TRITON_TIMEOUT}s: httpx.ReadTimeout. "
            "This can happen for extremely long prompts or large batches. "
            "Try a shorter prompt, or sending requests more slowly."
        )


prompt_too_long_pattern = re.compile(
    r"[Pp]rompt length \(\d+\) exceeds maximum input length \(\d+\)"
)


def parse_triton_error(error_message: str) -> Exception:
    if match := prompt_too_long_pattern.search(error_message):
        raise UserError(f"E1002 PromptTooLong: {match.group()}")
    if "the first token of the stop sequence IDs was not" in error_message:
        raise TritonError(
            f"E2102 TritonTokenizerError: Tokenizer error: {error_message}"
        )
    raise TritonError(f"E2100 TritonUnknownError: Unknown Triton error: {error_message}")


class Predictor(BasePredictor):
    async def setup(self, weights: str = "") -> None:
        self.log_performance_metrics = bool(os.getenv("LOG_PERFORMANCE_METRICS", False))

        COG_TRITON_CONFIG = os.getenv("COG_TRITON_CONFIG", "config.yaml")
        if not os.path.exists(COG_TRITON_CONFIG):
            print(f"Config file {COG_TRITON_CONFIG} not found. Defaults will be used.")
        else:
            print(f"Loading cog-triton config from {COG_TRITON_CONFIG}")
            config = load_yaml_config(COG_TRITON_CONFIG)
            print("----------------------")
            print("cog-triton config:")
            print(config)
            print("----------------------")
            if "server" in config["instantiate"]:
                generate_configs(config["instantiate"]["server"])

        engine_dir = os.environ.get(
            "ENGINE_DIR", "/src/triton_model_repo/tensorrt_llm/1"
        )

        self.system_prompt_exists = os.getenv("SYSTEM_PROMPT", None)
        self.end_id = int(os.getenv("END_ID", 2))
        self.pad_id = int(os.getenv("PAD_ID", 2))

        if weights:
            self.log(f"Downloading model files from {weights}...")
            maybe_download_tarball_with_pget(
                url=weights,
                dest=engine_dir,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(engine_dir)

        with open(f"{engine_dir}/config.json", "r") as f:
            self.trt_llm_config = config = json.load(f)
            print(f"tensorrt_llm config: {config}")

        max_seqlen_env = os.getenv("MAX_SEQUENCE_LENGTH", None)
        if max_seqlen_env:
            self.max_sequence_length = int(max_seqlen_env)
        else:
            try:
                self.max_sequence_length = self.trt_llm_config["pretrained_config"][
                    "max_position_embeddings"
                ]
            except KeyError:
                self.log(
                    "`max_seq_len` not found in ENV and not found in `config.json. Not enforcing `max_seq_len`."
                )

        # if engine_dir is empty, stop here
        if not os.listdir(engine_dir):
            print("Engine directory is empty. Exiting.")
            self.model_exists = False
            return
        self.model_exists = True
        self.client = httpx.AsyncClient(timeout=TRITON_TIMEOUT)
        await self.ensure_triton_started()
        # we expect this to throw a timeout or some other error in the case of failures
        self._testing = True
        generator = self.predict(**(self._defaults | {"max_tokens": 3, "prompt": "hi"}))
        test_output = "".join([tok async for tok in generator])
        print("Test prediction output:", test_output)
        self._testing = False

    async def ensure_triton_started(self):
        for i in range(3):
            if await self.start_triton():
                return
        raise Exception(f"Couldn't start Triton (exit code {self.proc.poll()})")

    async def start_triton(self) -> bool:
        # # launch triton server
        # # python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/tensorrtllm_backend/triton_model
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        print("Starting Triton")
        cmd = ["mpirun", "--allow-run-as-root"]
        for i in range(world_size):
            cmd += [
                "-n",
                "1",
                str(TRITONSERVER_DIST_DIR / "bin" / "tritonserver"),
                "--backend-dir",
                TRITONSERVER_BACKEND_DIR,
                # "--log-verbose=3", "--log-file=triton_log.txt",
                "--model-repository",
                "/src/triton_model_repo",
                f"--backend-config=python,shm-region-prefix-name=prefix{i}_",
            ]
            if i != 0:
                cmd += ["--model-control-mode=explicit", "--load-model=tensorrt_llm"]
            cmd += [":"]
        self.proc = subprocess.Popen(cmd)
        # Health check Triton until it is ready or for 5 minutes
        for i in range(TRITON_START_TIMEOUT_MINUTES * 60):
            try:
                response = await self.client.get(
                    "http://localhost:8000/v2/health/ready"
                )
                if response.status_code == 200:
                    print("Triton is ready.")
                    return True
            except httpx.RequestError:
                pass
            await asyncio.sleep(1)
        print(f"Triton was not ready within {TRITON_START_TIMEOUT_MINUTES} minutes (exit code: {self.proc.poll()})")
        self.proc.terminate()
        await asyncio.sleep(0.001)
        self.proc.kill()
        return False

    async def predict(
        self,
        prompt: str = Input(description="Prompt to send to the model."),
        system_prompt: str = Input(
            description="System prompt to send to the model. This is prepended to the prompt and helps guide system behavior.",
            default=os.getenv("SYSTEM_PROMPT", ""),
        ),
        max_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens.",
            ge=1,
            default=512,
        ),
        min_tokens: int = Input(
            description="Minimum number of tokens to generate. To disable, set to -1. A word is generally 2-3 tokens.",
            ge=-1,
            default=None,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.0,
            le=5,
            default=0.7,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens.",
            ge=0.0,
            le=1.0,
            default=0.95,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens.",
            ge=-1,
            default=0,
        ),
        stop_sequences: str = Input(
            description="A comma-separated list of sequences to stop generation at. For example, '<end>,<stop>' will stop generation at the first instance of 'end' or '<stop>'.",
            default=os.getenv("STOP_SEQUENCES"),
        ),
        length_penalty: float = Input(
            description="A parameter that controls how long the outputs are. If < 1, the model will tend to generate shorter outputs, and > 1 will tend to generate longer outputs.",
            ge=0.0,
            le=5.0,
            default=1.0,
        ),
        presence_penalty: float = Input(
            description="A parameter that penalizes repeated tokens regardless of the number of appearances. As the value increases, the model will be less likely to repeat tokens in the output.",
            default=0.0,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed.",
            default=None,
        ),
        prompt_template: str = Input(
            description="Template for formatting the prompt. Can be an arbitrary string, but must contain the substring `{prompt}`.",
            default=os.getenv("PROMPT_TEMPLATE", "{prompt}"),
            min_length=1,
        ),
        log_performance_metrics: bool = False,
        max_new_tokens: int = Input(
            description="This parameter has been renamed to max_tokens. max_new_tokens only exists for backwards compatibility purposes. We recommend you use max_tokens instead. Both may not be specified.",
            ge=1,
            default=None,
        ),
        min_new_tokens: int = Input(
            description="This parameter has been renamed to min_tokens. min_new_tokens only exists for backwards compatibility purposes. We recommend you use min_tokens instead. Both may not be specified.",
            ge=-1,
            default=None,
        ),
    ) -> AsyncConcatenateIterator[str]:
        if not self.model_exists:
            self.log(
                "Your model directory is empty, so there's nothing to do. Remember, you can't run this like a normal model. You need to YOLO!"
            )
            return

        formatted_prompt = format_prompt(
            prompt=prompt, system_prompt=system_prompt, prompt_template=prompt_template
        )
        if formatted_prompt == "":
            raise UserError(
                "E1001 PromptRequired: A prompt is required, but your formatted prompt is blank"
            )

        # compatibility with older language models
        if max_new_tokens:
            # 512 is the default
            if max_tokens == 512 or max_tokens is None or max_tokens == max_new_tokens:
                max_tokens = max_new_tokens
            else:
                raise UserError(
                    f"E1102 InvalidArgumentMaxTokens: Can't set both max_tokens ({max_tokens}) and max_new_tokens ({max_new_tokens})"
                )
        if min_new_tokens:
            if min_tokens is None or min_tokens == min_new_tokens:
                min_tokens = min_new_tokens
            else:
                raise UserError(
                    f"E1101 InvalidArgumentMinTokens: Can't set both min_tokens ({min_tokens}) and min_new_tokens ({min_new_tokens})"
                )

        n_prompt_tokens = self._get_n_tokens(formatted_prompt)
        args = self._process_args(
            prompt=formatted_prompt,
            n_prompt_tokens=n_prompt_tokens,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            length_penalty=length_penalty,
            presence_penalty=presence_penalty,
            stop_words=stop_sequences,
            seed=seed,
        )

        req = self.client.stream(
            "POST",
            "http://localhost:8000/v2/models/ensemble/generate_stream",
            json=args,
        )

        output = ""
        generation_length = 0
        stop_sequence_handler = StreamingTokenStopSequenceHandler(
            stop_sequences=args["stop_words"]
        )

        start_time = time.time()
        n_tokens = 0
        tokens = np.array([], dtype=np.int32)
        first_token_time = None

        async with wrap_httpx_error(req) as resp:
            async for event in receive_sse(resp):
                # Output is the _entire_ sequence, from the beginning
                try:
                    event_data = event.json()
                except json.JSONDecodeError as e:
                    raise UserError(
                        f"E2103 TritonMalformedJSON: Triton returned malformed JSON: {event}"
                    ) from e
                if error_message := event_data.get("error"):
                    if "output_ids" in event_data:
                        self.log(
                            "output_ids and error are both set, this shouldn't happen"
                        )
                    raise parse_triton_error(error_message)
                if (token := event_data.get("output_ids", ...)) is ...:
                    raise TritonError(
                        f"E2104 TritonMalformedEvent: Triton returned malformed event (no output_ids or error key): {event_data}"
                    )

                n_tokens += 1
                if n_tokens == 1:
                    first_token_time = time.time()

                tokens = np.append(tokens, token)
                output = self.tokenizer.decode(tokens, skip_special_tokens=True)
                # Catches partial emojis, waits for them to finish
                output = output.replace("\N{REPLACEMENT CHARACTER}", "")
                # Remove the tokens that were already yielded
                current_output = output[generation_length:]

                if current_output:
                    # Process the output for stop sequences
                    current_output = stop_sequence_handler(current_output)
                    # If we have a partial stop sequence match or a full match,
                    # `current_output` will be `None` and we shouldn't yield
                    if current_output:
                        yield current_output

                # Update generation length
                generation_length = len(output)

            # Handles the case where the generation ends in the middle of a valid stop sequence
            current_output = stop_sequence_handler.finalize()
            if current_output:
                yield current_output

        end_time = time.time()
        if self.log_performance_metrics or log_performance_metrics:
            latency = end_time - start_time
            actual_tps = n_tokens / latency
            self.log(f"Tokens processed: {n_tokens}\n")
            self.log(f"Serverside tokens per second: {round(actual_tps, 2)}\n")
            self.log(f"Serverside execution time: {round(latency, 2)} seconds\n")
            if first_token_time:
                time_to_first_token = first_token_time - start_time
                self.log(
                    f"Serverside time to first token: {round(time_to_first_token, 2)} seconds\n"
                )
        if not self._testing:
            cog.emit_metric("input_token_count", n_prompt_tokens)
            cog.emit_metric("output_token_count", n_tokens)
        self.log(f"Random seed used: `{args['random_seed']}`\n")
        self.log(
            "Note: Random seed will not impact output if greedy decoding is used.\n"
        )
        self.log(f"Formatted prompt: `{formatted_prompt}`")

        self.log(f"Random seed used: `{args['random_seed']}`\n")
        self.log(
            "Note: Random seed will not impact output if greedy decoding is used.\n"
        )
        self.log(f"Formatted prompt: `{formatted_prompt}`")

    def _process_args(
        self,
        prompt: str,
        n_prompt_tokens: int,
        max_tokens: int = 250,
        min_tokens: Optional[int] = None,
        top_k: int = 0,
        top_p: float = 0.0,
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop_words: Optional[str] = None,
        seed: Optional[int] = None,
        stream: bool = True,
    ):
        stop_words_list = stop_words.split(",") if stop_words else []
        min_tokens = 0 if min_tokens is None else min_tokens

        pad_id = self.pad_id
        end_id = self.end_id

        decoding_mode = "top_k_top_p"

        if top_k <= 0:
            top_k = None
            decoding_mode = "top_p"

        if top_p == 0.0:
            if decoding_mode == "top_p":
                raise UserError(
                    "E1105 InvalidArgumentTopKTopP: Can't set both top_k and top_p to 0"
                )
            decoding_mode = "top_k"
            top_p = None

        if not seed:
            seed = int(np.random.randint(0, 100000))

        if self.max_sequence_length:
            token_budget = self.max_sequence_length - n_prompt_tokens
            max_tokens = min(max_tokens, token_budget)
            min_tokens = min(min_tokens, token_budget)

        if min_tokens <= 0:
            min_tokens = None

        args = {k: v for k, v in {
            "text_input": prompt,
            "max_tokens": max_tokens,
            "min_length": min_tokens,
            "top_k": top_k,
            "temperature": temperature,
            "top_p": top_p,
            "length_penalty": length_penalty,
            "presence_penalty": presence_penalty,
            "stop_words": stop_words_list,
            "stream": stream,
            "random_seed": seed,
            "pad_id": pad_id,
            "end_id": end_id,
            "decoding_mode": decoding_mode,
        }.items() if v is not None}

        return args

    _defaults = {
        key: param.default.default
        for key, param in inspect.signature(predict).parameters.items()
        if hasattr(param.default, "default")
    }

    def _get_n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])
