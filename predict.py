import asyncio
import json
import os
import subprocess
import time
import multiprocessing as mp
from typing import Optional

import cog
from cog import BasePredictor, ConcatenateIterator, Input

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
        raise Exception(
            "E008: You have submitted both a prompt and a prompt template that doesn't include '{prompt}'. "
            "Your prompt would not be used. "
            "If don't want to use formatting, use your full prompt for the prompt argument and set prompt_template to '{prompt}'."
        )
    return prompt_template.format(
        system_prompt=system_prompt or "",
        prompt=prompt,
    )


@contextlib.asynccontextmanager
async def wrap_httpx_error(
    req: contextlib._AsyncGeneratorContextManager,
) -> AsyncIterator[httpx.Response]:
    try:
        with req as resp:
            yield resp
    except httpx.ReadTimeout:
        raise TritonError(
            f"E007: Triton timed out after {TRITON_TIMEOUT}s: httpx.ReadTimeout. "
            "This can happen for extremely long prompts or large batches. "
            "Try a shorter prompt, or sending requests more slowly."
        )


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
            "ENGINE_DIR", "/src/triton_model_repo/tensorrt_llm/1/"
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
        # Health check Triton until it is ready or for 3 minutes
        for i in range(180):
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
        print(f"Triton was not ready within 3 minutes (exit code: {self.proc.poll()})")
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
    ) -> ConcatenateIterator:
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
                "E001: A prompt is required, but your formatted prompt is blank"
            )

        # compatibility with older language models
        if max_new_tokens:
            # 512 is the default
            if max_tokens == 512 or max_tokens is None:
                max_tokens = max_new_tokens
            else:
                raise UserError(
                    f"E002: Can't set both max_tokens ({max_tokens}) and max_new_tokens ({max_new_tokens})"
                )
        if min_new_tokens:
            if min_tokens is None:
                min_tokens = min_new_tokens
            else:
                raise UserError(
                    f"E003: Can't set both min_tokens ({min_tokens}) and min_new_tokens ({min_new_tokens})"
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
                    raise json.JSONDecodeError(f"E009: Triton returned malformed JSON: {event}") from e
                if error_message := event_data.get("error"):
                    if "exceeds maximum input length" in error_message:
                        raise UserError(
                            f"E004: Prompt length exceeds maximum input length. Detail: {error_message}"
                        )
                    if (
                        "the first token of the stop sequence IDs was not"
                        in error_message
                    ):
                        raise TritonError(f"E005: Tokenizer error: {error_message}")
                    # should we raise an exception if there's both output_ids and error?
                    if "output_ids" not in event_data:
                        raise TritonError(f"E000: Unkown error: {error_message}")
                if not token := event_data.get("output_ids"):
                    raise KeyError(
                        f"E006: Triton returned malformed event (no output_ids or error key): {event_data}"
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

        if top_k < 0:
            top_k = 0
        if min_tokens < 0:
            min_tokens = 0

        if not seed:
            seed = int(np.random.randint(0, 100000))

        if self.max_sequence_length:
            token_budget = self.max_sequence_length - n_prompt_tokens
            max_tokens = min(max_tokens, token_budget)
            min_tokens = min(min_tokens, token_budget)

        args = {
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
        }

        return args

    def _get_n_tokens(self, text: str) -> int:
        return len(self.tokenizer(text)["input_ids"])
