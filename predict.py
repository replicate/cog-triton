# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import subprocess
import time
import inspect
from typing import AsyncIterator

import httpx
from cog import BasePredictor, ConcatenateIterator, Input

from sse import receive_sse
from utils import (
    maybe_download_tarball_with_pget,
    StreamingTokenStopSequenceHandler,
)
import webrtc


class Predictor(BasePredictor):
    def setup(self, weights: str = None) -> None:
        self.system_prompt_exists = os.getenv("SYSTEM_PROMPT", None)

        engine_dir = os.environ.get(
            "ENGINE_DIR", "/src/triton_model_repo/tensorrt_llm/1/"
        )

        if weights:
            print(f"Downloading model files from {weights}...")
            maybe_download_tarball_with_pget(
                url=weights,
                dest=engine_dir,
            )

        # if engine_dir is empty, stop here
        if not os.listdir(engine_dir):
            print("Engine directory is empty. Exiting.")
            self.model_exists = False
            return

        self.model_exists = True
        world_size = os.getenv("WORLD_SIZE", "1")
        # # launch triton server
        # # python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/tensorrtllm_backend/triton_model
        subprocess.run(
            [
                "python3",
                "/src/tensorrtllm_backend/scripts/launch_triton_server.py",
                f"--world_size={world_size}",
                "--model_repo=/src/triton_model_repo",
            ]
        )
        # Health check Triton until it is ready
        while True:
            try:
                response = httpx.get("http://localhost:8000/v2/health/ready")
                if response.status_code == 200:
                    print("Triton is ready.")
                    break
            except httpx.RequestError:
                pass
            time.sleep(1)

        self.client = httpx.AsyncClient(timeout=10)

    def webrtc_helper(self, offer: str) -> AsyncIterator[str]:
        self.log(f"creating connection for offer {offer}")
        webrtc.log.info = self.log
        rtc = webrtc.RTC(offer)

        @rtc.on_message
        async def handler(args: dict) -> AsyncIterator[dict[str, str | int | float]]:
            start = time.time()
            token_count = 0
            # that's right, this calls into predict recursively!
            # in principle you could create new/forwarded sessions from an existing connection?
            # resolve Input to the correct default values
            stream = self._predict(**(self.defaults | args))
            while True:
                # while-next() seems to express timing more clearly than for-in here
                tok_start = time.time()
                tok = await anext(stream, None)
                if tok is None:
                    break
                now = time.time()

                token_count += 1
                yield {
                    "text": tok,
                    "token_gen_latency": round((now - start) * 1000),
                    "gen_time": round((now - tok_start) * 1000),
                    "idx": token_count,
                }
            elapsed = time.time() - start
            tps = token_count / elapsed

            self.log(f"finished generating in {elapsed:.3f}, {tps:.2f} tok/s")
            yield {"status": "done", "tokens_per_second": round(tps, 3)}

        return rtc.answer()
        # return rtc.yield_output()

    async def predict(
        self,
        webrtc_offer: str = Input(
            description="instead of a single prediction, handle a WebRTC offer as json, optionally with an ice_server key of ICE servers to use for connecting",
            default=None,
        ),
    ) -> str:
        return await self.webrtc_helper(webrtc_offer)
            # async for output in self.webrtc_helper(webrtc_offer):
            #     if isinstance(output, dict): # this is very silly
            #         yield output["text"]
            #     else:
            #         yield output
            # return

    async def _predict(
        self,
        prompt: str,
        system_prompt: str = os.getenv("SYSTEM_PROMPT", None),
        max_new_tokens: int = 250,
        min_length: int = None,
        top_k: int = 0,
        top_p: float = 0.0,
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop_words: str = None,
        prompt_template: str = os.getenv("PROMPT_TEMPLATE", None),
    ) -> ConcatenateIterator:
        if not self.model_exists:
            self.log(
                "Your model directory is empty, so there's nothing to do. Remember, you can't run this like a normal model. You need to YOLO!"
            )
            return

        formatted_prompt = self._format_prompt(
            prompt=prompt, system_prompt=system_prompt, prompt_template=prompt_template
        )

        args = self._process_args(
            prompt=formatted_prompt,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            length_penalty=length_penalty,
            presence_penalty=presence_penalty,
            stop_words=stop_words,
        )

        req = self.client.stream(
            "POST",
            "http://localhost:8000/v2/models/tensorrt_llm_bls/generate_stream",
            json=args,
        )

        output = ""
        generation_length = 0
        stop_sequence_handler = StreamingTokenStopSequenceHandler(
            stop_sequences=args["stop_words"]
        )

        async with req as resp:
            async for event in receive_sse(resp):
                # Output is the _entire_ sequence, from the beginning
                output = event.json()["text_output"]
                # Catches partial emojis, waits for them to finish
                output = output.replace("\N{Replacement Character}", "")
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

        self.log(f"Formatted prompt: `{formatted_prompt}`")

    def _process_args(
        self,
        prompt: str,
        max_new_tokens: int = 250,
        min_length: int = None,
        top_k: int = 0,
        top_p: float = 0.0,
        temperature: float = 1.0,
        length_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop_words: str = None,
        stream: bool = True,
        end_id: int = 2,
        pad_id: int = 2,
    ):
        stop_words_list = stop_words.split(",") if stop_words else []
        min_length = 0 if min_length is None else min_length

        args = {
            "text_input": prompt,
            "max_tokens": max_new_tokens,
            "min_length": min_length,
            "top_k": top_k,
            "temperature": temperature,
            "top_p": top_p,
            "length_penalty": length_penalty,
            "presence_penalty": presence_penalty,
            "stop_words": stop_words_list,
            "stream": stream,
            "pad_id": pad_id,
            "end_id": end_id,
        }

        return args

    def _format_prompt(
        self, prompt: str, prompt_template: str, system_prompt: str
    ) -> str:
        if not prompt_template:
            return prompt
        if "system_prompt" in prompt_template:
            system_prompt = system_prompt if system_prompt else ""
            formatted_prompt = prompt_template.format(
                system_prompt=system_prompt, prompt=prompt
            )
            return formatted_prompt
        formatted_prompt = prompt_template.format(prompt=prompt)
        return formatted_prompt

    # resolve Input to the correct default values
    defaults = {
        key: param.default.default
        for key, param in inspect.signature(_predict).parameters.items()
        if hasattr(param.default, "default")
    }
