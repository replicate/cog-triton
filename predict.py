import asyncio
import os
import subprocess
from typing import Optional
import httpx
from cog import BasePredictor, ConcatenateIterator

from sse import receive_sse
from utils import (
    maybe_download_tarball_with_pget,
    StreamingTokenStopSequenceHandler,
)


class Predictor(BasePredictor):
    async def setup(self, weights: str = "") -> None:
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
        self.client = httpx.AsyncClient()
        # self.ready = False
        # self.triton_is_starting = True
        # self.last_healthcheck = refreshing_value(1, self.triton_healthcheck)
        # self.last_poll = refreshing_value(0.02, self.triton_is_running)
        await self.start_triton()
        # asyncio.create_task(self.monitor_triton_health())
        # asyncio.create_task(self.monitor_triton_poll())

    async def triton_healthcheck(self) -> Optional[httpx.Response]:
        try:
            return await self.client.get("http://localhost:8000/v2/health/ready")
        except httpx.RequestError:
            return None

    async def start_triton(self):
        world_size = os.getenv("WORLD_SIZE", "1")
        # # launch triton server
        # # python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/tensorrtllm_backend/triton_model
        self.proc = subprocess.Popen(
            [
                "python3",
                "/src/launch_triton_server.py",
                f"--world_size={world_size}",
                "--model_repo=/src/triton_model_repo",
            ],
            close_fds=False,
        )
        # Health check Triton until it is ready
        while True:
            response = await self.triton_healthcheck()
            if response and response.status_code == 200:
                print("Triton is ready.")
                # self.ready = True
                # self.triton_is_starting = False
                break
            await asyncio.sleep(1)

    # async def monitor_triton_poll(self):
    #     while True:
    #         self.triton_running = self.proc and self.proc.poll() is None
    #         if not self.triton_running:
    #             self.ready = False
    #             if self.triton_is_starting is False:
    #                 self.triton_is_starting = True
    #                 await self.start_triton()
    #         # we don't want to do a syscall 128 times a second,
    #         # but syscalls are still pretty cheap
    #         await asyncio.sleep(0.02)

    # async def monitor_triton_health(self):
    #     while True:
    #         try:
    #             response = await self.triton_healthcheck()
    #             self.last_healthcheck = response and response.json()
    #             self.ready = response and response.status_code == 200
    #         except httpx.RequestError:
    #             self.last_healthcheck = None
    #         # http requests are more heavyweight than one syscall, one second is fine
    #         await asyncio.sleep(1)
    #     # if triton was not running, start it
    #     # if triton crashed, restart it
    #     # if we're already starting it don't start it again
    #     # if the healthcheck is not ready, don't serve requests

    # def triton_is_running(self) -> bool:
    #     return self.proc and self.proc.poll() is None and self.last_healthcheck

    async def predict(
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
            print(
                "Your model directory is empty, so there's nothing to do. Remember, you can't run this like a normal model. You need to YOLO!"
            )
            return
        # if not self.ready:
        #     raise Exception(
        #         f"triton is not ready. last healthcheck was {self.last_healthcheck}. triton subprocess is {self.proc and self.proc.wait()}"
        #     )

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

        try:
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
        except httpx.RequestError as e:
            code = self.proc.poll()
            if code is not None:
                print(f"triton exited with code {code}")
                # if not self.triton_is_starting:
                #     self.triton_is_starting = True
                #     self.start_triton()
                # check if triton actually started...
                raise Exception(
                    f"triton exited unexpectedly with code {code}, maybe restarted triton, please retry"
                ) from e
            raise Exception(
                "triton client http error, but triton is still running"
            ) from e
        print(f"Formatted prompt: `{formatted_prompt}`")

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
