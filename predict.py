# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import time
import subprocess

from cog import BasePredictor, ConcatenateIterator
from utils import maybe_download_tarball_with_pget
import httpx
from sse import receive_sse
import string


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
        # # launch triton server
        # # python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/tensorrtllm_backend/triton_model
        subprocess.run(
            [
                "python3",
                "/src/tensorrtllm_backend/scripts/launch_triton_server.py",
                "--world_size=1",
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

        self.client = httpx.AsyncClient()

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
        frequency_penalty: float = 0.0,
        stop_words: str = None,
        prompt_template: str = os.getenv("PROMPT_TEMPLATE", None),
    ) -> ConcatenateIterator:
        if not self.model_exists:
            print(
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
            frequency_penalty=frequency_penalty,
            stop_words=stop_words,
        )

        req = self.client.stream(
            "POST",
            "http://localhost:8000/v2/models/tensorrt_llm_bls/generate_stream",
            json=args,
        )
        output = ""
        async with req as resp:
            async for event in receive_sse(resp):
                next_output = event.json()["text_output"]
                yield next_output.removeprefix(output)
                output = next_output

        print("Formatted prompt:\n", formatted_prompt)

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
        frequency_penalty: float = 0.0,
        stop_words: str = None,
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
            "stream": True,
        }

        return args

    def _format_prompt(
        self, prompt: str, prompt_template: str, system_prompt: str
    ) -> str:

        if not prompt_template:
            return prompt

        elif "system_prompt" in prompt_template:

            system_prompt = system_prompt if system_prompt else ""
            formatted_prompt = prompt_template.format(
                system_prompt=system_prompt, prompt=prompt
            )
            return formatted_prompt

        else:
            formatted_prompt = prompt_template.format(prompt=prompt)
            return formatted_prompt
