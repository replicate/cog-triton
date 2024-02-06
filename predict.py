# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import asyncio
import os
import subprocess

from cog import BasePredictor, ConcatenateIterator, Input, Path
from utils import maybe_download_tarball_with_pget
import httpx


URL = "https://replicate.delivery/pbxt/qkRFtudUXCoAKlntnVLc3dBhRutRoW02L127bU3Q4778emHJA/engine.tar"


class Predictor(BasePredictor):
    def setup(self, weights: str  = None) -> None:
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
        self.client = httpx.AsyncClient()

    async def predict(
        self, prompt: str, max_new_tokens: int = 500
    ) -> ConcatenateIterator:
        if not self.model_exists:
            print(
                "Your model directory is empty, so there's nothing to do. Remember, you can't run this like a normal model. You need to YOLO!"
            )
            return
        args = {"text_input": prompt, "max_tokens": max_new_tokens, "stream": True}
        req = client.stream(
            "POST",
            "http://localhost:8000/v2/models/ensemble/generate_stream",
            json=args,
        )
        async with req as resp:
            async for event in receive_sse(req):
                yield event.json()["text_output"]
