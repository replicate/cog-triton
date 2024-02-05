# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import asyncio
import os
import queue
import subprocess
import threading

from cog import BasePredictor, ConcatenateIterator, Input, Path
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer

from trtllm_client import TRTLLMClient
from utils import maybe_download_tarball_with_pget

URL = "https://replicate.delivery/pbxt/qkRFtudUXCoAKlntnVLc3dBhRutRoW02L127bU3Q4778emHJA/engine.tar"


class FakeQueue(asyncio.Queue):
    put = asyncio.Queue.put_nowait

class Predictor(BasePredictor):
    def setup(self, weights: Path = None) -> None:
        # print("Downloading model files...")

        # get tokenizer_dir from os.environ
        tokenizer_dir = os.environ.get(
            "TOKENIZER_DIR", "/src/triton_model_repo/tensorrt_llm/1/"
        )

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
        else:
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

        # Maybe need to special case llama tokenizer loading
        # should probably pull this out of setup so we can support
        # variety of contexts.

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            padding_side="left",
            trust_remote_code=True,
        )


    # def token_emitter(self):
    #     while True:
    #         # Wait for the callback to put a result in the queue
    #         try:
    #             result = self.client.user_data._completed_requests.get(timeout=0.1)
    #         except queue.Empty:
    #             continue  # Continue waiting for the callback

    #         # Try to get the decoded tokens
    #         try:
    #             decoded_token = self.client.user_data._decoded_tokens.get_nowait()
    #             print(decoded_token)
    #             yield decoded_token
    #             print("wtf now")
    #         except queue.Empty:
    #             pass  # No decoded tokens available yet

    async def predict(self, prompt: str) -> ConcatenateIterator:
        async for token in self._predict(str):
            yield token

    async def _predict(self, prompt: str):
        if not self.model_exists:
            print(
                "Your model directory is empty, so there's nothing to do. Remember, you can't run this like a normal model. You need to YOLO!"
            )
            return
        client = TRTLLMClient(tokenizer=self.tokenizer)

        # Ensure the client is configured with the shared queue
        client.user_data._completed_requests = (
            FakeQueue()
        )  # Reset for each call if necessary
        client.user_data._decoded_tokens = (
            FakeQueue()
        )  # Reset for each call if necessary

        # Start Triton client inference in a separate thread
        client_thread = threading.Thread(target=lambda: client.run(text=prompt))
        client_thread.start()

        # Stream tokens as they arrive from Triton
        while True:
            try:
                decoded_token = await client.user_data._decoded_tokens.get(
                    # timeout=0.1
                )  # Adjust timeout as needed
                if decoded_token is None:  # Check for stream termination signal
                    break
                yield decoded_token
            except queue.Empty:
                break  # Optionally handle timeout or continue waiting

        client_thread.join()  # Ensure client thread has finished

    # def predict(
    #     self, prompt: str = Input(description="Enter a prompt", default="")
    # ) -> ConcatenateIterator:
    #     # Start the client in a separate thread to avoid blocking
    #     client_thread = threading.Thread(target=lambda: self.client.run(text=prompt))
    #     client_thread.start()

    #     # Emit tokens as they arrive
    #     for token in self.token_emitter():
    #         yield token

    #     # Optionally, signal the token emitter to stop by putting None into the queue
    #     self.client.user_data.put(None)
    #     client_thread.join()
