# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, ConcatenateIterator
from utils import maybe_download_tarball_with_pget
import subprocess
from trtllm_client import TRTLLMClient
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer
import queue
import threading

URL = "https://replicate.delivery/pbxt/qkRFtudUXCoAKlntnVLc3dBhRutRoW02L127bU3Q4778emHJA/engine.tar"


class Predictor(BasePredictor):
    def setup(self) -> None:
        # print("Downloading model files...")
        # maybe_download_tarball_with_pget(
        #     url=URL,
        #     dest="./engine",
        # # )
        # # launch triton server
        # # python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/tensorrtllm_backend/triton_model
        subprocess.run(
            [
                "python3",
                "/src/tensorrtllm_backend/scripts/launch_triton_server.py",
                "--world_size=1",
                # "--model_repo=/src/triton_model",
                "--model_repo=/src/triton_model_repo",
            ]
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "/src/gpt2",
            padding_side="left",
            trust_remote_code=True,
        )

        self.client = TRTLLMClient(tokenizer=tokenizer)

    def token_emitter(self):
        while True:
            # Wait for the callback to put a result in the queue
            try:
                result = self.client.user_data._completed_requests.get(timeout=0.1)
            except queue.Empty:
                continue  # Continue waiting for the callback

            # Try to get the decoded tokens
            try:
                decoded_token = self.client.user_data._decoded_tokens.get_nowait()
                print(decoded_token)
                yield decoded_token
                print("wtf now")
            except queue.Empty:
                pass  # No decoded tokens available yet

    def predict(self, prompt: str) -> ConcatenateIterator:
        # Ensure the client is configured with the shared queue
        self.client.user_data._completed_requests = (
            queue.Queue()
        )  # Reset for each call if necessary
        self.client.user_data._decoded_tokens = (
            queue.Queue()
        )  # Reset for each call if necessary

        # Start Triton client inference in a separate thread
        client_thread = threading.Thread(target=lambda: self.client.run(text=prompt))
        client_thread.start()

        # Stream tokens as they arrive from Triton
        while True:
            try:
                decoded_token = self.client.user_data._decoded_tokens.get(
                    timeout=0.1
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
