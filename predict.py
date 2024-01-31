# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, ConcatenateIterator
from utils import maybe_download_tarball_with_pget
import subprocess
from trtllm_client import TRTLLMClient
from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer

URL = "https://replicate.delivery/pbxt/qkRFtudUXCoAKlntnVLc3dBhRutRoW02L127bU3Q4778emHJA/engine.tar"


class Predictor(BasePredictor):
    def setup(self) -> None:
        # print("Downloading model files...")
        # maybe_download_tarball_with_pget(
        #     url=URL,
        #     dest="./engine",
        # )
        # launch triton server
        # python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/tensorrtllm_backend/triton_model
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

    def predict(
        self,
        prompt: str = Input(description="Enter a prompt", default=""),
    ) -> ConcatenateIterator:
        output = ""
        for decoded_token in self.client.run(text=prompt):
            output += decoded_token
            yield decoded_token

        print(output)
