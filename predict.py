# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from utils import maybe_download_tarball_with_pget

URL = "https://replicate.delivery/pbxt/qkRFtudUXCoAKlntnVLc3dBhRutRoW02L127bU3Q4778emHJA/engine.tar"

class Predictor(BasePredictor):
    def setup(self) -> None:
        print("Downloading model files...")
        maybe_download_tarball_with_pget(
            url=URL,
            dest="./engine",
        )
        

    def predict(
        self,
        prompt: str = Input(description="Enter a prompt", default=""),
    ) -> str:

        return "Hello world!"