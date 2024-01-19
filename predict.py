# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from utils import maybe_download_tarball_with_pget


class Predictor(BasePredictor):
    def setup(self) -> None:
        print("Downloading model files...")
        maybe_download_tarball_with_pget(
            url="https://replicate.delivery/pbxt/s1rSbePwyeihDEAAX7jXejuoYSoIy4ZZUS0ePiDSFr27SLvIB/engine.tar.gz",
            dest="./engine",
        )
        

    def predict(
        self,
        prompt: str = Input(description="Enter a prompt", default=""),
    ) -> str:

        return "Hello world!"