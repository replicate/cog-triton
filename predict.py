# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from utils import maybe_download_with_pget


class Predictor(BasePredictor):
    def setup(self) -> None:
        
        maybe_download_with_pget(
            path="engine/",
            remote_path="https://replicate.delivery/pbxt/s1rSbePwyeihDEAAX7jXejuoYSoIy4ZZUS0ePiDSFr27SLvIB/",
            remote_filenames=[
               "engine.tar.gz"
            ],
        )
    def predict(
        self,
    ) -> str:

        return "Hello world!"