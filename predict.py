# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def predict(
        self,
    ) -> str:

        return "Hello world!"