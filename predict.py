from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    # The arguments and types the model takes as input
    def predict(self) -> str:
        """Run a single prediction on the model"""
        return "hello!"
