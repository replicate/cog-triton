# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import yaml
import subprocess
import os

from trt_llm_builder import TRTLLMBuilder
from downloader import Downloader
from config_parser import ConfigParser

if not os.path.isdir("/src/TensorRT-LLM"):
    raise Exception(
        "TensorRT-LLM is not available. Please make sure the TensorRT-LLM repository is available at /src/TensorRT-LLM."
    )
else:
    TRTLLM_DIR = "/src/TensorRT-LLM"
    # list subdirs in examples dir
    EXAMPLE_NAMES = os.listdir(os.path.join(TRTLLM_DIR, "examples"))


class Predictor(BasePredictor):
    def setup(self) -> None:
        # TODO: Might be nice to print relevant system specs at setup

        self.downloader = Downloader(base_local_model_dir="/src/models")
        self.config_parser = ConfigParser()
        self.builder = TRTLLMBuilder()

    def predict(
        self,
        config: Path = Input(
            description="Path to your config file. If not provided, a default config will be used if available.",
            default=None,
        ),
        hf_token: str = Input(description="Hugging Face API token", default=None),
    ) -> Path:
        config = self.config_parser.load_config(config) if config else {}
        config = self.config_parser.update_config(
            config,
        )
        self.config_parser.print_config(config)
        if hf_token:
            config["hf_token"] = hf_token
        # check if a hf token was provided
        if "hf_token" in config:
            from huggingface_hub._login import _login
            _login(token=config.hf_token, add_to_git_credential=False)
        local_model_dir = self.downloader.run(config.model_id)

        output = self.builder.run(
            config=config,
            local_model_dir=local_model_dir,
        )

        return Path(output)

if __name__ == "__main__":
    p = Predictor()
    p.setup()
    p.predict(config='starcoder_config.txt')