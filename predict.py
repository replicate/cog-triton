# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import yaml
import subprocess
import os

from trt_llm_builder import TRTLLMBuilder
from downloader import Downloader
from config_parser import ConfigParser
from utils import get_gpu_info

# Temporarily using the Triton w/ TRT-LLM backend image
# Set: TRTLLM_DIR="/src/tensorrtllm_backend/TensorRT-LLM",
# When we switch back to a TRT-LLM image
TRTLLM_DIR = "/src/tensorrtllm_backend/tensorrt_llm/"
if not os.path.isdir(TRTLLM_DIR):
    raise Exception(
        "TensorRT-LLM is not available. Please make sure the TensorRT-LLM repository is available at /src/TensorRT-LLM."
    )
else:
    # list subdirs in examples dir
    EXAMPLE_NAMES = os.listdir(os.path.join(TRTLLM_DIR, "examples"))


class Predictor(BasePredictor):
    def setup(self) -> None:
        # TODO: Might be nice to print relevant system specs at setup

        self.downloader = Downloader(base_local_model_dir="/src/models")
        self.config_parser = ConfigParser()
        self.builder = TRTLLMBuilder(trtllm_dir=TRTLLM_DIR)

        print("*" * 30)
        print("*" * 30)
        print("*" * 30)
        print(f"GPU info:\n{get_gpu_info()}")
        print("*" * 30)
        print("*" * 30)
        print("*" * 30)

    def predict(
        self,
        config: Path = Input(
            description="Path to your config file. If not provided, a default config will be used if available.",
            default=None,
        ),
    ) -> Path:
        config = self.config_parser.load_config(config) if config else {}
        config = self.config_parser.update_config(
            config,
        )

        self.config_parser.print_config(config)

        local_model_dir = self.downloader.run(config.model_id)

        output = self.builder.run(
            config=config,
            local_model_dir=local_model_dir,
        )

        return Path(output)
