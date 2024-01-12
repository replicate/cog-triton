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
        # model_id: str = Input(
        #     description="Provide the ID of the model that you want to compile.",
        #     default=None,
        # ),
        # example_name: str = Input(
        #     description="Provide the name of the TRT-LLM build example that you want to run.",
        #     choices=EXAMPLE_NAMES,
        #     default=None,
        # ),
        # model_name: str = Input(
        #     description="Provide the name of the model that you want to compile.",
        #     default=None,
        # ),
        config: Path = Input(
            description="Path to your config file. If not provided, a default config will be used if available.",
            default=None,
        ),
    ) -> Path:
        config = self.config_parser.load_config(config) if config else {}
        config = self.config_parser.update_config(
            config,
            model_id=model_id,
            example_name=example_name,
            model_name=model_name,
        )

        self.config_parser.print_config(config)

        local_model_dir = self.downloader.run(config.model_id)

        output = self.builder.run(
            example_name=example_name,
            local_model_dir=local_model_dir,
            config=config,
        )

        return Path(output)
