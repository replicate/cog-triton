# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from cog import BasePredictor, Input, Path
import yaml
import subprocess
from trt_llm_builder import TRTLLMBuilder
from downloader import Downloader
from config_parser import ConfigParser
from utils import get_gpu_info

from huggingface_hub._login import _login

# Temporarily using the Triton w/ TRT-LLM backend image
# Set: TRTLLM_DIR="/src/tensorrtllm_backend/TensorRT-LLM",
# When we switch back to a TRT-LLM image
TRTLLM_DIR = os.environ.get("TRTLLM_DIR", "/src/tensorrtllm_backend/tensorrt_llm/")
if not os.path.isdir(TRTLLM_DIR):
    raise Exception(
        "TensorRT-LLM is not available. Please make sure the TensorRT-LLM repository is available at /src/TensorRT-LLM."
    )
else:
    # list subdirs in examples dir
    EXAMPLE_NAMES = os.listdir(os.path.join(TRTLLM_DIR, "examples"))


class Predictor(BasePredictor):
    def setup(self) -> None:

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
        hf_token: str = Input(description="Hugging Face API token", default=None),
    ) -> Path:

        config = self.load_config(config)
        self._login_to_hf_if_token_provided(config, hf_token)
        self.config_parser.print_config(config)

        local_model_dir = self.downloader.run(
            config.model_id,
            weight_format=config.weight_format,
            model_tar_url=config.model_tar_url,
        )

        output = self.builder.run(
            config=config,
            local_model_dir=local_model_dir,
        )

        return Path(output)

    def _post_process_config(self, config: dict) -> dict:
        # We should do proper validation one day
        if "weight_format" not in config:
            config["weight_format"] = None

        if "model_tar_url" in config and not config.model_tar_url.endswith(".tar"):
            raise ValueError(
                f"model_tar_url must be a URL to a .tar file, but got {config.model_tar_url}"
            )

        if "model_tar_url" not in config:
            config["model_tar_url"] = None

        return config

    def _login_to_hf_if_token_provided(self, config: dict, hf_token: str) -> dict:
        # Prioritize hf_token from input over config
        if not hf_token:
            if "hf_token" in config:
                # pop hf_token from config
                hf_token = config.pop("hf_token")

        if hf_token:
            print("Logging in to Hugging Face Hub...")
            _login(token=hf_token, add_to_git_credential=False)
        else:
            print("No HuggingFace token, not logging in, gated tokenizers and models cannot be downloaded")

    def load_config(self, config_path: str) -> dict:
        """Load a config file from a path.

        Args:
            config_path (str): Path to the config file.

        Returns:
            dict: Dictionary containing the config.
        """
        config = self.config_parser.load_config(config_path) if config_path else {}
        config = self.config_parser.update_config(config)
        config = self._post_process_config(config)
        return config


# if __name__ == "__main__":
#     p = Predictor()
#     p.setup()
#     config_path = os.path.join(os.getcwd(), "examples", "gpt", "starcoder.yaml")
#     p.predict(config=config_path, hf_token=None)
