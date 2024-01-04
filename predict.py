# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import yaml
import subprocess
import os

from trt_llm_builder import TRTLLMBuilder


if not os.path.isdir("/src/TensorRT-LLM"):
    raise Exception(
        "TensorRT-LLM is not available. Please make sure the TensorRT-LLM repository is available at /src/TensorRT-LLM."
    )
else:
    BASE_LOCAL_MODEL_DIR = "/src/models"
    TRTLLM_DIR = "/src/TensorRT-LLM"
    # list subdirs in examples dir
    EXAMPLE_NAMES = os.listdir(os.path.join(TRTLLM_DIR, "examples"))


class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def predict(
        self,
        example_name: str = Input(
            description="Provide the name of the TRT-LLM build example that you want to run.",
            choices=EXAMPLE_NAMES,
            default=None,
        ),
        model_name: str = Input(
            description="Provide the name of the model that you want to compile.",
            default=None,
        ),
        config: str = Input(
            description="Path to your config file. If not provided, a default config will be used if available.",
            default=None,
        ),
    ) -> str:
        trt_llm_builder = TRTLLMBuilder(
            example_name=example_name,
            model_name=model_name,
            config_path=config,
            base_local_model_dir=BASE_LOCAL_MODEL_DIR,
            trtllm_dir=TRTLLM_DIR,
        )

        trt_llm_builder.run_build_workflow()
