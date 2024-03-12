# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
import subprocess
import time

import httpx
from cog import BasePredictor, ConcatenateIterator

#from sse import receive_sse
#from utils import (
#    maybe_download_tarball_with_pget,
#    StreamingTokenStopSequenceHandler,
#)

import pytriton.utils.distribution

TRITONSERVER_DIST_DIR = pytriton.utils.distribution.get_root_module_path() / "tritonserver"


class Predictor(BasePredictor):
    def setup(self) -> None:
        # # launch triton server
        # # python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/tensorrtllm_backend/triton_model
        subprocess.Popen(
            [
                str(TRITONSERVER_DIST_DIR / "bin" / "tritonserver"),
                "--backend-dir", str(TRITONSERVER_DIST_DIR / "backends"),
                "--model-repository", "/src/triton_model_repo",
            ]
        )
        # Health check Triton until it is ready
        while True:
            try:
                response = httpx.get("http://localhost:8000/v2/health/ready")
                if response.status_code == 200:
                    print("Triton is ready.")
                    break
            except httpx.RequestError:
                pass
            time.sleep(1)

        self.client = httpx.AsyncClient(timeout=10)

    async def predict(self) -> str:

        return "hello!"

