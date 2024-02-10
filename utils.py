from huggingface_hub import list_repo_files, snapshot_download
from tempfile import TemporaryDirectory
from distutils.dir_util import copy_tree
from pathlib import Path
import subprocess
import torch


def get_gpu_info():
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{i}")
        gpu_info.append(
            {
                "name": torch.cuda.get_device_name(device),
                "capability": torch.cuda.get_device_capability(device),
            }
        )
    return gpu_info


if __name__ == "__main__":
    # setup args and parser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        help="The model_id of the model you want to download.",
        required=True,
    )

    args = parser.parse_args()

    downloader = Downloader()
    downloader.download("gpt2")
    print("Done!")
