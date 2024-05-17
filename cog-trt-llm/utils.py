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
