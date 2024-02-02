import os
import subprocess
import requests
import time
import subprocess
import sys
from pathlib import Path


import os
import subprocess
from pathlib import Path
import logging


def maybe_download_tarball_with_pget(
    url: str,
    dest: str,
):
    """
    Downloads a tarball from url and decompresses to dest if dest does not exist. Remote path is constructed
    by concatenating remote_path and remote_filename. If remote_path is None, files are not downloaded.

    Args:
        url (str): URL to the tarball
        dest (str): Path to the directory where the tarball should be decompressed

    Returns:
        path (str): Path to the directory where files were downloaded

    """
    print("Checking for tarball...")
    if not os.path.exists(dest):
        print("Downloading tarball...")
        command = ["pget", url, dest, "-x"]
        subprocess.check_call(command, close_fds=True)

    return dest

    # subprocess.run(
    #     [
    #         "python3",
    #         "/src/tensorrtllm_backend/scripts/launch_triton_server.py",
    #         "--world_size=1",
    #         "--model_repo=/src/triton_model_repo",
    #     ]
    # )


import argparse
import subprocess
import sys
from pathlib import Path


class TritonHandler:
    def __init__(
        self,
        world_size=1,
        tritonserver="/opt/tritonserver/bin/tritonserver",
        grpc_port="8001",
        http_port="8000",
        metrics_port="8002",
        force=False,
        log=False,
        log_file="triton_log.txt",
        model_repo=None,
    ):
        if model_repo is None:
            model_repo = str(Path(__file__).parent.absolute()) + "/../all_models/gpt"
        self.world_size = world_size
        self.tritonserver = tritonserver
        self.grpc_port = grpc_port
        self.http_port = http_port
        self.metrics_port = metrics_port
        self.force = force
        self.log = log
        self.log_file = log_file
        self.model_repo = model_repo

    def get_cmd(self):
        cmd = ["mpirun", "--allow-run-as-root"]
        for i in range(self.world_size):
            cmd += ["-n", "1", self.tritonserver]
            if self.log and (i == 0):
                cmd += ["--log-verbose=3", f"--log-file={self.log_file}"]
            cmd += [
                f"--grpc-port={self.grpc_port}",
                f"--http-port={self.http_port}",
                f"--metrics-port={self.metrics_port}",
                f"--model-repository={self.model_repo}",
                "--disable-auto-complete-config",
                f"--backend-config=python,shm-region-prefix-name=prefix{i}_",
                ":",
            ]
        return cmd

    def start(self):
        res = subprocess.run(
            ["pgrep", "-r", "R", "tritonserver"], capture_output=True, encoding="utf-8"
        )
        if res.stdout:
            pids = res.stdout.replace("\n", " ").rstrip()
            msg = f"tritonserver process(es) already found with PID(s): {pids}.\n\tUse `kill {pids}` to stop them."
            if self.force:
                print(msg, file=sys.stderr)
            else:
                raise RuntimeError(msg + " Or use --force.")
        cmd = self.get_cmd()
        process = subprocess.Popen(cmd)

        try:
            # Exponential backoff
            max_retries = 10
            delay = 0.01  # initial delay
            for i in range(max_retries):
                try:
                    response = requests.get(f"http://localhost:{self.http_port}")
                    if response.status_code == 200:
                        print("Server started successfully.")
                        return True
                except requests.exceptions.ConnectionError:
                    pass
                time.sleep(delay)
                delay *= 2  # double the delay

            stdout, stderr = process.communicate()
            error_message = stderr.decode("utf-8") if stderr else ""
            raise RuntimeError(f"Server failed to start.")

        except RuntimeError as e:
            process.terminate()
            raise e
