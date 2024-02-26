import os
import subprocess
import requests
import time
import subprocess
import sys
from pathlib import Path
import shutil
import typing as tp
from collections import deque


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

    # if dest exists and is not empty, return
    if os.path.exists(dest) and os.listdir(dest):
        print(f"Files already present in the `{dest}`, nothing will be downloaded.")
        return dest

    # if dest exists but is empty, remove it so we can pull with pget
    if os.path.exists(dest):
        shutil.rmtree(dest)

    print("Downloading model assets...")
    command = ["pget", url, dest, "-x"]
    subprocess.check_call(command, close_fds=True)

    return dest


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


class StreamingTextStopSequenceHandler:
    def __init__(
        self,
        stop_sequences: tp.List[str] = None,
        eos_token: str = "<encountered-stop-sequence>",
    ):
        self.stop_sequences = stop_sequences or []
        self.eos_token = eos_token
        self.cache = []

        self.stop_sequence_tracker = [0] * len(self.stop_sequences)
        self.stop_sequence_lens = [len(seq) for seq in self.stop_sequences]
        self.stop_sequence_fulfilled = False

    def get_match_length(self, text: str, stop_sequence: str):
        """
        Checks if the end of the provided text matches the beginning of the stop sequence.
        Returns the length of the matched stop sequence if it exists, otherwise returns 0.
        """
        matched_len = 0
        for i in range(1, len(stop_sequence) + 1):
            if stop_sequence[:i] in text:
                matched_len = i

        return matched_len

    def process(self, token):
        partial_match = False
        output = None

        text = "".join(self.cache) + token
        for idx, stop_sequence in enumerate(self.stop_sequences):
            match_length = self.get_match_length(text, stop_sequence)

            if match_length:
                if match_length == self.stop_sequence_lens[idx]:
                    self.cache.append(token)
                    text_before_stop_sequence = "".join(self.cache).split(
                        stop_sequence, maxsplit=1
                    )[0]
                    self.cache.clear()
                    self.stop_sequence_tracker = [0] * len(self.stop_sequences)
                    if text_before_stop_sequence:
                        self.cache = [text_before_stop_sequence]
                        self.stop_sequence_fulfilled = True

                        return None
                    else:
                        self.stop_sequence_fulfilled = True
                        return None

                elif stop_sequence.startswith(text[-match_length]):
                    partial_match = True
                    self.stop_sequence_tracker[idx] = max(
                        match_length, self.stop_sequence_tracker[idx]
                    )

            else:
                self.stop_sequence_tracker[idx] = 0

        if not partial_match:
            output = text
            self.cache.clear()

        elif partial_match:
            reset_tracker = any(
                i < j
                for i, j in zip(
                    self.stop_sequence_tracker, [0] * len(self.stop_sequences)
                )
            )
            if reset_tracker:
                output = "".join(self.cache)
                self.cache.clear()

            self.cache.append(token)

        return output

    def __call__(self, token):
        if self.stop_sequences:
            return self.process(token)
        else:
            return token

    def finalize(self):
        if self.cache:
            final_output = "".join(self.cache)
            self.cache.clear()
            return final_output
        return None
