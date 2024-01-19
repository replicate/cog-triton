import os
import json
import requests
import time
import re
import multiprocessing
import subprocess
import pytest
import threading
import socketserver
import http
from threading import Thread, Lock

ERROR_PATTERN = re.compile(r"ERROR:")


def docker_image_exists(image_name):
    try:
        output = subprocess.check_output(["docker", "images", "-q", image_name])
        return output.decode("utf-8").strip() != ""
    except subprocess.CalledProcessError:
        return False


def get_image_name():
    image_name = os.getenv("COG_TRITON_IMAGE", None)
    if not image_name:
        raise RuntimeError(
            "You must set the COG_TRITON_IMAGE environment variable to the name of the image you want to run tests with."
        )
    return image_name


@pytest.fixture(scope="session", autouse=True)
def http_server_for_local_files():
    os.chdir("./tests/smoke")

    # Define the handler to be the SimpleHTTPRequestHandler
    handler = http.server.SimpleHTTPRequestHandler

    # Define the socket server, binding to localhost on port 8000
    httpd = socketserver.TCPServer(("0.0.0.0", 8000), handler)

    # Start the server in a new thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.start()

    # Yield control back to the test function
    yield

    httpd.shutdown()  # Shut down the server
    server_thread.join()  # Wait for the server thread to finish


@pytest.fixture(scope="session")
def cog_server():
    image_name = get_image_name()

    command = [
        "docker",
        "run",
        "-p",
        "5000:5000",
        "--gpus=all",
        "--workdir",
        "/src",
        "--net=host",
        image_name,
    ]

    print("\n**********************STARTING SERVER**********************")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print_lock = Lock()

    stdout_thread = Thread(target=capture_output, args=(process.stdout, print_lock))
    stdout_thread.start()

    stderr_thread = Thread(target=capture_output, args=(process.stderr, print_lock))
    stderr_thread.start()

    wait_for_server_to_be_ready("http://localhost:5000/health-check")

    yield process

    process.terminate()
    process.wait()


def process_log_line(line):
    line = line.decode("utf-8").strip()
    try:
        log_data = json.loads(line)
        return json.dumps(log_data, indent=2)
    except json.JSONDecodeError:
        return line


def capture_output(pipe, print_lock, logs=None, error_detected=None):
    for line in iter(pipe.readline, b""):
        formatted_line = process_log_line(line)
        with print_lock:
            print(formatted_line)
            if logs is not None:
                logs.append(formatted_line)
            if error_detected is not None:
                if ERROR_PATTERN.search(formatted_line):
                    error_detected[0] = True


def wait_for_server_to_be_ready(url, timeout=300):
    """
    Waits for the server to be ready.

    Args:
    - url: The health check URL to poll.
    - timeout: Maximum time (in seconds) to wait for the server to be ready.
    """
    start_time = time.time()
    while True:
        try:
            response = requests.get(url)
            data = response.json()

            if data["status"] == "READY":
                return
            elif data["status"] == "SETUP_FAILED":
                raise RuntimeError(
                    "Server initialization failed with status: SETUP_FAILED"
                )

        except requests.RequestException:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError("Server did not become ready in the expected time.")

        time.sleep(5)  # Poll every 5 seconds
