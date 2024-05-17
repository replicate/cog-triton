import pytest
import requests
import subprocess
import os
from threading import Thread, Lock
import socket
import http.server
import socketserver
import threading
from abc import ABC, abstractmethod

from tests.test_utils import (
    get_image_name,
    capture_output,
    wait_for_server_to_be_ready,
    docker_image_exists,
    cog_server,
    http_server_for_local_files,
)

# Constants
SERVER_URL = "http://localhost:5000/predictions"
HEALTH_CHECK_URL = "http://localhost:5000/health-check"


class BaseTests(ABC):
    @abstractmethod
    def test_prediction(self):
        pass


class TestLocal(BaseTests):
    def test_health_check(self, cog_server):
        response = requests.get(HEALTH_CHECK_URL)
        assert (
            response.status_code == 200
        ), f"Unexpected status code: {response.status_code}"
        print(response.json())

    def test_http_config_server(
        self,
        http_server_for_local_files,
    ):
        response = requests.get(
            "http://localhost:8000/gpt_convert_and_build_config.yaml"
        )
        assert (
            response.status_code == 200
        ), f"Unexpected status code: {response.status_code}"
        print(response.text)

    def test_prediction(self, cog_server, http_server_for_local_files):
        data = {
            "input": {
                "config": f"http://localhost:8000/gpt_convert_and_build_config.yaml",
            }
        }
        response = requests.post(SERVER_URL, json=data)
        assert (
            response.status_code == 200
        ), f"Unexpected status code: {response.status_code}"
        print("\n**********************RESPONSE**********************")
        assert response.json()["status"] != "failed"


if __name__ == "__main__":
    is_local = os.getenv("LOCAL_TEST")

    if is_local:
        print("Running local tests")
        pytest.main(["-k", "TestLocal", "-s"])
