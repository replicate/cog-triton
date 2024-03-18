# Use the CUDA 12.1.0 devel base image
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install required dependencies
RUN apt-get update && apt-get -y install \
    python3.10 \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev

# Install the latest preview version of TensorRT-LLM
# RUN pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# Install the latest stable version (corresponding to the release branch) of TensorRT-LLM.
RUN pip3 install tensorrt_llm -U --extra-index-url https://pypi.nvidia.com
