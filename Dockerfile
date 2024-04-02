#syntax=docker/dockerfile:1.4
# Use the CUDA 12.1.0 devel base image
FROM nvcr.io/nvidia/tritonserver:24.03-py3

# Install required dependencies
RUN apt-get update && apt-get -y install \
    python3.10 \
    python3-pip \
    openmpi-bin \
    libopenmpi-dev

# Install the latest preview version of TensorRT-LLM
# RUN pip3 install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com

# Install the latest stable version (corresponding to the release branch) of TensorRT-LLM.
RUN pip3 install tensorrt_llm==0.8.0 --extra-index-url https://pypi.nvidia.com

RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin:/usr/local/tensorrt/targets/x86_64-linux-gnu/lib/
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PATH="/usr/bin:$PATH"


RUN TINI_VERSION=v0.19.0; \
    TINI_ARCH="$(dpkg --print-architecture)"; \
    curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
    chmod +x /sbin/tini

RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.5.6/pget_linux_x86_64" && chmod +x /usr/local/bin/pget

# Set the working directory
WORKDIR /src

# Expose the necessary port
EXPOSE 5000

# Set the environment variables for TRT-LLM
# ENV CCACHE_DIR=/src/TensorRT-LLM/cpp/.ccache
# ENV CCACHE_BASEDIR=/src/TensorRT-LLM

# Define entrypoint and command
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["python", "-m", "cog.server.http"]

COPY tensorrtllm_backend /src/tensorrtllm_backend

# pip install requirements and prerelease cog
COPY requirements.txt /tmp/requirements.txt
RUN pip install https://r2.drysys.workers.dev/tmp/cog-0.10.0a6-py3-none-any.whl -r /tmp/requirements.txt 
# prevent replicate from downgrading cog
RUN ln -sf $(which echo) $(which pip)
COPY triton_model_repo /src/triton_model_repo
COPY triton_templates /src/triton_templates
COPY *.py *.yaml /src/
