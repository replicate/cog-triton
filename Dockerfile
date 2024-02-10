#syntax=docker/dockerfile:1.4
FROM triton_trt_llm as deps
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin:/usr/local/tensorrt/targets/x86_64-linux-gnu/lib/
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PATH="/usr/bin:$PATH"

# Install necessary packages
RUN --mount=type=cache,target=/var/cache/apt set -eux; \
    apt-get update -qq; \
    apt-get install -qqy --no-install-recommends curl make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev git ca-certificates; \
    rm -rf /var/lib/apt/lists/*

# Install Tini
RUN TINI_VERSION=v0.19.0; \
    TINI_ARCH="$(dpkg --print-architecture)"; \
    curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}"; \
    chmod +x /sbin/tini

# Copy and install the Python wheel
COPY .cog/tmp/build1194876799/cog-0.0.1.dev-py3-none-any.whl /tmp/cog-0.0.1.dev-py3-none-any.whl
RUN pip install /tmp/cog-0.0.1.dev-py3-none-any.whl

# pip install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.5.6/pget_linux_x86_64" && chmod +x /usr/local/bin/pget

# Set the working directory
WORKDIR /src

# Copy the examples
COPY examples /src/examples

# Expose the necessary port
EXPOSE 5000

# Set the environment variables for TRT-LLM
ENV CCACHE_DIR=/src/TensorRT-LLM/cpp/.ccache
ENV CCACHE_BASEDIR=/src/TensorRT-LLM

# Define entrypoint and command
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["python", "-m", "cog.server.http"]

COPY tensorrtllm_backend /src/tensorrtllm_backend
COPY *.py *.yaml /src/
