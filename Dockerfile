# First stage: Set up Triton TRT-LLM environment
FROM nvcr.io/nvidia/tritonserver:23.12-trtllm-python-py3 as triton_trt_llm

# Set the working directory in the container
WORKDIR /src

# Install git-lfs
RUN apt-get update && apt-get install -y git-lfs
 
# Clone the tensorrtllm_backend repository
RUN git clone https://github.com/triton-inference-server/tensorrtllm_backend.git -b v0.7.1 /src/tensorrtllm_backend \
    && cd /src/tensorrtllm_backend \
    && git lfs install \
    && git submodule update --init --recursive

# Install other dependencies
RUN apt-get install -y python3.10 python3-pip

# # Install tensorrt_llm
ENV TENSORRT_LLM_VERSION=0.7.1
RUN pip3 install tensorrt_llm==$TENSORRT_LLM_VERSION --extra-index-url https://pypi.nvidia.com

# # Second stage: Build upon the first stage and add additional dependencies and configurations
# #syntax=docker/dockerfile:1.4
FROM triton_trt_llm as final_stage

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/nvidia/bin:/usr/local/tensorrt/targets/x86_64-linux-gnu/lib/
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV PATH="/usr/bin:$PATH"

# Install necessary packages
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -qq \
    && apt-get install -qqy --no-install-recommends curl make build-essential libssl-dev zlib1g-dev \
       libbz2-dev libreadline-dev libsqlite3-dev wget llvm libncurses5-dev libncursesw5-dev \
       xz-utils tk-dev libffi-dev liblzma-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Tini
RUN TINI_VERSION=v0.19.0 \
    && TINI_ARCH="$(dpkg --print-architecture)" \
    && curl -sSL -o /sbin/tini "https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TINI_ARCH}" \
    && chmod +x /sbin/tini

# Install Python wheel and requirements
COPY .cog/tmp/build433494478/cog-0.0.1.dev-py3-none-any.whl /tmp/cog-0.0.1.dev-py3-none-any.whl
COPY requirements.txt /tmp/requirements.txt
RUN pip install /tmp/cog-0.0.1.dev-py3-none-any.whl \
    && pip install -r /tmp/requirements.txt

# Install pget
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.5.6/pget_linux_x86_64" \
    && chmod +x /usr/local/bin/pget

# Expose the necessary port
EXPOSE 5000

# Define entrypoint and command
ENTRYPOINT ["/sbin/tini", "--"]
CMD ["python", "-m", "cog.server.http"]

# # Copy application files
# COPY tensorrtllm_backend /src/tensorrtllm_backend
COPY *.py *.yaml /src/
