#syntax=docker/dockerfile:1.4
FROM triton_trt_llm as deps
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
RUN pip install https://r2.drysys.workers.dev/tmp/cog-0.9.4.dev81+g11986a1-py3-none-any.whl -r /tmp/requirements.txt 
COPY *.py *.yaml /src/
COPY triton_model_repo /src/triton_model_repo

# prevent replicate from downgrading cog
RUN ln -sf $(which echo) $(which pip)
COPY ./cog.yaml ./sse.py ./predict.py ./utils.py /src/
