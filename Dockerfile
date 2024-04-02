#syntax=docker/dockerfile:1.4
# Use the CUDA 12.1.0 devel base image
FROM nvcr.io/nvidia/tritonserver:24.03-trtllm-python-py3

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

ENV MPICC=/usr/bin/mpicc

COPY tensorrtllm_backend/tensorrt_llm/docker/common/install_mpi4py.sh /tmp/

# Update PATH and LD_LIBRARY_PATH to include the Open MPI installation
ENV PATH="/opt/hpcx/ompi/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/hpcx/ompi/lib:/usr/local/nvidia/lib64:/usr/local/nvidia/bin:/usr/local/tensorrt/targets/x86_64-linux-gnu/lib/:$LD_LIBRARY_PATH"

# Now set CFLAGS to point to the include directory found in /opt/hpcx/ompi
ENV CFLAGS="-I/opt/hpcx/ompi/include"

# Install mpi4py using pip
# RUN pip3 install mpi4py

# # Download mpi4py and extract it
# RUN wget https://github.com/mpi4py/mpi4py/releases/download/3.1.3/mpi4py-3.1.3.tar.gz \
#     && tar -zxf mpi4py-3.1.3.tar.gz \
#     && rm mpi4py-3.1.3.tar.gz

# # Set environment variable for MPI compiler to include the non-standard header location
# ENV CFLAGS="-I/usr/lib/x86_64-linux-gnu/openmpi/include"

# RUN apt-get remove -y --purge libopenmpi-dev openmpi-bin && apt-get install -y libopenmpi-dev openmpi-bin

# # Before building mpi4py, set environment variables for MPI compilers to include the non-standard header location
# ENV MPI_INCLUDE_PATH=/usr/lib/x86_64-linux-gnu/openmpi/include
# ENV MPI_LIB_PATH=/usr/lib/x86_64-linux-gnu/openmpi/lib
# ENV MPI_BIN_PATH=/usr/lib/x86_64-linux-gnu/openmpi/bin

# # Build and install mpi4py, specifying the custom include path for MPI headers
# RUN cd mpi4py-3.1.3 \
#     && python setup.py build --mpicc="/usr/bin/mpicc -I$MPI_INCLUDE_PATH" \
#     && python setup.py install

# Clean up the mpi4py source directory if you wish
# RUN rm -rf mpi4py-3.1.3

# RUN bash /tmp/install_mpi4py.sh && rm /tmp/install_mpi4py.sh