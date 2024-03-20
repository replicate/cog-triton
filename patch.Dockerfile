#FROM r8.im/replicate-internal/cog-triton@sha256:b4dc15ac254131ca79b72be5882b61373708749e406a7054c21223611acf4d08
FROM r8.im/replicate-internal/mistral-instruct-fp8-triton@sha256:36c4ff3cd593bc9fd1e513f148703b5f654a1a86ffd196af5c42afda3d64d800
RUN python3.10 -m pip install aiortc https://r2.drysys.workers.dev/tmp/cog-0.10.0a6.dev77+g55c6cf2-py3-none-any.whl
COPY ./predict.py ./webrtc.py ./utils.py /src/
RUN uname -a |tee /tmp/uname
