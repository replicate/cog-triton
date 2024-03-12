FROM r8.im/replicate-internal/cog-triton@sha256:b4dc15ac254131ca79b72be5882b61373708749e406a7054c21223611acf4d08
RUN python3.10 -m pip install aiortc
#RUN pip install https://r2.drysys.workers.dev/tmp/cog-0.9.4.dev81+g11986a1-py3-none-any.whl
COPY ./predict.py ./webrtc.py /src/
