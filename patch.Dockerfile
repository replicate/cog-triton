FROM r8.im/replicate-internal/staging-gpt2-triton-trt-llm@sha256:69668ff28d97a066bc36621cc11820d7c8adb5ddc9fe32df963705effddd1414
#RUN pip install https://r2.drysys.workers.dev/tmp/cog-0.9.4.dev81+g11986a1-py3-none-any.whl
RUN python3.11 -m pip install aiortc
#COPY ./cog.yaml ./sse.py ./predict.py /src/
