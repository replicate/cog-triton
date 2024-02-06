FROM r8.im/replicate-internal/staging-gpt2-triton-trt-llm@sha256:69668ff28d97a066bc36621cc11820d7c8adb5ddc9fe32df963705effddd1414
RUN pip install httpx
RUN pip install https://r2.drysys.workers.dev/tmp/cog-0.10.0a4.dev70+g8ffb906-py3-none-any.whl
RUN ln -sf $(which echo) $(which pip)
COPY ./predict.py /src/predict.py

