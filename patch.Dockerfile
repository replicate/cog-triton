FROM r8.im/replicate-internal/staging-gpt2-triton-trt-llm@sha256:69668ff28d97a066bc36621cc11820d7c8adb5ddc9fe32df963705effddd1414
RUN pip install https://r2.drysys.workers.dev/tmp/cog-0.10.0a4.dev65+g511a4a7.d20240205-py3-none-any.whl 
COPY ./predict.py /src/predict.py
RUN ln -sf $(which echo) $(which pip)

