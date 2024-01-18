Docker launch the Triton server:

```
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /path/to/tensorrtllm_backend:/tensorrtllm_backend nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 bash

```

With the cog image:

docker run --rm -it -p 5000:5000 --gpus=all --workdir /src  --volume $(pwd)/.:/src/. cog-triton /bin/bash


cog predict with docker

```
docker run -d -p 5000:5000 --gpus=all cog-trt-llm:latest
curl http://localhost:5000/predictions -X POST \
-H "Content-Type: application/json" \
-d '{"input": {
  }}'
```