# cog-triton
A cog implementation of Nvidia's Triton server

# Development

## Running a dev environment

1. Build the image

```
cog build --dockerfile Dockerfile
```
or
```
docker build -t cog-trt-llm .
```

2. Run the image

docker run --rm -it -p 5000:5000 --gpus=all --workdir /src  --net=host --volume $(pwd)/.:/src/. cog-triton /bin/bash

3. Start cog server in image

```
python -m cog.server.http
```

4. Expose configs via HTTP so they can be "downloaded" by cog

```
python3 -m http.server 8000 --bind 0.0.0.0
``` 

http://localhost:8000/examples/gpt/config.yaml

5. Make a request

```
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d $'{
    "input": {
    }
  }' \
  http://localhost:5000/predictions

```

## Run tests

### Implementation Notes

We currently assume that the trt-llm model is downloaded to:
/src/tensorrtllm_backend/triton_model/tensorrt_llm/1/. 

If you are working with a fresh pull of tensorrtllm_backend, you need to configure the Triton model:

```sh
mkdir -p tensorrtllm_backend/triton_model/
cp -r tensorrtllm_backend/all_models/inflight_batcher_llm/* tensorrtllm_backend/triton_model/
```

Configure preprocessing

```
python3 tensorrtllm_backend/tools/fill_template.py tensorrtllm_backend/all_models/inflight_batcher_llm/preprocessing/config.pbtxt \
    tokenizer_dir:/src/tensorrtllm_backend/triton_model/tensorrt_llm/1,tokenizer_type:auto \
     > tensorrtllm_backend/triton_model/preprocessing/config.pbtxt
```

```
python3 tensorrtllm_backend/tools/fill_template.py tensorrtllm_backend/all_models/inflight_batcher_llm/postprocessing/config.pbtxt \
    tokenizer_dir:/src/tensorrtllm_backend/triton_model/tensorrt_llm/1,tokenizer_type:auto \
    > tensorrtllm_backend/triton_model/postprocessing/config.pbtxt
```

Configure model

```
python3 tensorrtllm_backend/tools/fill_template.py tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
    gpt_model_type:V1,gpt_model_path:/src/tensorrtllm_backend/triton_model/tensorrt_llm/1,batch_scheduler_policy:max_utilization \
    > tensorrtllm_backend/triton_model/tensorrt_llm/config.pbtxt
```