# cog-triton
A cog implementation of Nvidia's Triton server

# Development

## End-to-end build process 

Cog-triton is pre-release and not stable. This build process is not optimal and will change. However, here we document every step we took to generate a deployable cog-triton image.

1. Clone the cog-triton image:

```
git clone https://github.com/replicate/cog-triton 
```

2. Update submodules
```
export TENSORRTLLM_VERSION=0.7.1
git lfs install
git submodule update --init --recursive
cd tensorrtllm_backend
git checkout v${TENSORRTLLM_VERSION}
```

3. Build TensorRT-LLM Backend

```
# Use the Dockerfile to build the backend in a container
# For x86_64
DOCKER_BUILDKIT=1 docker build -t triton_trt_llm -f ./dockerfile/Dockerfile.trt_llm_backend .
```

4. Build cog-triton

```
cd ..
docker build -t cog-triton .
```

## (Optional) Build a TRT-LLM Model Locally

1. Generate the default `triton_model` directory, which is where triton artifacts and configs will be stored, and generate default configs. Alternatively, you can provide your own config.

```
chmod +x ./scripts/generate_default_configs.sh
./scripts/generate_default_configs.sh
```

2. 

```
docker run --rm --gpus=all --workdir /src --volume $(pwd):/src cog-triton \
  bash -c "rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2 && \
  pushd gpt2 && rm -f pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd"
```

3. Convert the model from HF Transformers to FT

```
docker run --rm --gpus=all --workdir /src --volume $(pwd):/src cog-triton \
  bash -c "python /src/tensorrtllm_backend/tensorrt_llm/examples/gpt/hf_gpt_convert.py \
  -p 8 \
  -i /src/gpt2 \
  -o ./c-model/gpt2 \
  --tensor-parallelism 1 \
  --storage-type float16"
```

4. Build TensorRT engine

```
docker run --rm --gpus=all \
  --workdir /src \
  --volume $(pwd):/src cog-triton \
  bash -c "python3 /src/tensorrtllm_backend/tensorrt_llm/examples/gpt/build.py \
  --model_dir=/src/c-model/gpt2/1-gpu/ \
  --world_size=1 \
  --dtype float16 \
  --use_inflight_batching \
  --use_gpt_attention_plugin float16 \
  --paged_kv_cache \
  --use_gemm_plugin float16 \
  --remove_input_padding \
  --use_layernorm_plugin float16 \
  --hidden_act gelu \
  --parallel_build \
  --output_dir=/src/engines/fp16/1-gpu"
```

5. Copy TensorRT engine to triton model

```
cp ./engines/fp16/1-gpu/* triton_model_repo/tensorrt_llm/1
```

6. 

```
docker run --rm -it -p 5000:5000 -p 8000:8000 --gpus=all --workdir /src  --net=host --volume $(pwd)/.:/src/. cog-triton \
  bash -c "python -m cog.server.http"
```

7. Curl a request

You can curl directly to the Triton server:
```
curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'
```


or to cog

```
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d $'{
    "input": {
        "prompt": "What is machine learning?"
    }
  }' \
  http://localhost:5000/predictions
```
## Run with cog



back to what the rfucking fuck
python3 hf_gpt_convert.py -p 8 -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16

python3 build.py --model_dir=./c-model/gpt2/1-gpu/ \
                 --world_size=1 \
                 --dtype float16 \
                 --use_inflight_batching \
                 --use_gpt_attention_plugin float16 \
                 --paged_kv_cache \
                 --use_gemm_plugin float16 \
                 --remove_input_padding \
                 --use_layernorm_plugin float16 \
                 --hidden_act gelu \
                 --parallel_build \
                 --output_dir=engines/fp16/1-gpu

                 
mkdir triton_model_repo

# Copy the example models to the model repository
cp -r all_models/inflight_batcher_llm/* triton_model_repo/
rm -rf triton_model_repo/tensorrt_llm_bls

# Copy the TRT engine to triton_model_repo/tensorrt_llm/1/
cp tensorrt_llm/examples/gpt/engines/fp16/1-gpu/* triton_model_repo/tensorrt_llm/1

* preprocessing
export HF_GPT_MODEL=/src/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2

python3 tools/fill_template.py -i triton_model_repo/preprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,preprocessing_instance_count:1"

* post processing
python3 tools/fill_template.py -i triton_model_repo/postprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,postprocessing_instance_count:1"

* tensorrt_llm
python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt "triton_max_batch_size:4,decoupled_mode:True,engine_dir:/src/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:100,batch_scheduler_policy:max_utilization"

*ensemble
python3 tools/fill_template.py -i triton_model_repo/ensemble/config.pbtxt "triton_max_batch_size:4"


python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/tensorrtllm_backend/triton_model_repo

Modify the model configuration


* preprocessing 

export HF_GPT_MODEL=/src/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2
python3 tools/fill_template.py -i triton_model_repo/preprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,preprocessing_instance_count:1"



docker run --rm -it --net host --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all -v $(pwd):/src -w /src cog-triton bash

python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/triton_model/





## This is working

python3 hf_gpt_convert.py -p 8 -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16

python3 build.py --model_dir=./c-model/gpt2/1-gpu/                  --world_size=1                  --dtype float16                  --use_inflight_batching                  --use_gpt_attention_plugin float16                  --paged_kv_cache                  --use_gemm_plugin float16                  --remove_input_padding                  --use_layernorm_plugin float16                  --hidden_act gelu                  --parallel_build                  --output_dir=engines/fp16/1-gpu



cd /src/tensorrtllm_backend/

mkdir triton_model_repo

cp -r all_models/inflight_batcher_llm/* triton_model_repo/
rm -rf triton_model_repo/tensorrt_llm_bls

cp tensorrt_llm/examples/gpt/engines/fp16/1-gpu/* triton_model_repo/tensorrt_llm/1

export HF_GPT_MODEL=/src/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2

python3 tools/fill_template.py -i triton_model_repo/preprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,preprocessing_instance_count:1"

python3 tools/fill_template.py -i triton_model_repo/postprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,postprocessing_instance_count:1"

python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt "triton_max_batch_size:4,decoupled_mode:True,engine_dir:/src/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:100,batch_scheduler_policy:max_utilization"

python3 tools/fill_template.py -i triton_model_repo/ensemble/config.pbtxt "triton_max_batch_size:4"


## see if this works, different directories

# convert 

docker run --rm --gpus=all --workdir /src --volume $(pwd):/src cog-triton \
  bash -c "python3 /src/tensorrtllm_backend/tensorrt_llm/examples/gpt/hf_gpt_convert.py -p 8 -i gpt2 -o /src/tensorrtllm_backend/tensorrt_llm/examples/gpt/c-model/gpt2 --tensor-parallelism 1 --storage-type float16"


# build

docker run --rm --gpus=all --workdir /src --volume $(pwd):/src cog-triton \
  bash -c "python3 /src/tensorrtllm_backend/tensorrt_llm/examples/gpt/build.py --model_dir=/src/tensorrtllm_backend/tensorrt_llm/examples/gpt/c-model/gpt2/1-gpu/                  --world_size=1                  --dtype float16                  --use_inflight_batching                  --use_gpt_attention_plugin float16                  --paged_kv_cache                  --use_gemm_plugin float16                  --remove_input_padding                  --use_layernorm_plugin float16                  --hidden_act gelu                  --parallel_build                  --output_dir=/src/tensorrtllm_backend/tensorrt_llm/examples/gpt/engines/fp16/1-gpu"

# 

mkdir ./tensorrtllm_backend/triton_model_repo
cp -r tensorrtllm_backend/all_models/inflight_batcher_llm/* tensorrtllm_backend/triton_model_repo/
rm -rf tensorrtllm_backend/triton_model_repo/tensorrt_llm_bls

cp tensorrtllm_backend/tensorrt_llm/examples/gpt/engines/fp16/1-gpu/* tensorrtllm_backend/triton_model_repo/tensorrt_llm/1

export HF_GPT_MODEL=/src/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2

python3 tensorrtllm_backend/tools/fill_template.py -i tensorrtllm_backend/triton_model_repo/preprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,preprocessing_instance_count:1"

python3 tensorrtllm_backend/tools/fill_template.py -i tensorrtllm_backend/triton_model_repo/postprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,postprocessing_instance_count:1"

python3 tensorrtllm_backend/tools/fill_template.py -i tensorrtllm_backend/triton_model_repo/tensorrt_llm/config.pbtxt "triton_max_batch_size:4,decoupled_mode:True,engine_dir:/src/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:100,batch_scheduler_policy:max_utilization"

python3 tensorrtllm_backend/tools/fill_template.py -i tensorrtllm_backend/triton_model_repo/ensemble/config.pbtxt "triton_max_batch_size:4"


docker run --rm -it -p 5000:5000 -p 8000:8000 --gpus=all --workdir /src  --net=host --volume $(pwd)/.:/src/. cog-triton   bash -c "python -m cog.server.http"

curl -s -X POST   -H "Content-Type: application/json"   -d $'{
    "input": {
        "prompt": "What is machine learning?"
    }
  }'   http://localhost:5000/predictions



## see if this works, target directories

rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2 && \
  pushd gpt2 && rm -f pytorch_model.bin model.safetensors

wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd

# convert 

docker run --rm --gpus=all --workdir /src --volume $(pwd):/src cog-triton \
  bash -c "python3 /src/tensorrtllm_backend/tensorrt_llm/examples/gpt/hf_gpt_convert.py -p 8 -i gpt2 -o /src/c-model/gpt2 --tensor-parallelism 1 --storage-type float16"


# build

docker run --rm --gpus=all --workdir /src --volume $(pwd):/src cog-triton \
  bash -c "python3 /src/tensorrtllm_backend/tensorrt_llm/examples/gpt/build.py --model_dir=/src/c-model/gpt2/1-gpu/                  --world_size=1                  --dtype float16                  --use_inflight_batching                  --use_gpt_attention_plugin float16                  --paged_kv_cache                  --use_gemm_plugin float16                  --remove_input_padding                  --use_layernorm_plugin float16                  --hidden_act gelu                  --parallel_build                  --output_dir=/src/engines/fp16/1-gpu"

# 

rm triton_model_repo
mkdir ./triton_model_repo
cp -r tensorrtllm_backend/all_models/inflight_batcher_llm/* ./triton_model_repo/
rm -rf ./triton_model_repo/tensorrt_llm_bls

cp tensorrtllm_backend/tensorrt_llm/examples/gpt/engines/fp16/1-gpu/* ./triton_model_repo/tensorrt_llm/1

export HF_GPT_MODEL=/src/gpt2

python3 tensorrtllm_backend/tools/fill_template.py -i ./triton_model_repo/preprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,preprocessing_instance_count:1"

python3 tensorrtllm_backend/tools/fill_template.py -i ./triton_model_repo/postprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,postprocessing_instance_count:1"

python3 tensorrtllm_backend/tools/fill_template.py -i ./triton_model_repo/tensorrt_llm/config.pbtxt "triton_max_batch_size:4,decoupled_mode:True,engine_dir:/src/triton_model_repo/tensorrt_llm/1,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:100,batch_scheduler_policy:max_utilization"

python3 tensorrtllm_backend/tools/fill_template.py -i ./triton_model_repo/ensemble/config.pbtxt "triton_max_batch_size:4"


docker run --rm -it -p 5000:5000 -p 8000:8000 --gpus=all --workdir /src  --net=host --volume $(pwd)/.:/src/. cog-triton   bash -c "python -m cog.server.http"

curl -s -X POST   -H "Content-Type: application/json"   -d $'{
    "input": {
        "prompt": "What is machine learning?"
    }
  }'   http://localhost:5000/predictions




## see if this works, target directories, readme build

rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium gpt2 && \
  pushd gpt2 && rm -f pytorch_model.bin model.safetensors

wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd

# convert 

docker run --rm --gpus=all --workdir /src --volume $(pwd):/src cog-triton \
  bash -c "python3 /src/tensorrtllm_backend/tensorrt_llm/examples/gpt/hf_gpt_convert.py -p 8 -i gpt2 -o /src/c-model/gpt2 --tensor-parallelism 1 --storage-type float16"


# build

docker run --rm --gpus=all --workdir /src --volume $(pwd):/src cog-triton \
  bash -c "python3 /src/tensorrtllm_backend/tensorrt_llm/examples/gpt/build.py --model_dir=/src/c-model/gpt2/1-gpu/                  --world_size=1                  --dtype float16                  --use_inflight_batching                  --use_gpt_attention_plugin float16                  --paged_kv_cache                  --use_gemm_plugin float16                  --remove_input_padding                  --use_layernorm_plugin float16                  --hidden_act gelu                  --parallel_build                  --output_dir=/src/engines/fp16/1-gpu"

# 

rm triton_model_repo
mkdir ./triton_model_repo
cp -r tensorrtllm_backend/all_models/inflight_batcher_llm/* ./triton_model_repo/
rm -rf ./triton_model_repo/tensorrt_llm_bls

cp ./engines/fp16/1-gpu/* ./triton_model_repo/tensorrt_llm/1

export HF_GPT_MODEL=/src/gpt2

python3 tensorrtllm_backend/tools/fill_template.py -i ./triton_model_repo/preprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,preprocessing_instance_count:1"

python3 tensorrtllm_backend/tools/fill_template.py -i ./triton_model_repo/postprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,postprocessing_instance_count:1"

python3 tensorrtllm_backend/tools/fill_template.py -i ./triton_model_repo/tensorrt_llm/config.pbtxt "triton_max_batch_size:4,decoupled_mode:True,engine_dir:/src/triton_model_repo/tensorrt_llm/1,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:100,batch_scheduler_policy:max_utilization"

python3 tensorrtllm_backend/tools/fill_template.py -i ./triton_model_repo/ensemble/config.pbtxt "triton_max_batch_size:4"


docker run --rm -it -p 5000:5000 -p 8000:8000 --gpus=all --workdir /src  --net=host --volume $(pwd)/.:/src/. cog-triton   bash -c "python -m cog.server.http"

curl -s -X POST   -H "Content-Type: application/json"   -d $'{
    "input": {
        "prompt": "What is machine learning?"
    }
  }'   http://localhost:5000/predictions


## end this working

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
        "prompt": "Hello, my name is"
    }
  }' \
  http://localhost:5000/predictions

```


python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/tensorrtllm_backend/triton_model

docker run --rm -it -p 5000:5000 --gpus=all --workdir /src  --net=host --volume $(pwd)/.:/src/. cog-triton /bin/bash

docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v /path/to/tensorrtllm_backend:/tensorrtllm_backend nvcr.io/nvidia/tritonserver:23.10-trtllm-python-py3 bash


python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer-dir /src/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2 --streaming




* preprocessing 

export HF_GPT_MODEL=/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2
python3 tools/fill_template.py -i triton_model_repo/preprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,preprocessing_instance_count:1"

* tensorrt_llm
python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt "triton_max_batch_size:4,decoupled_mode:False,engine_dir:/src/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1,batching_strategy:V1,max_queue_delay_microseconds:100"


python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt "triton_max_batch_size:4,decoupled_mode:False,engine_dir:/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1,batching_strategy:V1,max_queue_delay_microseconds:100"


* postprocessing
python3 tools/fill_template.py -i triton_model_repo/postprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,postprocessing_instance_count:1"

* ensamble
python3 tools/fill_template.py -i triton_model_repo/ensemble/config.pbtxt "triton_max_batch_size:4"


python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/tensorrtllm_backend/triton_model_repo



* preprocessing 

export HF_GPT_MODEL=/src/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2
python3 tools/fill_template.py -i triton_model_repo/preprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,preprocessing_instance_count:1"

* tensorrt_llm
python3 tools/fill_template.py -i triton_model_repo/tensorrt_llm/config.pbtxt "triton_max_batch_size:4,decoupled_mode:False,engine_dir:/src/tensorrtllm_backend/triton_model_repo/tensorrt_llm/1,batching_strategy:V1,max_queue_delay_microseconds:100"

* postprocessing
python3 tools/fill_template.py -i triton_model_repo/postprocessing/config.pbtxt "triton_max_batch_size:4,tokenizer_dir:${HF_GPT_MODEL},tokenizer_type:auto,postprocessing_instance_count:1"

* ensamble
python3 tools/fill_template.py -i triton_model_repo/ensemble/config.pbtxt "triton_max_batch_size:4"


python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer-dir /src/tensorrtllm_backend/tensorrt_llm/examples/gpt/gpt2

python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/tensorrtllm_backend/triton_model_repo

## Run tests

### Implementation Notes



####

Building trt-llm backend 

```
git clone https://github.com/triton-inference-server/server.git
cd server
git checkout tags/v2.41.0
```

```

```
BASE_CONTAINER_IMAGE_NAME=nvcr.io/nvidia/tritonserver:23.12-py3-min
TENSORRTLLM_BACKEND_REPO_TAG=release/v0.7.1
PYTHON_BACKEND_REPO_TAG=r23.10

# Run the build script. The flags for some features or endpoints can be removed if not needed.
./build.py -v --no-container-interactive --enable-logging --enable-stats --enable-tracing \
              --enable-metrics --enable-gpu-metrics --enable-cpu-metrics \
              --filesystem=gcs \
              --endpoint=http --endpoint=grpc \
              --backend=ensemble --enable-gpu --endpoint=http --endpoint=grpc \
              --image=base,${BASE_CONTAINER_IMAGE_NAME} \
              --backend=tensorrtllm:${TENSORRTLLM_BACKEND_REPO_TAG} \
              --backend=python:${PYTHON_BACKEND_REPO_TAG}
```




If you run into arcane Triton errors like:

```
Assertion failed: d == a + length (/app/tensorrt_llm/cpp/tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.cpp:418)
```

the prebuilt TRT-LLM library that ships with the Triton image might be stale. You should make sure that the TRT-LLM build is _exactly_ the same as the one you used to build your image.


cp tensorrt_llm/build/lib/tensorrt_llm/libs/* /opt/tritonserver/backends/tensorrtllm/

cp plugins/* /opt/tritonserver/backends/tensorrtllm/

python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/tensorrtllm_backend/triton_model
python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/te[56/1573§ backend/triton model


curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'



```
BASE_CONTAINER_IMAGE_NAME=nvcr.io/nvidia/tritonserver:23.12-py3-min
TENSORRTLLM_BACKEND_REPO_TAG=release/0.7.1
PYTHON_BACKEND_REPO_TAG=r23.10
```

```
# Run the build script. The flags for some features or endpoints can be removed if not needed.
./build.py -v --no-container-interactive --enable-logging --enable-stats --enable-tracing \
              --enable-metrics --enable-gpu-metrics --enable-cpu-metrics \
              --filesystem=gcs \
              --endpoint=http --endpoint=grpc \
              --backend=ensemble --enable-gpu --endpoint=http --endpoint=grpc \
              --image=base,${BASE_CONTAINER_IMAGE_NAME} \
              --backend=tensorrtllm:${TENSORRTLLM_BACKEND_REPO_TAG} \
              --backend=python:${PYTHON_BACKEND_REPO_TAG}
```



python3 hf_gpt_convert.py -p 8 -i gpt2 -o ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16

python3 build.py --model_dir=./c-model/gpt2/1-gpu/ \
                 --world_size=1 \
                 --dtype float16 \
                 --use_inflight_batching \
                 --use_gpt_attention_plugin float16 \
                 --paged_kv_cache \
                 --use_gemm_plugin float16 \
                 --remove_input_padding \
                 --use_layernorm_plugin float16 \
                 --hidden_act gelu \
                 --parallel_build \
                 --output_dir=engines/fp16/1-gpu


cp /src/tensorrtllm_backend/tensorrt_llm/examples/gpt/engines/fp16/1-gpu/* /src/tensorrtllm_backend/triton_model/tensorrt_llm/1
python3 scripts/launch_triton_server.py --world_size=1 --model_repo=/src/tensorrtllm_backend/triton_model


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
    tokenizer_dir:/src/tensorrtllm_backend/triton_model/tensorrt_llm/1,tokenizer_type:auto,triton_max_batch_size:8 \
     > tensorrtllm_backend/triton_model/preprocessing/config.pbtxt
```

```
python3 tensorrtllm_backend/tools/fill_template.py tensorrtllm_backend/all_models/inflight_batcher_llm/postprocessing/config.pbtxt \
    tokenizer_dir:/src/tensorrtllm_backend/triton_model/tensorrt_llm/1,tokenizer_type:auto,triton_max_batch_size:8 \
    > tensorrtllm_backend/triton_model/postprocessing/config.pbtxt
```

Configure model

```
python3 tensorrtllm_backend/tools/fill_template.py tensorrtllm_backend/all_models/inflight_batcher_llm/tensorrt_llm/config.pbtxt \
    batching_strategy:inflight_fused_batching,engine_dir:/src/tensorrtllm_backend/triton_model/tensorrt_llm/1,batch_scheduler_policy:max_utilization,decoupled_mode:True,triton_max_batch_size:8 \
    > tensorrtllm_backend/triton_model/tensorrt_llm/config.pbtxt
```