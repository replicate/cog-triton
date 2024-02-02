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
git checkout tags/v0.7.1
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

python3 inflight_batcher_llm/client/inflight_batcher_llm_client.py --request-output-len 200 --tokenizer-dir /src/gpt2 


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
