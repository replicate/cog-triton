# cog-triton
A cog implementation of Nvidia's Triton server

## Create a Replicate Model with cog-triton

Currently, we use [yolo](https://github.com/replicate/yolo), a CLI tool we've built to help with non-standard Replicate workflows. To get started, install yolo:

```
sudo curl -o /usr/local/bin/yolo -L "https://github.com/replicate/yolo/releases/latest/download/yolo_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/yolo
```

Once you have yolo installed, follow these steps:

1. **Compile a TensorRT engine with cog-triton**

2. **If it doesn't exist already, you'll need to create the Replicate Model to which you'll push your cog-triton model**

You can create a new Replicate Model via web or our API. To keep things simple, we'll use the latter method.

First, set a Replicate API token.

```
export REPLICATE_API_TOKEN=<your-api-token>
```

```
curl -s -X POST -H "Authorization: Token $REPLICATE_API_TOKEN" \
    -d '{"owner": "my-username", "name": "my-new-model", "visibility": "private", "hardware": "gpu-a40-large"}' \
    https://api.replicate.com/v1/models 
```


We'll call our model `staging-gpt2-triton-trt-llm`

2. **Instantiate a cog-triton model with your TRT-LLM engine**

staging-gpt2-triton-trt-llm

```
yolo push \
--base r8.im/replicate-internal/cog-triton@sha256:5d784bf5f449a0578ceb903265bb756dae146a267fc075b4c77021babedc6637 \
--dest r8.im/replicate-internal/staging-gpt2-triton-trt-llm \
-e COG_WEIGHTS=https://replicate.delivery/pbxt/CUDp32x5hO6GMBWprN8o24vWOLZbnYm7AAoRTxLfe0CUfglkA/engine.tar
```


# Run cog-triton locally

To run cog-triton locally, you must either pull the cog-triton Replicate image or build your own image.


## Preparation to run cog-triton locally with Replicate image

### Pull and tag the cog-triton image

Go [here](https://replicate.com/replicate-internal/cog-triton/versions) and pick the version you want to run locally. For our purposes, we'll set the version ID as an environment variable so that the code chunks below won't get stale.

```
export COG_TRITON_VERSION=<version-id-here>
```


Then, click the version hash. We need to set our Replicate API Token and you can do that manually, or navigate to the HTTP tab in your browser and copy the export command.

```
export REPLICATE_API_TOKEN=<token-here>
```

Next, navigate to the `Docker` tab under Input. This will display a code chunk like with a `Docker run` command like:

```
docker run -d -p 5000:5000 --gpus=all r8.im/replicate-internal/cog-triton@sha256:2db2b5c2e199975fef07ed9045608ed7adc7796744041fa54d3ae9d13db6c3cf
```

We'll use the image reference to write a pull command:

```
docker pull r8.im/replicate-internal/cog-triton@sha256:${COG_TRITON_VERSION}
```

After the image has been pulled, you should tag it so that all the docker commands in this README will work. First, find the `IMAGE ID` for the image you just pulled, e.g. via `docker images`. Then run the command below after replacing `<image id>` with your image id.

```
docker tag <image id> cog-triton:latest
```

### Pull and initialize `tensorrtllm_backend` and it's submodules

```
git lfs install
git submodule update --init --recursive
```


### Run an engine built with cog-trt-llm

Copy all model artifacts from `cog-trt-llm/engine_outputs/` to `triton_model_repo/tensorrt_llm/1/`:

```
cp -r ../cog-trt-llm/engine_outputs/* triton_model_repo/tensorrt_llm/1/
```

Run the cog-triton image:

```
docker run --rm -it -p 5000:5000 --gpus=all --workdir /src --volume $(pwd)/.:/src/. --ulimit memlock=-1 --shm-size=20g cog-triton /bin/bash
python -m cog.server.http
```


Make a request:


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

# Performance tests with test_perf.py

```
time python3 scripts/test_perf.py --target cog-triton --rate 8 --unit rps --duration 30 --n_input_tokens 100 --n_output_tokens 100
```

# Development

## End-to-end build process 

Cog-triton is pre-release and not stable. This build process is not optimal and will change. However, here we document every step we took to generate a deployable cog-triton image.

1. Clone the cog-triton repo:

```
git clone https://github.com/replicate/cog-triton 
```

2. Update submodules
``` 
export TENSORRTLLM_VERSION=0.8.0
git lfs install
git checkout tags/v${TENSORRTLLM_VERSION}
git submodule update --init --recursive
# currently, tensorrt_llm is at commit==5955b8a and cutlass==39c6a83f
cd ..
```

3. Build cog-triton image

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


curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "What is machine learning?", "max_tokens": 20, "bad_words": "", "stop_words": ""}'

curl -X POST localhost:8000/v2/models/ensemble/generate -d '{"text_input": "Water + Fire = Steam\nEarth + Water = Plant\nHuman + Robe = Judge\nCow + Fire = Steak\nKing + Ocean = Poseidon\nComputer + Spy =", "max_tokens": 20, "bad_words": "", "stop_words": ""}'



curl -s -X POST \
  -H "Content-Type: application/json" \
  -d $'{
    "input": {
        "prompt": "Water + Fire = Steam\nEarth + Water = Plant\nHuman + Robe = Judge\nCow + Fire = Steak\nKing + Ocean = Poseidon\nComputer + Spy ="
    }
  }' \
  http://localhost:5000/predictions

  curl -s -X POST \
  -H "Content-Type: application/json" \
  -d $'{
    "input": {
        "prompt": "Water + Fire = Steam\nEarth + Water = Plant\nHuman + Robe = Judge\nCow + Fire = Steak\nKing + Ocean = Poseidon\nComputer + Spy ="
    }
  }' \
  http://localhost:5000/predictions

# Local Testing

## Test cog performance

The `mock_cog_triton` directory provides a mocked `cog-triton` server that emits tokens at a fixed rate. It also includes a performance test script that reports client side and server-side performance metrics.

This should eventually be integrated into a test suite, but to maintain some visibility into performance continuity, we can run it manually and eyeball.

To do this, start the `mock_cog_triton` cog server:
```
docker run --rm -it -p 5000:5000  --workdir /src  --net=host --volume $(pwd)/.:/src/. cog-triton bash -c "cd mock_cog_triton && python -m cog.server.http"
```

and execute the performance script:

```
python3 mock_cog_triton/test_perf.py  --unit batch --duration 10 --tps 100 --n_output_tokens 128 --output_method yield --rate 24
```

Expected output with these input parameters is shown below. Note that Single-stream TPS metrics are quite close to the server-side metrics.

```
------------------------------
Test Configuration:
------------------------------
Output Method: yield
Mode: batch
Rate: 24.0 batch
Duration: 60 seconds
Output tokens: 128
------------------------------
Concurrency levels:
Mode concurrency: 24
Mean concurrency: 13.8055
Median concurrency: 24.0
Max concurrency: 24
Min concurrency: 0
------------------------------
Statistics for completed predictions:
------------------------------
Single-stream TPS:
SSTPS - Std: 1.498
SSTPS - Median: 93.177
SSTPS - Mean: 93.082
SSTPS - Max: 96.346
SSTPS - Min: 87.296
------------------------------
Latency - Std: 0.023 seconds
Median response latency: 1.374 seconds
Mean response latency: 1.375 seconds
Max response latency: 1.466 seconds
Min response latency: 1.329 seconds
------------------------------
Server-side metrics:
------------------------------
Server-side TPS
--Expected mean: 100.000, Actual mean: 95.557
--Expected std: 0.000, Actual std: 1.198
--Expected median: 100.000, Actual median: 95.535
--Expected min: 100.000, Actual min: 90.650
--Expected max: 100.000, Actual max: 97.710
Response Latency
--Expected mean: 1.280, Actual mean: 1.340
--Expected std: 0.000, Actual std: 0.018
--Expected median: 1.280, Actual median: 1.340
--Expected min: 1.280, Actual min: 1.310
--Expected max: 1.280, Actual max: 1.410
Time to First Token
--Expected mean: 0.010, Actual mean: 0.010
--Expected std: 0.000, Actual std: 0.000
--Expected median: 0.010, Actual median: 0.010
--Expected min: 0.010, Actual min: 0.010
--Expected max: 0.010, Actual max: 0.010
------------------------------
Total requests made: 600
Total requests started: 600
Total requests completed: 600
Failure rate: 0.000, Total failures: 0
Cog already running prediction: 0
E2E throughput: 9.988 rps
```