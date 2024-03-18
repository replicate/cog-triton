# cog-trt-llm

**Note:** This is a prelease version that is likely to change in the future.

Cog-trt-llm is a wrapper around [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0). You can use it to build TensorRT-LLM (TRT-LLM) engines on Replicate infrastructure! 


### Why use cog-trt-llm?

* Cog-trt-llm wraps a TRT-LLM build that supports Ampere through Hopper architectures, which means you can use it to compile models for a wide range of devices.

* Cog-trt-llm runs on Replicate, which means you can compile TRT-LLM engines without spinning up a VM or building an image. 

* Cog-trt-llm streamlines deploying TRT-LLM engines on Replicate. For example, you can deploy with Triton simply by passing the `engine.tar.gz` output to cog-triton.

# Quick Start

Cog-trt-llm expects a config that specifies the TRT-LLM steps that need to be executed, as well as any requisite parameters for executing those steps. Currently, we provide minimal minimal TRT-LLM documentation, so you will need to consult the TRT-LLM docs to determine how to specify the build parameters for the engine you want to build. 

Currently we wrap [TRT-LLM's examples](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples). This means that you must specify an `example_name` in your config and cog-trt-llm will use that and the example `script` you specify in order to execute the build you specify.

To get you started, you can use the following config to build a TRT-LLM engine for `tiny-gpt2` for a single GPU (e.g. `tensor-parallelism=1`).

This model is a `gpt` model, so we will use the `gpt` example and the scripts it contains. Specifically, we'll rely on two scripts. The first script converts the HF model to a required format. Such a conversion is occasionally required prior to building a TRT-LLM engine, so we support that as a pipeline step called `convert_to_ft`. In this case, the conversion script is [`hf_gpt_convert.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/gpt/hf_gpt_convert.py).

Finally, two last bit of complexity: TRT-LLM scripts do not yet provide strict, common APIs. E.g., conversion scripts for different model classes might use different argument names! This means that you _must_ be sure to use the right argument names in your config as we currently do not provide any validation. 

Further, because we do not maintain an internal representation of script-specific arguments, cog-trt-llm has no way of tracking exact output or input directories. And, even worse, sometimes TRT-LLM will modify the output directory you specify! E.g., in the script below, the `output_dir` of the `convert_to_ft` step is actually `./c-model/gpt2/1-gpu` and _not_ `./c-model/gpt2/`, which is the argument provided to the script. This is because the TRT-LLM conversion script outputs to a subdirectory that depends on the `tensor-parallelism` argument. E.g., if `tensor-parallelism` was set to `2`, the output directory would be `os.path.join('./c-model/gpt2/', '2-gpu')`. 

We highlight these details because they have a very serious consequence! You need to specify the correct output for each step in the pipeline. 

```yaml
# this is an identifier for downloading the model. e.g., 
# https://huggingface.co/<model_id>
model_id: sshleifer/tiny-gpt2
# this is the name of the trt-llm example directory to use for the build
# See possible examples here: 
example_name: gpt
convert_to_ft:
  script: hf_gpt_convert.py
  output_dir: ./c-model/gpt2/1-gpu
  args:
    # Must be the path to the downloaded model
    # Model will always be downloaded to
    # /src/models/<model_name>/
    in-file: /src/models/sshleifer/tiny-gpt2
    out-dir: ./c-model/gpt2
    tensor-parallelism: 1
    storage-type: float16
build:
  script: build.py
  output_dir: ./engine_outputs/
  args:
    model_dir: ./c-model/gpt2/1-gpu
    output_dir: ./engine_outputs/
    # Keys without values are treated as flags
    use_gpt_attention_plugin: 
    remove_input_padding: 
```

# Run cog-trt-llm locally

To run cog-trt-llm locally, you must either pull the cog-trt-llm Replicate image or build your own image.

## Preparation to run cog-trt-llm locally with Replicate image

### Pull and tag the cog-trt-llm image

Go [here](https://replicate.com/replicate-internal/cog-trt-llm/versions) and pick the version you want to run locally. For our purposes, we'll set the version ID as an environment variable so that the code chunks below won't get stale.

```
export COG_TRT_LLM_VERSION=<version-id-here>
```


Then, click the version hash. We need to set our Replicate API Token and you can do that manually, or navigate to the HTTP tab in your browser and copy the export command.

```
export REPLICATE_API_TOKEN=<token-here>
```

Next, navigate to the `Docker` tab under Input. This will display a code chunk like with a `Docker run` command like:

```
docker run -d -p 5000:5000 --gpus=all r8.im/replicate-internal/cog-trt-llm@sha256:2db2b5c2e199975fef07ed9045608ed7adc7796744041fa54d3ae9d13db6c3cf
```

We'll use the image reference to write a pull command:

```
docker pull r8.im/replicate-internal/cog-trt-llm@sha256:${COG_TRT_LLM_VERSION}
```

After the image has been pulled, you should tag it so that all the docker commands in this README will work. First, find the `IMAGE ID` for the image you just pulled, e.g. via `docker images`. Then run the command below after replacing `<image id>` with your image id.

```
docker tag <image id> cog-trt-llm:latest
```

### Pull and initialize [`tensorrtllm_backend`](https://github.com/triton-inference-server/tensorrtllm_backend) and it's submodules

From the root of this repo, run the following:

```bash
sudo apt-get install git-lfs
git lfs install
git submodule update --init --recursive
```

## Build with your local cog-trt-llm image

1. Run a cog-trt-llm image and start the cog server
``` 
docker run --rm -it -p 5000:5000 --gpus=all --workdir /src  --net=host --volume $(pwd)/.:/src/. --ulimit memlock=-1 --shm-size=20g cog-trt-llm /bin/bash
python -m cog.server.http
```

2. In a separate terminal instance, tmux session, etc, start a local HTTP server. We'll use this to expose local build configs so that the cog server can "download" them.

```
python3 -m http.server 8003 --bind 0.0.0.0
```

3. Build a TRT-LLM engine with cog-trt-llm

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d $'{
    "input": { 
        "config":"http://localhost:8003/<local-path-to-config>"
    }
  }'   http://localhost:5000/predictions
```

The `<local-path-to-config>` references the root of this repo, specifically the yaml file `[examples/](https://github.com/replicate/cog-trt-llm/tree/main/examples/)` folder. For example, if you want to compile the model defined in [examples/llama/llama-2-7b-chat-int8-1gpu.yaml](examples/llama/llama-2-7b-chat-int8-1gpu.yaml), the curl command will look like this:

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d $'{
    "input": { 
        "config":"http://localhost:8003/examples/llama/llama-2-7b-chat-int8-1gpu.yaml"
    }
  }' \
  http://localhost:5000/predictions
```

Once you run this `curl` command, you can go back to your running docker container and see the logs associated with the model compilation.

# Development

## Running a dev environment
 
1. Initialize TRT-LLM

```
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs

git lfs install
git lfs pull
git submodule update --init --recursive
```

2. Set TRT-LLM Version.
Note: This _must_ match the version used in the Triton server. 

```
export TRT_LLM_VERSION=0.7.1
cd TensorRT-LLM
git checkout v${TRT_LLM_VERSION}
```

3. Build TRT-LLM 

```
make -C docker release_build
```
1. Build the image

```
cog build --dockerfile Dockerfile
```
or
```
docker build -t cog-trt-llm .
```

2. Run the Cog Server

```
docker run --rm -it -p 5000:5000 --gpus=all --workdir /src  --net=host --volume $(pwd)/.:/src/. cog-trt-llm /bin/bash
```
3. Start cog server in image

```
python -m cog.server.http
```

Note: You can also just call the following to run the image and start the server with one command:

```
docker run --rm -it -p 5000:5000 --gpus=all --workdir /src  --net=host --volume $(pwd)/.:/src/. cog-trt-llm bash -c "python -m cog.server.http"
```

4. Expose configs via HTTP so they can be "downloaded" by cog

```
python3 -m http.server 8003 --bind 0.0.0.0
``` 

5. Make a request

```
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d $'{
    "input": { 
        "config":"http://localhost:8003/examples/gpt/config.yaml"
    }
  }' \
  http://localhost:5000/predictions

```

## Tests 

Current tests are fragile and minimal.


### Smoke tests

This will run a local smoke test that does the following:
* Expose a GPT test config via a local http server
* Run cog-trt-llm in a container
* Execute the GPT conversion and build steps specified in the config

```
cog build --dockerfile Dockerfile
export COG_TRITON_IMAGE=cog-trt-llm
LOCAL_TEST=true python3 -m pytest tests/smoke/test_gpt_convert_and_build.py
```

The server components are abstracted and can be reused in other test modules. To build out a test for a new config, you can start with this implementation and point to a new config.

## Implementation Notes

First, build the TRT-LLM image.

```
sudo apt-get update && sudo apt-get -y install git git-lfs

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs install
git lfs pull
make -C docker release_build
```

Next, run the image and build TRT-LLM:

```
make -C docker run

# To build the TensorRT-LLM code.
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt

# Deploy TensorRT-LLM in your environment.
pip install ./build/tensorrt_llm*.whl
```

Now, you should have a local `tensorrt_llm/release:latest` image and you can build the cog Dockerfile:

```
docker build -t cog-trt-llm .
```
or
```
cog build --dockerfile Dockerfile
```


Finally, to make a request, we recommend running the image:
```
docker run --rm  -it -p 5000:5000 --gpus=all --workdir /src --entrypoint /bin/bash  --volume $(pwd)/.:/src/. cog-trt-llm
```

and make a request against the cog server:

```
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d $'{
    "input": {
        "config":"./examples/gpt/config.yaml"
    }
  }' \
  http://localhost:5000/predictions
```
 
Alternatively, if you just want to use TRT-LLM locally, you can:

```
docker run --rm  -it -p 5000:5000 --gpus=all --workdir /src --entrypoint /bin/bash  --volume $(pwd)/.:/src/. cog-trt-llm
```
