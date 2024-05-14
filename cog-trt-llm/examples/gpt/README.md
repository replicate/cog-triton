
This is just a brain dump and needs to get cleaned up!





# Weights & config
rm -rf gpt2 && git clone https://huggingface.co/gpt2-medium models/gpt2
pushd models/gpt2 && rm pytorch_model.bin model.safetensors && wget -q https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin && popd
```


docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --gpus=all \
    --volume /home/joe/cog-trt-llm/TensorRT-LLM:/code/tensorrt_llm \
    --env "CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache" \
    --env "CCACHE_BASEDIR=/code/tensorrt_llm" \
    --workdir /code/tensorrt_llm \
    --hostname joe-a40-devel \
    --name tensorrt_llm-devel-joe \
    --tmpfs /tmp:exec \
    tensorrt_llm/release:latest

docker run --rm -it --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --gpus=all \
    --volume $(pwd)/.:/src/. \
    --env "CCACHE_DIR=/src/TensorRT-LLM/cpp/.ccache" \
    --env "CCACHE_BASEDIR=/src/TensorRT-LLM" \
    --workdir /src/. \
    --hostname joe-a40-devel \
    --name tensorrt_llm-devel-joe \
    --tmpfs /tmp:exec \
     --entrypoint /bin/bash \
    cog-trt-llm:latest


docker run --rm -it -p 5000:5000 --gpus=all --workdir /src  --volume $(pwd)/.:/src/. cog-trt-llm

curl -s -X POST \
  -H "Content-Type: application/json" \
  -d $'{
    "input": {
        "config":"./examples/gpt/config.yaml"
    }
  }' \
  http://localhost:5000/predictions


python hf_gpt_convert.py --in-file gpt2/ --out_dir ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16

cd TensorRT-LLM/examples/gpt/ && python hf_gpt_convert.py --in-file models/gpt2/ --out_dir ./c-model/gpt2 --tensor-parallelism 1 --storage-type float16