# cog-trt-llm
A cog wrapper around trt-llm

## Development Notes

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



docker run -d --rm -p 5000:5000 --gpus=all cog-trt-llm

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