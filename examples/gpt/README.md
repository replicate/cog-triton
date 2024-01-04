```
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


docker run --rm -p 5000:5000 --gpus=all --workdir /src --volume $(pwd)/.:/src/. \cog-trt-llm

curl -s -X POST \
  -H "Content-Type: application/json" \
  -d $'{
    "input": {
        "config":"./examples/gpt/config.yaml"
    }
  }' \
  http://localhost:5000/predictions


  root@bfd2d67dd98a:/# find / -name libnvinfer.so.9
/usr/local/tensorrt/targets/x86_64-linux-gnu/lib/libnvinfer.so.9


  joe@joe-a40:~/cog-trt-llm$ docker run --rm -p 5000:5000 --gpus=all --workdir /src --volume $(pwd)/.:/src/. \cog-trt-llm
{"logger": "uvicorn.error", "timestamp": "2024-01-03T15:19:33.400602Z", "severity": "INFO", "message": "Started server process [19]"}
{"logger": "uvicorn.error", "timestamp": "2024-01-03T15:19:33.400721Z", "severity": "INFO", "message": "Waiting for application startup."}
{"logger": "uvicorn.error", "timestamp": "2024-01-03T15:19:33.694073Z", "severity": "INFO", "message": "Application startup complete."}
{"logger": "cog.server.probes", "timestamp": "2024-01-03T15:19:33.694704Z", "severity": "INFO", "message": "Not running in Kubernetes: disabling probe helpers."}
{"logger": "uvicorn.error", "timestamp": "2024-01-03T15:19:33.696568Z", "severity": "INFO", "message": "Uvicorn running on http://0.0.0.0:5000 (Press CTRL+C to quit)"}
{"prediction_id": null, "logger": "cog.server.runner", "timestamp": "2024-01-03T15:19:48.884708Z", "severity": "INFO", "message": "starting prediction"}
Running conversion script from `TensorRT-LLM/examples/gpt` with the following command:
['python', 'hf_gpt_convert.py', '--in-file models/gpt2/', '--out_dir ./c-model/gpt2', '--tensor-parallelism 1', '--storage-type float16']
/src
----------------------
Files in build_dir:
README.md
__pycache__
build.py
c-model
gpt2
hf_gpt_convert.py
merge_ptuning_tables.py
nemo_ckpt_convert.py
nemo_lora_convert.py
nemo_prompt_convert.py
requirements.txt
run_hf.py
smoothquant.py
utils
visualize.py
weight.py
Traceback (most recent call last):
File "/src/TensorRT-LLM/examples/gpt/hf_gpt_convert.py", line 32, in <module>
from utils.convert import split_and_save_weight
File "/src/TensorRT-LLM/examples/gpt/utils/convert.py", line 22, in <module>
from tensorrt_llm._utils import torch_to_numpy
File "/usr/local/lib/python3.10/dist-packages/tensorrt_llm/__init__.py", line 15, in <module>
import tensorrt_llm.functional as functional
File "/usr/local/lib/python3.10/dist-packages/tensorrt_llm/functional.py", line 26, in <module>
import tensorrt as trt
File "/usr/local/lib/python3.10/dist-packages/tensorrt/__init__.py", line 71, in <module>
from .tensorrt import *
ImportError: libnvinfer.so.9: cannot open shared object file: No such file or directory
Traceback (most recent call last):
File "/usr/local/lib/python3.10/dist-packages/cog/server/worker.py", line 218, in _predict
result = predict(**payload)
File "/src/predict.py", line 26, in predict
build_dir = self._convert_to_ft(
File "/src/predict.py", line 73, in _convert_to_ft
subprocess.run(cmd, cwd=build_dir, check=True)
File "/usr/lib/python3.10/subprocess.py", line 526, in run
raise CalledProcessError(retcode, process.args,
subprocess.CalledProcessError: Command '['python', 'hf_gpt_convert.py', '--in-file models/gpt2/', '--out_dir ./c-model/gpt2', '--tensor-parallelism 1', '--storage-type float16']' returned non-zero exit status 1.
{"error": "Command '['python', 'hf_gpt_convert.py', '--in-file models/gpt2/', '--out_dir ./c-model/gpt2', '--tensor-parallelism 1', '--storage-type float16']' returned non-zero exit status 1.", "prediction_id": null, "logger": "cog.server.runner", "timestamp": "2024-01-03T15:19:58.756299Z", "severity": "INFO", "message": "prediction failed"}