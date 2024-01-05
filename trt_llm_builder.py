import os
import subprocess
import logging
import tarfile
from transformers import AutoTokenizer
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BuildPrinter:
    convert_to_ft_msg = "Converting {model_name} weights to FT format..."
    quantize_msg = "Quantizing {model_name} weights..."
    build_msg = "Building {model_name} weights..."

    @classmethod
    def print(cls, model_name, step):
        if step == "convert_to_ft":
            msg = cls.convert_to_ft_msg.format(model_name=model_name)
        elif step == "quantize":
            msg = cls.quantize_msg.format(model_name=model_name)
        elif step == "build":
            msg = cls.build_msg.format(model_name=model_name)

        print(f"\n==========================================")
        print(f"================={step}===================")
        print(msg)


class TRTLLMBuilder:
    def __init__(
        self,
        trtllm_dir="/src/TensorRT-LLM",
    ):
        self.trtllm_dir = trtllm_dir

    def run(self, example_name, local_model_dir, config):
        model_id = config.model_id
        example_name = config.example_name
        example_dir = self._get_example_dir(example_name)

        target_model_dir = local_model_dir

        for step in ["convert_to_ft", "quantize", "build"]:
            if step in config:
                step_config = getattr(config, step)
                BuildPrinter.print(model_id, step)

                self._run_script(
                    example_dir,
                    step_config.script,
                    step_config.args,
                )

                target_model_dir = step_config.output_dir

        self._prepare_model_artifacts_for_upload(
            target_model_dir, local_model_dir, config
        )

        output = None
        return output

    def _try_to_load_tokenizer(self, local_model_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        except:
            tokenizer = None
        return tokenizer

    def _run_script(
        self,
        example_dir,
        script,
        args,
    ):
        script = os.path.join(example_dir, script)
        cmd = self._assemble_subprocess_cmd(script, args)

        assert os.path.isfile(script), (
            f"Script `{script}` does not exist."
            f"But, the following files do {os.listdir(example_dir)}!"
        )

        print("Using the following command:\n")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)

    def _assemble_subprocess_cmd(self, conversion_script, args):
        cmd = ["python", conversion_script]
        for k, v in args.items():
            cmd += ["--" + str(k)]
            cmd += [str(v)] if v else []
        return cmd

    def _get_example_dir(
        self,
        example_name,
    ):
        base_example_dir = os.path.join(self.trtllm_dir, "examples")
        example_dir = os.path.join(base_example_dir, example_name)
        assert os.path.isdir(example_dir), (
            f"Example `{example_name}` is not available. "
            f"Must be one of {os.listdir(base_example_dir)}"
        )
        return example_dir

    def _prepare_model_artifacts_for_upload(
        self, target_model_dir, local_model_dir, config
    ):
        # write config to target_model_dir
        with open(os.path.join(target_model_dir, "cog-trt-llm-config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(config))

        # try to load tokenizer and include in tarball if available
        tokenizer = self._try_to_load_tokenizer(local_model_dir)
        if tokenizer:
            tokenizer.save_pretrained(os.path.join(target_model_dir, "tokenizer"))

        with tarfile.open("./engine.tar.gz", "w:gz") as tar:
            tar.add(target_model_dir, arcname=os.path.basename(target_model_dir))
