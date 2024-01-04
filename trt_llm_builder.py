import os
import subprocess
import yaml
import logging

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
        example_name,
        model_name,
        config_path,
        base_local_model_dir="/src/models",
        trtllm_dir="/src/TensorRT-LLM",
    ):
        self.trtllm_dir = trtllm_dir

        if config_path:
            self.config = self._read_config(config_path)
        else:
            raise Exception(
                "Config is currently required. One day we'll provide defaults, but today is not that day."
            )

        self.example_name, self.model_name = self._get_example_and_model_names(
            example_name, model_name, self.config
        )

        self.example_dir = self._get_example_dir(self.example_name)
        self.local_model_dir = os.path.join(base_local_model_dir, self.model_name)
        assert self._maybe_download_model(self.local_model_dir)

    def run_build_workflow(self):
        target_model_dir = self.local_model_dir
        if "convert_to_ft" in self.config:
            target_model_dir = self._run_build_step("convert_to_ft", target_model_dir)

        if "quantize" in self.config:
            target_model_dir = self._run_build_step("quantize", target_model_dir)

        if "build" in self.config:
            target_model_dir = self._run_build_step("build", target_model_dir)

        self._prepare_model_for_upload(target_model_dir)

    def _run_build_step(self, step, target_model_dir):
        step_config = self.config[step]
        BuildPrinter.print(self.model_name, step)

        target_model_dir = self._run_script(
            target_model_dir, self.example_dir, **step_config
        )

        return target_model_dir

    def _run_script(
        self,
        target_model_dir,
        example_dir,
        output_dir,
        script,
        args,
        input_dir=None,
    ):
        script = os.path.join(example_dir, script)
        cmd = self._assemble_subprocess_cmd(script, args)

        assert os.path.isfile(script), (
            f"Script `{script}` does not exist."
            f"But, the following files do {os.listdir(example_dir)}!"
        )

        print("Using the following command:\n")
        print(" ".join(cmd))
        print(
            f"\n\nNote: Script output will be saved according to your command."
            f" However, some TRT-LLM scripts modify the output directory."
            f" Accordingly, you should use `output_dir` in your config to match the actual output directory that your TRT-LLM script writes to."
            f" It's up to you to make sure that your specified `output_dir` (i.e. {output_dir}) matches the real output directory for the script."
        )

        subprocess.run(cmd, check=True)
        return output_dir

    def _assemble_subprocess_cmd(self, conversion_script, args):
        cmd = ["python", conversion_script]
        for k, v in args.items():
            cmd += ["--" + str(k)]
            cmd += [str(v)] if v else []
        return cmd

    def _get_example_and_model_names(self, example_name, model_name, config):
        """
        This function determines the example and model to use for the build. It checks if the user provided
        the example and model as input arguments, and if not, it checks if they are specified in the config file.
        """
        if not example_name and not model_name and not config:
            raise Exception(
                "You must either specify `example_name` and `model_name`, provide a `config` file, or both."
                "If you only provide a `config` file, you must specify `example_name` and `model_name` in the config file."
                "If you do not provide a `config` file, a default config will be used if available."
            )

        if example_name and config.get("example_name", None):
            raise Exception(
                "You specified `example_name` as an input argument and in your config, this is not allowed."
            )
        elif config.get("example_name", None):
            example_name = config["example_name"]

        if model_name and config.get("model_name", None):
            raise Exception(
                "You specified `model_name` as an input argument and in your config, this is not allowed."
            )
        elif config.get("model_name", None):
            model_name = config["model_name"]

        return example_name, model_name

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

    def _maybe_download_model(self, local_model_dir):
        return True

    def _read_config(self, config_path):
        print("=====================================")
        print(f"Reading config from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    def _prepare_model_for_upload(self, target_model_dir):
        pass
