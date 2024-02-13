from huggingface_hub import list_repo_files, snapshot_download
from tempfile import TemporaryDirectory
from distutils.dir_util import copy_tree
from pathlib import Path
import subprocess


class Downloader:
    def __init__(self, base_local_model_dir="/src/models"):
        self.base_local_model_dir = base_local_model_dir

    def run(self, model_id, revision=None, weight_format=None):
        print(f"Downloading model artifacts for {model_id}...")
        output_dir = Path(self.base_local_model_dir) / model_id

        if self._is_model_present(output_dir):
            print(
                f"Model {model_id} is already present in {output_dir}. Skipping download."
            )
            return output_dir

        # if model is cached on replicate, download with pget
        # else, download from HF Hub
        self._download_from_hf_hub(
            model_id, output_dir, revision=revision, weight_format=weight_format
        )

        print(f"Finished downloading {model_id}...")

        return output_dir

    def _is_model_present(self, output_dir):
        return output_dir.exists() and any(output_dir.iterdir())

    def _download_from_hf_hub(
        self, model_id, output_dir, revision=None, weight_format=None
    ):
        """
        This will download the model from the HuggingFace Hub.
        Currently, `snapshot_download` caches in a default location and then creates
        symlinks in `output_dir`. The benefit is that models will only be downloaded once.
        However, there are reasons to avoid that behavior, e.g. to ensure that weights don't leak
        across runs of the cog model. However, right now, we're biased toward minimizing number of downloads
        for a given set of weights.
        """
        # Check for .safetensors files:
        files = list_repo_files(model_id)
        allow_patterns = ["*.json", "tokenizer.model", "*.py"]

        # weight format can be .bin, .safe_tensors, or .pt
        if weight_format:
            if weight_format == "bin":
                allow_patterns.append("*.bin")
            elif weight_format == "safetensors":
                allow_patterns.append("*.safetensors")
            elif weight_format == "pt":
                allow_patterns.append("*.pt")
            else:
                raise Exception(
                    "Invalid weight format. Must be one of: bin, safetensors, pt"
                )
        else:
            if any(filename.endswith(".bin") for filename in files):
                allow_patterns.append("*.bin")
            elif any(filename.endswith(".safetensors") for filename in files):
                allow_patterns.append("*.safetensors")
            elif any(filename.endswith(".pt") for filename in files):
                allow_patterns.append("*.pt")
            else:
                raise Exception(
                    "No valid model files found in the repo. Must be one of: .bin, .pt, .safetensors"
                )

        with TemporaryDirectory() as tmpdir:
            # print tmpdir
            print(f"Downloading {model_id} to {tmpdir}...")

            snapshot_dir = snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=tmpdir,
                allow_patterns=allow_patterns,
                ignore_patterns=[
                    "*.onnx",
                    "*.tflite",
                    "*.pb",
                    "*.h5",
                    "*.hdf5",
                    "*onnx*",
                ],
                local_dir=output_dir,
                local_dir_use_symlinks=False,
            )

    def _download_with_pget(self, url, file_name, weight_format=None):
        raise NotImplementedError()


if __name__ == "__main__":
    # setup args and parser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        help="The model_id of the model you want to download.",
        required=True,
    )

    args = parser.parse_args()

    downloader = Downloader()
    downloader.download("gpt2")
    print("Done!")
