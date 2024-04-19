{ pkgs, config, extendModules, ... }:
let
  deps = config.deps;
  python3 = config.python-env.deps.python;
  cudaPackages = pkgs.cudaPackages_12_2;
  site = python3.sitePackages;
  pythonDrvs = config.python-env.pip.drvs;
  inherit (pkgs) lib;
  cfg = config.cog-triton; # defined in interface.nix
in
{
  imports = [ ./interface.nix ];
  cog.build = {
    python_version = "3.10";
    cog_version = "0.10.0-alpha5";
    cuda = "12.1"; # todo: 12.2
    gpu = true;
    # echo tensorrt_llm==0.8.0 | uv pip compile - --extra-index-url https://pypi.nvidia.com -p 3.10 --prerelease=allow --annotation-style=line
    python_packages = [
      "--extra-index-url"
      "https://pypi.nvidia.com"
      "tensorrt_llm==0.8.0"
      "torch==2.1.2"
      "tensorrt==9.2.0.post12.dev5"
      "tensorrt-bindings==9.2.0.post12.dev5"
      "tensorrt-libs==9.2.0.post12.dev5"
      "nvidia-pytriton==0.5.2" # corresponds to 2.42.0
      "httpx"
      "nvidia-cublas-cu12<12.4"
      "nvidia-cuda-nvrtc-cu12<12.4"
      "nvidia-cuda-runtime-cu12<12.4"
      "omegaconf"
      "hf-transfer"
      "tokenizers"
    ];
    # don't ask why it needs ssh
    system_packages = [ "pget" "openssh" "openmpi" ];
  };
  python-env.pip.uv = {
    enable = true;
    # todo: add some constraints to match cudaPackages
    constraints = [
      "nvidia-cudnn-cu12<9"
    ];
    overrides = [
      "tokenizers==0.19.0"
      "transformers==4.40.0"
    ];
  };
  cognix.includeNix = true;
  # allow taking subsets of the above python_packages
  python-env.pip.rootDependencies = lib.mkIf (config.cog-triton.rootDependencies != null)
    (lib.mkForce (lib.genAttrs ([ "cog" ] ++ cfg.rootDependencies) (x: true)));
  python-env.pip.drvs = {
    # tensorrt likes doing a pip invocation from it's setup.py
    # circumvent by manually depending on tensorrt_libs, tensorrt_bindings
    # and setting this env variable
    tensorrt.env.NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP = true;
    tensorrt.mkDerivation.propagatedBuildInputs = with pythonDrvs; [
      tensorrt-libs.public
      tensorrt-bindings.public
    ];
    tensorrt-bindings.mkDerivation.propagatedBuildInputs = [ pythonDrvs.tensorrt-libs.public ];
    # fixed in torch 2.2
    torch.mkDerivation.postFixup = ''
      pushd $out/${site}/torch/lib
      ln -s libcudart-*.so.12 libcudart.so.12
      ln -s libnvrtc-*.so.12 libnvrtc.so.12
      ln -s libnvToolsExt-*.so.1 libnvToolsExt.so.1
      popd
    '';
    tensorrt-libs.mkDerivation.postFixup = ''
      pushd $out/${site}/tensorrt_libs
      ln -s libnvinfer.so.9 libnvinfer.so
      ln -s libnvonnxparser.so.9 libnvonnxparser.so
      popd
    '';
    tensorrt-libs.env.appendRunpaths = [ "/usr/lib64" "$ORIGIN" ];
    tensorrt-llm = {
      mkDerivation.buildInputs = [ cudaPackages.nccl ];
      mkDerivation.propagatedBuildInputs = with pythonDrvs; [
        tensorrt-libs.public # libnvinfer, onnxparse
      ];
      env.appendRunpaths = [ "/usr/lib64" "$ORIGIN" ];
      env.autoPatchelfIgnoreMissingDeps = ["libcuda.so.1"];
    };
    # has some binaries that want cudart
    tritonclient.mkDerivation.postInstall = "rm -r $out/bin";
    # replace libtritonserver-90a4cf82.so with libtritonserver.so
    # so backends don't have to know about the hash
    nvidia-pytriton.mkDerivation.postInstall = ''
      pushd $out/${site}/nvidia_pytriton.libs
      ln -s libtritonserver-*.so libtritonserver.so
      patchelf --replace-needed libtritonserver-*.so libtritonserver.so $out/${python3.sitePackages}/pytriton/tritonserver/bin/tritonserver
      popd
      pushd $out/${site}/pytriton/tritonserver
      mv python_backend_stubs/${python3.pythonVersion}/triton_python_backend_stub backends/python/
      rm -r python_backend_stubs/
      ln -s ${deps.trtllm-backend}/backends/tensorrtllm backends/
      popd
    '';
    cog = {
      version = lib.mkForce "0.10.0a6";
      mkDerivation.src = pkgs.fetchurl {
        url = "http://r2.drysys.workers.dev/tmp/cog-0.10.0a6-py3-none-any.whl";
        hash = "sha256-LWntNtgfPB9mvusmEVg8bxFzUlQAuIeeMytGOZcNdz4=";
      };
    };
    nvidia-cublas-cu12.mkDerivation.postInstall = ''
      pushd $out/${python3.sitePackages}/nvidia/cublas/lib
      for f in ./*.so.12; do
        chmod +w "$f"
        rm $f
        ln -s ${cudaPackages.libcublas.lib}/lib/$f ./$f
      done
      popd
    '';
    nvidia-cudnn-cu12.mkDerivation.postInstall = ''
      pushd $out/${python3.sitePackages}/nvidia/cudnn/lib
      for f in ./*.so.8; do
        chmod +w "$f"
        rm $f
        ln -s ${cudaPackages.cudnn.lib}/lib/$f ./$f
      done
      popd
    '';
  };
  # TODO: open-source, switch to fetchFromGitHub
  deps.cog-trt-llm = builtins.fetchGit {
    url = "git@github.com:replicate/cog-trt-llm.git";
    rev = "1f092d891b3cefeea5e0b4d39eb4406ebc60d99a";
    ref = "main";
  };
  deps.tensorrt-src = pkgs.fetchFromGitHub {
    owner = "NVIDIA";
    repo = "TensorRT";
    rev = "93b6044fc106b69bce6751f27aa9fc198b02bddc"; # release/9.2 branch
    hash = "sha256-W3ytzwq0mm40w6HZ/hArT6G7ID3HSUwzoZ8ix0Q/F6E=";
  };
  # todo: replace with lockfile
  deps.pybind11-stubgen = python3.pkgs.buildPythonPackage rec {
    pname = "pybind11-stubgen";
    version = "2.5";
    src = pkgs.fetchPypi {
      inherit pname version;
      hash = "sha256-lqf+vKski/mKvUu3LMX3KbqHsjRCR0VMF1nmPN6f7zQ=";
    };
  };
  deps.tensorrt-llm = pkgs.callPackage ./nix/tensorrt-llm.nix {
    inherit python3 cudaPackages pythonDrvs;
    # TODO: turn into config option
    withPython = false;
    inherit (cfg) architectures;
    inherit (deps) pybind11-stubgen tensorrt-src;
  };
  deps.trtllm-backend = pkgs.callPackage ./nix/trtllm-backend.nix {
    inherit python3 cudaPackages pythonDrvs;
    inherit (deps) tensorrt-llm tensorrt-src;
  };
}
