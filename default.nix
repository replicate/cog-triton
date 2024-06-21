{ pkgs, config, extendModules, ... }:
let
  deps = config.deps;
  python3 = config.python-env.deps.python;
  cudaPackages = pkgs.cudaPackages_12_1;
  site = python3.sitePackages;
  pythonDrvs = config.python-env.pip.drvs;
  inherit (pkgs) lib;
  cfg = config.cog-triton; # defined in interface.nix
in
{
  imports = [ ./interface.nix ];
  cog.build = {
    python_version = "3.10";
    cog_version = "0.10.0-alpha13";
    cuda = "12.1"; # todo: 12.2
    gpu = true;
    # inspiration: echo tensorrt_llm==0.8.0 | uv pip compile - --extra-index-url https://pypi.nvidia.com -p 3.10 --prerelease=allow --annotation-style=line
    python_packages = [
      "--extra-index-url"
      "https://pypi.nvidia.com"
      "tensorrt_llm==0.9.0"
      "torch==2.2.2"
      "tensorrt==9.3.0.post12.dev1"
      "tensorrt-bindings==9.3.0.post12.dev1"
      "tensorrt-libs==9.3.0.post12.dev1"
      "nvidia-pytriton==0.5.2" # corresponds to 2.42.0
      "httpx"
      "nvidia-cublas-cu12<12.2"
      "nvidia-cuda-nvrtc-cu12<12.2"
      "nvidia-cuda-runtime-cu12<12.2"
      "omegaconf"
      "hf-transfer"
      "tokenizers"
    ];
    # don't ask why it needs ssh
    system_packages = [ "pget" "openssh" "openmpi" ];
  };
  python-env.pip = {
    uv.enable = true;
    # todo: add some constraints to match cudaPackages
    constraintsList = [
      "nvidia-cudnn-cu12<9"
    ];
    overridesList = [
      "tokenizers==0.19.0"
      "transformers==4.40.0"
    ];
  };
  cognix.includeNix = true;
  cognix.nix.extraOptions = ''
    extra-trusted-public-keys = replicate-1:rbU0MI8kgUmqLINtKfXoDkrl9NxXQMw6//+LHHDYflk=
    extra-substituters = https://storage.googleapis.com/replicate-nix-cache-dev/
  '';
  python-env.pip.drvs = {
    # tensorrt likes doing a pip invocation from it's setup.py
    # circumvent by manually depending on tensorrt_libs, tensorrt_bindings
    # and setting this env variable
    tensorrt.env.NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP = true;
    # TODO remove upon next rebuild:
    tensorrt.mkDerivation.propagatedBuildInputs = with pythonDrvs; [
      tensorrt-libs.public
      tensorrt-bindings.public
    ];
    tensorrt-bindings.mkDerivation.propagatedBuildInputs = [ pythonDrvs.tensorrt-libs.public ];
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
      pushd $out/${site}/pytriton/tritonserver/python_backend_stubs
      # remove every python stub but the current python version
      for d in *; do
        if [ "$d" != "${python3.pythonVersion}" ]; then
          rm -r $d
        fi
      done
      popd
    '';
    # patch in cuda packages from nixpkgs
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
  deps.backend_dir = pkgs.runCommand "triton_backends" {} ''
    mkdir $out
    tritonserver=${pythonDrvs.nvidia-pytriton.public}/${site}/pytriton/tritonserver
    cp -r $tritonserver/backends $out/
    chmod -R +w $out/backends
    cp $tritonserver/python_backend_stubs/${python3.pythonVersion}/triton_python_backend_stub $out/backends/python/
    cp -r ${deps.trtllm-backend}/backends/tensorrtllm $out/backends/
    for f in $out/backends/tensorrtllm/*; do
      chmod +w $f
      patchelf --add-rpath ${pythonDrvs.nvidia-pytriton.public}/${site}/nvidia_pytriton.libs $f
    done
  '';
  deps.tensorrt-src = pkgs.fetchFromGitHub {
    owner = "NVIDIA";
    repo = "TensorRT";
    rev = "6d1397ed4bb65933d02725623c122a157544a729"; # release/9.3 branch
    hash = "sha256-XWFyMD7jjvgIihlqCJNyH5iSa1vZCDhv1maLJqMM3UE=";
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
