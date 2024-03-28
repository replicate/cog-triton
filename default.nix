{ pkgs, config, extendModules, ... }:
let
  deps = config.deps;
  py3 = pkgs.python310;
  py3Pkgs = py3.pkgs;
  cudaPkgs = pkgs.cudaPackages_12_2;
  site = py3.sitePackages;
  inherit (pkgs) lib;
  cfg = config.cog-triton; # defined in interface.nix
in
{
  imports = [ ./interface.nix ];
  public.extendModules = extendModules;
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
    ];
    # don't ask why it needs ssh
    system_packages = [ "pget" "openssh" ];
  };
  cognix.includeNix = true;
  # limit to runner image
  python-env.pip.rootDependencies = lib.mkIf cfg.runnerOnly (lib.mkForce (lib.genAttrs [
    "cog" "nvidia-pytriton" "transformers"
  ] (x: true)));
  python-env.pip.drvs = let pyPkgs = config.python-env.pip.drvs; in {
    mpi4py.mkDerivation = {
      buildInputs = [ pkgs.openmpi ];
      nativeBuildInputs = [ pkgs.openmpi ];
    };
    # tensorrt likes doing a pip invocation from it's setup.py
    # circumvent by manually depending on tensorrt_libs, tensorrt_bindings
    # and setting this env variable
    tensorrt.env.NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP = true;
    tensorrt.mkDerivation.propagatedBuildInputs = with pyPkgs; [
      tensorrt-libs.public
      tensorrt-bindings.public
    ];
    tensorrt-bindings.mkDerivation.propagatedBuildInputs = [ pyPkgs.tensorrt-libs.public ];
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
      mkDerivation.buildInputs = [ cudaPkgs.nccl ];
      mkDerivation.propagatedBuildInputs = with pyPkgs; [
        tensorrt-libs.public # libnvinfer, onnxparse
      ];
      env.appendRunpaths = [ "/usr/lib64" "$ORIGIN" ];
      env.autoPatchelfIgnoreMissingDeps = ["libcuda.so.1"];
    };
    # has some binaries that want cudart
    tritonclient.mkDerivation.postInstall = "rm -r $out/bin";
    nvidia-pytriton.mkDerivation.postInstall = ''
      pushd $out/${site}/pytriton/tritonserver
      mv python_backend_stubs/${py3.pythonVersion}/triton_python_backend_stub backends/python/
      rm -r python_backend_stubs/
      ln -s ${deps.trtllm_backend}/backends/tensorrtllm backends/
      popd
    '';
  };
  # TODO: open-source, switch to fetchFromGitHub
  deps.cog-trt-llm = builtins.fetchGit {
    url = "git@github.com:replicate/cog-trt-llm.git";
    rev = "ee50f890461c4d39eb6e7937aa364abc814e9683";
    ref = "yorickvp/flexibility";
  };
  deps.triton_repo_common = pkgs.fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "common";
    rev = "00b3a71519e32e3bc954e9f0d067e155ef8f1a6c";
    hash = "sha256-KyFicnB0716nIteSNo43RoiDzuVbj17KM4tIbmN6F+s=";
  };
  deps.triton_repo_backend = pkgs.fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "backend";
    rev = "a06e9a1157d6b5b9b34b6d05a07bb84d517f17c9";
    hash = "sha256-Ju2zV/jHUuciTs6GbkqcPG8U0y2lkIWSdAsX78DrpV4=";
  };
  deps.triton_repo_core = pkgs.fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "core";
    rev = "5d4a99c285c729a349265ce8dd7a4535e59d29b1";
    hash = "sha256-WP8bwplo98GmNulX+QA+IrQEc2+GMcTjV53K438vX1g=";
  };
  deps.googletest = pkgs.fetchFromGitHub {
    owner = "google";
    repo = "googletest";
    rev = "9406a60c7839052e4944ea4dbc8344762a89f9bd";
    hash = "sha256-pYoL34KZVjg/bpUQJBEBkjhU6XDDe6yzc1ehe0JfREg=";
  };
  deps.tensorrt_src = pkgs.fetchFromGitHub {
    owner = "NVIDIA";
    repo = "TensorRT";
    rev = "93b6044fc106b69bce6751f27aa9fc198b02bddc"; # release/9.2 branch
    hash = "sha256-W3ytzwq0mm40w6HZ/hArT6G7ID3HSUwzoZ8ix0Q/F6E=";
  };
  # todo: replace with lockfile
  deps.pybind11-stubgen = py3Pkgs.buildPythonPackage rec {
    pname = "pybind11-stubgen";
    version = "2.5";
    src = pkgs.fetchPypi {
      inherit pname version;
      hash = "sha256-lqf+vKski/mKvUu3LMX3KbqHsjRCR0VMF1nmPN6f7zQ=";
    };
  };
    
  deps.tensorrt_llm = pkgs.callPackage ({
    stdenv, cmake, ninja,
      cudaPackages, lib,
      python, withPython ? true }: stdenv.mkDerivation rec {
    pname = "tensorrt_llm";
    version = "0.8.0";
    src = pkgs.fetchFromGitHub {
      owner = "NVIDIA";
      repo = "TensorRT-LLM";
      rev = "v${version}";
      fetchSubmodules = true;
      fetchLFS = true; # libtensorrt_llm_batch_manager_static.a
      hash = "sha256-10wSFhtMGqqCigG5kOBuegptQJymvpO7xCFtgmOOn+k=";
    };
    outputs = if withPython then [ "python" "out" "dist" ] else [ "out" ];
    setSourceRoot = "sourceRoot=$(echo */cpp)";
    nativeBuildInputs = [
      cmake
      ninja
      python
    ];
    buildInputs = with pkgs; [
      cudaPackages.cuda_nvcc
      cudaPackages.cudnn
      cudaPackages.nccl
      openmpi
    ] ++ (lib.optionals (!withPython) [
      # torch hates the split cuda, so only do it without torch
      cudaPackages.cuda_cudart
      cudaPackages.cuda_nvcc.dev
      cudaPackages.cuda_cccl
      cudaPackages.libcublas.lib
    ]) ++ (lib.optionals withPython [
      cudaPackages.cudatoolkit
      python.pkgs.pybind11
      python.pkgs.setuptools
      python.pkgs.wheel
      python.pkgs.pip
      deps.pybind11-stubgen
    ]);
    propagatedBuildInputs = lib.optionals withPython (with config.python-env.pip.drvs; builtins.map (x: x.public or x) [
      accelerate #==0.25.0
      build
      colored
      # concerning statement from trtllm's requirements.txt:
      cuda-python # "Do not override the custom version of cuda-python installed in the NGC PyTorch image."
      diffusers # ==0.15.0
      lark
      mpi4py
      numpy
      onnx # >=1.12.0
      polygraphy
      psutil
      pynvml # >=11.5.0
      sentencepiece # >=0.1.99
      tensorrt # ==9.2.0.post12.dev5
      tensorrt-bindings # missed transitive dep
      tensorrt-libs
      torch # <=2.2.0a
      nvidia-ammo # ~=0.7.0; platform_machine=="x86_64"
      transformers # ==4.36.1
      wheel
      optimum
      evaluate
      janus
    ]);

    cmakeFlags = [
      "-DBUILD_PYT=${if withPython then "ON" else "OFF"}"
      "-DBUILD_PYBIND=${if withPython then "ON" else "OFF"}" # needs BUILD_PYT
      "-DBUILD_TESTS=OFF" # needs nvonnxparser.h
      # believe it or not, this is the actual binary distribution channel for tensorrt:
      "-DTRT_LIB_DIR=${config.python-env.pip.drvs.tensorrt-libs.public}/${site}/tensorrt_libs"
      "-DTRT_INCLUDE_DIR=${deps.tensorrt_src}/include"
      "-DCMAKE_CUDA_ARCHITECTURES=${builtins.concatStringsSep ";" cfg.architectures}"
      # "-DCUDAToolkit_INCLUDE_DIR=${cudaPkgs.cuda_cudart}/include"
      "-DCUDAToolkit_INCLUDE_DIR=${cudaPkgs.cudatoolkit}/include"
      "-DCMAKE_SKIP_BUILD_RPATH=ON" # todo test without this, might fail /build check
    ];
    postBuild = lib.optionalString withPython ''
      pushd ../../
      chmod -R +w .
      mkdir ./libs
      cp -r cpp/build/tensorrt_llm/libtensorrt_llm.so ./libs
      cp -r cpp/build/tensorrt_llm/thop/libth_common.so ./libs
      cp -r cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so* ./libs
      cp -r cpp/build/tensorrt_llm/pybind/bindings.*.so .
      python -m pybind11_stubgen -o . bindings
      mv bindings libs bindings.*.so tensorrt_llm
      python setup.py bdist_wheel
      popd
    '';
    # todo pythonOutputDistHook
    installPhase = ''
      mkdir -p $out
      cp -r $src/cpp $out/
      chmod -R u+w $out/cpp
      mkdir -p $out/cpp/build/tensorrt_llm/plugins
      pushd tensorrt_llm
      cp ./libtensorrt_llm.so $out/cpp/build/tensorrt_llm/
      cp ./plugins/libnvinfer_plugin_tensorrt_llm.so* $out/cpp/build/tensorrt_llm/plugins/
      for f in $out/cpp/build/tensorrt_llm/plugins/*.so*; do
        patchelf --add-rpath '$ORIGIN/..' $f
      done
      popd
    '' + (lib.optionalString withPython ''
      mv ../../dist $dist
      pushd $dist
      python -m pip install ./*.whl --no-index --no-warn-script-location --prefix="$python" --no-cache
      popd
    '');
    postFixup = lib.optionalString withPython ''
      mv $out/nix-support $python/
    '';
  }) {
    python = pkgs.python310;
    cudaPackages = cudaPkgs;
    # TODO: turn into config option
    withPython = false;
  };
  deps.trtllm_backend = let
    trt_lib_dir = "${config.python-env.pip.drvs.tensorrt-libs.public}/${site}/tensorrt_libs";
    # this package wants gcc12
    oldGccStdenv = pkgs.stdenvAdapters.useLibsFrom pkgs.stdenv pkgs.gcc12Stdenv;
    # todo don't mix this and cudaPkgs.cudnn:
    cudnn = "${config.python-env.pip.drvs.nvidia-cudnn-cu12.public}/${site}/nvidia/cudnn";
  in oldGccStdenv.mkDerivation rec {
    pname = "tensorrtllm_backend";
    version = "0.8.0";
    src = pkgs.fetchFromGitHub {
      owner = "triton-inference-server";
      repo = "tensorrtllm_backend";
      rev = "v${version}";
      hash = "sha256-5t8ByQxzfF4Td4HfnOYioVxJfZxOX2TV8a5Qg6YDmSQ=";
    };
    nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.python310 ];
    buildInputs = [ pkgs.rapidjson cudaPkgs.cudatoolkit pkgs.openmpi ];
    sourceRoot = "source/inflight_batcher_llm";
    cmakeFlags = [
      "-DFETCHCONTENT_SOURCE_DIR_REPO-COMMON=${deps.triton_repo_common}"
      "-DFETCHCONTENT_SOURCE_DIR_REPO-BACKEND=${deps.triton_repo_backend}"
      "-DFETCHCONTENT_SOURCE_DIR_REPO-CORE=${deps.triton_repo_core}"
      "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${deps.googletest}"
      "-DFETCHCONTENT_SOURCE_DIR_JSON=${pkgs.nlohmann_json.src}"
      "-DTRT_LIB_DIR=${trt_lib_dir}"
      "-DTRT_INCLUDE_DIR=${deps.tensorrt_src}/include"
      "-DTRTLLM_DIR=${deps.tensorrt_llm}" # todo: not src
    ];
    # buildInputs = [ pkgs.tensorrt-llm ];
    # linking to stubs/libtritonserver.so is maybe a bit shady
    # $out/lib/stubs
    # todo fix that hash
    # todo I think tensorrt_llm.so itself should have the cudnn dep
    postFixup = ''
      patchelf $out/backends/tensorrtllm/libtriton_tensorrtllm.so --add-rpath ${trt_lib_dir}:${deps.tensorrt_llm}/cpp/build/tensorrt_llm:${deps.tensorrt_llm}/cpp/build/tensorrt_llm/plugins:${cudnn}/lib --replace-needed libtritonserver.so libtritonserver-90a4cf82.so --add-needed libcudnn.so.8
    '';
  };
}
