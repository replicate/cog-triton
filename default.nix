{ pkgs, config, ... }:
let
  deps = config.deps;
  py3Pkgs = pkgs.python310.pkgs;
  cudaPkgs = pkgs.cudaPackages_12_2;
  site = pkgs.python310.sitePackages;
in
{
  cog.build = {
    python_version = "3.10";
    cog_version = "0.8.6";
    cuda = "12.1";
    gpu = true;
    python_packages = [
      "--extra-index-url"
      "https://pypi.nvidia.com"
      "tensorrt_llm==0.7.1"
      "torch==2.1.2"
      # from tensorrt
      "tensorrt_libs==9.2.0.post12.dev5"
      "tensorrt_bindings==9.2.0.post12.dev5"
      # fixed in torch 2.2
      "nvidia-nccl-cu12"
      "nvidia-pytriton"
    ];
  };
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
    tensorrt-llm = {
      mkDerivation.propagatedBuildInputs = with pyPkgs; [
        tensorrt-libs.public # libnvinfer, onnxparse
        # fixed in torch 2.2
        nvidia-nccl-cu12.public
      ];
      env.appendRunpaths = [ "/usr/lib64" "$ORIGIN" ];
      env.autoPatchelfIgnoreMissingDeps = ["libcuda.so.1"];
    };
    # has some binaries that want cudart
    tritonclient.mkDerivation.postInstall = "rm -r $out/bin";
    # todo put the python backend stub in the right location
    nvidia-pytriton.mkDerivation.postInstall = ''
      rm $out/${site}/pytriton/tritonserver/python_backend_stubs/3.{8,9,11}/triton_python_backend_stub
    '';
  };
  deps.triton_repo_common = pkgs.fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "common";
    rev = "bf4b16304c9ba1baff3a0d0a4f7c2e1ce949f510";
    hash = "sha256-ztvpjYeaRU7jAcRhLbJkjFVA6/SSa2Y+BphvOzaPfOM=";
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
    rev = "854bdcbe13fec20a695130f21811ca22cb4e1a13";
    hash = "sha256-JWmvULKt3YkaFLYL0WqQ/T+psnjznys2YTMyeZO4CLg=";
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
  deps.pybind11-stubgen = let py3Pkgs = pkgs.python310.pkgs; in py3Pkgs.buildPythonPackage rec {
    pname = "pybind11-stubgen";
    version = "2.4.2";
    src = pkgs.fetchPypi {
      inherit pname version;
      hash = "sha256-6b992wbUqpSobzkP5y+9P2awsGXbJGLaCJVVMElnSxw=";
    };
  };
    
  deps.tensorrt_llm = pkgs.callPackage ({
    stdenv, cmake, ninja,
      cudaPackages, lib,
      python, withPython ? true }: stdenv.mkDerivation rec {
    pname = "tensorrt_llm";
    version = "0.7.1";
    src = pkgs.fetchFromGitHub {
      owner = "NVIDIA";
      repo = "TensorRT-LLM";
      rev = "v${version}";
      fetchSubmodules = true;
      fetchLFS = true; # libtensorrt_llm_batch_manager_static.a
      hash = "sha256-CezACpWlUFmBGVOV6UDQ3EiejRLinoLFxXk2AOfKaec=";
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
      # todo separate cublas and cuda_cudart
      # cudaPackages.cuda_cudart
      cudaPackages.cudatoolkit # should be just cublas, torch cmake hates it
      # cudaPackages.cuda_nvcc.dev
      # cudaPackages.cuda_cccl
      openmpi
    ] ++ (lib.optionals withPython [
      python.pkgs.pybind11
      python.pkgs.setuptools
      python.pkgs.wheel
      python.pkgs.pip
      deps.pybind11-stubgen
    ]);
    propagatedBuildInputs = lib.optionals withPython (with config.python-env.pip.drvs; builtins.map (x: x.public or x) [
      accelerate # 0.20.3
      build
      colored
      torch
      numpy
      cuda-python # 12.2.0
      diffusers # 0.15.0
      lark
      mpi4py
      onnx # >= 1.12.0
      polygraphy
      tensorrt # = 9.2.0.post12.dev5
      tensorrt-bindings # = 9.2.0.post12.dev5
      tensorrt-libs # = 9.2.0.post12.dev5
      sentencepiece # >=0.1.99
      transformers # 4.33.1
      wheel
      optimum
      evaluate
    ]);

    cmakeFlags = [
      "-DBUILD_PYT=${if withPython then "ON" else "OFF"}"
      "-DBUILD_PYBIND=${if withPython then "ON" else "OFF"}" # needs BUILD_PYT
      "-DBUILD_TESTS=OFF" # needs nvonnxparser.h
      # believe it or not, this is the actual binary distribution channel for tensorrt:
      "-DTRT_LIB_DIR=${config.python-env.pip.drvs.tensorrt-libs.public}/${site}/tensorrt_libs"
      "-DTRT_INCLUDE_DIR=${deps.tensorrt_src}/include"
      "-DCMAKE_CUDA_ARCHITECTURES=86-real" # just a5000, a40, ain't got all day
      # "-DCUDAToolkit_INCLUDE_DIR=${cudaPkgs.cuda_cudart}/include"
      "-DCUDAToolkit_INCLUDE_DIR=${cudaPkgs.cudatoolkit}/include"
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
      cp ./libtensorrt_llm_static.a $out/cpp/build/tensorrt_llm/
      cp ./plugins/libnvinfer_plugin_tensorrt_llm.so* $out/cpp/build/tensorrt_llm/plugins/
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
    withPython = false;
  };
  deps.trtllm_backend = let
    trt_lib_dir = "${config.python-env.pip.drvs.tensorrt-libs.public}/${site}/tensorrt_libs";
    # this package wants gcc12
    oldGccStdenv = pkgs.stdenvAdapters.useLibsFrom pkgs.stdenv pkgs.gcc12Stdenv;
  in oldGccStdenv.mkDerivation rec {
    pname = "tensorrtllm_backend";
    version = "0.7.1";
    src = pkgs.fetchFromGitHub {
      owner = "triton-inference-server";
      repo = "tensorrtllm_backend";
      rev = "v${version}";
      hash = "sha256-5+a/+YCl7FuwlQFwWMHgEzPArAr8npLH7qTJ+Sm5Cns=";
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
    postFixup = ''
      patchelf $out/backends/tensorrtllm/libtriton_tensorrtllm.so --add-rpath $out/lib/stubs:${trt_lib_dir}:${deps.tensorrt_llm}/cpp/build/tensorrt_llm/plugins
    '';
  };
}
