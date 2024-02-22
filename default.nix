{ pkgs, config, ... }:
let deps = config.deps; in
{
  cog.build = {
    python_version = "3.10";
    cuda = "12.1";
    gpu = true;
    python_packages = [
      "--extra-index-url"
      "https://pypi.nvidia.com"
      "tensorrt_llm"
      "torch==2.1.2"
      # from tensorrt
      "tensorrt_libs==9.2.0.post12.dev5"
      "tensorrt_bindings==9.2.0.post12.dev5"
      # fixed in torch 2.2
      "nvidia-nccl-cu12"
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
    tensorrt-bindings.mkDerivation.propagatedBuildInputs = [ pyPkgs.tensorrt-libs.public ];
    # fixed in torch 2.2
    torch.mkDerivation.postFixup = ''
      pushd $out/lib/python3.10/site-packages/torch/lib
      ln -s libcudart-*.so.12 libcudart.so.12
      ln -s libnvrtc-*.so.12 libnvrtc.so.12
      ln -s libnvToolsExt-*.so.1 libnvToolsExt.so.1
      popd
    '';
    tensorrt-libs.mkDerivation.postFixup = ''
      pushd $out/lib/python3.10/site-packages/tensorrt_libs
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
    
  deps.tensorrt_llm = pkgs.stdenv.mkDerivation rec {
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
    sourceRoot = "source/cpp";
    nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.python3 ];
    buildInputs = with pkgs; [
      rapidjson # not sure?
      # python bindings need torch
      cudaPackages_12_1.cudatoolkit
      cudaPackages_12_1.cudnn
      cudaPackages_12_1.nccl
      openmpi
    ];
    cmakeFlags = [
      "-DBUILD_PYT=OFF" # needs torch
      "-DBUILD_PYBIND=OFF"
      "-DBUILD_TESTS=OFF" # needs nvonnxparser.h
      # believe it or not, this is the actual binary distribution channel for tensorrt:
      "-DTRT_LIB_DIR=${config.python-env.pip.drvs.tensorrt-libs.public}/lib/python3.10/site-packages/tensorrt_libs"
      "-DTRT_INCLUDE_DIR=${deps.tensorrt_src}/include"
      "-DCMAKE_CUDA_ARCHITECTURES=86-real" # just a5000, a40, ain't got all day
    ];
    installPhase = ''
      mkdir -p $out
      cp -r $src/cpp $out/
      chmod -R u+w $out/cpp
      mkdir -p $out/cpp/build/tensorrt_llm/plugins
      pushd tensorrt_llm
      cp ./libtensorrt_llm.so $out/cpp/build/tensorrt_llm/
      cp ./libtensorrt_llm_static.a $out/cpp/build/tensorrt_llm/
      cp ./plugins/libnvinfer_plugin_tensorrt_llm.so* $out/cpp/build/tensorrt_llm/plugins/
    '';
  };
  deps.trtllm_backend = let
    trt_lib_dir = "${config.python-env.pip.drvs.tensorrt-libs.public}/lib/python3.10/site-packages/tensorrt_libs";
  in pkgs.stdenv.mkDerivation rec {
    pname = "tensorrtllm_backend";
    version = "0.7.1";
    src = pkgs.fetchFromGitHub {
      owner = "triton-inference-server";
      repo = "tensorrtllm_backend";
      rev = "v${version}";
      hash = "sha256-5+a/+YCl7FuwlQFwWMHgEzPArAr8npLH7qTJ+Sm5Cns=";
    };
    nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.python3 ];
    buildInputs = [ pkgs.rapidjson pkgs.cudaPackages_12_1.cudatoolkit pkgs.openmpi ];
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
    # buildInputs = [ pkgs.tensorrt-llm pkgs.inflight_batcher_llm ];
    # linking to stubs/libtritonserver.so is maybe a bit shady
    postFixup = ''
      patchelf $out/backends/tensorrtllm/libtriton_tensorrtllm.so --add-rpath $out/lib/stubs:${trt_lib_dir}:${deps.tensorrt_llm}/cpp/build/tensorrt_llm/plugins
    '';
  };
}
