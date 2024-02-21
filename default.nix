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
  deps.tensorrt = pkgs.fetchFromGitHub {
    owner = "NVIDIA";
    repo = "TensorRT";
    rev = "v9.2.0";
    hash = "sha256-Yo9CsHwu8hIPQwigePIwHu7UWtfROuMQFYtC/QIMTO0=";
  };
    
  deps.tensorrt_llm = pkgs.stdenv.mkDerivation rec {
    pname = "tensorrt_llm";
    version = "0.7.1";
    src = pkgs.fetchFromGitHub {
      owner = "NVIDIA";
      repo = "TensorRT-LLM";
      rev = "v0.7.1";
      hash = "sha256-24BrTOs8S1q2+Y9p1laW0BRPhDPkzv8DbqKZ6EjZvhQ=";
    };
    sourceRoot = "source/cpp";
    nativeBuildInputs = [ pkgs.cmake pkgs.ninja pkgs.python3 ];
    buildInputs = [ pkgs.rapidjson pkgs.cudaPackages_12_1.cudatoolkit pkgs.openmpi ];
    cmakeFlags = [
      "-DBUILD_PYT=OFF"
      "-DBUILD_PYBIND=OFF"
      "-DTRT_LIB_DIR=${config.python-env.pip.drvs.tensorrt-libs.public}/lib/python3.10/site-packages/tensorrt_libs"
      "-DTRT_INCLUDE_DIR=${deps.tensorrt}/include"
      "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${deps.googletest}"
    ];
  };
  deps.trtllm_backend = pkgs.stdenv.mkDerivation rec {
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
      #"-DCMAKE_INSTALL_PREFIX=${out}"
      #"-DCMAKE_BUILD_TYPE=Release"
      "-DFETCHCONTENT_SOURCE_DIR_REPO-COMMON=${deps.triton_repo_common}"
      "-DFETCHCONTENT_SOURCE_DIR_REPO-BACKEND=${deps.triton_repo_backend}"
      "-DFETCHCONTENT_SOURCE_DIR_REPO-CORE=${deps.triton_repo_core}"
      "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${deps.googletest}"
      "-DFETCHCONTENT_SOURCE_DIR_JSON=${pkgs.nlohmann_json.src}"
      "-DTRT_LIB_DIR=${config.python-env.pip.drvs.tensorrt-libs.public}"
      "-DTRT_INCLUDE_DIR=${deps.tensorrt}/include"
      "-DTRTLLM_DIR=${deps.tensorrt_llm.src}" # todo: not src
    ];
    # buildInputs = [ pkgs.tensorrt-llm pkgs.inflight_batcher_llm ];
  };
}
