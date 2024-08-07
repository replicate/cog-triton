{
  stdenv,
  lib,
  stdenvAdapters,
  gcc12Stdenv,
  fetchFromGitHub,
  cudaPackages,
  cmake,
  ninja,
  openmpi,
  rapidjson,
  nlohmann_json,
  python3,
  pythonDrvs,
  tensorrt-llm,
  tensorrt-src,
}:
let
  deps.triton_repo_common = fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "common";
    rev = "0f2072bbc2d4e85f68b10cf60c0ed4e4ebfc766b";
    hash = "sha256-7DdJ1zkHrFEImI137Gt/pDKZhBvoQu0lg2ulqA/yLFA=";
  };
  deps.triton_repo_backend = fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "backend";
    # update for tritons after may 28, 2024
    rev = "a06e9a1157d6b5b9b34b6d05a07bb84d517f17c9";
    hash = "sha256-Ju2zV/jHUuciTs6GbkqcPG8U0y2lkIWSdAsX78DrpV4=";
  };
  deps.triton_repo_core = fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "core";
    rev = "bbcd7816997046821f9d1a22e418acb84ca5364b";
    hash = "sha256-LWLxMvtV0VQYMQQIfztm10xzQreNAoN9zAexf+5ktHo=";
  };
  deps.googletest = fetchFromGitHub {
    owner = "google";
    repo = "googletest";
    rev = "9406a60c7839052e4944ea4dbc8344762a89f9bd";
    hash = "sha256-pYoL34KZVjg/bpUQJBEBkjhU6XDDe6yzc1ehe0JfREg=";
  };

  inherit (python3) sitePackages;
  trt_lib_dir = "${pythonDrvs.tensorrt-cu12-libs.public}/${sitePackages}/tensorrt_libs";
  # this package wants gcc12
  oldGccStdenv = stdenvAdapters.useLibsFrom stdenv gcc12Stdenv;
in
oldGccStdenv.mkDerivation rec {
  pname = "tensorrtllm_backend";
  version = "0.12.0.dev2024073000";
  src = fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "tensorrtllm_backend";
    rev = "b25d578a48422db3b2d5bd89b16c235dd85c4300";
    hash = "sha256-UxuMdhkMv89Ozxi4jXioOfR1gf/cYr/bCxt/RG6CdZw=";
  };
  nativeBuildInputs = [
    cmake
    ninja
    python3
    cudaPackages.cuda_nvcc
  ];
  buildInputs = [
    rapidjson
    openmpi

    cudaPackages.cuda_cudart
    cudaPackages.cuda_cccl
    cudaPackages.libcublas.lib
    cudaPackages.libcublas.dev
    cudaPackages.cuda_nvml_dev.lib
    cudaPackages.cuda_nvml_dev.dev
  ];
  sourceRoot = "source/inflight_batcher_llm";
  cmakeFlags = [
    "-DFETCHCONTENT_SOURCE_DIR_REPO-COMMON=${deps.triton_repo_common}"
    "-DFETCHCONTENT_SOURCE_DIR_REPO-BACKEND=${deps.triton_repo_backend}"
    "-DFETCHCONTENT_SOURCE_DIR_REPO-CORE=${deps.triton_repo_core}"
    "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${deps.googletest}"
    "-DFETCHCONTENT_SOURCE_DIR_JSON=${nlohmann_json.src}"
    "-DTRT_LIB_DIR=${trt_lib_dir}"
    "-DTRT_INCLUDE_DIR=${tensorrt-src}/include"
    "-DTRTLLM_DIR=${tensorrt-llm}"
  ];
  postInstall = ''
    mkdir -p $out/backends/tensorrtllm
    cp libtriton_*.so trtllmExecutorWorker $out/backends/tensorrtllm
    rm -r /build/source/inflight_batcher_llm/build/_deps/repo-core-build
    rm -r /build/source/inflight_batcher_llm/build/libtriton_tensorrtllm_common.so
  '';
  # buildInputs = [ tensorrt-llm ];
  postFixup = ''
    patchelf $out/backends/tensorrtllm/libtriton_tensorrtllm.so \
      --add-rpath '$ORIGIN:${trt_lib_dir}:${tensorrt-llm}/cpp/build/tensorrt_llm:${tensorrt-llm}/cpp/build/tensorrt_llm/plugins:${cudaPackages.cudnn.lib}/lib'
    patchelf $out/backends/tensorrtllm/libtriton_tensorrtllm_common.so \
      --add-rpath '$ORIGIN:${trt_lib_dir}:${tensorrt-llm}/cpp/build/tensorrt_llm:${tensorrt-llm}/cpp/build/tensorrt_llm/plugins:${cudaPackages.cudnn.lib}/lib'
    patchelf $out/backends/tensorrtllm/trtllmExecutorWorker  \
      --add-rpath '$ORIGIN:${trt_lib_dir}:${tensorrt-llm}/cpp/build/tensorrt_llm:${tensorrt-llm}/cpp/build/tensorrt_llm/plugins:${cudaPackages.cudnn.lib}/lib:${tensorrt-llm}/cpp/build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention:${tensorrt-llm}/cpp/build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper'
  '';
}
