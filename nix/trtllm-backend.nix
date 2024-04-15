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
    rev = "00b3a71519e32e3bc954e9f0d067e155ef8f1a6c";
    hash = "sha256-KyFicnB0716nIteSNo43RoiDzuVbj17KM4tIbmN6F+s=";
  };
  deps.triton_repo_backend = fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "backend";
    rev = "a06e9a1157d6b5b9b34b6d05a07bb84d517f17c9";
    hash = "sha256-Ju2zV/jHUuciTs6GbkqcPG8U0y2lkIWSdAsX78DrpV4=";
  };
  # todo: update with trt-llm 0.9?
  deps.triton_repo_core = fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "core";
    rev = "5d4a99c285c729a349265ce8dd7a4535e59d29b1";
    hash = "sha256-WP8bwplo98GmNulX+QA+IrQEc2+GMcTjV53K438vX1g=";
  };
  deps.googletest = fetchFromGitHub {
    owner = "google";
    repo = "googletest";
    rev = "9406a60c7839052e4944ea4dbc8344762a89f9bd";
    hash = "sha256-pYoL34KZVjg/bpUQJBEBkjhU6XDDe6yzc1ehe0JfREg=";
  };

  inherit (python3) sitePackages;
  trt_lib_dir = "${pythonDrvs.tensorrt-libs.public}/${sitePackages}/tensorrt_libs";
  # this package wants gcc12
  oldGccStdenv = stdenvAdapters.useLibsFrom stdenv gcc12Stdenv;
in
oldGccStdenv.mkDerivation rec {
  pname = "tensorrtllm_backend";
  version = "0.9.0";
  src = fetchFromGitHub {
    owner = "triton-inference-server";
    repo = "tensorrtllm_backend";
    rev = "v${version}";
    hash = "sha256-aNjVYu7sDrIj/lse/wS3vYaR/vmjtZfxzBWYi3z3KqQ=";
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
    cp libtriton_*.so triton_tensorrtllm_worker $out/backends/tensorrtllm
    rm -r /build/source/inflight_batcher_llm/build/_deps/repo-core-build
    rm -r /build/source/inflight_batcher_llm/build/libtriton_tensorrtllm_common.so
  '';
  # buildInputs = [ tensorrt-llm ];
  postFixup = ''
    patchelf $out/backends/tensorrtllm/libtriton_tensorrtllm.so \
      --add-rpath '$ORIGIN:${trt_lib_dir}:${tensorrt-llm}/cpp/build/tensorrt_llm:${tensorrt-llm}/cpp/build/tensorrt_llm/plugins:${cudaPackages.cudnn.lib}/lib'
    patchelf $out/backends/tensorrtllm/libtriton_tensorrtllm_common.so \
      --add-rpath '$ORIGIN:${trt_lib_dir}:${tensorrt-llm}/cpp/build/tensorrt_llm:${tensorrt-llm}/cpp/build/tensorrt_llm/plugins:${cudaPackages.cudnn.lib}/lib'
    patchelf $out/backends/tensorrtllm/triton_tensorrtllm_worker \
      --add-rpath '$ORIGIN:${trt_lib_dir}:${tensorrt-llm}/cpp/build/tensorrt_llm:${tensorrt-llm}/cpp/build/tensorrt_llm/plugins:${cudaPackages.cudnn.lib}/lib'
  '';
}
