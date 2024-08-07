{
  stdenv,
  cmake,
  ninja,
  runCommand,
  cudaPackages,
  lib,
  fetchFromGitHub,
  openmpi,
  tensorrt-src,
  architectures,
  python3,
  pythonDrvs,
  pybind11-stubgen ? null,
  withPython ? true,
  rsync,
  zstd,
  autoPatchelfHook,
  patchelfUnstable,
}:
stdenv.mkDerivation (o: {
  pname = "tensorrt_llm";
  version = "0.12.0.dev2024073000";
  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "TensorRT-LLM";
    rev = "a681853d3803ee5893307e812530b5e7004bb6e1";
    fetchSubmodules = true;
    fetchLFS = true; # libtensorrt_llm_batch_manager_static.a
    hash = "sha256-Uvx8+Lhuo8lT4TqKjYSL0Mt/QI8jS5T9kxdsNGKJZzU=";
  };
  outputs =
    if withPython then
      [
        "python"
        "out"
        "dist"
      ]
    else
      [ "out" ];
  setSourceRoot = "sourceRoot=$(echo */cpp)";
  nativeBuildInputs = [
    patchelfUnstable
    zstd
    cmake
    ninja
    python3
    cudaPackages.cuda_nvcc
    rsync
    autoPatchelfHook
  ];
  buildInputs =
    [
      cudaPackages.cudnn.lib
      cudaPackages.cudnn.dev
      cudaPackages.nccl
      openmpi
      python3.pkgs.setuptools
    ]
    ++ (with cudaPackages; [
      cuda_cudart
      cuda_nvcc.dev
      cuda_nvrtc.dev
      cuda_nvrtc.lib
      cuda_nvml_dev.lib
      cuda_nvml_dev.dev
      cuda_cccl
      libcublas.lib
      libcublas.dev
      libcurand.dev
      cuda_profiler_api
    ])
    ++ (lib.optionals withPython (with cudaPackages; [
      cuda_nvtx.dev cuda_nvtx.lib
      libcusparse.dev libcusparse.lib
      libcusolver.dev libcusolver.lib
      python3.pkgs.pybind11
      python3.pkgs.wheel
      python3.pkgs.pip
      pybind11-stubgen
    ]));
  env.pythonRelaxDeps = "nvidia-cudnn-cu12";
  propagatedBuildInputs = lib.optionals withPython (
    with pythonDrvs;
    builtins.map (x: x.public or x) [
      accelerate
      build
      colored
      cuda-python # Do not override the custom version of cuda-python installed in the NGC PyTorch image.
      diffusers
      lark
      mpi4py
      numpy
      onnx
      polygraphy
      psutil
      pynvml
      pulp
      pandas
      h5py
      strenum
      sentencepiece
      tensorrt
      torch
      nvidia-modelopt
      transformers
      pillow
      wheel
      optimum
      evaluate
      janus
      mpmath
    ]
  );
  autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" "libnvidia-ml.so.1" ];

  # tries to run cutlass's `python setup.py develop`
  PYTHONUSERBASE = "/tmp/python";
  preConfigure = ''
    mkdir -p $PYTHONUSERBASE
    chmod -R +w ../3rdparty/cutlass/python
    export PYTHONPATH=$PYTHONPATH:$src/3rdparty/cutlass/python
  '';

  cmakeFlags = [
    "-DBUILD_PYT=${if withPython then "ON" else "OFF"}"
    "-DBUILD_PYBIND=${if withPython then "ON" else "OFF"}" # needs BUILD_PYT
    "-DBUILD_TESTS=OFF" # needs nvonnxparser.h
    # believe it or not, this is the actual binary distribution channel for tensorrt:
    "-DTRT_LIB_DIR=${pythonDrvs.tensorrt-cu12-libs.public}/${python3.sitePackages}/tensorrt_libs"
    "-DTRT_INCLUDE_DIR=${tensorrt-src}/include"
    "-DCMAKE_CUDA_ARCHITECTURES=${builtins.concatStringsSep ";" architectures}"
    # "-DFAST_BUILD=ON"
    "-DCMAKE_SKIP_BUILD_RPATH=ON"
  ];
  # workaround: cuda_nvcc exposes a gcc12 that uses a gcc13 libc
  # however, cmake finds the gcc12 libc somehow, which is wrong
  postConfigure = ''
    sed -i 's#${cudaPackages.cuda_nvcc.stdenv.cc.cc.lib}#${stdenv.cc.cc.lib}#g' build.ninja
  '';
  # include cstdint in cpp/tensorrt_llm/common/mpiUtils.h after pragma once
  postPatch = ''
    sed -i 's/#include <mpi.h>/#include <mpi.h>\n#include <cstdint>/' include/tensorrt_llm/common/mpiUtils.h
    sed -i 's/#pragma once/#pragma once\n#include <cuda_runtime.h>/' tensorrt_llm/kernels/lruKernel.h
  '';
  # configurePhase = "true";
  # buildPhase = ''
  #   tar xf ${/home/yorick/datakami/r8/cog-triton-r8/build-dir.tar.zst}
  #   cd source/cpp/build/
  #   runHook postBuild
  # '';
  # libtensorrt_llm.so _sometimes_ wants libcudnn, so --add-needed to prevent it from being shrunk out
  postBuild = ''
    patchelf --add-needed 'libcudnn.so.8' --add-rpath ${cudaPackages.cudnn.lib}/lib tensorrt_llm/libtensorrt_llm.so
  '' + (lib.optionalString withPython ''
    pushd ../../
    chmod -R +w .
    mkdir -p ./libs
    cp -ar cpp/build/tensorrt_llm/libtensorrt_llm.so ./libs
    cp -ar cpp/build/tensorrt_llm/thop/libth_common.so ./libs
    cp -ar cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so* ./libs
    cp -ar cpp/build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/libtensorrt_llm_nvrtc_wrapper.so ./libs
    cp -ar cpp/build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention.so ./libs
    mkdir -p ./bin
    cp -r cpp/build/tensorrt_llm/executor_worker/executorWorker ./bin
    cp -r cpp/build/tensorrt_llm/pybind/bindings.*.so .

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${cudaPackages.cuda_cudart.stubs}/lib python -m pybind11_stubgen -o . bindings
    rm -rf tensorrt_llm/{bin,bindings,libs}
    mv bin bindings libs bindings.*.so tensorrt_llm
    patchelf --replace-needed libnvinfer_plugin_tensorrt_llm.so.10 libnvinfer_plugin_tensorrt_llm.so --add-rpath '$ORIGIN/../libs' ./tensorrt_llm/bin/executorWorker
    python setup.py bdist_wheel
    popd
  '');
  # noAuditTmpdir = true;
  # todo pythonOutputDistHook
  # Install isn't well-defined, -backend just expects the build directory to exist somewhere.
  # Since we just copy build outputs, cmake doesn't get a chance to relink with the correct rpath.
  installPhase =
    ''
      mkdir -p $out
      rsync -a --chmod=u+w --include "tensorrt_llm/kernels/" --include "tensorrt_llm/kernels/kvCacheIndex.h" --exclude "tensorrt_llm/kernels/*" $src/cpp $out/
      pushd $src/cpp/tensorrt_llm
      find . '(' '(' -type f -executable ')' -or -type l ')' -print0 | rsync -av --chmod=u+w --files-from=- --from0 ./ $out/cpp/tensorrt_llm/
      popd
      # rsync -a --chmod=u+w $src/cpp/tensorrt_llm/kernels $out/cpp/tensorrt_llm/
      pushd tensorrt_llm
      mkdir -p $out/cpp/build/tensorrt_llm/
      find . '(' '(' -type f -executable ')' -or -type l ')' -print0 | rsync -av --chmod=u+w --files-from=- --from0 ./ $out/cpp/build/tensorrt_llm/
      popd
    ''
    + (lib.optionalString withPython ''
      mv ../../dist $dist
      pushd $dist
      python -m pip install ./*.whl --no-index --no-warn-script-location --prefix="$python" --no-cache --no-deps
      popd
    '');
  # manually call autoPatchelf so it doesn't cross-link the outputs
  dontAutoPatchelf = true;
  # move the propagatedBuildInputs to $python
  postFixup = lib.optionalString withPython ''
    mv $out/nix-support $python/
    autoPatchelf $out
    autoPatchelf $python
  '';
  # imports check, wants nvml
  # pushd $python/${python3.sitePackages}
  # python -c "import tensorrt_llm.bindings"
  # popd
  passthru.examples = runCommand "trt-examples" {} ''
    mkdir $out
    cp -r ${o.src}/examples $out/examples
  '';
  passthru.pythonModule = python3;
})
