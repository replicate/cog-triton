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
}:
stdenv.mkDerivation (o: {
  pname = "tensorrt_llm";
  version = "0.11.0";
  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "TensorRT-LLM";
    rev = "v${o.version}";
    fetchSubmodules = true;
    fetchLFS = true; # libtensorrt_llm_batch_manager_static.a
    hash = "sha256-J2dqKjuEXVbE9HgoCzhUASZAnsn/hsC+qUTHL6uT4nU=";
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
    cmake
    ninja
    python3
    cudaPackages.cuda_nvcc
    rsync
  ];
  buildInputs =
    [
      cudaPackages.cudnn.lib
      cudaPackages.cudnn.dev
      cudaPackages.nccl
      openmpi
      python3.pkgs.setuptools
    ]
    ++ (lib.optionals (!withPython) [
      # torch hates the split cuda, so only do it without torch
      cudaPackages.cuda_cudart
      cudaPackages.cuda_nvcc.dev
      cudaPackages.cuda_nvrtc.dev
      cudaPackages.cuda_nvrtc.lib
      cudaPackages.cuda_nvml_dev.lib
      cudaPackages.cuda_nvml_dev.dev
      cudaPackages.cuda_cccl
      cudaPackages.libcublas.lib
      cudaPackages.libcublas.dev
      cudaPackages.libcurand.dev
      cudaPackages.cuda_profiler_api
    ])
    ++ (lib.optionals withPython [
      cudaPackages.cudatoolkit
      python3.pkgs.pybind11
      python3.pkgs.wheel
      python3.pkgs.pip
      pybind11-stubgen
    ]);
  propagatedBuildInputs = lib.optionals withPython (
    with pythonDrvs;
    builtins.map (x: x.public or x) [
      accelerate # ==0.25.0
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
      tensorrt-cu12-bindings # missed transitive dep
      tensorrt-cu12-libs
      torch # <=2.2.0a
      nvidia-ammo # ~=0.7.0; platform_machine=="x86_64"
      transformers # ==4.36.1
      wheel
      optimum
      evaluate
      janus
    ]
  );
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
  ];
  # include cstdint in cpp/tensorrt_llm/common/mpiUtils.h after pragma once
  postPatch = ''
    sed -i 's/#include <mpi.h>/#include <mpi.h>\n#include <cstdint>/' /build/source/cpp/include/tensorrt_llm/common/mpiUtils.h
    sed -i 's/#pragma once/#pragma once\n#include <cuda_runtime.h>/' /build/source/cpp/tensorrt_llm/kernels/lruKernel.h
  '';
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
  # Install isn't well-defined, -backend just expects the build directory to exist somewhere.
  # Since we just copy build outputs, cmake doesn't get a chance to relink with the correct rpath.
  # sed the rpath in place manually
  # Also, libtensorrt_llm.so _sometimes_ wants libcudnn, so --add-needed to prevent it from being shrunk out
  installPhase =
    ''
      mkdir -p $out
      rsync -a --chmod=u+w --include "tensorrt_llm/kernels/" --include "tensorrt_llm/kernels/kvCacheIndex.h" --exclude "tensorrt_llm/kernels/*" $src/cpp $out/
      # rsync -a --chmod=u+w $src/cpp/tensorrt_llm/kernels $out/cpp/tensorrt_llm/
      pushd tensorrt_llm
      mkdir -p $out/cpp/build/tensorrt_llm/
      find . '(' '(' -type f -executable ')' -or -type l ')' -print0 | rsync -av --chmod=u+w --files-from=- --from0 ./ $out/cpp/build/tensorrt_llm/
      patchelf --add-needed 'libcudnn.so.8' --add-rpath ${cudaPackages.cudnn.lib}/lib $out/cpp/build/tensorrt_llm/libtensorrt_llm.so
      for f in $out/cpp/build/tensorrt_llm/plugins/*.so* $out/cpp/build/tensorrt_llm/executor_worker/executorWorker; do
        if [ ! -L "$f" ]; then
          new_path=$(patchelf --print-rpath "$f" |
            sed 's#/build/source/cpp/build/tensorrt_llm#$ORIGIN/..#g' |
            sed 's#/build/source/cpp/tensorrt_llm#$ORIGIN/../../../tensorrt_llm#g'
          )
          patchelf --set-rpath "$new_path" "$f"
        fi
      done
      new_path=$(patchelf --print-rpath $out/cpp/build/tensorrt_llm/libtensorrt_llm.so |
        sed 's#/build/source/cpp/tensorrt_llm#$ORIGIN/../../tensorrt_llm#')
      patchelf --set-rpath "$new_path" $out/cpp/build/tensorrt_llm/libtensorrt_llm.so
      popd
    ''
    + (lib.optionalString withPython ''
      mv ../../dist $dist
      pushd $dist
      python -m pip install ./*.whl --no-index --no-warn-script-location --prefix="$python" --no-cache
      popd
    '');
  postFixup = lib.optionalString withPython ''
    mv $out/nix-support $python/
  '';
  passthru.examples = runCommand "trt-examples" {} ''
    mkdir $out
    cp -r ${o.src}/examples $out/examples
  '';
})
