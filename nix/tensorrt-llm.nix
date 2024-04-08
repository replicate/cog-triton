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
}:
stdenv.mkDerivation (o: {
  pname = "tensorrt_llm";
  version = "0.8.0";
  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "TensorRT-LLM";
    rev = "v${o.version}";
    fetchSubmodules = true;
    fetchLFS = true; # libtensorrt_llm_batch_manager_static.a
    hash = "sha256-10wSFhtMGqqCigG5kOBuegptQJymvpO7xCFtgmOOn+k=";
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
  ];
  buildInputs =
    [
      cudaPackages.cuda_nvcc
      cudaPackages.cudnn
      cudaPackages.nccl
      openmpi
    ]
    ++ (lib.optionals (!withPython) [
      # torch hates the split cuda, so only do it without torch
      cudaPackages.cuda_cudart
      cudaPackages.cuda_nvcc.dev
      cudaPackages.cuda_cccl
      cudaPackages.libcublas.lib
    ])
    ++ (lib.optionals withPython [
      cudaPackages.cudatoolkit
      python3.pkgs.pybind11
      python3.pkgs.setuptools
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
      tensorrt-bindings # missed transitive dep
      tensorrt-libs
      torch # <=2.2.0a
      nvidia-ammo # ~=0.7.0; platform_machine=="x86_64"
      transformers # ==4.36.1
      wheel
      optimum
      evaluate
      janus
    ]
  );

  cmakeFlags = [
    "-DBUILD_PYT=${if withPython then "ON" else "OFF"}"
    "-DBUILD_PYBIND=${if withPython then "ON" else "OFF"}" # needs BUILD_PYT
    "-DBUILD_TESTS=OFF" # needs nvonnxparser.h
    # believe it or not, this is the actual binary distribution channel for tensorrt:
    "-DTRT_LIB_DIR=${pythonDrvs.tensorrt-libs.public}/${python3.sitePackages}/tensorrt_libs"
    "-DTRT_INCLUDE_DIR=${tensorrt-src}/include"
    "-DCMAKE_CUDA_ARCHITECTURES=${builtins.concatStringsSep ";" architectures}"
    # "-DCUDAToolkit_INCLUDE_DIR=${cudaPackages.cuda_cudart}/include"
    "-DCUDAToolkit_INCLUDE_DIR=${cudaPackages.cudatoolkit}/include"
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
  installPhase =
    ''
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
