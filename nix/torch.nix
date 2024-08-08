{ python3, magma-cuda-static, cudaPackages }:
(python3.pkgs.torchWithCuda.override {
  torchWithCuda = null; # ?!, not used
  cudaSupport = true;
  inherit cudaPackages;
  magma-cuda-static = magma-cuda-static.override { inherit cudaPackages; };
  future = null;
  tensorboard = null;
  hypothesis = null;
  cffi = null;
  openai-triton = null;
}).overridePythonAttrs
  (o: {
    nativeBuildInputs = o.nativeBuildInputs ++ [ python3.pkgs.setuptools ];
    dependencies = o.dependencies ++ [ python3.pkgs.requests ];
    USE_CUDNN = 0;
    USE_KINETO = 0;
    USE_QNNPACK = 0;
    USE_PYTORCH_QNNPACK = 0;
    USE_XNNPACK = 0;
    INTERN_DISABLE_ONNX = 1;
    ONNX_ML = 0;
    USE_ITT = 0;
    USE_FLASH_ATTENTION = 0;
    USE_MEM_EFF_ATTENTION = 0;
    USE_FBGEMM = 0;
    USE_MKLDNN = 0;
  })
