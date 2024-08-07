{ pkgs, config, extendModules, ... }:
let
  deps = config.deps;
  python3 = config.python-env.deps.python;
  inherit (config.cognix) cudaPackages;
  site = python3.sitePackages;
  pythonDrvs = config.python-env.pip.drvs;
  inherit (pkgs) lib;
  cfg = config.cog-triton; # defined in interface.nix
  trtllm-env = config.python-env.public.extendModules {
    modules = [{
      _file = ./.;
      pip.rootDependencies = lib.mkOverride 49 { tensorrt-llm = true; hf-transfer = true; };
      pip.drvs.pydantic = let mkMoreForce = lib.mkOverride 49; in {
        version = mkMoreForce "2.8.2";
        mkDerivation.src = mkMoreForce (pkgs.fetchurl {
          sha256 = "73ee9fddd406dc318b885c7a2eab8a6472b68b8fb5ba8150949fc3db939f23c8";
          url = "https://files.pythonhosted.org/packages/1f/fa/b7f815b8c9ad021c07f88875b601222ef5e70619391ade4a49234d12d278/pydantic-2.8.2-py3-none-any.whl";
        });
      };
    }];
  };
  trtllm-pythonDrvs = trtllm-env.config.pip.drvs;
  toCudaCapability = cmakeArch: {
    "70-real" = "7.0";
    "80-real" = "8.0";
    "86-real" = "8.6";
    "89-real" = "8.9";
    "90-real" = "9.0";
  }.${cmakeArch};
in
{
  imports = [ ./interface.nix ];
  cog.build = {
    python_version = "3.10";
    cog_version = "0.10.0-alpha18";
    cuda = "12.1"; # todo: 12.2
    gpu = true;
    # inspiration: echo tensorrt_llm==0.10.0 | uv pip compile - --extra-index-url https://pypi.nvidia.com -p 3.10 --prerelease=allow --annotation-style=line
    python_packages = [
      "--extra-index-url"
      "https://pypi.nvidia.com"
      "tensorrt_llm==0.12.0.dev2024073000"
      "tensorrt-cu12==10.2.0.post1"
      "torch==2.3.1"
      "nvidia-pytriton==0.5.8" # corresponds to 2.46.0
      "omegaconf"
      "hf-transfer"
      "tokenizers>=0.19.0"
    ];
    # don't ask why it needs ssh
    system_packages = [ "pget" "openssh" "openmpi" ];
  };
  # patch in cuda packages from nixpkgs
  cognix.merge-native = {
    cudnn = "force";
    cublas = true;
  };
  python-env.pip = {
    constraintsList = [
      "datasets>2.15.0" # picks older fsspec but newer datasets
      "mpi4py<4" # recent release with breaking changes
    ];
    # HACK: cog requires pydantic <2, but we do need the extra deps pydantic2 brings in
    overridesList = [
      "pydantic>=2.0"
    ];
    drvs.pydantic = {
      version = lib.mkForce "1.10.17";
      mkDerivation.src = pkgs.fetchurl {
        sha256 ="371dcf1831f87c9e217e2b6a0c66842879a14873114ebb9d0861ab22e3b5bb1e";
        url = "https://files.pythonhosted.org/packages/ef/a6/080cace699e89a94bd4bf34e8c12821d1f05fe4d56a0742f797b231d9a40/pydantic-1.10.17-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
      };
    };
  };
  cognix.includeNix = true;
  cognix.nix.extraOptions = ''
    extra-trusted-public-keys = replicate-1:rbU0MI8kgUmqLINtKfXoDkrl9NxXQMw6//+LHHDYflk=
    extra-substituters = https://storage.googleapis.com/replicate-nix-cache-dev/
  '';
  python-env.pip.drvs = {

    torch.public = lib.mkIf cfg.torchSourceBuild
      (lib.mkForce config.deps.minimal-torch);
    tensorrt-llm.public = lib.mkIf cfg.trtllmSourceBuild
      (lib.mkForce config.deps.tensorrt-llm.override {
        withPython = true;
      });

    nvidia-modelopt.mkDerivation.propagatedBuildInputs = [
      pythonDrvs.setuptools.public
    ];
    # tensorrt likes doing a pip invocation from it's setup.py
    # circumvent by manually depending on tensorrt_libs, tensorrt_bindings
    # and setting this env variable
    tensorrt-cu12.env.NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP = true;
    tensorrt-cu12.mkDerivation.buildInputs = [ python3.pkgs.pip ];
    tensorrt-cu12-bindings.mkDerivation.propagatedBuildInputs = [
      pythonDrvs.tensorrt-cu12-libs.public
    ];
    # fixes tensorrt-llm build
    tensorrt-cu12-libs.mkDerivation.postFixup = ''
      pushd $out/${site}/tensorrt_libs
      ln -s libnvinfer.so.10 libnvinfer.so
      ln -s libnvonnxparser.so.10 libnvonnxparser.so
      popd
    '';
    tensorrt-cu12-libs.env.appendRunpaths = [ "/usr/lib64" "$ORIGIN" ];
    tensorrt-llm = {
      mkDerivation.buildInputs = [ cudaPackages.nccl ];
      mkDerivation.propagatedBuildInputs = with pythonDrvs; [
        tensorrt-cu12-libs.public # libnvinfer, onnxparse
      ];
      env.appendRunpaths = [ "/usr/lib64" "$ORIGIN" ];
      env.autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" "libnvidia-ml.so.1" ];
      mkDerivation.postInstall = ''
        pushd $out/${site}/tensorrt_llm/bin
        patchelf --replace-needed libnvinfer_plugin_tensorrt_llm.so{.10,} executorWorker
        popd
      '';
    };
    # has some binaries that want cudart
    tritonclient.mkDerivation.postInstall = "rm -r $out/bin";
    # replace libtritonserver-90a4cf82.so with libtritonserver.so
    # so backends don't have to know about the hash
    nvidia-pytriton.mkDerivation.postInstall = ''
      pushd $out/${site}/nvidia_pytriton.libs
      ln -s libtritonserver-*.so libtritonserver.so
      patchelf --replace-needed libtritonserver-*.so libtritonserver.so $out/${python3.sitePackages}/pytriton/tritonserver/bin/tritonserver
      popd
      pushd $out/${site}/pytriton/tritonserver/python_backend_stubs
      # remove every python stub but the current python version
      for d in *; do
        if [ "$d" != "${python3.pythonVersion}" ]; then
          rm -r $d
        fi
      done
      popd
    '';
    mpi4py.mkDerivation.nativeBuildInputs = [ pkgs.removeReferencesTo ];
    mpi4py.mkDerivation.postInstall = ''
      pushd $out/${site}/mpi4py
      remove-references-to -t ${pkgs.openmpi.dev} mpi.cfg MPI.*.so
      popd
    '';
  };
  deps.backend_dir = pkgs.runCommand "triton_backends" {} ''
    mkdir $out
    tritonserver=${pythonDrvs.nvidia-pytriton.public}/${site}/pytriton/tritonserver
    cp -r $tritonserver/backends $out/
    chmod -R +w $out/backends
    cp $tritonserver/python_backend_stubs/${python3.pythonVersion}/triton_python_backend_stub $out/backends/python/
    cp -r ${deps.trtllm-backend}/backends/tensorrtllm $out/backends/
    for f in $out/backends/tensorrtllm/*; do
      chmod +w $f
      patchelf --add-rpath ${pythonDrvs.nvidia-pytriton.public}/${site}/nvidia_pytriton.libs $f
    done
  '';
  deps.tensorrt-src = pkgs.fetchFromGitHub {
    owner = "NVIDIA";
    repo = "TensorRT";
    rev = "v10.2.0";
    hash = "sha256-Euo9VD4VTpx8XJV97IMETTAx/YkPGXiNdA39Wjp3UMU=";
  };
  # make a python3 environment with all the pkgs from lock.json *and* nixpkgs.python
  # mainly used to build torch, which additionally requires astunparse
  deps.python3-with-nixpkgs = python3.override {
    packageOverrides = pyself: pysuper: (lib.mapAttrs (_: v: v.public.out) trtllm-pythonDrvs) // {
      # todo: replace with lockfile?
      pybind11-stubgen = pyself.buildPythonPackage rec {
        pname = "pybind11-stubgen";
        version = "2.5";
        src = pyself.fetchPypi {
          inherit pname version;
          hash = "sha256-lqf+vKski/mKvUu3LMX3KbqHsjRCR0VMF1nmPN6f7zQ=";
        };
      };
      # prevent infinite loop, don't override torch itself
      inherit (pysuper) torch;
    };
  };
  deps.tensorrt-llm = pkgs.callPackage ./nix/tensorrt-llm.nix {
    inherit python3 cudaPackages;
    pythonDrvs = config.deps.trtllm-env.config.pip.drvs;
    withPython = false;
    inherit (cfg) architectures;
    inherit (deps.python3-with-nixpkgs.pkgs) pybind11-stubgen;
    inherit (deps) tensorrt-src;
  };
  deps.trtllm-env = trtllm-env;
  deps.trtllm-backend = pkgs.callPackage ./nix/trtllm-backend.nix {
    inherit python3 cudaPackages pythonDrvs;
    inherit (deps) tensorrt-llm tensorrt-src;
  };
  deps.minimal-torch = pkgs.callPackage ./nix/torch.nix {
    python3 = deps.python3-with-nixpkgs;
    # todo: match/modify config.cognix.cudaPackages
    cudaPackages = (pkgs.extend (self: super: {
      config = super.config // {
        cudaCapabilities = map toCudaCapability cfg.architectures;
      };
    })).cudaPackages_12_1;
  };
}
