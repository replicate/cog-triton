{
  nixConfig = {
    extra-trusted-public-keys = "replicate-1:rbU0MI8kgUmqLINtKfXoDkrl9NxXQMw6//+LHHDYflk=";
    extra-substituters = "https://storage.googleapis.com/replicate-nix-cache-dev/";
  };
  inputs = {
    cognix.url = "github:datakami/cognix/24.07";
  };

  outputs = { self, cognix }@inputs: (cognix.lib.cognixFlake inputs {}) // {
    packages.x86_64-linux = let
      sourceRev = self.sourceInfo.rev or self.sourceInfo.dirtyRev or null;
      callCognix = args: cognix.legacyPackages.x86_64-linux.callCognix ({ lib, ... }: {
        paths.projectRoot = self;
        imports = [ args ];
        dockerTools.streamLayeredImage.config = lib.mkIf (sourceRev != null) {
          Labels."org.opencontainers.image.revision" = sourceRev;
        };
      }) "${self}";
      makeRunner = name: architectures: env: callCognix ( {config, lib, ... }: {
        inherit name;
        # only grab deps of nvidia-pytriton, transformers
        cognix.python_root_packages = [ "nvidia-pytriton" "transformers" "tokenizers" ];

        cognix.environment.TRITONSERVER_BACKEND_DIR = "${config.deps.backend_dir}/backends";

        cog-triton.architectures = architectures;
        # don't need this file in a runner
        python-env.pip.drvs.tensorrt-cu12-libs.mkDerivation.postInstall = lib.mkAfter ''
          rm $out/lib/python*/site-packages/tensorrt_libs/libnvinfer_builder_resource*
        '';
      });
      makeBuilder = name: callCognix ( { config, lib, pkgs, ... }: {
        inherit name;
        # only grab deps of tensorrt-llm, omegaconf, hf-transfer
        cognix.python_root_packages = [ "omegaconf" "hf-transfer" "transformers" "torch" ];


        # override cog.yaml:
        cog.concurrency.max = lib.mkForce 1;
        cognix.rootPath = lib.mkForce "${./cog-trt-llm}";
        # this just needs the examples/ dir
        cognix.environment.TRTLLM_DIR = config.deps.tensorrt-llm.examples;
        # HACK: cog needs pydantic v1, but trt-llm needs pydantic v2
        cognix.environment.TRTLLM_PYTHON = (config.python-env.public.extendModules {
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
        }).config.public.pyEnv;
      });
    in {
      cog-triton-builder = makeBuilder "cog-triton-builder";
      cog-triton-runner-80 = makeRunner "cog-triton-runner-80" ["80-real"] {};
      cog-triton-runner-86 = makeRunner "cog-triton-runner-86" ["86-real"] {};
      cog-triton-runner-90 = makeRunner "cog-triton-runner-90" ["90-real"] {};
      # mistral example, update for new engine
      # default = self.packages.x86_64-linux.cog-triton-mistral-7b;
      # cog-triton-mistral-7b = makeRunner "cog-triton-mistral-7b" ["86-real"] {
      #   COG_WEIGHTS = "https://replicate.delivery/pbxt/9yf58OhSA5VZCCiflRRmgfVSnxujfuLfXSk6P24Yyu54Db7TC/engine.tar";
      #   SYSTEM_PROMPT = "You are a very helpful, respectful and honest assistant.";
      #   PROMPT_TEMPLATE = "<s>[INST] {system_prompt} {prompt} [/INST]";
      # };
    };
  };
}
