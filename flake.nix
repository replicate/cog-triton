{
  nixConfig = {
    extra-trusted-public-keys = "replicate-1:rbU0MI8kgUmqLINtKfXoDkrl9NxXQMw6//+LHHDYflk=";
    extra-substituters = "https://storage.googleapis.com/replicate-nix-cache-dev/";
  };
  inputs = {
    cognix.url = "github:datakami/cognix/yorickvp/uv";
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
        cog-triton = {
          inherit architectures;
          # only grab deps of nvidia-pytriton, transformers
          rootDependencies = [ "nvidia-pytriton" "transformers" "tokenizers" ];
        };
        cognix.environment.TRITONSERVER_BACKEND_DIR = "${config.deps.backend_dir}/backends";
        # don't need this file in a runner
        python-env.pip.drvs.tensorrt-libs.mkDerivation.postInstall = lib.mkAfter ''
          rm $out/lib/python*/site-packages/tensorrt_libs/libnvinfer_builder_resource*
        '';
      });
      makeBuilder = name: callCognix ( { config, lib, ... }: {
        inherit name;
        cog-triton = {
          # only grab deps of tensorrt-llm, omegaconf, hf-transfer
          rootDependencies = [ "tensorrt-llm" "omegaconf" "hf-transfer" ];
        };
        # override cog.yaml:
        cog.concurrency = lib.mkForce 1;
        # copy cog-trt-llm source into /src
        cognix.postCopyCommands = ''
          cp ${config.deps.cog-trt-llm}/{*.py,cog-trt-llm-config.yaml} $out/src/
        '';
        # this just needs the examples/ dir
        cognix.environment.TRTLLM_DIR = config.deps.tensorrt-llm.examples;
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
