# syntax = ghcr.io/reproducible-containers/buildkit-nix:v0.1.1@sha256:7d4c42a5c6baea2b21145589afa85e0862625e6779c89488987266b85e088021
{
  nixConfig = {
    extra-trusted-public-keys = "replicate:+uoDlJCmq7Z6lxxacj4tSgsNa24qHqTZTiyRX2CDhsA=";
    extra-substituters = "https://cache.yori.cc/replicate";
  };
  inputs = {
    cognix.url = "github:datakami/cognix";
  };

  outputs = { self, cognix }@inputs: (cognix.lib.cognixFlake inputs {}) // {
    packages.x86_64-linux = let
      lib = cognix.inputs.nixpkgs.lib;
      callCognix = args: cognix.legacyPackages.x86_64-linux.callCognix {
        paths.projectRoot = self;
        imports = [ args ];
      } "${self}";
      makeRunner = name: architectures: env: callCognix {
        inherit name;
        cog-triton = {
          runnerOnly = true;
          inherit architectures;
        };
        cognix.environment = env;
      };
    in {
      default = self.packages.x86_64-linux.cog-triton-mistral-7b;
      cog-triton-builder = callCognix ({config, pkgs, ... }: {
        name = "cog-triton-builder";
        # only grab deps of cog, tensorrt-llm
        python-env.pip.rootDependencies = lib.mkForce (lib.genAttrs [
          "cog" "tensorrt-llm" "omegaconf" "hf-transfer"
        ] (x: true));
        # override cog.yaml:
        cog = {
          concurrency = lib.mkForce 1;
        };
        cognix.postCopyCommands = ''
          cp ${config.deps.cog-trt-llm}/{*.py,cog-trt-llm-config.yaml} $out/src/
        '';
        # this just needs the examples/ dir
        cognix.environment.TRTLLM_DIR = pkgs.runCommand "trt-examples" {} ''
          mkdir $out
          cp -r ${config.deps.tensorrt_llm.src}/examples $out/examples
        '';
      });
      cog-triton-runner-80 = makeRunner "cog-triton-runner-80" ["80-real"] {};
      cog-triton-runner-86 = makeRunner "cog-triton-runner-86" ["86-real"] {};
      cog-triton-runner-90 = makeRunner "cog-triton-runner-90" ["90-real"] {};
      cog-triton-mistral-7b = makeRunner "cog-triton-mistral-7b" ["86-real"] {
        COG_WEIGHTS = "https://replicate.delivery/pbxt/9yf58OhSA5VZCCiflRRmgfVSnxujfuLfXSk6P24Yyu54Db7TC/engine.tar";
        SYSTEM_PROMPT = "You are a very helpful, respectful and honest assistant.";
        PROMPT_TEMPLATE = "<s>[INST] {system_prompt} {prompt} [/INST]";
      };
    };
  };
}
