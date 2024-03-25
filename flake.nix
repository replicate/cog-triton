# syntax = ghcr.io/reproducible-containers/buildkit-nix:v0.1.1@sha256:7d4c42a5c6baea2b21145589afa85e0862625e6779c89488987266b85e088021
{
  inputs = {
    cognix.url = "github:datakami/cognix";
  };

  outputs = { self, cognix }@inputs: (cognix.lib.cognixFlake inputs {}) // {
    packages.x86_64-linux = let
      lib = cognix.inputs.nixpkgs.lib;
      makeRunner = name: architectures: env: cognix.legacyPackages.x86_64-linux.callCognix {
        paths.projectRoot = self;
        inherit name;
        cog-triton = {
          runnerOnly = true;
          inherit architectures;
        };
        cognix.environment = env;
      } "${self}";
    in {
      default = self.packages.x86_64-linux.cog-triton-ci-mistral-86;
      "cog-triton-builder" = cognix.legacyPackages.x86_64-linux.callCognix {
        paths.projectRoot = self;
        name = "cog-triton-builder";
      } "${self}";
      "cog-triton-runner-80" = makeRunner "cog-triton-runner-80" ["80-real"] {};
      "cog-triton-runner-86" = makeRunner "cog-triton-runner-86" ["86-real"] {};
      "cog-triton-runner-90" = makeRunner "cog-triton-runner-90" ["90-real"] {};
      "cog-triton-mistral-7b" = makeRunner "cog-triton-mistral-7b" ["86-real"] {
        COG_WEIGHTS = "https://replicate.delivery/pbxt/9yf58OhSA5VZCCiflRRmgfVSnxujfuLfXSk6P24Yyu54Db7TC/engine.tar";
        SYSTEM_PROMPT = "You are a very helpful, respectful and honest assistant.";
        PROMPT_TEMPLATE = "<s>[INST] {system_prompt} {prompt} [/INST]";
      };
    };
  };
}
