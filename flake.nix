# syntax = ghcr.io/reproducible-containers/buildkit-nix:v0.1.1@sha256:7d4c42a5c6baea2b21145589afa85e0862625e6779c89488987266b85e088021
{
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
    cognix.url = "github:datakami/cognix";
    cognix.inputs.nixpkgs.follows = "nixpkgs";
    cognix.inputs.dream2nix = {
      url = "github:yorickvp/dream2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, cognix }@inputs: cognix.lib.singleCognixFlake inputs "cog-triton-nix";
}
