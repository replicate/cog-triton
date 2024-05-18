{ lib, ... }:
{
  options.cog-triton = with lib; {
    architectures = mkOption {
      # https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/CMakeLists.txt#L152C34-L152C73
      type = types.listOf (types.enum [ "70-real" "80-real" "86-real" "89-real" "90-real" ]);
      default = [ "70-real" "80-real" "86-real" "89-real" "90-real" ];
      description = "Architectures to build TRT-LLM for";
      # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
      # 80: A100, 86: A5000, A40, A800, 89: L40, 90: H100
    };
  };
}
