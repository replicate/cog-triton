FROM r8.im/replicate-internal/triton-base-sm80@sha256:c7cb9d7813f438a178a25cc3687984d2d9457fbcfe3a8dca5f4c00fc7060cc6c
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install tokenizers==0.19.0
