# 0.1.0a1 release preparation

Implemented standalone prompt-to-text generation, pinned public MLX runtime, text-only Bonsai2 loader, greedy selector, Core ML backbone, target verification/accepted-prefix commit, EOS and per-prompt reset. The runtime does not import an oMLX app bundle.

Prepared HF model directory: six Core ML packages, original selector-only BF16 tensors, config, portable manifest, source revision/hash, model card and Apache license. Same model bundle is intended for standalone MLX and a future oMLX backend. Source repository: https://github.com/uqer1244/dflash-coreml (public). The HF repository has been created; compact weights are uploading.

Source ZIP standalone validation passed four prompts up to128 tokens, repeated reset, no-thinking and11 tests in a new environment. Compact context state/output parity passed264 commits.

A fresh Core ML instance showed48–56 second first generation in CLI checks; same-instance repeated request1.91 seconds. Cause remains unisolated. See `CLI_COLD_WARM_VALIDATION.json`.

Remaining release scope: isolate/reduce first-request delay; complete the HF weight upload and verify live download and replace model README placeholders; port a full independent conversion CLI; integrate oMLX backend and server lifecycle; extend long-context/multi-device correctness and fresh throughput/energy controls. The selector fidelity gate remains failed. Standalone source import/CLI and local model-bundle execution do not prove live HF download of the unpublished bundle.

`tools/bundle.py` creates the source ZIP/checksums. It excludes environments, model-release weights, runtime builds, local manifests and generation logs. Source and weights belong in separate GitHub/Hugging Face repositories.
