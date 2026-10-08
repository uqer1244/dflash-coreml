# 0.1.0a1 release preparation

Implemented standalone prompt-to-text generation, pinned public MLX runtime, text-only Bonsai2 loader, greedy selector, Core ML backbone, target verification/accepted-prefix commit, EOS and per-prompt reset. The runtime does not import an oMLX app bundle.

Prepared HF model directory: six Core ML packages, original selector-only BF16 tensors, config, portable manifest, source revision/hash, model card and Apache license. Same model bundle is intended for standalone MLX and a future oMLX backend. No model/code repository has been published by these tools.

Remaining release scope: publish model/source repositories and replace README placeholders; port a full independent conversion CLI; integrate oMLX backend and server lifecycle; extend long-context/multi-device correctness and fresh throughput/energy controls. The selector fidelity gate remains failed. Standalone source import/CLI and local model-bundle execution do not prove live HF download of the unpublished bundle.

`tools/bundle.py` creates the source ZIP/checksums. It excludes environments, model-release weights, runtime builds, local manifests and generation logs. Source and weights belong in separate GitHub/Hugging Face repositories.
