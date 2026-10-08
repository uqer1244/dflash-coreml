# dflash-coreml

GitHub repository: **dflash-coreml**. Hugging Face model: **Qwen3.8-27B-DFlash2-CoreML**. [Distribution status](docs/DISTRIBUTION.md).

**Source preview:** Core ML weights have not been published to Hugging Face yet. The HF example becomes runnable when the model repository is available; local prepared bundles can be used now. oMLX integration/PR is on hold.

Run an MLX target with a **Core ML / ANE DFlash2 drafter**, from a prompt to generated text. A Python standalone runner owns the MLX target/verification and calls a Swift Core ML bridge for the drafter backbone. No installed oMLX app is required.

One Hugging Face Core ML model bundle is intended for two integration routes:

| Route | Current status |
|---|---|
| **Standalone MLX** | Implemented: local/HF model resolution, target + drafter loading, greedy generation, EOS, reset and verification/rollback |
| **oMLX** | Planned: opt-in backend adapter is not yet connected to oMLX/DFlashEngine |

The model bundle contains six `.mlpackage` directories, selector-only weights and a SHA manifest. Users do not need the entire original BF16 drafter checkpoint. The target model remains a separate download. The converted Core ML model repository has not been published yet.

## 1. Standalone MLX

[Python execution example](examples/mlx_generate.py) downloads your HF drafter bundle and combines it with the MLX target:

```sh
python examples/mlx_generate.py --hf-owner YOUR_HF_ACCOUNT \
  --prompt "What is the capital of South Korea?" --no-thinking
```


Apple Silicon/macOS15+, Python3.11 and Xcode Command Line Tools. Tested on M3 Pro18GB/macOS27.0.1. Install the pinned public runtime in your own environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[generate]'
./native/build.sh

dflash-coreml generate \
  --model /path/to/compatible-Bonsai2-target \
  --draft-model /path/to/CoreML-model-bundle \
  --prompt 'What is the capital of South Korea? Answer in one sentence.' \
  --max-tokens 128 --no-thinking
```

`--model` and `--draft-model` also accept Hugging Face `owner/repo` IDs. Once the Core ML weights are published, replace the local bundle path with your published model ID and pin its revision:

```sh
dflash-coreml generate \
  --model nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX \
  --draft-model YOUR_HF_ACCOUNT/Qwen3.8-27B-DFlash2-CoreML \
  --coreml-revision YOUR_PUBLISHED_COMMIT \
  --prompt 'Explain a CPU, GPU and NPU briefly.' --max-tokens 128 --no-thinking
```

Repository/commit placeholders are intentional; there is no live Core ML model URL yet. `--target-revision` pins the target snapshot; the bundle rejects target/selector hashes that differ from the verified pairing. `--bridge` overrides the default `build/ane-bridge`. Downloads are weights/config/tokenizer files, not executable remote Python.

Options include `--prompt-file`, `--json-output` (token IDs, timings and finish reason), `--proposal-count 1..5` (default2), and `--history stock|deferred` (defaultdeferred). The default chat template retains thinking; `--no-thinking` requests direct answers. Text is printed after generation; this preview does not implement token streaming or sampling.

For a normal MLX target-only control:

```sh
dflash-coreml generate --model /path/to/compatible-Bonsai2-target \
  --vanilla --prompt 'What is the capital of South Korea?' \
  --max-tokens 128 --no-thinking
```

Vanilla uses ordinary singleton target forwards without speculative state capture. Comparing it to the drafter path requires checking output as well as timing; general singleton/batched numerical equivalence is not established.

## 2. oMLX integration

The same model bundle can be consumed by a future oMLX backend. **This is not currently an installable oMLX plugin.** The remaining work is to wire reset/draft/context-commit/close into the DFlash runtime, expose opt-in settings and validate cancellation, unload, cache state and fallback. See [integration contract](docs/INTEGRATION.md). No app-bundle patch command or completed upstream PR is claimed.

## Model publisher preparation

From the existing research workspace, generate a manifest with the matching target and prepare the HF upload directory:

```sh
python tools/local_manifest.py --workspace /path/to/research-workspace \
  --target /path/to/compatible-Bonsai2-target --output local-manifest.json
python tools/export_model.py --manifest local-manifest.json \
  --draft /path/to/original-DFlash2-checkpoint --output model-release
```

`model-release/` contains a model card, Apache license, model packages, compact selector weights and portable `model-manifest.json`. Export verifies hashes and never uploads. On APFS it attempts copy-on-write cloning for local package preparation. The exporter currently targets the exact recorded Qwen3.8-27B-DFlash2 source revision; conversion is still a separate research workflow. [Conversion status](docs/CONVERSION.md).

## Backbone API

```python
from dflash_coreml import CoreMLBackbone

with CoreMLBackbone(manifest_path, bridge_path) as backbone:
    backbone.reset(position=0)
    hidden, timings = backbone.forward(embedding, committed_features)
```

Core ML runs fusion, all five drafter layers, final norm and KV updates as2+2+1 shards, block6/capacity31. Shared LM head, Top16, selector and target verification run outside Core ML. Earlier committed features use context-only graphs; the final one computes the proposal block. Each instance is single-request/synchronous.

## Validation and limits

Normal selector replay **91.07% is below the95% fidelity gate**. The runner is experimental/model-specific; arbitrary DFlash2 targets, general losslessness, server batching and GPU superiority are not established. See [results](docs/RESULTS.md), [release status](docs/RELEASE.md) and the hardware/standalone validation records in `docs/`.

Run `python -m unittest discover -s tests -v`. Existing model packages, checkpoint or runtime changes require new hardware correctness validation. Historical budget gains and drafter-only power measurements are not standalone GPU speed/total-energy proof.

Project code is MIT except the adapted Bonsai2 target loader (`target.py`, Apache-2.0). Model artifacts remain Apache-2.0 derived from the original DFlash2 checkpoint. See [NOTICE](NOTICE) and `LICENSES/Apache-2.0.txt`.
