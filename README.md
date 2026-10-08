# dflash-coreml

Core ML / ANE DFlash2 drafter for Qwen3.8-27B targets running on MLX.

## Requirements

- Apple Silicon Mac with macOS 26 or later
- Python 3.11
- Xcode Command Line Tools (`xcode-select --install`)
- A compatible MLX target based on Qwen3.8-27B
- Core ML drafter: [Qwen3.8-27B-DFlash2-CoreML](https://huggingface.co/uqer1244/Qwen3.8-27B-DFlash2-CoreML)

The target and drafter are separate models. The drafter download is approximately 2.9 GB. No installed oMLX app is required. Tested on an M3 Pro with 18 GB of unified memory.

## Usage

### Install

```sh
git clone https://github.com/uqer1244/dflash-coreml.git
cd dflash-coreml

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install '.[generate]'
sh native/build.sh
```

### Generate

Use an existing compatible Qwen3.8-27B target directory. The Core ML drafter is downloaded from Hugging Face automatically.

The current runner is validated with a Bonsai2 text-only repack and checks its exact config/weight hashes. Other Qwen3.8-27B variants, including the official Prism bundle, need additional loader/binding validation.

```sh
dflash-coreml generate \
  --model /path/to/Qwen3.8-27B-target \
  --coreml-model uqer1244/Qwen3.8-27B-DFlash2-CoreML \
  --coreml-revision c6d1fc8900c481d36392f5c68ba96927b263d541 \
  --prompt 'What is the capital of South Korea?' \
  --max-tokens 128 --no-thinking
```

Python example:

```sh
python examples/mlx_generate.py \
  --hf-owner uqer1244 \
  --revision c6d1fc8900c481d36392f5c68ba96927b263d541 \
  --model /path/to/Qwen3.8-27B-target \
  --prompt 'What is the capital of South Korea?' \
  --max-tokens 128 --no-thinking
```

Remove `--no-thinking` to use the model's default thinking mode. Add `--vanilla` to run the MLX target alone. If HF downloads stall, set `HF_HUB_DISABLE_XET=1`.

This is an experimental, greedy, text-only runner. The first Core ML request can be slow; tested fresh-process runs took approximately 73–75 seconds including loading. General output equivalence and a speed advantage over GPU drafting are not established.

## Todo

- Reduce model loading and first-request latency
- Improve selector fidelity (current replay: 91.07%; goal: at least 95%)
- Support and validate additional Qwen3.8-27B target variants
- Extend correctness and performance tests across prompts, context lengths and devices
- Measure end-to-end memory use and energy against GPU drafting
- Add an oMLX adapter and validate cancellation, unload and fallback
- Provide a standalone conversion CLI for generating Core ML packages
