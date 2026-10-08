# Distribution names and scope

- GitHub source repository: **dflash-coreml**.
- Hugging Face model repository: **Qwen3.8-27B-DFlash2-CoreML** under the publisher's HF account.
- Runtime package: `dflash-coreml`; imports: `dflash_coreml`.

The source repo contains the reusable drafter backend and `examples/mlx_generate.py`: download the HF Core ML bundle and generate using a separate MLX target. The target model is not bundled with the drafter. The HF directory contains six Core ML packages, compact selector weights, original config, SHA manifest, Apache license and model card.

`YOUR_HF_ACCOUNT` remains a placeholder until the HF owner is supplied. Neither model/source publication nor an oMLX PR has been performed. oMLX PR work is explicitly deferred.

Local validation currently includes backbone hardware parity and a32-token standalone result matching the research token IDs. The attempted full128-token standalone suite was interrupted at the user's pause request and has no completed PASS report. Do not present it as completed validation. Live HF download of this unpublished bundle also remains untested.
