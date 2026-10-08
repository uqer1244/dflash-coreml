# Distribution names and scope

- GitHub source repository: **dflash-coreml**.
- Hugging Face model repository: **Qwen3.8-27B-DFlash2-CoreML** under the publisher's HF account.
- Runtime package: `dflash-coreml`; imports: `dflash_coreml`.

The source repo contains the reusable drafter backend and `examples/mlx_generate.py`: download the HF Core ML bundle and generate using a separate MLX target. The target model is not bundled with the drafter. The HF directory contains six Core ML packages, compact selector weights, original config, SHA manifest, Apache license and model card.

Source is public at https://github.com/uqer1244/dflash-coreml. The public HF repository https://huggingface.co/uqer1244/Qwen3.8-27B-DFlash2-CoreML has been created; compact weight upload is in progress. Until its model manifest and weight files are committed, use the prepared local bundle. No oMLX PR has been submitted; that work is explicitly deferred.

Release-source ZIP validation is complete: extracted outside the project, installed non-editably into a fresh venv using public pinned runtime packages, built the Swift bridge, passed11 tests and model hashes, and generated four fixtures up to128 tokens. Token IDs and EOS/length match the prior research references, repeated reset matches, and no-thinking answers Seoul. See `STANDALONE_VALIDATION.json`. Live HF download remains untested pending upload.

Context compaction reduced the complete prepared bundle from5.041 GB to2.885 GB (2.687 GiB). Four-fixture actual state/output parity passed80 cycles /264 feature commits. Disk reduction is not a runtime memory measurement. See `CONTEXT_COMPACTION.md`.
