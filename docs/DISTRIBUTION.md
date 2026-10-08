# Distribution names and scope

- GitHub source repository: **dflash-coreml**.
- Hugging Face model repository: **Qwen3.8-27B-DFlash2-CoreML** under the publisher's HF account.
- Runtime package: `dflash-coreml`; imports: `dflash_coreml`.

The source repo contains the reusable drafter backend and `examples/mlx_generate.py`: download the HF Core ML bundle and generate using a separate MLX target. The target model is not bundled with the drafter. The HF directory contains six Core ML packages, compact selector weights, original config, SHA manifest, Apache license and model card.

Source is public at https://github.com/uqer1244/dflash-coreml. The converted weights are public at https://huggingface.co/uqer1244/Qwen3.8-27B-DFlash2-CoreML, revision `c6d1fc8900c481d36392f5c68ba96927b263d541`. All24 uploaded files match the local hashes. No oMLX PR has been submitted; that work remains explicitly deferred. No tagged GitHub release has been created; tests downloaded public commit archives.

Local source-ZIP validation covers four generation fixtures up to128 tokens, repeated reset and no-thinking. Separate live-HF validation downloaded the GitHub source into a new folder and installed it in a fresh venv, downloaded the Core ML bundle into a fresh HF cache without authentication, and reused existing Bonsai2 target weights as requested. Vanilla MLX, Core ML CLI and the unmodified GitHub example all passed. CLI token IDs and EOS match for the9-token Seoul answer. Execution was offline after the downloads, without importing research or oMLX app code. See `HF_STANDALONE_VALIDATION.json` and `HF_UPLOAD_VALIDATION.json`.

HF snapshot manifests must retain their snapshot-relative base path. Before compilation, the runner prepares ordinary package files from HF cache symlinks using APFS clones when available. These fixes are covered by13 unit tests and the actual HF CLI/example run.

The pinned standalone MLX wheels require macOS26+ on Apple Silicon; the Core ML bridge alone targets macOS15+. Core ML cold CLI/example wall times were72.88/74.97 seconds for the9-token smoke. Cold latency remains work; these checks do not establish steady-state performance. See `CLI_COLD_WARM_VALIDATION.json` for the earlier same-instance repeated request.

Context compaction reduced the prepared bundle from5.041 GB to2.885 GB (2.687 GiB). Four-fixture actual state/output parity passed80 cycles /264 feature commits. Disk reduction is not a runtime memory measurement. See `CONTEXT_COMPACTION.md`.
