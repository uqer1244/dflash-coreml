# Changelog

## 0.1.0a1 — experimental preview

- Extracted portable stateful Core ML backbone adapter from the experimental Native protocol.
- Added explicit model manifest, SHA validation and lifecycle timeout/cleanup.
- Included Swift bridge/build source, examples and selected sanitized research evidence.
- Verified adapter hidden parity on four fixtures and capacity31 rollover.
- Removed unreferenced context weights: complete bundle5.041→2.885 GB; hardware state/output byte parity passed.
- Tested extracted source ZIP in a fresh standalone environment, including four prompts up to128 tokens.
- Published compact Core ML weights on Hugging Face and verified24 remote file hashes.
- Fixed HF snapshot-relative manifests and prepared regular package files for Core ML compilation.
- Passed fresh GitHub install + live HF download with existing Bonsai2 target: CLI and unmodified example;13 tests.
- Full conversion CLI, cold-request latency work and oMLX integration remain pending.
