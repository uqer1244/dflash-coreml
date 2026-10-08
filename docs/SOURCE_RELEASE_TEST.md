# Extracted source release test — 2026-10-08

Tested the prepared `dflash-coreml-0.1.0a1-source.zip` outside the project, with a new Python3.11 venv and a non-editable install of `.[generate]` from public packages. The Swift bridge was built from the extracted source. The target and compact converted bundle were local model directories; the target's cache location does not require an installed oMLX app. Live HF snapshot download was not part of this gate.

- Swift release build: PASS.
- Unit tests:11 PASS.
- Model manifest and all six package SHA inventories: PASS.
- Four generation fixtures: token IDs and EOS/length match prior research results.
- Repeated prompt reset:32-token reference match.
- No-thinking smoke: `The capital of South Korea is Seoul.`

| Fixture | Generated tokens | Finish | Reference IDs/EOS | One-run output tok/s |
|---|---:|---|---|---:|
| short |41|eos|PASS|4.87|
| medium |128|length|PASS|7.26|
| code |128|length|PASS|9.03|
| reasoning |128|length|PASS|8.98|

These are correctness runs, not repeated performance comparisons. See `STANDALONE_VALIDATION.json` for runtime versions and archive identity. Source tests used the same runtime source files as the final bundle; validation documentation was added afterward.

The compact bundle is approximately2.885 GB /2.687 GiB versus5.041 GB before unused context weights were removed. Every referenced constant byte is preserved. Actual Core ML state and the next proposal output match across264 commits /80 cycles /four fixtures, including capacity31 rollover. See `CONTEXT_COMPACTION.md`.

HF weight upload and live download validation remain pending. oMLX integration/PR remains deferred. General selector losslessness, other model combinations, incremental resident memory and total generation energy are outside this gate.

## CLI and first-request timing

Both `generate --coreml-model` and `generate --vanilla` return the same9 Seoul-answer tokens and EOS. Fresh Core ML runs took56.44 and48.23 seconds for the first generation. An immediate repeated request in the same instance took1.91 seconds; vanilla took1.95 seconds. The slow first request is reproducible and its cause has not been isolated. Treat startup/first-request latency as remaining release work. These are individual checks while HF upload was running, not a speed comparison. See `CLI_COLD_WARM_VALIDATION.json`.
