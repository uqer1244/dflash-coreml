# Measured preview results

2026-10-08, M3 Pro18GB. These are historical research-run measurements; packaging does not constitute another performance run.

- Sequential context-only/cached-copy backbone: about59ms per cycle; full state/next-output parity passed on frozen tests. Actual ANE trace recorded144/144 shard predictions with ANE intervals.
- Drafter-only combined system power, three alternating18-second probes: GPU21.778W, ANE8.380W. Draft precision and runtime differ; this is not total generation power or energy/output token.
- Deferred GDN history reduced MLX peak9.321→8.091GiB in the measured target workload. This target optimization is not part of this package's adapter.
- Verification proposal budget5→2: four prompts, five alternating fresh-process repeats, whole-output median7.342→7.924tok/s(+7.9%), steady7.833→8.611(+9.9%). Same final IDs at32/128-token limits and EOS; widths2–6 common prefixes and78 continuation intervals byte-exact. This is an ANE-path policy comparison, not a fresh GPU/vanilla advantage.
- Shared rotation/fused gate-up did not improve throughput; alternate gather dispatch differed numerically and was not adopted.

Open gates: normal drafter selector91.07% below95%; S=2 context batching failed state parity; existing singleton versus batched target arithmetic is not byte-exact. Four prompts do not establish general losslessness. Longer contexts, additional Macs, same-policy GPU controls and whole-generation energy remain required.

The research workspace retains detailed reports and machine-readable timings. Public evidence export is produced by `tools/export_evidence.py` from that workspace, with user paths sanitized and raw models/traces excluded.

Included evidence: [summary](evidence/verification-summary.json), [five-repeat confirmation](evidence/verification-confirmation.json), [long output](evidence/long-output-gate.json), [continuation](evidence/continuation-gate.json), [width gate](evidence/width-gate.json).

Packaging validation: the new adapter's actual Core ML hidden outputs match the frozen Phase4.1 Native adapter byte-for-byte on four fixtures, two calls each, reset at position17 and capacity31 rollover. [Hardware check](../docs/ADAPTER_VALIDATION.json). This validates backbone extraction, not a new target-generation or speed benchmark.
