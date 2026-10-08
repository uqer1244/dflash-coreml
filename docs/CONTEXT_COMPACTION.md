# Context weight compaction

The original context-only graphs retained the entire full-graph weight.bin after graph pruning. Repacking referenced constants reduces the context packages from 2,391,744,494 bytes to 236,040,288 bytes. The complete converted model bundle is approximately 2.885 GB / 2.687 GiB including the selector. Full-package weights are unchanged; source research packages are preserved.

`tools/compact_context.py` uses coremltools 9 blob storage to copy referenced INT8/FP16 tensors, updates only blob offsets, verifies every tensor byte, and checks that the protobuf graph is unchanged after restoring offsets. `tools/export_model.py --compact-context` applies this during export.

Hardware gate: CPU_AND_NE, four frozen fixtures, 80 cycles / 264 sequential feature commits. Actual MLState tensors and the next full-graph output match the full reference byte for byte, including capacity31 rollover. See `CONTEXT_COMPACTION.json` and `COMPACT_CONTEXT_PARITY.json`.

These measurements concern disk size and correctness. Incremental resident memory and post-compaction power have not been measured.
