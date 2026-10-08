# Model conversion status

The backend loads six already-converted `.mlpackage` directories: three full drafter shards and three matching context-only graphs. No public weight download is configured.

The current research exporters live in `ane-dflash/phase3_state.py`, related `bf16_ane_layer.py`/composition helpers, mixed-precision recovery scripts, and `phase4_1/v1/build_context.py`. They remain in the research workspace because their captured fixtures, local checkpoint and reference implementation dependencies are not yet portable. This preview does not present them as a general standalone conversion CLI.

Before releasing weights or a conversion tool: pin the exact source checkpoint revision/hash and license, package a portable exporter, preserve the chosen mixed precision and state schema, and rerun hidden/state/selector gates. Do not claim support for arbitrary DFlash2 checkpoints.

`tools/local_manifest.py` can create a SHA256 manifest for the current six packages when run from this research workspace. It is a local preparation utility, not a model converter.
