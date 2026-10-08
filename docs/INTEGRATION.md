# Backend contract and oMLX integration proposal

The public adapter accepts FP16-compatible embeddings [1,6,5120] and committed target features [1,N,25600]. Features concatenate target layers5/19/33/47/61 in that order. `reset(position)` sets the first absolute context position; each feature advances drafter state exactly once. The context branch forwards identical fused context across three shards. Only the last feature computes the full proposal hidden block.

The caller retains target prefill, shared LM head, Top16/CandidateSelector, target verification, accepted-prefix rollback, correction/bonus and EOS. The standalone Generator implements this greedy loop; streaming is not implemented. KV capacity31 and chronological masking must match the verified draft contract. One instance is single-request/synchronous; no concurrent use.

Suggested upstream change: opt-in drafter backend selection, model-manifest and bridge configuration, lifecycle reset/close, graceful initialization fallback, and request cancellation. This adapter is not yet bound to `DFlashEngine` or `dflash-mlx`. After a mid-request failure, a caller must rebuild/replay state or restart generation; blindly switching backends using stale state is invalid.

Existing oMLX ANE prefill uses private ANE APIs for target projection slices. Our public Core ML path accelerates drafter layers during generation. See official [ANE prefill documentation](https://github.com/jundot/omlx/blob/main/docs/experimental/qwen35_ane_prefill.md) and [DFlash target prefill proposal](https://github.com/jundot/omlx/pull/3058). Their prefill and this draft backend could complement each other, but their combined numerical and performance effects are untested.

The standalone example supports proposal budget2 and guarded deferred GDN history. These target-runtime policies are separate from the reusable CoreMLBackbone API. oMLX integration and upstream PR work are on hold at the user’s request.
