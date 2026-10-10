# First public-model adaptation protocol — October 10, 2026

Parent accepts Astra High's prospective recipe before any adaptation outcomes. This is a new conservative adaptation experiment; it does not reproduce the author's training schedule. Preserve the best0.950 entry and the frozen public checkpoint.

Hypothesis: two low-learning-rate passes over the original masked weak labels may improve the inherited full knee model without forgetting its useful representation. Public training-population exposure makes local differences development evidence; they cannot establish independent generalization or yield clean OOF teachers.

| Component | Fixed choice |
| --- | --- |
| Population | Existing4345 eligible studies, original fixed fold0/quarantine. Expected3475 training/870 development; verify exact membership and observed-label support before launch. No gold or bootstrap targets. |
| Inputs | Public96-slice layout,2–98% span,140mm crop directly to384, central320 crop and public normalization. Preserve partial masks and boundary policy. No corpus44 substitution. |
| Initial state | Strict full encoder/head state SHA7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef, explicit320 geometry. |
| Optimizer | AdamW, betas(0.9,0.999), eps1e-8, encoderLR1e-6, complete headLR1e-5. Matrix/kernel decay0.01; biases and normalization affine parameters decay0. All intended parameters included. |
| Duration | Two complete epochs, seed42, batch1, accumulate16 contributing studies. Flush the partial group with its actual denominator; expected218 updates/epoch,436 total,3-study tail. Counts are checked, not forced. |
| Schedule/precision |20 successful-update warmup then constant LR. FP32, norm clip1.0 after accumulation. Pilot fixes checkpointing/windowchunk4 settings; no silent precision fallback. |
| Views |12 uniformly sampled admissible centers per study; no replacement when sufficient, otherwise replacement. Deterministic epoch/study RNG. No spatial/intensity augmentation; original head dropout0.2 during training. |
| Normalization | Freeze BN running statistics/counters, train affine parameters; reapply BN-eval after model.train(). LayerNorm remains trainable. |
| Loss | Unweighted masked soft BCE, observed-cell mean within each contributing study, then study mean. Unknown cells stay masked; no class weights/distillation. |
| Endpoint | FP32 arithmetic mean of the two epoch-end parameter states. Preserve original BN buffers; nonfloating buffers must agree. No metric-selected checkpoint. |

Evaluate only the unchanged public control and fixed averaged endpoint on the entire fold0 using identical94-window anatomical-mirror inference. Record per-target AUC/support/undefined targets and unweighted defined-target macro AUC. A2000-resample patient/report-group cluster bootstrap (seed42) quantifies the paired macro difference. Do not open intermediate validation or gold metrics to change this recipe.

Reject mechanical failures, wrong lineage/geometry, missing updates, nonfinite state, BN drift, broken optimizer/RNG resume, or strict reload/parity failure. Reject clear macro forgetting when the paired95% interval's upper bound is negative, or coherent target damage when at least3 targets lose≥0.01 AUC with≥20 observations in each class. These are experiment triage rules. A nonpositive point estimate is unpromising; do not expand folds/epochs. Positive macro with no damage may justify one separately rehearsed public candidate, with uncertainty stated. Local gains do not automatically justify a blend.

Before fitting: actual full-state CUDA qualification PASS (done);32-study nongold real-image cache/window parity pilot; a disposable exact-AdamW pilot with at least two16-study updates and a3-study tail, unchanged BN, full optimizer/RNG/sampler save-resume parity; measured extraction/loading/training/evaluation throughput and actual remote storage. Discard pilot weights and reset to the pinned public checkpoint for the fit.

The first mechanical CPU pilot will establish exact96×384 bytes, masks and central320 windows; frozen model prediction parity is a separate CUDA prerequisite. Central320 caching can reduce roughly57.3GiB to39.8GiB after proving crop/mirror equivalence, because this recipe has no crop jitter. Crop only after the public full-frame quantization; do not recompute percentiles. Use study-aligned32–64-study shards, bounded resident buffers, validated private storage and exact shuffle/identity manifests. This reduced cache cannot support later arbitrary native ROIs. Verify actual platform disk/output limits before scaling; no arbitrary spending reserve.

Evidence: Astra addendum `outputs/analysis/astra-method-decision-20261010/recommendation.md`; parent [warm-start qualification](public-warmstart-qualification-20261010.md). No adaptation fit or full cache rebuild launched at this checkpoint.
