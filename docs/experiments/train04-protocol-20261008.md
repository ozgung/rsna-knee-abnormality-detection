# TRAIN-04: stronger own image reference and controlled soft targets

Prospective protocol, October 8, 2026. No TRAIN-04 fit has started. This replaces the old unequal-sample spending gate for this experiment; the old failures remain recorded. The goal is a useful independently fitted image model and a defensible supervision comparison, not a promised public score.

## Hypothesis and comparisons

A generic ImageNet-22k/1k ConvNeXt-Tiny, with randomized coverage of six MRI sequence slots and more training studies, should improve upon the fixed-view 512-study pilot. This first run deliberately changes several factors and establishes a new reference; it cannot attribute gains to any single factor. Once fixed, the second run changes only training targets to the previously audited four-table soft consensus on originally observed cells. Validation targets, observation masks, confidence weights, initialization, image sampling, optimizer and schedule stay identical.

The reference comparisons are the original train-only prevalence predictor, the selected 512-study image models (BCE 0.49586048 and 0.48837184), and the 2,000-study frozen DINO head on exactly these 200 development studies (BCE 0.44760065, weak macro AUC 0.85756882). The latter used 300 studies for its original selection, so it is descriptive rather than a clean paired training experiment. Public knee checkpoints are excluded from initialization and teaching.

## Frozen membership and supervision

- Reconstruct the original 2,000 training assignments, keeping their order and whole transitive report/patient components. Require every row to have `training_candidate` in the full identity manifest. Do not refill an excluded row from development or sealed data. If any original training assignment conflicts with that manifest, stop and reconcile before a fit.
- Development is exactly the unchanged 200 IDs, order and Steven-v1 labels in plan `1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942`. All 300 original development rows remain excluded from training, including the 100 not evaluated here. All gold, gold relatives, quarantined and 300 sealed rows remain excluded. Validate component disjointness and subset fingerprints before decoding or constructing targets.
- Pin full identity manifest fingerprint `911f1605497f5c181402e3fbf70c26d588ffb9c73d6cb2d67e45f22eec52245d` and file SHA `21c71745703d742585684298494a6269ae4217fadc5fc4c26f3bf0d10a289467`. Identity anonymization and undetected paraphrases remain limitations.
- Reference targets use original Steven-v1 file SHA `96444cd3c7d396bae4a71454317d9cbe0689759b5278b30939781e04d0774537`. Exact 0.5 is unknown; confidence is `2*abs(v1-0.5)`. Use target-normalized confidence-weighted BCE. Fit prevalence and any class-bias initialization on training only.
- The controlled alternative averages finite probabilities from Steven-v4, Pilkwang-v2, Lixin and the named older Qwen table, clipped to [0.01, 0.99], using the exact hashes in the [label audit](../../outputs/analysis/label-discordance/aggregate.json). Require valid source status where supplied and complete selected-row coverage. Do not silently substitute newer versions. Keep original-v1 confidence weights, including zero for all originally unknown cells. A consensus of 0.5 on an observed cell is a valid soft target with its original nonzero weight; target and weight must be separate tensors. This tests soft target quality, not increased supervision density. Derive the alternative training prior using these same weights; share the reference initial classifier bias to avoid another changed factor.
- No gold or sealed metrics, no reports in inference, and no public competition-trained predictions in either run. The consensus sources are correlated and partly gold-guided; neither they nor v1 establish independent clinical truth. A future OOF experiment is needed to test image-derived supervision of unknown cells.

## Images and bounded CPU cache

Reuse pinned Apache-2.0 `preprocess.py` and `knee.py` from the public reader; preserve their source hashes. Select the largest usable series per each of six plane/fat-suppression slots, deterministically, with a stable series-ID tie break. Sequence names do not imply T1. Preserve the public canonical orientation, physical 153.6 mm field, normalization and maximum 64 slices; then resize each slice to 256px with area interpolation and store uint8. Keep every resulting slice in that selected series, rather than four fixed triplets. This is a compact new recipe, not exact public training reproduction at cached 384px.

Store variable-length volumes by slot in compressed per-study files with explicit lengths/masks, source-series provenance and hashes in private manifests. Never concatenate planes when drawing a triplet. Missing slots remain masked; empty studies, malformed geometry, nonfinite arrays and unreadable selected series fail the strict cache build. Keep public inference's resilience work separate from this new strict training cache.

First decode 64 studies selected by a fixed label-blind ID hash across the selected cohort. Cap the pilot at 30 CPU minutes and the complete remote cache at six CPU hours, 16 GiB total output and a 5 GiB free-disk reserve. Expansion requires measured decode/compression time extrapolated with 1.5x margin, actual bytes with 1.25x margin, plus measured setup overhead, all within those limits. Actual free disk must also cover the projected output plus reserve. Stop on a failed gate; revise and record the protocol before another build. Preserve completed pilot arrays and rehash them when reused.

Only aggregate receipts and at most two QA arrays come to this Mac; its 2 GiB cache ceiling remains unchanged (22 GiB disk free at this session's check). Full arrays stay in private Kaggle outputs. The training notebook mounts the cache, rehashes every file once, and loads one study at a time through a bounded host cache. Test slot boundaries, deterministic evaluation, variable randomized training centers, masks, short-series handling and quantization before real caching; visually compare the two QA studies with the earlier gallery.

## Fixed training recipe and GPU contract

| Setting | Predeclared value |
| --- | --- |
| Initialization | Generic `convnext_tiny.fb_in22k_ft_in1k`; full encoder adaptation; record actual downloaded revision/config and tensor SHA before any update; alternative must use the exact same initial state |
| Architecture | Pinned public six-slot projection, slot/position embeddings, two-layer transformer and finding attention; reject all-missing studies |
| Sampling | Eight stratified random three-slice windows per present slot during training; twelve deterministic centers per slot for development/train diagnostics; positions relative to that series |
| Image augmentation | Pinned public per-series rotation/scale/translation/gain/gamma at 256px and 0.92 field fraction; no additional flip/TTA |
| Optimization | Seed42; batch2; FP16 with GradScaler; AdamW encoder LR1e-4/head5e-4, weight decay0.02, gradient clip3; all parameters trainable |
| Horizon | Twelve epochs; cosine schedule across the fixed horizon, no restart from previous pilot; skipped AMP updates do not advance scheduler or EMA |
| EMA | 0.998 with existing early-step ramp; evaluate EMA each epoch |
| Selection | Minimum unchanged-v1 confidence-weighted macro development BCE; earliest epoch wins exact ties; no gold/LB selection |
| Diagnostics | Fixed label-blind train64, deterministic twelve-window evaluation; per-target BCE/AUC/support, undefined targets and AMP skips recorded |
| Budget | Each run: at most 600 seconds setup plus 10,200 seconds including cache verification, timing gate, fit, validation and checkpointing; combined pair at most21,600seconds (six GPU hours); retain >=8 Kaggle GPU hours before launch |

Measure 16 warm-up and 48 timed supervised forward/backward batches without optimizer updates; restore RNG and clear gradients afterward. Also measure validation and train-diagnostic throughput. Require all twelve epochs to fit the remaining budget with 1.5x timing margin and 180 seconds for final checkpointing; do not silently shorten the horizon, alter batch size or reduce coverage if it fails. Record peak memory. Abort nonfinite loss/predictions, loss of all supervision, resource exhaustion, or more than five consecutive AMP overflows. Budget stops save an incomplete run and cannot be reported as twelve-epoch completion.

Save model, EMA, optimizer, scheduler, scaler, RNG, shuffle/cursor, cumulative elapsed time and selected-checkpoint evidence. Any resume must match the entire source/config/cache/split/initialization contract and consume the original remaining budget. No assumption of CUDA resume correctness follows from CPU tests.

## Decisions after results

The primary comparison is all-12-target weak macro AUC at the BCE-selected checkpoint; it is undefined if any target lacks a class. BCE selects the checkpoint and is a secondary aggregate comparison. Always report positive/negative/unknown support per target and report-group paired bootstrap intervals (2,000 draws, seed20261008); reused development data and weak labels limit interpretation.

1. Run the controlled target comparison only if the reference completes, has all twelve AUCs defined, beats its training-prior BCE by at least0.05, and achieves development macro AUC >=0.82. This is a prospective resource gate, not a claim of significance. If it fails, diagnose the fixed recipe once before any further fit; no duration sweep.
2. Prefer the alternative for further work only if its selected checkpoint improves macro AUC by >=0.005 and worsens BCE by no more than0.005 relative to the new reference. Otherwise retain the reference and report the ambiguity, including confidence intervals. Do not select different labels per finding from this cohort.
3. Consider an own-model competition rehearsal only if the chosen run has macro AUC >=0.855 and BCE <=0.457601 (the descriptive 2,000-row frozen reference plus0.01). These bounds govern spending, not predicted leaderboard performance. Any entry also needs independent image-only offline inference, exact schema checks and measured runtime/memory limits. A failed gate does not permit an ad hoc leaderboard probe.
4. Only then consider grouped OOF teachers or a complementary backbone. Any changed gate or recipe must be recorded prospectively; the earlier512-row failures remain unchanged. The 0.936 submission stays preserved regardless of these results.

Implementation is pending. Runtime contracts must bind all source hashes, exact data and target fingerprints, environment, initialization tensor hash, selected checkpoint, predictions and timings. Passing unit tests alone does not establish real cache, CUDA training or competitive performance.
