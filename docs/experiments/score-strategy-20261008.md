# Strategy review: a credible route toward public AUC >0.950

2026-10-08. Research only: no model training, submission, weight download or image download in this review. Source notebooks were pulled through authenticated Kaggle CLI and inspected without executing them. The desired score is a target, not a forecast. A reported 0.949 public checkpoint does not establish either eligibility or an independent validation estimate.

## Decision now

1. Finish the standalone public ConvNeXt reader's image-only rehearsal. Its author-reported 0.929 is a useful inexpensive reference, conditional on the separate provenance audit and our actual runtime checks. Preserve its exact preprocessing and three checkpoints; exclude the notebook's large public stack.
2. If the 0.949 CoAtNet's OAI eligibility is cleared, predeclare one global **0.8 CoAtNet / 0.2 ConvNeXt rank blend**, then rehearse and submit that fixed candidate. Do not choose its weight or targets using Gold58, our weak300 or leaderboard probes. If OAI remains unresolved, do not use that checkpoint as an initializer, ensemble member or distillation teacher. An independently audited non-OAI 0.924 reader is a possible alternative: use a single predeclared **0.5 / 0.5** blend with ConvNeXt, if its independent lineage and runtime justify it. This alternative is being audited by the parent task; it was not independently verified here. These are mutually exclusive prospective recipes, not a menu to sweep on the leaderboard.
3. Start our own **generic ImageNet-initialized ConvNeXt-Tiny, six-slot, 2.5D pilot on Kaggle**. First benchmark decoding/cache size on 64–128 studies in a CPU-only job; then cap the initial training pilot at 90 GPU minutes. Keep data within Kaggle initially. Colab becomes useful after a compact permitted cache and working transport exist, not merely because credits are available.
4. Scale a successful single model and validate another fold before broadening the ensemble. More seeds over our existing frozen features have already produced only +0.001 public AUC.

The current public-reference path can cheaply improve the submitted score. Fresh training creates a defensible model-comparison path and the possibility of a genuinely complementary model. Neither guarantees crossing 0.950.

## What the actual public code adds

| Source inspected | Evidence | Consequence |
| --- | --- | --- |
| [nartaa Anatomical Mirror](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) | One CoAtNet checkpoint, SWA epochs 14/11/12; five slots with 26/22/18/12/18 slices, 96 total; 94 triplets; native 384px, 140mm preprocessing, 320px center crop; equal original/anatomical-mirror probability averaging. Source explicitly acknowledges Gold58 reuse and controlled-access OAI. | This is a fine-tuned high-coverage reference, not evidence that mirroring alone explains the score. Reproduce exact inputs only after provenance clearance. |
| [nartaa Efficient 224 Crop](https://www.kaggle.com/code/nartaa/rsna-knee-0945-efficient-224crop) | A different 224-trained checkpoint and SWA epochs, cropping native 384 to 307 then resizing to 224. | Its reported 0.945 is not a controlled resolution ablation of the 0.949 checkpoint. Do not resize the latter to 224 and promise the same result. |
| [goodpjw2008 ConvNeXt reader](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944) | Independent knee reader; six plane/fat-suppression slots, physical orientation/spacing, adjacent-slice triplets, slot and relative-position embeddings, two-layer study transformer and finding-specific attention. Author reports reader alone 0.929 and stack blend 0.944. | A small trained CNN with sensible MRI inputs is a stronger first own-training candidate than a 20-member reproduction. Public score claims remain author reports until reproduced. |
| [Reader training files](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader) | Apache-2.0 dataset metadata; `train.py`, `make_labels.py`, `oof.py` downloaded individually. Default backbone uses `pretrained=True`, 256px, 8 training/12 evaluation windows per slot, 14 epochs, EMA. Labels average four public tables including Steven v4. | “Independent reader” describes a separate training lineage, not random initialization or independent gold labels. Reuse the architecture, not its validation claims. |
| [Apex Grandmaster Stack](https://www.kaggle.com/code/sujanmajhisuzan/rsna-knee-apex-grandmaster-stack) | Reuses the same 0.949 checkpoint and same three ConvNeXt checkpoints, with finding-specific rank weights justified using gold strengths. Broad exception fallback can return CoAtNet alone. | It is not a new independent model family. Its place at the top of a CLI sorted listing does not provide a numeric score. Do not copy its gold-derived weights or silent fallback behavior. |

Two implementation details deserve explicit separation between reproduction and new training:

- CoAtNet `_eval_centers` spans the concatenated 96-slice volume. With all slots present, **8 of its 94 triplets straddle one of four slot boundaries**, mixing sequences or planes; internal missing slots are not independently excluded by that center selection. This is a source-derived count, not a measured performance defect. Preserve it for an exact checkpoint reference; construct triplets *within individual series* for fresh training. “94 adjacent windows” does not mean 94 anatomically contiguous windows.
- Its mirror reverses sagittal triplet channels and flips image width for coronal/axial triplets. A universal horizontal flip is different. The nominal 320 crop covers about 116.7mm of the 140mm native field, versus our current 160mm input; crop coverage and resolution both change. The smaller field could omit useful peripheral anatomy, so retain a broader field as the default for fresh training until a controlled comparison supports cropping.

The reader training files also expose practical hazards: hardcoded CUDA BF16, absolute local paths, `persistent_workers=True` even if workers become zero, and validation binarization that turns exact 0.5 into a negative. T4 training needs supported precision, normally FP16 with gradient scaling. Its optional mirror argument defaults false and the downloaded training call does not enable it; do not claim the released reader was trained with that option. Its report grouping covers weak studies but does not explicitly purge weak duplicates of gold. Keep our stronger exclusions.

The reader's own notebook reports that mixed model/report labels improved weak OOF AUC from 0.888 to 0.903 while its five-fold public score fell from 0.930 to 0.929. The explanation offered by the author is a hypothesis. The concrete warning is that better agreement with model-derived labels need not improve the competition endpoint. The optional grade mapping also converts “not addressed” to low probabilities and infers synovitis probabilities from effusion; do not import these assumptions without a training-only evidence basis.

## Our current limitations, versus verified infrastructure

The current [first experiment](20261008-first-baseline.md) uses an entirely frozen DINOv2-Small encoder, three selected series, eight individually encoded slices per plane, grayscale repeated into RGB, and 224px/160mm inputs. The head has plane embeddings but no slice position. It never receives a local three-slice image neighborhood or learns MRI-specific encoder features. It trains on 2,000 weak studies, rather than all eligible weak training material. These are large representation and supervision differences from the strong public readers; their separate causal effects remain untested.

There is also unused sequence information: non-fat-suppressed sagittal and coronal series cover 4,266 and 3,406 of 4,407 studies. Our one-series-per-plane policy usually discards these. The metadata's fluid-sensitive and fat-suppression fields are identical in all 24,371 rows; non-fat-suppressed must not be renamed “T1” without DICOM evidence. Synovitis supervision is sparse in the current sample: 329/2,000 training and 41/300 validation entries have nonzero confidence.

The existing pipeline still supplies useful tested components: train-only statistics, label masks, strict schema, explicit fallback accounting, zero decode errors in the bounded extraction and raw-image/cache prediction parity within about 2e-7. There is no evidence here of a submission-column bug explaining the entire gap. Preserve these contracts while replacing the representation. Frozen-feature seed sweeps cannot recover discarded slices, series or spatial information.

## Validation contract before another training run

**Public-reference track.** Gold58 was already used by public model authors and public label developers. Public checkpoints trained on competition weak studies include much of our weak300. A newly shuffled split cannot undo that history. Their local gold/weak metrics may diagnose an ordering, preprocessing or catastrophic loading error; they must not select weights, crops, epochs, TTA or label thresholds. A label-free rehearsal verifies inference, not accuracy. Report this distinction with every public-reference result.

**Own-training track.** Initialize from generic ImageNet weights, or generic DINO weights for a later controlled alternative; no competition-trained teacher or initializer enters the first clean comparison. Freeze a label-table hash and its provenance before running models. Our current public v1 report labels have unverified gold-independent prompt development, so even a correctly held-out new model measures performance against a *weak-label proxy*, not independently established clinical truth.

- Keep related patient/study/series/slice data together. Use patient identity if it can actually identify repeat patients; unique anonymized PatientID hashes alone do not establish that. Otherwise group normalized duplicate reports, audit near duplicates without exporting report text, and state the residual repeat-patient limitation. Preserve gold and the four known gold-related weak exclusions.
- Use the existing weak300 as a development comparison set for the first pilot, so current baseline predictions remain comparable. It is already a development set, not a pristine final test. Before new modeling, deterministically reserve an additional approximately 300 eligible study groups from the previously unused material as a sealed weak-proxy check; freeze their manifest and do not inspect candidate metrics there. Sizes depend on grouping and target support.
- Once a pilot works, create grouped folds over the remaining eligible material, with all transformations and fitted label mappings restricted to the training fold. Obtain true out-of-fold predictions from fresh initializations. The sealed set stays out of every fold's training and out of teacher fitting.
- For weak-label metrics, define observed entries as finite labels other than the source's unknown sentinel 0.5. Use `label > 0.5` only on observed entries. Report per-target positive/negative/support counts. The primary comparison is the mean of all 12 observed-label AUCs **only when all are defined**; otherwise mark that primary undefined, list missing targets, and label any supported-target mean as secondary. Confidence-weighted BCE remains the epoch-selection metric, fixed in advance, and a secondary architecture check. An AUC against thresholded soft reports is not the official image-label AUC.
- Retain `2*abs(p-0.5)` confidence masking for the first pilot rather than changing architecture and label semantics together. Report support and weighted training priors per target. Do not turn unmentioned findings into negatives or tune synovitis from gold. Evaluate the training-prior reference on the same held-out rows.
- After the architecture and fold ensemble are frozen, use the sealed proxy once and report Gold58 once as descriptive, exposure-limited results. Do not return to tune on either. The available gold cannot supply an unbiased estimate after historical reuse; independent expert image labels would be needed to resolve that, beyond this pilot's scope.

## Ranked pilots and escalation gates

These are proposed execution bounds, not measured training times. Each run must record the code/config/seed, source and split hashes, environment, checkpoint state, elapsed time, masks, support and stopping reason. Save optimizer/scaler/scheduler/EMA and RNG state so a bounded run can resume without silently starting over.

| Priority | Hypothesis and bounded comparison | Budget / promotion gate |
| --- | --- | --- |
| 0: exact eligible reference | Finish three-fold ConvNeXt rehearsal; then test one separately eligible independent reader and a predeclared global blend. Measure actual wall time and peak memory; use all actual hidden test IDs. | Initial 16–32 study benchmark. Extrapolate conservatively before a full run; retain at least a 2x runtime safety margin under the verified notebook limit. No gold-based selection and no silent substitution if a member fails. |
| 1: cache and trainability | CPU-only 64–128 study decode, geometry galleries, missing-slot masks and cache byte measurements. Then a 32-study overfit check, followed by **512 training / existing weak300**: generic ImageNet ConvNeXt-Tiny, six slots, 4 triplets/slot, 256px, slot/position tokens, finding-specific attention and EMA. | At most 10 GPU minutes for the overfit diagnostic and 90 for the pilot, or 6 epochs, whichever first. Batch 1 with accumulation and encoder chunking is a starting point, not a claim it fits. Stop immediately on nonfinite values, label/ID overlap or geometry failure. Fit must beat the training-prior BCE and learn; a tiny noisy AUC improvement alone is insufficient for scale. |
| 2: useful single model | Expand the successful recipe to one grouped fold of the eligible weak corpus, preserving the sealed proxy. Use a frozen generic encoder with the *same input/head* as a small matched control if needed to isolate adaptation from richer coverage. | Benchmark first 100 optimizer steps, then cap the first complete fold at 6 GPU hours and 14 epochs. If projected cost exceeds that, reduce sampled windows before training, or move the measured cache to suitable Colab hardware. Continue only with meaningful weak AUC improvement and non-worse BCE against the comparable baseline; inspect per-target failures before scaling. |
| 3: independent confirmation | Train a second grouped fold, then a third only if the first two support the same direction. Average own OOF predictions; quantify paired uncertainty by study group. | No automatic five-fold run. A proposed practical promotion threshold is +0.01 macro weak AUC over the current baseline, with improvement in both first folds and no systematic target collapse; record this threshold before launch. This is a resource decision rule, not proof of clinical or leaderboard gain. |
| 4: one-factor improvements | On the successful own model, compare 4 versus 8 windows/slot, then 256 versus 320px at fixed physical field, or properly canonicalized anatomical mirroring. Select only one expensive branch at a time. | One controlled pair per question. Require consistent held-out benefit or useful error diversity before another full fold. Do not simultaneously change label table, crop, backbone and ensemble weight. |
| 5: complementary architecture | Partial generic DINO adaptation or an efficient ResNet/CoAtNet on the same split and views; compare OOF residuals as well as standalone score. | Only after a trained CNN exists. Prefer one complementary model to many seeds. A clean OOF global average can be selected with a small predeclared candidate set; public competition checkpoints cannot participate in that “clean OOF” claim. |

Cross-fitted distillation is a later option, not the first pilot. Any teacher predicting held-out training rows must itself exclude those rows and their groups, and eligibility applies to teachers too. Keep fixed report-derived validation targets; do not celebrate agreement with newly teacher-mixed targets as independent progress. A teacher trained on all competition weak rows invalidates an ostensibly held-out student's model-selection signal even if the student excludes those rows.

## Compute and data logistics

The parent verified **28.91 Kaggle GPU hours, 20 TPU hours, 351.8 Colab credits and zero active Colab sessions** on October 8; quotas can change. Reserve at least **8 Kaggle GPU hours** for offline reference checks, final inference and failure recovery. Treat the first own-training wave as at most approximately 8 GPU hours, including the pilot and first fold, before reviewing evidence. These ceilings fit the current free allocation and leave room to change course. Do not port the entire PyTorch stack to TPU simply because quota exists.

Kaggle is the economical first execution venue because the authorized competition images are already mounted. Run CPU preprocessing separately from GPU fitting when possible. Store grayscale slices once, reconstruct triplets from indices, and preserve series/geometry/position/mask metadata; avoid caching three overlapping channels repeatedly. Measure compression and bytes on the initial sample before building the full cache. For perspective, even 96 grayscale uint8 256px slices per study across 4,407 studies are about **27.7GB uncompressed**, before metadata. “Compact” cannot mean it will fit our 2GiB local starter budget.

Keep the full cache remote and private. Confirm competition permissions and platform access before any Kaggle-to-Colab transfer; do not publish it as a data-transfer workaround or copy reports unnecessarily. Test a bounded archive's checksum, upload/download, and image parity first. Public cache eligibility is a separate question from checkpoint eligibility. Start Colab with a 2-hour cap, measure actual credit burn and examples/second, then budget a complete fold. Do not assume a T4 credit rate applies to an A100 or that a larger GPU is cheaper per trained study. Only extend toward the three-fold plan once throughput, validation and usable credit balance justify it; no purchases are needed for the proposed first stage.

Rehearsals must measure report-free inference, CPU/RAM/disk peaks and total elapsed time with all intended ensemble members attached and internet disabled. A notebook can run successfully yet omit a member through a broad exception handler; require exact member receipts and hashes. Use a single global percentile-rank blend to avoid incompatible probability scales, calculated per target over the actual inference cohort with a fixed tie rule. Validate finite [0,1] outputs and identical target/ID alignment before blending. Ranking changes calibration, so retain original probabilities separately for diagnostics.

## Evidence record and acceptance

Source-only snapshots and a SHA256 manifest are under ignored `outputs/research/score-strategy/` in this review's isolated worktree. Preserve them before archiving it. Main notebook hashes:

| Snapshot | SHA256 |
| --- | --- |
| Anatomical Mirror | `8f1f96530b08543bd6caeed48489eea7c01657d6994f22e21a8400047fd4e100` |
| Efficient 224 Crop | `b80d3b72a1c34f2b39b3ea0d2952f039b8d27c3feb5c4db0a7d0ab62ae75354e` |
| ConvNeXt reader/stack | `941802c45fbc21a763c259d54d35f5fe1600f3a269a39d6773edcbf4185f457e` |
| Apex stack | `c3b62e2f8be016abe541e9d5e537217fec73edce41f6fef15d95527c2dbfc75a` |

The reviewed 0.949 checkpoint identity is `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`; no bytes of that checkpoint were downloaded here. Its [published weights dataset](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007) is evidence of distribution, not a resolution of the OAI eligibility question.

Acceptance for this review: primary code inspected, limitations distinguished from measured defects, a ranked bounded plan with honest validation available to the parent, Markdown content/link checks and `git diff --check`. No training or inference accuracy claim is made by this document. The parent owns implementation, eligibility decisions and STATUS integration.

## Additional public scan, October8 01:55UTC

A bounded scan of100 score-sorted and100 recently run competition notebooks,
six source notebooks, four model cards and three discussions did not establish a
stronger reproducible non-OAI checkpoint. API notebook lists expose no numeric
score, so list ordering is not leaderboard-score evidence. Raptor student and PJ D4
extra student inputs were inaccessible in this audit; several other notebooks reuse
known/OAI arms. No new weights or images were downloaded and no extra job was allocated.
Evidence and hashes: `outputs/research/2026-10-08/complement-scan/audit-summary.json`.

[CoolinLai claims0.954 with fivefold ResNet50 at224px](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3529410);
[Tanuki claims0.955 with fivefold single-model CV](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3533599).
Neither comment supplies a reproducible recipe, weights or training-data provenance.
These are author claims, not verified notebook results or proof of non-OAI training.
They support testing training/data coverage before increasing backbone size, without
supporting a forecast for our own score.

After the current Tiny pilot's predeclared spending gates, the next complementary
hypothesis is fresh generic DINOv2-Small, adapting its last six blocks and final norm
on the same six-slot cache/head/split. It needs a CUDA diagnostic and a declared
<=90minute run contract. The user has authorized such experiments, but none has
started; progression remains gated by actual pilot results. A future
comparison would predeclare one equal probability blend with our own ConvNeXt and
require useful held-out improvement rather than low correlation alone. Avoid public
knee teachers in independent holdout claims because they may have seen those studies.
