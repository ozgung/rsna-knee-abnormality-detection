# First image-only baseline — 2026-10-08

Status: **first submission COMPLETE, public score0.756; second submission pending scoring**. Source revision `359c831`; bundled source SHA256 `64a0f7501c8d4097999c62b5b81d764b5a7b4c4409561c42f99237fe999d0bb3`.

## Hypothesis and scope

Frozen DINOv2-Small features with a confidence-weighted, finding-specific attention head can rank study abnormalities better than a constant training-prevalence predictor. This is the first real-data experiment, not a fine-tuned image encoder. Data budget: at most 2,000 weakly labeled training studies, 300 weak-label validation studies, and all 58 official labeled studies. Seed 42. Stop after at most 5,400 seconds of extraction and 900 seconds/80 epochs of head training; early stopping patience 12. Private Kaggle notebook cap 7,200 seconds, within the observed 30 free GPU hours. The image extraction ran on Kaggle; a later cache-only head comparison used a bounded Colab T4 session, now stopped.

## Evidence and data

The real CSV audit found 4,407 studies, 24,371 series, 58 fully labeled studies, and no patient identifier in the CSV. All studies have metadata for the three planes. Four unlabeled studies share normalized reports with the official labeled set and are excluded. No train/test study or series overlap. The eligible full split is 4,345 train / 58 validation / 4 excluded; its SHA256 is `f8e5651f280966b95f58d9cf31ba282e37dbed8912b482a1409ee37b31c5167a`. Subsampling and weak validation are deterministic at report-group level. Observed DICOM patient hashes, if available, join transitive report/patient components and exclude related training/weak-validation rows. Missing or anonymized identity cannot establish repeat-patient separation.

Raw data remains unchanged in ignored storage. [DINOv2-Small](https://www.kaggle.com/models/metaresearch/dinov2/PyTorch/small/1) is an independently pretrained Apache-2.0 model. [Steven Lee Hans's public v1 labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) are CC0; exact 0.5 denotes an unmentioned finding and receives zero loss weight. Other confidence weights are `2*abs(p-0.5)`. We do not use v2/v4's documented gold-guided adjustments. Even v1's upstream prompt development independence from gold is unverified; gold AUC is exploratory, not an untouched independent estimate.

## Pipeline and evaluation

Three MRI planes, one deterministic fluid-sensitive-preferred series each, up to eight geometrically ordered slices, 160 mm physical crop, 224 px. Rescale/padding/MONOCHROME1 handling, per-study percentile normalization, slice and plane masks are shared between training and inference. Features concatenate DINO CLS and mean patch embeddings (768 dimensions). Feature normalization is fitted only on valid training slices. The image encoder is frozen; the head uses per-finding attention, dropout, AdamW, and masked soft-label BCE.

Epoch selection uses only weighted BCE on the weak-label holdout. Gold AUC is reported per target with positive/negative counts; the primary macro is defined only if all 12 target AUCs exist. A secondary macro excludes and lists undefined targets. All 12 targets have both classes in the gold data, so the initial constant-probability reference is exactly 0.5. Training prevalence was fitted from the selected training labels only. Test inference reads only images, series descriptors, test IDs, and weights.

## Execution and reproducibility

- [Training notebook](https://www.kaggle.com/code/ozgung/knee-dino-train-20261008), version 1.
- Rebuild: `python3.12 scripts/build_kaggle.py outputs/kaggle/train knee-dino-train-20261008 scripts/train_kaggle_entry.py --internet`.
- Run: `kaggle kernels push -p outputs/kaggle/train --timeout 7200`.
- Inspect: `kaggle kernels status ozgung/knee-dino-train-20261008`; `kaggle kernels logs ozgung/knee-dino-train-20261008 --follow`.
- Remote artifacts: `/kaggle/working/artifacts/features.npz`, `extraction.json`, `attention/head.pt`, `attention/training.json`, `wheels/`, and wheel hashes. Private linkage information never belongs in Git.
- Retrieval: `kaggle kernels output ozgung/knee-dino-train-20261008 -p outputs/experiments/dino-frozen-20261008`.
- No intermediate extraction checkpoint: a failed extraction must rerun. Runtime terminates at completion or its declared bound.

## Verification and failures

60 local tests pass, covering CSV/submission invariants, grouped exclusion, exact target coverage, AUC support policy, DICOM geometry/intensity/padding, missing-plane masks, train-only feature statistics, and unknown-label gradients. Independent CPU checkpoint reconstruction tests matched both head types exactly, including missing-image prior behavior. These do not establish real-image quality or offline Kaggle success.

The initial environment probe failed before project code because the generated notebook lacked `kernelspec`; fixed in the builder. Probe version2 completed: TeslaT4, all30 sampled headers contained geometry and PatientID (five distinct values), DINO shape [2,257,384]. Extraction and the private CPU gallery both completed; measurements are below. Initial automatic approval rejection for Kaggle execution was resolved by verifying the private code-only payload, authenticated account destination, and the user's explicit first-submission authorization. No rejected action was bypassed.

The CPU gallery completed and both fixed examples were visually checked: three planes and 24 decoded slices each, no errors, knee anatomy retained. Private images/audit: `outputs/kaggle/gallery-result/`. No report text or identifiers are drawn. The original pipeline schematic and primary-source research brief are under `outputs/research/2026-10-08/`.

Pre-exclusion supervision audit confirms every target has both positive-leaning and negative-leaning weak probabilities in train/weak-validation. Synovitis is sparsest:329/2000 supervised training labels and41/300 weak-validation labels. Unknowns remain masked. Aggregate details: `outputs/validation/20261008/weak-support-pre-exclusions.json`.

An optional comparison is predeclared: attention seeds43/44 and meanmax seed42, same cache/split, up to300s and80epochs each. Compare singles, all pairs and all-four equal-probability averages only by weak-validation BCE. Preserve baseline; no target-specific fitted blending. This small repeated comparison can overfit the weak holdout. The comparison completed on Colab after the cache was ready. Input packaging includes only project code, compact features and baseline artifacts; no raw images or reports.

## First results and offline acceptance

- Extraction: 2807.88s, 2,358 complete studies, 7,074 series, 56,592 decoded slices; zero errors/missing planes/empty studies. Cache80,772,745bytes. Peak RSS1,770,040KiB; CUDA peak249,929,216bytes.
- All2,358 studies have observed PatientID hashes, one unique hash per study. No observed patient-hash or exact normalized-report overlap across train2000 / weak validation300 / gold58. Anonymized PatientID semantics still limit repeat-patient guarantees. `outputs/validation/20261008/selected-cache-audit.json` records the aggregate audit.
- Training: 5.61s, early stopping; selected epoch11 using weak BCE0.46322939, versus training-prior weak BCE0.59380212. Gold macro AUC **0.72014923**, prevalence macro0.5; all12 targets defined. Gold remains exploratory for the provenance reasons above.
- Offline rehearsal: private `ozgung/knee-dino-rehearsal-20261008` v1, source`efad291`. All58 gold studies decoded again; max cached-versus-raw prediction difference1.78814e-7. Repeated visible-test predictions identical. No decoder errors, missing planes or fallback studies. Temporary raw-image symlink removed before output collection.
- Gold58 inference109.59s including loading; mean1.418s/study, p953.384s. Visible3 inference25.88s including loading; mean2.045s/study. These imply approximately31–44minutes for1300studies, an extrapolation rather than a hidden-test measurement. Peak rehearsal CUDA252,073,472bytes; RSS1,778,056KiB. Verified9-hour limit has substantial measured headroom.
- Downloaded visible submission hash/schema/exact test-ID order and finite[0,1] probabilities verified. Private pure test-only notebook `ozgung/knee-dino-submission-20261008` v1 completed with internet disabled and no weak-label dataset attached.
- **First submission56922773**, sent2026-10-07 22:13:58UTC (October8 local), COMPLETE with public score **0.756**, verified2026-10-07 22:42UTC via logged-in UI and official CLI. The first hidden-test run succeeded; observed end-to-end time from submission to verification was under29minutes (includes queue/scoring; exact hidden inference runtime is not exposed).

| Target | Positive / negative gold studies | AUC |
| --- | --- | --- |
| ACL | 24 / 34 | 0.6924 |
| MCL | 9 / 49 | 0.5873 |
| Medial Meniscus | 26 / 32 | 0.7308 |
| Lateral Meniscus | 23 / 35 | 0.7913 |
| Medial OA | 15 / 43 | 0.8109 |
| Lateral OA | 11 / 47 | 0.7292 |
| PF OA | 21 / 37 | 0.7426 |
| Effusion | 35 / 23 | 0.8584 |
| Synovitis | 27 / 31 | 0.6643 |
| Baker's | 12 / 46 | 0.6522 |
| Contusion | 19 / 39 | 0.7422 |
| Fracture | 18 / 40 | 0.6403 |

Artifacts: `outputs/experiments/dino-frozen-20261008/artifacts/`, `outputs/kaggle/rehearsal-result/`, `outputs/kaggle/submission-result/`. First encoder remains frozen; this score is a starting baseline, substantially below current top public leaderboard scores. Subsequent comparisons must preserve this entry.

## Bounded second-entry comparison

Attention seeds43/44 and meanmax seed42 reused the exact frozen cache, split, normalization policy and labels. At most300s/80epochs per new head, patience12; actual total19.63s. Four singles, six pairs and the all-four equal probability average were ranked by weak-validation BCE alone.

| Candidate | Weak BCE | Exploratory gold macro AUC |
| --- | --- | --- |
| Preserved attention42 | 0.46322938 | 0.72014923 |
| Attention43 | 0.46086096 | 0.71651002 |
| Attention44 | 0.46654191 | 0.71585761 |
| Meanmax42 | 0.47434029 | 0.70155636 |
| Selected all-four average | **0.45957642** | 0.72072733 |

The small improvement on the selection metric does not establish a better competition score. No gold labels or target-specific fitted blend weights entered selection. Results and checkpoint hashes: `outputs/experiments/head-comparison-20261008/comparisons/comparison.json`. Colab used Python3.13.15, torch2.11.0+cu130, numpy2.1.3 and a TeslaT4. Kaggle training used torch2.11.0+cu128 and transformers5.16.1; Colab's transformers5.18.0 was not used to re-encode images. All inference heads share one encoder forward pass.

Colab input transfer: full81MB archive rejected by the contents endpoint400. Installed CLI0.6.0 sent chunk1 without the final marker; successful4MB uploads initially remained invisible. Finalizing chunks through the same ContentsClient materialized20parts at filesystem root, while the kernel cwd was `/content`. Two guarded attempts stopped before training. Corrected root-path reassembly verified archive and per-file SHA256 before the successful comparison. Results were retrieved and checked, session `knee-heads-20261008` terminated, and `colab sessions` confirmed no active sessions. No raw images/reports transferred.

Kaggle rejected2.2MB weight-embedded notebooks with an explicit source-size limit below1MB; no notebook execution began. A private dataset upload containing a gold-prediction fixture was rejected by automatic approval review. The safer payload contains **only four own trained heads plus dataset metadata**, verified to exclude IDs, labels, reports, images and prediction fixtures. Private dataset `ozgung/knee-own-heads-20261008` v1 was created successfully. Rehearsal derives its reference from the feature cache already attached on Kaggle. Pure inference never reads that cache.

Private ensemble rehearsal and pure inference v1 notebooks were pushed with source`5f66959` and sourceSHA`97f06aa2db0373c46f6cb2593c03d001cc4fca4eec7319cbab5288351ad7376e`. Internet disabled; hash-verified weights and codecs. Both notebooks completed. Raw-DICOM gold parity maximum error1.0430813e-7; repeated visible-test predictions identical; no errors or fallback studies. Gold58 inference100.94s including loading, mean1.3691s/study. Rehearsal peakCUDA252,470,784bytes, RSS1,785,860KiB. Pure visible3 inference26.82s including loading. Visible CSV SHA256`be43e19b805c97174f9d0bc051b15f0f170f51c670f7739c0446004ab2398ac8`, official target order, actual test ID order and finite probabilities independently checked. **Second entry56923149** submitted2026-10-07 22:36:29UTC; pending scoring. Three daily submissions remain. Artifacts: `outputs/kaggle/ensemble-rehearsal-result/ensemble-rehearsal/rehearsal.json` and `outputs/kaggle/ensemble-submission-result/`.


After both entries were frozen and submitted, a descriptive paired study bootstrap (2,000 draws, seed20261008;1,999 draws retained because all12 target AUCs were defined) gave baseline macro AUC95% percentile interval[0.66666,0.76795] and ensemble[0.66943,0.76807]. Paired ensemble-minus-baseline difference0.000578 had interval[-0.00978,0.01182]. The small gold gain is consistent with sampling noise. These intervals condition on the current labels/models and do not correct upstream gold influence, model-selection bias or distribution shift. No selection changed. Reproduction script and aggregate JSON: `outputs/analysis/20261008/gold_uncertainty.py`, `gold-uncertainty.json`.


## Community comparison prompted by the public-score gap

Checked the actual rendered notebooks and logs on October8 after the user asked about the gap. Pilkwang source had been inspected before training; Barun's ResNet34 source was first inspected in this follow-up. Neither community run was independently reproduced.

[Pilkwang v15](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1/notebook) visibly reports **public score0.891**. Its [execution log](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1/log) confirms the scored notebook loads20 shared knee-model members, one336px/130mm decode group, six slots,12 cached slices per slot and10 windows per member, then rank-averages20 models. The source constructs partially trainable DINOv2-Small with the last6 blocks and LayerNorm trainable (10.7M encoder parameters), separate backbone/head learning rates, contrast/laterality recovery and finding-specific slot attention. Its fallback training path uses report-derived labels across the corpus and includes gold studies in training when their report group is not held out. The exact historical training provenance of every attached member remains unverified; do not equate the fallback code with proof of each weight's full training history.

Our0.756 first entry uses a **frozen** vanilla encoder, one selected series per plane,224px/160mm physical sampling, eight separately encoded grayscale slices per plane and a small head trained on2,000 weakly labeled studies. The four-head second entry reuses exactly the same representation. Pilkwang's finer physical sampling (130/336≈0.387mm vs160/224≈0.714mm), sequence coverage, learned encoder and model/TTA diversity are plausible reasons for a substantial performance difference. No ablation attributes a precise share of the0.135 public-score gap to any component. Our weak-BCE selection objective also differs from the final ranking metric. Keeping gold entirely outside our training supports the exploratory evaluation policy; it does not invalidate Pilkwang's legitimate public score or imply that using provided gold labels for training is prohibited.

[Barun's ResNet34 v3](https://www.kaggle.com/code/barun2104/baseline-rsna-knee-mri-2-5d-resnet34/notebook) is **not demonstrated to beat this entry**. The displayed run filters to58 manually labeled studies, trains on46 and validates on12. It concatenates preprocessed series, uniformly samples24 slices as24 input channels, adapts ImageNet ResNet34's first convolution and trains for10epochs. Best displayed validation AUC is0.6086 (epoch9); undefined targets are omitted from its mean. Pseudo-label pretraining is listed as future work, not used by that run. No public competition score or submission pipeline was shown. That tiny-fold validation figure is not comparable with our public score.

Plan correction: the frozen first entry established working infrastructure. Seed/pooling averaging was a cheap diagnostic, but the next score-focused work should reproduce a strong published reference under its exact preprocessing/weights and independently test a partially fine-tuned encoder, then sequence/crop coverage. Stop adding head seeds as the main improvement strategy. Check the public weight provenance before use and preserve current evaluation boundaries. No third training run or submission was launched during this comparison.
