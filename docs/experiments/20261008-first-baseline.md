# First image-only baseline — 2026-10-08

Status: **running; no model quality claim yet**. Source revision `359c831`; bundled source SHA256 `64a0f7501c8d4097999c62b5b81d764b5a7b4c4409561c42f99237fe999d0bb3`.

## Hypothesis and scope

Frozen DINOv2-Small features with a confidence-weighted, finding-specific attention head can rank study abnormalities better than a constant training-prevalence predictor. This is the first real-data experiment, not a fine-tuned image encoder. Data budget: at most 2,000 weakly labeled training studies, 300 weak-label validation studies, and all 58 official labeled studies. Seed 42. Stop after at most 5,400 seconds of extraction and 900 seconds/80 epochs of head training; early stopping patience 12. Private Kaggle notebook cap 7,200 seconds, within the observed 30 free GPU hours. No Colab runtime allocated for this run.

## Evidence and data

The real CSV audit found 4,407 studies, 24,371 series, 58 fully labeled studies, and no patient identifier in the CSV. All studies have metadata for the three planes. Four unlabeled studies share normalized reports with the official labeled set and are excluded. No train/test study or series overlap. The eligible full split is 4,345 train / 58 validation / 4 excluded; its SHA256 is `f8e5651f280966b95f58d9cf31ba282e37dbed8912b482a1409ee37b31c5167a`. Subsampling and weak validation are deterministic at report-group level. Observed DICOM patient hashes, if available, join transitive report/patient components and exclude related training/weak-validation rows. Missing or anonymized identity cannot establish repeat-patient separation.

Raw data remains unchanged in ignored storage. [DINOv2-Small](https://www.kaggle.com/models/metaresearch/dinov2/PyTorch/small/1) is an independently pretrained Apache-2.0 model. [Steven Lee Hans's public v1 labels](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels) are CC0; exact 0.5 denotes an unmentioned finding and receives zero loss weight. Other confidence weights are `2*abs(p-0.5)`. We do not use v2/v4's documented gold-guided adjustments. Even v1's upstream prompt development independence from gold is unverified; gold AUC is exploratory, not an untouched independent estimate.

## Pipeline and evaluation

Three MRI planes, one deterministic fluid-sensitive-preferred series each, up to eight geometrically ordered slices, 160 mm physical crop, 224 px. Rescale/padding/MONOCHROME1 handling, per-study percentile normalization, slice and plane masks are shared between training and inference. Features concatenate DINO CLS and mean patch embeddings (768 dimensions). Feature normalization is fitted only on valid training slices. The image encoder is frozen; the head uses per-finding attention, dropout, AdamW, and masked soft-label BCE.

Epoch selection uses only weighted BCE on the weak-label holdout. Gold AUC is reported per target with positive/negative counts; the primary macro is defined only if all 12 target AUCs exist. A secondary macro excludes and lists undefined targets. All 12 targets have both classes in the gold data, so the initial constant-probability reference is exactly 0.5. Training prevalence will be fit from the selected training labels only. Test inference reads only images, series descriptors, test IDs, and weights.

## Execution and reproducibility

- [Training notebook](https://www.kaggle.com/code/ozgung/knee-dino-train-20261008), version 1.
- Rebuild: `python3.12 scripts/build_kaggle.py outputs/kaggle/train knee-dino-train-20261008 scripts/train_kaggle_entry.py --internet`.
- Run: `kaggle kernels push -p outputs/kaggle/train --timeout 7200`.
- Inspect: `kaggle kernels status ozgung/knee-dino-train-20261008`; `kaggle kernels logs ozgung/knee-dino-train-20261008 --follow`.
- Remote artifacts: `/kaggle/working/artifacts/features.npz`, `extraction.json`, `attention/head.pt`, `attention/training.json`, `wheels/`, and wheel hashes. Private linkage information never belongs in Git.
- Retrieval: `kaggle kernels output ozgung/knee-dino-train-20261008 -p outputs/experiments/dino-frozen-20261008`.
- No intermediate extraction checkpoint: a failed extraction must rerun. Runtime terminates at completion or its declared bound.

## Verification and failures

35 local tests pass, covering CSV/submission invariants, grouped exclusion, exact target coverage, AUC support policy, DICOM geometry/intensity/padding, missing-plane masks, train-only feature statistics, and unknown-label gradients. Independent CPU checkpoint reconstruction tests matched both head types exactly, including missing-image prior behavior. These do not establish real-image quality or offline Kaggle success.

The initial environment probe failed before project code because the generated notebook lacked `kernelspec`; fixed in the builder. Probe version 2 and the first experiment were then launched. Initial automatic approval rejection for Kaggle execution was resolved by verifying the private code-only payload, authenticated account destination, and the user's explicit first-submission authorization. No rejected action was bypassed.

Results, offline resource measurements, and submission score will replace this pending section after verification.
