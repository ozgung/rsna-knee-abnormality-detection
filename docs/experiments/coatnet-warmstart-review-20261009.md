# CoAtNet warm-start review — October 9, 2026

Static review completed at19:36UTC while both round1 pairs remained running. No training, credential transfer or submission launched.

## Geometry and initialization

The qualified public mirror constructs `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` with **`img_size=320`**, then strictly loads the checkpoint whose metadata says384. The384 suffix therefore does not impose a universal384-only restriction. This does not establish arbitrary dynamic-resolution support. The original training builder omits `img_size`; changing `--res` only changes the tensors, leaving default backbone geometry. Actual selected-resolution forward/backward remains required.

The current training checkpoint loader extracts `backbone.*` and uses `strict=False`, then initializes a new attention/classification head. That is backbone transfer. Retaining the full public model requires an explicit full-model strict load, pinned checkpoint hash, matching target order, architecture and all state shapes. The public and training heads have corresponding `norm`, `att.0`, `att.3`, `clsW` and `clsb` structure; static similarity is not a successful training-runtime load.

Public preprocessing uses96 slices at384, a central320 crop,2–98% slice coverage and94 evaluation windows. The current corpus has44 slices at336,15–85% coverage and interpolation to requested resolution. Upsampling it to384 is an adaptation, not reproduction of the0.949 recipe. CoAtNet224 versus ConvNeXt256 changes resolution as well as backbone.

## Next bounded comparison

Use one fixed eligible fold and identical construction, corpus44 preprocessing, pinned label artifact, seed, epochs, sampling, optimizer and selection policy. Compare generic initialization against **full** public MRI initialization; do not simultaneously change label source or geometry between arms. Start with an actual training smoke: finite evaluation/loss/gradients, encoder and head updates, planned checkpointing/chunking/accumulation, and save/reload prediction parity. Record installed timm/torch and device. No new local backend smoke was possible: timm is absent from the local ML environment.

Report per-target support and masked report-label AUC descriptively. Public all-data/OAI initialization exposes the nominal heldout population; current bootstrap labels also carry indirect fold exposure. Neither produces independent clinical CV or a clean OOF teacher. Do not select labels, epochs or recipe using gold. The existing0.950 artifact remains intact.

## Evidence

- `outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py`:118,177–234,285–305,420–430.
- `outputs/kaggle/trackA-convnext-20261009/train_knee_original.py`:295–319,323–359,525.
- `outputs/kaggle/mirror-rehearsal-20261009-result/reference-rehearsal/rehearsal.json`: existing PASS; checkpoint SHA `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`,292840787bytes.
- [Reproduced mirror entry](mirror-submission-20261009.md), [current strategy](top10-strategy-20261009.md).

Parent checked source excerpts and the archived PASS receipt after a bounded Sol review. Documentation/content checks only this turn; no new CUDA or performance claim.
