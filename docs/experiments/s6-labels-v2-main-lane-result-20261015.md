# Main lane: S6 ConvNeXt on labels v2 — fold-0 result (2026-10-16)

Protocol: [s6-labels-v2-main-lane-protocol-20261015.md](s6-labels-v2-main-lane-protocol-20261015.md) (pre-registered Oct 15, ~19:20 UTC). Aggregates only; no IDs, reports or label rows. Live log and artifact paths: `outputs/s6v2/STATUS.md` (ignored).

## Read-out

TBD

## What was run

| item | value |
|---|---|
| Code | heliosli's pinned scripts from `nartaa/rsna-knee-s6fall-weights` `s6_extras.bin`; all 293 members of nartaa's `code_manifest.json` match on the VM. The repo copies under `ws2-a-prime-project/source` (compact_train 04df7e2e…, compact_infer 8401812c…) do **not** match nartaa's pins and were not used. |
| Trainer | `compact_train_s6v2.py` = `scripts/s6v2/patch_trainer.py` over pinned `compact_train.py` 14767c7a…; nartaa's own wrapper (ae18d2e1…) is not public. Edits: folds from the targets CSV (folds_v0) instead of sha1 % 5; output-bias init fitted on the fold's training rows (pinned code uses the whole pool); all-data switch; per-fold membership receipts; lossless PNG row store. No model, loss, sampler or optimiser change. |
| Recipe | nartaa's S6fall run_config: convnext_small from timm `fb_in22k_ft_in1k` (`model.safetensors` 2457488b…), 16 epochs, batch 2, 64 train windows, crop 288 / cache 320, views v4 with SKM-TEA anchors (anatomy.pt 5904dd16…), 3 slices at 3.5 mm, lr 1e-4 / head 1e-3, wd 0.05, spatial attention, gradient checkpointing, arm E_prior, seed 42, `--only-folds 0`, all_data false. Loader workers 5 per arm (two arms share 12 vCPUs; loader count does not change the sampled data). |
| Environment | Colab A100-SXM4-40GB, Python 3.12.15, torch 2.8.0+cu128 (= nartaa), timm 1.0.20, numpy 2.0.2, opencv-headless 4.12.0.88, pydicom 3.0.1. |
| Pixel cache | Built on the VM from the official archive (HTTP range reads per study, DICOMs deleted after rendering) through `compact_infer.CStudies.rows`, i.e. heliosli's DICOM → native → geometry → SKM-TEA anchors → `px_cache.StudySlices.build` path, the one compact_infer scores test studies with. Rows stored as lossless PNG shards (raw ~310 GB does not fit the disk). TBD rows / studies / errors. |
| Split | folds_v0 (nartaa/rsna-knee-hpo-assets, sha 12bebc89…). No sealed-identity component spans two folds. Fold 0 held out. Gold-58 rows carry blank targets (trainer asserts it and never pools them); the 4 gold-relative studies quarantined by the sealed identity audit are excluded from both arms. Membership receipts: TBD. |

## Targets

- **J0** = mean of nartaa `labels_v1_blend` and riadmohamed42 JEV `HYBRID` on every non-gold cell (both tables are dense on non-gold rows), prior fill for any cell neither addresses (fitted on training folds only; it fills 0 cells), weight 1 per cell, weight-normalised BCE (arm E_prior).
- **V2** = `labels_v2_base_own224_w30` with the same rule (dense: fills 0 cells).
- **Reconstruction gaps.** nartaa's blend weights are not published (0.5/0.5 assumed). "Prior fill for unaddressed cells" is read literally as filling missing cells, which is a no-op on these tables; if nartaa instead replaced report-silent cells (steven 0.25 / pilkwang UNK) with the class prior, J0 differs from their targets on 6–84 % of cells per finding. nartaa's pool is 4,263 studies (sha 378f782f…); its 86 exclusions are not public, so ours is 4,345 minus cache failures.
- Codex accepted the mapping (`docs/experiments/s6-v2-target-mapping-acceptance-20261015.md`).

## Results

TBD

## Limits

- n = 58; macro SE about 0.017. The rule-2 threshold is a point-estimate rule.
- **Gold-58 is selection-exposed for V2, not clean.** labels v2 (`base_own224_w30`) was chosen among about ten label variants by its gold-58 score; excluding gold-58 from gradient training does not restore independence, so V2's gold-58 read-out is a selection-exposed development comparison (the protocol's "clean for our arms" holds for J0 only). nartaa also selected their models on gold-58. Thresholds are kept as pre-registered.
- V2's teacher (own CoAtNet-224 OOF) was trained on our grouped folds, not folds_v0, so V2's training targets for folds 1–4 carry information from teacher models that saw fold-0 studies' report labels. This can favour V2 on the fold-0 OOF (secondary); gold-58 is clean.
- nartaa's fold-0 OOF 0.907 reference definition is unknown; ours is scored against J0 targets ≥ 0.5 (report-derived) and labels_v1_blend ≥ 0.5.
- A gold-58 gain is not proven to be a leaderboard gain.
