# Main lane: S6 ConvNeXt on labels v2 — fold-0 result (2026-10-16)

Protocol: [s6-labels-v2-main-lane-protocol-20261015.md](s6-labels-v2-main-lane-protocol-20261015.md) (pre-registered Oct 15, ~19:20 UTC). Aggregates only; no IDs, reports or label rows. Live log and artifact paths: `outputs/s6v2/STATUS.md` (ignored).

## Read-out

- **Verdict under the pre-coded rules: BLOCKED (rule 1 not met), so no promotion and no all-data fit was launched.** J0 is in the gold band, but its fold-0 OOF misses the ±0.01 band by 0.0004 under the evaluator's pre-coded reference.
- **Primary endpoint (gold-58, raw DICOM → pinned compact_infer, stride 1):** J0 **0.9230**, V2 **0.9294**. **V2 − J0 = +0.0064 [−0.0067, +0.0197]**, P(Δ>0) 0.83 (paired bootstrap 2,000, seed 42, no draws skipped). V2 wins 7 of 12 findings, ties 1.
- **Rule 1 (J0 sanity).** Gold 0.923 is in 0.931 ± 0.02. Fold-0 OOF against J0 targets ≥ 0.5 (the reference coded before training) is 0.8966, outside 0.907 ± 0.01 (floor 0.897). Against nartaa's own report-label table (`labels_v1_blend` ≥ 0.5) it is **0.9107** (stride 1) / **0.9074** (stride 3), inside the band. The protocol does not define nartaa's OOF reference. The debug finding is that the miss is a reference-definition issue, not a pipeline fault. Supporting this, nartaa's public all-data S6fall weights score **0.9314** on gold-58 through the same DICOM path.
- **Rule 2 raw thresholds are met** (gold Δ +0.0064 ≥ +0.005; OOF Δ +0.007 ≥ −0.010). Rule 3 (kill) is not met. Whether rule 1 counts as satisfied under the `labels_v1_blend` reference is a decision for the user/coordinator. If it does, V2 is promoted and `scripts/s6v2/vm_alldata.sh` runs the all-data fit after a cache restore. As registered, rule 2 is a point-estimate rule; the CI includes 0.
- **The cache and DICOM paths agree.** Trainer re-evaluation from the pixel cache at stride 1 gives J0 0.9231 and V2 0.9293 (Δ +0.0062). The trainer's stride-3 predictions give 0.9229 and 0.9284 (Δ +0.0055).

## What was run

| item | value |
|---|---|
| Code | heliosli's pinned scripts from `nartaa/rsna-knee-s6fall-weights` `s6_extras.bin`; all 293 members of nartaa's `code_manifest.json` match on the VM. The repo copies under `ws2-a-prime-project/source` (compact_train 04df7e2e…, compact_infer 8401812c…) do **not** match nartaa's pins and were not used. |
| Trainer | `compact_train_s6v2.py` = `scripts/s6v2/patch_trainer.py` over pinned `compact_train.py` 14767c7a…; nartaa's own wrapper (ae18d2e1…) is not public. Edits: folds from the targets CSV (folds_v0) instead of sha1 % 5; output-bias init fitted on the fold's training rows (pinned code uses the whole pool); all-data switch; per-fold membership receipts; lossless PNG row store. No model, loss, sampler or optimiser change. |
| Recipe | nartaa's S6fall run_config: convnext_small from timm `fb_in22k_ft_in1k` (`model.safetensors` 2457488b…), 16 epochs, batch 2, 64 train windows, crop 288 / cache 320, views v4 with SKM-TEA anchors (anatomy.pt 5904dd16…), 3 slices at 3.5 mm, lr 1e-4 / head 1e-3, wd 0.05, spatial attention, gradient checkpointing, arm E_prior, seed 42, `--only-folds 0`, all_data false. Loader workers 5 per arm (two arms share 12 vCPUs; loader count does not change the sampled data). |
| Environment | Colab A100-SXM4-40GB, Python 3.12.15, torch 2.8.0+cu128 (= nartaa), timm 1.0.20, numpy 2.0.2, opencv-headless 4.12.0.88, pydicom 3.0.1. |
| Pixel cache | Built on the VM from the official archive (HTTP range reads per study, DICOMs deleted after rendering) through `compact_infer.CStudies.rows`, i.e. heliosli's DICOM → native → geometry → SKM-TEA anchors → `px_cache.StudySlices.build` path, the one compact_infer scores test studies with. Rows stored as lossless PNG shards (raw ~310 GB does not fit the disk). **4,407/4,407 studies, 0 errors, 2,617,852 rows, 133 GB in 17 shards.** The complete-cache preflight passed with all shard sha256s bound. The cache is backed up as private Kaggle datasets `ozgung/knee-s6-px-v4-320-20261015` (index) + `-s000…-s016`. |
| Split | folds_v0 (nartaa/rsna-knee-hpo-assets, sha 12bebc89…). No sealed-identity component spans two folds. Fold 0 held out. Gold-58 rows carry blank targets (trainer asserts it and never pools them); the 4 gold-relative studies quarantined by the sealed identity audit are excluded from both arms. Trainer membership receipts equal the preflight: pool 4,345 (`dbd83070…`), fold-0 train 3,475 (`db246e0a…`), holdout 870 (`5a0d33a0…`), gold 58 (`3918e0c2…`), train ∩ gold = 0, identical in both arms. The two `run_config.json` files differ only in `targets` and `out`. |

## Targets

- **J0** = mean of nartaa `labels_v1_blend` and riadmohamed42 JEV `HYBRID` on every non-gold cell (both tables are dense on non-gold rows), prior fill for any cell neither addresses (fitted on training folds only; it fills 0 cells), weight 1 per cell, weight-normalised BCE (arm E_prior).
- **V2** = `labels_v2_base_own224_w30` with the same rule (dense: fills 0 cells).
- **Reconstruction gaps.** nartaa's blend weights are not published (0.5/0.5 assumed). "Prior fill for unaddressed cells" is read literally as filling missing cells, which is a no-op on these tables; if nartaa instead replaced report-silent cells (steven 0.25 / pilkwang UNK) with the class prior, J0 differs from their targets on 6–84 % of cells per finding. nartaa's pool is 4,263 studies (sha 378f782f…); its 86 exclusions are not public, so ours is 4,345 minus cache failures.
- Codex accepted the mapping (`docs/experiments/s6-v2-target-mapping-acceptance-20261015.md`).

## Results

Gold-58 per finding (compact_infer, stride 1):

| finding (gold +) | J0 | V2 | Δ |
|---|---|---|---|
| ACL (24) | .988 | .983 | −.005 |
| MCL (9) | .923 | .939 | +.016 |
| Medial Meniscus (26) | .962 | .964 | +.002 |
| Lateral Meniscus (23) | .930 | .897 | −.034 |
| Medial OA (15) | .972 | .988 | +.016 |
| Lateral OA (11) | .822 | .810 | −.012 |
| PF OA (21) | .894 | .860 | −.035 |
| Effusion (35) | .955 | .966 | +.011 |
| Synovitis (27) | .798 | .835 | +.037 |
| Baker's (12) | .967 | .967 | .000 |
| Contusion (19) | .950 | .972 | +.022 |
| Fracture (18) | .914 | .972 | +.058 |
| **macro** | **.9230** | **.9294** | **+.0064 [−.0067, +.0197]** |

The V2 gains are on the findings where labels v2's image teacher helped on gold (Fracture, Synovitis, Contusion, Medial OA, MCL, Effusion). V2 loses on PF OA and Lateral Meniscus, and slightly on Lateral OA and ACL. Lateral OA and PF OA are where the labels-v2 table itself was weaker than the report blend.

Fold-0 OOF (870 studies, trainer re-evaluation at stride 1; partial fold-0 predictions, not full OOF; report-derived, descriptive):

| reference | J0 | V2 | Δ |
|---|---|---|---|
| J0 targets ≥ 0.5 (pre-coded rule-1 reference) | .8966 | .9038 | +.0071 |
| `labels_v1_blend` ≥ 0.5 (nartaa's report table) | .9107 | .9180 | +.0074 |
| same, trainer stride 3 | .8932 / .9074 | .9013 / .9159 | +.0081 / +.0086 |

Spearman V2 vs J0 on fold 0: 0.86 (Fracture) to 0.98 (Medial Meniscus). Per-target support counts and the macro policy (unweighted mean of all 12 AUCs; undefined unless every target has both classes) are in `outputs/s6v2/vmout/readout_ci.json` (ignored).

Training: 16 epochs, 27,792 optimizer steps per arm, 21,609 s per arm, run concurrently on one A100 (0.78 s/step). Final epoch loss: J0 0.3908, V2 0.3791 (nartaa's all-data final: 0.3911). Fold-0 models: J0 sha256 `ba78bfcd…`, V2 `229a066c…`. They are backed up with the last checkpoints in private `ozgung/knee-s6v2-ckpt-20261015`.

Artifacts (ignored): `outputs/s6v2/vmout/` (preflight, readouts, predictions, run configs, membership, pip freeze, code verify) and `outputs/s6v2/STATUS.md` (sessions, timings, deployed script hashes).

## Limits

- n = 58; macro SE about 0.017. The rule-2 threshold is a point-estimate rule.
- **Gold-58 is selection-exposed for V2, not clean.** labels v2 (`base_own224_w30`) was chosen among about ten label variants by its gold-58 score; excluding gold-58 from gradient training does not restore independence, so V2's gold-58 read-out is a selection-exposed development comparison (the protocol's "clean for our arms" holds for J0 only). nartaa also selected their models on gold-58. Thresholds are kept as pre-registered.
- V2's teacher (own CoAtNet-224 OOF) was trained on our grouped folds, not folds_v0, so V2's training targets for folds 1–4 carry information from teacher models that saw fold-0 studies' report labels. This can favour V2 on the fold-0 OOF (secondary); gold-58 is clean.
- nartaa's fold-0 OOF 0.907 reference definition is unknown; ours is scored against J0 targets ≥ 0.5 (report-derived) and labels_v1_blend ≥ 0.5.
- A gold-58 gain is not proven to be a leaderboard gain.
