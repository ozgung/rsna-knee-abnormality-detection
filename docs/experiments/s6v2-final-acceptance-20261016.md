# V2-all and matched S6fall export/CPU acceptance — October 16

Accepted **2026-10-16 16:15 UTC** for actual bytes, provenance receipts and synthetic CPU execution only. Shared solo wrapper integration/sealing and real-image offline T4 qualification remain pending. [Frozen prospective scope](s6-v2all-solo-scope-20261016.md).

## Actual identities

| Arm | Immutable dataset/version | File / bytes | Full SHA256 |
| --- | --- | --- | --- |
| V2 | ozgung/knee-s6v2-v2all-20261016 / 1, private | E_prior_s42_v2all.pt / 200188027 | e32e5976c60a8d2f33954e348f4b310e49e9739daf4b2012335071cbedb8a972 |
| Control | nartaa/rsna-knee-s6fall-weights / 1 | E_prior_s42_fall.pt / 200185003 | 0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12 |

All five V2 files and all three control files were downloaded individually using explicit version1 paths, with actual size/hash verification. V2 official status is ready/current_version_number1 and metadata private=true. Control's owner-only status endpoint returned404; explicit version1 listing/downloads succeed, so that endpoint is not evidence of absence. Public control score0.950 and training claims remain publisher-attributed.

Actual files, full file-map receipt, CPU receipt, descriptors and replay live under [outputs/analysis/s6v2-final-acceptance-20261016/](../../outputs/analysis/s6v2-final-acceptance-20261016/). [Acceptance](../../outputs/analysis/s6v2-final-acceptance-20261016/acceptance.json), [provenance replay](../../outputs/analysis/s6v2-final-acceptance-20261016/audit_export.py), [CPU receipt](../../outputs/analysis/s6v2-final-acceptance-20261016/cpu-pair.json). CPU receipt SHA256 **3dc1da57c52398d53982a4e12ffa404e06a294145776fcab101eeb2d061a948f**.

## Restricted load, constructor, state and masks

[scripts/qualify_s6solo_cpu.py](../../scripts/qualify_s6solo_cpu.py) verifies exact captured compact_train14767c7a… and compact_infer34f433fe… before using the public constructor. Constructor: convnext_small, no initialization file/download, crop288, three channels, spatial attention enabled, extent/state heads disabled, hidden384/drop0.1/freeze16. Canonical12 target order matches. Actual weights_only CPU loads and strict model loads PASS; each raw state has362 finite tensors.

| Arm | Full state fingerprint |
| --- | --- |
| Control | d3864b6581c733a73720cb56d80c303fd3e51a864209576706249d8027bc1f39 |
| V2 | 007b74ed56aaadf40401e4f7fbd35e9585f82d3a09889e2a4a4d9861ff838c27 |

Both constructor-buffer fingerprints **ef435a71100fab4f78e3b5129fb647079454df7a6e418b80ba0ea376f7b10267**; inventory is normalization mean/std float32[1,3,1,1]. Three actual forwards per checkpoint use identical synthetic uint8[1,3,3,288,288] windows, metadata[1,3,12], mask[true,false,true] and production chunk192. Outputs finite[1,12], sigmoid probabilities finite[0,1]; exact repeat difference0, masked image+metadata perturbation difference0, full state unchanged. One view is present; this verifies masked synthetic pooling, not real missing-plane reader behavior. The pinned forward retains its native encoder padding to32. Forward totals9.18s/control and9.16s/V2, not production runtime estimates.

Environment: existing fastgen Python3.12.3, torch2.10.0, timm1.0.24, CPU fp32/two threads. No package installation or model mutation. This is not the T4 fp16 environment and establishes no CUDA/real-reader/runtime qualification.

Exact successful command:

```bash
~/miniconda3/envs/fastgen/bin/python scripts/qualify_s6solo_cpu.py \
  --train outputs/s6v2/dl/extras/heliosli/compact_train.py \
  --infer outputs/s6v2/dl/extras/heliosli/compact_infer.py \
  --control outputs/analysis/s6v2-final-acceptance-20261016/control/E_prior_s42_fall.pt \
  --v2 outputs/analysis/s6v2-final-acceptance-20261016/v1/E_prior_s42_v2all.pt \
  --out outputs/analysis/s6v2-final-acceptance-20261016/cpu-pair.json
```

Use a fresh receipt filename on replay. Different source/model bytes fail before model forward.

## Training and exposure bindings

All four published V2 metadata files match the independently downloaded owner copies. SHA256SUMS declares the genuine export-source path /content/bk_v2all/E_prior_s42_v2all.pt; that path is retained, not rewritten as a local filename. Actual checkpoint hash matches.

Export config agrees seed42/16epochs/batch2/windows64/crop288/slices3/phys3.5/spatialtrue/extentfalse/no state head/no holdout/no review inclusion. Recomputed membership from the local targets_V2_all.csv matches all-data4345, gold58, empty holdout and train∩gold0. Canonical gold supervision columns remain blank; excluded review/quarantine rows are not silently included. Train hash dbd83070…/gold3918e0c2…/empty-holdout e3b0c442… agree with actual export and preflight. Targets full hash e2283884e3b445eba0c62aa3f011811aef3e9fd1d3732259fb80de9e75ac0377. Bound trainer full hash12e430a350e4e0915fb439d5119ffce6a2f4fc6db164f079539bcf0615cbddcc matches the actual local training source.

Owner-collected preflight reports4407 complete features/no errors. Genuine owner-collected V2_all.log has exactly epochs0–15 then ALL DONE, final loss0.3785 at11863s. These are consistent completion receipts, not independent observation or a cryptographic proof of remote execution. Claude reports34752steps/11984s including final work, every epoch backed up, s6v2e stopped15:12 and no active Colab. No duplicate training/backup or VM inspection by Codex. Earlier backupv4 small receipts remain accepted; its full600MB resume tensor was not downloaded or restored.

Gold58 is selection-exposed: V2 was chosen among about ten variants, and teacher/report exposure persists. Trainer stride3gold0.932 is descriptive. Original fold0 BLOCKED_DEBUG_J0_SANITY is unchanged; this all-data fit follows separate post-hoc user authorization. Train∩gold0 does not establish clean validation or isolate label causation. The matched control trained4263 studies per publisher; V2 uses4345 plus different targets, so any competitive difference combines training/population changes.

## Preparation and next admission

Real descriptors now exist. Dependency-free validate_pair verifies captured source6e87350d… and common settings but deliberately returns UNSEALED. Actual byte staging of both checkpoints PASS in temporary exclusive directories, each with exactly one loader-discoverable E_prior_s*_f*.pt file; directories were removed afterward. V2 uses identical-byte alias E_prior_s42_fall_v2.pt. No public control co-load or inference-loader change.

Claude was requested to implement the substantial shared solo39 wrapper/rehearsal in an isolated worktree; Codex owns targeted integration/sealing. Keep original A, source/dependency pins, both boards, full-population rank/recovery, failure deletion, exact target/ID coverage, native guards and ordinary fresh runtime gates. No old publisher DONE assertions or prior runtime waiver. These receipts authorize packet preparation, not submission; live job/quota/slot reconciliation and fresh private/offline T4 evidence for both checkpoints remain required.
