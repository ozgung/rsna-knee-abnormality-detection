# Project status and session handoff

Updated **2026-10-10 09:31 UTC**. Read this snapshot, then [Fable/Astra handoff](docs/claude-astra-handoff.md). Prior recovery details and receipts: [archived status](docs/status-archive/status-before-20261010-fullfive.md). Durable rules: AGENTS.md; setup: README.md.

## Current snapshot

- Target top10; best public **0.950**, entry56986393. No reproducible top10 solution established. Preserve this entry.
- **Original-label CoAtNet224 all five folds complete.** Parent141-check report/artifact audit PASS; pooled raw-report AUC0.900342 across4345 studies. Folds0–4:0.901842/0.902008/0.896921/0.897643/0.906643. All selected checkpoint hashes, metadata and486 finite tensors per model verified. [Full-five audit](docs/experiments/coatnet224-fullfive-20261010.md).
- Three-fold CUDA rehearsal passed exact raw/corpus/window/probability/repeat parity and16 reader stresses. **Full-five rehearsal v1 PASS** (69.27s); final-visible notebook v1 PASS (94.27s internal, exact CSV parity). **Entry57037296 accepted08:52:58UTC, PENDING; score unknown.** [Subset evidence](docs/experiments/coatnet224-subset-rehearsal-20261010.md), [final wrapper](docs/experiments/coatnet224-submission-preparation-20261010.md).
- Bootstrap384 fold0 recovery/export verified: reportAUC0.918185; all486 exported tensors exactly match local reconstruction. Diagnostic model, not default ensemble member. [Recovery export](docs/experiments/coat384-recovery-export-20261010.md). Original-label384 Colab fold0 logged all12epochs, selected[10,11,9], bestepoch reportAUC0.9126; SWA/export not yet audited.
- ConvNeXt bootstrap branch closed after public0.874 versus0.878 original. Fable launched CoAtNet bootstrapR2 pair in the other Kaggle slot, despite prior disagreement about gold-guided method selection. Treat it as exploratory; gold58 has influenced these iterative decisions and cannot be claimed independent audit evidence. No automatic expansion or blend promotion.
- October10 entries last checked this turn: two used, three remaining. Rules last verified October9:5/day, offline inference≤9h; entry/merge deadlineOctober15, final deadlineOctober22 23:59UTC. No purchases/new terms.

## Ready queue and ownership

| Owner / task | State / next action |
| --- | --- |
| Parent — original224 entry | Entry57037296 PENDING. Inspect score/runtime when complete; do not resubmit v1. |
| Fable — Colab384 original-label family | Fold0 trained; finalbackup ready, payload/export pending. Fable relaunched fold1 after reported reliability fixes; A100 session and epoch0 backup confirmed09:24UTC. Qualification slot released08:53UTC for prepared export; respect actual capacity. |
| Fable — persistence fixes | Re-review:14 focused tests PASS; state separation/final wait/locking/strict resumes improved. Exact remote payload binding and recovery/skip evidence still need correction; preserve healthy training. |
| Fable — exploratory R2 label pair | Already running; preserve fixed comparison and failure records. Report selected-fold/globalteacher/gold-selection limitations; no automatic new round. |

## Exact live jobs and recovery

| Job | Current evidence |
| --- | --- |
| `ozgung/knee-own-coatnet224-rehearsal-fivefold-20261010` v1 | COMPLETE/PASS. Build `outputs/kaggle/own-coatnet224-rehearsal-fivefold-20261010/`. Command `/Users/ozgun/miniconda3/bin/kaggle kernels push -p outputs/kaggle/own-coatnet224-rehearsal-fivefold-20261010`. Outputs `coatnet224_rehearsal/rehearsal-report.json` and `visible.csv`; no local controller. |
| `ozgung/knee-own-coatnet224-submission-20261010` v1 | COMPLETE/PASS, entry57037296 PENDING. Build `outputs/kaggle/own-coatnet224-submission-20261010/`; push with same KaggleCLI. Collect `coatnet224_submission/operational-receipt.json` and root `submission.csv`; no local controller. |
| `ozgung/knee-tracka-coatnet224-r2-fold01-20261010` v1 | Fable launched08:22UTC, occupies second GPU slot. `labels_bootstrap_v2`; do not duplicate or promote automatically. |
| Colab | Fable-owned `knee-c384r-f1-a1`, A100 endpoint `gpu-a100-s-kkb-ass1c2-20ivyft9r6y5b`, active09:24UTC. Fold1 epoch0 completed; serverversion1 backup logged READY09:22UTC. Log `outputs/colab/c384raw-loop.log`; maintained controller `scripts/colab_tracka_folds.sh` queues folds1–4. Fable owns stop/resume; do not launch a competing controller. |

Colab launch ownership is now **Fable only**, with stops coordinated in sharedhandoff except clear runaway/duplicate work. Parent/Sol must not launch another loop. Completed original384fold0 backup `ozgung/knee-colab-resume-c384r-fold0` is serverREADY; smallmetadata confirms12epochs,partialfalse,selected[10,11,9], resume size2,049,835,743bytes. Payload remains remote/restore-unverified. Recovery receipt `outputs/analysis/colab-raw384-reconcile-20261010/receipt.json`. Failed download/cleanup logs preserved as `outputs/colab/c384raw-loop-fold0.log`; stoppedfold1 log `outputs/colab/c384raw-loop.log`, frozen localsources/receipt `outputs/analysis/colab-raw384-fold1-launch-20261010/`. Do not restart oldloops63024/79276. Fable prepared `outputs/kaggle/trackA-export-c384r-fold0-20261010/`; may now push: parent released qualification/entry slot08:53UTC. Fable reports implementing classified retries, strict identity, staged-payload manifests and final-backup verification before this fresh fold1. Parent confirmed active session/first READY log and reviewed the critical paths after Sol re-review. Fourteen focused tests pass; exact durable payload/recovery provenance is still unqualified. No fixed compute reserve or training-time cap is imposed.

Persistence re-review:14 focused synthetic tests PASS, no skips. Accepted/verified retry state separation, final backup waiting, strict resume and loop locking improved. Remaining material gaps: latest listing size/time is not exact accepted-version/payload verification; controller promotes staged payload_epoch to verified and may skip a completed fold without hash/history/fold checks; restored files are marked verified without provenance; unreadable payloads can use sibling metadata epoch. Evidence `outputs/analysis/colab-hardening-rereview-20261010/report.json`. Fable owns corrections before automatic recovery/skip decisions; healthy training continues. Source review also confirms inherited10h Colab cap still enabled; remove from future packages, not active scientific settings. Proxy refresh remains a candidate fix (`outputs/analysis/colab-transport-rootcause-20261010/report.json`); local401/404 does not prove a hard runtime cap.

Completed original224 sources: `ozgung/knee-tracka-coatnet224-fold01-20261009` v3, `...-fold23-20261009` v2, `...-fold4-20261009` v2. Results in matching `outputs/kaggle/trackA-coatnet224-fold{01,23,4}-20261009-result/trackA/`. Staged symlinks `outputs/kaggle/own-coatnet224-inputs-20261010/`. Parent downloads finished; no active downloader. Audit/checkpoint receipts: `outputs/analysis/coatnet224-{parent,fullfive}-audit-20261010/`. Full-five contractSHA `873ea579607fe42159225f316f4c92709dcfba7e7e28c3d54a1d61d02ff494c1`; notebookSHA `2613253fdcd2d15bd173de73a1adde4dc0883d8a8baf326e98178ae4b86c1798`.

Accidental seed42 repeat `ozgung/knee-tracka-coatnet224-s43-fold01-20261010` v1 was canceled after555.4s; preserve its notebook/logs and do not restart. Any future seed43 run must prove resolved command seed first. Old queue6782 and lost bootstrapColab loop32148/32151 must not restart. Recovery backup and exact selected tensor fingerprint remain in linked archive/recovery docs.

## Scientific decisions and verification

- Original224 fixed recipe: native `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`,224px,k8train/k16eval,10epochs,bs4/accum2,gradcheckpoint,windowchunk8,seed42,generic timm initialization,original `labels_v1_blend`. Keep fixed family; scheduler/AMP update ambiguity and dropped final accumulation microbatch are documented, not silently changed.
- Eligibility4345 =4407 minus58gold and4related quarantined studies; known patient/report components do not cross fixed folds. Historical600 release applies only to newCV, old comparisons remain frozen. Fullfold report AUC selects SWA, so OOF is not untouched outerfold evidence. Globalteacher bootstrap adds indirect exposure. Reports may supervise training; inference remains image-only.
- Public MRI/OAI warm-start remains a separate hypothesis requiring strict full-model load and explicit geometry. DINO/TRAIN04/MaViT retries deferred. No blind fusion or gold-based claimed-independent promotion.
- Parent141 full-five report checks PASS; all5 selectedcheckpoint byte/metadata/finite checks PASS. Subset CUDA actual PASS.16 focused runtime/builder tests previously passed without skips; scientific sources unchanged. Full-five real CUDA PASS; final-visible image-only receipt and exact CSV parity PASS; hidden-test entry57037296 remains pending. No new model-training tests required for documentation/artifact collection.

## Git and tools

Branch `codex/token-efficient-instructions`, local Git only/no remote. Parent commits reviewed work, preserves user AGENTS edits and Fable's untracked implementation. Prior checkpointc9fc8a5; final entry receipt is recorded in this update. Protected managedworktree cannot be archived; do not force removal. Main's separate checkout is not claimed current.

ML `.venv/bin/python`; Kaggle CLI `/Users/ozgun/miniconda3/bin/kaggle`; SDK `/Users/ozgun/miniconda3/bin/python`; Colab `/Users/ozgun/.local/bin/colab`. Respect actual platform limits (observed2 batchGPU sessions, user45h/week); no assistant-imposed reserves. Current source/artifact metadata pinned in evidence, not inferred from notebook titles.

Latest09:24UTC check: entry57037296 still PENDING; exploratory R2 pair RUNNING. Original384fold0 export slug is not accessible through the API, so launch is unconfirmed. Sol independently reran14 focused persistence tests successfully; the full-suite374 passing/65 skipped result remains Fable-reported. Raw384 export preflight found stale k8/10epoch metadata versus actual k12/12epochs; Fable owns correction, receipt `outputs/analysis/raw384-export-preflight-20261010/review.json`.
