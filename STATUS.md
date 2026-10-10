# Project status and session handoff

Updated **2026-10-10 04:38 UTC**. Read this snapshot and queue, then the latest [Fable/Astra handoff](docs/claude-astra-handoff.md). Rules are in AGENTS.md, setup in README.md. Older details: [October10 archive](docs/status-archive/status-before-20261010-0435-reconciliation.md).

## Current snapshot

- **Target: top10. Best public score: 0.950**, entry56986393. The user's0.962+ waypoint is provisional; last October9 leaderboard read put rank10 at0.961. No reproducible top10 method is established.
- **ConvNeXt bootstrap round1 closed:** entry57028725 completed at **0.874**, versus round0 entry57016778 at0.878. Exact image-only qualification passed for both. Pooled raw-report AUC increased0.882137→0.893651, but that improvement did not transfer to the public metric. No further round1 fits, default fusion, or recursive bootstrap from this result. [Result and evidence](docs/experiments/own-model-r1-submission-20261010.md).
- **Original-label CoAtNet224 is the primary own candidate.** Folds0/1/4 completed at raw-report AUC0.901842/0.902008/0.906643; parent57-check audit and allthree selected checkpoint hashes/metadata/finite tensors passed. Folds2/3 remain RUNNING. Gains support the combined architecture/resolution/training recipe, not backbone causality alone. [Audit](docs/experiments/coatnet224-partial-audit-20261010.md).
- **Colab384 trained14epochs but failed during export and lost its session.** Epoch13 recovery dataset reports ready; actual restore remains unverified. Fable owns export-only recovery and persistence fixes. Colab API at04:33UTC reports **no active sessions**, resolving the previously listed unknown assignment's current presence, not proving past billing or a hard session cap.
- October10 usage: one of five entries, four remaining after57028725; recheck before submission. Official rules last verified October9:5entries/day, offline inference≤9h; entry/merge deadlineOctober15, final deadlineOctober22 at23:59UTC. No new terms or compute purchases.

## Ready queue and ownership

| Owner / task | State | Next action |
| --- | --- | --- |
| Parent — CoAtNet224 qualification | Separate rehearsal implemented; subset0/1/4 built only | Collect and audit23 when complete, verify both weights, build fullfive and run short offline CUDA rehearsal at next free slot. Require exact reader/window/probability/repeat parity and fullfive PASS before final build. |
| Parent — CoAtNet224 final wrapper | Implemented and reviewed;16 combined tests PASS | After real fullfive rehearsal PASS, build and run final-visible inference; require exact CSV parity before entry. [Preparation](docs/experiments/coatnet224-submission-preparation-20261010.md). |
| Fable — existing training | Original-label23v2 and bootstrap-label01v1 RUNNING | Finish fixed comparisons. Bootstrap arm is diagnostic, not a default ensemble member. Preserve next free slot for short224 qualification, then384 export recovery before more long fits. |
| Fable — Colab384 recovery | Training complete, export FAILED, backup ready | Pin and verify epoch13 state; recover selectedSWA and export with fewer workers. Fix inaccurate training provenance in export receipts. No retraining needed merely to recover completed training. |
| Fable — Colab reliability | Accepted changes not parent-verified | Require resume identity/config/source/labels/folds binding, fail on missing explicit resume, verify processed version before success, retain previous verified version, reconcile assignment ownership before retry. |
| Parent/Fable — next experiment | No new bootstrap generation approved | Gold58 remains descriptive, never a trigger for selecting labels/epochs. Propose training-only nested/crossfit evidence or a prospectively fixed comparison before recursive bootstrap. Public warm-start option remains a separate hypothesis. |
| Fable — B1 archival integration | Deferred | Fix benchmark provenance claims and cached CSV hash binding. Current pinned training artifacts unaffected. |

## Live jobs and recovery

Latest read-only receipt: `outputs/analysis/reconcile-20261010-0435/receipt.json`. No parent training launch, submission, controller or downloader this turn.

| Exact job | Verified state |
| --- | --- |
| `ozgung/knee-tracka-coatnet224-fold23-20261009` v2 | RUNNING, pushed by parent October10 at02:45UTC. Fixed original-label224 recipe. Do not duplicate. |
| `ozgung/knee-tracka-coatnet224-r1-fold01-20261010` v1 | RUNNING, pushed by Fable03:08UTC. Same224 recipe with ConvNeXt-teacher `labels_bootstrap_v1`; paired diagnostic only. |
| `ozgung/knee-tracka-coatnet224-fold01-20261009` v3 | COMPLETE; selectedSWA0:[8,9,7],1:[9,8,7]. Earlier geometry/OOM versions preserved. |
| `ozgung/knee-tracka-coatnet224-fold4-20261009` v2 | COMPLETE; selectedSWA[8,9,7]. |
| `ozgung/knee-own-convnext-r1-submission-20261010` v1 / entry57028725 | Notebook PASS, competition COMPLETE0.874. Do not resubmit. |

CoAtNet fixed recipe: native `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`,224px, k8 training/k16 evaluation, generic timm initialization,10epochs, batch4/accumulation2, gradient checkpointing, windowchunk8, original `labels_v1_blend`. Despite earlier coworker wording, these are blended labels; selection evaluates raw report labels. Fullfold report AUC selects top3SWA. Existing logs show scheduler/AMP update uncertainty and an unstepped final accumulation microbatch; preserve current family and fix only in a separately identified future recipe.

Selected weights0/1/4 are292826563bytes each,486finite tensors. Exact hashes and supports: `outputs/analysis/coatnet224-parent-audit-20261010/{audit.json,checkpoint-review.json}`. Real result directories: `outputs/kaggle/trackA-coatnet224-fold{01,4}-20261009-result/trackA/`. Staged symlinks: `outputs/kaggle/own-coatnet224-inputs-20261010/`. Downloader finished; preserve receipts and do not restart it.

Subset preparation build: `outputs/kaggle/own-coatnet224-rehearsal-fold014-20261010/`, notebookSHA `09ae6f54d979b56a8644891605cf4eaa72f0d4fc7c5ef1f747c0acfb96665496`, contractSHA `1ffaf9753272873da56fa5237c07e68edd27adde12db33b33a9faa4e51b73eae`. **Not launched; not CUDA-qualified; not a fullfive entry.** Once23 finishes, build fullfive directly rather than treating this subset as qualification.

Colab lost session: `knee-c384-f0-a1`; logs `outputs/colab/c384-fold0-loop.log`, failure snapshot `outputs/analysis/reconcile-20261010-0335/colab-export-failure.log`. It trained14epochs on bootstrap-v1,384px,k12/k24,batch8, with selectedSWA[13,12,11] and best logged reportAUC0.9184. Export worker was killed, then download returned404/401. OOM is plausible, not established. Loop32148/driver32151 exitedrc1; **do not restart the old loop**, whose first attempt starts fresh.

Recovery dataset: **`ozgung/knee-colab-resume-c384-fold0` version14**, epoch13, about2.05GB including full resume/top-k state. Ready status verified, downloaded payload/restore not yet verified. Fable's `scripts/build_tracka_export_kaggle.py` supports rebuildingSWA from resume and fewer export workers. Current review finds K_TRAIN default8 and EPOCHS default10 do not represent the actual12/14 fit; current export sources and synthetic train_cmd must be separated from original training provenance. The existing parked package may target the old epoch3 backup: **verify exact attached dataset/state before any push.** Earlier two failed Colab attempts and backup remain archived; no blind retraining.

Old queuePID6782 was deliberately stopped; never restart `queue_push2.sh`, which would duplicate jobs. Launcher20983 successfully pushed round1 rehearsal and exited; oldwatch75269 is gone. Original logs remain under `/private/tmp/claude-501/-Users-ozgun-Dev-rsna-knee-abnormality-detection/5710ec33-d376-4baa-809b-95617f931beb/scratchpad/`. No replacement controller was started.

## Scientific and operational decisions

- Fixed eligibility:4,407 studies minus58 gold and4 related quarantined studies = **4,345**. Known patient/report components do not cross fixed folds after exclusions. The user released the historical600reserved studies for new CV; prior comparisons remain immutable.
- Heldout labels select epochs/SWA, so selected-fold OOF is not untouched outer-fold evidence. GlobalOOF bootstrap adds indirect validation exposure. Do not claim clinical generalization from report AUC, or select new label schemes by gold58 performance.
- Existing v1-bootstrap fits may finish as diagnostics; no expansion based on biased report gains. Two public scores do not quantify leaderboard noise or establish a single causal explanation.
- Public MRI/OAI checkpoint permission was reviewed; no new OAI data/terms. A full warm start requires strict full-model loading and explicit geometry. Existing loader transfers backbone only; public384-named checkpoint qualified with explicit320 geometry. Public all-data weights cannot establish cleanOOF teachers. [Compatibility review](docs/experiments/coatnet-warmstart-review-20261009.md).
- DINO deferred after negative complementarity; TRAIN04 retired; MaViT hidden reruns failed twice. Preserve best0.950 and avoid blind fusion/retries. See linked history, not new experiments.
- User removed self-imposed GPU reserves/time budgets. Respect actual account/platform limits. Kaggle45GPUh/week and observed two simultaneous batchGPU sessions. Existing Colab resources authorized; no purchases.

## Verification, Git and next checkpoint

ML interpreter `.venv/bin/python`; Kaggle CLI `/Users/ozgun/miniconda3/bin/kaggle`; SDK Python `/Users/ozgun/miniconda3/bin/python`; Colab CLI `/Users/ozgun/.local/bin/colab`. Use bounded output and retain exact evidence under ignored outputs.

CoAtNet rehearsal commit **9bff86c**:8focused tests PASS, no skips, actual pinned224 window arithmetic and16reader stresses. Synthetic smoke PASS (`torch_forward=false`). Full discovery352tests:291passed,60optional evidence/backend skips, one existing multiprocessing test blocked by sandbox shared-memory manager; exacttest passed with process permission. No actual CoAtNet architecture forward/CUDA qualification yet. Failure handoff commit908bbae.

Local Git only, no remote; branch `codex/token-efficient-instructions`. Parent commits reviewed changes. Preserve user AGENTS edits and Fable's untracked scripts. Main has a separate checkout and is not claimed current. Worktree `/Users/ozgun/.codex/worktrees/coatnet224-rehearsal/rsna-knee-abnormality-detection` contains the integrated rehearsal/final-wrapper source; app refused earlier archive because a pinned task/workspace protects it. No force removal.

Final-wrapper code and tests are ready; no real final build or CUDA qualification is claimed. Parent16 combined tests passed with no skips; synthetic smoke passed and existing224 rehearsal source hashes are unchanged. Next meaningful checkpoint: original-label23 completion/fullfive qualification or384 recovery result. Round1 branch is closed. Update this handoff on changed evidence; avoid unchanged polling and duplicate jobs.

## References

- [Shared plan](docs/claude-astra-handoff.md), [top10 strategy](docs/experiments/top10-strategy-20261009.md)
- [CoAtNet224 audit and rehearsal](docs/experiments/coatnet224-partial-audit-20261010.md)
- [Round1 public result](docs/experiments/own-model-r1-submission-20261010.md), [round0 result](docs/experiments/own-model-submission-20261009.md)
- [Colab persistence review](docs/experiments/colab-resume-review-20261010.md)
- [Public0.950 fusion](docs/experiments/versia-fusion-20261009.md), [illustrated0.936 notebook](docs/experiments/0936-explainer-20261008.md)
