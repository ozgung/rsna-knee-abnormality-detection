# Project status and session handoff

Updated **2026-10-10 07:35 UTC**. Read this snapshot and queue, then the latest [Fable/Astra handoff](docs/claude-astra-handoff.md). Rules are in AGENTS.md, setup in README.md. Older details: [October10 archive](docs/status-archive/status-before-20261010-0435-reconciliation.md).

## Current snapshot

- **Target: top10. Best public score: 0.950**, entry56986393. The user's0.962+ waypoint is provisional; last October9 leaderboard read put rank10 at0.961. No reproducible top10 method is established.
- **ConvNeXt bootstrap round1 closed:** entry57028725 completed at **0.874**, versus round0 entry57016778 at0.878. Exact image-only qualification passed for both. Pooled raw-report AUC increased0.882137→0.893651, but that improvement did not transfer to the public metric. No further round1 fits, default fusion, or recursive bootstrap from this result. [Result and evidence](docs/experiments/own-model-r1-submission-20261010.md).
- **Original-label CoAtNet224 is the primary own candidate.** Folds0/1/4 completed at raw-report AUC0.901842/0.902008/0.906643; parent57-check audit and allthree selected checkpoint hashes/metadata/finite tensors passed. Folds2/3 have logged all ten epochs and remain RUNNING for final artifacts. Gains support the combined architecture/resolution/training recipe, not backbone causality alone. [Audit](docs/experiments/coatnet224-partial-audit-20261010.md).
- **Bootstrap384 recovery export COMPLETE.** Raw-report macro AUC0.918185 on870studies; parent28-check artifact/metric audit PASS. All486 exported tensors exactly match local recovery;10 tensor checks PASS. Optimizer continuation remains unverified. [Export audit](docs/experiments/coat384-recovery-export-20261010.md). [Recovery](docs/experiments/colab384-artifact-recovery-20261010.md). Fable also launched original-label384 fold0 on Colab A100,12epochs,k12/24,batch8, with in-VM export disabled. Training reached epoch4/reportAUC0.8898; its epoch4 backup shows Successful in Kaggle. Persistence fixes remain open; no verified2h20session cap is established.
- October10 usage: one of five entries, four remaining after57028725; recheck before submission. Official rules last verified October9:5entries/day, offline inference≤9h; entry/merge deadlineOctober15, final deadlineOctober22 at23:59UTC. No new terms or compute purchases.

## Ready queue and ownership

| Owner / task | State | Next action |
| --- | --- | --- |
| Parent — CoAtNet224 qualification | Subset0/1/4 rehearsal v1 PASS; full-five pending | Collect and audit23 when complete, verify both weights, build fullfive and run short offline CUDA rehearsal at next free slot. Require exact reader/window/probability/repeat parity and fullfive PASS before final build. |
| Parent — CoAtNet224 final wrapper | Implemented and reviewed;16 combined tests PASS | After real fullfive rehearsal PASS, build and run final-visible inference; require exact CSV parity before entry. [Preparation](docs/experiments/coatnet224-submission-preparation-20261010.md). |
| Fable — existing training | Original-label23v2 RUNNING; bootstrap-label01v1 COMPLETE; original-label384fold0 ACTIVE on Colab | Completed paired audit:69 checks PASS; pooled report AUC+0.003114, mixed folds, no bootstrap expansion. Preserve next free Kaggle slot for224 qualification; subset224 rehearsal completed; keep released slot available for full-five qualification. |
| Fable — Colab384 export | Export COMPLETE;28 parent checks PASS | Cross-check reconstructed tensors against parent fingerprint. Submitted export metadata still says k8/10epochs instead of actualk12/14; preserve receipt and append explicit provenance correction. |
| Fable — Colab reliability | Accepted changes not parent-verified | Require resume identity/config/source/labels/folds binding, fail on missing explicit resume, verify processed version before success, retain previous verified version, reconcile assignment ownership before retry. |
| Parent/Fable — next experiment | No new bootstrap generation approved | Gold58 remains descriptive, never a trigger for selecting labels/epochs. Propose training-only nested/crossfit evidence or a prospectively fixed comparison before recursive bootstrap. Public warm-start option remains a separate hypothesis. |
| Fable — B1 archival integration | Deferred | Fix benchmark provenance claims and cached CSV hash binding. Current pinned training artifacts unaffected. |

## Live jobs and recovery

Latest receipt: `outputs/analysis/reconcile-20261010-0735/receipt.json`. Parent canceled Fable's accidental seed42 duplicate `ozgung/knee-tracka-coatnet224-s43-fold01-20261010` v1 (UI confirmed555.4s; notebook/logs preserved). The requested seed43 patch had not applied. Parent successfully ran subset224 rehearsal v1 in that slot: PASS,101.99s measured runtime. Original23 continues; no duplicate controller. Reserve subsequent capacity for full-five qualification before more exploratory fits.

| Exact job | Verified state |
| --- | --- |
| `ozgung/knee-tracka-coatnet224-fold23-20261009` v2 | RUNNING, pushed by parent October10 at02:45UTC. Fixed original-label224 recipe. Do not duplicate. |
| `ozgung/knee-tracka-coatnet224-r1-fold01-20261010` v1 | COMPLETE. Parent69-check audit PASS: reportAUC0.899828/0.909633, pooled+0.003114; weight bytes not audited. Diagnostic only. |
| `ozgung/knee-tracka-export-c384-fold0-20261010` v1 | COMPLETE. Input resume hash matches local epoch13 recovery; reportAUC0.918185. Exported486 tensors exactly match local recovered model; no transfer active. |
| `ozgung/knee-own-coatnet224-rehearsal-fold014-20261010` v1 | COMPLETE/PASS. Parent verified contract, exact probabilities,16 stress cases and authoritative visible CSV. Three-fold wiring check only. |
| `ozgung/knee-tracka-coatnet224-s43-fold01-20261010` v1 | CANCELED by parent: actualseed42 duplicate. Do not restart; seed43 must be demonstrated in resolved training command before any future run. |
| `ozgung/knee-tracka-coatnet224-fold01-20261009` v3 | COMPLETE; selectedSWA0:[8,9,7],1:[9,8,7]. Earlier geometry/OOM versions preserved. |
| `ozgung/knee-tracka-coatnet224-fold4-20261009` v2 | COMPLETE; selectedSWA[8,9,7]. |
| `ozgung/knee-own-convnext-r1-submission-20261010` v1 / entry57028725 | Notebook PASS, competition COMPLETE0.874. Do not resubmit. |

CoAtNet fixed recipe: native `coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k`,224px, k8 training/k16 evaluation, generic timm initialization,10epochs, batch4/accumulation2, gradient checkpointing, windowchunk8, original `labels_v1_blend`. Despite earlier coworker wording, these are blended labels; selection evaluates raw report labels. Fullfold report AUC selects top3SWA. Existing logs show scheduler/AMP update uncertainty and an unstepped final accumulation microbatch; preserve current family and fix only in a separately identified future recipe.

Selected weights0/1/4 are292826563bytes each,486finite tensors. Exact hashes and supports: `outputs/analysis/coatnet224-parent-audit-20261010/{audit.json,checkpoint-review.json}`. Real result directories: `outputs/kaggle/trackA-coatnet224-fold{01,4}-20261009-result/trackA/`. Staged symlinks: `outputs/kaggle/own-coatnet224-inputs-20261010/`. Downloader finished; preserve receipts and do not restart it.

Subset preparation build: `outputs/kaggle/own-coatnet224-rehearsal-fold014-20261010/`, notebookSHA `09ae6f54d979b56a8644891605cf4eaa72f0d4fc7c5ef1f747c0acfb96665496`, contractSHA `1ffaf9753272873da56fa5237c07e68edd27adde12db33b33a9faa4e51b73eae`. **Version1 COMPLETE/PASS after the earlier capacity rejection. Three-fold CUDA wiring verified; not a full-five entry.** Launch command: `/Users/ozgun/miniconda3/bin/kaggle kernels push -p outputs/kaggle/own-coatnet224-rehearsal-fold014-20261010`. Collected and verified `coatnet224_rehearsal/rehearsal-report.json` and `visible.csv`; require full-five rehearsal after23 artifacts are audited. [Measured subset result](docs/experiments/coatnet224-subset-rehearsal-20261010.md). Prepared audit: `outputs/analysis/coatnet224-fullfive-audit-20261010/audit.py`, not yet run while23 artifacts are pending. No local controller; stop only through this exact notebook’s Active Events control if a diagnosed fault requires it.

Colab lost session: `knee-c384-f0-a1`; logs `outputs/colab/c384-fold0-loop.log`, failure snapshot `outputs/analysis/reconcile-20261010-0335/colab-export-failure.log`. It trained14epochs on bootstrap-v1,384px,k12/k24,batch8, with selectedSWA[13,12,11] and best logged reportAUC0.9184. Export worker was killed, then download returned404/401. OOM is plausible, not established. Loop32148/driver32151 exitedrc1; **do not restart the old loop**, whose first attempt starts fresh.

Recovery dataset: **`ozgung/knee-colab-resume-c384-fold0` version14**, epoch13, about2.05GB including full resume/top-k state. Parent now downloaded exactserverlistedsize and safely loaded fullstate;14epochhistory/top3/modeltensor checks PASS. SelectedSWA recovered locally under `outputs/colab/coat384-fold0-recovery-20261010/`; exactoptimizercontinuation andmodelinference remain unverified. Detailed hashes/limits in recovery-review.json and linkedrecoverydocument. Fable's `scripts/build_tracka_export_kaggle.py` supports rebuildingSWA from resume and fewer export workers. Current review finds K_TRAIN default8 and EPOCHS default10 do not represent the actual12/14 fit; current export sources and synthetic train_cmd must be separated from original training provenance. The active export uses the correct resume dataset; verify exact version14/epoch13 state and selected tensor hash when results arrive. The older parked package is not the active export. Earlier two failed Colab attempts and backup remain archived; no blind retraining.

New original-label Colab run: session **`knee-c384r-f0-a1`**, assignment **`gpu-a100-s-kkb-usc1f1-2ss4kbfvl4vz0`**, A100. Controller63024 runs `colab_raw_folds.sh 0 1 2 3 4` from the existing Claude scratchpad; driver63057 runs `scripts/colab_tracka.sh --session knee-c384r-f0-a1 --fold-a 0 --fold-b "" --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --k 12 --k-eval 24 --bs 8 --epochs 12 --labels-file labels_v1_blend.parquet --grad-ckpt --skip-export --persist-dataset ozgung/knee-colab-resume-c384r-fold0`. Log `outputs/colab/c384raw-loop.log`; per-session logs `outputs/colab/trackA-knee-c384r-f0-a1/`. Remote states `/content/trackA/raptor_resume_fold0.pt` and private dataset above. Do not start another loop. Resume only after verifying compatible ready state and old assignment resolution; use a new session and `--resume-from-dataset`. To stop deliberately, verify identities then stop controller/driver and `colab stop -s knee-c384r-f0-a1`. Current persistence still deletes old versions by default and treats CLI acknowledgement as ready; Fable owns fixes. Loop retries arbitrary rc1 and can advance after four failed attempts: correction requested before next retry/fold. Healthy current training retained.

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

Final-wrapper code and tests are ready; subset CUDA rehearsal passed, but no full-five qualification or final build is claimed. Parent16 combined tests passed with no skips; synthetic smoke passed and existing224 rehearsal source hashes are unchanged. Completed label-arm audit:69 checks PASS, original-label candidate priority unchanged. [Paired results](docs/experiments/coatnet224-label-arm-20261010.md). Next meaningful checkpoint: original-label23 completion/fullfive qualification or384 export result. Parent recovered384SWA without retraining;506 artifactchecks include486 per-tensorchecks, not model evaluations. Round1 branch is closed. Update this handoff on changed evidence; avoid unchanged polling and duplicate jobs.

## References

- [Shared plan](docs/claude-astra-handoff.md), [top10 strategy](docs/experiments/top10-strategy-20261009.md)
- [CoAtNet224 audit and rehearsal](docs/experiments/coatnet224-partial-audit-20261010.md)
- [Round1 public result](docs/experiments/own-model-r1-submission-20261010.md), [round0 result](docs/experiments/own-model-submission-20261009.md)
- [Colab persistence review](docs/experiments/colab-resume-review-20261010.md)
- [Public0.950 fusion](docs/experiments/versia-fusion-20261009.md), [illustrated0.936 notebook](docs/experiments/0936-explainer-20261008.md)
