# Project status and session handoff

Updated **2026-10-09 23:38 UTC**. Read this snapshot and queue, then the latest entries in [the Fable/Astra handoff](docs/claude-astra-handoff.md). Durable rules are in [AGENTS.md](AGENTS.md); setup is in [README.md](README.md). Earlier detailed history is preserved in [the status archive](docs/status-archive/status-before-20261009-1830-reconciliation.md).

## Current snapshot

- **Target: top10. Best public score:0.950**, fusion entry56986393. The user's0.962+ waypoint is provisional; the last October9 leaderboard read showed rank10 at0.961. No reproducible top10 recipe is established.
- **Own round0 entry57016778 scored0.878**, COMPLETE, observed October9 at18:43UTC. Five ConvNeXt folds trained on4,345 eligible studies. Image-only qualification and final inference passed exactly; the score does not support the planned blind equal-rank fusion with the0.949 public member. [Own result](docs/experiments/own-model-submission-20261009.md), [revised strategy](docs/experiments/top10-strategy-20261009.md).
- **Round1 ConvNeXt allfive folds COMPLETE and audited.** Fold01/23/4 passed61/65/45 checks; allfive selected checkpoint hashes verified. Pooled raw-report AUC0.882137→0.893651 (+0.011513). Fullfive rehearsal built; launch initially rejected byactual2GPU-sessionlimit. A one-shot launcher owns the next free slot. [Partial audit](docs/experiments/round1-partial-audit-20261009.md). The exact bootstrap label artifact passed134 independent checks. GlobalOOF teacher exposure makes round1 local validation biased; it is not independent clinical CV. No gold-driven label or epoch selection.
- **Last recorded October9 usage: four of five entries.** Recheck current UTC-day allowance before submitting; entries:56986393,56989116,57003785,57016778. Recheck before another entry. Official rules last checked October9 at09:24UTC:5entries/day, offline inference≤9h; entry/merge deadline October15, final deadline October22 at23:59UTC. No new terms accepted.
- Local Git only, no remote. Current work branch `codex/token-efficient-instructions`; parent commits reviewed changes. Main has a separate checkout and has not been claimed up to date with this turn. Preserve the user's unstaged AGENTS changes and Fable's untracked analysis/training/Colab files.

## Ready queue and ownership

| Owner / task | State | Concrete next action |
| --- | --- | --- |
| Parent — fixed round1 fit | Allfive COMPLETE; full artifact/metric audit and weight hashes PASS | Retain exact fixed comparison; no recursive bootstrap expansion based on biased validation. |
| Parent — round1 inference | Fullfive built; launcher20983 waiting for actual free GPU slot | Collect exact rehearsal PASS, build/run final-visible inference, require exact CSV parity, check daily allowance and submit the fixed standalone comparison. |
| Fable — CoAtNet224 | Revisedfold4v2 and01v3 RUNNING; fold23 held behind qualification | Failure logs preserved. Replacement native224 package usesbs4/accum2, gradient checkpointing and windowchunk8. Require actual finite forward/backward/checkpoint evidence; changed recipe is not yet qualified. |
| Fable — Colab CoAtNet384 | Attempt2 `knee-coat384b` RUNNING; epoch4 logged, localepoch3 weights verified | Attempt1 preserved as lost. Backup loads with486finite tensors but has nooptimizerstate; exact resume remains unsupported. Fable owns driver/snapshot loop. [Incident](docs/experiments/colab-session-loss-20261009.md). |
| Parent/Fable — next scientific hypothesis | Static compatibility review complete; training smoke pending | Public384-named CoAtNet already qualified with explicit `img_size=320`; current training omits this argument. Its loader transfers only the backbone, discarding the trained head. Require explicit geometry and strict full-model loading for a full warm start. See [review](docs/experiments/coatnet-warmstart-review-20261009.md). No new fit launched. |
| Parent/Fable — B1 archival integration | Deferred reporting/cache corrections | Benchmark remains descriptive. “Not flagged” does not prove gold-independent provenance; nongold equality cannot establish gold equality. Bind or regenerate cached parquet-to-CSV conversions. Current pinned training artifacts are unaffected. |

## Active jobs and recovery

Exact live-state receipt: `outputs/analysis/reconcile-20261009-2330/receipt.json`. At23:33UTC round1fold4 COMPLETE; CoAtNet4v2/01v3 RUNNING. Rehearsal push rejected by actual capacity; parent launcher now waits for the next free slot. No entry submitted.

| Job | State / evidence |
| --- | --- |
| `ozgung/knee-tracka-convnext-r1-fold01-20261009` v1 | COMPLETE. Small artifacts under `outputs/kaggle/trackA-convnext-r1-fold01-20261009-result/trackA/`; exact metadata and61-check independent audit PASS. Selected SWA epochs0:[9,8,7],1:[8,9,7]. |
| `ozgung/knee-tracka-convnext-r1-fold23-20261009` v1 | COMPLETE. Artifact/metrics65-check audit PASS; both selected checkpoints SHA-verified. Result `outputs/kaggle/trackA-convnext-r1-fold23-20261009-result/trackA/`. |
| `ozgung/knee-tracka-convnext-r1-fold4-20261009` v1 | COMPLETE.45-check artifact/metrics audit PASS; selectedSWA[9,8,7] bytes verified. Fullfive staged in `outputs/kaggle/own-convnext-r1-fivefold-inputs-20261009/`. |
| `ozgung/knee-tracka-coatnet224-fold01-20261009` v3 / `…-fold23-20261009` | Memory-revised01v3 RUNNING; corrected23 not launched. Prior geometry/OOM failures preserved separately, including `outputs/kaggle/trackA-coatnet224-fold01-20261009-failed-v2/`. |
| `ozgung/knee-tracka-coatnet224-fold4-20261009` v2 | RUNNING memory-revised replacement; queue log confirms successful version2 push. Actual selected-checkpoint result pending. Prior failed settings remain distinct. |

Parent stopped queuePID**6782** after verifying its identity, to prioritize the now-ready qualification over the remaining long fit. Both active training kernels are untouched. **Do not restart `queue_push2.sh`: its original list would duplicate jobs.** The only remaining long fit is correctedCoAtNetfold23; release it after rehearsal and final-visible checks. Fable's watchPID**75269** remains alive; original queue/watch logs remain in the previously recorded `/private/tmp/claude-501/-Users-ozgun-Dev-rsna-knee-abnormality-detection/5710ec33-d376-4baa-809b-95617f931beb/scratchpad/`.

One-shot launcherPID**20983**, managedexec session**88904**, command `python3.12 outputs/analysis/reconcile-20261009-2330/wait_rehearsal_slot.py`, is **WAITING_FOR_PLATFORM_SLOT**. It verifies the immutable notebook hash, checks both active fits, and pushes only `ozgung/knee-own-convnext-r1-rehearsal-fivefold-20261009`; it cannot submit or launch training. It checks every300s, exits after accepted/existing job, retries only explicitcapacityrejection, and stops on unknownpushoutcome or uncertainstate. Files `launch-state.json`, `launch.log`, `launch.pid`, `queued-push-responses.txt` are beside it. Before any restart inspect these plus remotejobstate and process; never launch a duplicate. Stop only its verifiedPID if required. Initial detachedstartup exited and a subsequent startup rejected an unrecognized404; both are preserved in the log, exactmissing-job404 handling corrected before the current waiter. No push was accepted in those attempts.

Retrieve completed training artifacts with `/Users/ozgun/miniconda3/bin/kaggle kernels output <kernel> -p <new-result-dir> --file-pattern <small-file-pattern>`; retain failed runs. Expected remote paths are `trackA/run_foldN.json`, `oof_foldN.csv`, logs and selected SWA checkpoints. No running logs are normally visible. Verify checkpoint bytes against run records before packaging.

No parent downloader or GPU batch remains active yet. Allfive checkpoint transport completed; receipts under `outputs/analysis/reconcile-20261009-{2030,2130,2330}/`. Fullfive immutable rehearsal `outputs/kaggle/own-convnext-r1-rehearsal-fivefold-20261009/`, built from6b289e5: notebookSHA `1f5be813a5b19f767cb13ed12c519c54e81adb6e69298e05553d79406fc8fdba`, contractSHA `256b8623d0820c66cca19ea687e70ed428443f4e6bcd26049d2fd8d5ba618630`. Prior twofold build stays archived and was never launched. Fullfive has not yet passed CUDA qualification. Existing heartbeat continues; no duplicate automation.

## Frozen evidence and working decisions

- Dataset:4,407 studies,58 gold and4 related nongold studies excluded from new training, leaving**4,345**. User released the former600reserved studies for new group-disjoint CV. Historical TRAIN04 splits and archived comparisons remain immutable. Known report/patient components do not cross the fixed public folds after exclusions.
- Round0: ConvNeXt-T256,44-slice corpus, k8 training/k16 evaluation,10epochs/fold, generic timm initialization, top3SWA selected by full heldout report-mask AUC. Export AUCs:0.880946/0.882341/0.881449/0.878534/0.889545; these did not predict a0.95-class clinical model. Allfive checkpoint hashes now verified.
- Round1 accepted label file: `labels_bootstrap_v1.parquet`, SHA `b260412732573fa9601907501b753f73d7d4725c3d0942cb463428510c1d48f4`. Exact4,345 coverage, original teacher OOF consistency, source/fold/selection bindings and all52,140 fixed0.5/0.8 blend cells passed. Artifact-specific acceptance does not approve blanket `allow_partial` for future tables. [Audit](docs/experiments/bootstrap-artifact-audit-20261009.md).
- Round1 inference metadata now binds the exact kernel family, label filename/hash, ten full epochs, fixed folds/exclusions and gold-free selection. Legacy round0 archives still validate. Scientific reader/model closure did not change. [Implementation](docs/experiments/own-model-submission-20261009.md).
- DINO is deferred: fullfold report AUCs0.842/0.811/0.838/0.731/0.834. Fixed equal-rank addition to ConvNeXt on folds0/1 reduced pooled AUC by0.01341 (95% group-bootstrap CI−0.01705 to−0.00988). No weight search or gold selection. [Analysis](docs/experiments/dino-complementarity-20261009.md).
- Blind round0/public fusion is deferred after0.878. Preserve the0.950 best. Finish the already-running fixed round1 comparison; prioritize initialization, architecture and supervision hypotheses using controlled changes, not longer training by assumption.
- TRAIN04 retired after12epochs: selected weakAUC0.52359, BCE0.59456 versus prior0.59642. MaViT entries56989116/57003785 failed hidden reruns despite bounded diagnostics; no more retries without a concrete new cause. Detailed evidence remains archived and linked below.
- Existing public MRI/OAI checkpoint use was reviewed under current competition section2.6 and its research/competition permission. No OAI data acquisition or new terms acceptance. Public all-data weights cannot establish clean heldout teacher evidence. [Provenance/limits](docs/experiments/mirror-candidate-20261009.md).

## Qualification and resource notes

Round0 fullfive rehearsal `ozgung/knee-own-convnext-rehearsal-fivefold-20261009` v1 PASS70.491s: all25 pixel/mask slots, windows, probability and repeat comparisons exact;16reader stresses passed. Final `ozgung/knee-own-convnext-submission-20261009` v1 PASS38.9999s, peakGPU0.764GB/host1.573GB, exact CSV SHA `f442a47616691a98392dd0b33f45b6fc9255ff6efb8d4a683109ab9eda9dcd22`. Submitted entry57016778 scored0.878; hidden runtime was not exposed. [Rehearsal](docs/experiments/own-model-rehearsal-20261009.md), [entry](docs/experiments/own-model-submission-20261009.md).

ConvNeXt23 downloads were recovered with4MiB verified HTTP ranges and exact final hashes. Receipts and reusable transport script: `outputs/analysis/reconcile-20261009-1721/weight-range-transport.json` and `range_weights.py`. Recovery is complete; do not restart it. Allfive local weights/run records are linked under ignored `outputs/kaggle/own-convnext-fivefold-inputs-20261009/`.

User removed self-imposed time budgets/GPU reserves; respect actual account limits. Kaggle account45GPUh/week; actual two-session batchGPUlimit rejected the short rehearsal. This is platform capacity, not an assistant reserve. Colab existing~350units authorized; no purchase.

Attempt1 `knee-coat384` lost its assignment afterepoch3; local wait stopped, failure/logs preserved in `outputs/analysis/reconcile-20261009-2230/` and `outputs/colab/coat384-fold0-attempt1-preempted.log`. Cause remains unknown; no originalcheckpoint recovered. Fable started attempt2 **`knee-coat384b`**, driverPID**14464**, same14epoch/384/k12/k_eval24/bs8/bootstrap/gradientcheckpoint recipe, without `--keep`. Command: `scripts/colab_tracka.sh --session knee-coat384b --fold-a 0 --fold-b '' --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --k 12 --k-eval 24 --bs 8 --epochs 14 --labels-file labels_bootstrap_v1.parquet --grad-ckpt`. Log `outputs/colab/coat384-fold0.log`, results `outputs/colab/trackA-knee-coat384b/`, remote `/content/trackA/`. Fable owns monitoring/recovery; reconcile active assignments before any restart.

BackupPID**14667** runs `scratchpad/colab_snapshot.sh` every15min into `outputs/colab/trackA-knee-coat384b-partial/`. Parent safely loaded an epoch3 copy:292826199bytes,486finite modeltensors, expectedCoAtNet384, **no optimizerstate**. Receipt `outputs/analysis/reconcile-20261009-2330/colab-partial-checkpoint.json`; this is recoverable best-model weights, not exact resumable trainingstate. Epoch4 logged in attempt2; runtime-loss detection/full-state persistence remain unfinished. Parent did not relaunchColab or transfercredentials. Local disk last checked17GiB before recentcheckpoints; no full raw-corpus download.

ML interpreter `.venv/bin/python`; Kaggle CLI `/Users/ozgun/miniconda3/bin/kaggle`; Kaggle SDK Python `/Users/ozgun/miniconda3/bin/python`. Dependency-free commands and required validation are in AGENTS. Latest parent checks:11submission tests +5rehearsal tests PASS (no skips, actual public-source paths and isolatedOpenCV supplied), synthetic smoke PASS (`torch_forward=false`), actual legacy round0 archive/report validation PASS. No round1 CUDA result claimed.

## Latest checkpoint

This turn: finalfold audit45checksPASS, allfive selected weights verified, fullfive rehearsal built. Actual platformcapacity rejected launch; one-shot waiter20983 now owns nextfree slot, oldlongfitqueue6782 stopped safely. Metric discrepancy reconciled: raw-reporttargets0.882137→0.893651 versus blendedtargets0.867392→0.879288, same masks. Colabattempt2/backupepoch3 verified; no fulloptimizerresume. No new entry or prediction/runtime code change. Next: collect rehearsal once launcherpushes, qualify final-visible, submit fixedround1 afterchecking allowance, then release correctedCoAtNet23. Best0.950 unchanged.

## Reference records

- [Shared plan and coworker log](docs/claude-astra-handoff.md)
- [Top10 strategy](docs/experiments/top10-strategy-20261009.md)
- [Public0.950 fusion](docs/experiments/versia-fusion-20261009.md), [public0.949 mirror](docs/experiments/mirror-submission-20261009.md)
- [Illustrated0.936 notebook](docs/experiments/0936-explainer-20261008.md)
- [TRAIN04 retirement](docs/experiments/train04-diagnostic-20261009.md), [MaViT failure](docs/experiments/mavit-hidden-diagnostic-20261009.md)
- [Earlier complete status and recovery history](docs/status-archive/status-before-20261009-1830-reconciliation.md)
