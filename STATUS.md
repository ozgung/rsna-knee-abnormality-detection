# Project status and session handoff

Updated **2026-10-10 02:46 UTC**. Read this snapshot and queue, then the latest entries in [the Fable/Astra handoff](docs/claude-astra-handoff.md). Durable rules are in [AGENTS.md](AGENTS.md); setup is in [README.md](README.md). Earlier detailed history is preserved in [the status archive](docs/status-archive/status-before-20261009-1830-reconciliation.md).

## Current snapshot

- **Target: top10. Best public score:0.950**, fusion entry56986393. The user's0.962+ waypoint is provisional; the last October9 leaderboard read showed rank10 at0.961. No reproducible top10 recipe is established.
- **Own round0 entry57016778 scored0.878**, COMPLETE, observed October9 at18:43UTC. Five ConvNeXt folds trained on4,345 eligible studies. Image-only qualification and final inference passed exactly; the score does not support the planned blind equal-rank fusion with the0.949 public member. [Own result](docs/experiments/own-model-submission-20261009.md), [revised strategy](docs/experiments/top10-strategy-20261009.md).
- **Round1 ConvNeXt allfive folds COMPLETE and audited.** Fold01/23/4 passed61/65/45 checks; allfive selected checkpoint hashes verified. Pooled raw-report AUC0.882137→0.893651 (+0.011513). Fullfive rehearsal and final-visible qualification PASS; entry **57028725 PENDING**, submitted October10 at02:45:28UTC. [Qualification/entry](docs/experiments/own-model-r1-submission-20261010.md). [Partial audit](docs/experiments/round1-partial-audit-20261009.md). The exact bootstrap label artifact passed134 independent checks. GlobalOOF teacher exposure makes round1 local validation biased; it is not independent clinical CV. No gold-driven label or epoch selection.
- **October10 usage: one of five entries; four remaining**, confirmed by accepted submission57028725. Recheck before another entry. Official rules last checked October9 at09:24UTC:5entries/day, offline inference≤9h; entry/merge deadline October15, final deadline October22 at23:59UTC. No new terms accepted.
- Local Git only, no remote. Current work branch `codex/token-efficient-instructions`; parent commits reviewed changes. Main has a separate checkout and has not been claimed up to date with this turn. Preserve the user's unstaged AGENTS changes and Fable's untracked analysis/training/Colab files.

## Ready queue and ownership

| Owner / task | State | Concrete next action |
| --- | --- | --- |
| Parent — fixed round1 fit | Allfive COMPLETE; full artifact/metric audit and weight hashes PASS | Retain exact fixed comparison; no recursive bootstrap expansion based on biased validation. |
| Parent — round1 inference | Rehearsal/final PASS; entry57028725 PENDING | Inspect hidden score/failure when complete; do not resubmit. Best0.950 preserved. |
| Fable — CoAtNet224 | Fold01v3 COMPLETE and audited; fold4v2 RUNNING; parent released fold23v2 | Finish fixed family; verify checkpoint bytes and build separately pinned224 inference. [Partial audit](docs/experiments/coatnet224-partial-audit-20261010.md). |
| Fable — Colab CoAtNet384 | New run `knee-c384-f0-a1` RUNNING after attempt2 loss | First full resume dataset ready (1.464GB); downloaded-byte/restore verification still pending. Active loop32148; Fable owns persistence identity/retention fixes. [Review](docs/experiments/colab-resume-review-20261010.md). |
| Parent/Fable — next scientific hypothesis | Static compatibility review complete; training smoke pending | Public384-named CoAtNet already qualified with explicit `img_size=320`; current training omits this argument. Its loader transfers only the backbone, discarding the trained head. Require explicit geometry and strict full-model loading for a full warm start. See [review](docs/experiments/coatnet-warmstart-review-20261009.md). No new fit launched. |
| Parent/Fable — B1 archival integration | Deferred reporting/cache corrections | Benchmark remains descriptive. “Not flagged” does not prove gold-independent provenance; nongold equality cannot establish gold equality. Bind or regenerate cached parquet-to-CSV conversions. Current pinned training artifacts are unaffected. |

## Active jobs and recovery

Latest receipt: `outputs/analysis/reconcile-20261010-0245/receipt.json`. Fullfive round1 rehearsal/final v1 COMPLETE and PASS; competition entry57028725 PENDING. CoAtNet01v3 COMPLETE; fold4v2 RUNNING; correctedfold23v2 newly accepted. Colab currentrun loggingepoch7 with fullstate uploadversion8; processing/readiness and restore of the latest version not independently verified.

| Job | State / evidence |
| --- | --- |
| `ozgung/knee-tracka-convnext-r1-fold01-20261009` v1 | COMPLETE. Small artifacts under `outputs/kaggle/trackA-convnext-r1-fold01-20261009-result/trackA/`; exact metadata and61-check independent audit PASS. Selected SWA epochs0:[9,8,7],1:[8,9,7]. |
| `ozgung/knee-tracka-convnext-r1-fold23-20261009` v1 | COMPLETE. Artifact/metrics65-check audit PASS; both selected checkpoints SHA-verified. Result `outputs/kaggle/trackA-convnext-r1-fold23-20261009-result/trackA/`. |
| `ozgung/knee-tracka-convnext-r1-fold4-20261009` v1 | COMPLETE.45-check artifact/metrics audit PASS; selectedSWA[9,8,7] bytes verified. Fullfive staged in `outputs/kaggle/own-convnext-r1-fivefold-inputs-20261009/`. |
| `ozgung/knee-tracka-coatnet224-fold01-20261009` v3 / `…-fold23-20261009` v2 | Fold01 COMPLETE,56 independent checks PASS; selected weights not yet byte-verified. Correctedfold23 v2 pushed02:45UTC; do not push again. Prior geometry/OOM failures preserved separately, including `outputs/kaggle/trackA-coatnet224-fold01-20261009-failed-v2/`. |
| `ozgung/knee-tracka-coatnet224-fold4-20261009` v2 | RUNNING memory-revised replacement; queue log confirms successful version2 push. Actual selected-checkpoint result pending. Prior failed settings remain distinct. |

Parent stopped queuePID**6782** after verifying its identity, to prioritize the now-ready qualification over the remaining long fit. Both active training kernels are untouched. **Do not restart `queue_push2.sh`: its original list would duplicate jobs.** CorrectedCoAtNetfold23 is now released asv2; the old queue remains stopped and must not restart. Fable's watchPID**75269** remains alive; original queue/watch logs remain in the previously recorded `/private/tmp/claude-501/-Users-ozgun-Dev-rsna-knee-abnormality-detection/5710ec33-d376-4baa-809b-95617f931beb/scratchpad/`.

One-shot launcherPID**20983** accepted rehearsalv1 at02:37:40UTC (`launch-state.json`: PUSH_ACCEPTED). Its work is complete; do not restart. Files `launch-state.json`, `launch.log`, `launch.pid`, `queued-push-responses.txt` remain under `outputs/analysis/reconcile-20261009-2330/`. Initial startup/capacity failures are preserved; no duplicate rehearsal or controller was launched. Parent subsequently built and pushed finalv1 once, qualified it, and submitted entry57028725 once.

Retrieve completed training artifacts with `/Users/ozgun/miniconda3/bin/kaggle kernels output <kernel> -p <new-result-dir> --file-pattern <small-file-pattern>`; retain failed runs. Expected remote paths are `trackA/run_foldN.json`, `oof_foldN.csv`, logs and selected SWA checkpoints. No running logs are normally visible. Verify checkpoint bytes against run records before packaging.

No parent downloader remains active. Final notebook completed; hidden competition rerun is pending. Allfive checkpoint transport completed; receipts under `outputs/analysis/reconcile-20261009-{2030,2130,2330}/`. Fullfive immutable rehearsal `outputs/kaggle/own-convnext-r1-rehearsal-fivefold-20261009/`, built from6b289e5: notebookSHA `1f5be813a5b19f767cb13ed12c519c54e81adb6e69298e05553d79406fc8fdba`, contractSHA `256b8623d0820c66cca19ea687e70ed428443f4e6bcd26049d2fd8d5ba618630`. Prior twofold build stays archived and was never launched. Fullfive CUDA qualification PASS62.333s; final-visible PASS55.198s, exact CSV SHA `d062b8a194d38d24d11051a504943c02b0c876f8d393492fc796a91078ca5f38`. Existing heartbeat continues; no duplicate automation.

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

Colab attempts1/2 `knee-coat384` and `knee-coat384b` both lost their assignments; preserve their logs. Attempt2 reached loggedepoch11 but only epoch3 bestweights were recovered. That292826199byte backup safely loaded with486finite tensors and nooptimizerstate; it cannot supply exact resume. Old driver/snapshot loop stopped by Fable.

Fable started **`knee-c384-f0-a1`**, loopPID**32148**, driverPID**32151**, same fold0/14epoch/384/k12/k_eval24/bs8/bootstrap recipe with new per-epoch full-state persistence. Loop script `scratchpad/colab_fold_loop.sh 0` launches up tofive attempts only on rc75, with later attempts passing `--resume-from-dataset ozgung/knee-colab-resume-c384-fold0`; log `outputs/colab/c384-fold0-loop.log`, results `outputs/colab/trackA-knee-c384-f0-a1/`, remote `/content/trackA/`. Do not start the loop again: attempt1 resets to fresh fitting. Reconcile assignment/process ownership before any stop/retry. Parent did not start this loop or transfercredentials.

Dataset **`ozgung/knee-colab-resume-c384-fold0` is ready**, first fullstate1464064119bytes created01:33:08UTC afterepoch0; parent verified status/filelist, **not downloaded-byte integrity or restore**. Active helper publishes at most every8min; Fable reports local exact-continuation tests but parent did not rerun them. Review found missing experiment identity binding, unverified success bookkeeping, delete-old-versions default, missing localresume fallback and retry ownership gaps; [review](docs/experiments/colab-resume-review-20261010.md). Fable owns corrections; activefitleftuntouched. This is not yet a qualified resumed run. No paidcompute or raw-corpus expansion.

ML interpreter `.venv/bin/python`; Kaggle CLI `/Users/ozgun/miniconda3/bin/kaggle`; Kaggle SDK Python `/Users/ozgun/miniconda3/bin/python`. Dependency-free commands and required validation are in AGENTS. Latest parent checks:11submission tests +5rehearsal tests PASS (no skips, actual public-source paths and isolatedOpenCV supplied), synthetic smoke PASS (`torch_forward=false`), actual legacy round0 archive/report validation PASS. Round1 CUDA rehearsal and final-visible passed, with exact CSV parity; hidden competition performance pending. No runtime source changes this turn, so unchanged tests were not rerun.

## Latest checkpoint

This turn: qualified and submitted fixed round1 entry57028725; four entries remain today. Released correctedCoAtNet23v2 after short checks. IndependentSol audit confirms CoAtNet01 reportAUC0.901842/0.902008 and56 checks PASS; do not call selected-fold predictions unbiased teachers. Future recipes should record skippedAMP steps and flush partial gradient accumulation; preserve this ongoing comparison. Colab currentfit remains active and unchanged, latestepoch7 reportAUC0.9115; Fable owns accepted persistence fixes between folds. Next: collect round1 hidden result, finish/audit CoAtNet family, then qualify separately pinned224 image-only inference. Best0.950 unchanged.

## Reference records

- [Shared plan and coworker log](docs/claude-astra-handoff.md)
- [Top10 strategy](docs/experiments/top10-strategy-20261009.md)
- [Public0.950 fusion](docs/experiments/versia-fusion-20261009.md), [public0.949 mirror](docs/experiments/mirror-submission-20261009.md)
- [Illustrated0.936 notebook](docs/experiments/0936-explainer-20261008.md)
- [TRAIN04 retirement](docs/experiments/train04-diagnostic-20261009.md), [MaViT failure](docs/experiments/mavit-hidden-diagnostic-20261009.md)
- [Earlier complete status and recovery history](docs/status-archive/status-before-20261009-1830-reconciliation.md)
