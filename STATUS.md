# Project status and session handoff

Updated **2026-10-09 18:52 UTC**. Read this snapshot and queue, then the latest entries in [the Fable/Astra handoff](docs/claude-astra-handoff.md). Durable rules are in [AGENTS.md](AGENTS.md); setup is in [README.md](README.md). Earlier detailed history is preserved in [the status archive](docs/status-archive/status-before-20261009-1830-reconciliation.md).

## Current snapshot

- **Target: top10. Best public score:0.950**, fusion entry56986393. The user's0.962+ waypoint is provisional; the last October9 leaderboard read showed rank10 at0.961. No reproducible top10 recipe is established.
- **Own round0 entry57016778 scored0.878**, COMPLETE, observed October9 at18:43UTC. Five ConvNeXt folds trained on4,345 eligible studies. Image-only qualification and final inference passed exactly; the score does not support the planned blind equal-rank fusion with the0.949 public member. [Own result](docs/experiments/own-model-submission-20261009.md), [revised strategy](docs/experiments/top10-strategy-20261009.md).
- **Round1 ConvNeXt folds0/1 and2/3 are RUNNING** under Fable's existing queue. Fold4 is next. The exact bootstrap label artifact passed134 independent checks. GlobalOOF teacher exposure makes round1 local validation biased; it is not independent clinical CV. No gold-driven label or epoch selection.
- **One October9 submission remains.** Entries today:56986393,56989116,57003785,57016778. Recheck before another entry. Official rules last checked October9 at09:24UTC:5entries/day, offline inference≤9h; entry/merge deadline October15, final deadline October22 at23:59UTC. No new terms accepted.
- Local Git only, no remote. Current work branch `codex/token-efficient-instructions`; parent commits reviewed changes. Main has a separate checkout and has not been claimed up to date with this turn. Preserve the user's unstaged AGENTS changes and Fable's untracked analysis/training/Colab files.

## Ready queue and ownership

| Owner / task | State | Concrete next action |
| --- | --- | --- |
| Fable — fixed round1 fit | Folds01/23 RUNNING;4 queued | Collect terminal runJSON/OOF/logs; check exact label hash, selected checkpoints, per-target support and failures. Keep the predeclared recipe fixed; no recursive bootstrap expansion. |
| Parent — round1 inference | Build support implemented/tested at7c1dd41; no round1 GPU qualification | Once verified checkpoints exist, build with `--training-round 1`; rehearse allfive, run final visible inference and require exact parity before considering the remaining entry. |
| Fable — CoAtNet224 | Old fold01/23 v1 ERROR; native224 replacement queued | Both failures are196-vs576 attention geometry mismatches from native384 construction with224 inputs. Preserve failed versions; require actual forward/backward smoke of the corrected architecture before full fits. |
| Parent — next scientific hypothesis | Preparation only | Review a controlled warm start from the stronger public MRI checkpoint. Verify strict state compatibility and preprocessing first. Such all-data/OAI initialization is a separate competition member, never a clean OOF teacher. No new fit launched. |
| Parent/Fable — B1 archival integration | Deferred reporting/cache corrections | Benchmark remains descriptive. “Not flagged” does not prove gold-independent provenance; nongold equality cannot establish gold equality. Bind or regenerate cached parquet-to-CSV conversions. Current pinned training artifacts are unaffected. |

## Active jobs and recovery

Exact live-state receipt: `outputs/analysis/reconcile-20261009-1830/receipt.json`. API reads during18:43–18:51UTC; no duplicate run launched this turn.

| Job | State / evidence |
| --- | --- |
| `ozgung/knee-tracka-convnext-r1-fold01-20261009` v1 | RUNNING; pushed17:28UTC. Package `outputs/kaggle/trackA-convnext-r1-fold01-20261009/`. |
| `ozgung/knee-tracka-convnext-r1-fold23-20261009` v1 | RUNNING; pushed17:55:36UTC after two recorded account-capacity rejections. Package `outputs/kaggle/trackA-convnext-r1-fold23-20261009/`. |
| `ozgung/knee-tracka-convnext-r1-fold4-20261009` | Inaccessible through API; queued next by the existing controller. Do not infer a completed or running job. |
| `ozgung/knee-tracka-coatnet224-fold01-20261009` / `…-fold23-20261009` | Old v1 ERROR. Fold01 logs preserved under `outputs/kaggle/trackA-coatnet224-fold01-20261009-failed-v1/`; fold23 diagnosis under `outputs/analysis/reconcile-20261009-1721/`. Corrected packages are not yet qualified. |
| `ozgung/knee-tracka-coatnet224-fold4-20261009` | Inaccessible; queued. |

Fable owns queuePID**75657** and watchPID**75269**, both confirmed alive. Scripts and logs are under `/private/tmp/claude-501/-Users-ozgun-Dev-rsna-knee-abnormality-detection/5710ec33-d376-4baa-809b-95617f931beb/scratchpad/`: `queue_push2.sh`, `queue_push.log`, `watch_all.sh`, `watch_all.log`. Latest observed queue log18:47:34UTC reported two occupied sessions and round1fold4 next. Queue advances only after a successful push. Coordinate changes through the shared log; do not start another controller. For an intentional stop, coordinate with Fable and terminate only the recorded controller PID; preserve logs. A restart must reconcile remote versions and the remaining queue first.

Retrieve completed training artifacts with `/Users/ozgun/miniconda3/bin/kaggle kernels output <kernel> -p <new-result-dir> --file-pattern <small-file-pattern>`; retain failed runs. Expected remote paths are `trackA/run_foldN.json`, `oof_foldN.csv`, logs and selected SWA checkpoints. No running logs are normally visible. Verify checkpoint bytes against run records before packaging.

No parent GPU batch, downloader or implementation worker remains active. The first delegated status reconciliation returned no result and was interrupted; parent completed the API checks directly. The round1 implementation worker completed its bounded task. Existing heartbeat remains the campaign follow-up; no duplicate automation.

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

User removed self-imposed time budgets/GPU reserves; respect actual account limits. Kaggle account is45GPUh/week and two concurrent batchGPU sessions were observed; recheck current quota when scheduling. Colab A100 entitlement was confirmed and probes stopped; existing~350units authorized, no purchase. Fable owns its pending credential/session workflow; no current parent Colab/MPS training. Last local disk check18GiB free before two112MBcheckpoint downloads. Do not download the full raw corpus locally.

ML interpreter `.venv/bin/python`; Kaggle CLI `/Users/ozgun/miniconda3/bin/kaggle`; Kaggle SDK Python `/Users/ozgun/miniconda3/bin/python`. Dependency-free commands and required validation are in AGENTS. Latest parent checks:11submission tests +5rehearsal tests PASS (no skips, actual public-source paths and isolatedOpenCV supplied), synthetic smoke PASS (`torch_forward=false`), actual legacy round0 archive/report validation PASS. No round1 CUDA result claimed.

## Latest checkpoint

This turn: verified0.878 score, revised strategy, preserved CoAtNet01 failure logs, reconciled live controllers/jobs, and integrated round1 packaging at7c1dd41. No new training run or entry. Next: inspect terminal round1 results and selected checkpoint evidence; qualify the exact family before another entry. One dailyentry remains, best0.950 unchanged.

## Reference records

- [Shared plan and coworker log](docs/claude-astra-handoff.md)
- [Top10 strategy](docs/experiments/top10-strategy-20261009.md)
- [Public0.950 fusion](docs/experiments/versia-fusion-20261009.md), [public0.949 mirror](docs/experiments/mirror-submission-20261009.md)
- [Illustrated0.936 notebook](docs/experiments/0936-explainer-20261008.md)
- [TRAIN04 retirement](docs/experiments/train04-diagnostic-20261009.md), [MaViT failure](docs/experiments/mavit-hidden-diagnostic-20261009.md)
- [Earlier complete status and recovery history](docs/status-archive/status-before-20261009-1830-reconciliation.md)
