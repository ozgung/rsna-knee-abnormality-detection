# Completed R2 pair and 384 fold2 export — October11

Local source/config/aggregate audit: **129 PASS,0 FAIL,4 UNKNOWN**. A fresh SolMedium reviewed immutable builds against terminal artifacts; parent checked receipt consistency and export source. Detailed per-target AUC/support and hashes: `outputs/analysis/completed-family-review-20261011/{report.md,receipt.json}`. These are receipt checks, not fresh prediction/AUC recomputation or a tensor restore.

| Run | Report macro-AUC | Gold macro-AUC | Epochs / selected | Report change versus original224 |
| --- | ---: | ---: | --- | ---: |
| R2 fold2 | 0.909005 | 0.901849 | 10 / [8,9,7] | +0.012085 |
| R2 fold3 | 0.909011 | 0.908421 | 10 / [8,9,7] | +0.011367 |
| Original-label384 fold2 export | 0.912046 | 0.901528 | 12 / [9,11,10] | Different recipe; no matched original384 control |

R2 source/config/fold/exclusion/label hashes and runtime versions match the reviewed packages. All12 report targets are defined with explicit unweighted macro averaging; class supports match original comparisons. Report gains are broad: fold2 synovitis declines0.000587 and fold3 ACL declines0.000159; small gold58 target changes are mixed. GlobalOOF teachers, gold-guided development, fullfold checkpoint selection and altered class weights prevent independent or labels-only causal claims. No public-score forecast follows.

The384 export's consumed-resume SHA claim matches the parent-fetched exactfold2v14 manifest. Export source rebuilds selected top-k averages and strictly loads the resulting model, but lacks an independent all-tensor equality oracle and resume identity check. Receipt params are exporter defaults, not proven training config; `stopped_by_time_budget=true` is a hardcoded resume-path flag despite12completedepochs, and zero trainingtime is a placeholder. Preserve these provenance limits rather than treating them as a training interruption.

Next owner task: Fable should bind expected datasetversion/resumeSHA/trainingidentity before export, emit selected epochs and an independent reload-versus-topk tensor equality/digest receipt, and distinguish completed training from exporter defaults. Apply this to the next384 export without changing active training recipes; legacy exports remain unqualified for promotion until equivalent evidence exists. Image-only runtime/schema rehearsal remains required before a candidate submission.

At00:30UTC, publicadaptation and R2fold4 wereRUNNING; R2fold23 and384fold2export wereCOMPLETE. Queue18429 exited after one acceptedfold4push00:09; no duplicatecontroller. Colabfold3ep6 remainshealthy, but old uploader verifiesep0only due lagged versioncounter. Future min_version/deferred-binding fix passed3 targeted parent tests; actual new-code publication/restore remainspending. Reconciliation `outputs/analysis/live-jobs-20261011-0030/receipt.json`; persistence review `outputs/analysis/colab-fold3-version-lag-20261011/parent-review.json`.

## R2 fold4 terminal review — October11

Fold4 completed ten epochs with selected SWA epochs `[8,9,7]`: report macro-AUC **0.914071** versus original224 fold4 **0.906643** (Δ+0.007428); gold58 **0.899827** versus **0.897501** (Δ+0.002326). Local audit: **117 PASS,0 FAIL,4 UNKNOWN**. Detailed target AUC/support, signed deltas, immutable source/config/label hashes and executable checks: [review report](../../outputs/analysis/r2-fold4-review-20261011/report.md), [receipt](../../outputs/analysis/r2-fold4-review-20261011/receipt.json). All12 targets are defined under unweighted defined-target macro averaging; original support counts match. Training receipt records10186.86s (2.830h), CUDA/TeslaT4, torch2.10.0+cu128/timm1.0.26, and successful fresh completion; terminal log agrees.

Build/manifest/notebook embedded sources and config plus local label bytes bind to receipt claims. The local terminal copy contains receipt/log only: separately recovered runtime sources, selected checkpoint SHA/bytes and prediction/ID coverage remain unverified. No new tensor restore or prediction/AUC recomputation occurred; aggregate macro arithmetic alone was checked. GlobalOOF teachers, prior gold-guided development, fullfold checkpoint selection and changed class weights preserve the same development exposure and causal limits. Gold58 is not independent validation; the embedded receipt's “only clean number” claim is overstated. Parent next integrates the completed family evidence; image-only artifact/rehearsal gates still precede candidate promotion.
