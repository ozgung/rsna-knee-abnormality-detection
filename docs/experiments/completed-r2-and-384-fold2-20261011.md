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
