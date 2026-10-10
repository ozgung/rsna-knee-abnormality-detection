# Original-label CoAtNet224 full-five audit — October 10, 2026

All five native224 models completed their fixed ten-epoch recipe. Parent report/artifact audit: **141 checks PASS**, pooled masked report AUC **0.900342** over4,345 eligible studies. All twelve targets are defined; macro aggregation is the unweighted mean over targets with both classes present. Unmentioned labels remain masked.

| Fold | Report AUC | Selected SWA epochs |
| --- | ---: | --- |
| 0 | 0.901842 | [8, 9, 7] |
| 1 | 0.902008 | [9, 8, 7] |
| 2 | 0.896921 | [8, 9, 7] |
| 3 | 0.897643 | [9, 8, 7] |
| 4 | 0.906643 | [8, 9, 7] |

The audit checked frozen source/recipe/label/truth/fold/exclusion hashes, exact eligible membership and support, all ten epochs, and report-selected top-three checkpoints. Original-label means `labels_v1_blend.parquet`, not raw report labels; evaluation uses `train_labels_v1.parquet` independently. Checkpoint selection uses the evaluated report fold, so these are selected-fold estimates rather than untouched outer-fold validation.

Initial audit failures were resolved with evidence: local train_meta files for folds2/3 were from the prior failed384 run and were replaced with exact completed version2 outputs. Fold1's command uses oof_every870 (raw fold size) while eligible coverage is866. Trainer source checks the bound covers all eligible IDs and explicitly evaluates the complete held-out fold. The audit now checks both the exact raw bound and this source behavior; it does not weaken full-fold coverage. Failed receipt preserved as `audit-before-metadata-refresh.json`.

Evidence: `outputs/analysis/coatnet224-fullfive-audit-20261010/{audit.py,audit.json}`. Fold2/3 source: `ozgung/knee-tracka-coatnet224-fold23-20261009` v2. Checkpoint and full-five rehearsal checks subsequently passed below. Final-visible parity remains required before one authorized code entry.

Fable's later round2 bootstrap experiment was triggered by gold58 comparisons. Consequently those repeated gold-guided choices cannot claim independent gold audit evidence. Global teacher exposure further biases report OOF. Preserve the separate exploratory pair; do not use its gold or report improvements to promote it automatically. The original-label qualification and fixed submission recipe remain independent of that new experiment.


All five selected checkpoint hashes and metadata/finite checks passed. Folds2/3 files are292,826,563bytes each,486 tensors; SHA256 respectively `3c4c9c000ca15db79260d1b140ff594112c85e5d41e35f1ce849e8df790b23c9` and `d7a32474b016e97ad66b8a935e005ec19e80cda646c4e7c85b62cba11a406d9e`. Receipt: `outputs/analysis/coatnet224-fullfive-audit-20261010/checkpoint-review.json`.

**Full-five CUDA rehearsal v1 PASS.** All five strict loads, exact raw/corpus/mask/window/probability/repeat parity,16 reader stresses and authoritative three-study visible CSV passed. Internal runtime69.27seconds; peak hostRSS2,448,723,968bytes. This remains a bounded measurement, not hidden-test completion. Exact contract873ea579… and visibleCSV SHA `d13957c828844e6f08b4737209e185501142f88ebfef0ca2038bb1e99afeb114`. Kernel `ozgung/knee-own-coatnet224-rehearsal-fivefold-20261010` v1, build/result folders with matching names under outputs/kaggle.

Final image-only notebook `ozgung/knee-own-coatnet224-submission-20261010` v1 accepted. ContractSHA `368e1bffab6fca3edd2b74987552216ceaea758bb7c95168540d4df0b023be8e`, notebookSHA `1f3d6b3b7c6ad30f1b8f9cb3301457f1cc56bddb1c7ab42e3ed4272d7a8590b6`. It reconstructs the exact qualified archive, streams test images, enforces official orderedcoverage and finite probabilities, uses an equal five-model probability mean, and enforces the9hour runtime limit. Final-visible completion/parity and competition entry remain pending at this checkpoint.


**Final-visible PASS; competition entry accepted.** Notebook v1 completed in115.9s wall execution; internal inference/setup94.2726s, peak GPU2,001,337,344bytes, peak host2,128,908,288bytes. All five checkpoint hashes/strict loads, source contract, label-free receipt, exact authoritative coverage and both written CSV copies passed parent checks. CSV SHA remained exactly `d13957c828844e6f08b4737209e185501142f88ebfef0ca2038bb1e99afeb114`. Receipt: `outputs/analysis/coatnet224-fullfive-audit-20261010/final-review.json`. This three-study runtime is bounded evidence, not a measured hidden-test runtime.

Entry **57037296** accepted2026-10-10 08:52:58UTC, PENDING at08:53UTC, score unknown. Three daily entries remain. Best public0.950 entry56986393 remains preserved. Next: inspect hidden-test completion/public result; Fable may use the released batch slot for original384fold0 export.

**Public result verified10:28UTC: entry57037296 COMPLETE,0.914.** Best0.950 remains preserved. Finish active original384fold1 and export0–1, then hold generic folds2–4 while qualifying full-public adaptation and its frozen control.
