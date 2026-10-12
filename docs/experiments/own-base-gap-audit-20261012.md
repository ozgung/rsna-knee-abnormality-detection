# Own-base gap audit — October 12, 2026

The next diagnostic holds our original224 fold0 checkpoint fixed and compares its existing16 evaluation windows with42 candidate windows in the same44-slice cache. This tests evaluation sampling density before committing to a new96-slice corpus and training. It does not add source slices or establish native-detail improvement.

## Recipe evidence

- Own224 already used `labels_v1_blend`, ten epochs and report-selected top-three SWA. Fold0 selected epochs8,9,7; report AUC0.901842; full-five public0.914. TrainingK8/evaluationK16.
- Original own384 c384r used the same blended labels,12 epochs and top-three SWA, trainingK12/evaluationK24. Do not confuse it with the older14-epoch bootstrap c384b artifact.
- Public .945/.949 models also differ in knee-specific initialization, OAI supervision, label recipe, source depth, inference coverage and checkpoint selection. Their score gap does not isolate any factor or establish a competition-only ceiling.
- A labels-v1-to-blend arm and single-checkpoint-to-SWA arm would repeat features already present. A longer schedule is possible but late own224 epochs plateaued; it is a weaker first diagnostic.
- A96-slice inference builder exists in cached `outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py`. The exact public training cache/builder is unavailable in inspected artifacts. A controlled input comparison needs explicit series allocation, span, intensity and masks, not just changing an array dimension.

Detailed read-only evidence: `outputs/analysis/base-gap-20261012/recipe-audit.md`. Existing141-check own224 audit remains authoritative for the original family. No new metric or GPU result is claimed.

## Fixed-window diagnostic acceptance

Use the same fold0 SWA bytes,870 IDs, original corpus/labels/truth/masks and model preprocessing. K16 must reproduce saved held-out predictions within a prospectively documented backend tolerance before interpreting K42. Change only evaluation window count. Preserve original candidate-center behavior, including known masked/series-boundary limitations, and report them; do not silently fix those while comparing K. Report per-target AUC/support, explicit macro policy, paired report-component uncertainty and measured runtime/memory. This is repeatedly used development data with checkpoint-selection exposure, not an untouched outer test. No leaderboard submission or training follows automatically.

SolMedium implements a separate evaluator/builder/tests in managed worktree `own224-window-diagnostic`; parent owns review, commit and one qualified run. Production inference is unchanged.

## Native-ROI feasibility findings

The SKM-TEA localizer weights have verified upstream identity, but public anchor-selection heuristics explicitly used gold58 montages. Thus the complete preprocessing is not gold-unexposed. The previously qualified mechanical center90 crop failed anatomical review; do not reuse it as a successful ROI result. Existing S6 full-field-plus-local union passed a limited visible anatomy screen, which does not qualify a new24-view subset. Current native crop helper hardcodes320 resampling;224/384 need an explicit contract. Measure actual resampling, out-of-field and depth-boundary projections, fallback routes, and preserve full context. Hidden-test failure rates remain unknown. Evidence: `outputs/analysis/native-roi-feasibility-20261012/source-review.json`.

## Diagnostic implementation qualification — 13:06 UTC

Separate evaluator/builder/tests implemented; original production source unchanged. Parent review caught gold58 included in a derived training count (3533 instead of3475), fixed the exclusion and added regression coverage. Scoring membership/groups are checked before GPU work. Actual pinned486-tensor checkpoint loaded safely with finite/identity checks. Local real-data preflight reproduces saved macro0.901842101355689,870 development IDs,3475 training IDs and843 report components; gold58 excluded. Failed pre-fix receipt preserved.

Focused new tests10PASS/1pyarrow skip; combined legacy/new15PASS/4 dependency skips. The installed-pyarrow interpreter separately exercised actual truth/grouping/metrics, and source-pinned sampler/tensor parity was checked synthetically by Sol. CPU smoke and diff checks PASS. These do not establish CUDA prediction parity; runtime requires K16 versus saved probabilities absolute error≤1e-4 before interpreting K42. Both arms use the historical CSV-export configuration batch4/chunk0/FP16 (not the training evaluator batch2/chunk8). Exact source output remains completedv3, verified13:05UTC.

Both arms are evaluation-only; hypothesis, endpoint and stop condition are fixed above. The bounded cohort is870 studies. Runtime remains unmeasured for this evaluator; approximately tens of minutes is a planning estimate, not a cap. Use an existing Kaggle T4 allocation (1.808h available12:57UTC), preserve failures/checkpoints, and stop on source/schema/parity/resource failure without silently reducing batch size or relaxing tolerance. No competition submission is authorized by this diagnostic result alone.

Evidence: `outputs/analysis/own224-window-diagnostic-20261012/{local-endpoint-preflight.json,local-endpoint-preflight-before-fix.json,input-source-live.json}`.

## Failed historical reproduction and prospective same-run endpoint

October12 v1/kernel138332643 completed870 K16 predictions then failed its frozen1e-4 historical probability gate. Maximum absolute error.00173277, median.00004410; pairwise masked macro .901842101356 historical versus .901836297777 reproduced. IDs exact, pinned checkpoint/source/metadata match; model state check passed. K42 never ran. The failed protocol and all outputs remain preserved; this is not evidence of a density gain. Numerical/source audit found no demonstrated semantic mismatch; original workers2/subprocess versus direct/notebook execution differs.

A separate first32 probe v1/kernel138336007 checks tensor identity, repeated FP16, FP32 and CPU loader/thread paths. It cannot qualify K42. Acceptance of Fable13:34's proposal changes the NEXT experiment, declared before any K42 output: same870 K42 minus same-run K16, paired2000-component bootstrap, same pinned model/inputs/truth/precision/batch grouping. Historical probabilities become a descriptive comparison in this explicitly named `same_run_v2` protocol, not a relaxed reproduction claim. Stop on source/schema/state/nonfinite/resource failures; no fallback. K-dependent numerical differences need not cancel. Positive macro delta plus bootstrap lower bound above zero is the initial promotion signal; inspect supported-target damage and runtime before further experiments, never submit automatically. This remains exposed development, not independent validation.

No training, optimizer updates, external/OAI data, or inference production changes. Bounded870 studies on existing T4 allocation; expected minutes based on283-second original K16 evaluation including setup, not an artificial cap. Full target supports and candidate window artifacts are mandatory outputs. Evidence: `outputs/analysis/own224-window-diagnostic-20261012/{parity-failure-v1.json,source-audit.md}`.
