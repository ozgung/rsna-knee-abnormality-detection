# Label-table benchmark against gold 58 (task B1, 2026-10-09)

Purpose: before any table feeds training, check each public CC0 report-derived label table for gold leakage, then score it against the 58 fully labelled studies (CPU only, aggregate numbers only). Outputs (ignored): `outputs/analysis/label-benchmark-20261009/{summary.json,per_table.csv,agreement.csv}`.

```bash
# downloads: kaggle datasets download <ref> -f <file> -p data/weak-labels/<owner>_<slug> -q   (CLI: /Users/ozgun/miniconda3/bin/kaggle)
.venv/bin/python -I scripts/benchmark_label_tables.py   # defaults: --data-dir data/raw --tables-dir data/weak-labels --out-dir outputs/analysis/label-benchmark-20261009
```
Parquet files are read through a CSV conversion made with miniconda's pyarrow 26.0.0 (the `.venv` has no pyarrow; `pip install pyarrow` was run in miniconda only). Bootstrap: 2,000 study-level draws, seed 20261008; one draw (index 1365, MCL has a single class) is skipped for every table.

## Tables, leak verdicts, gold-58 macro AUC (rows = table rows, gold = gold studies present; SHA256 = first 12 hex, full in `summary.json`)

| table (ref : file) | rows (gold) | SHA256 | verdict | macro AUC [95% CI] |
|---|---|---|---|---|
| steven_v4_blend (stevenleehans/rsna-knee-llm-report-labels : llm_labels_v4_blend.csv) | 4407 (58) | eb4c2b29e7e8 | not leaked | 0.893 [0.860, 0.922] |
| local llm_labels_v2.csv (provenance unrecorded) | 4407 (58) | 3c082a987939 | not leaked | 0.887 [0.853, 0.918] |
| local llm_labels_full.csv (provenance unrecorded) | 4407 (58) | 96444cd3c7d3 | not leaked | 0.878 [0.844, 0.909] |
| pilkwang_v2 (pilkwang/rsna-knee-llm-labels : report_labels_v2.csv) | 4406 (57) | 6f704a7bdb2f | not leaked | 0.870 [0.835, 0.902] |
| lixin_sol56 (lixin73/rsna-knee-llm-report-labels-sol56 : labels_llm_gpt56sol.csv) | 4407 (58) | a79b05e60933 | not leaked (hard 0/1) | 0.835 [0.806, 0.865] |
| laymond qwen3-8b (laymond/rsna-knee-abnormality-qwen3-8b-weak-labels : qwen_knee_weak_labels.csv) | 4407 (58) | 26adba022842 | not leaked (hard 0/1) | 0.818 [0.785, 0.850] |
| nartaa labels_v0raw.parquet (nartaa/rsna-knee-hpo-assets) | 4349 (0) | a63f9b0679ab | no gold rows; non-gold rows identical to steven v4 | inherits 0.893 |
| nartaa labels_v1_gemlow.parquet | 4349 (0) | 77a3857a25a8 | no gold rows; non-gold rows identical to train_labels_v1 | inherits 0.888 |
| nartaa labels_v1_blend.parquet / labels_v1_perlabel.parquet | 4349 (0) | d2cdb8a37e0c / 56db71b667e9 | no gold rows, unscorable here | n/a |
| nartaa train_labels_v0.parquet | 4407 (58) | 43dc6ef04941 | not leaked; gold rows identical to steven (duplicate for gold) | 0.893 [0.860, 0.922] |
| nartaa train_labels_v1.parquet (= gemlow + gold rows) | 4407 (58) | a134cfce2f75 | not leaked (30% of gold cells within 0.01) | 0.888 [0.854, 0.916] |
| jev HYBRID_labels_scores.csv (riadmohamed42/jev-knee-labels) | 4407 (58) | 40f4d6c46699 | not leaked, but gold-informed upstream | 0.904 [0.870, 0.932] |
| jev FINAL2_labels_scores.csv | 4407 (58) | 6f70de34e75d | not leaked, gold-informed weights | 0.874 [0.844, 0.902] |
| rayanbabur train_targets_soft.csv (rayanbabur/rsna-knee-2026-calibrated-soft-targets) | 4407 (58) | 6dac0e2d9ba7 | LEAKED: 100% of gold cells within 0.01 of official, 12/12 AUC = 1.0, gold rows binary vs soft elsewhere | 1.000 (meaningless) |
| dreaddevelopment/rsna-knee-labels : labels_llm_soft.csv | 4349 (0) | 8cec477fb555 | no gold rows, unscorable here | n/a |

Also fetched: `gold_benchmark_58.csv` (58 rows, equals official labels), `gold74_eval_set.csv` (74 rows: the 58 plus 16 hand-checked studies, equal to official on the 58), `train_targets_weights.csv` (weights, not scored). Only 1 of 16 tables leaks; the handoff's "five leak" claim is not reproduced for this list. JEV README: hybrid = 0.95 x per-target swap of steven (9 targets) / pilkwang (ACL, MCL, Fracture) chosen on gold-58, + 0.05 x ensemble weighted on 67 gold studies; its non-gold values correlate r = 1.000 with those sources. Silent cells: steven marks 19% of cells exactly 0.25 (Baker's 48%, Fracture 43%); local v2/full use 0.5 (19%/25%).

## Per-target gold-58 AUC (top 3 non-leaked, plus recommended triple)

| table | ACL | MCL | MedMen | LatMen | MedOA | LatOA | PF OA | Effus | Synov | Baker | Contus | Fract |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jev_hybrid (gold-informed) | .999 | .981 | .947 | .872 | .930 | .855 | .902 | .877 | .785 | .920 | .870 | .904 |
| steven_v4_blend | .987 | .968 | .948 | .879 | .932 | .833 | .902 | .877 | .790 | .944 | .860 | .793 |
| nartaa gemlow (train_labels_v1) | .996 | .984 | .922 | .859 | .905 | .829 | .934 | .837 | .789 | .905 | .789 | .903 |
| mean(steven, gemlow, pilkwang) | .995 | .983 | .951 | .876 | .932 | .802 | .916 | .862 | .799 | .938 | .856 | .840 |

## Ensembles (gold-58 macro AUC [CI]; paired difference vs steven_v4)

top3 non-leaked (jev_hybrid, steven, gemlow) 0.897 [0.862, 0.926], +0.004 [-0.006, +0.013]; all 9 distinct non-leaked 0.893 [0.858, 0.924], +0.000 [-0.011, +0.011]; all incl. duplicates 0.894; top3 gold-uninformed (steven, gemlow, v2) 0.895 [0.861, 0.924]; all gold-uninformed (7) 0.896 [0.861, 0.926]; one per labeller family (steven, pilkwang, gemlow, lixin, qwen) 0.897 [0.862, 0.928], +0.005 [-0.006, +0.014]; recommended triple 0.896 [0.861, 0.926], +0.003 [-0.008, +0.012]. No ensemble is distinguishable from steven_v4 alone.

## Agreement on 4,349 non-gold studies (3,814 Latin, 318 Greek, 217 Cyrillic; gold has 52/3/3, so scripts cannot be checked on gold)

Over 118 distinct table pairs the directional disagreement is 8.2% Latin, 8.3% Greek (+0.1 pp [-0.5, +0.7]), 7.9% Cyrillic (-0.3 pp [-1.0, +0.4]); Pearson 0.840/0.829/0.814. For five independent labellers (15 pairs) it is 9.3/10.0/9.3% (Greek +0.7 pp [-0.1, +1.4]). So Greek/Cyrillic do not disagree more overall, but specific cells do: Cyrillic x Medial Meniscus 17.6% vs 6.1% Latin (+11.5 pp [+8.7, +14.3], elevated for all 16 tables, +8 to +19 pp), Greek x Synovitis +5.4 pp [+4.0, +6.8], Contusion +4.2 pp in both scripts, Greek x Fracture +2.3 pp. Targeted local re-read candidates: Cyrillic medial meniscus (217 studies), Greek synovitis (318), Contusion in both (535). Disagreement is not error; no table can be called right here.

## Limitations

Only 58 gold studies (SE of macro AUC about 0.016-0.017, MCL has 9 positives); table differences below about 0.03 are unresolved and the top-3 pick is selected on the same 58, so it is optimistic. Tables may be gold-informed upstream even when not leaked (JEV documented; others unknowable). Leak rule (a) is uninformative for natively binary tables (lixin/qwen score 0.80/0.83 accuracy). The unscorable tables (v1_blend, v1_perlabel, dread) rely on author-reported numbers. No report text was used beyond the dominant-script label.

## Recommendation

Average steven_v4_blend, nartaa v1_gemlow (Gemini re-read; use `labels_v1_gemlow` for the 4,349 non-gold rows, identical to `train_labels_v1`) and pilkwang_v2: three distinct labellers, none gold-informed, and complementary (pilkwang is strongest on ACL/MCL, gemlow on Fracture, steven on Baker's and menisci). Expect at most a small gain over steven alone; judge it by OOF and LB, not by gold 58. Avoid rayanbabur soft targets (gold labels are inside the table, so it cannot be validated and would contaminate gold-58), jev HYBRID/FINAL2 as additional members (steven/pilkwang repackaged with gold-selected weights, so they double-count them and their AUC is optimistic), local v2/full (older steven variants; full degrades Synovitis to 0.678), and lixin/qwen except as low-weight extras (hard 0/1, lowest AUC, weak on Effusion and PF OA). Do not use v1_blend, v1_perlabel or dread labels until their gold-58 behaviour is reproduced.
