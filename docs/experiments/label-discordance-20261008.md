# Fixed-cohort weak-label source diagnostic

October8,2026. TRAIN-03 completed but failed its larger-data gate. Before another image-model fit, this CPU-only diagnostic tested whether changing the weak teacher on the same known development cells reverses the fresh-versus-frozen ranking. It did not. No model was fitted, no new predictions or checkpoints selected, and no gold/sealed labels or metrics accessed.

The audit used the exact512training/200development plan. Four currently available named public CSVs matched all712rows: Steven`llm_labels_v4_blend.csv`, Pilkwang`report_labels_v2.csv`, Lixin`labels_llm_gpt56sol.csv`, and Qwen`qwen_knee_weak_labels.csv`. File SHA-256s, metadata hashes, licenses and current listing dates are preserved in the [aggregate receipt](../../outputs/analysis/label-discordance/aggregate.json). Their historical checkpoint-era hashes cannot be established, so the reconstruction is explicitly **candidate current consensus**, not proof of the exact labels used for the0.936 checkpoints. Four downloaded files total14.15MB; they remain private/ignored. Qwen contains raw/report columns that the analysis does not access.

On200development rows,72of1,827originally known cells changed direction (3.94%);13high-confidence disagreements occurred among1,657comparable cells. All573previously unknown cells acquired directional consensus values, but they stayed masked in this comparison. Original Steven-v1 confidence weights stayed fixed for both target sources. Consensus0.5 is handled as a soft target for BCE and excluded from the common AUC mask. All12AUCs were defined.

| Selected model, unchanged predictions | Original v1 BCE | Candidate consensus BCE | Original v1 AUC | Candidate consensus AUC |
|---|---:|---:|---:|---:|
| Fresh four-epoch reader |0.49586048|0.48305872|0.80547772|0.79164765|
| Frozen DINO single head |0.44760065|0.43420126|0.85756882|0.83988796|
| Frozen DINO four-head average |0.44628505|0.43460893|0.86033998|0.84191143|

Candidate-consensus fresh-minus-four-head differences: BCE+0.04844979, report-group bootstrap95%interval[0.02374704,0.07286066]; AUC−0.05026378, interval[−0.08725584,−0.01989126]. Each comparison used2,000valid draws/zero invalids, seed20261008. These are selected models on reused weak development data; intervals do not remove selection optimism or establish clinical truth. Compute was0.44CPU seconds, no GPU.

Interpretation: switching the weak source on existing known cells does not explain away the performance gap. This does not rule out benefits from denser training supervision or better labels. The four teachers are not independent: Stevenv4 includes Lixin-derived information and was gold-guided; the named Qwen file is now deprecated, but newer labels were not silently substituted. The old frozen heads also trained2,000rows versus fresh512, and their original selection used300development rows.

Next authorized bounded experiment: one weak-only frozen-DINO attention head on exactly512training/200development rows, same original head recipe and training-only normalization/prior, CPUfloat32/two threads, whole-run cap300seconds. No feature extraction, GPU, gold evaluation, model submission, or gate change. This is a diagnostic control, not another submission seed sweep. Its comparison with fresh512 removes the training-count imbalance; comparison with the old2,000-row head remains descriptive because selection cohort/device differ. Implementation/review precedes the fit.

Evidence: [verification](../../outputs/analysis/label-discordance/verification.json), [analysis script](../../outputs/analysis/label-discordance/analyze.py), [detailed next-step recommendation](../../outputs/analysis/label-discordance/recommendation.md). Existing larger-data thresholds remain unchanged.
