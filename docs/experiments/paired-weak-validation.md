# Paired fresh/frozen weak-validation analysis

`knee.paired_validation` compares the selected fresh EMA checkpoint with the preserved frozen-DINO single head and fixed four-head average. It performs no fitting, model selection, blend search, image decoding or gold evaluation. NumPy is optional until analysis runs; importing the module needs only the standard library and project audit modules.

Run after downloading the selected fresh checkpoint and its private sidecar:

```bash
python -m knee.paired_validation \
  --plan outputs/kaggle/fresh-cache-pilot-result/fresh_reader_cache/plan-private.json \
  --fresh-evidence outputs/kaggle/fresh-train-result/fresh_reader_training/best-weak-validation-private.json \
  --baseline-dir outputs/experiments/dino-frozen-20261008/artifacts \
  --comparison-dir outputs/experiments/head-comparison-20261008/comparisons \
  --output outputs/analysis/fresh-pilot-baselines/paired-selected.json
```

`best.pt` must be beside the sidecar, or supplied through `--fresh-checkpoint`. It is hashed, never deserialized. The command rejects an existing output path. Inputs remain private and ignored; the resulting JSON contains only aggregate metrics, supports, fingerprints and uncertainty intervals.

The command verifies the final post-header plan fingerprint, exact ordered fresh IDs/targets/labels, finite logits and sigmoid consistency, checkpoint SHA256 and selected BCE. It verifies the preserved feature cache, original training/validation roles, report groups, weak labels, frozen checkpoint hashes, full original weak-prediction coverage and target order. Saved frozen rows may be differently ordered; explicit ID alignment produces the exact plan order. The original 300-row BCE values and exact fixed four-member probability mean must still reproduce their recorded comparison. Features, gold arrays and reports are never requested from the feature archive.

BCE uses the original float32-cast soft labels, masks exact `0.5`, weights by `2*abs(y-0.5)`, normalizes each supported target, then averages targets. Fresh loss is recomputed stably from saved logits in float64, preserving finite loss when sigmoid rounds to zero or one; the saved float32 selection loss must match within `2e-6` absolute/relative tolerance. Frozen probability loss retains its original float64 machine-epsilon clipping convention. Weak AUC thresholds known labels at `>0.5`, handles ties by average rank, and requires every target for the primary macro. Per-target counts and undefined targets are always reported.

Two paired bootstrap schemes use seed **20261008** and **2,000 draws each**: sampling studies, and sampling whole normalized-report groups. Every model receives the same multiplicities. Group draws sample the original number of groups with replacement, preserve all members, and may contain different numbers of studies. Reported differences are **fresh minus frozen**: negative BCE and positive AUC favor fresh. Intervals are the 2.5/97.5 percentiles with linear interpolation. BCE intervals keep the original supported-target set; losing any of those targets invalidates the draw. Macro-AUC draws require both known classes for every target. Valid and invalid draw counts accompany each interval; intervals condition on valid draws and are null if none remain.

These intervals condition on already-selected models and epochs. The reused 200-study development cohort is not an untouched holdout; selection optimism remains. Baselines used 2,000 training studies versus at most 512 fresh, so this is not a matched training-budget comparison. Report groups do not establish independent patient identities. Shared weak-label provenance prevents interpreting these metrics as independent clinical validation or a prediction of leaderboard performance.
