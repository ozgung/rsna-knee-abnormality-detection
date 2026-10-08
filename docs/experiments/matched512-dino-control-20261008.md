# Matched512 frozen-DINO CPU control

**Prospective; no real fit launched.** The12-epoch fresh ConvNeXt experiment selected epoch6 at weak BCE0.48837184/AUC0.80396701, below the old2,000-trained frozen-DINO head. One CPU control removes the512-versus2,000 training-count difference using the exact immutable fresh512-training/200-development plan and original image-feature cache. The current failed larger-data gate remains unchanged.

The fixed `matched512-dino-attention-v1` recipe uses `StudyHead`, `fit_feature_stats` and `masked_loss` directly: attention, hidden128, dropout0.2, seed42, AdamW lr0.001/weight decay0.02, batch64, gradient clip5, at most80epochs, patience12 and improvement tolerance1e-5. Feature statistics and confidence-weighted prevalence use only the512 training rows. Exact0.5 weak labels are unknown; confidence is2×distance from0.5. Checkpoint selection uses confidence-weighted macro BCE on the same200 development rows. Per-target BCE/AUC/support and the training-prior baseline are reported explicitly, including undefined targets.

Execution is CPU float32 with two PyTorch threads and a setup-inclusive300-second hard deadline, checked before/after each batch and enforced with the existing POSIX timer. No device fallback, feature extraction, remote builder, resume, second fit or configurable recipe is added. An output directory must be new. A budget stop is an incomplete comparison; it does not authorize extending the cap or imply convergence. No GPU fit or submission follows automatically.

The runner pins exact plan/cache/extraction/sealed-manifest files and the audited feature/head helper source. It verifies exact ordered IDs, labels, targets, masks and original roles/groups. The whole original assignment hash is recomputed separately from the historical fresh builder's label-enriched digest; those two valid hashes differ and neither is rewritten. Full identity components must keep training outside all original development rows, gold and sealed rows. Original cohorts remain unchanged.

The compressed feature member necessarily decompresses all original rows before NumPy selects the712 plan rows; unselected feature rows are immediately discarded. The `gold` archive member is never opened. No gold/sealed/unused-development row enters statistics, the model, inference or metrics. The runner does not call `experiment.train`, which would evaluate gold. Private outputs contain the selected head, exact ordered200-row logits/probabilities/weak labels and linkage needed for paired analysis. The aggregate report contains no study IDs or report text. Keep all outputs ignored/private.

After review and integration, the parent may launch exactly one local fit:

```bash
.venv/bin/python -m knee.matched_dino \
  --plan outputs/kaggle/fresh-cache-result/fresh_reader_cache/plan-private.json \
  --artifacts outputs/experiments/dino-frozen-20261008/artifacts \
  --sealed-manifest outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json \
  --output outputs/experiments/matched512-dino-20261008
```

Outputs are `run-contract.json`, `training-report.json`, `best.pt` and `weak-predictions-private.json`. The experiment fingerprint binds input hashes, actual runner/helper bytes, fixed recipe, revision and ordered cohort assignment. Best checkpoint and private evidence bind the same contract and selected epoch; the evidence pins the checkpoint SHA. The report records environment, timing/stop reason, steps, epoch history, selected all12 metrics and prevalence baseline. Failure receipts exclude raw exception text that could contain paths/identifiers.

Interpretation is limited to a matched training count. Frozen features are3planes×8slice tokens, whereas fresh images are6slots×4three-frame windows; preprocessing and representation differ. This is not a causal architecture comparison. The old2,000-trained head also selected on300 development rows and ran CUDA. Shared weak labels and repeated use of the200 development rows remain limitations; no sealed or clinical generalization claim is possible here.

Initial checks: eight focused synthetic regressions pass, including a real two-epoch synthetic head fit, source/label/role/mask rejection, train-only statistics/prior invariance, protected-component crosslinks, unopened object-valued gold labels, exact prediction order, a per-batch stop and an actual timer interrupt during setup. A read-only real input check passes exact512/200 alignment and712×3×8×768 finite features with nonempty masks. No real model fit has run. Independent review passed. All174 ML tests pass with the existing pinned external source/receipt fixtures enabled; default dependency-free smoke, source compilation and diff checks pass. The full-suite log is `/tmp/matched-dino-tests.log` in this implementation session.


## Actual result, October8 at13:59UTC

Integrated source`ad609f3`; one CPU run completedPASS. Exact documented command used output`outputs/experiments/matched512-dino-20261008`; local session43200 ended with exit0. Log`outputs/research/2026-10-08/matched512-live.log`. Root focused8tests and default smoke passed before fitting; author full suite174/174 and independent review passed. No GPU or further fit was used.

Runtime1.622seconds,27completed epochs/216successful updates, early stopping; selectedepoch15 by the fixed BCE rule. Stable recomputed BCE0.476427977433, all12target AUC0.827198608813; no undefined targets. Best checkpoint SHA`f785d8754f4b7087ef472abe23da403d445fd0fe4ee612462bea0390de633e6a`. Private evidence and actual environment/config/source/ordered-cohort fingerprints are in the output folder. CUDA/MPS were not exercised.

Independent final verification checked source/input/contract/checkpoint/evidence hashes, exact ordered200IDs/groups/labels, checkpoint metadata, epoch selection, all target supports and metrics. [Aggregate paired verification](../../outputs/analysis/matched512-dino/paired-verification.json), [analysis script](../../outputs/analysis/matched512-dino/verify_compare.py), [summary](../../outputs/analysis/matched512-dino/summary.md).

| Control minus comparator | BCE difference (95% report-group interval) | AUC difference (95% interval) |
|---|---|---|
| Fresh four-epoch reader |−0.01943[−0.04110,0.00468]|+0.02172[−0.00780,0.05606]|
| Fresh12-epoch selected reader |−0.01194[−0.03350,0.01012]|+0.02323[−0.00351,0.05301]|
| Old2,000-row frozen single head |+0.02883[0.01351,0.04176]|−0.03037[−0.04823,−0.01507]|

Each bootstrap scheme used2,000valid draws,163report groups and zero invalids. Matching training count narrows the apparent fresh-versus-frozen gap; both fresh comparisons include zero. This is not proof of equivalence or architectural superiority. The old2,000-row comparison retains the300-row selection-cohort/CUDA confounds. No gold/sealed metrics, public submission or larger-data gate change follows from this result.

Next low-cost diagnostic: a separately declared2,000-row frozen-head CPU control using this same200-row selection cohort and exact head recipe, bounded at300seconds. It is not implemented or fitted yet. This would improve the sample-size comparison before choosing an image-model data expansion versus representation change. Do not extend the completed512run or silently reuse its contract.
