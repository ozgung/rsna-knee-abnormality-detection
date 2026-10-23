# TRAIN04 disposable trainability diagnostic

Prospective protocol, October9. The immutable original reference failed after11 completed epochs on the predicted wholly unknown pair. Its selected epoch9 weak BCE .596467853 is near its exact prior .596422255; all completed development macro AUCs are .510–.542. Training64 BCE improves only .007236 versus its own prior. Preserve the failed run. The next objective comparison remains useful; first establish the new coverage recipe can update its encoder and fit a small training subset. This is an engineering diagnostic, not a weak-validation score gate.

The package embeds all eight original `6af8902` modules and both public sources verbatim from the archived notebook, plus only `knee/coverage_diagnostic.py`. It verifies the immutable build digest, every source hash, original cache receipts, authentic initial file, reference report/run identity and exact runtime package/device environment. Only existing cache and reference kernel outputs are attached. Internet, competition inputs, extra datasets, pretrained downloads and external report processing are disabled.

1. Restore the exact original model and saved Python/NumPy/torch/CUDA RNG. Use the original seeded first training permutation's first pair, random eight-window sampling and normal augmentation. Run until one accepted FP16 update, recording encoder, non-bias head and bias gradient norms, preclip norm, clipping, AMP scale, parameter movement and online/EMA training metrics. Overflow retries are explicit; more than five consecutive overflows fail as in the reference. An unexpectedly wholly unknown first pair fails this diagnostic.
2. Independently restore original model/RNG again and create fresh optimizer/scaler/scheduler/EMA. Select32 training studies by the existing label-independent ID-hash policy. Use deterministic eight windows, fixed study order and no geometric/intensity augmentation, retaining original train-mode dropout. Cycle pairs until64 accepted updates. Evaluate the same deterministic eight-window inputs initially and after each16 attempted batches/final update, online and EMA. Report per-target weighted BCE/support, undefined targets, all-training prior and soft-label entropy floor; the comparators refer to this fixed32 cohort. No development selection/evaluation occurs.

Both phases retain batch2, original confidence-weighted batchmacro loss, AdamW LR1e-4 encoder/5e-4 head, weight decay.02, clip3, original EMA formula and12,000-successful-update cosine horizon. Post-forward wholly unknown batches skip; AMP skips may update only scaler state, not optimizer/scheduler/EMA/accepted count. Fixed32 windows are recomputed deterministically from the cache rather than storing a second image cache. No fitted model, optimizer or RNG checkpoint is saved or reused. Report PASS means execution completed; inspect finite gradients, actual encoder/non-bias-head movement and the reduction in training loss above entropy before claiming learning. This does not establish generalization.

The64-update limit defines the diagnostic experiment. There is no assistant wall-clock budget or accuracy threshold. The actual12h Kaggle platform limit remains enforced. Runtime/device/memory, every attempt, skips, source/config/init/selection fingerprints and sanitized failure type are retained. No identifiers or predictions are emitted in aggregate receipts. A failure or non-learning result triggers a specific investigation; it does not authorize an architecture sweep.

Build only from the integrated primary repository:

```bash
repo=~/Dev/rsna-knee-abnormality-detection
primary=~/Dev/rsna-knee-abnormality-detection
cd "$repo"
"$primary/.venv/bin/python" scripts/build_coverage_diagnostic.py \
  "$primary/outputs/kaggle/train04-reference-20261009/notebook.ipynb" \
  "$primary/outputs/kaggle/train04-cache-full-v2-result/coverage_cache" \
  "$primary/outputs/kaggle/train04-reference-20261009-result/coverage_training" \
  "$primary/outputs/kaggle/train04-diagnostic-reviewed-20261009" \
  knee-train04-diagnostic-20261009 \
  --cache-kernel ozgung/knee-train04-cache-full-20261008 \
  --reference-kernel ozgung/knee-train04-reference-20261009
```

The builder refuses an existing destination. After parent review/integration, rebuild in a fresh destination to bind the final commit and reviewed file hashes before launching. No launch is part of implementation acceptance. Retrieve only `coverage_diagnostic/diagnostic-report.json` and, on failure, `train04-diagnostic-failure.json`; no diagnostic weights exist. `initial.pt` is required; reference `last.pt` and `best.pt` are not.

Acceptance checks: focused synthetic CPU tests exercise accepted updates, non-bias grouping, true GradScaler overflow, empty-supervision state preservation and nonfinite loss refusal. Fixture tests verify the original archive and reject one-byte source drift; packaging the authentic FAIL reference must work without score/completion gates. Run the complete fixture-enabled unit suite, dependency-free smoke, compile built notebook and isolated bundled import. Actual ConvNeXt FP16/CUDA learning remains untested until the parent launches and inspects receipts.

## Failed-notebook attachment recovery

The first diagnostic`ozgung/knee-train04-diagnostic-20261009`v1 was accepted but Kaggle explicitly omitted the ERRORreference as an invalid kernel source; it failed before training. Preserve that run and its receipts. A private dataset`ozgung/knee-train04-initial-assets-20261009` transports only exact`initial.pt`, `run-contract.json` and `training-report.json` plus license/README. No images, clinical report text, study identifiers or predictions are included; byte hashes match the original outputs. Metadata license is`other`, original source license retained.

For the repaired build add`--reference-dataset ozgung/knee-train04-initial-assets-20261009`, use fresh destination`outputs/kaggle/train04-diagnostic-assets-20261009` and slug`knee-train04-diagnostic-assets-20261009`. `reference_kernel` remains original provenance, but only the complete cache is attached as a kernel. Dataset resolution changes transport only; original source and all three artifact hashes remain enforced.243testsPASS before final flat-path adjustment; focused7testsPASS after it. Remote execution remains pending.

Privateassetcreation completed anddatasetstatusready. Repaired`ozgung/knee-train04-diagnostic-assets-20261009`v1 pushedOctober8at23:51UTC from20b7d84; Kaggleacceptedbothinputs, API RUNNING. Build/log`outputs/kaggle/train04-diagnostic-assets-20261009/`, stream86524. Originalattachmentfailureelapsed0.040s, beforeupdates; sanitizedreceiptpreserved. Actualdiagnosticlearningresultsremainpending.

## October 9 mount-failure recovery

At 08:20 UTC, `ozgung/knee-train04-global-objective-assets-20261009` version 1 changed to ERROR. The UI log reports `ERRORED_MOUNTING_DATASET` for the cache notebook output (source version 356480955), exhausting 30 mount attempts. Output is 0 B; the API failure-message field is empty and its downloaded log is `[]`. There is no evidence of application execution, optimizer steps or training metrics from this attempt. This is a platform input-mount failure, not a measured failure of the corrected objective.

The cache inventory remains available. A fresh download of `coverage_cache/cache-report.json` matches the original pinned SHA256 and PASS status. At approximately 08:26 UTC, the parent launched one recovery job, `ozgung/knee-train04-global-recovery-20261009` version 1. Its notebook is byte-identical to the failed attempt; only notebook identity and title changed. Scientific revision remains `3c205e1`, with the same model initialization/RNG, labels, 2,000/200 split, objective, schedule, environment and 12-hour platform limit. This is a fresh attempt from the original initialization, not a checkpoint resume. The original failed notebook remains preserved.

Build: `outputs/kaggle/train04-global-recovery-20261009/`. Launch: `~/miniconda3/bin/kaggle kernels push -p outputs/kaggle/train04-global-recovery-20261009 --timeout 43200`. Evidence: `outputs/analysis/train04-global-objective-20261009/mount-failure-0823.json`, `cache-source-check/`, and `recovery-build.json`. No source or scientific changes; verified exact notebook hash, metadata-only differences and input-receipt hash. Runtime tests were not repeated for identical code.

Inspect recovery logs and small reports before downloading any selected checkpoint. If the same input-mount failure recurs, investigate a different cache transport rather than looping retries. Do not revive the old local controller. MaViT's distinct train2000 diagnostic remains active; its UI showed 24 minutes running, but no logs or optimizer/inference progress were established.

## Global-objective recovery result — October9,11:25 follow-up

`ozgung/knee-train04-global-recovery-20261009` v1 completed all12 epochs. Contract/source/selected-prediction validators passed. It performed11,995 accepted optimizer updates,4AMP skips and1zero-supervision skip in8,468.19seconds; peakCUDA12.99GB. Earliest minimum-BCE EMA selected epoch4: **BCE0.5945647, weak macroAUC0.5235904**, versus training-only prevalence BCE0.5964223. All per-target support/evidence is preserved in `outputs/analysis/reconcile-20261009-1125/receipt.json` and the downloaded small reports.

The previous incomplete11epoch reference selected BCE0.5964679/AUC0.527841. The corrected objective improved BCE by0.001903 on the fixed cohort, but useful discrimination is not established. This is not a matched comparison with Fable's new full-pool/public-label folds. No further TRAIN04 fit or model submission is planned; its artifacts and completed result remain immutable. Selected prediction evidence was reconstructed exactly and BCE recomputed; fitted checkpoint bytes remain remote and were not independently downloaded/rehashed in this follow-up.
