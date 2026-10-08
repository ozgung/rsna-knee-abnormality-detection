# Same-cohort twelve-epoch duration experiment

**Prospective; implemented locally, not launched.** The four-epoch pilot completed with weak-validation BCE0.49586048/AUC0.80547772, improving fixed-train diagnostics and epoch4 selected. It passed the predeclared duration spending gate and failed the larger-data gate. Hypothesis: additional optimization on the same512 training/200 development studies improves the selected EMA checkpoint. Compare against the unchanged four-epoch pilot and frozen four-head same-200 baseline (BCE0.44628505/AUC0.86033998). Weak BCE remains the primary selection metric; report all12 target AUC/support and paired differences without interpreting reused weak labels as independent clinical truth.

The sole named profile is `same512-duration12-v1`. It requires the exact existing cache plan `1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942`. The effective config differs **only** in `max_epochs:4→12`; cache/plan bytes, labels, split, seed42, loss, confidence masking, learning rates, weight decay, EMA, inputs, train64 diagnostic and BCE selection remain unchanged. Initialize fresh generic ImageNet weights, optimizer, EMA and a new12-epoch cosine schedule. The initial encoder tensor SHA must be `b1d4d0986c66bb1d02eddf7bf4bceea7fde12aced5def666c28d3b89951b569e`; a changed download fails. Never warm-start the completed four-epoch checkpoint or extend its cosine schedule.

A distinct immutable experiment fingerprint binds the cache plan, effective config, public source hashes, all bundled runtime helpers and builder source bytes, backbone, initial encoder SHA, selection rule and timing policy. The builder, run contract, best/last checkpoints, training report and version2 selected evidence carry it. The cache-plan fingerprint remains unchanged. The profile does not evaluate gold or the sealed cohort, and it changes no identity-audit code.

The GPU gate runs exactly64 consecutive batches/128 training rows in the same original order with no optimizer updates:16 warmup and48 timed batches. Every batch must complete a supervised backward pass; an unknown-only batch fails the gate. Gradient cleanup and Python/NumPy/PyTorch RNG restoration run on success and failure. The gate times validation and fixed-train64 diagnostic evaluation, charges all setup/warmup time against5,400seconds, reserves180seconds for checkpoint work, and applies a1.5margin. **All12 projected epochs must fit or training fails**; the horizon is not shortened. The notebook's external launch timeout is5,700seconds, while its internal whole-stage cap stays5,400seconds.

Initial build (no checkpoint attachment, no launch):

```bash
.venv/bin/python scripts/build_fresh_reader.py \
  outputs/research/2026-10-08/convnext-audit \
  outputs/kaggle/fresh-duration12 knee-fresh-duration12-20261008 \
  --stage train --training-profile same512-duration12-v1 \
  --cache-kernel ozgung/knee-fresh-cache-20261008 \
  --plan-fingerprint 1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942
```

The notebook is private, GPU enabled and cache-only; internet permits generic ImageNet initialization. Parent reviews the build then may launch with `kaggle kernels push -p outputs/kaggle/fresh-duration12 --timeout 5700`. No competition/report/weak-label/public knee-weight attachments are added.

Resume is limited to an incomplete run of this exact experiment and runner, with its matched best/last pair,12-epoch scheduler and selected evidence. The builder requires `--resume-kernel` plus a downloaded `--resume-contract PATH/run-contract.json` and adjacent `training-report.json`. Both are hash-pinned and checked against mounted copies. Reviewed prior elapsed time is subtracted from the outer notebook deadline **before attachment resolution**. The runner checks that small report and remaining budget before loading checkpoint tensors, hashing cache arrays or constructing a model; checkpoint elapsed cannot exceed its report. Cache verification, initialization and restoration use the remaining cumulative budget. Exhausted runs fail without allocating a model or receiving a new90-minute allowance. The original4-epoch runner/checkpoints remain immutable and cannot resume through this profile.

Paired analysis explicitly accepts genuine version1 evidence without requiring missing historical run files, limits its selected epoch to the original≤4 horizon, and rejects it if adjacent artifacts advertise a named experiment. Version2 evidence must match its run/report, original recorded source hashes, selected checkpoint SHA and exact plan predictions/labels. Future analysis may read archived source hashes without substituting current ones. No checkpoint deserialization is added to analysis.

Local verification:166/166 ML tests pass with existing audited fixtures; dependency-free166 checks pass with80 optional skips. Eight new duration tests cover the config/source contract, all12 budget gate, no-update/RNG/gradient behavior, missing supervised timing batches, legacy/new evidence mixing, own checkpoint pairing and near-exhausted resume rejection before tensor/model work. Existing cache/decoder/validation tests remain green. GPU throughput, full12-epoch fitting and CUDA resume remain unverified until actual run receipts exist.
