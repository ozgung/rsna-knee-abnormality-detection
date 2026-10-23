# Same-cohort twelve-epoch duration experiment

**Completed/PASS on 2026-10-08; verified at 13:34 UTC. Selected epoch6 weak BCE0.48837184/AUC0.80396701; the larger-data spending gate failed.** The four-epoch pilot completed with weak-validation BCE0.49586048/AUC0.80547772, improving fixed-train diagnostics and epoch4 selected. It passed the predeclared duration spending gate and failed the larger-data gate. Hypothesis: additional optimization on the same512 training/200 development studies improves the selected EMA checkpoint. Compare against the unchanged four-epoch pilot and frozen four-head same-200 baseline (BCE0.44628505/AUC0.86033998). Weak BCE remains the primary selection metric; report all12 target AUC/support and paired differences without interpreting reused weak labels as independent clinical truth.

The sole named profile is `same512-duration12-v1`. It requires the exact existing cache plan `1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942`. The effective config differs **only** in `max_epochs:4→12`; cache/plan bytes, labels, split, seed42, loss, confidence masking, learning rates, weight decay, EMA, inputs, train64 diagnostic and BCE selection remain unchanged. Initialize fresh generic ImageNet weights, optimizer, EMA and a new12-epoch cosine schedule. The initial encoder tensor SHA must be `b1d4d0986c66bb1d02eddf7bf4bceea7fde12aced5def666c28d3b89951b569e`; a changed download fails. Never warm-start the completed four-epoch checkpoint or extend its cosine schedule.

A distinct immutable experiment fingerprint binds the cache plan, effective config, public source hashes, all bundled runtime helpers and builder source bytes, backbone, initial encoder SHA, selection rule and timing policy. The builder, run contract, best/last checkpoints, training report and version2 selected evidence carry it. The cache-plan fingerprint remains unchanged. The profile does not evaluate gold or the sealed cohort, and it changes no identity-audit code.

The GPU gate runs exactly64 consecutive batches/128 training rows in the same original order with no optimizer updates:16 warmup and48 timed batches. Every batch must complete a supervised backward pass; an unknown-only batch fails the gate. Gradient cleanup and Python/NumPy/PyTorch RNG restoration run on success and failure. The gate times validation and fixed-train64 diagnostic evaluation, charges all setup/warmup time against5,400seconds, reserves180seconds for checkpoint work, and applies a1.5margin. **All12 projected epochs must fit or training fails**; the horizon is not shortened. The notebook's external launch timeout is5,700seconds, while its internal whole-stage cap stays5,400seconds.

Executed initial build (no checkpoint attachment):

```bash
.venv/bin/python scripts/build_fresh_reader.py \
  outputs/research/2026-10-08/convnext-audit \
  outputs/kaggle/fresh-duration12 knee-fresh-duration12-20261008 \
  --stage train --training-profile same512-duration12-v1 \
  --cache-kernel ozgung/knee-fresh-cache-20261008 \
  --plan-fingerprint 1bd346c05c0aac20f54b82b790e356d3d6c1b95606accaa4a5d1da7e3e421942
```

The notebook is private, GPU enabled and cache-only; internet permits generic ImageNet initialization. After the build review, the parent-authorized launch used `kaggle kernels push -p outputs/kaggle/fresh-duration12 --timeout 5700`. No competition/report/weak-label/public knee-weight attachments are added.

Resume is limited to an incomplete run of this exact experiment and runner, with its matched best/last pair,12-epoch scheduler and selected evidence. The builder requires `--resume-kernel` plus a downloaded `--resume-contract PATH/run-contract.json` and adjacent `training-report.json`. Both are hash-pinned and checked against mounted copies. Reviewed prior elapsed time is subtracted from the outer notebook deadline **before attachment resolution**. The runner checks that small report and remaining budget before loading checkpoint tensors, hashing cache arrays or constructing a model; checkpoint elapsed cannot exceed its report. Cache verification, initialization and restoration use the remaining cumulative budget. Exhausted runs fail without allocating a model or receiving a new90-minute allowance. The original4-epoch runner/checkpoints remain immutable and cannot resume through this profile.

Paired analysis explicitly accepts genuine version1 evidence without requiring missing historical run files, limits its selected epoch to the original≤4 horizon, and rejects it if adjacent artifacts advertise a named experiment. Version2 evidence must match its run/report, original recorded source hashes, selected checkpoint SHA and exact plan predictions/labels. Future analysis may read archived source hashes without substituting current ones. No checkpoint deserialization is added to analysis.

Local verification:166/166 ML tests pass with existing audited fixtures; dependency-free166 checks pass with80 optional skips. Eight new duration tests cover the config/source contract, all12 budget gate, no-update/RNG/gradient behavior, missing supervised timing batches, legacy/new evidence mixing, own checkpoint pairing and near-exhausted resume rejection before tensor/model work. Existing cache/decoder/validation tests remain green. Actual GPU throughput and all12 completed epochs are verified below. CUDA resume remains unexercised.

## Actual launch,13:13:52UTC

Private Kaggle notebook [`ozgung/knee-fresh-duration12-20261008`](https://www.kaggle.com/code/ozgung/knee-fresh-duration12-20261008), **version1**, was pushed from root revision `c5c7047676bea9e3bfccc43db50e033624308ab8` and confirmed `KernelWorkerStatus.RUNNING`. Root fixture tests166/166, default smoke and diff checks passed immediately before launch; log `outputs/research/2026-10-08/tests-duration12-integrated.log`. Quota was28.26GPUhours remaining; receipt `outputs/research/2026-10-08/kaggle-quota-before-duration12.json`. No model submission was made.

The documented build command produced a182,073-byte notebook with only `ozgung/knee-fresh-cache-20261008` attached. Local verification receipt: `outputs/kaggle/fresh-duration12/local-build-verification.json`. Experiment fingerprint `be6a0259487a33ecd76b8389d12ee728a41ef2f7554af9233172d08f0079b412`; build SHA `ce9bde00757db6a4d7b623aa1343be604714b2de9e4840657f9518f234616532`; notebook SHA `a17050a0e125d714bfb72dd978fe08ab42a1d14728b9ce5160f01766c7066028`. Preserve this saved notebook/source for any own-run resume; later source edits change the experiment contract.

Executed launch command: `~/miniconda3/bin/kaggle kernels push -p outputs/kaggle/fresh-duration12 --timeout 5700`; response saved to `outputs/kaggle/fresh-duration12/push.log`. Monitor with `kaggle kernels status ozgung/knee-fresh-duration12-20261008` and `kaggle kernels logs ozgung/knee-fresh-duration12-20261008 -f`; initial local stream session23098 writes `outputs/kaggle/fresh-duration12/live.log`. Stream or persisted-output failures do not establish a worker failure. Use backoff rather than duplicate launches. If stopping becomes necessary, use the saved version's running controls in Kaggle; no cancellation session ID has been inferred.

Expected remote artifacts are under `fresh_reader_training/`: `gpu-gate.json`, `run-contract.json`, `training-report.json`, `best.pt`, `last.pt` and `best-weak-validation-private.json`. Retrieve the small receipts and selected best checkpoint/evidence first; retain optimizer/last checkpoint remotely unless a matching own-run resume is needed. Resume requires this exact source, matched best/last and both run/report receipts, with prior elapsed charged to the same5,400-second budget. The timing gate passed on the actual GPU:712 cache files/1,346,975,371bytes verified in13.7185seconds; initialization completed at33.8009seconds. The exact16warmup/48timed supervised batches completed in73.1673seconds, projecting183.1453seconds per epoch with the1.5margin, so all12epochs qualified. All12 epochs subsequently completed; final results and artifact checks follow.


## Verified result, 13:34 UTC

Kaggle reported `COMPLETE`; the application report is `PASS` with **12/12 epochs, 3,070 successful optimizer updates and two AMP overflow skips** (one in each of epochs4 and12; 3,072 attempted supervised batches). Application elapsed time was **1,058.022seconds** and peak allocated CUDA memory was **3,653,647,360bytes**. The immutable c5c7047 run used Python3.12.13, PyTorch2.10.0+cu128, timm1.0.26 and a TeslaT4. The local live stream paused during epoch7; the persisted final log and receipts confirm training continued. This was delayed log delivery, not an application failure. Stream session23098 was stopped after completion; download session24579 completed.

The minimum weak-validation BCE selected **epoch6**, as declared. The run's float32 selection BCE is0.4883718789; the paired analysis recomputed0.4883718437 from saved logits. Both refer to the same checkpoint. Its SHA256 is `cd009bbd000bb80fd5a40e3a59b5b310cf4c93eab24d8e8b7d36b57848e29245`. Exact builder/run/report/evidence experiment bindings, source hashes, generic encoder SHA, checkpoint SHA, configuration/target order and finite checkpoint tensors passed local verification. The selected checkpoint was loaded with `weights_only=True` and kept private. Last/optimizer state remains remote and was not independently deserialized.

| Fixed comparison | Weak BCE ↓ | All12 weak AUC ↑ |
| --- | ---: | ---: |
| Fresh duration12, selected epoch6 | 0.48837184 | 0.80396701 |
| Fresh duration4, selected epoch4 | 0.49586048 | 0.80547772 |
| Frozen DINO single, same200 | 0.44760065 | 0.85756882 |
| Frozen DINO four-head, same200 | 0.44628505 | 0.86033998 |

[Verified learning curve](../../outputs/analysis/fresh-duration12/learning-curve.png).

Duration12 minus duration4 is BCE−0.00748864/AUC−0.00151071. Thus longer training gives a small BCE improvement without an AUC improvement. After epoch6, validation BCE worsens to0.51986557 at epoch12 while fixed train64 BCE falls0.37193915→0.27948868, consistent with overfitting this training cohort. A later epoch's AUC does not replace the BCE-selected checkpoint.

The existing paired analysis passed exact200-row alignment and both2,000-draw bootstrap schemes, with no invalid draws. Report-group fresh-minus-four-head95% intervals are BCE **+0.04208679 [0.02287838, 0.06454345]** and AUC **−0.05637297 [−0.08970132, −0.03090177]**. These condition on selected models and reused weak development labels; frozen baselines trained on2,000 rows versus512 fresh. They do not establish independent clinical validation or public-score performance.

The prospective larger-data spending gate **fails both requirements**: selected BCE must be≤0.456285 and AUC≥0.850340. No further duration run, larger-data fit, gold/sealed evaluation, or model submission was launched. Reassess the recipe before additional training; this completed cosine run cannot be extended as a resume.

Selected-epoch target support is explicit below. All12 AUCs are defined and the primary macro includes all12; unknown labels are masked. Synovitis remains particularly sparse.

| Target | Known | Positive | Negative | Weak AUC |
| --- | ---: | ---: | ---: | ---: |
| ACL | 188 | 38 | 150 | 0.670702 |
| MCL | 186 | 31 | 155 | 0.737565 |
| Medial Meniscus | 189 | 88 | 101 | 0.771152 |
| Lateral Meniscus | 185 | 51 | 134 | 0.798507 |
| Medial OA | 170 | 69 | 101 | 0.863395 |
| Lateral OA | 161 | 57 | 104 | 0.869096 |
| PF OA | 170 | 83 | 87 | 0.872871 |
| Effusion | 187 | 111 | 76 | 0.894737 |
| Synovitis | 28 | 20 | 8 | 0.662500 |
| Baker's | 77 | 40 | 37 | 0.760811 |
| Contusion | 163 | 42 | 121 | 0.886659 |
| Fracture | 123 | 12 | 111 | 0.859610 |

Private selected artifacts: `outputs/kaggle/fresh-duration12-result/fresh_reader_training/`. Full final log: `outputs/kaggle/fresh-duration12/final.log` and the downloaded kernel log beside that result folder. Aggregate verification, per-target metrics and complete epoch trace: `outputs/analysis/fresh-duration12/verification.json`; paired intervals and integrity hashes: `outputs/analysis/fresh-duration12/paired-selected.json`.

Executed analysis:

```bash
.venv/bin/python -m knee.paired_validation \
  --plan outputs/kaggle/fresh-cache-result/fresh_reader_cache/plan-private.json \
  --fresh-evidence outputs/kaggle/fresh-duration12-result/fresh_reader_training/best-weak-validation-private.json \
  --baseline-dir outputs/experiments/dino-frozen-20261008/artifacts \
  --comparison-dir outputs/experiments/head-comparison-20261008/comparisons \
  --output outputs/analysis/fresh-duration12/paired-selected.json
```

Post-run quota at13:39UTC: **27.96 GPU hours remaining** (2.04/30 used), TPU20hours unused, provider refreshOctober10 at00:00. Receipt: `outputs/research/2026-10-08/kaggle-quota-after-duration12.json`. No active training or local log stream remains.
