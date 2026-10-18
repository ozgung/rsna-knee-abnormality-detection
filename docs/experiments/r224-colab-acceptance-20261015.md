# R224 integration and Colab acceptance — October 15, 2026

Prospective Codex contract, frozen before a new inference run. The current task is A′ = A with R224: `0.5 * (0.65 * rank(R384) + 0.35 * rank(R224)) + 0.5 * rank(R7)`. This is a public-component integration, not independent model validation. Claude owns the separate Colab T4 execution through his existing CLI; Codex owns the immutable package and targeted acceptance. No execution into Claude’s sagittal runtime.

## Current source and arithmetic

The downloaded latest heliosli notebook has SHA256 `85c694c903ec6841488b49d5ecbe2838b8fad7db283f1f952c59f48d936e0b24`; the [research scan](../research/public-0957-scan-20261015.md) identifies its origin. Static source inspection at lines 1233–1235 and 1534–1550 of the downloaded text shows that the final blend uses **raw R224 probabilities**, written from float32, with pandas average percentile ranks. This supersedes the October 13 ignored WS2 protocol’s first argsort rank followed by a second percentile rank; that older approach breaks raw ties and is not the new primary. Preserve the old protocol and partial files as historical/unaccepted work. Test ties and the exact current arithmetic before freezing the new build.

Checkpoint: nartaa/rsna-knee-publication-swa-weights-20261007 version1, `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt`, 292,842,755bytes, SHA256 `3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6`. Strict restricted deserialization into native224 architecture; no pretrained download, grid interpolation or384-model substitution. Image reader uses guarded384 uint8 volume before any old320 crop: 307 centre crop at offset38, bilinear224 with align_corners=False, then ImageNet normalization, K94. One-view fp16 is the primary; mirror is deferred to a separately bound prospective variant.

Keep frozen A’s source closure, reader, R384 anatomical mirror, R7 assets/stackers/compile and numeric conversion. Do not substitute expanded current training modules for the archived inference closure. Preserve explicit whole-arm typed data-local masks and global genuine-arm means, union floor(0.03*N) across all three arms, complete canonical population ranks and exact IDs. Unknown, programming, resource and platform failures are fatal; no optional R224 timeout, unprocessed-row fallback or probability-based failure identification. This recovery deliberately differs from the publisher’s per-model success ranks and other-arm substitution.

## Colab package and execution

Provide an immutable source manifest, resolved config, private frozen39 input registry and command/script interface. Image inputs stay private and are staged on the VM through approved access; no raw corpus relay through the laptop. Use the same actual39 images/order for A and A′, fresh worker output directories, the same frozen precision/compile/mirror settings and package/environment pins. Synthetic timing cannot stand in for real reader or model timing. Frozen R7 requires Python3.12, torch2.10.0+cu128 and timm1.0.26; report an actual environment mismatch rather than weakening its gate.

Measure the complete back-to-back passes and each setup, staging/decode, model load/compile, production, repeat/reload, merge and shutdown interval. Save physical GPU identity, host/GPU memory and disk peaks/free space, complete image-reader/output/state ledgers and private raw probabilities. Check all39 coverage, exact source/checkpoint/input/state bindings, CPU/CUDA crop parity, repeat/reload and unchanged A-member outputs with the existing tolerances. Preserve failures, logs and recovery/stop instructions; stop the owned diagnostic VM after collecting and verifying its evidence.

## Runtime estimates and qualification

The archived passing A v7 conservative projection is 26,553.01589682397seconds (7.375837749h); its conservative policy includes 25% more R7 windows and15% shard imbalance. A Colab paired ratio `historical_A_conservative * (Colab_Aprime / Colab_A)` may be reported as a **ratio-calibrated estimate** against the unchanged8.3h gate. Record both actual elapsed values, included phases and state/order/environment bindings. Do not label that formula a measured Kaggle combined projection.

A single-T4 ratio does not mathematically cancel platform differences: two concurrent readers change CPU/disk pressure; one-time setup/compile and per-study production scale differently; compilation and session speed varied across A’s real sessions. Report these transfer assumptions. Use measured phase/production evidence and sensitivities, not just a caller success flag or subtraction from historical totals.

A successful Colab run establishes its own mechanics/parity and gives runtime evidence. It remains `PENDING_KAGGLE_QUALIFICATION`. Before a pure competition entry, use the actual remaining Kaggle quota for the short fresh2×T4 offline combined rehearsal/commit check: exact boards, package, all39 images, all three worker states/ledgers, unchanged A parity, full resource measurement and the frozen projection≤8.3h. Verify actual quota rather than automatically waiting for October17. Actual9h termination is fatal. No assistant-chosen experiment-hour cap is added.

The [earlier own-arm preparation checker](combined-rehearsal-harness-20261014.md) applies to target-specific qualified sagittal additions. It cannot qualify this global R224 rank change. Entry generation must bind the actual new complete rehearsal and strip private fixtures/IDs from the pure payload. Scientific generalization remains unestablished; a public displayed tie-or-better decision follows official COMPLETE only.

## CPU acceptance and current handover

October14 22:23 UTC preparation: six new Python files implement the native224 loader/window path, fixed three-arm recovery, actual worker/rehearsal, immutable archived-source notebook builder/result checker, paired one-T4 Colab runner and regression tests. Parent independently passed **11 installed-ML regressions/zero skips**, **4 dependency-free regressions/one optional ML-class skip**, default smoke and torch/CPU smoke. The model constructor test uses a fake timm backbone; real checkpoint/grid loading and CUDA remain pending.

Parent verified the actual preparation package against the pinned v7 notebook and build receipt: all237 archived source files unchanged, exactly3 new payload modules, all240 source files parse, complete private36+3 input registry, and all7 package file hashes. The10 protected full worker/builder files remain byte-identical. [Parent acceptance receipt](../../outputs/analysis/r224-integration-20261015/parent-acceptance.json), [static tie-policy check](../../outputs/analysis/r224-integration-20261015/rank-policy-reconciliation.json).

Immutable package: `outputs/r224-colab/package/`; manifest SHA256 `866b92b441b5f24f565f22b844e2fdb5d9ff0bb9d0e8a1c4a568f44204c1169c`, notebook SHA256 `e1074b19128cf5dfd84e6cd21e184a6560e1df1efed38b5a54c9f0231e3ba8e6`. These are **PREPARATION_ONLY**, not inference qualification. The manifest and `input-requirements.json` bind the external assets, exact private39 UID/train-versus-test registry and pinned runtime; no raw image/weight bytes are packaged.

Claude’s launch after VM-side manifest/input verification and environment setup:

```bash
python run_r224_diag.py --inputs /content/r224_inputs --out /content/r224_out
```

`/content/r224_inputs/inputs.json` follows the packaged `inputs.example.json`. Actual A and A′ run separately in fresh output directories, each restaging/decoding all39; the runner saves both passes’ outputs/ledgers/resources, compares preserved A member inputs/state/probabilities and computes a separately measured timing ratio. Success is `DIAGNOSTIC_ONLY_PENDING_KAGGLE_QUALIFICATION`. Results return to `outputs/r224-colab/result/`; Claude owns the session/log/stop record.

The standalone full Kaggle evidence checker requires the exact preparation build and pinned archive before reading actual fresh six-worker ledgers:

```bash
python scripts/build_public_ws2_entry.py --verify-result outputs/r224-kaggle-result/ws2-a-prime --rehearsal-build outputs/r224-colab/package --archive-rehearsal outputs/kaggle/public-r7-recovery-rehearsal-20261013-v7/notebook.ipynb --archive-rehearsal-receipt outputs/kaggle/public-r7-recovery-rehearsal-20261013-v7/build-receipt.json
```

The result path above is a future example, not an existing qualified result. The checker replays captured actual setup/merge inputs, exact frozen870-window distribution and both frozen-base/combined projections; arbitrary PASS/gate flags and foreign preparation bindings are insufficient. Pure entry building still requires this real fresh qualification.

## Colab T4 run 1: failure and pre-registered v2 repair (Oct 15, 00:17 UTC, Claude)

**Run:** `r224-t4`, Tesla T4, 22:27–00:16 UTC. Package v1 (manifest 866b92b4…) ran unchanged on Python 3.12.13. Python 3.12.3 had failed first, on the mirror source-transform hash, which depends on the Python patch version. **Exit 1 with `ONE_T4_WORKER_crop224_FAILED`.**

- **Pass A completed:** 39/39 studies, 2,483.8 s in total (old 989.9 s, s6 1,491.4 s).
- **Pass A′:** old (984 s) and s6 (1,292 s) completed. crop224 failed 12 s in, on the first sentinel, with `CROP224_CPU_CUDA_PARITY`.
- **Measured CPU/CUDA difference** on the 94×3×224×224 float32 window tensors, in the range −2.12 to 2.64:

  | Sentinel | max_abs | mean_abs | Elements > 1e-5 | Elements > 1e-4 |
  |---|---|---|---|---|
  | 0 | 1.12e-5 | 7.3e-8 | 1.3e-6 | 0 |
  | 1 | 2.16e-5 | 5.1e-8 | 2.1e-5 | 0 |

  This is float32 rounding between the CPU and CUDA bilinear kernels, not a crop or offset bug.
- **Peaks:** host memory 7.66 GB, GPU 6.16 GB allocated, lowest free disk 181.9 GB.
- **Evidence:** `outputs/r224-colab/result/` (ignored).

**Pre-registered v2 repair:** the only code change is the crop parity tolerance.

- **New rule:** `max_abs ≤ 1e-4` **and** `mean_abs ≤ 1e-6`, applied in the worker (`knee/public_ws2_entry.py`) and in the result checker.
- **Unchanged:** production windows stay on CUDA. Every other gate stays as it is, including the reload probability tolerance of 1e-6, the source and state bindings, the 8.3 h gate, and failures being fatal.
- **Rebuild:** binding version `ws2-a-prime-r224-v2` into a new immutable `outputs/r224-colab/package-v2/`. Package v1 is preserved.
- **Rerun:** run the v2 package on a fresh T4 with Python 3.12.13 and a fresh output directory. This rerun is a new run, not a resume of run 1.

## Resumed v2 qualification — 2026-10-15 08:46 UTC

Claude committed the prospective tolerance repair as df958e4. Parent independently passed12 focused installed-ML regressions/zero skips and exact static verification of both `outputs/r224-colab/package-v2/` and fresh `outputs/kaggle/ws2-r224-v2-qual-20261015/`. Their notebook bytes, inference payload, binding and all source hashes match; only the requested metadataID differs. Parent preserved237 archived sources/3 new modules. [Build receipt](../../outputs/analysis/r224-integration-20261015/kaggle-v2-preparation.json).

Claude owns the fresh Colab v2b execution and the parallel binding Kaggle qualification. The earlier v2 VM was reclaimed during staging without inference results. Parent independently lists the new Colab T4 allocation; actual staging/inference receipts remain pending.

Kaggle accepted qualification version1 at08:36 under **`ozgung/a-prime-r224-rehearsal`**, derived from the title. Use that actual slug; the requested localID does not resolve. Parent independently confirmed RUNNING and pulled exact accepted source/settings: code cell/payload match the frozen build, private/offline2×T4/pinned Docker/competition match. API dataset-ref normalization does not replace runtime asset hash checks. [Accepted-source proof](../../outputs/analysis/r224-integration-20261015/accepted-kernel-v1-verification.json). Fix title/id consistency after this frozen qualification is consumed, before a future build; do not mutate it now.

Actual GPU quota at08:34:10.82h remaining/49.18h used/60h total. Recheck before another launch. The CLI’s refreshAt field has no explicit timezone; do not infer local time. [Quota receipt](../../outputs/analysis/r224-integration-20261015/quota-resume-20261015.json).

The current full checker uses `--rehearsal-build outputs/kaggle/ws2-r224-v2-qual-20261015` and future collected result `outputs/kaggle/ws2-r224-v2-qual-20261015-result/ws2-a-prime`, with the same pinned archive arguments above. A supplementary ignored auditor `outputs/analysis/r224-integration-20261015/audit_kaggle_v2.py` adds all39 native-versus-baseline source hashes and finite/nonnegative/plausible crop-statistics checks without modifying the running code. Only its syntax has been checked; actual terminal execution is pending. No entry qualifies from preparation/flags or Colab ratio alone.

Parent’s completed v1 partial audit passed441 assertions: all240 saved runtime files exact, four completed39-row old/R7 ledgers, source/environment/state consistency and zero preserved-arm probability delta across A/A′. V1 remains FAILED with R224 completed0; no combined ratio, efficacy or qualification. Saved input hashes were audited; raw images were not locally replayed. [Partial audit](../../outputs/analysis/r224-integration-20261015/colab-v1-partial-audit.json).
