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
