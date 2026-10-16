# A‴ private offline rehearsal preparation

October 16, 2026, 14:19 UTC. Implements Claude's13:05 request under the [frozen scope](na-full-aprime-swap-scope-20261016.md). **Ready for Claude's rehearsal launch; no GPU qualification or entry claimed.** [Acceptance](../../outputs/analysis/na-full-aprime-rehearsal-20261016/acceptance.json).

Packet: `outputs/kaggle/nafull-aprime-rehearsal-20261016/`, kernel/title `ozgung/nafull-aprime-rehearsal-20261016` / `nafull-aprime-rehearsal-20261016`. Private, internet disabled, NvidiaTeslaT4, original offline dependencies and competition attachment. No timeout option; inherited nine-hour alarm remains.

| Binding | Identity |
| --- | --- |
| Source revision | `66b26506677a40c1f7f10c01f6701c5cfa9b7d8f` |
| Notebook SHA256 | `3a15def2d7c2b1300e022a9522698619b6afee1ac146222f3c7a2f8d1fbd6f7d` |
| Payload SHA256 | `8472a05d0c47965776eb27b8f51d4d6967cfceff2ffb81490b196cca24702d97` |
| Build receipt SHA256 | `0c9dce1bd735f5e4efe1c9199b0e86fdf3bbf675e692e43f10bb9c4d56dff016` |
| Manifest SHA256 | `01f650bb83e5282ae63e729896ad0f0e10299bbe70c6ae559d4fa9a1becee341` |
| Notebook size/source files | 728611 bytes /250 |

Na-full uses final private dataset **version1**, checkpoint c5cda6ab…/292826627bytes, strict CPU receipt b80f59ed…. [Export, transport recovery and CPU evidence](na-full-export-acceptance-20261016.md). Blend remains `.5*(.65 rank(na-full K62 one-view)+.35 rank(R224))+.5 rank(R7)`, with unchanged whole-population recovery and R224 CSV precision. `old320` is only an inherited ledger key for the new na-full arm, explicitly mapped in the binding.

The new two-arm facade reuses R7 worker/run/merge code and the B0 K62 old-worker code with new checkpoint bindings. The outer facade preserves the R224 worker and launcher. Every source already in A′'s archived payload and its top-level R224/R7 input configuration remains byte-identical. Original shared `raptor_input.py` is pruned for A′; the full frozen B0 reader9dee404a… lives separately in `public_nafull_raptor_input.py` and is aliased only within the na-full worker process, with restoration on failure. Its four common function ASTs equal A′'s original definitions. No shared reader or old320 gate was globally relaxed.

Actual saved archived R7/R224 all39 genuine ledgers, input hashes and unchanged states were reconciled as references. R7 probability tolerance remains1e-3 for independently compiled kernels; R2241e-6, within-board strict reload1e-6. The earlier combined projection FAIL is preserved as ancestry. New na-full probabilities have no historical weight reference; strict-source/reload, both-board parity and reader/count checks apply.

Checks: seven focused tests PASS in fastgen; default dependency-free smoke PASS; actual strict CPU K62 check PASS separately. Deterministic builder replay, all250 source compilation/import closure, source/config preservation, synthetic arithmetic at39/100/1300 with union-cap negatives, counterfactual base/outer checker positives and negatives, and mocked bootstrap source extraction/model mounts/offline install arguments PASS. Synthetic checker receipts are temporary test inputs, never real qualification evidence. No new pydicom reader test in fastgen because pydicom is absent; exact frozen B0 reader reused, and its16-case plus guarded10-case tests remain mandatory in the fresh pinned rehearsal.

Verify the immutable preparation without rebuilding or launching:

```bash
/Users/ozgun/miniconda3/envs/fastgen/bin/python scripts/build_public_nafull_aprime.py \
  --cpu-receipt outputs/analysis/na-full-export-acceptance-20261016/cpu-k62.json \
  --verify-build outputs/kaggle/nafull-aprime-rehearsal-20261016
```

Claude owns actual account/job reconciliation and launch (`kaggle kernels push -p outputs/kaggle/nafull-aprime-rehearsal-20261016`, no timeout). Do not relaunch an existing run. Expected result root is `/kaggle/working/nafull-aprime/`, with `base/` old/R7 and two crop224 ledgers. Record the accepted server notebook version/source identity and terminal result. Parent acceptance must replay saved worker receipts/ledgers, exact39 coverage, reader/source/state/sentinel and R7/R224 parity, final all-population ranks/CSV, resources, fresh≤8.3h projection and actual<9h. Preserve failures and no inherited waiver. No competitive entry builder or submission has been produced; a qualified result is the next gate.

Temporary implementation worktree removed after integration/evidence preservation; no matching broad-test process remained. Recovery branch `codex/na-full-aprime-rehearsal` retained. No full-suite pass is claimed.
