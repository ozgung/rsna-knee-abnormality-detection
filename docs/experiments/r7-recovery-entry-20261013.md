# R7 recovery entry (WS1): R7 replaces S6 in the 0.954 capped-recovery entry

Hypothesis (docs/research/next-direction-20261013.md, WS1): heliosli's public 0.957 recipe (Raptor original320 K94 mirror + R7 at fixed 0.5/0.5 per-finding percentile ranks) inside our capped-recovery runtime scores at least 0.956. Only the compact arm and the blend weights change; the old arm, recovery policy/guards, schema checks and offline mode are the qualified 0.954 code, shipped byte-identical.

## What changed

- Compact slot (still named `s6` in ledgers): heliosli R7 `heliosli/rsna-knee-compact-r7cnxfsss` pinned at dataset version 1 (the only version; manifest sha256 `c4bb0f73…`, bundle `8c7b666e27dc0e8f`), R7 `compact_infer.py`/`compact_train.py`/`probe_dual.py` taken as literals from heliosli's "0.957 latest dataset score in one click" notebook (sha256 `cb2a0f57…`, parsed with `ast`, never executed locally). All other scientific sources (training/*, px_cache, anchor bank, side model code) are byte-identical between S6 and R7 and are verified as such at build time.
- Stackers: fluid stacker adopts Effusion (D2 sagittal volume, `eff_sag_p99_open`), state stacker adopts Baker's (z + six state probabilities, no volume). Ported to `knee/public_r7_stackers.py` with the source's exact numpy/math expressions; on Kaggle the worker cross-checks the port against the pinned source (`fluid_stack` exact equality, loaded coefficients and layout equality).
- Blend: 0.5/0.5 on every finding (heliosli's OA override is also 0.5); R7 values quantized with the R7 worker's `'%.9g'`. `knee/public_r7_recovery.py` is the frozen recovery algorithm with only weights/precision parameterized; a test proves it reproduces the frozen 70:30 function bit for bit.

## Declared deviations from heliosli's notebook

1. torch.compile applied in place (`nn.Module.compile`, same inductor mode, shapes, 1e-3 gate and eager fallback) so state digests/strict reloads are unchanged; the check runs on the first pilot sentinel. v1 rehearsal ran eager (see below).
2. Non-finite fp16 logits and stacker exceptions are fatal (heliosli: fp32 rescue, keep model probability). Non-finite stacker logits keep the model probability, as in the source.
3. No 8.3 h deadline guard. Its fallback (unreached studies take the other model's rank, up to 50%) is incompatible with the frozen 3% capped arm-mean recovery; the 9 h alarm and the rehearsal projection gate remain.
4. Failure recovery: capped global arm means ranked over all rows (heliosli: each model ranked over its own scored rows). Identical when nothing fails.
5. Encoder chunk 192 (compact_infer default; the S6 entry used 16). Our pinned Docker image/environment, not heliosli's. Submission values written as float64 (heliosli casts to float32).
6. R7 state digest is measured on both rehearsal boards and bound into the entry (no CPU pin, because downloaded R7 code is not executed locally).

## Results

Rehearsal v1 (eager; `ozgung/knee-public-r7-recovery-rehearsal-20261013` v1, 2×T4, 29.7 min): all 39 studies genuine on both arms, both boards identical R7 state, pilot-input parity with the S6 pilot on all 32 probes, stacker recomputation, fluid self-check (cv2 4.13.0). Visible-3 parity with heliosli's published output: Raptor arm max |Δp| 8.3e-5, R7 arm max |Δp| 2.3e-4 (OA/Fracture; compile vs eager), blend max |Δ| 3.3e-8 (float32 rounding), i.e. per-target rank difference 0 on all 12 targets. **Failed** the frozen nine-hour gate: projection base 6.89 h, conservative 9.46 h (S6 entry: 5.97 / 8.19 h). R7 GPU time 16.6 s/study vs S6 15.1 s; the unchanged old arm was also ~10% slower in this session.

Rehearsal v2 (v2 of the same kernel, 32.8 min, status PASS): every gate passed; identical parity (blend rank Δ 0 on all 12 targets; R7 2.3e-4; Raptor 8.3e-5); R7 state `f90ddf4b…` on both boards; fluid ok 36/39 (3 without a fluid-sensitive sagittal series take the source's missing path); zero stacker fallbacks. Compile did **not** activate: the receipt update passed `study_dprob` twice (TypeError), so both workers fell back to eager after ~300 s. Projection base 6.57 h, **conservative 8.993 h** (gate < 9 h), i.e. almost no margin; the wasted compile time is inside the measured overhead. Entry `public-r7-recovery-entry-20261013-v1` is built from this exact rehearsed code and pushed as kernel `ozgung/knee-public-r7-recovery-entry-20261013` v1 (not submitted). Coordinator decision: do not submit entry v1 (margin too thin).

Rehearsal v3 (compile receipt fixed and regression-tested; R7 contract tightened to require active compile with no fallback on both workers and conservative projection ≤ 8.3 h): compile active on both workers (check decisions within 1e-3; max |dp| 2.4e-4 / 3.6e-4), R7 GPU 11.6 s/study (eager 16.0 s), **conservative projection 7.87 h** (base 5.80 h). Visible parity: blend rank Δ 0 on all 12 targets; R7 2.6e-4; Raptor 8.3e-5. **Failed** `CROSS_BOARD_SENTINEL_PARITY` (1e-6): independently compiled graphs differ across boards by up to 2.6e-4 in probability (logits 7.9e-3), while within-board strict reload stays bit-exact and the old arm is exactly equal across boards. An offline check with only that R7 tolerance set to 1e-3 passes every other gate on the v3 receipt. Not shipped; relaxing that gate needs a decision and a v4 rehearsal.

Coordinator decision (option 1): R7 slot cross-board sentinel tolerance 1e-3 (heliosli's compiled-vs-eager tolerance); within-board strict reload and the old arm stay at 1e-6. This tolerance does not guarantee hidden rank parity: shards compiled separately can swap close ranks. The evidence is the measured visible blend rank difference of 0, not a claim about AUC.

Rehearsal v4 (`…rehearsal-20261013` v4) is a diagnostic only: Codex's acceptance review found two blockers in that code. (1) An opaque `fluid_d2` exception would score Effusion on the missing path outside the recovery cap. (2) Production could continue eager after a compile failure, which projects to 8.99–9.46 h.

Fix (qualifying build v6; v5 was never built or pushed):
- `fluid_d2`'s catch-all is AST-replaced by an allowlist boundary. The recovery's typed data-local classes (DataLocalFailure, registered geometry/decode data classes, InvalidDicomError, audited pixel-buffer error) become a counted, capped study failure. Every other exception, including resource or programming causes anywhere in the chain, is fatal.
- An `error:` status is then fatal (it means the boundary was not installed). The source's no-fluid-series and no-region statuses keep the missing-volume path.
- Production now requires compiled encoders: a failed compile check or a failing compiled forward is fatal before or at inference, with no eager continuation.
- v6 runs under its own slug `ozgung/knee-public-r7-recovery-rehearsal-v6-20261013` so the v4 diagnostic can finish. No retroactive PASS for v4/v5.

Codex then asked for documentation-only fixes: the fluid-gate docstring, and compile metadata stating no eager fallback in production. Source bytes changed, so v6 (already pushed) is a diagnostic and the qualifying rehearsal is v7.

| Rehearsal | Code | Result | Conservative / base projection (h) | Visible blend rank Δ | R7 / Raptor vs heliosli |
|---|---|---|---|---|---|
| v4 (diagnostic) | tolerance only | FAIL `R7_HELIOSLI_HORIZON_PROJECTION_GATE` | 8.97 / 6.63 | 0 | 2.6e-4 / 8.3e-5 |
| v6 (diagnostic) | + boundary, compile contract | PASS | 8.09 / 5.96 | 0 | 2.6e-4 / 8.3e-5 |
| **v7 (qualifying)** | + doc-only fixes | **PASS** | **7.38 / 5.44** | **0** | 2.6e-4 / 8.3e-5 |

- **Compile:** active on both workers in v4, v6 and v7, never with a fallback. R7 GPU time 10.4–11.3 s/study.
- **v4:** session-level host slowdown. The unchanged old arm took 144 s against 115 s, and R7 production 536 s against 480 s. Across four compiled sessions the conservative projection ranged 7.38–8.97 h, so session variance is material against the 9 h limit.
- **v7:** `ozgung/knee-public-r7-recovery-rehearsal-v7-20261013` v1, 29.5 min.
  - All 39 studies genuine; R7 state `f90ddf4b…` on both boards; R7 cross-board sentinel |Δ| 0.0.
  - Fluid: ok 36, no fluid-sensitive series 3; no stacker fallbacks.
  - Peak GPU 5.8 GB per board; peak host RSS 15.8 GB.
  - Rehearsal notebook `712ab44f…`. Frozen sources in `outputs/analysis/r7-recovery-20261013/frozen-v7-sources/`.
- **Entry v2:** `public-r7-recovery-entry-20261013-v2`, built from exactly the v7 bytes and gated by the v7 contract plus measured visible blend rank Δ 0. Notebook `be3e6c06…`, pushed as v2 of `ozgung/knee-public-r7-recovery-entry-20261013`. Not submitted.
- **Entry v2 visible run:** COMPLETE, receipt PASS in 8.7 min.
  - Compile active on both workers, no fallback; R7 state `f90ddf4b…`; zero recovered rows.
  - `submission.csv` (sha256 `05bb8080…`): header equals `sample_submission.csv`, IDs equal `test.csv` in order and unique, 12 finite values in [0, 1] per row.
  - Byte-identical to the v7 rehearsal visible CSV.
- **Entry v1** (eager): visible run COMPLETE; superseded, not to be used.

## WS3: selection-B probes without OAI-trained weights

**Provenance:** R7's fastMRI auxiliary training and the licence permission of its weights are unresolved (`outputs/analysis/r7-recovery-20261013/provenance-primary-sources.md`). B1 and B2 hedge only against an OAI ruling, not against an R7/fastMRI ruling.

Both probes reuse the v7-frozen R7 code, with the builder checking the pinned bytes. No nartaa (OAI-trained) dataset is attached, and the boot fails if one is mounted.

- **B1 (R7 alone):**
  - Only the R7 arm is launched; the Raptor weights are neither attached nor run.
  - Recovery is the frozen capped arm-mean policy on the single arm: at most floor(3%) classified rows take the R7 genuine mean ('%.9g').
  - The submission is the R7 average percentile rank (weight 1.0), a monotone transform of the filled R7 probabilities, so AUC equals submitting those probabilities.
  - Datasets: R7 v1, orthofoundation (code and anatomy), goodpjw2008 (offline codec wheels).
- **B2 (Dread + R7, 0.5/0.5):**
  - The B0 Dread CC0 Raptor v4 K62 old-arm worker, byte-identical to entry 57119585, plus the v7 R7 worker.
  - Default merge/run core; frozen 0.5/0.5 recovery at '%.9g'.
  - Datasets: R7 v1, orthofoundation, goodpjw2008 and `dreaddevelopment/raptor-knee-widedense`.
- **Qualification gates:**
  - The v7 R7 gates: compiled with no fallback, fluid boundary, conservative projection ≤ 8.3 h, R7 cross-board tolerance 1e-3, and visible R7 parity with heliosli within 5e-3.
  - B2 also has the B0 old-arm gates (raptor62 visible receipt parity, strict pydicom and reload repeats; old cross-board 1e-6).
  - There is no published reference for these blends. The entry build therefore requires the rehearsal's visible R7 arm to match entry v2's R7 arm within 1e-3 (reference `r7-entry-v2-visible.json`, sha256 `09c3c3e0…`).

| Probe | Rehearsal (2×T4, 39 studies) | Conservative / base projection (h) | R7 vs entry v2 / heliosli | Entry visible run |
|---|---|---|---|---|
| B1 | `ozgung/knee-r7-only-b1-rehearsal-20261013` v1, PASS, 19.7 min | 6.34 / 4.47 | 1.0e-4 / 2.6e-4 | `ozgung/knee-r7-only-b1-entry-20261013` v1 COMPLETE, receipt PASS, CSV checks pass, CSV equals rehearsal |
| B2 | `ozgung/knee-r7-dread-b2-rehearsal-20261013` v1, PASS, 25.7 min | 7.21 / 5.29 | 1.0e-4 / 2.6e-4 | `ozgung/knee-r7-dread-b2-entry-20261013` v1 COMPLETE, receipt PASS, CSV checks pass, CSV equals rehearsal |

- **B1:** only the R7 worker ran, compiled on both workers with no fallback. R7 state `f90ddf4b…`; cross-board R7 Δ 0.0.
- **B2:**
  - The Dread K62 old arm took 5.0 s/study, with Dread state `e0f0e7e1…` and raptor62 visible parity inside the contract.
  - R7 compiled with no fallback; cross-board R7 Δ 5.2e-5.
- Frozen selection-B sources: `outputs/analysis/r7-recovery-20261013/frozen-selb-sources/`. Not submitted.

## WS5: B3 = B2 + own dr-full at 0.2 of the Dread slot (not qualified)

**Inference contract.** dr-full's trainer/export eval path equals the B0 Dread worker apart from K:
- the same `knee.raptor_input.prepare_study` 64-slot volume (cache built with the audited Dread reader);
- the same deterministic eval centres;
- the window is divided by 255, resized 336→384 bilinear with `align_corners=False`, and ImageNet-normalized;
- the original RaptorClassifier head; SWA checkpoint `{model, arch, res, lab}`.

So the dr arm runs the B0 old-arm worker bytecode with only its `'old'` string constant replaced by `'dr'`, at K42, with dr-full pins.

**CPU qualification** (`scripts/qualify_b3_drfull_cpu.py`; fastgen Python 3.12.3, torch 2.10.0):
- strict load into the audited Dread RaptorClassifier;
- state `a3dbd31b…`; grid `67ae62b6…`;
- finite repeatable K42 outputs; SWA epochs [9, 10, 11];
- the same method reproduces B0's CUDA Dread pins.

**Blend.** The frozen capped recovery generalized to three arms: 0.5 × (0.8 rank Dread + 0.2 rank dr-full) + 0.5 rank R7 = 0.4 / 0.1 / 0.5. The dr-full dataset `ozgung/knee-raptor64-dr-full-fold0-final-20261013` is attached privately, pinned at version 1.

**Rehearsal** `ozgung/knee-r7-dread-drfull-b3-rehearsal-20261014` v1 (37.3 min): **FAIL `NINE_HOUR_PROJECTION_GATE`**.
- **Projection:** conservative 9.48 h, base 7.13 h.
- **Functional gates:** all 39 studies genuine on all three arms; dr-full CUDA state equals the CPU pin on both boards; R7 compiled with no fallback; R7 cross-board Δ 9.5e-6; R7 vs entry v2 1.0e-4.
- **Measured production (39 studies, slower board):**

| Arm | Production | Per study | Phase overhead |
|---|---|---|---|
| Dread K62 | 142.5 s | 7.0 s | 385 s |
| dr-full K42 | 77.5 s | 3.8 s | 286 s |
| R7 | 512.5 s | 25.3 s | 808 s |

- **Session speed:** this session was slow; the same Dread arm took 102 s in the B2 rehearsal.
  - Removing dr-full's measured terms gives a B2-equivalent of about 8.57 h in this session, against B2's own 7.21 h.
  - dr-full adds about 0.9 h conservative here (0.83 h production + 0.08 h overhead).

**Sanity.** dr-full vs Dread per-target Spearman over the 39 rehearsal studies is 0.83–0.96 (Fracture 0.83, Lateral OA 0.84, MCL 0.89, … Medial Meniscus 0.96). Visible-3 probabilities are recorded in `b3-sanity.json`.

Not shipped; the options are reported to the coordinator.

Evidence (ignored): `outputs/analysis/r7-recovery-20261013/` (pinned R7 v1 assets, heliosli visible output, build arguments, push/status logs, summaries); builds and results under `outputs/kaggle/public-r7-recovery-*-20261013-*`.
