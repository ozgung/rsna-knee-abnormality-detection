# Lane 3b protocol: output-preserving R7 prefetch for A′ runtime headroom, 2026-10-14

Pre-registered 2026-10-14 at about 16:45 UTC, before any implementation. Author: Fable (Claude), strategy. Replaces the finished lane 3 ([result](../research/rank-value-selection-20261014.md)). **Owner of execution: a Claude implementation agent** (this modifies a qualified entry worker, so not a routine cheap-model task). **Codex keeps entry integration, build pins, rehearsal launch and acceptance**, under its 15:24 UTC terms. Lane 1 (axial PF arm), lane 2 (sagittal joint crop) and Codex's M4 ROI v2 are untouched: this lane changes no model, no blend, no arm, no data path and no Kaggle GPU quota before the Oct 17 refresh. Labels: **[measured]** = number in a receipt or repo file; **[literature]** = source code or another participant's text; **[estimate]** = my judgement.

Inputs: v7 qualifying rehearsal receipt `outputs/kaggle/public-r7-recovery-rehearsal-20261013-result-v7/s6-recovery-entry/receipt.json`; the v1–v6, B1, B2 and B3 rehearsal receipts under `outputs/kaggle/*rehearsal*result*`; `knee/public_r7_recovery_entry.py` (worker loop, lines 349–400); `knee/public_s6_recovery_entry.py` (`launch`, `projection`, lines 518–570); heliosli's pinned `compact_infer.py` (`CStudies._setup`, `__getitem__`, `main`); [R7 recovery entry note](r7-recovery-entry-20261013.md); [B1 hidden-error diagnosis](b1-hidden-error-diagnosis-20261014.md); [plan of record](../research/plan-update-20261014.md) step 2.

## Why this lane and not the alternatives

Lane 3 priced a true +0.001 on A at about 356 places and showed that hidden slots go to A′. The only live sources of gain are the lane 1 and lane 2 arms (+0.0004 to +0.0013 and +0.0003 to +0.0010 macro [estimate]), and both can enter A′ only if A′ passes the 8.3 h conservative gate on 2×T4. That gate is currently a coin flip on host speed, not on our code:

| Compiled session, identical v7-class code | Conservative projection | Headroom vs 8.3 h |
|---|---|---|
| v7 (qualifying, A entry v2 built from it) | **7.38 h** | +0.92 h |
| v6 | 8.09 h | +0.21 h |
| v3 | 7.87 h | +0.43 h |
| v4 (slow host) | **8.97 h** | **−0.67 h** |
| B2 (Dread + R7) | 7.21 h | — |

[measured; the plan's "7.87 h / 0.4 h headroom" is the v3 figure]. The two arms add roughly 0.4–0.8 h conservative together (5–15 T4 minutes each, two boards, ×1.44 projection multipliers, plus load overhead [estimate]). Without this lane, P(A′ with both arms passes the gate) is about the share of fast sessions, 0.4–0.5 [estimate].

| Candidate | Verdict | Reason |
|---|---|---|
| **R7 study-pipeline prefetch (this lane)** | run | 53 % of R7 production is serial CPU work with the GPU idle [measured, below]; the fix is output-preserving by construction and restores the concurrency heliosli's own `compact_infer.main` uses (`DataLoader`, `--workers` default 2, spawn context) [literature, pinned source]. |
| A +0.001 science lane on A itself | not run | Nothing validatable: R7 changes cannot be evaluated (the 870 are exposed to R7/S6 with +0.021 inflation, lane 3 Q0); Raptor's recipe is at K94 mirror already; per-target weights are closed; the metadata prior failed its transfer gate. |
| Decode cache shared between arms | not run | Raptor's `old_public` reader and R7's `native_cache`/`px_cache` are different pinned decoders; no identity proof without a deep audit, and the Raptor arm is only 1.18 h conservative. |
| Dynamic shard assignment (remove the 1.15 imbalance term) | rejected | Changes which board scores each study; R7's separately compiled graphs differ across boards by up to 2.6e-4, so outputs would depend on timing. The 1.15 is a projection assumption, so the saving would be a gate change. |
| Shipping the inductor cache (compile ≈ 300 s per board) | rejected | About 0.08 h, fragile across drivers, and it would couple the two boards' kernels. |
| Counting only entry-relevant overhead in the projection | rejected | A projection-policy relaxation, Codex's domain, not engineering. |
| Raptor (old arm) prefetch | deferred | The 0.954 code is shipped byte-identical and qualified; 1.18 h conservative. Only if this lane passes and more headroom is still needed. |

## Measured runtime anatomy of A (v7 session, 39 studies, 2×T4, 4 vCPUs, 2 workers)

Frozen projection (`public_s6_recovery_entry.projection`): conservative = setup + Σ phase overhead + old production × 1300/39 × 1.15 + R7 production × (1300 × 577.8 × 1.25 / 22,885 windows) × 1.15, production = the slower board's per-study sum.

| Term | Seconds (v7) | Conservative hours | Share |
|---|---|---|---|
| Setup | 12.9 | 0.00 | 0 % |
| Phase overhead (old 492.8 + R7 708.3; R7 compile check 298 s) | 1,201 | 0.33 | 4.5 % |
| Old (Raptor) production, 110.9 s for 20 studies, 5.5 s/study | 4,251 | 1.18 | 16 % |
| **R7 production, 447.0 s for 20 studies, 22.4 s/study** | **21,090** | **5.86** | **79 %** |

R7 per-study time (board 0, n = 20; board 1 agrees within 0.2 s) [measured from `preprocess_seconds` = (decode, render, anchor), `four_model_seconds`, `production_seconds`]:

| Component | Mean s/study | Where it runs | Overlappable |
|---|---|---|---|
| Four-model compiled forward (`four_model_seconds`) | 10.53 | GPU, main thread | no (this is the floor) |
| DICOM decode and geometry (includes the small side-net forward) | 3.21 | CPU (side net on GPU) | yes |
| Render (`px_cache.StudySlices.build`) | 1.82 | CPU | yes |
| Anchor segmentation (SKM-TEA bank) | 1.24 | GPU (`dev = "cuda"` in `_setup`) | partly (serialises on the default stream) |
| Unattributed main-thread work: 8 SHA-256 fingerprints over about 0.5–1 GB of arrays per study, scratch `rglob`, receipt save, checks, stackers, fluid (0.07) | 5.55 | CPU, main thread | yes (fingerprints depend only on the item) |
| **Total production** | **22.35** | | |

The forward correlates 1.00 with window count (17–18 ms/window, stable at 17–22 ms across all compiled sessions); the CPU part correlates 0.4–0.75 and is the session-sensitive term: 11.8 s/study in v7, 16.0 s/study in v4 [measured]. Peak host RSS per R7 worker 6.6–6.8 GB, flat over the run; combined phase peak 15.8 GB (v7) to 18.2 GB (B1) on a 29–31 GB host [measured].

## Hypothesis

A one-ahead prefetch of the whole per-study input stage (decode, geometry, anchors, render, fluid measurement and our fingerprint hashing) for study i+1, run on a worker thread while the main thread runs the four-model forward of study i, leaves every array, fingerprint, probability and ledger field identical and reduces R7 production per study from about 22 s to about max(CPU side, GPU side) + ε ≈ 12–14 s in a v7-speed session (−35 to −45 %), and from about 27 s to 16–18 s in a v4-speed session. Expected A conservative projection: **4.7–5.6 h** in a v7-speed session and **5.8–6.8 h** in a v4-speed session [estimate], so A′ with both arms passes the 8.3 h gate in either, with 1.5–3 h to spare. P(the speed criterion below is met) about 0.7 [estimate]; the main uncertainty is CPU contention of four busy threads on four vCPUs.

## Baseline

The v7 worker bytes (`knee/public_r7_recovery_entry.py`, sha pinned in `scripts/build_public_r7_selection_b.py` `R7_FROZEN_SHA` and hashed by `scripts/build_public_r7_recovery_entry.py` `R7_FILES`), the v7 receipt above, A entry v2 (57157384) built from them, and the frozen projection policy, which this lane does not touch.

## Method (what may and may not change)

Allowed, all inside our own worker `knee/public_r7_recovery_entry.py`:
1. After the pilot sentinels (which reload models and recreate `ds`, so they stay strictly serial), the evaluation loop obtains study i+1's `item = ds[i+1]`, its `capture` contents (anchors record, built item, rows, source diagnostics, scratch bytes) and the eight `fp` fingerprints from a single prefetch thread with a queue depth of exactly one. The main thread keeps, in the existing order: `r['phase']`/`current_ordinal` and receipt saves, every `require` gate (`UNCLASSIFIED_FAILED_STUDY`, `S6_INPUT_SCHEMA`, `TRIPLET_SERIES_VIEW`, `fluid_gate`, compile on the first study, `S6_EVAL_STATE`, `NONFINITE_FP16_NO_RESCUE`, `MEAN_SIGMOID_RECONSTRUCTION`, `PORTED_FLUID_EQUALS_SOURCE`, `FINITE_STACKER_LOGITS`, `PER_STUDY_SCRATCH_NOT_REMOVED`, probe gates), the forward, stackers and the row.
2. `capture` becomes per-item (the observe hooks write into the object the prefetch stage hands over), so no state is shared between the study on the GPU and the study being prepared.
3. Any exception raised during the prefetch of study i+1 is stored and re-raised in the main thread at the point where study i+1 would have been processed, through the unchanged classification path; `failure-private.json`, the recovery cap and the fatal boundaries are unchanged. The prefetch thread holds no signal handlers; `PlatformInterruption` stays in the main thread, which joins the thread with a timeout on any exit.
4. GPU work inside the item stage (anchor bank, side net) runs under `torch.inference_mode()` in the thread; both threads use the default stream, so kernels serialise in issue order and outputs are unchanged.
5. `production_seconds` is redefined as the main-thread wall time from the previous study's record to this study's record (including any wait for the prefetch), so Σ production remains the real production wall and the frozen projection stays honest. `preprocess_seconds` and `four_model_seconds` keep their meanings; a new `prefetch_wait_seconds` field is added to each record.

Not allowed: any change to heliosli's pinned files (`compact_infer.py`, `compact_train.py`, `probe_dual.py`, trainer MD5, asset bindings), to `px_cache`/`native_cache`/anchor code, to chunk 192, fp16, compile settings or tolerances, to the Raptor worker, to the parent `launch`/`merge`/`projection`, to the blend or recovery arithmetic, or to any gate threshold. A process-based `DataLoader` (heliosli's path) is the fallback only if the thread design cannot pass the synthetic parity tests; it would need its own CUDA context in the child and a different capture transport, and is pre-registered as a lower-preference variant.

## Parity checks (pre-registered)

| Check | Where | Pass |
|---|---|---|
| P1 Synthetic serial vs prefetch | `tests/` with a fake `CStudies`, fake models and injected hooks, CPU | Identical record dicts except the timing fields; identical `fp`; identical probabilities; prefetch depth never exceeds one (counter assertion); pilots unchanged; an injected `item['error']` at ordinal k raises `UNCLASSIFIED_FAILED_STUDY` at ordinal k with the same receipt state as serial; an injected fatal exception in the item stage raises the same type at the same ordinal; a simulated SIGTERM leaves status `INTERRUPTED` and no live thread. |
| P2 Real-data serial vs prefetch, CPU soak | Kaggle **CPU** kernel (private, offline, competition data attached, no GPU quota; precedent: the ROI qualification kernels), 39 frozen rehearsal studies, two concurrent processes mimicking two boards, anchor bank on CPU, forward stubbed | All 39 × 8 fingerprints identical between serial and prefetch runs; no deadlock or timeout; combined RSS peak < 22 GB; CPU-side seconds per study in the two-process run ≤ 1.3 × the one-process run. Fingerprints equal to v7's `input` fingerprints are recorded as informational only (CPU vs CUDA anchors may differ). |
| P3 Qualifying rehearsal | Kaggle 2×T4, Codex-owned, the A′ rehearsal after the Oct 17 refresh | The unchanged v7 contract (compiled with no fallback, fluid boundary, cross-board 1e-3, visible R7 vs entry v2 within 1e-3, blend rank Δ 0, conservative ≤ 8.3 h) **plus** a new rehearsal-mode gate `EVALUATION_INPUT_PARITY_V7`: every evaluation study's eight fingerprints equal the v7 receipt's for the same study, and sentinel strict-reload probabilities within the existing 1e-6. |

## Acceptance

- **Correctness (required):** P1, P2 and P3 all pass. Any P3 parity failure reverts the A′ rehearsal to the v7 worker bytes; no second attempt in this lane.
- **Speed (the lane's success criterion):** in P3, R7 `production_seconds` / `four_model_seconds` per study, averaged over evaluation studies on the slower board, **≤ 1.40** (serial baseline 2.12 in v7, 2.45 in v4 [measured]); this is session-robust because it is normalised by the forward. Secondary: conservative projection of A′ including the admitted arms ≤ 8.3 h, and the A′-minus-arms conservative figure ≤ 6.5 h.
- **Resources:** peak host RSS per R7 worker ≤ 9.5 GB and combined phase peak ≤ 22 GB; peak CUDA allocated recorded (the anchor bank may now overlap the forward's working set).
- Correctness passing with speed failing is recorded as "output-preserving, no gain" and the worker still ships (nothing to revert); the arms then compete for the unchanged headroom.

## Kill rules

- **K0 (profile):** close if the non-forward share of R7 production in the v7 session were < 30 %. It is 53 % [measured], so the lane proceeds.
- **K1 (design):** close if the thread design cannot pass P1 without touching a pinned heliosli file, or P1 is not green by **Oct 15 22:00 UTC**.
- **K2 (soak):** a P2 failure gets one fix iteration; a second failure closes the lane.
- **K3 (schedule):** if the branch, P1 and the P2 receipt are not handed to Codex by **Oct 16 12:00 UTC**, the Oct 17 rehearsal uses the v7 worker. The lane then continues only if an arm exists and that rehearsal failed the 8.3 h gate, with a second rehearsal no later than Oct 19 and the plan's rule of no new hidden run after Oct 20 16:00 UTC.
- **K4 (rehearsal):** any P3 parity gate failure or the RSS bound breached reverts to v7 bytes; the lane is closed as failed.

## Compute and timeline

| Stage | Owner | Compute | When |
|---|---|---|---|
| 0 Receipt profile (done in this protocol; written up with the result) | Claude | local CPU | Oct 14 |
| 1 Implementation + P1 tests | Claude implementation agent | local CPU, `python3.12` ML env; checks: `python3.12 -m unittest discover -s tests -v`, `python3.12 -m knee.smoke` | Oct 14 17:00 → Oct 15 22:00 UTC |
| 2 P2 CPU soak | Claude (kernel), evidence to Codex | one private offline Kaggle **CPU** kernel, about 1–2 h wall, **zero GPU quota** | Oct 15 → Oct 16 12:00 UTC |
| 2b Optional T4 speed preview | Claude, only with the user's explicit authorisation to stage the 39 studies' DICOMs to Colab via a private Kaggle dataset (external image processing per AGENTS) | Colab T4, about 1 h | skipped by default |
| 3 P3 qualifying rehearsal = the A′ rehearsal | **Codex** (build, pins, launch, acceptance) | Kaggle 2×T4, about 0.5 h quota, after the Oct 17 refresh | Oct 17 |

Deadline for the lane's deliverable: **Oct 16 12:00 UTC** (K3). Hard end: the Oct 17 rehearsal verdict.

## Coordination with Codex

- Claude delivers on branch `claude/lane3b-r7-prefetch-20261014`: the worker diff, tests, the soak script and its receipt, and a result note; it does not build, pin, launch or submit. The handoff entry states the new worker sha256 and the P1/P2 evidence paths.
- Codex owns: review against its 15:24 UTC terms (exact preservation of the other eleven targets is automatic here, since the worker's outputs are identical; the fresh worker/ledger/resource rehearsal is P3), `R7_FROZEN_SHA` and build pins in `scripts/build_public_r7_selection_b.py` and the build-time hashes in `scripts/build_public_r7_recovery_entry.py`, the `EVALUATION_INPUT_PARITY_V7` reference binding (v7 receipt fingerprints), the rehearsal launch, acceptance, and STATUS reconciliation. If Codex prefers to make the pin edits itself, Claude's branch leaves them untouched.
- One rehearsal serves lanes 1, 2 and 3b. Order of admission to A′ if the gate still binds: lane 1 arm, then lane 2 arm (as pre-registered in lane 2), with this worker in both cases.
- Codex may veto the worker for the Oct 17 rehearsal on review grounds; then K3 applies and nothing else changes.

## Files the implementation touches

- `knee/public_r7_recovery_entry.py`: evaluation-loop prefetch (a small, testable iterator), per-item capture, fingerprinting moved into the item stage, exception replay, `production_seconds` semantics, `prefetch_wait_seconds`, the `EVALUATION_INPUT_PARITY_V7` rehearsal gate (reference supplied through `cfg`).
- `knee/public_r7_recovery_rehearsal.py`: only if the injection hook needs the per-item capture interface; expected unchanged.
- `tests/test_public_r7_recovery.py` and a new focused `tests/test_public_r7_prefetch.py`: P1.
- `scripts/soak_r7_prefetch_cpu.py` (new): P2 harness reusing the worker's iterator with a stubbed forward; receipt under `outputs/analysis/lane3b-r7-prefetch-20261014/`.
- At Codex integration: `scripts/build_public_r7_selection_b.py` (`R7_FROZEN_SHA`), `scripts/build_public_r7_recovery_entry.py` (reference binding for the new gate). `knee/public_r7_b1_entry.py` and B1′ inherit the worker unchanged.
- Result note: `docs/experiments/lane3b-r7-prefetch-headroom-result-2026101x.md` with the stage-0 profile table, P1/P2/P3 evidence, measured ratio and projections; a dated handoff entry.

## Not in this lane

Raptor-arm prefetch, decode sharing across arms, dynamic shard assignment, compile-cache shipping, projection-policy edits, R7 flip-TTA or any model/TTA change, DataLoader spawn workers (fallback only), and any change to the arms or the blend.

## Expected value [estimate]

P(at least one of lanes 1–2 passes) about 0.5; P(A′ with the passing arms fits the gate) rises from about 0.45 to about 0.95 with this lane; E[gain | an arm passes] about +0.0008 macro, about 280 places at lane 3's 356 places per +0.001. Expected contribution about 0.5 × 0.5 × 280 ≈ 70 places, plus one avoided failed rehearsal day in a slow session and a larger margin against the hard 9 h limit in the hidden run. Cost: about 6–10 agent hours, one CPU kernel, no Kaggle GPU quota before Oct 17, no Colab by default.
