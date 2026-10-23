# Runtime headroom and the A″ plan, 2026-10-15

Written 2026-10-15 12:10 UTC by Fable (Claude), analysis only: no GPU, no Colab, no Kaggle push or submission. Labels: **[measured]** = a number in a receipt, ledger or repo file; **[literature]** = another participant's text or source; **[estimate]** = my judgement or a model built on measured inputs. Scratch scripts: `~/.claude/jobs/74614526/tmp/headroom/{breakdown,scenarios}.py` (read-only over the ledgers; no study IDs printed).

## Verdict

1. **R7 is the clock.** On 1,300 studies R7 (heliosli's four-fold ConvNeXt) is about 65 % of A′'s projected time, the R384 two-view arm 16 %, R224 one-view 9–11 %, fixed overhead 6 % [measured, scaled]. Inside R7, only 52 % of per-study time is the four-model forward; 26 % is serial CPU preprocessing and 22 % is serial CPU bookkeeping (input fingerprint hashing, stackers, transfers) while the GPU idles [measured]. The same DICOM volume is decoded three times in A′ (R384, R7 and R224 each decode in their own process) but the duplicated R384/R224 decode costs only 0.36 s per study (0.1 h); R7's decoder is a different pinned reader, so decode sharing is not where the time is.
2. **The frozen projection is conservative against every hidden run we can bound** (actual including queue was 0.89× the conservative figure for 57096002, ≤ 0.95× for A, ≤ 0.79× for B1), but it has one blind spot: R7's per-study state digest (about 2.5 s per study [estimate]) sits outside `production_seconds`, so it is counted once as overhead rather than 650 times per board. The 1.25 × 1.15 multipliers absorb it, and the hidden-run evidence still shows net conservatism. The A′ probe at 8.59 h conservative most likely runs 7.4–8.2 h [estimate]; P(it hits the 9 h limit) about 0.1–0.2 [estimate].
3. **The only output-preserving lever large enough to matter is the parked lane 3b R7 prefetch**: −1.1 to −1.5 h conservative on this session's ledgers (−1.8 h on v7's per lane 3b's own model). Its P2 soak failed a confounded criterion (1.52 vs 1.3, of which 1.37 is plain two-process contention and 1.11 is the prefetch); the real test is the paired 2×T4 rehearsal that the protocol already pre-registers as P3. A simpler fallback, "ledger-light" entry mode (skip evaluation-study input hashing), is worth −0.8 h and about twenty lines, but does not fund the mirror.
4. **Member trades do not pay.** Swapping R7 for nartaa's S6fall frees 2.7 h but has expected Δ about −0.0003 (range −0.0015 to +0.0008, P(Δ > 0) ≈ 0.4) [estimate] and needs a new worker, binding and slot. Fewer R7 folds free 0.8–1.6 h at an expected −0.0005 to −0.0015 on A [estimate]. S6fall as a fifth member fits only with prefetch and is worth 0 to +0.0006 with no validation path.
5. **Recommended A″ = A′ (R384 + R224 one-view at 0.65/0.35 in the Raptor half, R7 at 0.5) + the lane 3b prefetch worker + the R224 anatomical mirror, admitted in that order and gated by the frozen 8.3 h projection.** Projected 7.65 h conservative (central contention) or 8.10 h (conservative contention); if the mirror does not fit, A′ + prefetch at 7.07–7.53 h still turns the gate-waived A′ into a qualified entry. Expected Δ vs A′: +0.0001 to +0.0004, P(Δ > 0) 0.65 [estimate, from the scan]; vs A: +0.0003 to +0.0012, P(Δ > 0) ≈ 0.8, conditional on today's A′ probe (57229227) not showing a displayed deficit. **If the probe displays below 0.956, nothing on this list is worth a slot**: the mirror only rides on R224, and the R7-side trades have non-positive expected value. Prefetch would then still be worth integrating as runtime insurance for whichever A-class entry is final 1, without a new slot.

## Recommended plan (Oct 16–20)

| Item | Content |
|---|---|
| Change set | (a) R7 worker `knee/public_r7_recovery_entry.py` → lane 3b bytes `09ad6adb…` (branch `claude/r7-prefetch-lane3b-20261014`, head a415ce5; P1 55 tests green; Astra's six blocking findings addressed in 457c44a). (b) R224 arm: second forward on nartaa's `_mirror_tri` views (sagittal slots reverse the three slice channels; coronal/axial flip width), probabilities averaged, then the existing 0.35 rank slot. (c) New versioned recipe `ws2-a-prime-r224-v3`; nothing else in the blend, recovery or gates changes. |
| Expected Δ | Mirror on R224: +0.0001 to +0.0004, P(Δ > 0) 0.65, P(Δ ≥ +0.001) 0.05 [estimate, `public-0957-scan-20261015.md` §2 #2; one public rank move 357 → 289 [literature]]. Prefetch: 0 by construction (identical arrays, fingerprints and probabilities; P1 proves it synthetically, P3 must prove it on CUDA). Rank value near 0.956: 356 places per true +0.001 [measured on the lane 3 model], so the mirror is worth about 35–140 places in expectation and is below the public paired SD (0.0004–0.0009); its reading can neither confirm nor refute it, only a deficit rejects. |
| Runtime (frozen formula, v2 ledgers) | A′ + prefetch: **7.07 h** central / **7.53 h** conservative contention. A″ with mirror: **7.65 h** / **8.10 h**, both ≤ 8.3 h. Without prefetch the mirror cannot fit (9.17 h). Point estimate at multipliers 1.0/1.0: A′ 6.52 h, so the hidden run should take roughly 6–7 h. |
| Order of admission | Prefetch first (it is the enabler), then R224 (already in A′), then the mirror only if the rehearsal projection stays ≤ 29,880 s. A P3 parity failure reverts the R7 worker to the v7 bytes (protocol K4) and A″ collapses to A′ unchanged. |
| Effort | Codex (integration, about 0.5–1 day): pin `R7_FROZEN_SHA` (and the B3 pin that hashes the selection-B builder), bind `evaluation_input_reference` from the v7 receipt in rehearsal cfg, add the paired serial control (`cfg['r7_prefetch']=False`) as a second R7 phase in the rehearsal (+8–10 min) and run `scripts/compare_r7_prefetch_p3.py`, implement the mirror in the crop224 worker with sentinel and cross-board parity, 3–5 tests, rebuild the sealed package. Claude (GPU): review a415ce5 against the Astra packet, one 2×T4 rehearsal (about 0.65 session-hours: 1,986 s baseline + control phase + mirror), collection and parent checks, entry build, visible run (about 8 min), submission, hidden-run reconciliation. |
| Kaggle quota | 9.76 h left now; refresh 2026-10-17 00:00 (timezone unverified). Rehearsal + visible run ≈ 0.8 h, so no need to wait for the refresh. Hidden reruns do not draw on the user's quota as far as I know [estimate; verify if it matters]. Slots: 4 left today, 5 per day after; A″ needs one. |
| Timeline | Oct 15 about 19:00–20:30 UTC: 57229227 scores (submitted 11:54:48 UTC; hidden runs so far took 3.7–7.3 h). Oct 16: Codex integrates, Claude rehearses and builds. Oct 16 late or Oct 17: submit A″; scores the same day. Oct 18–20: one fallback window (drop the mirror, or ledger-light if P3 fails). Last new hidden submission Oct 20 16:00 UTC per the plan of record; final selection Oct 21. |

### Pre-registered acceptance (rehearsal, then entry)

1. Every existing gate unchanged and PASS: compiled with no fallback on both R7 boards; R7 cross-board sentinel 1e-3; strict reload 1e-6; old/crop224 cross-board exact; zero failed or imputed studies; blend rank Δ 0 on all 12 targets against the visible reference; CSV schema and finite values.
2. `EVALUATION_INPUT_PARITY_V7`: all 39 evaluation studies' eight fingerprints equal the v7 receipt on both boards (`checked == len(records)`).
3. P3 paired comparison, serial control vs prefetch, same session and worker bytes: identical ids, assignment and order; 39/39 fingerprints; probabilities within 1e-6 when one process produced both, 1e-3 across separately compiled processes; `max_concurrent_stages == 1`; `alive_at_exit` false; per-worker RSS ≤ 9.5 GB, combined phase peak ≤ 22 GB; CUDA allocated/reserved peaks recorded.
4. Speed: paired evaluation-wall ratio prefetch/serial on the slower board ≤ 0.85 (central estimate 0.72 from the v2 ledgers; 0.69–0.72 in lane 3b's v7 model [estimate]). Report production/forward alongside (protocol criterion ≤ 1.40, informational because the forward interval is interleaved). A ratio > 0.90 is recorded as "output-preserving, no gain": the worker still ships, the mirror is dropped.
5. Frozen projection: A″ conservative ≤ 29,880 s. If it fails with the mirror, rebuild as A′ + prefetch and require ≤ 29,880 s; if that fails too, close A″ and keep A′'s probe result as the only R224 evidence.
6. Mirror arm: per-study normal and mirrored window hashes recorded; crop224 sentinels strict reload 1e-6 and cross-board exact; visible three rows finite and in (0, 1).
7. Entry: visible run COMPLETE with every receipt PASS; CSV equals the rehearsal's within float32; hidden run COMPLETE.
8. Selection: A″ replaces the current final 1 only if displayed(A″) ≥ displayed(best COMPLETE of A, A′). A displayed deficit rejects the mirror (and, if A′ also trailed, R224). Timeout or ERROR is a failed probe and changes nothing.

Fallback if Codex vetoes the thread design or P3 fails parity: **ledger-light entry mode** in the R7 worker, skipping the eight evaluation-study input fingerprints when `cfg['mode']!='rehearsal'` (they are consumed only for pilots, sentinels and the rehearsal's cross-board/parity gates, lines 400–405 and 536–551 of `knee/public_r7_recovery_entry.py`; evaluation rows carry them as diagnostics). −0.78 h conservative (A′ 7.81 h, PASS); the mirror does not fit (8.38 h). Same acceptance minus items 2–4, plus a test that entry rows still carry `status`, `probabilities` and timings.

## 1. Time breakdown

Source: v2 qualification `outputs/kaggle/ws2-r224-v2-qual-20261015-result/ws2-a-prime/` (39 studies, 20 + 19 per board, 2×T4, 4 vCPU) [measured]. Scaling: 650 studies per board; R7 scaled by windows (sample mean 586.8, development-870 mean 577.8). Hours are the slower board without the 1.25/1.15 multipliers unless marked.

| Member / phase | Per study, slower board | Of which | 1,300-study hours (×1.0) | Share of A′ |
|---|---|---|---|---|
| Setup, merges | 14.1 s + 0.3 s once | staging, CSV | 0.00 | 0 % |
| R384 two-view (old) fixed overhead | 504.6 s once | model load, stress cases, 4 sentinels, reference passes (rehearsal-only work included) | 0.14 | 2 % |
| R384 two-view production | **5.82 s** (board 1: 5.53) | decode ≈ 0.36 s (same volume hash as R224, fast-header path), windows ≈ 0.1 s, two 94-window forwards ≈ 5.3 s | **1.05** | 16 % |
| R7 fixed overhead | 716.7 s once | compile check 297 s, model/stacker loads, 8 sentinel repeats with reloads, stress cases, plus the per-study state digests (see §2) | 0.20 | 3 % |
| R7 production | **23.95 s** (board 1: 22.16) | four-model forward 12.35 s (21.4 ms/window; board 1 17.5), preprocess 6.22 s (decode+geometry+side-net 3.19, render 1.76, anchors 1.27), input hashing ≈ 3.0 s, stackers/transfers/row ≈ 2.3 s, fluid 0.07 s | **4.26** | 65 % |
| R224 one-view fixed overhead | 87.8 s once | checkpoint load, 4 sentinels | 0.02 | 0 % |
| R224 production | **3.33 s** (board 1: 3.27) | decode 0.36 s, window 0.08 s, forward 2.70 s (94 windows at 224) | **0.60** | 9 % |
| **A′ total, ×1.0 / frozen conservative** | | | **6.52 h / 8.59 h** | |

Decode duplication: R384 and R224 decode the same volume (39/39 identical `volume_sha256`) in separate processes and phases; R7 decodes again through heliosli's pinned `native_cache`/`px_cache` reader with different geometry. The removable duplicate (R224's decode and its 88 s overhead if folded into the R384 worker) is about 0.10 h conservative. R7's 3.19 s decode+geometry step is pinned code [measured, lane 3b review].

Session variance across identical worker bytes [measured]:

| Session | R384 s/study (b0/b1) | R7 forward ms/window (b0/b1) | R7 preprocess s/study | R7 production s/study (b0/b1) | A conservative |
|---|---|---|---|---|---|
| v7, Oct 13 (A entry built from it) | 5.54 / 5.62 | 18.2 / 17.4 | 6.27 / 6.12 | 22.35 / 22.21 | 7.38 h |
| v4, Oct 13, slow host | 7.2 | ≈ 18 | 16.0 incl. hashing | 26.8 | 8.97 h |
| v1 cancelled, Oct 15 (13 + 12 R7 rows) | 7.18 / 7.16 | 16.7 / 17.2 | 6.74 / 7.57 | 22.3 / 24.5 | — |
| v2, Oct 15 | 5.82 / 5.53 | **21.4** / 17.5 | 6.22 / 6.12 | 23.95 / 22.16 | 7.86 h |

Board 0 of v2 had a 22 % slower GPU per window than board 1 with identical code; v1 had slow CPUs and fast GPUs. Host and board speed, not code, moved the conservative figure from 7.38 to 8.97 h.

## 2. Projection versus evidence

Frozen formula (`knee/public_s6_recovery_entry.projection`, `knee/public_ws2_entry.projection`) [measured]: conservative = setup + merges + Σ phase overhead (all rehearsal work, once) + old production × 1300/39 × 1.15 + R7 production × (1300 × 577.8 × 1.25 / 22,885 windows) × 1.15 + R224 production × 1300/39 × 1.15 + R224 overhead. "Production" is the slower board's per-study sum. The 1.25 "hidden windows" and 1.15 "imbalance" terms multiply R7's 19,656 s point term to 22,604 s (+2,948 s), the R384 term 3,877 → 4,459 s, R224 2,219 → 2,552 s. The nine-scenario grid is in both receipts.

Hidden-run bounds (submission timestamps from the Kaggle API, completion from the handoff log; queue time is inside every figure and unknown) [measured]:

| Submission | Entry | Conservative / point (h) | Pending at | Complete by | Actual incl. queue (h) | Actual / conservative |
|---|---|---|---|---|---|---|
| 57096002 | R384 + S6 70:30 | 8.19 / 5.97 | 12:07 UTC (7.22 h) | 12:11 UTC | **7.22–7.29** | **0.89** |
| 57157384 | A | 7.38 / 5.44 | 01:51 UTC (3.66 h) | 05:12 UTC | 3.66–7.01 | ≤ 0.95 |
| 57157388 | B2 (Dread + R7) | 7.21 / 5.29 | 01:51 UTC (3.66 h) | 05:12 UTC | 3.66–7.01 | ≤ 0.97 |
| 57168064 | B1 (R7 alone) | 6.34 / 4.47 | 08:19 UTC (3.11 h) | 10:13 UTC | 3.11–5.01 | **≤ 0.79** |
| 57157385 | B1 first attempt | — | — | ERROR at 3.6 h | platform fault | — |
| 57229227 | A′ probe | 8.59 / 6.52 | submitted 11:54:48 UTC | pending | — | — |

Reading: the one tightly bounded run sits 21 % above its point estimate and 11 % below its conservative figure, with queue inside. B1 shows the R7 term alone is at least 21 % conservative. A ran at most 7.0 h against 7.38 h. Nothing bounds A's actual time from below better than 3.7 h. For the A′ probe this implies about 7.4–8.2 h actual if the hidden host resembles the v7/v2 sessions [estimate]; a v4-class host (+22 % on CPU work) would put it near 9 h, hence P(timeout) 0.1–0.2 [estimate].

Where the projection is and is not conservative [measured unless marked]:

- Conservative: slower board for every arm; all rehearsal-only overhead counted (stress cases, 8 sentinel repeats, reference decodes/forwards, the 297 s compile check per board); R7's 1.25 window multiplier against a hidden mean that nartaa's kernel budgets as 1,284 studies with no sign of larger studies [literature].
- Not modelled: R7's per-study `S6_STATE_AFTER_EACH_STUDY` digest (SHA-256 over all parameters and buffers of four ConvNeXt-smalls, the anchor V-Net and the side net, about 1 GB) runs in the `finally` block after `production_seconds` is stamped (lines 413 and 424–425), so about 2.5 s per study [estimate] is counted once in "overhead" and never scaled: about 0.45 h per board of real hidden time. The lane 3b worker closes the interval properly and so would raise its own measured production by this amount; my scenario table adds it back as `D`. Also unmodelled: ledger I/O growth over 650 rows, and the hidden-set window distribution.
- Dynamic shard assignment would remove the 1.15 term on paper but change which board scores a study (separately compiled graphs differ by up to 2.6e-4); rejected in lane 3b as a gate change, and I agree.

## 3. R7 speedups

Savings are conservative-projection hours on the v2 ledgers for A′ (8.59 h baseline) from `scenarios.py`; real hidden-time savings are roughly 0.7× these.

| Lever | Saving (conservative h) | Accuracy effect | Evidence and status |
|---|---|---|---|
| **Lane 3b one-ahead prefetch** (item stage + hashing on a producer thread, quiescent boundaries) | **−1.06 (c = 1.52) to −1.52 (c = 1.11)**; lane 3b's own v7 model: −1.06 to −1.80 | none by construction; P3 must prove CUDA parity | P1 55/55 tests green; P2 fingerprints 39/39, RSS 8.7 GB, paired wall 0.86–0.875, but CPU-stage ratio 1.52 > 1.3 FAIL. The criterion compared two processes against one: two serial processes already give 1.37; the prefetch adds 1.11. **Single allowed K2 fix:** no second CPU soak (anchors on CPU make it 5× more CPU-bound than the T4 host); treat P2 as mechanics-only evidence and let P3 (the paired 2×T4 rehearsal, already pre-registered, serial control in the same session) decide. If Codex requires a code change for K2, the candidate is pinning the producer to one `torch` thread (`torch.set_num_threads(1)` inside the producer only), which attacks the 1.37 contention term; it changes no output. |
| **Ledger-light entry mode** (skip evaluation-study input fingerprints; keep pilots, sentinels, state before/after) | −0.78 | none (fingerprints are diagnostics for evaluation rows) | Hashing 3.0 s/study measured in the P2 soak; consumers listed above. Simple, but loses the hidden-run input ledger and is an alternative to, not additive with, prefetch (prefetch already hides hashing). Also skipping the per-study state digest saves another 2.5 s/study of real time (0.45 h per board) but 0 in the frozen projection; it weakens per-study drift attribution, not detection (state_after remains). |
| Shared decode or cache across members | −0.10 (R224 into the R384 worker) | none | R384/R224 volumes identical 39/39; R7's reader is different pinned code (lane 3b review). Not worth a worker change on its own. |
| Batch size, channels_last, CUDA graphs, compile mode, cudnn.benchmark | ≤ −0.1 to −0.4 each, uncertain | numerics change (algorithm selection) → requalification of every R7 gate | Already: fp16 autocast, `channels_last`, inductor `default` on six static shapes (32…192), chunk 192, peak 6.2 GB of 15 GB. `cudnn.benchmark=False`/deterministic is forced in three places. T4 is compute-bound at this batch (5.3 ms per window per fold); `reduce-overhead` (CUDA graphs) targets launch overhead and would also forbid the producer's concurrent CUDA work; `max-autotune` adds 10–30 min of compile per board, eating its own gain. No cheap safe lever here. |
| Members concurrently per GPU (R384 phase alongside R7 phase) | up to −1.0 on paper, likely ≈ 0 | none | Gain = R384's GPU time hidden behind R7's CPU stalls, but four processes on four vCPUs: the soak measured 1.37× CPU contention from two processes alone; R7's 11.5 s of CPU per study would grow by more than the 5.3 s hidden. Also a parent `launch` change (Codex domain) and +6 GB host RSS. Prefetch achieves the same overlap inside one process. Reject. |
| Fewer R7 folds | 3 folds −0.81; 2 folds −1.61 | **−0.0005 to −0.0015 on A** [estimate: single-family 4→2 fold ensemble loses 0.001–0.003 on the component, half weight]; no local check possible (the 870 are exposed to R7) | Each fold is 3.1 s/study on board 0. Reject: the loss is 2–5× the gains being funded. |
| Compile-cache shipping, counting only entry overhead | −0.08; policy change | — | Rejected in lane 3b; unchanged. |

## 4. Member trades

| Trade | Runtime (conservative, A′ basis) | Expected Δ on A | What the freed time funds | Effort and evidence |
|---|---|---|---|---|
| **S6fall replaces R7** (one all-data S6-class ConvNeXt-small at size 320 / crop 288, same `compact_infer` pipeline, no state or extent heads, no `fm_aux`) | A′ 8.59 → **5.84 h** (−2.75 h: one forward at 0.67× the area, same preprocess; compile check ¼) | **−0.0015 to +0.0008, median −0.0003, P(Δ > 0) ≈ 0.4** [estimate: S6fall 0.950 solo vs R7 0.949 [literature]; same architecture class and label family ("low decorrelation vs R7" in the scan), a single model against a four-fold mean, no OAI/fastMRI] | Mirror (+0.58 h), S6 four-fold as a second compact member (+2.6 h as a process), or nothing: 5.84 h leaves 2.5 h idle | New worker variant (plain S6 path, 320/288 geometry, no stackers), new asset binding (CC BY-NC 4.0 weights, nartaa's `s6_extras.bin` code), rehearsal, one slot; no OOF validation exists (all-data model). **Reject as a swap**: it spends the only certain asset (R7's four-fold mean) for a coin flip. |
| S6fall as a fifth member in-process (0.8 R7 / 0.2 S6fall; shared decode, separate 320-px render + one forward) | +1.01 h → 9.60 h FAIL; with prefetch (central) 8.07 h PASS | 0 to +0.0006, P(Δ > 0) 0.55, P(≥ +0.001) 0.05 [estimate, scan §2 #3] | — | Needs prefetch first and a new in-worker model path (different render size, so windows cannot be shared); no validation path; one slot. Lower priority than the mirror; revisit only if A″ passes with ≥ 0.7 h to spare. |
| S6fall as a separate fifth process | +3.7 h → 12.3 h | as above | — | Does not fit. |
| R224 anatomical mirror (the candidate A″ funds) | +0.58 h (forward doubles; decode and overhead unchanged) | +0.0001 to +0.0004, P 0.65 [estimate] | — | Small code change in the crop224 worker; fits only with prefetch. |
| R7 at 2 folds + mirror | 6.98 + 0.58 = 7.56 h | net −0.0004 to −0.0011 [estimate] | — | Reject. |

## Limits

- All runtime numbers come from 39 rehearsal studies on two sessions; hidden-set window counts, host speed and queue time are unobserved. The frozen multipliers are policy, not measured bounds; the hidden-run bounds above include unknown queue time.
- Prefetch savings use a contention model (c = 1.11 central from the soak's prefetch2/serial2, 1.52 conservative) and two estimated constants (hashing 3.0 s from a CPU kernel, digest 2.5 s unmeasured). Only P3 on 2×T4 measures them.
- Expected Δ figures are estimates from public displayed scores and one author's rank moves; the public paired SD (0.0004–0.0009) exceeds the mirror's expected gain, so the hidden slot cannot confirm it, only reject a deficit.
- S6fall per-study cost assumes heliosli's pipeline scales with crop area and that its preprocess equals R7's; its stackers path, manifest keys and side model were not run locally.
- No GPU, Colab or Kaggle execution was performed for this note; nothing here qualifies an entry.

## Evidence

- v2 qualification: `outputs/kaggle/ws2-r224-v2-qual-20261015-result/ws2-a-prime/{receipt.json,base/receipt.json,base/board*-{old,s6}/*/receipt.json,board*-crop224/crop224/receipt.json}`; `docs/experiments/r224-kaggle-v2-result-20261015.md`; `outputs/analysis/r224-integration-20261015/kaggle-v2-baseline-timing-comparison.json`.
- v1 cancelled partial: `outputs/kaggle/ws2-r224-v2-qual-20261015-result-v1-cancelled/`, `outputs/analysis/r224-integration-20261015/kaggle-v1-cancelled-summary.json`.
- v7 rehearsal: `outputs/kaggle/public-r7-recovery-rehearsal-20261013-result-v7/s6-recovery-entry/receipt.json`; `docs/experiments/r7-recovery-entry-20261013.md`.
- Projection code: `knee/public_s6_recovery_entry.py` lines 551–571; `knee/public_ws2_entry.py` lines 189–209; R7 worker loop `knee/public_r7_recovery_entry.py` lines 349–433.
- Lane 3b: `docs/experiments/lane3b-r7-prefetch-headroom-protocol-20261014.md`; branch `claude/r7-prefetch-lane3b-20261014` at a415ce5 (`docs/experiments/lane3b-r7-prefetch-headroom-result-20261014.md` there); `outputs/analysis/lane3b-r7-prefetch-20261014/{projection.json,p2-soak-receipt.json,astra-design-review.md}`.
- Hidden runs: `kaggle competitions submissions -c rsna-knee-abnormality-detection --csv` (saved to the scratch directory); `docs/archive/handoff/claude-astra-handoff.md` entries 2026-10-12 12:07/12:11, 2026-10-14 01:51/05:12/08:19/10:13; `outputs/analysis/resume-20261014/official-state.json`.
- Candidates: `docs/research/public-0957-scan-20261015.md` §1.2, §1.4, §2, §3; `docs/research/rank-value-selection-20261014.md`; `outputs/research/public-0957-20261015/datasets/nartaa__rsna-knee-s6fall-weights/checkpoint-manifest.json`; `docs/research/plan-update-20261014.md` step 7.
