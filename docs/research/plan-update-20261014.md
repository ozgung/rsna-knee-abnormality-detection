# Plan update to the Oct 22 final — 2026-10-14, about 11:00 UTC

Author: Fable (Claude), strategy. Read-only review: no compute, submission, commit, STATUS or handoff-log edit; this file is the only output. Labels: **[measured]** = a number in a linked receipt, log or repo file; **[literature]** = another participant's public claim or official text; **[estimate]** = my judgement.

Inputs: `knee-domain-research-20261014.md`, `iteration4-assessment-20261014.md`, `../experiments/raptor64-axial-pf-it4-protocol-20261014.md`, `iteration3-decision-20261014.md`, `../experiments/raptor64-original-methods-it3-protocol-20261014.md`, `shakeup-risk-20261013.md`, `../experiments/r7-recovery-entry-20261013.md`, the handoff log through 10:24 UTC, `STATUS.md` (08:39 UTC), one raw DICOM header and the Oct 10 header-audit summary.

## 1. Facts the plan rests on

- **Entries.** A = Raptor (nartaa OAI-trained CoAtNet) + heliosli R7 at fixed 0.5/0.5 ranks, public 0.956 (57157384); B1 = R7 alone, no OAI, 0.949 (57168064, COMPLETE today); B2 = Dread + R7 0.948; B0 0.940; own-only CoAtNet-224 0.914 (57037296, scored, selectable) [measured].
- **Runtime.** A's qualifying rehearsal projected 7.87 h conservative (5.80 h base) against the 8.3 h gate, so A has about 0.4 h of conservative headroom; B3 failed at 9.48 h [measured]. B1 runs one worker and has the largest margin [measured].
- **Leaderboard.** Top 10 at about 0.963, rank 1 at 0.966, about 800 teams at ≥ 0.956, gold line about 0.958 [literature, LB readings]. Forks of the heliosli recipe plateau at 0.956–0.957 and its author's own run shows 0.957 [literature].
- **Own-model class.** in-distil fold-0 K42 0.9149; dev-to-public offset about +0.012, so about 0.925 public, Dread parity [measured/estimate]. Iteration 3 interim: `in-distil-w1` epoch-11 K16 0.9152 vs in-distil's final K16 0.9124 (+0.0028); `in-distil-flip` 0.9123 (−0.0001) [measured, interim]. in-distil gained +0.0025 from K16 to K42, so w1's K42 projects to about 0.918 [estimate]; the full-data bar is 0.9218.
- **Shake-up.** The public/private split is unstated; at 30/70, one entry's public-to-private SD is about 0.004, but the *paired* SD between two fork-like entries is about 0.001 on the public set and smaller on private, and about 0.002–0.003 against a genuinely different model [measured on the 870 / estimate]. A true +0.002 therefore mostly survives to private; the rank order inside the 0.956 block is otherwise a lottery.
- **Compute.** Kaggle GPU about 9 h until the Oct 17 refresh, then 60 h/week [measured]. Colab A100 sessions are plentiful; wall-clock is the limit. Two A100s are in use: iteration 3 (final K42 eval due within about 1 h) and iteration 4 (building since about 10:25 UTC).
- **Submissions.** 5 per day, about 7 h hidden scoring each; 8 days leave about 40 slots, of which this plan uses 4–6. Whether a hidden scoring run draws GPU quota is unverified [estimate: it does not; check the quota page after the next submission].
- **Provenance.** R7's fastMRI auxiliary training and weight licence are unresolved, and both A and B1 carry R7; the no-OAI hedge covers only an OAI ruling [measured, `outputs/analysis/r7-recovery-20261013/provenance-primary-sources.md`].
- **Metadata.** One raw header (Philips Ingenia, 1.5 T) carries 63 tags including `PatientSex`, `Manufacturer`, `ManufacturerModelName`, `MagneticFieldStrength`, `SoftwareVersions`, `ReceiveCoilName`, `ContrastBolusAgent`, `SeriesDescription`, `Laterality` (empty here), sequence parameters, spacing; `PatientAge`, `InstitutionName` and `StudyDate` are absent [measured, one study]. Only one study's DICOMs are cached locally; the Oct 10 Kaggle header audit (819,078 headers, 71 CPU-min) saved aggregates only, no per-study table [measured].

## 2. (a) Is there any route to about 0.963 by Oct 22?

**No executable route reaches 0.963 in expectation; the remaining eight days should maximise expected private rank.** The reasoning, lever by lever:

| Lever | True gain on A | Why it stops there |
|---|---|---|
| Own distilled student into A or B (iteration 3) | ≤ +0.002 [estimate] | 0.925-class public; B2 vs B1 showed that a Dread-class arm adds nothing to R7 (0.948 vs 0.949) [measured]. w1's interim +0.003 dev is below the 0.9218 bar; P(bar met) about 0.05 [estimate]. |
| Axial patellofemoral arm (iteration 4) | +0.0004 to +0.0013 [estimate] | One target of twelve at a 0.3 weight. |
| Site/protocol prior from DICOM metadata (new, section 3 step 3) | 0 to +0.002 [estimate] | Image arms already encode scanner cues; no `PatientAge` in the headers. |
| Side canonicalisation + compartment pooling (item 2) | 0 at entry level unless the own recipe clears 0.9218 [estimate] | Own-model gain +0.003 to +0.005 on fold 0; no entry path on its own. |
| OAI pretraining of an own CoAtNet (the user has NDA access) | +0.02 on the own model [literature/estimate]; ≤ +0.004 on A | Download, curation and pretraining take more than the 8 days of wall-clock; A already holds an OAI arm; the host has not ruled. P(trained, qualified and scored by Oct 21) about 0.1. |
| More public components, mirror TTA, weight probes | each ≤ +0.002, some negative [measured/literature] | Against the user's direction, and the 800-team plateau shows they saturate at 0.957. |

Everything that can plausibly land sums to about +0.002 to +0.005 true gain, so a public reading of 0.957–0.960 with ±0.004 noise. The teams at ≥ 0.960 built 0.949-class *own* singles (OAI or fastMRI pretraining, anchored local views, five folds) over weeks; a jump of +0.007 needs one more such component, and none can be built, qualified and run inside 9 h on 2×T4 by Oct 21.

**Probabilities [estimate].**
- P(public ≥ 0.963 by Oct 22, any route we can execute): **about 0.02**, and mostly by sampling luck rather than true gain.
- P(public ≥ 0.960): about 0.10.
- P(private top 10): about 0.01.
- P(private rank inside the gold line, about 0.958 true): about 0.05 if we freeze A now; about 0.12 if we bank the +0.002 to +0.004 of robust small gains below.

**Why small gains still matter for rank.** About 650 teams sit between 0.956 and 0.958, so roughly 300 teams per 0.001 [estimate from the LB counts]. A true +0.002 that is robust (pre-registered, not fitted to the public LB) moves expected private rank by several hundred places, from roughly the middle of the block to its upper edge, which is where the bronze and silver thresholds lie if the field is about 3,000–4,000 teams [unverified]. In rank terms the small items are not noise; in score terms they are, and that is exactly why they must be gated on fold-0 CIs and not on public readings.

**What maximises expected private rank.** (1) Bank every robust +0.001 into A and mirror it into B1. (2) Keep the OAI hedge B1. (3) Resolve or hedge the R7/fastMRI question, which both finals share. (4) Protect against run failure: rehearsals after the Oct 17 refresh, no new hidden runs after Oct 20, finals selected on Oct 21.

## 3. (b) Ordered work plan through Oct 22

Owners: **Claude** (protocols, evaluation, decisions, commits), **Codex** (integration into the frozen R7 wrapper, acceptance checks, rehearsal launches), **Colab agent** (A100 training under a pre-registered protocol). One implementation task per owner at a time.

### Step 0 — Iteration 3 read-out and decision (today, about 11:30 UTC; Claude; no new compute)

Pre-commit: the K42 paired Δ against in-distil decides, with no rounding and no re-reads of K16. Outcomes:

| Outcome | Action |
|---|---|
| **w1 wins** (Δ ≥ +0.003, CI lower > 0) | Distillation is not saturated at 0.5; `w1` becomes the own-student recipe of record. The full-data bar (K42 ≥ 0.9218) stays; if unmet (expected), no full-data student and no second-seed replicate. Knowledge only. |
| **flip wins** (CI lower > 0) | Laterality invariance helps; promote item 2 (step 5) with the flip as its stage A. If w1 also won, one combined `w1+flip` fold-0 run is justified *only* if the two Δs sum to a projected K42 ≥ 0.9218; otherwise it is knowledge. |
| **both null** | Distillation saturated; mirror irrelevant. Close iteration 3, keep in-distil as the recipe of record, skip step 5, and close the own-model lane after iteration 4. The A100 goes to step 2 or is stopped. |

Writing: a result note under `docs/experiments/`, dataset verification, and a handoff entry. **Iteration 4 continues unchanged in all three cases.** It is a single-factor crop test on in-distil; changing its teacher weight to match a w1 win would confound the crop factor, and the weight is not worth a re-run.

### Step 1 — Iteration 4 read-out (today, about 16:00–18:00 UTC [estimate]; Colab agent trains, Claude evaluates)

Hypothesis, arms, metric and the five kill conditions are as pre-registered in the protocol; nothing changes. Compute: one A100 session, about 4–5 h for `ax-crop` and `ax-full` concurrently [estimate]. Pass → step 2. Fail → close the axial idea and skip to step 3; do not rebuild at native resolution.

### Step 2 — Full-data axial PF arm into A and B1 (Oct 15–18; Colab agent trains, Codex integrates and rehearses, Claude evaluates and commits)

- **Hypothesis:** the fold-0 PF OA gain carries to the entries at a fixed 0.3 rank weight on PF OA only.
- **Steps:** full-data retrain on all 4,345 studies with the pre-registered SWA epochs (about 3–4 h A100 [estimate]); secondary gate 5 (blend into the pinned Dread fold-0 predictions must not lower PF OA); Codex adds the arm as a per-target component of the frozen wrapper with weights only parameterised, plus a regression test that the other 11 targets are byte-identical; one 2×T4 rehearsal after the Oct 17 quota refresh (about 0.5 h quota); one hidden submission of A′ on Oct 17, B1′ on Oct 18.
- **Kill:** rehearsal conservative projection > 8.3 h drops the arm from A (B1 keeps it); any schema or coverage check failing drops it everywhere; a hidden run error reverts to A/B1 with no retry beyond one.
- **Expected entry gain:** +0.0004 to +0.0013 [estimate]. Quota: about 1 h T4 total.

### Step 3 — Site/protocol prior from DICOM metadata (Oct 14–16 for the fold-0 test; Claude; CPU only, no GPU quota)

- **Hypothesis:** per-study DICOM metadata carries report-label signal that the image arms under-use, mainly through site reporting style and protocol choice (Synovitis, Lateral OA, PF OA, Baker's). The test labels come from the same report pipeline, so a site prior is legitimately predictive of them. Inference stays image-file-only; the metadata is in the DICOM headers.
- **Data:** one Kaggle **CPU** kernel reads one header per series for `train_series` and `test_series` (about 17k headers, minutes) and writes a per-study table keyed by `StudyInstanceUID` with: `PatientSex`; site fingerprint = `Manufacturer` × `ManufacturerModelName` × `MagneticFieldStrength` × `SoftwareVersions` × `ReceiveCoilName`, bucketed with a minimum count of 20 and an "other" level; contrast flag; series counts by (plane, fluid-sensitive, fat-suppression) from `train_series.csv`; median slice thickness and pixel spacing per plane; series count; `Laterality` population rate. No `PatientID`, no pixel data. The kernel also asserts that the test header tag set equals train's.
- **Model (pre-registered here):** per-target L2 logistic regression on the 3,475 training studies with masked weak labels (the same label table in-distil used), C chosen by inner 5-fold CV on the training studies only. Combination on the 870: `logit(p) + 0.5 × (logit(p_meta) − logit(prevalence_target))`, applied to in-distil's K42 OOF probabilities; paired whole-component bootstrap vs in-distil, no weight tuning on the 870.
- **Kill:** macro Δ < +0.002, or CI lower ≤ 0, or any target drops > 0.005. **Transfer gate** before any entry work: the same fixed rule on the pinned Dread v4_full fold-0 predictions must not lower macro. If both pass, Codex integrates the shift as a post-hoc per-target term on the blended output of A′/B1′ at the same fixed 0.5, one unit test, and it rides the step-2 rehearsal and submissions (one extra hidden run if step 2 has already been submitted).
- **Expected gain:** own model +0.000 to +0.004 on fold 0; entry 0 to +0.002 [estimate]. Cost: 3–4 Claude hours, one CPU kernel, no GPU quota.

### Step 4 — Second-final composition and provenance (Oct 15–21; Claude; no compute)

- Final 1 = the best-scoring A-family entry that passed all gates (A, or A′ if ≥ A public). Final 2 = the best no-OAI entry (B1, or B1′ if ≥ B1). Do not build an own-heavy B: it hedges nothing the cheaper no-OAI entries do not and costs 0.012–0.017 against A [measured/estimate, shake-up note].
- Optional, only if slots are idle on Oct 19–20: an R7-heavy 0.75/0.25 R7 + Dread variant of B1′ (expected +0.001 [estimate]); skip if anything upstream slipped.
- **R7/fastMRI:** re-check the host threads (737950, 733965, 748480, 748276) on Oct 16 and Oct 20. If the host states that fastMRI-derived weights are disallowed, final 2 switches to Dread + S6 (57119585, 0.940) and, if OAI is also adverse, final 1 to the own-only 57037296 (0.914). If the host explicitly allows OAI before Oct 21, final 2 becomes the second-best A-family entry instead of B1, since B1 then hedges nothing that A′ does not.
- Kill: none; this is a decision with a pre-stated rule.

### Step 5 — Side canonicalisation + compartment-aware pooling, gated on the flip result (Oct 15–18; Claude designs, Colab agent trains; about 2R ≈ 6–8 h A100 including the side oracle [estimate])

- Run **only if `in-distil-flip` wins** in step 0. Stage A: flip plus a canonical frame from the SKM-TEA medial/lateral tibial-cartilage oracle (check `Laterality` population first; it was empty in the one sampled header). Stage B: soft compartment attention biases. Kill rules as in the cloud report item 2.
- Entry path: none unless the combined own recipe reaches K42 ≥ 0.9218; then one full-data student at a 0.2 slot in B1′ under the iteration-3 rule (rehearsal by Oct 19, hidden run by Oct 20). Otherwise knowledge and write-up.

### Step 6 — Codex M4 same-series ROI (Codex; Oct 16 kill date unchanged)

Fixed 120 mm sagittal / 140 mm coronal FOV or the extent-plus-margin rule, inside the pre-registered gate (truncation ≤ 2 % of native slots, fallback ≤ 30 %). Fail on Oct 16 → close; there is no time afterwards for a paired fit, a rehearsal and a hidden run.

### Step 7 — Freeze and select (Oct 20–22; Claude and Codex; the user selects)

- Oct 20 16:00 UTC: last new hidden submission launched. Oct 21: confirm both chosen entries COMPLETE in the official API, select the two finals, update STATUS and the handoff with submission IDs and the selection rule applied. Oct 22: buffer only; no code, no submissions. Deadline 23:59 UTC (02:59 TRT Oct 23).

### Calendar (UTC)

| Day | Plan |
|---|---|
| Oct 14 | Step 0 read-out; step 1 read-out; step 3 header kernel launched; decision log. |
| Oct 15 | Step 2 full-data arm if it4 passed; step 3 fold-0 test; step 5 design if flip won. |
| Oct 16 | Codex integration and tests (steps 2–3); M4 kill date; provenance re-check. |
| Oct 17 | Quota refresh; 2×T4 rehearsal of A′ (and B1′); A′ hidden submission. |
| Oct 18 | A′ scores; B1′ hidden submission; step 5 read-out if run. |
| Oct 19 | B1′ scores; one re-run slot reserved for a hidden-run error. |
| Oct 20 | Last submissions by 16:00 UTC; provenance re-check. |
| Oct 21 | Verify, select finals, STATUS/handoff. |
| Oct 22 | Buffer. |

### What this plan drops

The coronal specialist, R7/Raptor teachers, Dread-init students, native-resolution rebuilds, OAI pretraining, item 5 (two-slice pooling) and public inference tweaks stay closed. The own-model lane closes after iteration 4 unless the flip wins; the remaining Claude time goes to entry-level integration, rehearsals and provenance, which is where expected rank is actually won.

## 4. (c) The single next action for Claude, now

While iteration 3's K42 evaluation finishes and iteration 4 builds, **launch the step-3 Kaggle CPU header kernel**: one header per series over `train_series` and `test_series`, the per-study metadata table described above (no `PatientID`), and the train-vs-test tag-set assertion. It costs no GPU quota, takes under an hour, does not touch either A100, and unblocks the only new idea in this plan. Then run the step-0 decision tree at the read-out and record it.
