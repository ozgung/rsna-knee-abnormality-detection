# Project status and session handoff

Updated **2026-10-20 15:30 UTC** (Claude, solo; Codex unavailable until Oct 22). Read this snapshot first, then the [shared handoff](docs/claude-astra-handoff.md). Rules are in [AGENTS.md](AGENTS.md) and setup in [README.md](README.md). The previous detailed snapshot, with its recovery records for older jobs, is in the [archive](docs/status-archive/status-before-20261019-final-push.md).

## Current snapshot

- **Deadline:** Oct 22 23:59 UTC (verified via the Kaggle API on Oct 19). 5 submissions per day, 2 final picks. Plan for the last new submission at **Oct 22 10:00 UTC** and lock the selection by 22:00, because a hidden run still scoring at the deadline may not count.
- **Scored entries (public):**

  | Entry | Submission | Public |
  |---|---|---|
  | P2 (R6F-pure v2) | 57340072 | **0.958** |
  | A′-w | 57324282 | 0.957 |
  | P3 / P4 / P5 | 57366765 / 57368819 / 57368388 | 0.957 each |
  | A′ | 57229227 | 0.956 |
  | V2 and S6fall solos | 57285381 / 57285382 | 0.950 |
  | B0′ | 57240823 | 0.944 |
  | 57326253 (crash before salvage) | 57326253 | 0.500 |

  A‴ 57288634 failed the runtime limit.
- **Board (Oct 20 15:06):** 6,270 teams; 211 at ≥ 0.959 (all private recipes); 1,387 tied at 0.958 (ranks 212–1,598). Silver ≈ 314 (any 0.959 display), bronze ≈ 627 inside the tie. No public 0.959 recipe exists ([scan Oct 19](docs/research/public-scan-20261019.md); Oct 20 rescan of 60 + 80 kernels and 168 topics found nothing new).
- **Gap assessment (Oct 20, user question "what does 0.960 take?"):** [note](docs/research/silver-gap-assessment-20261020.md). The only ≥ 0.959 shot before the deadline is **P9**, P2 with the compact slot ½ rank(R6) + ½ rank(own S6 members); runtime forces R6 plain and no R224 unless a shared 384 + 320 px decode is built. Expected +0.001 to +0.003 true, P(display ≥ 0.959) 0.35–0.45. Optional parallel bet: labels v4 with out-of-fold own teachers (P ≈ 0.3 of a scored entry). **Awaiting the user's go; nothing launched.**
- **Plan:** [final-push plan](docs/research/final-push-plan-20261019.md) (Fable max). Lanes A–D run concurrently, and the selection rule is fixed in its §4.
- **User decisions (Oct 19):**
  - use all compute in concurrent lanes (Colab 547.87 units, Kaggle 54 h GPU / 20 h TPU);
  - no copying public recipes;
  - **Lane D approved**: fastMRI-derived student weights may enter a submission.

## Lanes (all running, Claude subagents in their own worktrees)

| Lane | Worktree / branch | Resource | Goal and deadline |
|---|---|---|---|
| A: P6 dual-compact | merged 14f85a6 | Kaggle GPU | **P6s′ SUBMITTED as 57414571** (Oct 20 00:17). Shared decode, no R224. Point 25,507 s (P6s with R224 was 27,183 s); visible PASS, fallback 0. Studies the guard cuts from R7 take R6 plain. **Public 0.957**: under the rule it does not replace P2; it is the final-2 candidate (ρ to P2 is 0.994, so little diversity). Lane finished. Gate: point ≤ 26,000 s (else P6s′ without R224 if ≤ 29,000 s); no submission after Oct 21 18:00. Earlier: STOPPED at the kill rule. Rehearsal point 32,085 s (P6′ replay 29,832 s), above the 26,000 s limit. Spearman of P6 vs P2 is 0.994, so there is little diversity. No entry and no slot used. [Note](docs/experiments/lane-a-p6-20261019.md) |
| B: R6/R7 out-of-fold columns | merged e483f37 | Colab #1, VM stopped 22:09 | **DONE.** Fold mapping confirmed (8/8). On 4,349 out-of-fold studies, dual − R6 flip = +0.0029 [+0.0019, +0.0039] (single-model columns; gold-58 4-model: −0.0022 [−0.0068, +0.0028]). Labels v3 READY (gold-58 0.9254 vs v2 0.9202, exposed). Colab rate 5.3 units per A100-h. [Note](docs/experiments/lane-b-oof-20261019.md) |
| C: S6 seeds → solo → P7′ | merged ed801d0; VMs stopped | Colab #2 + Kaggle | **DONE.** v3-s42 (gold 0.9283) and v3-s43 (gold 0.9380) exported privately; P7′ compact = mean rank of {v3-s42, v3-s43, fastMRI student}. **C1 solo 57432734 = 0.955; P7′ 57432784 = 0.955.** [Note](docs/experiments/lane-c-seeds-20261019.md) |
| D: fastMRI noisy student | merged 8f21dca; VM stopped 06:33 | Colab #3 + Kaggle | **DONE.** All gates passed; 745 fastMRI exams got P2-teacher rank-mapped targets. Student = S6 on labels v3 + fastMRI, private dataset `ozgung/knee-laned-fmall-s42-20261020`. Gold-58 0.9350 (exposed). **Solo 57426580 = 0.954** (S6 solos 0.950). Added to P7′'s compact by a parent amendment (08:40). **D2 DROPPED (10:20):** the agent's policy refused the T4-label shim even after the user approved it, and refused a direct fp16 compact_infer call too. VM laned2 stopped 09:20 (about 0.7 A100-h); no artifacts. Recommendation adopted: keep student 1 only, protect P7′'s Kaggle slots. [Note](docs/experiments/lane-d-fastmri-20261019.md) |
| E: local (parent) | main checkout | Mac | Oct 20 provenance recheck **DONE, not adverse** ([note](docs/research/provenance-refresh-20261020.md)); cross-entry Spearman / E[max] calculator; integration of lane branches |

Lanes write large outputs to `outputs/lane{a,b,c,d}-20261019/` and commit only in their own branch. The parent integrates their branches and updates this file and the handoff. Each lane stops its own VM when it finishes. If a lane agent is lost, stop the VM it was using with `colab --config ~/.config/colab-cli/sessions-<lane>.json stop -s <name>` (the session name is in the lane's note/log) before relaunching.

## Selection rule (fixed in the plan, §4)

- **Final 1:** P2, unless a challenger displays ≥ 0.958, passed every gate except the projection with zero visible fallback, and wins Lane B's out-of-fold comparison (Δ ≥ +0.0003, CI excluding 0).
- **Final 2:** P6 if it displays ≥ 0.957, else P7′ if ≥ 0.957, else A′-w. Rule 2 gives P6s′; the parent recommends P7′ instead as the only all-ours-compact hedge at the same zero E[max] cost ([evidence](docs/experiments/final-selection-20261020.md)). **User (Oct 20 15:45): decide on Oct 21; Claude sets the two checkboxes once chosen.** Checked Oct 20: the Kaggle CLI/API has no final-selection command and this session has no browser tool, so the checkboxes need the Kaggle UI (user, or a browser-capable session).
- **Provenance override (Oct 20 recheck):** if the recheck turns adverse, final 2 becomes P7′, or 57037296 if OAI is also ruled out.

## Verification and invariants

The last parent checks covered dependency-free unit tests and smoke at the R6F integration (Oct 18). Lane branches report their own tests, and the parent reruns them on merge. Keep patient/study grouping limits explicit, gold-58 exposure logged, image-only inference, deadline guard plus salvage-before-constant on every entry, and never use `kaggle kernels push --timeout`.

**Next (parent):** user decides whether to launch P9 (and the optional labels-v4 bet) per the [gap assessment](docs/research/silver-gap-assessment-20261020.md); if yes, start the seed-44/45 fits and the P9 variant build at once (submission target Oct 21 03:00 UTC). Otherwise: user decides final 2 on Oct 21 (P6s′ by rule, P7′ as the hedge); then set the two final-selection checkboxes and record it here. No lane is running; all VMs are stopped. Slots used today: 4 of 5.
