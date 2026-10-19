# Project status and session handoff

Updated **2026-10-19 22:30 UTC** (Claude, solo; Codex unavailable until Oct 22). Read this snapshot first, then the [shared handoff](docs/claude-astra-handoff.md). Rules are in [AGENTS.md](AGENTS.md) and setup in [README.md](README.md). The previous detailed snapshot, with its recovery records for older jobs, is in the [archive](docs/status-archive/status-before-20261019-final-push.md).

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
- **Board (Oct 19 16:15):** 6,218 teams; 64 at 0.959; 1,259 tied at 0.958 (ranks 186–1,444). The bronze line (about rank 622) falls inside that tie. No public 0.959 recipe exists ([scan](docs/research/public-scan-20261019.md)).
- **Plan:** [final-push plan](docs/research/final-push-plan-20261019.md) (Fable max). Lanes A–D run concurrently, and the selection rule is fixed in its §4.
- **User decisions (Oct 19):**
  - use all compute in concurrent lanes (Colab 547.87 units, Kaggle 54 h GPU / 20 h TPU);
  - no copying public recipes;
  - **Lane D approved**: fastMRI-derived student weights may enter a submission.

## Lanes (all running, Claude subagents in their own worktrees)

| Lane | Worktree / branch | Resource | Goal and deadline |
|---|---|---|---|
| A: P6 dual-compact | merged 0c49b21; reopened 22:30 | Kaggle GPU | **REOPENED as P6s (shared decode)** after Lane B's out-of-fold win. Gate: point ≤ 26,000 s (else P6s′ without R224 if ≤ 29,000 s); no submission after Oct 21 18:00. Earlier: STOPPED at the kill rule. Rehearsal point 32,085 s (P6′ replay 29,832 s), above the 26,000 s limit. Spearman of P6 vs P2 is 0.994, so there is little diversity. No entry and no slot used. [Note](docs/experiments/lane-a-p6-20261019.md) |
| B: R6/R7 out-of-fold columns | merged e483f37 | Colab #1, VM stopped 22:09 | **DONE.** Fold mapping confirmed (8/8). On 4,349 out-of-fold studies, dual − R6 flip = +0.0029 [+0.0019, +0.0039] (single-model columns; gold-58 4-model: −0.0022 [−0.0068, +0.0028]). Labels v3 READY (gold-58 0.9254 vs v2 0.9202, exposed). Colab rate 5.3 units per A100-h. [Note](docs/experiments/lane-b-oof-20261019.md) |
| C: S6 seeds → solo → P7′ | `.claude/worktrees/lanec-20261019` | Colab #2, `sessions-lanec-20261019.json`; Kaggle | seeds 43/44 on labels v2 now, then v3 or seed 45; P7′ all-ours compact submitted by Oct 21 18:00 |
| D: fastMRI noisy student | `.claude/worktrees/laned-20261019` | Colab #3, `sessions-laned-20261019.json` | go/no-go Oct 20 06:00; student export by Oct 21 20:00; P8 decision with the parent |
| E: local (parent) | main checkout | Mac | Oct 20 provenance recheck; cross-entry Spearman / E[max] calculator; integration of lane branches |

Lanes write large outputs to `outputs/lane{a,b,c,d}-20261019/` and commit only in their own branch. The parent integrates their branches and updates this file and the handoff. Each lane stops its own VM when it finishes. If a lane agent is lost, stop the VM it was using with `colab --config ~/.config/colab-cli/sessions-<lane>.json stop -s <name>` (the session name is in the lane's note/log) before relaunching.

## Selection rule (fixed in the plan, §4)

- **Final 1:** P2, unless a challenger displays ≥ 0.958, passed every gate except the projection with zero visible fallback, and wins Lane B's out-of-fold comparison (Δ ≥ +0.0003, CI excluding 0).
- **Final 2:** P6 if it displays ≥ 0.957, else P7′ if ≥ 0.957, else A′-w.
- **Provenance override (Oct 20 recheck):** if the recheck turns adverse, final 2 becomes P7′, or 57037296 if OAI is also ruled out.

## Verification and invariants

The last parent checks covered dependency-free unit tests and smoke at the R6F integration (Oct 18). Lane branches report their own tests, and the parent reruns them on merge. Keep patient/study grouping limits explicit, gold-58 exposure logged, image-only inference, deadline guard plus salvage-before-constant on every entry, and never use `kaggle kernels push --timeout`.

**Next (parent):** integrate lane reports as they arrive; run the Oct 20 provenance recheck; recompute E[max] with measured ρ before the Oct 21 selection.
