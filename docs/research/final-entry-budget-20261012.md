# Final-entry runtime and calendar budget (Oct 12, read-only)

Labels: **[M]** measured in a receipt, **[D]** derived from receipts, **[E]** my estimate. Hidden test ~1,300 studies, 2xT4, 9 h limit. Correction: the visible test has **3** studies; the "39-study" rehearsals are 3 visible + 36 training probes.

## 1. Per-arm runtime evidence

| Item | Number | Source |
| --- | --- | --- |
| Recovery rehearsal, 39 studies | 1,573.63 s [M] (visible-only pure entry: 170.17 s) | `outputs/analysis/s6-capped-recovery-20261012/parent-terminal-review-recovery-v1.json` |
| Production time, slower shard (~20 studies) | original320 107.8 s; S6 516.9 s; fixed phase overhead 483 s + 452 s [M] | same |
| Decomposition of 5.97 h base | scale 32.93 x slower-shard production: original320 **0.99 h**, S6 **4.73 h**, overhead 0.26 h [D, reproduces 5.97 exactly] | same receipt, `projection` |
| S6, full 870 studies, one board | 21,563 s = 24.8 s/study; 1,300 studies = 8.95 h on one board [M/D] | `docs/experiments/s6-fixed-evaluation-result-20261011.md` |
| Original320 K94 mirror alone | sequential p95 projection 4,378 s for 1,300 [D] | `docs/experiments/mirror-rehearsal-20261009.md` |
| DICOM decode, 96-slice reader (CPU) | 1.4 s/study warm, 2.05 s/study (32-study pilots) [M]; external claim that header reads dominate is unmeasured here | `docs/experiments/source-span-pilot-20261012.md`, `public-rich-cache-pilot-20261010.md` |
| Own224 K16, fold0, 870 studies | 283.49 s total (0.33 s/study), cached input, no DICOM decode [M] | `outputs/kaggle/own224-window-diagnostic-20261012-result-v1/own224_window_diagnostic/receipt.json` |
| K-window scaling | **Not measured.** Same-run K16/K42 job 138336842 is still RUNNING (since 14:49 UTC, ~1 h versus 10-20 min planned), so linear scaling is unverified | `STATUS.md` |
| Own224 training | ~3.6 h per fold on T4 before 96-slice I/O [E from history] | `docs/experiments/own-base-gap-audit-20261012.md` |

## 2. Hidden-test projection [E unless noted]

Assumptions: arms run as sequential phases split across both T4s. Decode uses 2.05 s/study over 2 workers, once for all folds (0.37 h). Setup is 0.05 h. Forward is 0.33 s/study per fold at K16, scaled linearly by 94/16 for K94 (likely optimistic) and doubled for mirror. "Conservative" is 8.19 h [M] plus 1.25 x the own-arm increment.

| Configuration | Own-arm increment | Base | Conservative | 9 h verdict |
| --- | --- | --- | --- | --- |
| Current entry | 0 | 5.97 h [M] | 8.19 h [M] | fits |
| + K16, 1 fold | 0.48 h | 6.45 | 8.78 | fits |
| + K16, 1 fold, mirror | 0.54 h | 6.51 | 8.86 | fits |
| + K16, 5 folds | 0.71 h | 6.69 | 9.08 | conservative marginal |
| + K16, 5 folds, mirror | 1.01 h | 6.98 | 9.45 | conservative exceeds |
| + K94, 1 fold | 0.77 h | 6.74 | 9.14 | conservative marginal |
| + K94, 1 fold, mirror | 1.11 h | 7.09 | 9.57 | conservative exceeds |
| + K94, 5 folds | 2.15 h | 8.12 | 10.87 | conservative exceeds |
| + K94, 5 folds, mirror | 3.88 h | 9.85 | 13.03 | exceeds on base |

To fit the exceeding rows, drop S6 (saves 4.73 h). K94 x 5 folds + mirror then totals about 5.1 h base [E]. Dropping original320 saves only 0.99 h and is not enough. An own-model-only entry (no original320 or S6) with K94 x 5 + mirror is about 3.9 h. Measure the real K94 cost before trusting any K94 row.

## 3. GPU quota

Live `kaggle quota` at 15:46 UTC [M]: 44.41/45 h used, **0.59 h remaining** (not 1.6 h), refreshAt **2026-10-17T00:00:00** (Saturday, matches `docs/claude-astra-handoff.md:115`; time zone presumed UTC). Used time rose 43.35 h (14:00) to 44.41 h (now) with job 138336842 running, so if it is billed by wall time it exhausts the quota near 16:20 UTC. Hidden-test reruns appear not to draw on the quota [D]: 43.14 h before the entry launch, about 43.19 h after, despite a >=6 h dual-T4 hidden rerun. One rehearsal plus visible entry costs about 0.44 + 0.05 = 0.5 h, so 0-1 fit now (3 if 1.6 h were real). After Oct 17 there are 45 h through Oct 22, and wall-clock is the constraint.

## 4. Calendar sketch (UTC) [E]

| Date | Plan |
| --- | --- |
| Mon Oct 12 | Same-run K16/K42 receipt, if quota allows. The four CPU cache shards finish. Stage and start fold-0 96wide on Colab Pro (browser must stay open; session loss was seen Oct 9). About 3.6 h on T4, less on A100 [E]. |
| Tue Oct 13 | Fold-0 result at K16 (primary) and K94, gate +.005 macro with CI above 0 against own224 fold0 (0.9018). Start fold 1. |
| Wed Oct 14 | Fold-1 replication. If both pass, queue folds 2-4; else keep the current entry. |
| Thu Oct 15 | Entry/merge deadline. Solo account already entered, so only confirm no merger is wanted. It does not restrict submission content or later submissions. |
| Fri Oct 16 | Colab Pro+ (background execution) begins; folds 2-4 unattended. Kaggle quota near zero. |
| Sat Oct 17 | Quota reset to 45 h. Build own-arm notebook, rehearsals (about 0.5 h each), measure K94 and mirror. Folds can also use two Kaggle T4s. |
| Oct 18-20 | LB probes (5/day): own arm alone, then blend with the current entry; final candidates submitted by Oct 20 with retry margin. A hidden run takes about 7 h to score (57096002: 04:53 to 12:11). |
| Oct 21 | Freeze. Select A = best-LB (57096002 or the blend) and B = own-model-heavy/no-OAI. |
| Thu Oct 22 | Last full-length submission must start by about 12:00 UTC; selections locked before 23:59 UTC. |

## 5. Revised resources and calendar after Pro+ (update at 16:00 UTC, Oct 12)

The user reports Colab **Pro+ active now with background execution and 850.56 compute units**, all authorized for this competition. This supersedes the October 16 Pro+ start in section 4. The unit rates below are typical published Colab rates and are not verified for this account: A100 about 12 to 13 units/h, L4 about 5, T4 about 2.

| Resource | Capacity | Notes |
| --- | --- | --- |
| Colab Pro+ | 850 units, about 65 to 70 A100-hours | User-supplied plan text: background execution up to 24 h per session, which survives a closed tab or a sleeping computer; priority A100 access; high-memory runtimes; units valid 90 days. The concurrent GPU session limit is not stated; probe it with the first two sessions. |
| Kaggle GPU | 15.02 h remaining of a raised 60 h weekly total at 16:21 (was 0.53 of 45 h at 15:50); refresh 2026-10-17 00:00 UTC | Rehearsals and visible runs are possible before Oct 17. The user attributes the increase to the Colab Pro+ activation (about 15:54), which fits the timing; whether 60 h is the new weekly total after the refresh is unverified. |
| Training cost per 96-wide 224 fold [E] | About 3.6 T4-h. On A100 about 1.2 to 1.8 h if the loader keeps up, roughly 15 to 25 units | Five folds about 100 units. A second recipe (Raptor-matched: longer schedule, mirror augmentation) at five folds is about another 150 units. Both fit comfortably. |

Revised calendar (estimates):

| Date (UTC) | Plan |
| --- | --- |
| Mon Oct 12 evening | Cache shards finish; stage to Colab. Codex launches fold-0 96-wide on A100. Once the pipeline is healthy, Claude launches the predeclared fold-1 replication from Codex's fold-1 package. |
| Tue Oct 13 | Fold-0 gate (K16 primary, +.005 with CI above 0 against own224 fold 0). Read fold 1 only if fold 0 passes. If both pass, launch folds 2 to 4 in parallel the same day. Run K94 and mirror secondaries (Claude's module) on the fold-0 checkpoint. |
| Wed Oct 14 | Folds 2 to 4 finish; pooled five-fold own-model OOF (encoder unexposed to its held-out fold, but scored on report-derived labels, with fold 0 reused for repeated decisions, so not clean validation). Start the Raptor-matched variant if the base passed. Begin the entry notebook (own arm at K16 × 5 folds plus original320, with or without S6) and CPU packaging. |
| Thu Oct 15 | Entry/merge deadline: solo account, confirmation only. Freeze the own-arm checkpoints and a blend rule declared in advance (own-arm weight fixed, not tuned on LB). The pooled OOF informs the rule but carries the report-label and repeated-selection limits above; blends with public arms cannot be validated locally. |
| Fri Oct 16 | Offline CPU rehearsal of the packaged entry with fixtures. Prepare two or three candidate entries for Oct 17. |
| Sat Oct 17 | Kaggle quota resets at 00:00. Rehearsals and visible runs, then submit two or three candidates (the hidden score takes about 7 h). |
| Sun Oct 18 to Mon Oct 19 | Read LB results and submit at most two or three refinements per day. Keep a budget of about 10 Kaggle GPU-h for visible runs. |
| Tue Oct 20 | Last new candidates submitted, leaving retry margin for a failed hidden run. |
| Wed Oct 21 | Freeze. Choose A = best LB, and B = own-model-heavy without OAI. |
| Thu Oct 22 | Selections locked. Only a retry of an already qualified entry is allowed, starting by about 12:00 UTC. |

The main schedule risk is hidden-test runtime. Section 2 shows that K94, mirror or extra folds on top of S6 exceed 9 h, so the own arm at K16 is the default in any entry that keeps S6.

**Qualifications (16:05 UTC, after Codex review).**
- All runtime, unit-burn and A100 figures are estimates until measured in actual sessions.
- The 5.97 h and 8.19 h projections are Codex's recorded projections; my per-arm split was reconstructed from receipt production seconds.
- The visible test has 3 studies; the 39-study rehearsals add 36 training probes.
- Any comparison involving public-model arms on the 870 development studies is exposed.
- Five-fold pooling of the own model does not make its OOF clean validation.
