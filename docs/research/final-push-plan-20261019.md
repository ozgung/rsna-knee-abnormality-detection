# Final-push plan — 2026-10-19 (data as of 16:45 UTC)

Author: Fable (Claude), strategy lead, read-only. No training, kernel push, submission, Colab VM, download or commit was made; this file is the only repo output. Labels: **[measured]** = a number in a receipt, CSV, API response, rules capture or pinned source; **[literature]** = another participant's claim; **[estimate]** = my judgement. Times are UTC (user: UTC+3).

## Verdict

1. **Position.** P2 (57340072, 0.958) is a unique prediction vector whose expected private score equals the public 0.958 recipe family's, minus about 0.0003–0.0005 for the two public knobs we do not run (R224 mirror, MIL reader) [estimate]. The bronze line (rank ≈ 622 of 6,218) falls inside the 1,259-team 0.958 block, whose members are mostly byte-identical copies that will share one private score each. Our bronze odds are therefore a coin flip decided by our own residual: **P(bronze) ≈ 0.4–0.5, P(silver) ≈ 0.02** [estimate, §1]. A true +0.0005 over the copy cluster would lift P(bronze) to about 0.75; that is the only prize within reach, and no lane below delivers it with more than about 0.45 probability.
2. **"Keep P2 + A′-w" is a sound floor, not an optimum.** A′-w adds ≈ +0.00001 to E[max(final1, final2)]; P3/P4/P5 add less (they are 0.99-correlated copies of P2 at a lower display). The second final should be an entry with P2's expected score and a different compact slot, which does not exist yet: **Lane A (P6 "dual-compact": R6 plain + R7 plain instead of R6 flip, runtime-neutral)** is both the best final-2 candidate (+0.0002 on E[max] at equal quality, §3) and the only cheap candidate with a positive expected gain for final 1 (+0.0002, range −0.0004…+0.0008 [estimate]).
3. **Validation without the public board is possible for the compact slot.** The pinned `compact_train.py` assigns folds by `sha1(uid)[:8] % K` over the non-gold pool and a fold-k model never sees fold k [measured, lines 59/2015]. Running the R6 and R7 fold models on their own held-out hash folds yields genuine out-of-fold columns on 4,344 training studies (Lane B), on which "R6 flip" vs "R6 + R7" vs "R6 plain" can be compared with a paired SD of about 0.0003 — ten times sharper than the public board. Nothing equivalent exists for the Raptor half (trained on every study), so the Raptor half stays frozen.
4. **Own-model lanes (C, D) cannot raise final 1 by Oct 22 in expectation.** Our best compact model is 0.950-class next to a 0.958-class blend; goodpjw measured S6 next to R6 at −0.001 [literature]. Lane C (V3 seed ensemble on labels v3) is run as the provenance hedge and the decorrelated final-2 fallback, not as a score bet. Lane D (fastMRI noisy student) is the only lane with a real upside and is a 0.2-probability moonshot with provenance and disk constraints; it runs only on a third Colab session and only with the user's explicit data-use decision.
5. **Clock.** Deadline **2026-10-22 23:59:00** [measured, Kaggle API]; **5 submissions/day, 2 finals** [measured, rules capture §2]. Whether a notebook still scoring at the deadline counts is not in the rules body; the community reading is that it is treated as late (§0). Plan conservatively: **last new submission Oct 22 10:00 UTC, retry of an already-qualified packet no later than 12:00, selection locked by 22:00.**

## 0. Facts verified today

| Item | Value | Source |
|---|---|---|
| Deadline | 2026-10-22 23:59:00 (Kaggle lists UTC) | `kaggle competitions list -s rsna-knee` [measured] |
| Daily limit / finals | "maximum of five (5) Submissions per day"; "select up to two (2) Final Submissions" | `outputs/provenance-20261016/rules-body-normalized.txt` §2 [measured; body unchanged since Oct 13] |
| Running at deadline | Not in the rules body. Kaggle forum readers say a notebook still scoring at the deadline is treated as a late submission and not ranked (kaggle.com/general/532407; community, not staff) | [literature, unverified] |
| Board (16:15 UTC) | 6,218 teams; 0.969 ×1 … 0.960 ×39 (83–121), 0.959 ×64 (122–185), **0.958 ×1,259 (186–1,444)**, 0.957 ×183, 0.956 ×310, 0.950 ×132. Our API rank 1,429 (0.958, late in the tie). Bronze top-10 % ≈ 622, silver ≈ 311, gold ≈ 22 | `outputs/analysis/lb-20261019/*.csv`, API [measured]; medal fractions [literature] |
| Our scored entries | P2 0.958; A′-w, P3, P4, P5 0.957; A′ 0.956; V2/S6fall solos 0.950; B0′ 0.944; 57326253 0.500 (pre-salvage crash); A‴ runtime failure | `kaggle competitions submissions` [measured] |
| Hidden walls | P2 ≤ 8.1 h (05:20 → scored by 13:25); A′ ≤ 7.65 h; solo S6 ≤ 2.35 h; A‴ > 9 h killed | submission/score times [measured] |
| Slow-board production (P5 rehearsal, 20 studies) | R384 two-view 5.83 s/study (+517 s fixed); na-full 4.1 s; **R6 flip 31.0 s/study**; setup 27 s. P5 point 28,913 s / conservative 38,807 s; P2 v3 point 24,930 s / conservative 33,474 s | `outputs/kaggle/r6flip-p5-rehearsal-20261018-out/r6flip/receipt.json`, handoff [measured] |
| Fold rule | `pool_fold = int(sha1(uid).hexdigest()[:8], 16) % K` over the non-gold pool; "A fold-k model never saw fold k's studies and the hash folds are the same"; R6/R7 manifests: folds 4, gold_in_training 0 | `outputs/research/public-0958-20261017/r6-v1-sources/compact_train.py:59,111,2015` [measured] |
| Colab | Two A100 sessions ran concurrently on Oct 16 (endpoints …2eaiu47jk8yg9 and …4avczupgxru7, 14:28–16:09); a third is unverified. 547.87 units (user). Token expiry 3,600 s: `outputs/s6v2/vm/refresher.sh` → `tools/reconnect.py` every 30 min. VM disk ≈ 185 GB free at start; the competition cache is 133 GB | `~/.config/colab-cli/colab.log` [measured]; disk [measured Oct 15] |
| Kaggle | 54 h GPU, 20 h TPU this week (user); hidden reruns did not draw on the quota on Oct 12 [measured once]; no XLA/TPU code path exists in the repo | `rg` over `knee/`, `scripts/` [measured] |

## 1. Position and the two finals

**Private ordering model.** Private ≈ 70 % of about 1,300 studies (≈ 910) [literature]; per-entry sampling SD ≈ 0.0023 at that size [measured on the 870, lane 3]. Teams running byte-identical public notebooks get byte-identical private scores and are ordered by submission time [assumption]. The 0.958 block therefore contains a few large clusters (blend-mine v1 weights; heliosli V2 weights + reader; goodpjw mirror + reader) plus unique entries. P2 differs from the blend-mine cluster only by our own R384/R224 readers (same weights) and the absent mirror/reader, so P2 − cluster = ε, ε ≈ N(−0.0002, 0.0004) [estimate]. Teams above the block on public (185) keep a true lead in perhaps half the cases; some 0.956–0.957 teams with different models will jump above. The cluster then starts near private rank 200–350 and ends near 1,200, with the bronze line inside it: **P2's bronze probability is essentially P(ε > 0) ≈ 0.4–0.5**; with a true +0.0005 it is ≈ 0.75–0.8; silver needs ≈ +0.0015 (P ≈ 0.05 even if a lane succeeds).

**Value of the second final.** Kaggle ranks the better private score of the two picks. With paired SD s = 0.0023·√(2(1−ρ)) and expected deficit d, the gain is s·φ(d/s) − |d|·(1−Φ(|d|/s)) [lane-3 model]:

| Final-2 candidate beside P2 | d | ρ | s | E[max] gain | Verdict |
|---|---|---|---|---|---|
| A′-w 57324282 (R7 no flip, v1 weights) | −0.0010 | 0.97 | 0.00056 | +0.00001 | floor only |
| P3 / P4 / P5 (na-full variants) | −0.0005 | 0.995 | 0.00023 | +0.000001 | dominated |
| **P6 dual-compact (Lane A)** | 0 ± 0.0004 | 0.97 | 0.00056 | **+0.00022** | build |
| P7′ all-ours compact (Lane C) | −0.0012 | 0.93 | 0.00086 | +0.00003 | hedge only |
| Solo V3 (Lane C) | −0.007 | 0.90 | 0.0010 | 0 | never a final |

**Recommendation.** Final 1 = P2 unless P6 displays ≥ 0.958 *and* wins the Lane-B out-of-fold comparison (§4 rule). Final 2 = P6 if it displays ≥ 0.957, else P7′ if ≥ 0.957, else A′-w. If the Oct 20 provenance recheck turns adverse for heliosli's compact bundles, final 2 = P7′ regardless of display; if adverse for OAI as well, final 2 = 57037296 (own CoAtNet224, 0.914, no external data), as pre-planned in the shake-up note.

What we deliberately do not do: verbatim public forks, the R224 mirror and MIL reader (user decision; expected cost ≈ 0.0003–0.0005 private, stated for the record), per-finding weight fitting on gold-58 or the 870 (closed rule), a TPU port (no XLA path; > 3 days of risk), paid provisioning or new data terms.

## 2. Lanes (all concurrent)

| Lane | Resource | What | Expected private effect | P(success) | Wall / compute | Latest start |
|---|---|---|---|---|---|---|
| **A. P6 dual-compact** | Kaggle 2×T4 (1 rehearsal + 1 visible ≈ 1 h); local build | compact slot = ½ rank(R6 plain) + ½ rank(R7 plain) replacing R6 flip; everything else as P2; R7 stage last under the guard | final 1: +0.0002 (−0.0004…+0.0008); final 2: +0.0002 on E[max] | 0.45 (true gain), 0.85 (scored ≥ 0.957) | build 4–6 h, rehearsal 0.8 h, chain 1 h, hidden 7–8.5 h | build by Oct 21 20:00; submit by Oct 22 10:00 (target Oct 20 12:00) |
| **B. R6/R7 out-of-fold columns** | Colab A100 #1 (≈ 5 h; split 3.5 + 2 h if a 3rd session exists) | per-study held-out-fold predictions of R6 (plain, flip) and R7 (plain) on 4,344 pool studies + gold-58 | none directly; decides A's design; yields labels v3 teacher; measures R6 memorisation | 0.8 (fold mapping confirmed) | 1 h prep, 5–8 h run, 1 h CPU analysis | start by Oct 20 12:00 (target Oct 19 20:00) |
| **C. V3 seed ensemble → solo probe + P7′ all-ours** | Colab A100 #2 (4 × 3.3 h fits, 2 concurrent per VM ≈ 6.5 h); Kaggle 2 rehearsals + 3 visibles ≈ 2 h | S6 recipe on the 133 GB cache with labels v3 (fallback labels v2), seeds 42–45, all data; solo V3-s42 probe; P7′ = Raptor half + 4-seed compact (no flip) | solo +0.001…+0.003 over V2 (P ≈ 0.5); P7′ 0.956–0.957; provenance hedge and ρ ≈ 0.93 final-2 fallback | 0.7 | labels v3 → fits → export 9 h; P7′ stage code 6 h; rehearsal 0.8 h | fits by Oct 20 12:00; P7′ build by Oct 21 18:00; submit by Oct 22 10:00 |
| **D. fastMRI noisy student (optional)** | Colab A100 #3 only (download, cache, teacher inference ≈ 10 h; student fit 4 h); local plane classifier | teacher = R384 + R6 flip + R224 on ≈ 1,000 fastMRI batch-2 exams (out-of-sample for every teacher); student = S6 recipe on competition (labels v3) + fastMRI soft targets | if it lands: +0.0003…+0.001 as the compact co-member of R6 plain (P8) — P ≈ 0.3 given completion | 0.2 (complete by Oct 21 20:00) | 20–26 A100-h; 30 h wall | go/no-go Oct 20 06:00 (cache + classifier validated); student exported by Oct 21 20:00 |
| **E. Local Mac** | CPU | paired-bootstrap analysis of B's columns; cross-entry Spearman table from 39-study ledgers; E[max] selection calculator; Oct 20 provenance recheck; D's header→plane classifier validated on 24,371 train series; STATUS/handoff | — | — | continuous | — |
| Kaggle TPU | — | **no lane** (no XLA path; porting the trainer is > 3 days of risk) | — | — | — | — |

### Lane A — P6 dual-compact

- **Hypothesis.** Two different four-fold bundles (R6: pseudo-label round 6, no extent head; R7: fastMRI auxiliary, extent head, different stackers) averaged at rank level reduce variance more than mirrored TTA of one bundle. Public evidence is a wash (goodpjw v26: R7 at 0.4 beside R6 flip displayed 0.958 = v25; v23: flip on R7 too = 0.957 [literature]); flip alone was worth ≈ +0.0003 (0.9573 → 0.9576). Prior: Δ = +0.0002 ± 0.0005 vs P2.
- **Runtime.** R6 flip measured 31.0 s/study on the slow board; R6 plain + R7 plain ≈ 15.5 + 15.5 s plus a second decode pass hidden behind the two loader workers (CPU ≈ 3 s/study wall per pass vs ≥ 15 s GPU) [estimate]. Point ≈ 25,000–25,600 s ≈ P2's 24,930 s; actual ≈ 8.1–8.5 h, so the guard may trim the tail of the R7 stage on a slow host: trimmed studies fall back to R6-only ranks (loss < 0.0001 per 10 % trimmed [estimate]). Fallback config P6′ drops R224 (−0.6 h) if the rehearsal point exceeds 26,000 s.
- **Validation (no public board needed).** (1) Lane B columns: on 4,344 out-of-fold studies, compare ½rank(R6)+½rank(R7) against rank(R6 flip) and rank(R6 plain) against `labels_v1_blend ≥ 0.5` masked truth, paired study bootstrap 2,000 draws seed 42 (`knee/paired_validation.py`); pre-registered rule: adopt P6 for final 1 only if Δ(dual − R6 flip) ≥ +0.0003 with CI excluding 0; adopt for final 2 if Δ ≥ −0.0003 (non-inferior). (2) Gold-58 sanity from Lane B (R6 flip reproduces goodpjw's ≈ 0.937 macro; dual within 0.01). (3) Rehearsal: every gate except projection PASS, `fallback_studies` 0, per-finding Spearman of the dual compact against the R6-flip output on the 39 studies reported (expect 0.90–0.98). (4) Visible run: receipt PASS, `fallback_studies` 0, pulled code byte-identical.
- **Kill.** OOF non-inferiority fails; rehearsal point > 26,000 s after P6′; any non-projection gate fails; visible fallback > 0.
- **Build on.** `knee/public_r6flip.py` (`VARIANTS`, `ALL_ARMS`, blend; add `p6` with a sealed `compact_arms` list, weights fixed ½/½, flip off), `knee/public_r6flip_entry.py` (`launch_guarded`/stage order; `merge_r6` generalised to a second compact folder; R7 last), `knee/public_r6_profile.py` as the pattern for an R7 profile with the pins already in `knee/public_r7_recovery_entry.py` (`R7_DATASET='heliosli/rsna-knee-compact-r7cnxfsss'` v1, bundle `8c7b666e27dc0e8f`, `flip_tta False`), `scripts/build_r6flip_entry.py --variant p6` (rehearsal → verify → waiver → entry), chain pattern `~/.claude/jobs/74614526/tmp/vchain.sh`, tests `tests/test_public_r6flip.py` (add: stage-order, half-weights, R7 fallback-to-R6 path).

### Lane B — out-of-fold R6/R7 columns

- **Job.** On an A100 VM: stage the training archive by HTTP range reads per study (`scripts/s6v2/vm_build_cache.py` `Archive`/`Remote`/`fetch_study`), run heliosli's pinned `compact_infer.py` in-process (`scripts/s6v2/vm_gold_infer.py` pattern) with the R6 bundle (`outputs/research/public-0958-20261017/r6-v1-assets/`, manifest 6508ce69…) and the R7 bundle (dataset v1, bundle 8c7b666e…) at their own geometry (384/352, v4, 3 slices, 3.5 mm), **one fold model per study restricted to its hash fold** (the s6solo wrapper shows compact_infer accepts a single checkpoint), configs: R6 plain, R6 flip, R7 plain (R7 flip if time allows). Also all four models on gold-58 (gold is out-of-fold for every model). Keep raw logits; no IDs in Git.
- **Cost.** ≈ 1.3 s/study decode+render (the 320-px cache build ran 48 studies/min) + 0.7 s per plain forward, 1.4 s per flipped forward on an A100 [estimate] → ≈ 4 s/study → ≈ 5 h for 4,402 studies; 8 h with R7 flip.
- **Mapping check (first hour).** For each fold model, AUC on its own hash fold must sit clearly below its AUC on the other three folds (in-sample echo; Dread's echo was 0.95–0.99 agreement). If the gap is absent, the fold rule is misread (or a "ruler fold" was excluded from every training set, `compact_train.py:1875`; then that fold is out-of-fold for all models): stop and re-derive before any use.
- **Outputs.** (i) Lane A decision table; (ii) `labels_v3` = `mean6_plus_dread` base + R6-flip-OOF teacher at w = 0.30 (MCL 0.15), the labels-v2 construction with the teacher swapped (`outputs/labels/labels-v2-20261015/build_labels_v2.py`), one table, scored on gold-58 once and recorded as exposed; (iii) memorisation note (in-fold vs out-of-fold AUC per fold) for the write-up.
- **Kill.** Mapping unconfirmed by +4 h; R6-flip gold-58 macro < 0.92 (pipeline fault; goodpjw's table implies ≈ 0.937); labels v3 gold-58 < labels v2's 0.920.

### Lane C — V3 seed ensemble, solo probe, P7′

- **Hypothesis.** A stronger image teacher (R6 out-of-fold, gold-58 ≈ 0.93 per fold-model class) improves the label table more than own-CoAtNet-224 did (+0.019 on gold-58); seed averaging of four all-data fits adds +0.002–0.004 solo [estimate]. Neither reaches R6 quality; the entry value is a no-heliosli compact slot (provenance hedge) and a ρ ≈ 0.93 final-2 option.
- **Runs.** Restore the cache (`scripts/s6v2/vm_restore_px.py`, 393 s), targets from labels v3 (`scripts/s6v2/build_targets.py`/`check_targets.py`), `scripts/s6v2/vm_alldata.sh` per seed with `vm_bk_v2all_ckpt.sh` per-epoch backups; two fits concurrently per A100 (≈ 22 GB GPU, 6.0 h measured for a pair of 16-epoch fits); export each to a private dataset with SHA256SUMS; strict CPU state/constructor/repeat checks as for V2-all (`scripts/check_s6_solo_pair.py`). If labels v3 is not ready by Oct 20 02:00, start seeds 43/44 on labels v2 instead (seed-ensemble value does not depend on v3).
- **Entries.** C1 solo V3-s42 through the s6solo packet with the checkpoint swapped (`knee/s6_solo_pair.py` validation, hidden ≤ 2.35 h): tie-or-better against V2's 0.950. C2 P7′: Raptor half (R384 .65 + R224 .35, v1 per-finding weights as sealed config) + compact = mean rank of 4 seeds, no flip, 320/288, eval stride 1; new `s6v3` compact stage in the R6F controller reusing the s6solo compact_infer arguments; point ≈ 4.5 h, actual ≈ 5.3 h [estimate].
- **Validation.** Gold-58 per seed and for the 4-seed mean through the pinned DICOM → compact_infer path (V2 0.9294, S6fall 0.9314 as references; read once, no selection); rehearsal gates; visible run.
- **Kill.** Epoch-0 loss > 0.55 (V2: 0.5106); 4-seed gold-58 macro < 0.925; P7′ rehearsal point > 27,000 s; solo probe < 0.949 displayed (then P7′ still proceeds as a hedge only).

### Lane D — fastMRI noisy student (optional, third session only)

- **Preconditions.** The user's explicit decision on NYU fastMRI terms (research use, derived weights entering a competition submission; the host's accessibility test in 743416 accepts registration-only data as we read it, but upstream permission is the user's call); a third concurrent Colab session; Lane E's header→plane/fluid classifier validated at ≥ 98 % on the 24,371 competition train series (`outputs/analysis/site-prior-20261014/kernel-v1/series_headers.csv` against `train_series.csv`).
- **Scope.** batch 2 only (≈ 30 GB tar; exam count unverified, planned ≈ 1,000 exams so the 320-px cache (≈ 30 GB) fits beside the 133 GB competition cache on one VM). Teacher inference with our own readers on the A100 (R384 two-view + R224 + R6 flip ≈ 12 s/study ≈ 3.5 h); targets = per-finding weighted mean of teacher probabilities (v1 Raptor weights), no ranks. Student = S6 recipe, joint pass over competition (labels v3) + fastMRI (teacher targets at weight 0.5), 16 epochs ≈ 4 h; export; gold-58 read-out; one solo probe; if its gold-58 and solo beat V3, P8 = P6 with R7 replaced by the student (same runtime class).
- **Kill.** Download not complete by Oct 20 02:00; classifier < 98 %; cache audit (`knee.reference`-style sample of 50 studies) fails; epoch-1 K16 < 0.80; export not done by Oct 21 20:00. No fastMRI pixels, file names or links in Git.

### Resource budget

Colab: Lane B 5–8 A100-h, Lane C 13–14 A100-h (+1 h exports/gold), Lane D 20–26 A100-h → 40–50 A100-h, within 548 units whether the rate is 5–6 or 12–13 units/h (both figures are unverified for this account; check unit burn after the first hour). Kaggle GPU: ≈ 5 h of 54 h (rehearsals, visibles, solo probes); if only two Colab sessions are allowed, run Lane B's R7 half on 2×T4 instead (≈ 5 h kernel). Kaggle TPU: unused. Local: Lane E throughout.

## 3. Diversity for the second final

- Correlation is dominated by shared members: every 0.956–0.958 entry we can build shares the Raptor half (ρ ≥ 0.93); the compact slot is the only lever. P6 changes the compact slot without lowering expected quality; P7′ changes it at a −0.001 cost. P3–P5 and A′-w change too little or lose too much.
- Measure, do not assume: Lane E tabulates per-finding Spearman between every candidate's rehearsal output (39 studies: 3 visible + 36 train probes; ledgers under `outputs/kaggle/*-out/`) and, from Lane B, between R6-flip, dual and V3 columns on 4,344 out-of-fold studies. The E[max] table in §1 is recomputed with measured ρ before selection.
- Uniqueness is already secured: all our vectors differ from every public copy (own readers, own ranks). Do not resubmit any composition twice; a second identical vector ties itself.

## 4. Submission-slot schedule and selection rule

| Day (UTC) | Slots | Planned use |
|---|---|---|
| Oct 19 | 5 (0 used) | none planned; nothing qualified is ready |
| Oct 20 | 5 | P6 (target 12:00); solo V3-s42 (target 15:00); 3 spare (P6′ retry, Lane D probe) |
| Oct 21 | 5 | P7′ (target 04:00); P8 or a P6 variant only if Lane B/D justify it; keep ≥ 2 spare for retries |
| Oct 22 | 5 | last new submission 10:00; retry of an already-qualified packet ≤ 12:00; nothing after; **selection locked by 22:00** |

Every entry: deadline guard T0 + 8.3 h (180 s reserve, compact self-stop 60 s), salvage-before-constant, no `--timeout`, title = id, visible run with `fallback_studies 0` before submitting; user's "Submit regardless" covers projection-only failures of guarded builds.

**Selection rule (fixed now).**
1. Final 1 = P2 (57340072). A challenger replaces it only if it displays ≥ 0.958, passed every non-projection gate with zero visible fallback, and (for P6/P8) its Lane-B out-of-fold Δ against R6 flip is ≥ +0.0003 with the CI excluding 0. A displayed tie with no OOF win keeps P2.
2. Final 2 = the candidate with the highest E[max] gain (§1 table, recomputed with measured ρ): P6 if ≥ 0.957, else P7′ if ≥ 0.957, else A′-w (57324282).
3. Provenance override (Oct 20 recheck): adverse for heliosli compact bundles → final 2 = P7′ (≥ 0.955); adverse for OAI as well → final 2 = 57037296.
4. Never select: a 0.500, ERROR or runtime-failed run; a guarded run with > 10 % compact fallback; P3/P4/P5 (dominated).

## 5. Risks and guards

| Risk | Guard |
|---|---|
| 9 h runtime kill (A‴ precedent) | guard on every stage; cheapest-to-lose stage last (R7 in P6, na-full pattern); rehearsal point ≤ 26,000 s (P6) / 27,000 s (P7′); never submit an unguarded heavier build; hidden host variance ×1.17–1.3 is assumed, not measured |
| Crash to 0.500 (57326253 precedent) | `salvage_csv` blends every ledger row before the constant (proved by 57340072); atomic ledgers per stage; visible run must show `fallback_studies 0`; a crash costs a slot and a day, never rank (P2 stays) |
| Provenance | R6 `r6fm` tag / R7 `fm_aux` unresolved; OAI via nartaa; Oct 20 recheck (rules hash vs Oct 16 capture; threads 743416, 749858, 750236–750260); P7′ (no heliosli bundles) and 57037296 (no external data) are the hedges; Lane D only after the user's fastMRI decision |
| Leakage / exposure | Lane B's fold mapping verified by the in-fold vs out-of-fold gap before use; labels v3 fitted on non-gold rows only; gold-58 read once per artifact and logged as exposed; no weight fitting on the 870 or gold-58; P6's ½/½ compact weights are pre-registered; study-level hash folds (patient grouping impossible for public bundles — documented limitation) |
| Public-board over-reading | 0.001 displayed differences are inside the ≈ 0.003 public SD; decisions rest on Lane B's paired OOF comparisons and pre-registered rules, the board only confirms ≥ 0.957/0.958 class |
| Wall-clock | Colab token refresh every 30 min on every VM; per-epoch checkpoint backups; VM loss → restart from backup, never from scratch past Oct 21 12:00; Kaggle concurrent-GPU cap → stagger rehearsals; 5 slots/day; last submission Oct 22 10:00 |
| Tie ordering | unverified assumption that equal private scores are ordered by submission time; our vectors are unique, so it does not bind us either way |

## 6. Assumptions not verified

Public/private split 30/70 and public SD ≈ 0.003 [literature]; late-scoring rule (community reading); medal cut-offs (top 10 % / 5 %) and tie ordering; a third concurrent Colab session; Colab units per A100-hour (5–6 per the user vs 12–13 in the Oct 12 note); VM disk 185 GB; Kaggle hidden reruns not drawing on quota (observed once); the 54 h / 20 h Kaggle figures (user-supplied); R6/R7 fold rule on study uid with K = 4 and no ruler fold (verifiable in Lane B's first hour); R6 plain ≈ half of R6 flip cost; fastMRI batch-2 exam count and tar layout; P2's hidden run completed without a guard cut (inferred from its 0.958 display); private transfer of the public per-finding weights.

## Evidence

`outputs/analysis/lb-20261019/rsna-knee-abnormality-detection-publicleaderboard-2026-10-19T16:15:09.csv`; `kaggle competitions list/submissions` (16:50 UTC); `outputs/provenance-20261016/rules-body-normalized.txt`; `outputs/kaggle/r6flip-p5-rehearsal-20261018-out/r6flip/receipt.json`; `outputs/research/public-0958-20261017/r6-v1-sources/compact_train.py`; `~/.config/colab-cli/colab.log`; `docs/research/strategy-revision-20261017.md`, `public-0958-scan-20261017.md`, `public-scan-20261019.md`, `rank-value-selection-20261014.md`, `shakeup-risk-20261013.md`, `external-data-lane-20261015.md`, `provenance-refresh-20261016.md`; `docs/experiments/labels-v2-20261015.md`, `s6-labels-v2-main-lane-result-20261015.md`, `r6flip-preparation-20261017.md`, `na-full-export-acceptance-20261016.md`; `outputs/s6v2/STATUS.md`; handoff log entries Oct 17–19.
