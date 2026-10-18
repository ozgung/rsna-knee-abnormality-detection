# Public scan beyond Raptor + R7 0.5/0.5 — 2026-10-14 (data as of 2026-10-13 22:50 UTC)

Author: Claude, read-only scan. No compute, submission, commit, kernel push, terms acceptance, STATUS or handoff-log edit. Kernel sources and dataset manifests were pulled with the Kaggle CLI into ignored `outputs/analysis/public-scan-20261014/` (`lists/`, `kernels/`, `code/`, `datasets/`, `lb/`). No downloaded code was executed; JSON was parsed with `python3 -I` from a separate tools directory. Labels: [M] measured, [P] public claim, [E] my estimate.

Limits: the CLI does not show notebook scores or discussion threads, so scores come from the notebooks' own text [P]. Private notebooks of the top teams are invisible.

## 1. Leaderboard (snapshot 2026-10-13 22:49 UTC, 5,948 teams) [M]

| Threshold | Teams now | Teams at 14:47 UTC Oct 13 |
|---|---:|---:|
| ≥ 0.966 (rank 1) | 2 | – |
| ≥ 0.963 | 19 | 16 |
| ≥ 0.960 | 53 | 51 |
| ≥ 0.958 | 95 | 86 |
| ≥ 0.957 | 163 | 141 |
| ≥ 0.956 | 803 | 637 |

Rank 10 = 0.963, rank 50 = 0.960, rank 100 = 0.957 [M]. Our team still showed 0.954 at rank 1,181 (last scored submission Oct 12 21:47), so the 0.957 entry had not posted yet [M]. The top 30 teams have 18 to 345 submissions each, and none has a public RSNA notebook newer than Sept 23 [M]. The 0.958+ block grew by only 9 teams in 8 h, so nothing public is moving people above 0.957 [E].

## 2. Notebooks (≈530 unique kernels across dateRun, voteCount, dateCreated and hotness sorts, 3 pages each) [M]

Since Oct 12 there is no new heliosli, nartaa or dreaddevelopment RSNA notebook. heliosli's only newer kernel (Oct 13) is an ARC-AGI entry [M]. Every notebook claiming 0.957 is heliosli's "0.957 latest dataset" or a fork of it. A code diff against heliosli's source shows these deltas [M]:

| Notebook (date) | Change vs heliosli 0.5/0.5 | Claimed score | Runtime | Flags |
|---|---|---|---|---|
| jingshuifu/rsna-knee-flip-tta (Oct 13 16:34) | Sets `manifest['flip_tta']=True` in place, which turns on R7's built-in `--flip-tta`. Every fold scores mirrored windows; logits are averaged with a medial/lateral class permutation; MCL uses the plain pass only | none posted | R7 model passes double [M]; +1.5 to 3 h on our 4.73 h R7 slot [E] | none new |
| varshitha / karttik / yuvrajlamba / rainycloudss V21 / goodpjw v11 | Adds nartaa's `c08_r224` at 35% of the Raptor half (effective 0.325 / 0.175 / 0.5) | 0.956 (subm. 57141154) [P]; rank 233→156 inside the 0.956 block [P] | 26 s on 3 visible studies [P]; about +0.4 h [E, our WS2] | none new |
| sujanmajhisuzan V13 "anti-tie" | Adds a `linspace(±1e-6)` row-order jitter, then re-ranks | none posted | ~0 | Harmful in expectation: it breaks true ties by row order, which is noise [E] |
| medvax r7-source-trial | Wrapper with `RAPTOR_WEIGHT` and `DISABLE_CALIBRATIONS` (turns off R7's Effusion/Baker stackers); default = source | 0.956 (subm. 57136151) [P] | 0 | – |
| degnonguidi, evgendvorkin Versia-7 (385 votes), mayankraaz, tahmidtawsif v13, omn1v3r53 | Identical code (Versia = heliosli, per karttik's fingerprint) or mount tweaks; omn1v3r53 uses R6 instead of R7 | 0.956 to "0.957" [P] | same | – |
| goodpjw2008 v11 (Oct 13) | R224 at 35%, plus its 2.5D reader cut to 1/10 weight because "at full weight it cost about 0.002" on the 0.956 base | 0.954 title [P] | +reader | Negative evidence for weak third arms [P] |
| ayodeji m01–m05 (Oct 13) | Raptor r384 plus dreaddevelopment `raptor_ft_coatnet_v5_full_swa.pt` (maxspan) with per-finding weights (ACL 0.75, LM/PF 0.4, others 0.05, Contusion 0); no R7 | none posted | +1 CoAtNet pass | Old Raptor-family arm |
| hukumjailer consensus-raptor-training (Oct 13) | Training kernel: CoAtNet-384 MIL on 5-source Dawid-Skene soft labels (its own dataset, Oct 13); no weights published yet | none | training only | Internet on; label provenance unverified |
| dmitriykolmikov "is +0.001 real" (Oct 12) | Simulation: a single score is ±0.004 (1 SD) on public and ±0.0025 on private; SD of a difference is 0.0014 at error correlation 0.95, 0.0028 at 0.8 | n/a | n/a | – |

## 3. Datasets updated since Oct 11 [M]

- **heliosli:** R7 `rsna-knee-compact-r7cnxfsss` last updated Oct 11 19:59 (no new version); R6 Oct 11 13:18; S6 `compact-v1` Oct 10. Nothing newer. R7 `manifest.json` has `flip_tta: false` and `run_config.mirror: 0.5`. The model was therefore trained with the mirror the flip TTA inverts [M].
- **nartaa:** `publication-swa-weights-20261007` unchanged since Oct 7. goodpjw2008's RSNA reader is unchanged since Oct 5; its Oct 13 dataset is for another competition (S6E10) [M].
- **New third-party weights, none LB-claimed:** `jessky/rsna-knee-local-cv-cnx50` (Oct 12; 5-fold ConvNeXt-S6 d50 plus compact code; its README warns of second-order leakage through an external OOF teacher, Apache-2.0). `rauleg/…-dense-r288-attn-swa-oaij` (Oct 12; ResNet34 attention, 5 folds, holdout 0.867–0.891, trained with OAI 1,660 knees). `syouyatobita/rsna-knee-g231-nf1-weights` (Oct 11; one CoAtNet-320 fine-tuned from nartaa's weights). `genesis96839/optionb` (Oct 12; DINOv2 and MedicalNet backbones only). R6 mirrors from several users [M].
- **Labels:** `hukumjailer/rsna-knee-consensus-labels-clean` (Oct 13), `gperdrizet/…-pseudolabeled-training-data` (Oct 12), `shogomiyawaki/rsna-knee-llm-labels` (Oct 12) [M]. These are useful only for training we have no days left for [E].
- **External-data flag:** nartaa's Raptor checkpoint name contains `oai` (`…_blendjev_oai_d96_r384_swa.pt`), so the 0.957 recipe already carries OAI-trained weights. rauleg's arm adds more OAI. Both bear on the unresolved host ruling, so keep a no-OAI final [M/E].

## 4. Assessment

No public notebook or dataset published since Oct 12 claims a score above 0.957 [M/P]. Nothing public explains the 0.958–0.966 block. Those teams hold private models [E]. The public frontier is still Raptor-384 + R7 0.5/0.5. All levers left are worth at most a few thousandths, inside the ±0.004 single-score blur [P/E].

## 5. Ranked candidate levers and cheapest test

1. **R7 flip-TTA (`--flip-tta`)** [new]. It is the only untested inference lever with a training-consistent rationale (mirror 0.5 in training, a medial/lateral class permutation and MCL excluded). Expected effect +0.000 to +0.002 on R7 alone, and about half that blended [E]. heliosli shipped it off, either for runtime or because it did not help their CV; we cannot tell which [E]. Cheapest test: (a) time R7 on about 50 cached studies with and without the flag on T4 (Colab/Kaggle, <0.5 h) to get the real model-forward share; (b) if A + flip ≤ 8.3 h, make one LB probe of A with only the flag changed. If it is over budget, probe flip on 2 of the 4 folds.
2. **R224 at 35% of the Raptor slot** (already WS2). It is LB-tested only as a sub-display move inside the 0.956 block [P]. It is cheap (+0.4 h) [E]. Cheapest test: the planned WS2 submission. Make sure the R224 failure mask is exported separately (karttik flags that the public recipe reuses the R384 mask) [P].
3. **Per-finding Raptor/R7 weights.** Public ratios of 40/60 to 60/40 are flat at display resolution [P]. Without R7 OOF, any per-finding tuning would be fitted on LB noise [E]. Cheapest test: none worth a slot unless we have R7 OOF predictions on our folds.
4. **R7 stacker ablation** (medvax's `DISABLE_CALIBRATIONS` for Effusion/Baker). It is a pure post-process on existing outputs at zero runtime. The expected sign is unknown and probably slightly negative, since heliosli refitted the stackers [E]. Cheapest test: offline only, by diffing ranks; spend no LB slot.
5. **New third arms** (jessky cnx50, rauleg r288, syouya G233, goodpjw reader). goodpjw's report that its reader cost about 0.002 at full weight [P], the earlier 0.93-class-arm arithmetic [E], and the leakage and OAI flags make these low value. Cheapest test: skip; at most a 0.1-weight nudge if runtime is free.
6. **Anti-tie jitter, mount/fork variants:** no lever; skip.
