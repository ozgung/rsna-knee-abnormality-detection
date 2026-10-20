# Lane C — S6 seed ensemble, solo probe, P7′ (Oct 19–22)

Owner: Claude (lane C agent). Plan: [final-push plan §2 Lane C](../research/final-push-plan-20261019.md). Times UTC.

## Pre-registration (written 2026-10-19 18:35 UTC, before any lane C fit, gold read-out or Lane B result)

- **Hypothesis.** Averaging all-data S6 fits over seeds (and, if Lane B delivers, a stronger label table) improves the
  compact slot over the single V2-all model (seed 42, solo public 0.950). Not a score bet for final 1: P7′ is the
  no-heliosli-bundle provenance hedge and the ρ ≈ 0.93 final-2 option.
- **Recipe.** V2-all exactly (`scripts/s6v2/vm_train.sh`, trainer 12e430a3…, 16 epochs, all 4,345 pool studies, no holdout),
  only `--seed-base` changed; two fits concurrently on one A100 (5 loader workers each instead of 8; augmentation RNG is
  per item, so worker count does not change batches).
- **Sequence.** Seeds 43 and 44 on labels v2 (targets_V2_all e2283884…) now. Next pair: if Lane B's
  `labels_v3/READY.json` exists by Oct 20 06:00, two seeds on labels v3 (seeds 42 and 43, so the v3 pair mirrors the v2
  seed numbering); otherwise seed 45 on labels v2.
- **Ensemble set (fixed now; no gold-58 input).** v2 and v3 seeds are never mixed. If both v3 fits complete and pass the
  epoch-0 kill rule, the P7′ compact = mean rank of the two v3 seeds (the plan's primary path is labels v3); otherwise
  the v2 set {42, 43, 44, 45 if run}. Mean rank = per seed percentile rank over the scored population, averaged over seeds.
- **Kill rules (plan).** Epoch-0 loss > 0.55 (V2-all 0.5106) stops a fit; chosen-set gold-58 macro < 0.925 kills P7′;
  P7′ rehearsal point > 27,000 s kills P7′; solo probe < 0.949 displayed leaves P7′ as a hedge only.
- **Solo probe C1.** The new seed with the highest gold-58 macro within the chosen label family (gold-58 is used here only to
  pick which probe to spend; it is selection-exposed and recorded as such). Not a final candidate.
- **Gold-58.** Read once per artifact through the pinned DICOM → compact_infer path (stride 1, bf16 on A100), logged as exposed.

## Runs

| Run | Labels | Seed | VM / session | Start | Epoch-0 loss | Final loss | Wall | Gold-58 | Export |
|---|---|---|---|---|---|---|---|---|---|
| v2-s43 (lost) | v2 | 43 | lanec1 | 2026-10-19 19:21 | 0.5083 (1,563 s) | — (VM lost after epoch 1) | | | |
| v2-s44 (lost) | v2 | 44 | lanec1 | 2026-10-19 19:22 | 0.5105 (1,661 s) | — (VM lost after epoch 1) | | | |
| v2-s43 | v2 | 43 | lanec2 | 2026-10-19 23:05 | 0.5083 (1,561 s) | stopped at epoch 3 (0.4248); not needed after amendment 1 | | | |
| v2-s44 | v2 | 44 | lanec2 | 2026-10-19 23:07 | 0.5112 (1,659 s) | stopped at epoch 3 (0.4279); not needed after amendment 1 | | | |
| v3-s42 | v3 | 42 | lanec2 | 2026-10-19 23:57 | 0.5185 (1,560 s) | 0.3713 (epoch 15, 25,761 s) | 7.2 h | 0.9283 | Model `knee-s6-lanec-seeds-20261019/pyTorch/v3-s42/1`, 03da2cac… |
| v3-s43 | v3 | 43 | lanec2 | 2026-10-19 23:59 | 0.5157 (1,662 s) | 0.3722 (epoch 15, 25,762 s) | 7.2 h | 0.9380 | Model `knee-s6-lanec-seeds-20261019/pyTorch/v3-s43/1`, a99ad897… |

## Recovery

- Colab config `~/.config/colab-cli/sessions-lanec-20261019.json`, session `lanec2`, endpoint
  gpu-a100-s-kkb-ass1c2-326w2ki99c9c0 (lanec1 = gpu-a100-s-kkb-ass1c0-1hv5ftoekaj8u, lost 20:38). Local tools and heartbeat log: `outputs/lanec-20261019/vm/`
  (`refresher.sh` every 30 min → `tools/reconnect.py`; `c.sh` = colab with the lane config).
- VM: boot `/content/vm_seed_boot.sh` (log `/content/boot.log`); fits `/content/vm_seed_fit.sh <tag> <seed> <targets>`
  (logs `/content/fit_<tag>_s<seed>.log`, `/content/runs/<tag>_s<seed>.log`); per-epoch backups to the private Kaggle Model
  `ozgung/knee-s6-lanec-ckpt-20261019`, variation `<tag>-s<seed>` (`/content/bk_<tag>_s<seed>.log`); exports to
  `ozgung/knee-s6-lanec-seeds-20261019`, variation `<tag>-s<seed>` (`scripts/s6v2/vm_kmodel.sh`, confirmed by remote listing).
- Resume after a VM loss: new session, rerun the boot (restore + preflight), put the backed-up checkpoint at
  `/content/runs/<tag>_s<seed>/ckpt/E_prior_s<seed>_f0.pt`, rerun the fit script (the trainer resumes).
- Stop: `outputs/lanec-20261019/vm/c.sh stop -s lanec1`, then kill the local refresher.

## Log

- 2026-10-19 19:20: cache restored on lanec1 (17 shards, every sha256 matches the original preflight; 52 min in
  asia-southeast1 vs 393 s on Oct 16), preflight on targets_V2_all passed, fits started. Epoch time ≈ 1,600 s per fit
  with two fits sharing the A100 (V2-all alone: 778 s) → ETA ≈ 02:45 Oct 20.
- Reference read-out, V2-all seed 42 (already exposed): gold-58 macro **0.9341** (compact_infer stride 1, bf16, 58/58,
  0 fallback; the scorer reproduces nartaa S6fall's 0.9314). Probe-36 A100 fp16 vs the s6solo-v2 T4 rehearsal: max
  |Δp| 0.0047 → the P7′ rehearsal reference gate uses 0.01.
- **Kaggle private-dataset quota is exhausted** (0 B left; 133 GB cache backup plus earlier lanes). Epoch-0 checkpoints
  of both fits were backed up; from epoch 1 every versioned backup fails. A VM loss now means restarting from epoch 1.
  Exports at fit end need space: plan = delete only this lane's own checkpoint datasets once the final models exist,
  then create the export datasets. No other dataset is touched.
- P7′ smoke rehearsal (not a candidate): `r6flip-p7smoke-rehearsal-20261019`, seeds = nartaa S6fall control (as seed
  41) + our V2-all s42, to exercise the s6c worker in the A-prime image before the lane C seeds exist.
- Smoke v1 (21:07) failed before any stage: `S6C_MODEL_SCHEMA` (the bootstrap adds each model's mounted path; the schema
  check now allows it, commit "s6c: accept the bootstrap-added model path"). **Smoke v2 PASS** (21:10–21:52, 2×T4,
  A-prime image torch 2.10.0+cu128): 39/39 genuine, 0 fallback, every R384/R224 gate and every s6c gate (sentinel
  repeats 5 per sentinel across fresh processes and boards, max |Δp| 0.0; state unchanged; source/anatomy pins), CSV
  recomputed; elapsed 2,169 s; projection point 15,125 s, conservative 19,294 s (two seeds; gate 29,880 s). Phase walls:
  old 946 s, crop224 498 s, s6c sentinels 287 + 125 s, s6c production 256 s; s6c peak CUDA 7.3 GB/board. The s42 per-seed
  probabilities on the 36 probes match the A100 fp16 reference within 0.0047 (the same as nartaa's image), so the
  exact S6 code tree runs unchanged in the A-prime image.
- **lanec1 lost at 20:38 UTC** (keep-alive 404 from 20:38:59; token refresh at 20:25 had succeeded; RAM/disk healthy at
  20:20; cause unknown, like s6v2d on Oct 16). Both fits had finished epoch 1. The "epoch 0 backup ok" lines were false:
  the kaggle CLI exits 0 on a quota error and no checkpoint dataset exists, so nothing could be resumed.
- Mitigation: private **Kaggle Models** still accept uploads (dataset quota does not apply). New publisher
  `scripts/s6v2/vm_kmodel.sh` confirms each upload by listing the remote version's files (tested: create + version).
  A CPU probe kernel showed the mount layout `/kaggle/input/models/<owner>/<model>/pytorch/<variation>/<version>/`;
  P7′ and C1 packets now take Model sources. lanec2 created 22:07; cache restore + both fits restarted from scratch
  (same seeds, same scripts apart from the backup target).
- lanec2: restore OK 22:56; epoch 0 s43 0.5083 (1,561 s), s44 0.5112 (1,659 s). First Model backups failed (the CLI's
  version listing returns every variation of the model; fixed by filtering on the variation), then epoch-0 checkpoints
  confirmed remotely (`ozgung/knee-s6-lanec-ckpt-20261019/pyTorch/v2-s4{3,4}/1`, 600,696,617 B each). ETA ≈ 06:15.
  Resume after a loss: new session, boot (restore + preflight), `kaggle models instances versions download
  ozgung/knee-s6-lanec-ckpt-20261019/pyTorch/v2-s<seed>/<latest> --untar` and place `ckpt__E_prior_s<seed>_f0.pt` as
  `/content/runs/v2_s<seed>/ckpt/E_prior_s<seed>_f0.pt`, rerun `vm_seed_fit.sh v2 <seed> <targets>`.
- Queued on lanec2 (`/content/after_pair.log`): when both exports exist, start seed 45 on v2 at once (GPU otherwise idle;
  if labels v3 is READY by 06:00 the v3 pair runs next and the pre-registered set is v3), then gold-58 read-out of
  s43+s44 (bf16, one decode) and the probe-36 fp16 reference of s42+s43+s44.
- **Labels v3 READY** (Lane B marker created 22:09 Oct 19, seen 23:47): table 3ceafe5a…, targets_V3_all 8d2ef527…
  (rebuilt here byte for byte with `build_targets_table.py`; `check_targets.py` OK: pool 4,345, gold blank, 4 quarantined).
  Lane B's exposed gold-58 of the label table: v3 0.9254 vs v2 0.9202.
- **Order changed (reversible):** v3 is READY 6 h before the 06:00 cut-off, so the pre-registered P7′ set is the v3 pair
  (seeds 42, 43). The v2 pair had only restarted at 23:06 after the VM loss; it is paused after its epoch-1 checkpoints are
  confirmed in the Model backups and resumes after the v3 pair as the fallback set. Running v3 first moves the primary path
  (C1 and P7′) about six hours earlier at no loss of v2 work. Seed 45 (v2) is dropped: the rule runs it only when v3 is
  not READY.

## Amendment 1 — P7′ ensemble set (2026-10-20 08:40 UTC, parent decision)

- **New set:** P7′ compact = equal-weight mean rank of {v3-s42, v3-s43, Lane D fastMRI student}. The student (S6 recipe on
  labels v3 + 745 fastMRI exams with P2-teacher soft targets at effective weight 0.5, seed 42, final loss 0.3694; private
  dataset `ozgung/knee-laned-fmall-s42-20261020` v1, checkpoint 6a13f52e…; gold-58 0.9350 exposed; solo probe 57426580
  public 0.954) counts as a v3-family member (same labels v3, architecture and geometry). If a v3 seed fails its kill rule,
  the remaining v3 seed(s) plus the student, equal weight; if both fail, the student alone. No v2 seed is mixed in.
- **Reason:** a new model arrived after the pre-registration; the parent decided before any lane C v3 read-out reached it.
- Every other rule unchanged: chosen-set gold-58 < 0.925 kills P7′; rehearsal point ≤ 27,000 s; 8.3 h guard; salvage;
  visible PASS; then submit.

## Read-outs and packets (Oct 20)

- Gold-58, read once per artifact, **exposed** (compact_infer stride 1, bf16 A100, 58/58, 0 fallback; one decode for both v3
  seeds; scorer reproduces nartaa S6fall 0.9314): v3-s42 0.9283, v3-s43 **0.9380**, v3 pair mean logit 0.9365; reference
  V2-all s42 0.9341; Lane D student 0.9350 (its own read-out). **P7′ set (amendment 1) mean rank 0.9357 ≥ 0.925 → no kill.**
- Local acceptance of both v3 exports (`scripts/build_s6_solo_seed.py accept`): SHA256SUMS, run_config (seed, 16 epochs,
  pool 4,345, targets_V3_all), membership (train dbd83070… = V2-all's pool, gold 58, holdout 0, overlap 0); strict CPU
  state/constructor/repeat check PASS (362 tensors, repeat and masked-perturbation Δ 0). State fingerprints:
  v3-s42 3048aae6…, v3-s43 aa2decf3…. v3-s43 was fetched from the VM because the Models download API returned 404 for it;
  a CPU mount probe kernel confirmed all three P7′ members mount in Kaggle with the expected sha256s.
- C1 = v3-s43 (highest gold-58 in the chosen family, exposed choice): rehearsal `s6solo-v3s43-qual-20261020` (payload
  ee6addb1…) pushed 08:39.
- P7′ rehearsal `r6flip-p7-rehearsal-20261018` (payload 296f10d7…; members aliased E_prior_s42_f0 = v3-s42,
  E_prior_s42_f1 = student, E_prior_s43_f0 = v3-s43; A100 fp16 probe reference d41d7952…) pushed after C1.
- lanec2 stopped 08:42 (`colab sessions` no longer lists it). A100 time: lanec1 2.2 h, lanec2 10.6 h.
- C1 rehearsal v1 (Model mount) failed at fixture build: the solo fixture links `input/<mount>` one level deep and cannot
  link the nested Models path. Fix without touching the shared solo sources: a private dataset copy of the v3-s43 export
  (`ozgung/knee-s6-v3all-s43-20261020` v1, same five files, quota freed by Lane D). **C1 rehearsal v2 PASS** (09:09–09:36,
  every check true incl. qualified state/reload, cross-board sentinels, all 39 genuine; elapsed 585 s; conservative
  projection 10,531 s). Entry `s6solo-v3s43-entry-20261020` built from that verified result; visible run → submit chain started.
- **P7′ rehearsal PASS** (08:42–09:39, 2×T4): 39/39 genuine, 0 fallback, every R384/R224 gate, every s6c gate
  (3 members; sentinel repeats max |Δp| 0.0; A100 reference 108 member-study checks, max |Δp| 0.0048 ≤ 0.01), CSV
  recomputed; elapsed 1,680 s; **projection point 16,369 s** (kill > 27,000 s: pass), conservative 21,256 s (gate 29,880 s).
  Phase walls: old 717 s, crop224 264 s, s6c sentinels 208 + 152 s, s6c production 309 s. Entry `r6flip-p7-entry-20261018`
  (payload aad4124d…) built from the verified result (no waiver); visible run → submit chain started.
- **C1 submitted: 57432734** (10:00 UTC) after visible run v1 COMPLETE (09:38–09:59): entry receipt PASS, 0 A fallbacks,
  0 B failed rows, elapsed 245 s, submission 3×12 finite in [0, 1], pulled code cells identical, private/offline.
- **P7′ submitted: 57432784** (10:01 UTC) after visible run v1 COMPLETE (09:47–09:59): receipt PASS, `fallback_studies` 0,
  every arm merged strictly, elapsed 458 s, pulled code cells identical. Kernel without `--timeout`; 8.3 h guard in force.

## Outcome (2026-10-20 14:31 UTC)

| Entry | Kernel | Submission | Public | Hidden wall |
|---|---|---|---|---|
| C1 solo v3-s43 (s6solo packet) | `s6solo-v3s43-entry-20261020` v1 | **57432734** | **0.955** | ≤ 4.5 h (submitted 10:00, scored by 14:31) |
| P7′ (Raptor half + v3-s42 / v3-s43 / fastMRI student mean rank, no flip, no heliosli bundle) | `r6flip-p7-entry-20261018` v1 | **57432784** | **0.955** | ≤ 4.5 h |

References on the board: V2-all and S6fall solos 0.950, Lane D student solo 0.954, P2 0.958, P6sp 0.957.

- Kill rules: none fired (epoch-0 losses 0.5185 / 0.5157 ≤ 0.55; chosen-set gold-58 0.9357 ≥ 0.925; rehearsal point
  16,369 s ≤ 27,000 s; solo 0.955 ≥ 0.949).
- Selection rule (§4): P7′ displayed 0.955 < 0.957, so it does not become final 2 under rule 2; it stays the provenance hedge
  (rule 3: if the Oct 20 recheck is adverse for the heliosli compact bundles, final 2 = P7′ at ≥ 0.955 — met).
- C1 at 0.955 is the best own compact solo so far (+0.005 displayed over V2-all's 0.950, one to two public SDs); a solo is
  never a final. Gold-58 remains selection-exposed; no private claim.
- Exposure ledger (each read once): v3-s42 0.9283, v3-s43 0.9380 (one decode), P7′ set mean rank 0.9357, s42-v2 reference
  0.9341. The probe-36 references (fp16) are not gold.
- Compute: Colab A100 lanec1 2.2 h (lost), lanec2 10.6 h; Kaggle GPU ≈ 1.1 h T4-pairs for rehearsals/visibles plus the
  two hidden runs. The v2 pair (s43/s44) stopped at epoch 3 and is not exported; their epoch-1 checkpoints sit in
  `ozgung/knee-s6-lanec-ckpt-20261019` (v2-s43/2, v2-s44/2) if ever needed.
- Branch `claude/lanec-20261019` (pushed): `knee/public_s6c.py`, `knee/public_s6c_worker.py`, R6F variant `p7`,
  `scripts/build_s6_solo_seed.py`, `scripts/s6v2/{vm_seed_*,vm_bk_seed_ckpt,vm_kmodel,build_targets_table,vm_seed_readout,score_gold_seeds}`,
  `tests/test_public_s6c.py` (18 tests) — 115 R6F/s6c/solo tests PASS, `python3.12 -m knee.smoke` passed.
