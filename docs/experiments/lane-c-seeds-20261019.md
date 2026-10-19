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
| v2-s43 | v2 | 43 | lanec2 | ≈ 2026-10-19 23:05 | 0.5083 (1,561 s) | paused after epoch 1 (resumable) | | | |
| v2-s44 | v2 | 44 | lanec2 | ≈ 2026-10-19 23:07 | 0.5112 (1,659 s) | paused after epoch 1 (resumable) | | | |
| v3-s42 | v3 | 42 | lanec2 | ≈ 2026-10-20 00:10 | | | | | |
| v3-s43 | v3 | 43 | lanec2 | ≈ 2026-10-20 00:12 | | | | | |

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
