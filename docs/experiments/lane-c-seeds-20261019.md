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
| v2-s43 | v2 | 43 | lanec1 | 2026-10-19 19:21 | 0.5083 (1,563 s) | | | | |
| v2-s44 | v2 | 44 | lanec1 | 2026-10-19 19:22 | 0.5105 (1,661 s) | | | | |

## Recovery

- Colab config `~/.config/colab-cli/sessions-lanec-20261019.json`, session `lanec1`, endpoint
  gpu-a100-s-kkb-ass1c0-1hv5ftoekaj8u. Local tools and heartbeat log: `outputs/lanec-20261019/vm/`
  (`refresher.sh` every 30 min → `tools/reconnect.py`; `c.sh` = colab with the lane config).
- VM: boot `/content/vm_seed_boot.sh` (log `/content/boot.log`); fits `/content/vm_seed_fit.sh <tag> <seed> <targets>`
  (logs `/content/fit_<tag>_s<seed>.log`, `/content/runs/<tag>_s<seed>.log`); per-epoch backups to private
  `ozgung/knee-s6-<tag>all-s<seed>-ckpt-20261019` (`/content/bk_<tag>_s<seed>.log`).
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
