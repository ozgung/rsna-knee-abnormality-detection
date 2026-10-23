# Raptor64 fold-0 protocol: Dread v4 input contract, 2×2 init × head (WS-B)

Written 2026-10-13 (UTC 2026-10-12 about 22:45), before any training. Status: **cache building on Kaggle CPU; no Colab session or training started.** Infrastructure is tested on synthetic data and real pinned metadata only; nothing here is evaluated on real images yet.

## Revision 2026-10-13 07:45 UTC (before any result)

- **Arms run:** `in-orig`, `in-plane`, `dr-orig`. `dr-plane` is dropped. The `in-plane − in-orig` head test is now the most informative pair: B0 (Dread v4_full in the S6 blend) scored public 0.940 against 0.954 for the nartaa version.
- **Optional `na-orig`:** nartaa's OAI SWA Raptor backbone (`7e5315dad125…`, 292,840,787 bytes; OAI host ruling unresolved). Its pin and synthetic loader tests exist, but there is no local copy, so a real missing=0/unexpected=0 load is not yet verified. It is not launched without confirmation.
- **Full-data `dr-full` (prepared, not launched):** Dread v4 init with the original head, trained on all 4,345 eligible studies (`--folds 0`). With no held-out set, top-3 SWA uses three pre-registered epochs, applied through an epoch indicator so the historical top-3 and SWA code is unchanged. The proposal is to take `dr-orig`'s fold-0 selected epochs once known, or 9/10/11 otherwise. Build: `scripts/build_raptor64_training.py DEST --evidence-root … --arm dr-full --swa-epochs E1 E2 E3`. The export writes `run_fold0.json` and the SWA checkpoint, with no predictions. The "fold0" in its file names is a tag only. It needs a new bundle (v2) and an LB-probe entry rehearsal before any submission.
- **Gates:** the 0.92 sanity gate is withdrawn, and the evaluator reports raw K42 macros and paired deltas only. The `dr-orig` epoch-1 K16 < 0.85 stop rule stays, as a training health check.
- **Execution:** session `raptor64-fold0-r2-20261013` (A100 40 GB, Python 3.13.15, torch 2.11.0+cu130, timm 1.0.26).
  - Cache staging PASS, 4,345 studies; content `0bdc189c…`.
  - Probe: missing 0, unexpected 0, finite, 2.74 GB training peak per arm at batch 4 × K8.
  - `in-orig` (controller PID 5942) and `in-plane` (PID 6025) were launched at 07:36 UTC and are running.
  - The GPU was at 100% with two arms, so `dr-orig` (config `fd4fd060…`) waits for a free slot.

## Question and hypotheses

Does the public Dread "widedense v4" input contract — 64 slots (18/14/12/8/12), 6–94% slice span, 336-px storage resized to 384 — with the public-recipe CoAtNet (`train_knee.py` RaptorClassifier lineage) give a stronger own fold-0 model than own44 at K42?

- **H1 (primary, clean):** the ImageNet-initialised original-head arm `in-orig` beats own44 K42 (0.905877) on the fixed 870 at K42 by at least +0.005, with the paired CI above zero.
- **H2 (head, clean):** the plane-aware head (`in-plane`) beats the original head (`in-orig`) with the same ImageNet init.
- **Dread-init arms (`dr-orig`, `dr-plane`) are exposed.** Dread trained on all non-gold studies, including our fold 0, so their fold-0 numbers are a sanity gate only, never evidence of generalisation.

## Arms (everything else identical)

| Arm | Initialiser | Head | Role |
| --- | --- | --- | --- |
| `in-orig` | ImageNet `timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, revision `2c31d823…`, safetensors SHA `91570c7a…` | public RaptorClassifier | H1; clean comparator |
| `in-plane` | same | plane-aware | H2 against `in-orig` |
| `dr-orig` | Dread CC0 `raptor_ft_coatnet_v4_full.pt` SHA `89606f05…`, backbone tensors only | public RaptorClassifier | exposed sanity gate, LB-probe candidate |
| `dr-plane` | same | plane-aware | exposed; head effect under Dread init |

- **Dread init:** `train_knee.py --ckpt` semantics: the `backbone.` prefix is loaded with `strict=False` and the head is fresh. The loader also enforces missing = 0 and unexpected = 0, the exact eight head keys, finite tensors, arch/res 384, target order, and pinned bytes. On the local v4_full copy, timm 1.0.26 loads 478 backbone tensors with missing 0 and unexpected 0 (`tests/test_raptor64_training.py`).
- **Plane-aware head** (`knee/raptor64_training.py:plane_variant`):
  - per-window metadata is the centre slice's plane × actual `Fluid_Sensitive` (6 codes, `Embedding(6,16)`) and normalised depth within its series (`Linear(1,8)`);
  - this is concatenated to `LayerNorm(h)` before the per-finding attention logits `att` and before the classifier `clsW`;
  - it is a subclass, so the pinned public class bodies are unchanged and AST-checked;
  - the sampler's RNG consumption is unchanged, so both head variants see identical training windows for a seed.
- **Metadata source:** the per-slot metadata (plane, selected, actual FS, series frame count) is recorded by the cache builder from the public reader's own ordering calls (no extra I/O) and stored in the private index.
- **Inference parity:** inference uses `inference_window_metadata(slots, mask, k)`, and a test proves it equals the training dataset's eval-mode metadata.
- **Fixed for all arms:**
  - labels and split: `labels_v1_blend` (SHA `d2cdb8a3…`); fixed fold-0 split, 3,475 train and 870 development, with 58 gold and 4 quarantined excluded (source96 pins);
  - schedule: seed 42; trainer-default OneCycle (pct_start 0.15) and **12 epochs** (the `train_knee.py` default);
  - batch and windows: batch 4 × accumulation 2 (effective 8, as own224/source96); K8 training windows; 384 input; ImageNet normalisation;
  - optimisation: FP16, gradient checkpointing with window chunks of 8, bb_lr 3e-5, head_lr 1e-3, wd 0.02, workers 2, no preload;
  - selection and export: top-3 SWA selected per epoch by K16 report-masked macro on the 870; final export at **K42**.

**Selection-K decision:** per-epoch selection uses K16, exactly as own44/source96, and the endpoint uses K42, exactly as the own44 K42 comparator (K16-selected, K42-evaluated). This keeps the comparator protocol matched and costs about 25% less per epoch than K42 selection. It is one constant (`FROZEN['k_eval']`) if the parent prefers K42 selection.

## Endpoint and analysis (pre-registered)

- **Primary endpoint:** masked report macro AUC on the fixed 870 at K42. The unweighted mean is taken over targets with both observed classes.
  - Truth, masks and 843 report-component groups come from the pinned own224 endpoint `outputs/kaggle/own224-window-samerun-20261012-result-v1/own224_window_diagnostic/endpoint-private.npz` (SHA `1c7a0d09…`).
  - The control is own44 K42 `k42-private.npz` (SHA `2e5d807b…`, macro 0.9058770197).
- **Method:** `scripts/evaluate_raptor64_endpoint.py` runs the qualified paired whole-component bootstrap (2,000 draws, seed 42). It reports per-target AUC with support, undefined targets, the prevalence baseline (0.5), deltas against own44 K42, and the contrasts `in-plane − in-orig`, `dr-plane − dr-orig` and `dr-orig − in-orig` (the last is exposed).
- **Decisions:**

| Arm or contrast | Rule | Outcome |
| --- | --- | --- |
| H1 (`in-orig`) | delta ≥ +0.005 and CI lower > 0 | promote to fold-1 replication |
| H1 (`in-orig`) | otherwise | close this contract for own-base use |
| H2 (`in-plane − in-orig`) | delta ≥ +0.003 and CI lower > 0, with no supported target worse by more than 0.02 | adopt the plane-aware head |
| H2 | otherwise | keep the original head |
| `dr-orig` | report the raw exposed K42 macro and paired deltas only (the earlier 0.92 gate was withdrawn as arbitrary, 2026-10-13) | a full-data fit and LB probe is a separate decision |

Each arm is also reported against own44 K42 as context. The comparison changes resolution, slices, span and epochs at once, so it is a recipe-level comparison, not a single-factor one.

- **Limits:** the development set is exposed and report-selected; patient identity is unavailable (study-level folds); persistent-worker RNG is not restored on resume; the historical own44 initialiser bytes are unresolved. None of this is a public-score forecast.

## Pipeline and status

1. **Cache (running).** Four private, offline, CPU-only Kaggle kernels at version 1, pushed 22:26 UTC:

| Kernel | ID | Studies |
| --- | --- | --- |
| `ozgung/knee-raptor64-cache-shard0-20261013` | 138397942 | 1,087 |
| `ozgung/knee-raptor64-cache-shard1-20261013` | 138397945 | 1,086 |
| `ozgung/knee-raptor64-cache-shard2-20261013` | 138397951 | 1,086 |
| `ozgung/knee-raptor64-cache-shard3-20261013` | 138397954 | 1,086 |

   - Builds are in `outputs/kaggle/raptor64-cache-shardN-20261013/`; the receipt is `outputs/analysis/raptor64-20261013/cache-push-receipt.json`.
   - The reader is the audited `knee.reference` raptor profile of the pinned Dread inference source (SHA `859aae12…`), called through the unchanged `knee.raptor_input.prepare_study`.
   - Exact parity with direct `build_study` and with the strict reference reader is tested on synthetic DICOM, together with the full production stress-case set.
   - Expected time is about 50–60 min per shard (the 96-slot shards took about 77 min).
2. **Qualify metadata locally** (no images): `scripts/collect_raptor64_cache_metadata.py`.
3. **Colab:** bundle, stage, initialisers, backups, configs, launch (commands below).
4. **Export** each completed arm privately: `scripts/colab_raptor64_export_terminal.py`.
5. **Evaluate locally** with `scripts/evaluate_raptor64_endpoint.py`.

## Resource estimate (unverified; record actuals)

- **Per epoch:** the 224 source96 epoch took 866 s solo on an A100 40 GB (27.8k training windows plus 13.9k K16 windows). Scaling by pixels (×2.94), a fully GPU-bound 384 epoch is about 2,550 s. That gives about 8.5 h per arm for 12 epochs, plus about 15 min for the K42 export of 36.5k windows.
- **Four arms:** about 35 GPU-hours if run sequentially. With two processes per GPU at near-solo speed (as measured at 224: 869/903 s overlapped against 866 s solo, 2.6 GB each), wall time is about 18 h on one A100. At 384 the GPU is more likely saturated, so measure epoch 0 before assuming overlap is free. An H100 (`colab new --gpu H100`) should be faster; check unit cost.
- **Memory:** each controller's runtime probe records `train_peak_cuda_bytes_bs4_k8` on real rows before launch. Run two arms together only if probe peak × 2 is below 70% of VRAM.
- **Disk:**
  - cache: about 4,345 × 5–6 MB ≈ 22–26 GB; the worst-case bound is 7.3 MB per study;
  - per arm: resume about 2 GB, persister staging about 2 GB and SWA about 0.3 GB.
- **Compute units:** about 12 units/h on A100 (unverified) × 35 h ≈ 420 of the 850 available; the background limit is 24 h per session. Each controller checkpoints every epoch and backs up privately (persister SHA `8923bd87…`), so a platform stop loses at most one epoch.

## Stop rules

- **Pre-launch:** stop if the runtime probe fails, which covers CUDA, staged 4,345 rows, timm 1.0.26, initialiser missing or unexpected ≠ 0, a non-finite forward or backward pass, and a K42 eval window count ≠ 42. Never launch an arm whose config fails `colab_raptor64_run.py validate`.
- **During training:** a trainer exception or non-finite loss makes the controller fail and preserves the resume and backup; debug, and do not relaunch blindly. A dead persister triggers a bounded stop.
- **Dread-init sanity:** if `dr-orig` epoch-1 K16 selection AUC is below 0.85, pause both `dr-*` arms and investigate. A fine-tune from a 0.92-class exposed model should not collapse; a collapse suggests a contract or init defect.
- **No assistant time caps.** Resume only through an explicit config with `resume: true` and a pinned `resume_sha256`.

## Exact parent launch commands (after all four caches are COMPLETE)

All paths are relative to the repository root after the reviewed commit; `S=raptor64-fold0-20261013`.

```bash
# 0. Local: qualify cache metadata (no images) and write the private stage config.
~/miniconda3/bin/python scripts/collect_raptor64_cache_metadata.py \
  --evidence-root ~/Dev/rsna-knee-abnormality-detection \
  --receipt outputs/analysis/raptor64-20261013/qualified-metadata-v1.json \
  --stage-config outputs/analysis/raptor64-20261013/raptor64-stage-config-private.json
# 1. Local: build the reviewed bundle (four packages, tools, accepted notebooks, pinned persister).
.venv/bin/python scripts/build_raptor64_colab_bundle.py outputs/analysis/raptor64-20261013/colab-bundle-v1 \
  --evidence-root ~/Dev/rsna-knee-abnormality-detection
#    Put bundle_sha256 into outputs/analysis/raptor64-20261013/vm/vm-extract.py and the
#    stage-config SHA into vm/vm-stage.py (REPLACE_WITH_... placeholders).
# 2. Allocate one GPU session (A100, or H100 if preferred and available).
colab new -s raptor64-fold0-20261013 --gpu A100
colab upload -s raptor64-fold0-20261013 outputs/analysis/raptor64-20261013/colab-bundle-v1/raptor64-colab-bundle.tar.gz /content/raptor64-colab-bundle.tar.gz
colab upload -s raptor64-fold0-20261013 outputs/analysis/raptor64-20261013/colab-bundle-v1/colab_tracka_persist.py /content/colab_tracka_persist.py
colab upload -s raptor64-fold0-20261013 outputs/analysis/raptor64-20261013/raptor64-stage-config-private.json /content/raptor64-stage-config-private.json
colab upload -s raptor64-fold0-20261013 ~/.kaggle/kaggle.json /content/kaggle.json
V=outputs/analysis/raptor64-20261013/vm
colab exec -s raptor64-fold0-20261013 -f $V/vm-extract.py --timeout 300
colab exec -s raptor64-fold0-20261013 -f $V/vm-setup.py --timeout 2400      # pip pins; ImageNet-384 + Dread v4 initialisers
colab exec -s raptor64-fold0-20261013 -f $V/vm-stage.py --timeout 7200      # about 23 GB private cache, 8 download workers
for A in in-orig in-plane dr-orig dr-plane; do
  colab exec -s raptor64-fold0-20261013 -f $V/vm-inputs-$A.py --timeout 600
  colab exec -s raptor64-fold0-20261013 -f $V/vm-bootstrap-$A.py --timeout 600   # creates/verifies PRIVATE ozgung/knee-raptor64-$A-fold0-20261013
  colab exec -s raptor64-fold0-20261013 -f $V/vm-config-$A.py --timeout 600      # prints config_sha256; review, then paste into vm-launch-$A.py
done
# 3. Launch in order; two concurrent if the probe memory allows, otherwise one at a time.
colab exec -s raptor64-fold0-20261013 -f $V/vm-launch-in-orig.py --timeout 900
colab exec -s raptor64-fold0-20261013 -f $V/vm-launch-in-plane.py --timeout 900
#    ...then dr-orig and dr-plane as slots free up.
colab exec -s raptor64-fold0-20261013 -f $V/vm-health.py --timeout 120
colab download -s raptor64-fold0-20261013 /content/raptor64-health.json outputs/analysis/raptor64-20261013/health-latest.json
# 4. After each arm's controller ends (receipt PASS), export privately on the VM:
#    python /content/raptor64-tools/scripts/colab_raptor64_export_terminal.py --arm ARM --workdir /content/raptor64-ARM-fold0 \
#      --config /content/raptor64-ARM-run-config.json --config-sha256 SHA --package-manifest-sha256 SHA --cache-content-sha256 SHA \
#      --runner /content/raptor64-tools/scripts/colab_raptor64_run.py --runner-sha256 SHA \
#      --dataset ozgung/knee-raptor64-ARM-fold0-final-20261013 --staging /content/raptor64-ARM-export-staging \
#      --verify-root /content/raptor64-ARM-export-verify --receipt /content/raptor64-ARM-export-receipt.json --source-revision <commit>
# 5. Local evaluation (after downloading each oof_fold0.csv from the verified private export):
.venv/bin/python scripts/evaluate_raptor64_endpoint.py \
  --endpoint outputs/kaggle/own224-window-samerun-20261012-result-v1/own224_window_diagnostic/endpoint-private.npz \
  --control outputs/kaggle/own224-window-samerun-20261012-result-v1/own224_window_diagnostic/k42-private.npz \
  --arm in-orig=PATH --arm in-plane=PATH --arm dr-orig=PATH --arm dr-plane=PATH \
  --out outputs/analysis/raptor64-20261013/endpoint-result-v1.json
```

The launch order follows the concurrency priority: `in-orig` and `in-plane` first, as the clean H1/H2 pair, then `dr-orig` and `dr-plane`. Monitor only through `colab download` of the health file while kernels are busy. Each arm has its own controller lock, workdir, inputs, persister staging and status file, and private backup dataset; stopping one arm means sending SIGTERM to that arm's controller PID from `/content/raptor64-ARM-launch.json`.

## Evidence and tests

The files are new and uncommitted in the worktree at the time of writing:
- **Code:**
  - `knee/raptor64_{cache,loader,training}.py`;
  - `scripts/{build_raptor64_cache,build_raptor64_training,build_raptor64_colab_bundle,prepare_raptor64_initializer,colab_raptor64_prepare,colab_raptor64_run,colab_raptor64_vm,colab_raptor64_export_terminal,collect_raptor64_cache_metadata,evaluate_raptor64_endpoint}.py`.
- **Tests:** `tests/test_raptor64_*.py`, `tests/test_colab_raptor64_*.py`, `tests/test_evaluate_raptor64_endpoint.py`. Among them:
  - real-reader parity on synthetic DICOM;
  - the real v4_full missing=0/unexpected=0 load;
  - a CPU end-to-end run of the generated trainer and K42 export for a plane-head and an original-head arm (tiny stand-in backbone, synthetic cache);
  - stager, runner and exporter fail-closed behaviour;
  - the endpoint evaluator against the pinned own44 K42 control.
