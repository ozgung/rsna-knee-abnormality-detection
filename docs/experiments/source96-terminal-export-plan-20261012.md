# Source96 terminal export plan (prepared 2026-10-12, not run)

Status: implemented and tested on synthetic data only (56 unit tests, 24 mutation checks; revised after Codex review: 403 create-only path, `--keep-tabular`, `KAGGLE_CONFIG_DIR=/content`, reconciliation restricted to fold 0 with original status FAIL). Not launched, not uploaded, no VM or Kaggle contact by this tool. The live fold 0 and fold 1 trainers were not touched. Parent decides when and whether to run it.

## Problem

Each fit ends with exports that exist only on the Colab VM: final top-3 SWA checkpoint, OOF predictions, run/meta JSON and controller receipts. The persister backs up only `raptor_resume_fold{k}.pt` to `ozgung/knee-source96-fold0-r2-20261012` and `ozgung/knee-source96-fold1-20261012`. Those datasets and their versions must stay untouched, so the exporter publishes to a new private dataset per fold.

## Tool

`scripts/colab_source96_export_terminal.py` (tests: `tests/test_colab_source96_export_terminal.py`). It runs on the VM after a fit has ended and never writes to the workdir.

Exact inputs (all required, no defaults):

| Input | Purpose |
|---|---|
| `--fold`, `--workdir` | `/content/trackA96-fold0` or `/content/trackA96-fold1` |
| `--config` + `--config-sha256` | the exact run config the controller was launched with (fold 0 `2590477c...`, fold 1 `49785f1b...`) |
| `--package-manifest-sha256` | fold 0 `c7dffd73...`, fold 1 `83d83b67...` |
| `--cache-content-sha256` | `c113958a...` (both folds) |
| `--runner` + `--runner-sha256` | `/content/colab_source96_run_v2.py`, `ccddb970...`; imported, so its own `primary_complete()` defines completeness |
| `--controller-sha256` (optional pin) | fold 0 `e8494e9d...`, fold 1 `ccddb970...` |
| `--dataset`, `--staging`, `--verify-root`, `--receipt` | destination and fresh paths; none may exist yet or sit inside the workdir |
| `--source-revision` | the parent's commit SHA containing the exporter (`unversioned` is accepted only as a fallback) |

Flags: `--new-version` (only for a dataset that already exists), `--create-fresh SLUG` (create-only for a fresh exact slug whose metadata is inaccessible, see below; must equal `--dataset`, cannot be combined with `--new-version`), `--accept-reconciled-receipt` (fold 0 only, see below), `--failed-receipts-only`, `--include-logs` (private logs, off by default), `--controller-log PATH`.

## Preflight (any failure refuses; nothing is uploaded)

1. Config file sha256 equals its pin; config fold, workdir, package-manifest hash and cache-content hash equal the request. Cache stage receipt and initializer descriptor re-checked against the config pins.
2. `source96-run-fingerprint.json`: fold, version, package manifest, cache content, and `config_sha256_canonical` equal the pinned config (canonical hash confirmed locally against the real fold 0 fingerprint). Fold 1 also requires the experiment split hash and the 4345/3479/866 counts.
3. `primary_complete(workdir, fold, 870|866, experiment_split, cache_content)` from the pinned runner. This is the runner's own function (10 epochs, `swa_topk`, `matched K16 primary`, `k_eval` 16, `n_val`, `n_gold` 0, three `selected_epochs`, history equals meta, `checkpoint_sha256` equals the file, no symlinks, OOF present). Plus: `selected_epochs` must be three distinct epochs in 0..9.
4. Controller receipt `source96-run-receipt.json` must match the fingerprint and be PASS with `checkpoint_sha256` equal to the run record and `backup.epoch` 9. Fold 0 only: the original controller runs the pre-fix checker and is expected to leave a FAIL receipt (`FileNotFoundError`, training and persister return codes 0, no signal, resume preserved). With `--accept-reconciled-receipt` that is accepted only for fold 0 and only when the original receipt status is exactly `FAIL` (checked first; fold 1, or a status such as `UNVERIFIED`, is refused even with a matching reconciliation hash), and then only if `source96-parent-completion-reconciliation.json` is PASS and binds the original receipt hash, the same checkpoint hash, backup epoch 9 and `checker_sha256` equal to the runner pin.
5. The fold's controller lock (`controller_lock` from the config) must exist and be free. The exporter then holds it for the whole run, so a resumed controller cannot start concurrently.
6. Free disk is at least 2x payload + 1 GiB.
7. Destination name must be `OWNER/knee-source96-fold{k}-final-YYYYMMDD[-rN]` (`-failed-` in failed mode), owner equal to the backup dataset owner, and never one of the two resume-backup datasets nor the config's persistence dataset.
8. Remote check using the runner's `private_dataset()` rule (identity `ownerUser/datasetSlug` or legacy `id`, `isPrivate` exactly true). Metadata 404 means absent: create. Present: refuse unless `--new-version`, and still refuse when public or misidentified. Metadata HTTP 403 (what Kaggle returned for the intended, not yet created `ozgung/knee-source96-fold0-final-20261012`, per `outputs/analysis/source96-training-20261012/exporter-readonly-transport-check.json`) is UNKNOWN and is never treated as absence: the run refuses unless `--create-fresh` names exactly the destination slug. With it the exporter runs create only (never a version, never overwrite). A conflict makes `create` fail; a create that silently produced version 2 is rejected (version must be exactly 1); and after the create an exact identity plus `isPrivate` true check is mandatory (retried while the new dataset is not yet visible, failing at `--ready-timeout`) before any verification download. The receipt records `existence` as `unknown_metadata_inaccessible`.

## Files copied to the fresh staging directory (hash-while-copy, then re-hash the copy)

From the workdir: `raptor_ft_fold{k}_swa.pt`, `oof_fold{k}.csv`, `run_fold{k}.json`, `raptor_ft_fold{k}.json`, `source96-run-fingerprint.json`, `source96-run-receipt.json`; fold 0 also `source96-parent-completion-reconciliation.json`; if present `train_meta_fold{k}.json`, `initial-state-record.json`.
From outside the workdir (named inside the dataset): `run-config.json`, `staging-receipt.json` (`stage_root/staging-receipt.json`), `initializer-descriptor.json`, `persist-status-fold{k}.json` (config `status_file`).
Only with `--include-logs`: `log_fold{k}.txt`, `log_export_fold{k}.txt`, `source96-training.log`, `source96-backup.log`, `source96-runtime-stdout.log`, `source96-runtime-stderr.log`, optional controller log. These are unsanitized private diagnostics; the dataset is private but they are off by default.
Excluded: `raptor_resume_fold{k}.pt` (already backed up), `raptor_ft_fold{k}.pt`, K94 secondary outputs, `train_knee_*.py`.

`MANIFEST.json`: schema, label `FINAL`, fold, dataset, `source_revision`, exporter and runner sha256, all pins, `receipt_mode`, per-file `size`/`sha256`/role/origin/source, `checkpoint_sha256` with its cross-check sources (run record, staged copy, controller or reconciliation receipt must all be equal), `selected_epochs`, `epochs_completed`, `k_eval`, `n_val`, the resume-backup dataset name (reference only). `dataset-metadata.json` uses the same CC0-1.0 license string as the existing persister datasets.

## Upload and verification

Create uses `kaggle datasets create -p STAGING -q --keep-tabular` with no `--public`, so the dataset is private by default and CSV bytes are not converted; new versions use `kaggle datasets version ... --keep-tabular` without `-d` (old versions kept). Right after upload the identity/privacy check is repeated (a public result fails before any download). The exporter then polls `datasets status` until `ready` with a version at least prior+1, downloads that exact version (`OWNER/SLUG/N`, `--unzip`) into a fresh directory under `--verify-root`, and compares every file's size and sha256 plus the downloaded `MANIFEST.json` byte-for-byte against the local one; extra, missing or symlinked files fail. Transient status/download errors and lagging versions are retried until `--ready-timeout` (default 900 s); an upload is never repeated automatically. After verification it re-checks that the workdir snapshot (names, sizes, mtimes, inodes) is unchanged. The receipt (`--receipt`) is written in every outcome with `state` `verified` or `failed` plus `phase`, `upload_attempted`, `problems`, `dataset_version`, `manifest_sha256`, per-file hashes, timings. A failed download is kept for inspection; a verified one is deleted. If `create` or `version` timed out after upload may have started, re-run with `--new-version` and a fresh staging path (the privacy check is repeated).

Failed mode (`--failed-receipts-only`, dataset `...-failed-...`) is allowed only when `primary_complete` rejects the workdir. It copies receipts and meta only, adds `FAILED.txt`, labels the manifest `FAILED` with a failure summary and a workdir name/size listing, and copies no checkpoint, OOF or resume.

## Order of operations

1. Now to fold 0 end (about 19:20 UTC): nothing runs on the VM. The exporter is inert until invoked; staging the file on the VM without exec (the parent knows how `colab_source96_run_v2.py` was staged) is the only prerequisite.
2. Fold 0 controller exits with the expected FAIL receipt (do not delete it). Fold 1 trains on until about 19:50.
3. The foreground supervision cell only ends when both controllers end, and Codex forbids queuing `colab exec` behind it. Two options:
   - Option A, recommended: wait for the supervision state `CONTROLLERS_ENDED` (about 19:50), then exec in this order: (a) `vm-fold0-final-reconcile.py` (already planned by Codex; produces the fold 0 reconciliation receipt), (b) export fold 0, (c) export fold 1 (no reconciliation needed). Fold 0 export is delayed about 30 minutes, but no GPU or CPU contention with the fold 1 trainer and no rule change. Risk in the 19:20 to 19:50 window: fold 0's final SWA/OOF exist only on the VM; its epoch-9 resume backup is verified by the persister's final request before the controller exits (the reconcile script re-checks it), so the SWA could be rebuilt from it if the VM were lost.
   - Option B, preservation copy only: at about 19:20 pull fold 0's small files (OOF, run/meta JSON, receipts, about 0.2 MB) and optionally the 293 MB SWA with REST `colab download` into ignored `outputs/`. This cannot replace the exporter (no reconciliation receipt, lock check or Kaggle verification without exec) and the exporter is not designed to run on the laptop. Large-file REST transfer has not been tried in this session.
   Recommendation: A, optionally with the small-file part of B as cheap insurance. Do not export fold 0 while fold 1 trains.

## Command (fold 0; run after the supervision cell ended and the reconciliation receipt exists)

```bash
KAGGLE_CONFIG_DIR=/content /usr/bin/python3 -u /content/colab_source96_export_terminal.py \
  --fold 0 --workdir /content/trackA96-fold0 \
  --config /content/source96-fold0-run-config.json \
  --config-sha256 2590477c0b21504cdf007ca12b11bfd2377995dfbfa1f162e95d71e7981df69c \
  --package-manifest-sha256 c7dffd7365e555387bc1936ef2f0ee9156d795eba2f4fd490647739b707ab2ff \
  --cache-content-sha256 c113958a0cc7595696fab688d0cd6832a45468625c61cecc08308844b7e538ca \
  --controller-sha256 e8494e9d604daac469606e7e0af62b3d4cdedc9978b403f9813fd5803f023881 \
  --runner /content/colab_source96_run_v2.py \
  --runner-sha256 ccddb970360c013e2d48d00ae0e174af0eb6bf99d53ae87e83582dbbdad5d06d \
  --accept-reconciled-receipt \
  --dataset ozgung/knee-source96-fold0-final-20261012 \
  --create-fresh ozgung/knee-source96-fold0-final-20261012 \
  --staging /content/source96-fold0-final-staging \
  --verify-root /content/source96-fold0-final-verify \
  --receipt /content/source96-fold0-final-export-receipt.json \
  --source-revision <PARENT_COMMIT_SHA>
```

Fold 1: `--fold 1`, workdir `/content/trackA96-fold1`, config `/content/source96-fold1-run-config.json` with `--config-sha256 49785f1b0cc7b0f5bcec5442dfa6189084a89006e91839068217ee853ed95736`, `--package-manifest-sha256 83d83b67a23bbcc2d18e535dd09a02c33a5fd1a68b42ddd8f795744373933aba`, `--controller-sha256 ccddb970360c013e2d48d00ae0e174af0eb6bf99d53ae87e83582dbbdad5d06d`, no `--accept-reconciled-receipt`, dataset and `--create-fresh` both `ozgung/knee-source96-fold1-final-20261012` (drop `--create-fresh` if metadata for the new slug returns 404 instead of 403), matching `fold1` staging/verify/receipt names. The real credential is `/content/kaggle.json`, hence `KAGGLE_CONFIG_DIR=/content` in both commands. Exit code 0 only for `verified`; one JSON summary line is printed and the full receipt is at `--receipt`.

## Resources (estimates, unmeasured on the VM)

- Payload per fold about 293 MB: SWA checkpoint 292.8 MB (the historical fold 1 file is 292,826,563 bytes), OOF about 0.18 MB, run JSON about 7 KB, the rest small receipts. Logs, if included, add an unknown amount.
- Disk: staging about 293 MB plus verification download about 293 MB; the tool refuses below about 1.6 GiB free.
- Memory: streaming 8 MiB chunks. CPU: the checkpoint is hashed four times (`primary_complete`, copy, staged re-hash, downloaded copy), likely seconds each.
- Upload and download time: not measured. The first real run will populate `staging_seconds`, `upload_seconds`, `verify_seconds` in the receipt.
- Kaggle quota: two new private datasets of about 0.3 GB each, additional to the existing resume backups.

## Checks and limits

- Unit tests: `.venv/bin/python -m unittest tests.test_colab_source96_export_terminal -v`. Synthetic workdirs and a fake transport only; `KaggleTransport` is tested through injected command and metadata functions, so the real CLI/SDK behaviour (status JSON fields, create privacy default and conflict wording, `--keep-tabular` effect, `slug/N` download) is unexercised until the first run. Metadata 403 for a not-yet-created slug and a 200 for an existing private dataset are the only real transport observations (read-only check above).
- The runner is imported from the pinned file; the tests use the repository copy of `scripts/colab_source96_run.py` (sha256 `ccddb970...`).
- Decisions (Codex): logs omitted (`--include-logs` stays off); staging is kept until an independently verified receipt has been recovered; the original fold 0 controller FAIL receipt is preserved and exported alongside the reconciliation with the explicit `--accept-reconciled-receipt` flag. Open: license string (`CC0-1.0`, copied from the persister).

## Parent integration,18:15UTC

Independent Sol review:56focused testsPASS/0skip, earlier fold1/non-FAIL reconciliation counterexamples refused. SourceSHA4afd7f883ecff8d557dea14aea38b7895e8915b949737084a7f9de04afc301ef. Real read-only metadata/status parsing passed on existingprivatebackup; proposedfreshslug returned403, motivating the explicit create-only option. One separate syntheticprivate transportprobe is in progress, no model/competition artifacts: `outputs/analysis/source96-training-20261012/terminal-transport-probe/receipt.json`. Actual finalexport remains unexercised.
