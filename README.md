# RSNA Knee Abnormality Detection

Offline-first Python project for [the Kaggle competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection). It implements a bounded DICOM-to-study pipeline, frozen DINOv2 feature extraction, weak-label head training, and offline Kaggle inference. Current run results and completion state are in [STATUS.md](STATUS.md). Tracked files contain no competition data, labels, credentials, or weights.

## Project management and AI-assisted development

[STATUS.md](STATUS.md) is the main handoff between sessions: verified state, tentative planning, prioritized tasks with acceptance criteria, blockers, decisions, test results, and the next action. [AGENTS.md](AGENTS.md) gives Codex the durable project instructions. Keep evolving status and planning in STATUS rather than duplicating them here.

Use one task per implementation session: specify the outcome, constraints, and acceptance check; implement and verify it; update the handoff. A useful prompt is: “Read STATUS.md and AGENTS.md. Complete task ID, run its relevant checks, and update STATUS.md with evidence and the next action.”

Recommended management setup:

- Start with Git, small task branches, this local queue, and reviewable diffs. Once a remote exists, use issues/PRs when helpful; a simple Backlog / Ready / In progress / Review / Done board is enough. Link issues from STATUS instead of duplicating their full descriptions.
- Keep current decisions in STATUS. Add `docs/decisions/` for substantial tradeoffs and `docs/experiments/` for actual run summaries when needed. Store large artifacts in ignored output directories and record their locations.
- For every experiment, record the hypothesis, data/split version, code revision, resolved configuration, seed, environment, metrics, runtime, and outcome. Begin with a small baseline and validate folds before scaling.
- Use Astra with medium reasoning effort as a starting point; use high for difficult debugging, leakage analysis, and architecture decisions. Select the model in Codex settings; AGENTS is instruction text and does not change the selected model.
- No extra skill or plugin is required now. Turn stable, repeated workflows into narrow skills later, such as recording experiments or rehearsing offline submissions. Use GitHub integration when there is a remote to manage, marimo when choosing live notebook work, and research tools for substantive literature tasks.

The concise-instructions approach follows [OpenAI's Astra guidance](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra). See the official [AGENTS.md documentation](https://learn.chatgpt.com/docs/agent-configuration/agents-md), [skills documentation](https://learn.chatgpt.com/docs/build-skills), and [configuration documentation](https://learn.chatgpt.com/docs/config-file/config-basic) for how those mechanisms differ.

## Run now, without downloads

```bash
cd ~/Dev/rsna-knee-abnormality-detection
python3.12 -m unittest discover -s tests -v
python3.12 -m knee.smoke
```

The default smoke check writes only temporary synthetic CSVs, validates 12 probabilities and row ordering, then cleans them up. With no PyTorch, automatic device selection returns CPU; this is not proof of a working GPU runtime.

## Reproducible local ML environment

Use Python **3.12.9** (recorded in `.python-version`) and **uv 0.9.13**, the tool version used for this setup. Runtime dependencies and distribution hashes are recorded in [uv.lock](uv.lock); the setuptools build backend is pinned in [pyproject.toml](pyproject.toml). From the project root:

```bash
uv sync --locked --extra ml --no-python-downloads --cache-dir cache/uv
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m knee.smoke
.venv/bin/python -m knee.smoke --torch --device cpu
env -u PYTORCH_ENABLE_MPS_FALLBACK .venv/bin/python -m knee.smoke --torch --device mps
```

`--locked` fails if the project metadata needs a lock update; upgrades are deliberate. See the [uv locking documentation](https://docs.astral.sh/uv/concepts/projects/sync/). Keep `--extra ml` when syncing this environment; omitting it removes the optional ML packages. `.venv/` and `cache/` are ignored. The default dependency-free commands above still work without installing ML packages. See [STATUS.md](STATUS.md) for the verified environment and backend results.

This environment is validated locally on macOS arm64 only. The universal lock also contains other platforms' dependencies, but CUDA/Kaggle compatibility has not been tested. On CUDA hosts, preserve the platform's preinstalled torch or use the [official PyTorch installer](https://pytorch.org/get-started/locally/) for its driver/runtime and record a separate tested environment. Do not copy Mac wheels to Kaggle. [MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html) describes Apple GPU availability checks. Explicit device requests fail when unavailable; the MPS check above disables inherited CPU fallback.

`configs/local.toml` starts with float32, batch size 1, 224px, 8 slices/plane, no loader workers and a 2 GiB cache budget. These are proposed experiment settings; the synthetic forward pass deliberately uses only 32px and 2 slices/plane. Device `auto` prefers CUDA, then MPS, then CPU. Explicit unavailable devices fail. There is no silent per-operation CPU fallback. The tiny MPS forward check passed; real-data extraction and offline inference were measured on Kaggle CUDA, as recorded in the experiment report. Real-loader throughput on MPS remains unmeasured. CUDA-specific mixed precision code should be enabled only on CUDA after testing.

## Competition facts and verification boundary

The October 8 official-page and real-CSV audit verified 12 study-level MRI abnormality probabilities scored by macro ROC-AUC, 4,407 training studies, 58 fully labeled studies, and 24,371 series. The full data listing is approximately 570 GB; it is not downloaded locally. Reports are training-only; hidden test inference (roughly 1,300 studies) works from images and series metadata. Exact target names and order are always read from the official sample CSV.

The live official pages verified on October 8 record entry deadline **October 15, 2026, 23:59 UTC**, final submission **October 22, 2026, 23:59 UTC**, notebook runtime up to 9 hours, internet disabled, and public/free pretrained weights allowed subject to the rules. Source pages: [overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview), [data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data?select=test_series), [rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules). The CLI can read these pages using `kaggle competitions pages -c rsna-knee-abnormality-detection --content --format json`. Recheck them before later calendar planning. This implementation uses attached public labels and no external report-processing API.

## Mac, Kaggle and Colab workflow

| Environment | Intended work | Resource plan |
| --- | --- | --- |
| Mac MPS | CSV audit, DICOM decoding checks, a few studies, model debugging | Keep a small cache (initially <=2 GiB); inspect free disk before expansion. Shared RAM is not all available to the model. |
| Kaggle | Full raw dataset access, preprocessing, validation, final offline inference | Attach competition data; stream selected series; store compact caches and weights. Benchmark full-study decoding as part of the 9-hour budget. |
| Colab | Selected CUDA experiments | Use the user-authorized existing 350-credit allocation for bounded project runs before expiry; see STATUS for remaining-budget evidence and expiry. Transfer only compact authorized caches. Record each run and release idle runtimes; new credit purchases need separate authorization. |

The complete raw corpus cannot fit the current free disk. Compact multi-plane 2D/2.5D models are plausible for local experiments; no throughput, full training duration or score is promised without measured batches. Avoid starting with a full-volume 3D model.

## Technical approach reference

These are implementation considerations, not a fixed execution sequence. Use [STATUS.md](STATUS.md) for the current tentative plan; report-derived supervision and other improvements depend on baseline evidence and available time.

1. User joins/accepts rules in Kaggle when ready. Audit the small metadata CSVs and the official sample submission before obtaining image samples. Do not expose credentials in notebooks or source control.
2. Inspect schemas, missing labels, report languages, patient/study/series identity and overlap. Use `python -m knee.audit path/to/file.csv` then `--id-column EXACT_COLUMN_NAME` from the observed header. This command reports counts/headers, not report text. Keep CSVs in ignored `data/`.
3. Establish folds at patient level when an identifier exists; otherwise document limitations of study-level grouping. All slices, series, repeat exams and report-derived examples belonging to a validation patient must stay outside training. Fit text-label mappings using training folds only.
4. Derive weak training labels from reports with language-aware negation, uncertainty and anatomy handling. Do not interpret an unmentioned finding as a confirmed negative. Keep uncertain labels masked; record provenance/confidence and manually assess errors on available trusted labels. Any external processing of reports must first satisfy official rules and data restrictions.
5. Build a compact 2D or 2.5D encoder with slice -> series/plane -> study pooling and 12 sigmoid outputs. Use images only at inference; reports can supervise training. `knee/model.py` is a toy fixed-bag 2D architecture for synthetic shape checks, not a DICOM loader or implemented training pipeline. Real loading needs orientation/order checks, intensity normalization, missing-plane masks and padding masks. Audit compressed DICOM transfer syntaxes before adding codecs.
6. Train initially on a bounded sample; compare against prevalence predictions using held-out study labels. Use masked binary cross-entropy for missing labels and confidence weights for weak labels. Report each target AUC and macro mean; explicitly flag targets with only one class where AUC is undefined. Tiny labeled validation sets will have high variance.
7. On Kaggle, run an offline rehearsal with all code, wheels (if needed) and allowed weights attached. Inspect the real sample CSV to set exact column names/order. `knee.submission.write_submission` validates an ID-to-12-probabilities mapping and preserves sample order; it does not infer probabilities or discover test studies. Confirm hidden-test handling, no train-report dependency, finite outputs, disk/RAM use and total wall-clock time with margin below 9 hours.

## Real-data pipeline

The executable baseline is separate from the toy `knee/model.py` shape check:

| Module | Responsibility |
| --- | --- |
| `knee.validation` | Real schema audit; patient/report grouping; prevalence and per-target AUC with explicit missing-label policy. |
| `knee.dicom` | Bounded series selection, physical ordering/crop/orientation, normalization and missing-data masks. |
| `knee.features` | Frozen DINOv2-Small CLS+mean-patch features and finding-specific attention/meanmax heads. |
| `knee.experiment` | Deterministic bounded sampling, train-only statistics, weak-label training, checkpoint and experiment records. |
| `knee.runtime` | Hash-verified local codec wheels; image-only inference; exact actual-test ID coverage. |
| `knee.compare` | Bounded three-head comparison; selection uses only weak-holdout BCE and preserves the baseline. |
| `knee.ensemble` | Compatible probability averaging with one shared encoder pass and verified cache/preprocessing. |
| `knee.rehearsal` | Raw-image versus cached prediction parity, report-free inputs, repeatability and failure cleanup. |

Audit already-authorized small metadata in ignored `data/raw/`:

```bash
python3.12 -m knee.validation --data-dir data/raw --output-dir outputs/validation/20261008
```

Build a private training notebook with the competition, vanilla DINOv2-Small and public v1 weak-label dataset attached:

```bash
python3.12 scripts/build_kaggle.py outputs/kaggle/train knee-dino-train-20261008 scripts/train_kaggle_entry.py --internet
kaggle kernels push -p outputs/kaggle/train --timeout 7200
```

The training entry downloads pinned Linux codec wheels once and saves them beside the feature cache and trained head. Kaggle/Colab preserve the host CUDA PyTorch; the local Mac lock is not installed there. Each run records actual package versions, code/source hash, data fingerprints, preprocessing, seed and metrics. See the [first experiment record](docs/experiments/20261008-first-baseline.md) for the exact bounds and label-provenance limitation.

After that version completes, build an independent offline rehearsal:

```bash
python3.12 scripts/build_kaggle.py outputs/kaggle/rehearsal knee-dino-rehearsal-20261008 scripts/rehearse_kaggle_entry.py --no-weak-labels --kernel-source ozgung/knee-dino-train-20261008
kaggle kernels push -p outputs/kaggle/rehearsal --timeout 1800
```

It reprocesses all gold images and compares predictions against the saved cache-based evaluation, then checks the visible test twice. The final entry only reads the actual test metadata/images and checkpoint:

```bash
python3.12 scripts/build_kaggle.py outputs/kaggle/submission knee-dino-submission-20261008 scripts/submit_kaggle_entry.py --no-weak-labels --kernel-source ozgung/knee-dino-train-20261008
kaggle kernels push -p outputs/kaggle/submission --timeout 28800
```

Only submit a completed, successfully rehearsed version. Inspect its actual version number and current quota first. `STATUS.md` records authorized submissions and their results. Do not overwrite an existing training notebook while a dependent submission is pending. Data, private linkage hashes, feature caches, outputs and weights stay out of Git.

### TRAIN04 coverage and target comparison

The [prospective TRAIN04 protocol](docs/experiments/train04-protocol-20261008.md) defines the new 2,000-training/unchanged-200-development reference. `knee.coverage_cache` stores variable-length selected series so training can draw new within-series triplets each epoch. `knee.coverage_training` retains original observation/confidence masks and compares alternate soft targets only after a useful reference completes. Both the old fixed-view pilot and scored public readers remain separate.

After the full remote cache completes, download its JSON receipts only, then build the first private training notebook. The helper uses 100-file pages with two-second pacing, stops when all requested reports arrive, and resumes from `.report-download-state.json` after a bounded rate-limit failure. Completed downloads are hash-checked on reuse; changing the requested kernel/report set requires a separate destination. Use the Python environment containing the Kaggle SDK:

```bash
/Users/ozgun/miniconda3/bin/python scripts/download_kaggle_reports.py \
  ozgung/knee-train04-cache-full-20261008 outputs/kaggle/train04-cache-full-v2-result \
  --required coverage_cache/plan-private.json coverage_cache/cache-report.json coverage_cache/index-private.json
.venv/bin/python scripts/build_coverage_training.py \
  outputs/research/2026-10-08/convnext-audit \
  outputs/kaggle/train04-reference knee-train04-reference-20261008 \
  outputs/kaggle/train04-cache-full-v2-result/coverage_cache \
  --cache-kernel ozgung/knee-train04-cache-full-20261008
```

The builder rejects incomplete or failed cache receipts. Inspect the private notebook metadata and current quota before pushing with Kaggle's43,200second GPU-session limit. The user removed self-imposed experiment budgets: measured throughput is recorded as an ETA and cannot veto the twelve-epoch fit. The runner verifies every cache file and saves state before the actual platform deadline. To package the controlled alternate targets, add both `--reference-result` (the completed reference's local receipt directory) and `--reference-kernel` (its private notebook reference). The builder checks the reference's prospective performance gate and identical code/cache/configuration; runtime additionally checks package/device identity and the exact saved initial model. No full-cache download to this Mac is required. CUDA training/resume verification state is tracked in STATUS.


For a selected ensemble, `scripts/build_ensemble_kaggle.py` builds the notebook and a separate **weights-only** asset directory. Supply `--assets-dir` and the private `--dataset-source`; use `--rehearsal` for cached-feature versus raw-image parity. Kaggle notebook source must be below1MB, so do not embed checkpoint bytes. Inspect the exact upload directory before creating its private dataset; the builder rejects unexpected files. Rehearsal computes reference predictions from the existing training notebook cache on Kaggle. The pure entry reads only its small heads and test images/metadata. Never upload gold prediction fixtures, reports or identifiers as part of the weights package.


### Selected-checkpoint export provenance

For resumed TrackA exports, `scripts/build_tracka_export_kaggle.py` requires an expected dataset version, source checkpoint SHA and training config JSON. Its runtime verifies source identity and every selected weight tensor before image inference, then records training and export settings separately. [Method, required CLI arguments and evidence](docs/experiments/selected-export-provenance-20261011.md). This qualification does not change existing training recipes or establish a competition score.
