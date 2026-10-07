# RSNA Knee Abnormality Detection

Local experiment scaffold for [the Kaggle competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection). This is setup code, not a trained solution. It contains no competition data, labels, credentials, or weights.

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

`configs/local.toml` starts with float32, batch size 1, 224px, 8 slices/plane, no loader workers and a 2 GiB cache budget. These are proposed experiment settings; the synthetic forward pass deliberately uses only 32px and 2 slices/plane. Device `auto` prefers CUDA, then MPS, then CPU. Explicit unavailable devices fail. There is no silent per-operation CPU fallback. The tiny MPS forward check passed; real-loader operations, memory use, and throughput still need measurement. CUDA-specific mixed precision code should be enabled only on CUDA after testing.

## Competition facts and verification boundary

The prior official-page review supplied with this setup records 12 study-level MRI abnormality probabilities scored by macro ROC-AUC; the data listing is 569.76 GB across 819,640 files. Training has a small directly labeled subset plus multilingual reports. Reports are training-only; hidden test inference (roughly 1,300 studies) must work from images alone. Exact counts such as 58 labeled or 4,407 report studies are **unverified until CSV audit** and must not be assumed.

The supplied official overview review records entry deadline **October 15, 2026, 23:59 UTC**, final submission **October 22, 2026, 23:59 UTC**, notebook runtime up to 9 hours, internet disabled, and public/free pretrained weights allowed subject to the rules. Source pages: [overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview), [data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data?select=test_series), [rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules). The web fetch in this setup returned no readable Kaggle page body, so recheck the live pages before entry or submission. External API use during training needs official clarification; an internet-off inference rule alone does not establish a blanket training API prohibition.

## Mac, Kaggle and Colab workflow

| Environment | Intended work | Resource plan |
| --- | --- | --- |
| Mac MPS | CSV audit, DICOM decoding checks, a few studies, model debugging | Keep a small cache (initially <=2 GiB); inspect free disk before expansion. Shared RAM is not all available to the model. |
| Kaggle | Full raw dataset access, preprocessing, validation, final offline inference | Attach competition data; stream selected series; store compact caches and weights. Benchmark full-study decoding as part of the 9-hour budget. |
| Colab | Selected CUDA experiments | Use only available free quota; transfer compact authorized caches, not the full corpus. Sessions/storage/GPU availability vary. No paid compute is configured. |

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

## Remaining implementation

Dataset access and schema audit, DICOM loader, folds, report labeling, masked training/evaluation, compact cache format, checkpointing and an offline Kaggle inference notebook remain to be built. No real patient data or performance metric has been evaluated. Keep data, reports, secrets, generated CSVs and weights out of Git using the included `.gitignore`.
