# RSNA Knee Abnormality Detection

Our entry to the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) Kaggle competition (August to October 2026): twelve findings per knee MRI study, scored by macro ROC AUC, 4,407 training studies with free-text radiology reports and only 58 expert-labelled studies.

| Entry | Public | Private | Note |
|---|---|---|---|
| Selected finals (S4, P2): public R6 + Raptor blends | 0.958 | 0.942 | rank 715 of 6,425 |
| P11: eight own ConvNeXt-S readers on image-teacher labels, no public weights | 0.956 | **0.944** | not selected; inside the silver band |
| Private winner | | 0.960 | gold line about 0.955, silver 0.943 |

**Start here:**

- [What the competition taught us](docs/postmortem/competition-review-20261023.md): results, the public-to-private shift, what the medal teams did, what worked and what did not, and a checklist for the next competition. Built from twelve competitor write-ups plus our own paired entries.
- [Our best entry, P11](docs/postmortem/own-solution-p11-20261023.md): the solution write-up for the own-model line, with the label rounds, members and the evidence from paired submissions.
- [Final-selection retrospective](docs/research/retrospective-20261023.md): why the pre-registered selection rule picked the wrong two entries, and the rules we take forward.

![Public score of every scored submission, by date](docs/postmortem/figures/review-01-journey.png)

## What is in this repository

Everything is code, configuration and notes. No competition data, labels, report text, caches, weights or credentials are tracked; `data/`, `outputs/`, `checkpoints/` and every binary data format are ignored.

| Area | What it holds |
|---|---|
| `knee/` | The Python package: DICOM-to-study preprocessing with masks and orientation checks, label tables and teacher blends, the own ConvNeXt (S6-recipe) training and qualification lanes, the controller that wraps the public R6 + Raptor recipe with deadline guards and ledger salvage, schema and submission validation. |
| `scripts/` | Entry points for building Kaggle notebooks and packets, running Colab fits, collecting receipts, benchmarking label tables and building the out-of-fold teacher columns (`scripts/laneb/`, `scripts/s6v2/`). |
| `tests/` | About 1,500 unit tests on synthetic fixtures: leakage, masks, schemas, device selection, submission integrity, controller stage order and fallbacks. |
| `docs/experiments/` | One note per experiment or qualification run, with the exact checks, numbers and limits (153 notes). |
| `docs/research/` | Strategy notes, public-notebook scans, domain research and the retrospective (32 notes). |
| `docs/postmortem/` | The two write-ups above, as HTML (source) and Markdown with rendered figures. |
| `docs/archive/` | The campaign's working state: the final `STATUS.md` snapshot, earlier snapshots, the Claude/Codex message log and the original README. History only. |
| `AGENTS.md`, `STATUS.md` | Durable working rules for AI-assisted sessions and the current (closed) project state. |

The pipeline that produced P11, in one paragraph: public report-derived label tables are averaged into a base, then rewritten in three rounds by out-of-fold image teachers (first the public R6 model, then our own fold models together with R6 and R7, then our own round-2 fold models). nartaa's public S6 recipe (ConvNeXt-small, 288 px crop, two views per slice, three-slice windows, per-finding attention, 16 epochs) is retrained on each label table, and eight such members are averaged at the logit level. Inference is one offline notebook on two T4s with a strict CPU load check for every checkpoint, sentinel repeats, per-stage deadlines and salvage of finished members.

## Running the checks

Dependency-free checks from the project root:

```bash
python3.12 -m unittest discover -s tests -v
python3.12 -m knee.smoke
```

With the ML environment (Python 3.12.9 and uv, pinned in `uv.lock`):

```bash
uv sync --locked --extra ml --no-python-downloads --cache-dir cache/uv
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m knee.smoke --torch --device cpu
```

Tests that need numpy, pydicom or torch skip or error without the ML extra; the dependency-free subset passes on a bare interpreter. The lock was validated on macOS arm64; Kaggle and Colab kept their preinstalled CUDA torch.

## Data, weights and licences

- Competition data is used under the RSNA research-use agreement and never leaves ignored directories. Report text, study identifiers and per-study predictions are not recorded in the notes; the notes carry aggregate numbers only.
- Public weights and code we built on are referenced by dataset slug and SHA-256, not vendored: nartaa's Raptor CoAtNet and S6 ConvNeXt (weights CC BY-NC 4.0 / "other", training code Apache-2.0), heliosli's R6 and R7 compact readers and anatomy checkpoint, goodpjw2008's 2.5D reader (Apache-2.0, licence text in `docs/licenses/`), Dread Development's Raptor pipeline (CC0 weights). `knee/fresh_reader.py` adapts goodpjw2008's preprocessing and says so.
- The fastMRI student member used NYU fastMRI knee DICOMs under NYU's data-use agreement; those weights stay private. No OAI-derived weights are in P11.
- Public label tables by pilkwang, stevenleehans, lixin73 and nartaa (CC0 or competition-use) feed the base label table.

This repository's own code and notes are released under the [Apache License 2.0](LICENSE).

## Acknowledgements

RSNA, the annotating radiologists and Kaggle for the data and the competition. nartaa, heliosli, goodpjw2008 and Dread Development for the public models the whole field built on. The write-up authors cited in the review for sharing what worked. The code and notes were written with Claude and Codex as coding agents; the decisions and the mistakes are ours.
