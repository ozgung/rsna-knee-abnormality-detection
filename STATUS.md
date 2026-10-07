# Project status and session handoff

Last updated: **2026-10-07 (Europe/Istanbul)**

Read this file at session start. Update it after meaningful changes and before handing off. Working rules are in [AGENTS.md](AGENTS.md); setup and the modeling outline are in [README.md](README.md).

## Current objective

Build a reproducible, image-only knee MRI baseline and an offline competition inference path. The next milestone is a verified metadata/schema audit and leakage-aware validation design, before real model training.

## Verified snapshot

| Area | State / evidence |
| --- | --- |
| Project | Local scaffold at `/Users/ozgun/Dev/rsna-knee-abnormality-detection` |
| Version control | No Git repository as of this session; `git status --short` reports “not a git repository.” No remote/branch/commit is established here. |
| Utilities | `knee/audit.py`, `knee/device.py`, and `knee/submission.py` exist; five unit tests pass. |
| Smoke check | `python3.12 -m knee.smoke` passes on CPU with synthetic CSVs; `torch_forward: false`. |
| Model | `knee/model.py` contains an untrained fixed-bag toy encoder. Its PyTorch forward pass was not run this session. |
| Configuration | `configs/local.toml`: 12 targets, seed 42, automatic device selection, initial 2 GiB cache budget. Several settings are proposed for future training, not an implemented training pipeline. |
| Data and results | No dataset, audit artifacts, folds, trained checkpoints, or measured real-data scores are present in this project. |
| Environment | No project `.venv` or dependency lockfile exists. Python 3.12 executes the dependency-free checks. ML environment setup remains pending. |
| Remote compute tools | `kaggle` and `colab` are installed and their `--help` commands succeed. Kaggle exposes competition/kernel operations; Colab exposes session creation, execution, and file transfer. Authentication, competition access, and runtime connectivity have not been tested. A Colab browser MCP tool is also available, but no browser connection was opened. |
| Handoff setup | `AGENTS.md` establishes startup/closeout rules, validation invariants, experiment records, and completion criteria. README links here. |

Historical setup notes reported an Apple M4 Max with 48 GiB RAM and about 34 GiB free disk; remeasure before resource-intensive work. They also reported manual Codex project registration. Neither hardware capacity nor sidebar registration was rechecked in this session.

## Work queue

States: `ready`, `active`, `blocked`, `done`. Keep one implementation task active. IDs are stable; update rows in place. This local queue is sufficient until external issue tracking is useful.

**This plan is tentative.** Reprioritize, revise, or drop work as experiments, data access, rules, available compute, and time remaining change. Record significant changes and their evidence in Working decisions. Protect a working submission candidate and leave time for offline rehearsals before the verified deadline.

| ID | State | Task | Acceptance / exit criteria |
| --- | --- | --- | --- |
| PM-01 | done | Establish session handoff and agent guidance | STATUS and AGENTS exist, README links them, baseline checks recorded. |
| SETUP-01 | ready | Establish local version control | Initialize Git, inspect ignored/sensitive files, and create a clean initial tracked snapshot; record commit. Remote publication is a separate action. |
| DATA-01 | blocked | Verify competition access and audit metadata | Current official rules/metric/schema checked; authorized metadata and sample submission available; aggregate audit records target order, missingness, identity fields, and overlap. |
| ENV-01 | ready | Establish reproducible ML environment | Isolated environment, recorded/locked dependencies appropriate to platform, and successful explicit CPU/MPS torch smoke check. |
| VAL-01 | blocked | Define folds and prevalence baseline | After DATA-01: persist reproducible grouped split, verify disjoint identities, define missing-label/AUC policy, and report baseline support/metrics. |
| PIPE-01 | blocked | Implement bounded DICOM-to-study loading | After DATA-01 and ENV-01: representative studies decode; ordering/orientation and masks tested; cache/time/memory measured within budget. |
| TRAIN-01 | blocked | Train and evaluate first real baseline | After VAL-01 and PIPE-01: reproducible run, held-out per-target/macro metrics versus prevalence, checkpoint, config, and experiment summary. |
| SUBMIT-01 | blocked | Rehearse offline Kaggle inference | After TRAIN-01: image-only run without internet, exact schema/IDs, finite predictions, measured resource headroom under verified limits. |

## Blockers and unresolved choices

- Competition access/terms acceptance and metadata availability have not been verified. The prior setup did not accept terms or download data.
- README competition counts, dates, and limits were inherited from an earlier review; the setup fetch could not read Kaggle page bodies. They are provisional, not newly verified facts. Recheck live sources before using them to schedule work or submit.
- Exact patient identity, label schema, report languages, missingness, and usable validation size depend on DATA-01. Do not invent these values.
- No training budget or measured throughput is established. Agree on a bounded run before launching substantial training.
- No active training/background jobs or resumable checkpoints were created by this session.
- CLI installation is verified; authenticated access and available compute quota still need read-only checks before remote execution.

## Working decisions

| Date | Decision | Reason / revisit trigger |
| --- | --- | --- |
| 2026-10-07 | Keep management in the repository: STATUS for current state and tentative planning, AGENTS for rules, README for setup. No separate roadmap. | Keep one source for progress and planning; revise priorities as evidence and competition constraints change. |
| 2026-10-07 | Use one active implementation task with a concrete acceptance check. | Keeps sessions reviewable and makes the next action unambiguous. |
| 2026-10-07 | Start with existing tools; no custom skills/plugins installed. | Extract narrow skills after workflows stabilize; add connectors when external tracking is needed. |
| 2026-10-07 | Prioritize schema/folds and a simple baseline before complex models or weak-label expansion. | Useful model comparisons depend on reliable validation. |
| 2026-10-07 | Recommend Astra medium effort normally, high for complex analysis. | A starting recommendation, not a benchmark result or a changed app/model setting. |

Add detailed decision records under `docs/decisions/` only when alternatives or consequences no longer fit in this table. Link them here.

## Latest verification

Executed **2026-10-07**, from the project root:

```text
python3.12 -m unittest discover -s tests -v
PASS: 5 tests

python3.12 -m knee.smoke
PASS: device=cpu, torch_forward=false, data=synthetic only, training=false
```

These checks were executed during the initial handoff setup on this date; they were not rerun for the subsequent documentation-only planning update. They cover CSV audit behavior, device selection, and submission validation/order, not DICOM loading, GPU execution, training, metrics, or Kaggle inference. Local Markdown links and formatting were checked after the planning update.

Remote-tool inspection on the same date: `command -v kaggle`, `command -v colab`, `kaggle --help`, and `colab --help` succeeded. These checks establish installed command capabilities only, not authenticated service access.

## Latest session handoff

- **Completed:** verified installed Kaggle and Colab CLI commands and recorded their capabilities/access boundary. Existing CLI tools are sufficient to start; a separate connector is optional for workflows such as interactive notebook editing. No runtime code changed.
- **Not performed:** Git initialization, dependency installation, data acquisition, model training, plugin installation, or app model/config changes.
- **Next action:** SETUP-01 for a general “continue” request; DATA-01 becomes the research priority once authorized competition metadata is available. ENV-01 can proceed independently of metadata.
- **Suggested session prompt:** “Read STATUS.md and AGENTS.md. Complete SETUP-01, verify the tracked files contain no data or secrets, and update STATUS.md with the result. Keep repository publication separate.”
