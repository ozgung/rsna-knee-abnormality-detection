# Project status and session handoff

Last updated: **2026-10-07 (Europe/Istanbul)**

Read this file at session start. Update it after meaningful changes and before handing off. Working rules are in [AGENTS.md](AGENTS.md); setup and the modeling outline are in [README.md](README.md).

## Current objective

Build a reproducible, image-only knee MRI baseline and an offline competition inference path. The next milestone is a verified metadata/schema audit and leakage-aware validation design, before real model training.

## Verified snapshot

| Area | State / evidence |
| --- | --- |
| Project | Local scaffold at `/Users/ozgun/Dev/rsna-knee-abnormality-detection` |
| Version control | Local Git initialized. Initial scaffold commit `0ae2128` on `main`; current task branch `codex/ml-environment` starts from `955596f`. No remote configured. |
| Utilities | `knee/audit.py`, `knee/device.py`, and `knee/submission.py` exist; five unit tests pass. |
| Smoke check | Dependency-free smoke passes; explicit PyTorch forward checks pass on CPU and host MPS with fallback unset. Synthetic data only. |
| Model | `knee/model.py` contains an untrained fixed-bag toy encoder. Its tiny forward pass is verified on CPU/MPS; no training or real-data validation. |
| Configuration | `configs/local.toml`: 12 targets, seed 42, automatic device selection, initial 2 GiB cache budget. Several settings are proposed for future training, not an implemented training pipeline. |
| Data and results | No dataset, audit artifacts, folds, trained checkpoints, or measured real-data scores are present in this project. |
| Environment | Isolated `.venv`: Python 3.12.9, uv 0.9.13, torch 2.14.1, numpy 2.5.3, pandas 3.0.6, pydicom 3.0.2, scikit-learn 1.9.1. Exact runtime dependencies/hashes in `uv.lock`; Python and build backend pinned. Locked sync/check and package compatibility pass. |
| Remote compute tools | `kaggle` and `colab` are installed and their `--help` commands succeed. Kaggle exposes competition/kernel operations; Colab exposes session creation, execution, and file transfer. Read-only Kaggle leaderboard, notebook list/source, discussion list/comments, and public model search/card access succeeded on 2026-10-07 with network-enabled CLI calls. Competition data entitlement, terms acceptance, and remote execution remain unverified. A Colab browser MCP tool is also available, but no browser connection was opened. |
| Handoff setup | `AGENTS.md` establishes startup/closeout rules, validation invariants, experiment records, and completion criteria. README links here. |

Host verified during ENV-01: Apple M4 Max, 48 GiB RAM, macOS 27.0.1 arm64; free disk 34.56 GiB before installation and 33.52 GiB after. `.venv` uses about 774 MiB and `cache/uv` about 780 MiB (below the initial 2 GiB cache budget). Remeasure before expanding caches. Sidebar registration remains unverified.

## Work queue

States: `ready`, `active`, `blocked`, `done`. Keep one implementation task active. IDs are stable; update rows in place. This local queue is sufficient until external issue tracking is useful.

**This plan is tentative.** Reprioritize, revise, or drop work as experiments, data access, rules, available compute, and time remaining change. Record significant changes and their evidence in Working decisions. Protect a working submission candidate and leave time for offline rehearsals before the verified deadline.

| ID | State | Task | Acceptance / exit criteria |
| --- | --- | --- | --- |
| PM-01 | done | Establish session handoff and agent guidance | STATUS and AGENTS exist, README links them, baseline checks recorded. |
| SETUP-01 | done | Establish local version control | Initial commit `0ae2128` contains 13 reviewed source/config/documentation files. Ignore coverage and dependency-free checks pass. Local only; no remote publication. |
| DATA-01 | blocked | Verify competition access and audit metadata | Current official rules/metric/schema checked; authorized metadata and sample submission available; aggregate audit records target order, missingness, identity fields, and overlap. |
| ENV-01 | done | Establish reproducible ML environment | Isolated environment installed from `uv.lock`; pinned Python/build backend; explicit CPU/MPS forward checks passed. Validated on local macOS arm64 only. |
| VAL-01 | blocked | Define folds and prevalence baseline | After DATA-01: persist reproducible grouped split, verify disjoint identities, define missing-label/AUC policy, and report baseline support/metrics. |
| PIPE-01 | blocked | Implement bounded DICOM-to-study loading | After DATA-01 and ENV-01: representative studies decode; ordering/orientation and masks tested; cache/time/memory measured within budget. |
| TRAIN-01 | blocked | Train and evaluate first real baseline | After VAL-01 and PIPE-01: reproducible run, held-out per-target/macro metrics versus prevalence, checkpoint, config, and experiment summary. |
| SUBMIT-01 | blocked | Rehearse offline Kaggle inference | After TRAIN-01: image-only run without internet, exact schema/IDs, finite predictions, measured resource headroom under verified limits. |

## Blockers and unresolved choices

- Competition access/terms acceptance and metadata availability have not been verified. The prior setup did not accept terms or download data.
- Official overview/data text was readable through the web search index on 2026-10-07: macro AUC over 12 targets, images-only test input, about 1,300 test studies, 9-hour offline notebook limit, entry/merge October 15 and final October 22 at 23:59 UTC. Direct page fetch remained unreadable and CLI pages parsing failed; recheck the live timeline before scheduling. Exact training counts still require CSV audit.
- Exact patient identity, label schema, report languages, missingness, and usable validation size depend on DATA-01. Do not invent these values.
- No training budget or measured throughput is established. Agree on a bounded run before launching substantial training.
- No active training/background jobs or resumable checkpoints. ENV-01 installation and checks completed.
- The sandbox hides MPS and causes uv 0.9.13 macOS configuration lookup failures; host execution passed. CUDA is unavailable here and untested. Other operations and real-data throughput remain unverified.
- Public Kaggle CLI reads are verified; competition data entitlement, available compute quota, and remote execution remain unverified.

## Working decisions

| Date | Decision | Reason / revisit trigger |
| --- | --- | --- |
| 2026-10-07 | Keep management in the repository: STATUS for current state and tentative planning, AGENTS for rules, README for setup. No separate roadmap. | Keep one source for progress and planning; revise priorities as evidence and competition constraints change. |
| 2026-10-07 | Established local `main` snapshot and task branches; keep commits local. | SETUP-01 completed; no remote configured. |
| 2026-10-07 | Use existing uv with a checked-in universal lock, Python 3.12.9 pin, and setuptools 84.0.0 pin. Next priority is DATA-01. | ENV-01 passed on macOS arm64. Other lockfile platforms are untested; preserve/test CUDA environments separately. Metadata and rules are now the blocking inputs. |
| 2026-10-07 | Use one active implementation task with a concrete acceptance check. | Keeps sessions reviewable and makes the next action unambiguous. |
| 2026-10-07 | Start with existing tools; no custom skills/plugins installed. | Extract narrow skills after workflows stabilize; add connectors when external tracking is needed. |
| 2026-10-07 | Prioritize schema/folds and a simple baseline before complex models or weak-label expansion. | Useful model comparisons depend on reliable validation. |
| 2026-10-07 | Recommend Astra medium effort normally, high for complex analysis. | A starting recommendation, not a benchmark result or a changed app/model setting. |

Add detailed decision records under `docs/decisions/` only when alternatives or consequences no longer fit in this table. Link them here.

## Latest verification

Executed **2026-10-07** for ENV-01 (base code revision `955596f`; runtime code unchanged):

```text
uv sync --locked --extra ml --no-python-downloads --cache-dir cache/uv
PASS: isolated .venv installed, 22 packages; repeat sync succeeded

uv sync --locked --extra ml --check --offline --no-python-downloads --cache-dir cache/uv
PASS (host): 22 packages audited, would make no changes

uv pip check --python .venv/bin/python --cache-dir cache/uv --offline
PASS: all 22 installed packages compatible

.venv/bin/python -m unittest discover -s tests -v
PASS: 5 tests

python3.12 -m knee.smoke
.venv/bin/python -m knee.smoke
PASS: CPU, torch_forward=false (sandbox)

.venv/bin/python -m knee.smoke --torch --device cpu
PASS: CPU, torch_forward=true

env -u PYTORCH_ENABLE_MPS_FALLBACK .venv/bin/python -m knee.smoke --torch --device mps
PASS (host): MPS, torch_forward=true; fallback unset
```

The MPS command first failed inside the sandbox with explicit device unavailable; rerunning with host GPU access passed. Initial sandbox uv lock/check calls failed in macOS system-configuration lookup; host calls succeeded. No fallback or device-selection code was changed. Host diagnostic imports of numpy, pandas, pydicom, sklearn and torch succeeded; MPS built/available, CUDA unavailable.

Detailed local environment evidence (all installed versions, lock SHA-256, hardware, revision and check summary): [outputs/environment/env-01.json](outputs/environment/env-01.json), ignored by Git. Reproduction commands are in [README.md](README.md#reproducible-local-ml-environment). The lockfile and Python/build pins are tracked. Environment/cache/output/build artifacts remain ignored. Local Markdown links and Git whitespace checks passed.

These checks verify installation and synthetic CSV/shape behavior, not DICOM decoding, training, CPU/MPS numerical equivalence, model quality, throughput, or offline Kaggle inference. No real data or weights were downloaded. Earlier Kaggle/Colab CLI help checks established installed commands only; authentication remains unverified.

## Latest session handoff

- **Completed:** ENV-01 on `codex/ml-environment`: created `.venv`, locked dependencies, pinned Python/build backend, documented repeatable setup, and passed CPU/MPS forward checks plus all five unit tests. Runtime code unchanged.
- **Not performed:** competition access checks, terms acceptance, metadata/image acquisition, training, CUDA execution, or remote publication. No jobs remain running.
- **Next action:** DATA-01: read-only verification of official rules/metric/deadlines and existing authenticated competition access; obtain only authorized small metadata and sample-submission files, then audit aggregate schema and identities. If terms acceptance is required, stop at that boundary for user authorization. Keep images/reports local and immutable.
- **Suggested session prompt:** “Read STATUS.md and AGENTS.md. Start DATA-01 with read-only competition access and official-rule verification. Audit authorized metadata and sample submission if accessible; do not accept terms, download image archives, or launch compute.”
