# Project status and session handoff

Last updated: **2026-10-07 (Europe/Istanbul)**

Read this file at session start. Update it after meaningful changes and before handing off. Working rules are in [AGENTS.md](AGENTS.md); setup and the modeling outline are in [README.md](README.md).

## Current objective

Build a reproducible, image-only knee MRI baseline and an offline competition inference path. The next milestone is a verified metadata/schema audit and leakage-aware validation design, before real model training.

## Verified snapshot

| Area | State / evidence |
| --- | --- |
| Project | Local scaffold at `/Users/ozgun/Dev/rsna-knee-abnormality-detection` |
| Version control | Local Git initialized. Initial scaffold commit `0ae2128` on `main`; current task branch `codex/colab-connection` starts from `b1d02c1`. No remote configured. |
| Utilities | `knee/audit.py`, `knee/device.py`, and `knee/submission.py` exist; five unit tests pass. |
| Smoke check | Dependency-free smoke passes; explicit PyTorch forward checks pass on local CPU/host MPS and Colab T4 CUDA. MPS fallback unset. Synthetic data only. |
| Model | `knee/model.py` contains an untrained fixed-bag toy encoder. Its tiny forward pass is verified on CPU/MPS and Colab CUDA; no training or real-data validation. |
| Configuration | `configs/local.toml`: 12 targets, seed 42, automatic device selection, initial 2 GiB cache budget. Several settings are proposed for future training, not an implemented training pipeline. |
| Data and results | No dataset, audit artifacts, folds, trained checkpoints, or measured real-data scores are present in this project. |
| Environment | Isolated `.venv`: Python 3.12.9, uv 0.9.13, torch 2.14.1, numpy 2.5.3, pandas 3.0.6, pydicom 3.0.2, scikit-learn 1.9.1. Exact runtime dependencies/hashes in `uv.lock`; Python and build backend pinned. Locked sync/check and package compatibility pass. |
| Remote compute tools | `kaggle` and `colab` are installed and their `--help` commands succeed. Kaggle exposes competition/kernel operations; Colab exposes session creation, execution, and file transfer. Read-only Kaggle leaderboard, notebook list/source, discussion list/comments, and public model search/card access succeeded on 2026-10-07 with network-enabled CLI calls. Competition data entitlement and terms acceptance remain unverified. Colab CLI authentication and T4 CUDA execution passed; browser MCP connection returned false. No active Colab sessions remain. |
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
| COLAB-01 | done | Verify Colab connection and CUDA execution | Authenticated CLI access, 5 remote tests and T4 CUDA forward check passed; environment evidence saved; no active runtimes after teardown. |
| ENV-01 | done | Establish reproducible ML environment | Isolated environment installed from `uv.lock`; pinned Python/build backend; explicit CPU/MPS forward checks passed. Validated on local macOS arm64 only. |
| VAL-01 | blocked | Define folds and prevalence baseline | After DATA-01: persist reproducible grouped split, verify disjoint identities, define missing-label/AUC policy, and report baseline support/metrics. |
| PIPE-01 | blocked | Implement bounded DICOM-to-study loading | After DATA-01 and ENV-01: representative studies decode; ordering/orientation and masks tested; cache/time/memory measured within budget. |
| TRAIN-01 | blocked | Train and evaluate first real baseline | After VAL-01 and PIPE-01: reproducible run, held-out per-target/macro metrics versus prevalence, checkpoint, config, and experiment summary. |
| SUBMIT-01 | blocked | Rehearse offline Kaggle inference | After TRAIN-01: image-only run without internet, exact schema/IDs, finite predictions, measured resource headroom under verified limits. |

## Blockers and unresolved choices

- Competition access/terms acceptance and metadata availability have not been verified. The prior setup did not accept terms or download data.
- Official overview/data text was readable through the web search index on 2026-10-07: macro AUC over 12 targets, images-only test input, about 1,300 test studies, 9-hour offline notebook limit, entry/merge October 15 and final October 22 at 23:59 UTC. Direct page fetch remained unreadable and CLI pages parsing failed; recheck the live timeline before scheduling. Exact training counts still require CSV audit.
- Exact patient identity, label schema, report languages, missingness, and usable validation size depend on DATA-01. Do not invent these values.
- User authorized use of the existing **350 Colab compute credits** on 2026-10-07, with preference to consume them before expiry. Balance and expiry are not independently verified; expiry requested. No authorization to buy more credits. Plan bounded, useful runs within that allocation; measured training throughput is still unknown.
- No running jobs, Colab runtimes, or resumable checkpoints. COLAB-01 completed and its T4 session was released; `colab sessions` confirmed none active.
- The sandbox hides MPS and causes uv 0.9.13 macOS configuration lookup failures; host execution passed. CUDA is unavailable locally; the Colab T4 forward check passed. Other operations and real-data throughput remain unverified.
- Public Kaggle CLI reads and Colab T4 execution are verified; competition data entitlement, actual remaining Colab balance/expiry, and billed credits for the test remain unverified. Colab image lacks pydicom; install/pin it before DICOM work. Preserve its working CUDA torch instead of applying the Mac environment unchanged.

## Working decisions

| Date | Decision | Reason / revisit trigger |
| --- | --- | --- |
| 2026-10-07 | Keep management in the repository: STATUS for current state and tentative planning, AGENTS for rules, README for setup. No separate roadmap. | Keep one source for progress and planning; revise priorities as evidence and competition constraints change. |
| 2026-10-07 | Established local `main` snapshot and task branches; keep commits local. | SETUP-01 completed; no remote configured. |
| 2026-10-07 | Use existing uv with a checked-in universal lock, Python 3.12.9 pin, and setuptools 84.0.0 pin. Next priority is DATA-01. | ENV-01 passed on macOS arm64. Other lockfile platforms are untested; preserve/test CUDA environments separately. Metadata and rules are now the blocking inputs. |
| 2026-10-07 | Prioritize existing 350 Colab credits for useful bounded project runs before expiry; user explicitly authorized their use. | COLAB-01 moved ahead of DATA-01 to verify access. Expiry and actual remaining balance still needed; no new purchases authorized. Release idle runtimes and record per-run budgets/results. |
| 2026-10-07 | Use one active implementation task with a concrete acceptance check. | Keeps sessions reviewable and makes the next action unambiguous. |
| 2026-10-07 | Start with existing tools; no custom skills/plugins installed. | Extract narrow skills after workflows stabilize; add connectors when external tracking is needed. |
| 2026-10-07 | Prioritize schema/folds and a simple baseline before complex models or weak-label expansion. | Useful model comparisons depend on reliable validation. |
| 2026-10-07 | Recommend Astra medium effort normally, high for complex analysis. | A starting recommendation, not a benchmark result or a changed app/model setting. |

Add detailed decision records under `docs/decisions/` only when alternatives or consequences no longer fit in this table. Link them here.

## Latest verification

Executed **2026-10-07** for COLAB-01, source revision `b1d02c1` (runtime code unchanged):

```text
colab sessions
PASS: authenticated server access; no sessions before test

colab run --gpu T4 --session knee-connect-20261007 --timeout 120 outputs/colab/20261007-connection-check.py
PASS: exit 0; 5 unit tests and knee.smoke --torch --device cuda

colab sessions
PASS: no active sessions after automatic teardown
```

- **Runtime:** Tesla T4, 15,637,086,208 GPU memory bytes; Python 3.13.15, torch 2.11.0+cu130, CUDA runtime 13.0, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.6.1. pydicom absent. Existing runtime packages were preserved; no installs performed.
- **Bound:** 120-second code-execution timeout. Checks took 11.39 seconds, excluding allocation/startup and initial imports; total billed credits unverified. Synthetic fixtures only; no training or real-data transfer.
- **Artifacts (ignored):** [runner](outputs/colab/20261007-connection-check.py), [log](outputs/colab/20261007-connection-check.log), [complete package/environment record](outputs/colab/20261007-environment.json). Session `knee-connect-20261007` is **terminated**, not resumable. Rerunning the command provisions a fresh runtime. Emergency stop if a future retry is interrupted: `colab stop -s knee-connect-20261007`; verify with `colab sessions`.
- **Access limitations:** browser MCP connection returned false; CLI 0.6.0 authenticated and executed successfully. Sandbox CLI invocation initially failed writing its own log; host execution succeeded. No credential contents were printed or transferred to project files.

Earlier ENV-01 checks passed: locked local sync and offline no-change audit (22 packages), package compatibility, all 5 unit tests, dependency-free smoke, explicit CPU/MPS forward checks. MPS and uv required host access because the sandbox hid GPU access/blocked macOS configuration lookup. Local environment evidence: [env-01.json](outputs/environment/env-01.json). Reproduction: [README.md](README.md#reproducible-local-ml-environment).

Local Markdown links and Git whitespace checks passed after this documentation update. These checks verify connection/installation and synthetic CSV/shape behavior, not DICOM decoding, training, CPU/MPS/CUDA numerical equivalence, model quality, real-data throughput, or offline Kaggle inference.

## Latest session handoff

- **Completed:** COLAB-01 on `codex/colab-connection`: authenticated CLI access, T4 allocation, 5 remote unit tests, CUDA toy forward check, evidence retrieval, and verified teardown. Recorded the user's authorization to consume the existing 350-credit allocation. Runtime code unchanged.
- **Unknown:** credit expiry (asked user), actual remaining balance and billed test cost. No new credit purchases, competition data transfer, training, or terms acceptance occurred.
- **Next action:** DATA-01: verify official rules/deadlines and existing data entitlement; audit authorized small metadata and sample submission, then grouped folds and a prevalence baseline. Prioritize bounded Colab training once these prerequisites exist, using the authorized credits and recording budget/stop conditions per run. If terms acceptance is needed, stop for user authorization.
- **Suggested session prompt:** “Read STATUS.md and AGENTS.md. Start DATA-01 with read-only competition access and official-rule verification, then audit authorized metadata. Use the existing authorized Colab credits for useful bounded compute once validation/data prerequisites are met; record expiry when available.”
