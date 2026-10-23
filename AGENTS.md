# Project instructions

## Session handoff

- Read `STATUS.md` first: current snapshot, queue, and latest checkpoint; read other sections only when relevant. Then inspect the relevant README sections and code. Reconcile stale claims against files and checks before acting.
- `STATUS.md` is the primary cross-session handoff and current task queue. Keep changing state there; keep setup in `README.md` and durable working rules here.
- The competition closed on 2026-10-22. The campaign's shared Claude/Codex message log and the final STATUS snapshot are archived under `docs/archive/`; read them only for history. New work starts from `STATUS.md` and the postmortem in `docs/postmortem/`.
- Treat the STATUS plan as tentative. After meaningful results or changes to data, rules, time, or compute, reorder, revise, or drop tasks and record the reason in its working decisions. Verify deadlines before calendar planning and preserve time for offline rehearsals.
- Follow the user's current request. For an open-ended request to continue, pick the first unblocked ready task in `STATUS.md`. Keep one implementation task active at a time.
- Before substantial work, state the intended outcome and acceptance check. Continue through implementation, relevant verification, and handoff updates within the authorized scope.
- Update `STATUS.md` at meaningful checkpoints and before ending any session that changes code, plans, environment, or project findings. Record the date, changes, exact checks/results, unresolved issues, and a concrete next action. A discussion with no project impact needs no update.
- Keep the handoff compact: current snapshot, ready queue, blockers, decisions, verification, and latest note. Move completed history and detailed receipts to linked artifacts; preserve active-job recovery details. Use normal readable prose, not concatenated words or repeated hashes.
- Record long-running jobs with environment, job/session ID, command, log and checkpoint locations, and resume/stop instructions. Distinguish running, completed, failed, and unverified work.

## Token efficiency

- Complete the authorized task with the smallest sufficient context. Do not skip correctness checks, scientific safeguards, or required verification to save tokens.
- Search with scoped `rg` queries, then read relevant ranges. Avoid whole-repository dumps and rereading unchanged files or instructions already in context. Expand reads when dependencies or uncertainty require it.
- Keep tool output bounded: save verbose logs under ignored `outputs/`, return summaries, failures, and evidence paths. For structured results, extract needed fields; do not print entire notebooks, datasets, or logs.
- Batch independent reads/checks. Reuse verified evidence for unchanged inputs; repeat checks only after relevant changes, failures, or new uncertainty. Back off unchanged job polling and prefer completion notifications when available.
- Keep updates and final answers brief: outcome, checks, material limits, next action. Avoid repeating plans, instructions, or tool output. Simple edits do not need elaborate plans or separate review agents.
- Default to one agent. When parallel work is explicitly requested, delegate only independent substantial tasks with concise context and result summaries; avoid duplicate investigations and routine review chains.
- Load only relevant skills and tools. Keep durable instructions here, setup in README, current state in STATUS, and detailed evidence in linked artifacts; avoid duplicating them across files.

## Project and scope

This is an offline-first Python scaffold for study-level knee MRI abnormality prediction. The existing model is a toy shape-check architecture. Do not describe it as a trained or validated baseline.

- Prefer small, reviewable changes with explicit acceptance criteria. Separate infrastructure fixes from model experiments when practical.
- Use the existing plain Python modules and TOML configuration. Add dependencies or abstractions when a concrete task needs them.
- Preserve the dependency-free audit, submission, and default smoke paths. Import ML dependencies only where needed.
- Inspect Git state before editing when Git exists; preserve unrelated changes. If no Git repository exists, report that limitation rather than inventing a branch or commit.
- Use task branches and reviewable diffs once Git is established. Use isolated worktrees for explicitly requested concurrent agent work; do not add routine multi-agent orchestration.
- Resolve reversible implementation choices independently. Ask only for missing information or consequential choices that block authorized work. Do not create recurring approval checkpoints.

## Validation and data invariants

- Read exact target names, target order, and ID schema from the real competition files. Treat counts, deadlines, rules, and runtime limits in prior notes as unverified until checked against current official sources.
- Split at patient level when identity is available; keep all related studies, series, slices, and derived labels in the same fold. Document limitations if only study-level grouping is possible.
- Fit preprocessing statistics and report-label mappings on training folds only. Keep validation labels independent of weak-label generation. Missing or unmentioned findings are not automatically negatives.
- Reports may supervise training; inference must work from images alone. Check applicable rules and data permissions before external processing of reports or images.
- Keep raw data immutable and excluded from Git. Do not commit credentials, report text, patient identifiers, image data, caches, or weights. Use synthetic fixtures in tests and aggregate summaries in handoffs.
- Test DICOM ordering/orientation, intensity handling, padding and missing-plane masks before trusting real-data pooling.
- Evaluate a prevalence baseline before judging learned models. Report per-target AUC, support/class counts, undefined targets, and the explicit macro aggregation policy. Do not infer generalization from smoke checks or training loss.
- Preserve the official submission schema/order and require finite probabilities in `[0, 1]` with exact study-ID coverage.

## Experiments and resources

- Before training, state the hypothesis, baseline, split, primary metric, bounded sample, compute budget, and stop condition. Change one major factor per comparison where practical.
- Record each run's ID, code revision (or explicitly unversioned state), resolved config, seed, data/split fingerprint, preprocessing version, environment/package versions, device, metrics, runtime, and artifact paths. Record failed runs too.
- Store large/generated artifacts in ignored `outputs/`, `data/`, or `checkpoints/`. Add a concise, non-sensitive experiment summary under `docs/experiments/` when the first real experiment exists; link it from `STATUS.md`.
- Use the initial 2 GiB local cache budget in `configs/local.toml`; inspect free disk before expansion. Full-corpus downloads are outside the local starter workflow.
- Start with bounded CPU/MPS checks locally; use CUDA/Kaggle for scale when authorized. Explicit unavailable devices should fail clearly. Do not silently enable CPU fallback or assume MPS/CUDA equivalence.
- The user has removed self-imposed experiment time budgets. Record runtime estimates and resource use, but do not abort authorized training because an estimate exceeds an assistant-chosen cap or GPU-hour reserve. Respect actual platform/account limits and explicit user constraints; checkpoint before platform termination. Do not provision paid compute, accept competition terms, or submit entries without user authorization.
- Before submission, rehearse image-only inference offline with attached code/dependencies/weights; measure time, memory, and disk against verified limits.

## Checks and definition of done

From the project root, the current dependency-free checks are:

```bash
python3.12 -m unittest discover -s tests -v
python3.12 -m knee.smoke
```

When an ML environment exists, use its interpreter. For model/device changes, also run the appropriate installed-backend check, e.g. `python -m knee.smoke --torch --device mps`. Never claim a backend passed if it was unavailable or not exercised.

- Add meaningful regression tests for changed behavior, especially leakage, masks, schemas, device selection, and submission integrity. Documentation-only edits need link/content checks, not new tests.
- Done means acceptance criteria met, relevant checks reported honestly, artifacts linked, and `STATUS.md` updated. Distinguish implemented, tested on synthetic data, tested on real data, and competitively evaluated.
- Report remaining risks and skipped checks with reasons. Do not claim a full pipeline works from unit-test success.

## Astra, skills, and plugins

- Give Astra an outcome, constraints, acceptance criteria, and relevant file paths. Keep instructions concise and project-specific; avoid scripted reasoning or repeating generic coding advice.
- User model-cost preference (October 12, supersedes October 9): Codex is on Plus using Sol high. Claude owns most strategy, research, substantial implementation, difficult debugging and result analysis. Codex owns concise coordination, targeted acceptance checks, integration, parent commits and compact state updates. Delegate token-intensive work to the continuing Claude collaborator through the shared handoff before starting additional Codex agents. Preserve existing scientific safeguards and explicit job ownership; avoid duplicate execution or repeated reviews. Use Astra only when explicitly needed for a consequential unresolved issue, not as the default planning lead.
- For new routine subagents, explicitly select the cheaper model with `fork_turns="none"` and a concise task/evidence handoff; do not inherit a full Astra conversation by default or reuse an Astra agent for routine follow-ups. Avoid agents for trivial edits, duplicate audits, repeated unchanged polling and routine review chains. Reuse verified evidence and keep outputs compact. These delegation choices do not change the parent model selected in Codex settings.
- No custom skill or extra plugin is required for the current scaffold. Use relevant installed skills when their workflow is needed, not for every session.
- Extract a narrow repo skill only after a workflow is repeatable, such as experiment recording or offline submission rehearsal. Keep executable validation in tested Python code, not only in skill prose.
- Add connectors only for an actual external workflow. GitHub issues/PRs can become useful after a remote exists; keep current state in `STATUS.md` and link external tasks rather than maintaining duplicate boards.
