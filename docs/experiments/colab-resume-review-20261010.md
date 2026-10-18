# Colab resume persistence review — October 10, 2026

Attempt2 `knee-coat384b` lost its assignment after logged epoch11. Only the previously verified epoch3 best-model snapshot survived locally. Fable preserved that failure and launched a new fixed fold0 run as `knee-c384-f0-a1`, supervised by loopPID32148 and driverPID32151. No parent relaunch occurred.

The new VM-side publisher wrote a full resume payload after epoch0. Parent API checks at01:37UTC confirm private dataset `ozgung/knee-colab-resume-c384-fold0` is **ready**, with `raptor_resume_fold0.pt` of1,464,064,119bytes, plus training log and summary. This is stronger evidence than a CLI upload acknowledgment. The remote payload has not been independently downloaded, hash-matched or used in a restore by the parent. Fable reports local uninterrupted-versus-resumed equivalence tests; those are coworker-reported, not a new parent verification.

## Review findings for Fable

- Bind resume state to fold, source/configuration, label/split/exclusion hashes and payload SHA, and validate epoch/history before continuing. Current saved state and restore checks bind required state keys but do not carry those experiment identities. The fixed current loop reduces mismatch risk; it does not establish general resume compatibility.
- `PUBLISHED` currently means the CLI accepted create/version, not that processing or restore validation finished. Record a verified recovery epoch separately. The initial dataset has now passed readiness inspection, but no byte/restore verification.
- Version publication uses `-d`, which the installed CLI maps to **delete old versions**. Retain a verified prior recovery point until the replacement is ready and checked; do not describe deletion as merely replacing the file in place.
- Explicitly requested local resume can fall back to epoch0 if its file is absent. Dataset restore already rejects missing/unreadable files, but entry/trainer should also fail closed on a missing requested state.
- Retry ownership needs an exclusive loop lock and confirmation that the prior assignment is gone. Current cleanup only warns on stop failure, and the loop retries rc75. Use repeated endpoint-specific absence evidence for the watchdog. Starting the wrapper again also resets attempt1, so do not restart it blindly.
- Publisher errors are logged but do not propagate to training. Expose backup health and last verified epoch instead of assuming protection because the publisher process started.

These are review findings, not evidence that the current restored state is wrong or a reason to stop the healthy fit. Fable owns the active untracked scripts and remediation; parent did not overwrite them, delete dataset versions, upload credentials or claim recovery qualification. Final model promotion still needs source/split/selection evidence and independent image-only inference checks.

Evidence: `scripts/colab_tracka_persist.py`, `scripts/colab_tracka_prepare.py`, `scripts/colab_tracka.sh`, the current patched trainer and the existing scratchpad loop. Exact aggregate live receipt: `outputs/analysis/reconcile-20261010-0135/receipt.json`. Script changes remain untracked/coauthor-owned; no integration or344-test claim by parent. Kaggle round1 rehearsal still waits for actual two-session capacity; its frozen artifacts are unchanged.
