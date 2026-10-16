# V2-all backup candidate acceptance

October16,11:32 UTC. Claude reports s6v2d lost between09:30 and09:48; the private checkpoint dataset returned403, so recovery remains unavailable/unverified. The VM-loss cause and historical backup execution are not established. Claude already owns diagnosis and one replacement launch; Codex has not connected, collected remotely or launched anything.

The **new per-epoch** shell candidate, `vm_bk_v2all_ckpt.sh` SHA256 **8dfd2c7bd82e8840e3c3a3cbbe34871da65db8b7b3cd8cbe7c0341a558e4a4d8**, was saved after the loss. These findings concern its future recovery behavior, not proof of what caused the original loss or missing backup. The candidate polls every60s and waits20s after checkpoint mtime, so it addresses first-hour delay in principle.

[Hash-bound four-case execution receipt](../../outputs/analysis/s6v2-backup-acceptance-20261016/acceptance.json), [frozen candidate](../../outputs/analysis/s6v2-backup-acceptance-20261016/candidate.sh), [replay](../../outputs/analysis/s6v2-backup-acceptance-20261016/audit_backup.py). The real shell was executed with its absolute content/Kaggle paths redirected to temporary local fixtures and fake date/stat/sleep/Python/Kaggle commands. Checkpoints are tiny synthetic payloads; no cloud, torch deserialization, model or live job was exercised.

| Case | Observed publication calls | Acceptance |
| --- | --- | --- |
| Successful initial publication, unchanged epoch | create once | PASS |
| Initial create fails, unchanged epoch | create once, no retry | FAIL |
| Initial create fails, next epoch appears | create then version | FAIL: remote dataset not confirmed created |
| Required membership metadata missing | create still runs | FAIL |

The script advances `last` even on failed publication, suppressing retry of that checkpoint. Local dataset-metadata.json also survives a failed create and selects version next time; it proves neither remote existence nor recoverability. Required metadata copy failure is ignored, allowing an incomplete upload attempt.

Claude should repair these transitions in the owned workflow: keep failed epochs pending, advance the checkpoint marker only after successful remote acknowledgement, distinguish a verified existing remote dataset from a local metadata file, and require fresh matching config/membership/hash before publication. A restarted VM needs explicit reconciliation of any existing remote dataset, not blind create/version guessing. Preserve first failure and retry receipts. Confirm an actual retrievable checkpoint identity/hash/epoch in the new live run before describing recovery as available; a successful local shell test is insufficient.

Retest the same cases on the corrected source hash, plus verified-existing-dataset versioning and repeated network failure. No training stop or recipe change is requested. This candidate fails backup acceptance; it does not invalidate already accepted model/input mechanics or establish a new training result.
