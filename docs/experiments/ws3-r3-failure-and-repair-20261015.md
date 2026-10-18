# WS3 r3 failure, saved P3 diagnostic and CPU repair

Recorded October 15, 2026, 14:03 UTC. Claude owns execution and collection; Codex owns this targeted acceptance. **The rehearsal failed; no v3 entry is qualified. GPU work is suspended pending A′ submission 57229227.** The earlier [prospective contract](ws3-prefetch-mirror-qualification-20261015.md) remains historical evidence, not an instruction to relaunch.

## Terminal evidence

`ozgung/ws3-prefetch-plain-qual-20261015-r3`, version 1, pushed 12:54:11 UTC, ended ERROR around 13:39. Claude collected 28 files into `outputs/kaggle/ws3-prefetch-plain-qual-20261015-r3-result/`; result root is `ws2-a-prime/`. Root elapsed was 2,624.987 seconds. Base passed in 1,691.939 seconds; serial control completed, but native224 never ran.

The comparator imported `knee.public_r7_prefetch_candidate`, a checkout-only alias. The sealed packet deliberately maps that source to `knee.public_r7_recovery_entry`. The missing alias caused `ModuleNotFoundError` before comparison or serial phase summary persistence. Serial worker receipts and NPZ files survived; the measured combined serial phase RSS did not. Summing worker peaks cannot reconstruct that measurement.

Preserved build: `outputs/kaggle/ws3-prefetch-plain-qual-20261015-r3/`, notebook SHA `e785a4547548a9ba6a548c301ab69433d9b7c0d0d6f1ad1630864c03aa63c1d6`, payload SHA `04183dc958669decca89d72d1289076763477d86dc553f5a27ac6b7925889249`, MANIFEST SHA `8214ca487041d77ec9870c072652d88712c94210c0fbf3a9513eb6da3f6d539c`. Original preparation was verified before repairs. These immutable bytes remain failed historical evidence; current repaired sources need a fresh build if rescue becomes necessary.

## Saved worker comparison

[Partial diagnostic](../../outputs/analysis/r224-integration-20261015/ws3-r3-partial-p3.json): **60/63 checks pass**, with all 39×8 input fingerprints exact and maximum model/final/ledger probability delta zero. State, reload/sentinel, compiled/source, producer/teardown and worker RSS checks pass. Prefetch combined peak RSS was 17,528,127,488 bytes, below 22e9. The missing serial combined peak remains unknown.

| Board | Prefetch evaluation wall | Serial evaluation wall | Paired ratio |
| --- | ---: | ---: | ---: |
| 0, slower prefetch board | 378.954332 s | 426.876256 s | 0.8877381362 |
| 1 | 359.451780 s | 398.798890 s | 0.9013359578 |

Evaluation-wall intervals equal the recorded production sums. Cold compilation outside those intervals does not change this preregistered metric. **Mirror is not admitted under ≤0.85.** No warming or ordering change will rescue this observed result. The base-only conservative projection was 23,259.978 seconds; it excludes the unexecuted native224 arm and cannot qualify the full recipe.

The other two failed checks are exact `native_scratch_bytes` equality on both boards. Prefetch totals differ by −119 to −301 bytes per study on approximately 30–235 MB caches. The counter includes all serialized cache files. Inspection of the pinned `training/native_cache.py` found absolute target paths and elapsed-time metadata; this supports a diagnostic-comparability explanation, although deleted cache files prevent exact byte accounting. It does not establish a new PASS or prove full cache-file equality.

Claude and Codex accept that classification prospectively: **only if rescue is needed**, implement per-file content hashes for cached arrays and retain raw byte totals as informational. Record and test the precise prospective policy before another GPU run. This policy is not implemented yet; r3 remains failed. [Frozen diagnostic script](../../outputs/analysis/r224-integration-20261015/compare_failed_ws3_r3.py) binds the original comparator hash and sealed manifest, preserving reproducibility after the import repair.

## CPU repair and acceptance

The comparator now imports the canonical mapped module. The coordinator saves serial worker/phase measurements in `serial-control-status.json` and the root receipt before importing/calling comparison, then saves completed comparison evidence before acceptance gates can fail. Candidate `09ad6adb10bb1346a8751b9dfe33c3d732ce1616692a1e42179abd5fa43419de`, model/reader/state logic, thresholds, projection and archived v7/v2 code remain unchanged.

- Installed ML focused checks: **38 tests PASS, zero skips**. [Log](../../outputs/analysis/r224-integration-20261015/ws3-repair-focused-tests.log).
- Bare Python WS3 checks: **14 tests PASS, six expected optional-dependency skips**. [Log](../../outputs/analysis/r224-integration-20261015/ws3-repair-dependency-free-tests.log).
- New isolated unpack regression fails with the old import and passes after repair. It disables checkout/editable-install fallback with `-I -S`.
- Actual 244-source packet unpack and canonical comparator import pass in isolation. [Receipt](../../outputs/analysis/r224-integration-20261015/ws3-repair-isolated-packet-import.json). Synthetic empty comparison exercises imports only; no GPU or real-reader inference.
- Dependency-free smoke PASS on synthetic CPU data, without a torch forward or training.
- A synthetic comparator exception confirms serial measurements survive in both receipts and native224 is not launched.

No full-suite repeat, fresh r4 preparation, GPU run, collection or submission was performed for these narrow repairs. Earlier broad-suite limitations remain in the prospective contract. No competitive performance follows from these checks.

## Conditional next action

Official submission 57229227 remained **PENDING at 13:59:06 UTC**, with no score. [Snapshot](../../outputs/analysis/r224-integration-20261015/aprime-after-ws3-r3-failure.json).

- If A′ is COMPLETE: close v3, with no r4 re-push; retain A and existing final-selection rules.
- If A′ times out or ERRORs: finish/test the prospective cache-array hash policy, seal a fresh plain r4, reconcile actual quota/jobs and have Claude run one rehearsal. Require complete saved acceptance and conservative projection ≤8.3 hours before any separately authorized submission.

The existing heartbeat remains active; this suspension concerns GPU work only. Do not duplicate Claude's monitoring, collection or execution.
