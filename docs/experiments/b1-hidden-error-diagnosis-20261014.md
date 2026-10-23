# B1 hidden-run "system error" diagnosis (2026-10-14)

Read-only diagnosis of submission **57157385** (B1, kernel `ozgung/knee-r7-only-b1-entry-20261013` v1, R7 alone). Kaggle returned ERROR with error_description "A system error. Please try resubmitting to resolve the error and contact Kaggle Support if it persists." and total_bytes 0, about 3.6 h after submission at 22:11 UTC. Nothing was executed or pushed for this note. Code paths cite the uncommitted worktree `.claude/worktrees/agent-a56a946aa62d1317f`. Pinned heliosli sources are under `outputs/r7work/heliosli0957/sources/` in that worktree.

## Two facts that constrain the ranking

1. **Every deliberate fatal gate surfaces as an ordinary Python exception.** The worker records `failure-private.json` and re-raises (`public_r7_recovery_entry.py:435-439`). The parent sees the non-zero exit code, sends SIGTERM to the sibling worker and raises `RuntimeError('Worker failure; no partial merge')` (`:455-465`). `run` then moves any CSV aside to `rejected-submission.csv` and re-raises (`public_r7_b1_entry.py:97-100`). The notebook cell saves `setup-failure-private.json` and raises again (`notebook.ipynb`, cell 0, final `except BaseException`). No code path ends cleanly without a `submission.csv`.
   - Two earlier hidden failures in this competition were raised Python exceptions. Kaggle labelled both "Notebook Threw Exception" / "Your notebook hit an unhandled error while rerunning your code" (57083497 in `docs/experiments/s6-dual-rehearsal-result-20261011.md:26`; 57086107 in `docs/archive/handoff/claude-codex-log-20261009-20261012.md:959`). Both entries used the same notebook wrapper as B1, and the 0.954 build has the same `out` path and the same `alarm` and `except` structure.
   - B1 got a **different** label. A raised gate is therefore not the most likely explanation. The label points to the kernel or container dying, or to a platform fault. Kaggle's exact mapping from failure to label is not documented, so this inference is unverified.
2. **The timing points to mid-run.** The rehearsal projects base 16,095 s (4.47 h) and conservative 22,826 s (6.34 h) (`outputs/kaggle/r7-b1-rehearsal-20261013-result-v1/s6-recovery-entry/receipt.json`). The 0.954 entry took about 7.3 h from submission to observation against a 5.97 h base (×1.22, queue included). At that ratio B1 would finish at about 5.5 h. The 3.6 h figure includes queue time, so the actual runtime was at most 3.6 h, roughly 60–80 % through production. End-of-run gates would normally fire at 4.5 h or later.

What is shared with the successful 0.954 entry (57096002), checked against the build receipts:
- All 232 source files other than `compact_infer.py`, `compact_train.py`, `probe_dual.py` and the four new `knee/public_r7_*` modules are byte-identical. That covers native_cache, dicom_geometry, px_cache, the anchor bank, the guards, staging and the parent launch code.
- `CStudies` `_setup`, `_prepare` and `_native_and_geometry` are AST-identical. The only change to `__getitem__` is the added `fluid_d2` call.

So the 0.954 run has already put the identical hidden studies through the identical decode, render and guard chain.

## Ranked termination paths

| # | Path (evidence) | When it would fire (B1 wall time) | Kaggle label we would expect | Likelihood for a "system error" at about 3.6 h | Shared by A / B2 |
|---|---|---|---|---|---|
| 1 | **Platform fault**: node or container loss, preemption, or a backend fault. A SIGTERM would reach `PlatformInterruption` (`public_adaptation_fit.py:145-158`); a SIGKILL or VM loss leaves no trace. | Any time | "System error" (that is the message's literal meaning) | **Highest.** Matches both the label and the mid-run timing, and nothing in our code produces this label. | Independent per run. A and B2 succeeding would support it. |
| 2 | **Host-RAM exhaustion that kills the kernel or container.** The R7 phase starts much higher than S6:<br>• Combined descendant RSS peaked at **18.2 GB** in the B1 rehearsal (not 15.9; A v7 17.0, B2 17.7, B1 visible entry 16.6). The 0.954 S6 rehearsal peaked at 8.74 GB.<br>• Each worker's ru_maxrss was 7.07–7.2 GB, flat over 20 studies. The rest of the total is the parent plus inductor compile-worker pools that live as long as each worker.<br>• Headroom is about 11–13 GB on a 29–31 GB machine.<br>• Triggers: an outlier study on both boards at once (per-study host arrays scale with windows, about 0.44 MB per window for images plus the rendered rows; the hidden maximum is unknown, 1,452 among the 870 dev studies) or a slow leak. ru_maxrss cannot show a leak below the 7.07 GB sentinel-phase plateau. | At the outlier study, or late in the run for a leak (a leak hitting the ceiling at about 75 % of the run fits 3.6 h) | A worker OOM-kill gives exit −9, so the parent raises the "unhandled error" label. Only a kill of the kernel or container would give the "system error" label (unverified). | **Moderate-low.** This is the main R7-specific scaling difference from 0.954. | Yes. The R7 phase runs alone in A and B2 too, because arms are sequential (`public_s6_recovery_entry.py:641-643`). It would fire later there, after the old arm. |
| 3 | **A data-dependent fatal gate in new R7 code, on some hidden study**:<br>• `fluid_boundary` turns any unknown exception from `fluid_d2` into `FatalSourceError` (`public_r7_recovery_entry.py:121-135`). One reachable example: `_fluid_disk` with pixel size 0 or NaN raises OverflowError or ValueError from `int(round(inf/nan))` (`compact_infer.py:84-96`). Other examples are cv2 errors and `int(NaN)` on the fluid flag (`:123`).<br>• `NONFINITE_FP16_NO_RESCUE` (`:382`), with no fp32 rescue, unlike heliosli `compact_infer.py:982-990`.<br>• A non-finite Baker's state logit is allowed in entry mode, but its NaN `features` go into the row (`public_r7_stackers.py:177-179`). `save(..., allow_nan=False)` (`public_s6_qualification.py:48-52`) then raises ValueError on the next receipt write (`:431`).<br>• `PORTED_FLUID_EQUALS_SOURCE` (`:388`) and `UNKNOWN_FLUID_STATUS` (`:116`). | At the offending study, anywhere from about 0.2 h to the end of production | "Unhandled error" (precedent) | **Low-moderate.** The timing fits but the label does not. It ranks above #4–#9 only because the label mapping is unverified. | **Yes, deterministic.** Same hidden data and the same modulo-2 assignment (`public_s6_recovery_entry.py:51-53`), so A and B2 would fail as well. |
| 4 | **Worker native crash** (a segfault in cv2 morphology, which only R7 uses, in CUDA or in the decoders). | At the offending study | The parent catches the non-zero exit code and raises (`:459-463`), so "unhandled error" | Low | Yes if it is data-dependent |
| 5 | **GPU OOM with chunk 192** (S6 used 16, at `public_s6_recovery_entry.py:446`). Peak allocated was 6.25 GB, set during the compile check and flat afterwards. Reserved peaked at 11.36 GB of about 15 GB. The part that grows with study size is about 0.74 MB per window (uint8 image plus the masked copy), so OOM would need a study of roughly 10,000 windows or more. | At a giant study | `torch.OutOfMemoryError` is a RuntimeError, so "unhandled error" | Very low | Yes |
| 6 | **Union recovery cap.** The per-board check is at `:426`; the final check is `UNION_RECOVERY_CAP` (`public_r7_b1_entry.py:38-39`). The R7 failure set equals the S6 set (identical preprocessing), plus `fluid_d2` data-local classes that cannot be reached on rendered arrays. The 0.954 run kept the old ∪ S6 union within the same 3 % cap on the same hidden data. | Per board: after 40 failures on one board. Final: end of run, 4.5 h or later | "Unhandled error" | Very low | Yes |
| 7 | **End-of-run gates**: merge (`public_s6_recovery_entry.py:143-163`), `recover` schema and probability checks, `write_csv`, `MAX_SECONDS` (`public_r7_b1_entry.py:89,94`). These are the only B1-specific code. | 4.5 h or later | "Unhandled error" | Very low (timing) | Merge and CSV yes; B1's single-arm `recover` no |
| 8 | **Compile**: fallback or a failed check is fatal through `require_compiled` (`:160-164,372`). This is decided on the first sentinel, a training study, so it is near-deterministic, and it passed in the visible entry and every rehearsal.<br>**Recompilation: verified impossible for the encoder.**<br>• Eval-mode `Net.forward` pads to a multiple of 32 and runs chunks of 192 (`compact_train.py:657-669`), so encoder batches are always one of {32, 64, …, 192}.<br>• All six are warmed through the same forward, dtype, channels_last and inference mode (`public_r7_recovery_entry.py:253-262`).<br>• Only `m.enc` is compiled; the head, whose shapes vary with window count, runs eager.<br>• Even past the limit of 64, dynamo falls back to eager rather than failing. | About 0.1–0.2 h | "Unhandled error" | Very low | Yes |
| 9 | **Disk**: output peaked at 0.46–0.58 GB, with at least 20.35 GB free. Per-study scratch is removed (`compact_infer.py:528`, `public_s6_recovery_runtime.py:227-234`). Ledger and receipt are rewritten in place. Worker stdout is not printed per study. The /tmp inductor cache is written once. The staging tree under `/kaggle/working` is the same as in the 0.954 entry. Nothing grows with study count except about 2.8 KB of JSON per record (about 1.8 MB at the end). | — | — | Very low | Yes |
| 10 | **9 h SIGALRM** (notebook cell 0) and the Kaggle 9 h limit | 9.0 h | "Notebook Timeout" or a TimeoutError exception | Excluded at 3.6 h | Yes |

Other items checked and ruled out as causes:
- The parent's 0.5 s `resource_snapshot` loop: an rglob of about 8k entries, which does not follow staged symlinks.
- Receipt growth, which is O(n) per save.
- Sentinel reloads, which happen only in the first 8 items.

## Comparison with the 0.954 S6 worker

These are the differences that could matter at scale:
1. Host RSS baseline roughly doubles: about 7.1 GB per worker plus compile pools, against about 4.4 GB, for 17–18 GB total against 8.7 GB.
2. Inductor compile and its subprocess pools, which persist for each worker's lifetime.
3. Chunk 192 against 16: GPU working set about 6 GB, but bounded and independent of study size.
4. `fluid_d2`, which adds cv2 morphology and a new exception boundary.
5. Aux state read-out plus stackers, with the JSON NaN path.
6. No fp32 rescue.

These do not differ: preprocessing, caching and cleanup, staging, launch and merge. Nothing accumulates across studies in either worker.

## Cheapest discriminating checks, in order

1. **Wait for A (57157384) and B2 (57157388). This costs nothing.** They run the identical R7 phase on identical data after their old arm.
   - Both score: deterministic data paths (#3–#8) are excluded. #1 is most likely, with #2 still possible but unlikely.
   - Both fail with "unhandled error" at about (old-arm duration + B1's offset): a deterministic R7 gate (#3–#5).
   - Both fail with "system error": a systematic R7-phase resource kill (#2) becomes the lead.
2. **If #2 needs ruling out: a scaled memory profile, about 1 h of 2×T4.**
   - Run the unchanged R7 worker in rehearsal mode on about 100–150 training studies. Include the 20 largest by window count (the 870-dev maximum is 1,452) and put the largest pairs on both boards at once.
   - Every 0.5 s, sample current per-worker **VmRSS** from /proc (ru_maxrss hides leaks), compile-pool RSS, `torch.cuda.memory_reserved`, and free space under /kaggle/working and /tmp. Log per-study deltas.
   - Pass criterion: leak slope × 650 studies plus the largest-study transient stays under (29 GB − 18.2 GB), with margin.
3. **If #3 needs ruling out: a CPU-only replay**, with no GPU and no downloaded code executed. Feed `fluid_d2`-equivalent inputs (pixel size 0 or NaN, NaN fluid flag) through our boundary port, and NaN `stv` through `stackers.apply` plus `save`. This shows which gates fire. It cannot show whether hidden data triggers them; only #1 can.
