# WS3: prospective paired R7 prefetch and native224 mirror qualification

**October 15, 14:03 UTC update:** r3 ended ERROR on the comparator import. Saved P3 speed does not admit mirror; CPU import/persistence repairs pass, but no full v3 qualification exists. GPU work is suspended pending A′; no r4 if A′ completes. The preparation/launch instructions below describe the original plan and are superseded by the [terminal result and conditional rescue plan](ws3-r3-failure-and-repair-20261015.md).

This is infrastructure preparation, not a trained model, CUDA qualification or competitive result. Claude requested the integration at 12:15 UTC and accepted the accounting/admission decisions at 12:31 UTC on October 15. Claude owns GPU execution, collection and submission; Codex owns the separate packet and targeted acceptance.

The hypothesis is that the unchanged lane 3b one-ahead producer reduces R7 evaluation wall time while preserving inputs, states and outputs. The second hypothesis is that nartaa's native224 anatomical mirror fits the recovered headroom; no accuracy gain is established here. Compare the exact same frozen 39 studies, assignment and boards in one 2×T4 session, using separately compiled serial-control and prefetch processes. No training, labels or generalization evaluation is involved. Respect actual quota and the platform's nine-hour limit; no assistant experiment-hour cap.

The original P2 CPU speed screen remains **failed**, with passing mechanics. Its result has not been relabelled. This is a prospectively authorized actual-GPU P3 test. Candidate worker bytes remain `09ad6adb10bb1346a8751b9dfe33c3d732ce1616692a1e42179abd5fa43419de`, including Astra's earlier fixes. The file is stored separately as [public_r7_prefetch_candidate.py](../../knee/public_r7_prefetch_candidate.py) and mapped to the old R7 runtime path only within the new sealed packet. All other archived files remain exact. Existing root v7/v2 workers, selection-B/B3 builders and pins stay unchanged, preserving A/A′ provenance; v3 has explicit packet/source bindings instead.

## Method and admission

The new coordinator runs the prefetched baseline once, then **one** serial-control R7 phase, then the native224 arm. It does not run a second prefetch phase. The all-39×8 input reference comes from the exact pinned v7 receipt. P3 checks complete assignment/order, exact fingerprints, model state, paired physical GPU/package identity, compiled-no-fallback and sentinels, complete NPZ hashes/finite genuine arrays, stacker/reader mechanics, one producer and quiescent teardown. Separate compilations retain the existing 1e-3 tolerance; individual reload remains 1e-6. Worker RSS ≤9.5e9 bytes and each combined phase peak ≤22e9 bytes; CUDA allocated/reserved peaks are retained.

Correctness/resources and speed are distinct outcomes. A slower-board paired evaluation-wall ratio ≤0.85 admits the mirror. Above 0.85 no mirror is admitted; above 0.90 the result is labelled “output-preserving, no gain.” Passing mechanics can still qualify a plain prefetch variant only if its full conservative projection passes. A parity/resource failure does not qualify either variant; preserve it and apply the lane's reversion/closure rule, without silently retrying a changed gate.

The v3 policy explicitly excludes **only the separately measured serial-control diagnostic** from the hidden-population projection because entry mode never runs it. Report that wall separately and include it in the actual rehearsal elapsed time, which remains below nine hours. Original measured per-arm overhead and the 25% R7-window/15% imbalance policy remain unchanged. The complete conservative projection must remain ≤29,880 s. This prospective v3 policy does not alter the failed v2 projection or the user's one-probe exception.

The prefetched baseline's three visible rows must match the pinned heliosli blend reference on all 12 targets after the original six-decimal comparison. The new final three-arm CSV is independently recomputed from all 39 saved ledgers; it is not required to equal the baseline A blend.

The mirror calls the pinned public `_mirror_tri` on **uint8 triplets before LUT, crop, resize and per-channel normalization**. Sagittal centres reverse the three slice channels; coronal/axial centres flip width. Reversing normalized sagittal channels is incorrect; flipping after the odd 307-pixel crop changes its pixel support. Keep original and mirrored window hashes, both probabilities, probability averaging, strict reload and cross-board checks. No fp32 rescue, optional-arm timeout or new failure substitution.

The builder requires an official submissions CSV showing **57229227 COMPLETE with publicScore≥0.956** before creating either a mirror rehearsal or a mirror entry. A plain packet can be prepared while that score is pending. Entry building additionally requires a genuinely passing fresh v3 rehearsal, exact prepared packet and complete saved P3/worker/CSV/resource replay. Entry configuration requires prefetch on, removes all private evaluation references, and cannot launch a control phase.

## Prepared artifact and execution ownership

Ready plain rehearsal: `outputs/kaggle/ws3-prefetch-plain-qual-20261015-r3/`, PREPARATION_ONLY. Notebook SHA `e785a4547548a9ba6a548c301ab69433d9b7c0d0d6f1ad1630864c03aa63c1d6`; payload SHA `04183dc958669decca89d72d1289076763477d86dc553f5a27ac6b7925889249`; MANIFEST SHA `8214ca487041d77ec9870c072652d88712c94210c0fbf3a9513eb6da3f6d539c`. Notebook 701,619 bytes. Private/offline, pinned Docker/datasets, 2×T4; id and title agree. Earlier unlaunched non-r3 directories are development drafts, not accepted launch artifacts.

Claude reviews the prospective gates, reconciles actual quota/owned jobs and claims the **one chosen** rehearsal. The plain packet is the fallback; create a separate mirror packet only after the official score admits it, using the same builder with `--mirror --mirror-admission <official-submissions.csv>`. Do not push both variants automatically.

```bash
kaggle kernels push -p ~/Dev/rsna-knee-abnormality-detection/outputs/kaggle/ws3-prefetch-plain-qual-20261015-r3
```

No `--timeout`. Verify accepted version/code/settings before considering the job bound. Claude collects terminal evidence to a fresh result directory; expected root `ws2-a-prime/` retains the legacy path name but records recipe `ws2-a-prime-r224-v3`. Preserve logs, eight worker ledgers (six main plus two serial control), references, comparison, resources and failure receipts. Post job/version/output/stop details in STATUS/shared handoff. Stop via owner-coordinated Kaggle UI if needed; no parent collector or blind repush.

Saved acceptance API from the project root:

```python
from scripts.build_public_ws3_entry import verify_preparation, verify_result
prepared = verify_preparation('outputs/kaggle/ws3-prefetch-plain-qual-20261015-r3', False)
evidence = verify_result('outputs/kaggle/ws3-prefetch-plain-qual-20261015-r3-result/ws2-a-prime', prepared['binding'])
```

For a mirror build pass `True` and its exact preparation/result paths. Save the returned aggregate evidence; no caller-only PASS can create an entry. The builder's `--mode entry` requires both `--rehearsal-build` and `--rehearsal-result`; a mirror entry also requires an official score snapshot meeting admission.

## Verification and material limits

36 focused installed-ML synthetic tests pass, zero skips; bare-Python WS3 checks pass 13 tests with five expected optional-ML skips (helper array tests now skip cleanly when NumPy is absent): preserved WS2/waiver tests plus mirror, source/reference, P3 bounds, admission, control-phase and projection tests. Dependency-free smoke passes (synthetic CPU; no torch forward). Static preparation reconstruction, source/hash/metadata/manifest checks and every packaged Python parse pass. The legacy R7 saved-rehearsal verifier is AST-identical in v7 and the candidate. [Static receipt](../../outputs/analysis/r224-integration-20261015/ws3-preparation-verification.json), [focused tests](../../outputs/analysis/r224-integration-20261015/ws3-integration-tests.log).

The full suite run before the last explicit visible-reference addition ran 1,312 tests with 191 skips and one sandbox error: PyTorch shared-memory manager launch was denied in the existing four-worker loader test. That exact synthetic test passed outside the sandbox, 1/1; no code change was needed. The final focused suite covers the added reference gate. [Full log](../../outputs/analysis/r224-integration-20261015/ws3-full-suite.log), [isolated retry](../../outputs/analysis/r224-integration-20261015/ws3-multiprocess-test.log).

No new GPU/real-reader P3/mirror execution, hidden result or entry qualification is asserted. Original P1 evidence is reused for the unchanged candidate bytes; the new packet still needs actual two-board qualification.

One plan detail needs explicit reconciliation before future entry acceptance: the rehearsal ranks a 39-study population, whereas a visible entry ranks its three actual test rows. Their final CSV numbers cannot generally be equal. Compare raw visible-arm outputs at the existing per-arm tolerances and recompute the blend for the **same actual population**; preserve the separate exact baseline-visible-reference gate. Do not copy private rehearsal predictions into entry or claim cross-population CSV equality. Claude accepted this clarification at 12:54 UTC; it does not waive a numerical check.

Claude claimed the plain rehearsal at12:54 UTC after checking no other owned kernel was running and actual quota9.62h. The plan now permits a separately claimed mirror rehearsal after official score admission, using measured P3 evidence; it is not an automatic duplicate launch. Parent independently verified accepted source/settings and RUNNING; version1 accepted12:54:11 is recorded by Claude at12:58.

Terminal collection correction: use regex `(receipt|status|failure|assignment|prefetch-p3|\.log$|projection|ledger|\.npz$|\.csv$)` with page-size200 and no page token. The installed SDK automatically paginates. This includes `base/assignment.json` required by the checker, `prefetch-p3.json`, all eight worker receipt/NPZ pairs and CSV/log artifacts. The initial owner filter omitted assignment and P3;25 pure example paths now match. [Coverage receipt](../../outputs/analysis/r224-integration-20261015/ws3-collection-coverage.json). Claude remains the sole collector.
