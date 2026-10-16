# Matched S6 solo-pair preparation acceptance — October 16

**17:37 UTC: implementation integrated; qualification launch and entry admission held for missing state/reload and actual-time checks.** This is acceptance of specific preparation checks, not GPU qualification. [Frozen scope](s6-v2all-solo-scope-20261016.md), [actual checkpoint CPU/export acceptance](s6v2-final-acceptance-20261016.md).

Claude delivered implementation5a54b58 and readiness note3d339f3 in `.claude/worktrees/s6-solo-pair-20261016`; the note had not reached the main shared log. Codex integrated only the six implementation files as **9e2b420**, preserving newer na-full repairs and STATUS. No remote push/run/submission occurred.

## Initial immutable packets

| Arm | Local preparation | Notebook SHA256 | Payload SHA256 |
| --- | --- | --- | --- |
| V2 | outputs/kaggle/s6solo-v2-qual-20261016 | 3c0a55c77ba742d7b74a61939259bd35990e4f6557bad1db9fa8648f4718125f | 884e1d3a5728ba722b7aff85fa4cb8e2df94c27dd935022ced0f4ec4f21bdda8 |
| Control | outputs/kaggle/s6solo-s6fall-qual-20261016 | 6950456112eb8b5db5f83b66fe1fb5df9670c7463fff2b8378c9a5488bb073f3 | 9a99ad20555332129a97a9b4730602d29c27d2ac65fa68f8e3e50e7a9baaa040 |

Both remain unlaunched initial snapshots; preserve them when preparing replacements. Private/offline/T4, title=id, captured publisher Docker image, no timeout field. Exact model identities remain V2e32e5976…/200188027bytes and control0c7297d5…/200185003bytes. V2 identical-byte runtime alias E_prior_s42_fall_v2.pt is staged alone; control's native filename is staged alone.

Parent `.venv/bin/python -m unittest tests.test_s6_solo_runtime tests.test_s6_solo_pair -v`: **31 tests PASS** (21 new,10 prior). Default dependency-free smoke PASS. Parent `scripts/check_s6_solo_pair.py` PASS: deterministic packet replay, fresh isolated `python -I` imports from each emitted tree, all six source files identical across arms, identical common config and derived controller. Only the per-arm weight/provenance bindings, encoded payload literals, notebook id/title and extra V2 dataset attachment differ. Common source bundle **67e75ad63e3343bbc22531b61328296713d158e73c101e7dc9fa0e1e21afc949**; common config **a3eeebdb8a1b29fab3588020fe911dfdf98fb33a11ae7c4e08ecac9eda490e40**.

[Parent test log](../../outputs/analysis/s6solo-pair-20261016/parent-tests.log), [isolated pair check](../../outputs/analysis/s6solo-pair-20261016/parent-pair-check.json). Tests exercise synthetic mocked subprocess/GPU layers, captured retry/deadline/cap behavior, full-population ranks, fixture mapping, failure deletion and builder admission mechanics. They establish no real-reader, GPU-state or runtime qualification.

## Code-mount version deviation accepted for packaging

Both arms mount public nartaa/rsna-knee-s6fall-weights **version3** for the code extras absent from version1; control also reads its weights from that mount. Its descriptor's already inspected version1 provenance remains explicit. This deviation is accepted prospectively because the checkpoint full hash/size is unchanged and remains mandatory at runtime; it does not imply the parent downloaded or loaded version3 weights.

Parent independently downloaded code_manifest.json, s6_extras.bin and checkpoint-manifest.json using explicit `/3` paths. Actual files match the builder's full pins and Claude's copies. All21 regular archive members have exactly the expected paths/sizes/hashes from the293-entry source manifest; Python files compile. Manifest compact_train14767c7a…/compact_infer34f433fe… equal actual CPU-qualified sources. Checkpoint manifest is byte-identical to the accepted version1 file; official version3 listing gives200185003bytes for the control checkpoint. Fresh runtime must still verify full0c7297d5… bytes before use.

[Independent mount receipt](../../outputs/analysis/s6solo-pair-20261016/parent-control-v3/acceptance.json), [replay](../../outputs/analysis/s6solo-pair-20261016/audit_code_mount.py). This accepts the packaging location, not any new publisher score/clinical claim or all-data generalization.

## Acceptance gaps demonstrated independently

The agreed scope requires actual constructor/full-state binding and both-board strict-reload/sentinel evidence. Current `knee/s6_solo_rehearsal.py:assemble` checks39 genuine rows, pixel hashes, B cross-board prediction parity, resources and projection. It does **not** collect or verify GPU constructor/buffer/full-state fingerprints before/after inference, or retained strict-reload evidence. The actual CPU-state pin in cfg is not consumed there. Pinned compact_infer does use strict=True at load; that source fact is distinct from the missing fresh per-board state/reload receipt. A has neither fresh state/reload evidence nor an explicit equivalence justification tied to exact reusable source/constructor/input/state evidence.

[Explicitly synthetic counterexample](../../outputs/analysis/s6solo-pair-20261016/missing-state-counterexample.json), [replay script](../../outputs/analysis/s6solo-pair-20261016/audit_missing_state.py): actual production assemble and verify_result accept PASS with a genuine expected CPU-state pin in config but no GPU state, constructor or reload files. A second counterexample changes only the measured finish time to **32401seconds**, beyond the registered actual<32400second limit; assemble and saved-result replay still accept it. All synthetic traces live in temporary directories and were deleted; no GPU, real image or checkpoint was loaded and no false real qualification receipt was created.

Claude was asked through the shared handoff to finish the substantive instrumentation/qualification checks in the existing isolated worktree. Preserve captured controller/inference bytes and full-population A; prefer observational rehearsal-only hooks, record overhead, bind physical board/model/input identities, compare B state against the actual CPU pins and verify unchanged state/constructor buffers with strict reload and finite same-input sentinels. For A, supply genuine fresh evidence or an explicit exact unchanged-evidence justification. Reject missing/wrong/changed state, absent strict reload, mismatched model/GPU/input and actual elapsed>=9h in saved replay and entry admission. Synthetic fixtures may exercise mechanics only and must remain labeled synthetic.

Next: receive Claude's fix commit and fresh separate packet identities, then targeted integration/negative acceptance before any owner launch. Existing captured recovery/caps/rho/deadlines, ordinary8.3h projection and actual9h guards remain fixed. No waiver, fitted model change, new automation, competing implementation or competitive entry is inferred from the passing preparation checks.
