# Iteration 4 result: axial patellofemoral arm, 2026-10-14 — BLOCKED, no verdict

Protocol: [raptor64-axial-pf-it4-protocol-20261014.md](raptor64-axial-pf-it4-protocol-20261014.md). Status: **implemented and tested; training started twice and lost twice to Colab runtime termination; no kill-rule read-out exists.** The arm is neither kept nor closed.

## Orientation (protocol step 1)

- `knee.raptor_input.prepare_study` does **not** canonicalise in-plane orientation. It calls the pinned public `build_study` with the reader's own `_make_reader` (source `outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py`, SHA `859aae12…`, lines 255–292). In that reader, `ImageOrientationPatient` is used only to sort slices along the normal (`pos = dot(IPP, cross(row, col))`). The pixel arrays keep DICOM storage order, with a central 140 mm crop and a resize to 336 px.
- I applied the protocol's heuristic as `knee/raptor64_training.py:axial_anterior_side`: the anterior side is the vertical tissue edge farther from the tissue centroid. I audited it by eye on the staged cache:

| Check | Result |
| --- | --- |
| 20 random studies | 19 of 20 anterior at the top. The one failure (row 4203, an oblique knee) was called "bottom". |
| All 4,345 studies | 4,285 called top, 60 called bottom |
| First 20 of the "bottom" calls | **20 of 20 wrong**: every one is anterior at the top |
| 10 largest column-skew studies (possible 90° rotation) | 0 rotated |

  Across 50 eye-checked studies, none was stored anterior-bottom. The heuristic only added errors.
- **Deviation, recorded before training:** the training rule is DICOM storage order with anterior at the top (`ORIENTATION_RULE = 'storage-order-anterior-top-v1 (centroid heuristic audited and rejected)'`). The heuristic is kept for audit only. The p = 0.5 anterior–posterior flip (labels and teacher unchanged) is applied as the protocol requires. Montages are in `outputs/it4/it4-orient-*.png` (ignored).
- **Axial presence:** all 4,345 staged studies have axial content. Metadata and masks agree on all 4,345. As a result, the missing-axial training exclusion and the evaluation fallback exclude no studies.

## Implementation (commits `05ae207`, `8f0ff34` on `claude/axial-pf-it4-20261014`)

| Component | Implementation |
| --- | --- |
| Arms | `ax-crop` and `ax-full` in `knee/raptor64_training.py` (`AXIAL_SPECS`, `FROZEN_AXIAL`, `axial_variant`, `axial_block`, `axial_crop_box`, `axial_presence`) |
| Input | Slots 52–63 only. `ax-crop` takes rows 0–184 × columns 59–276 of 336 (anterior 55% × central 65%), bilinear to 384. `ax-full` takes the full 336 slice, bilinear to 384. |
| Windows | All 10 windows: K10 for training (each centre used once), for per-epoch selection, and for export |
| Recipe | Same as in-distil: ImageNet CoAtNet-384, original attention-MIL head, report BCE + 0.5 × BCE to the teacher (`dab0621a…`), fold 0, seed 42, 12 epochs, top-3 SWA. The crop is the only difference between the two arms (tested). |
| Plumbing | Builder, controller (probe K and per-arm export K), VM tool, exporter and bundle accept the two arms. |
| Unchanged arms | The existing in-orig v1 and in-distil v5 packages still rebuild byte-identically (tested). |
| Evaluator | `scripts/evaluate_raptor64_axial_blend.py`: fixed 0.7/0.3 rank blend into in-distil K42 (`0eb638e2…`) and into Dread v4_full K42. Applies all five kill rules, with the paired component bootstrap (2,000 draws, seed 42), plus per-target arm-alone AUCs. Dry-run on existing OOFs: in-distil PF OA 0.8637, Dread PF OA 0.9295. |

**Tests:**
- Locally with `.venv/bin/python` (3.12, no timm): all raptor64 suites plus the evaluator ran **119 tests OK, 6 skipped**. The skips are the timm-dependent e2e tests and the real v4 checkpoint test.
- On the Colab VM (Python 3.13.15, torch 2.11.0+cu130, timm 1.0.26): **118 OK, 1 skipped** (no local v4_full checkpoint). This run includes the new CPU end-to-end `test_it4_axial_arms_end_to_end`. Log: `outputs/it4/it4-vm-tests.log`.

## Runs (both lost, not resumed)

| Session | Endpoint | Created → lost (UTC) | What happened |
| --- | --- | --- | --- |
| `raptor64-axial-it4-20261014` | `…ass1b0-2p4tme7ruo4t9` | 10:33 → about 11:48 | Staged cache (content `0bdc189c…`, 1,759 s). Orientation audit and VM tests passed. Both arms launched 11:20. The runtime then returned 404, before epoch 0 finished. |
| `raptor64-axial-it4-r2-20261014` | `…ass1c1-2mjiwjyaxc8r0` | 11:56 → about 13:12 | Staging hit one ChunkedEncodingError and then PASSED on resume. Both arms launched 12:34. The runtime returned 404 during epoch 1. |

- **Both runtimes vanished after an identical keep-alive duration** (4,578 s; 404 from iteration 76). The local token-refresh loop was running, and `colab sessions` then listed no server sessions.
- The it3 session ran 5.5 h this morning without this. The cause is unknown: possibly an account-level compute or runtime limit. I did not chain VMs around it.
- **Configs (r2):**
  - ax-crop `062f4a64…`, ax-full `8476b66d…`;
  - packages `343b44bb…` and `e18d906b…`;
  - bundle `b25279c6…`, runner `60bf96b0…`, autopilot plan `2ed4b2ac…`.
- **Measured epoch 0, two arms concurrent on an A100 40 GB:** 1,540 s per epoch at 7.9 GB and 100% utilisation, so about 5.2 h for 12 epochs. Epoch-0 K10 selection macro and PF OA:
  - ax-crop: macro 0.756, PF OA 0.708;
  - ax-full: macro 0.774, PF OA 0.759.

  These are single-epoch, report-selected development numbers, **not evidence**.
- **Epoch-0 resume backups exist** in the private datasets `ozgung/knee-raptor64-{ax-crop,ax-full}-fold0-20261013` (13:03 UTC). No final dataset was created and nothing was exported.

## Next action

- Resolve the Colab termination first, for example by checking the compute-unit balance or the session type.
- Then rerun on one runtime that will survive about 6 h: `outputs/it4/vm-r2-all.py` with the it4 bundle, followed by the autopilot.
- Alternatively, resume from the epoch-0 backups using a pinned `resume_sha256` config.
- Then evaluate with `scripts/evaluate_raptor64_axial_blend.py`. Pass `--best` with each arm's `raptor_oof_fold0_fold0.npz`, downloaded from the VM workdir before stopping, for kill rule 3.
