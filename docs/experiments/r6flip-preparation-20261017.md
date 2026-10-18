# R6-flip candidate preparation — 2026-10-17

Preparation only: no remote push, GPU run, download or submission. Not qualified on real images.

**Frame, frozen before any run:** Raptor half = A′ R384 two-view mirror ×0.65 + R224 one view ×0.35. These are A′'s own worker bytes and readers. Compact half = heliosli R6 `fe055406fad6a12f` from `heliosli/rsna-knee-r6-frozen-fe055406-w1` v1 (manifest 6508ce69…), run through compact_infer v4c `--flip-tta`. The final blend per finding is w·Raptor + (1−w)·rank(R6), with w = Lateral OA 0.2, PF OA 0.6, Synovitis 0.8, Baker's 0.35, and 0.5 for every other finding. The baseline is A′ (57229227, 0.956). Excluded: na-full, V2, extra readers and the R224 mirror. Our Raptor arms come from our own readers, so the predictions are our own, not a copy of the public notebook.

**Code (commit af8b369):** `knee/public_r6flip.py` holds the frame, recovery, independent flip maps and digests. `knee/public_r6flip_worker.py` runs compact_infer `main()` in process, unchanged and observed. `knee/public_r6flip_entry.py` is the controller. Two scripts support them: `scripts/build_r6flip_entry.py` builds the packets and `scripts/qualify_r6flip_cpu.py` produces the CPU pins. Tests are in `tests/test_public_r6flip.py`.

**Deadline guard (strategy revision):** every stage stops at T0 + 8.3 h minus a 180 s reserve. compact_infer gets its own `--deadline`. When every arm is complete, A′ recovery applies, with a 3% cap. Otherwise the public stage-3 fallback applies: each model is ranked over its own rows, a missing model takes the others' rank, and a study with nothing gets 0.5. The entry never aborts and always writes a complete CSV, using constant 0.5 only as a last resort. The receipt records `fallback_studies` and a per-arm breakdown. The rehearsal stays strict (no fallback allowed). A na-full Raptor arm can be added to `RAPTOR_ARMS` later, but the controller currently refuses it because no na-full stage exists yet.

**Packet:** `outputs/kaggle/r6flip-rehearsal-20261017/` contains notebook e1bf3bbe…, payload f521b0de… and kernel-metadata 441f8eaa…. It is private, offline, 2×T4, uses the A′ image 37c64f7d…, has title equal to id and no timeout field. Byte replay passes (`verify`). The A′ v2 payload's 240 sources are kept byte for byte, and four modules are added.

**Checks:**
- CPU pins from the actual checkpoints: `outputs/analysis/r6flip-20261017/cpu-pins.json` (3d25aefe…). It was produced with torch 2.10.0, timm 1.0.26 and uv offline. It covers 4 folds plus the side net, strict load, and flip maps equal to the pinned compact_infer.
- `.venv` targeted tests: 80 pass, 5 skipped (r6flip, r6 profile, ws2, r7 recovery). This includes a CPU end-to-end `execute()` run with two real spawn loader children, an injected failure and the ledger, plus a fresh `python -I` import of the unpacked payload.
- The system `python3.12` discover run gives 22 errors, all in unrelated optional-dependency suites; none are in r6flip. Smoke passes.

**Runtime estimate:** this is not measured. It is based on the A′ v2 ledgers from the slower board: R384 5.82 s/study, R224 3.33 s/study, and an R7 four-model forward of 12.35 s/study, which doubles under flip to about 24.7 s. Decode is hidden behind the 2 loader workers.
- Point estimate: about 6.4 to 6.8 h.
- Using A′'s ratio of actual to point time: about 7.5 to 7.9 h.
- Ordinary conservative estimate: about 8.8 to 9.0 h, which is expected to FAIL the 8.3 h gate. Dropping R224 brings it to about 8.1 to 8.3 h.
- The deadline guard bounds the hidden run in any case.

**Open risks:**
- No real-image, GPU, compile or DataLoader qualification has been run.
- The R7-reference input-parity gate (32 probes) assumes the loader-child decode equals the in-process decode.
- The fallback and the 0.5 last resort change A′'s fail-closed semantics; this was requested.
- A projection-only failure needs a named waiver file at entry build.
- R6 fastMRI provenance is unresolved.
