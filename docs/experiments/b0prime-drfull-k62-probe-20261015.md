# B0′: dr-full checkpoint swap at B0 K62

**Terminal update, 18:15 UTC:** r2 ended ERROR with a 9.346h conservative projection, failing qualification. Saved mechanics replay passes separately; ordinary entry creation rejects the failed receipt. [Terminal result and next action](b0prime-drfull-k62-result-20261015.md). The running-state and launch commands below are historical; do not relaunch from them.

Prepared October 15, 2026, 16:15 UTC, in response to Claude's 15:33 handoff. **Historical preparation/running-state record; see the terminal failure above. No qualified entry or score.** Claude owns launch, collection and submission; Codex owns integration and targeted acceptance.

The hypothesis is that our dr-full fine-tune improves the fixed B0 ensemble when it replaces Dread v4_full. The official saved submissions CSV confirms B0 submission **57119585 COMPLETE / 0.940**. B0′ changes the checkpoint while retaining B0's K62 windows, source/reader, no TTA, CUDA fp16, S6, 70:30 percentile-rank blend and capped recovery. It does not use B3's K42 window count or R7. dr-full was trained on all 4,345 eligible studies; this is an exposed-development leaderboard swap probe, not independent held-out validation. A displayed score difference tests this fixed ensemble intervention and remains subject to rounding and leaderboard noise.

## Binding and acceptance

- Private dataset `ozgung/knee-raptor64-dr-full-fold0-final-20261013`, version **1**; `raptor_ft_fold0_swa.pt`, **292,826,627 bytes**, SHA256 `35b7fc2cc95155b13271d3bc43ebd4931a915eaa490c572d2fd02d3cac2a061c`.
- State pin `a3dbd31bdd65f306e987808708ef7afe691f9fb61e67fb4ddd86843cd7ce7812`; constructor grid `67ae62b6e4ba164abf4757e0948b4fa9cc410b587538dd19ce7be58186d465bd`; SWA epochs 9/10/11. The K42 export provenance is retained separately from this K62 inference probe.
- Reuse all **235 archived B0 sources unchanged**, adding two separate runtime/wrapper modules. Original B0/default builders and modules remain pinned and unchanged. Old worker, S6 worker, launch, merge and run reuse exact B0 bytecode against the new bindings.
- Rehearse the same frozen **39 studies**: 32 S6 probes, four old historical probes and three visible studies; preserve assignment, grouping, all original source/reload/state/injection/environment gates. Bind all 39 image fingerprints and old reader counters to B0's evidence. Replay unchanged S6 probabilities on all 39 at the existing 1e-6 tolerance and check four-logit sigmoid consistency.
- Dread's old-weight probabilities are not a valid reference for dr-full. Replace only that final visible-probability gate with exact visible reader-count checks; retain fresh source-reference inference, strict reload and both-board parity for the new weights. No fake probability reference or old-receipt substitution.
- Require actual two physical T4s, private/offline pinned Docker/assets, complete four NPZ ledgers, exact schema/coverage/finite probabilities and independently recomputed visible blend. Preserve B0's original projection formula and sensitivity scenarios; persist setup/merge inputs for exact saved replay. New qualification requires conservative projection **≤29,880 seconds / 8.3 hours**, actual elapsed <9h, worker RSS ≤9.5e9 and combined phase RSS ≤22e9 bytes. Failures stay failed; no time-budget abort or general waiver.

B0 historically projected **8.69h conservative** under its older nine-hour gate. That cannot qualify this fresh 8.3h probe; session speed is a material risk. Reconcile actual quota and owned jobs before launch. CPU checks do not estimate T4 runtime or establish competitive performance.

## Verified preparation

Final packet: `outputs/kaggle/b0prime-drfull-k62-qual-20261015-r2/`, **PREPARATION_ONLY**, 596,103 notebook bytes, 237 sources. The earlier directory without `-r2` is an unlaunched development draft; do not push it.

| Artifact | SHA256 |
| --- | --- |
| notebook.ipynb | `92c92d3eced454083cffb862ee9d7ea2fe1b02252067076c0612ab125f4d6887` |
| kernel-metadata.json | `27e509a26bc535e84e768d1c845c808145af443d91dba09eda812be62b1ac7d5` |
| payload | `613e23784e4bb107e3b5781756b41c348ba89bfa1d218a650e885776aac27408` |
| MANIFEST.json | `a87bf2766176cfceba358ddbed7551352a478440400bc5f611dbae4f28296105` |

Title equals id slug; private/offline 2×T4, original Docker and S6 assets, versioned dr-full attachment; Dread weight attachment removed. Exact source hashes are bound in the receipt; source was uncommitted at sealing and is snapshotted by the parent integration commit. [Static reconstruction proof](../../outputs/analysis/b0prime-20261015/final-static-acceptance.json).

Fresh CPU check used `fastgen`, Python 3.12.3 / torch 2.10.0, synthetic inputs only: strict checkpoint load, 62 windows, finite outputs, repeat max delta **0**, unchanged model state, exact state/grid pins and successful reproduction of B0's original Dread pin method. Average forward 35.4484 seconds, two forwards; **CPU fp32 is not a CUDA probability reference**. [Receipt](../../outputs/analysis/b0prime-20261015/cpu-k62.json), SHA `7516a3df591773cc1a201a7adfbc79f77a92d9ec467a2c859b9b9a8b39bebb9c`.

Targeted checks ran **28 tests: 24 passed, four optional reader tests skipped** because the installed test environment lacks the complete cv2/reader dependency set. All **14 new B0′ tests passed**, including isolated actual packet imports with checkout/editable fallback disabled, original-receipt rejection, synthetic state/reload/failure mutations, four-ledger/projection/blend replay and rejection of an altered CSV. [Combined log](../../outputs/analysis/b0prime-20261015/combined-tests.log). An initial fastgen test attempt lacked pydicom; the suite was rerun in `.venv`, while the real checkpoint CPU check remained in fastgen. No dependencies were installed. Bare Python: 14 tests OK, seven expected optional-dependency skips. [Log](../../outputs/analysis/b0prime-20261015/dependency-free-tests.log). Dependency-free synthetic CPU smoke PASS, no torch forward/training. No full-suite repeat or new real-DICOM qualification; unchanged B0 reader evidence is retained, and fresh T4 execution is still required.

## Owner commands and recovery

Claude claimed/launched this **one** rehearsal at 16:18 after reconciling quota/jobs. The recorded command below is already executed; do not repeat it:

```bash
kaggle kernels push -p /Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/kaggle/b0prime-drfull-k62-qual-20261015-r2
```

No `--timeout`. Post actual version, accepted source/settings, logs, status and stop/recovery details. It ends naturally; owner coordinates any necessary stop. No Codex launch or duplicate collector. Collect terminal evidence into the fresh `...-r2-result/` directory:

```bash
kaggle kernels output ozgung/b0prime-drfull-k62-qual-20261015-r2 -p outputs/kaggle/b0prime-drfull-k62-qual-20261015-r2-result --file-pattern '(receipt|status|failure|assignment|\.npz$|\.csv$|\.log$)' --page-size 200
```

The output root retains `s6-recovery-entry/`. Preserve root/worker receipts, assignment, all four NPZ files, CSV and logs even on failure. The builder's `verify_preparation(root, cpu_receipt)` reconstructs the exact packet; `verify_result(result_root, binding)` replays the real saved evidence. This check runs automatically before entry creation:

```bash
.venv/bin/python scripts/build_public_b0prime.py --mode entry --cpu-receipt outputs/analysis/b0prime-20261015/cpu-k62.json --rehearsal-build outputs/kaggle/b0prime-drfull-k62-qual-20261015-r2 --rehearsal-result outputs/kaggle/b0prime-drfull-k62-qual-20261015-r2-result/s6-recovery-entry --destination outputs/kaggle/b0prime-drfull-k62-entry-20261015 --slug b0prime-drfull-k62-entry-20261015
```

No entry packet exists yet. A successful build removes private evaluation references and all rehearsal wrappers; inference remains image-only. Claude owns entry launch, visible raw-output/blend acceptance and any already-authorized competitive submission. A′ 57229227 is separately still PENDING at 16:01:55 UTC; its result continues to control the suspended v3 rescue lane.

**16:19 UTC execution update:** Claude reconciled quota 8.89 h/no other owned running kernels and claimed/launched r2 at 16:18. Parent independently sees RUNNING and exact accepted code/payload/settings/normalized dataset refs. [Accepted source proof](../../outputs/analysis/b0prime-20261015/accepted-source-verification.json). Pull/status do not expose exact server version or attachment versions; owner push receipt is pending, requested dr-full v1 is bound by the sealed metadata and runtime checkpoint SHA/bytes. Parent integration 7c00222 matches sealed source hashes. No qualification/result/entry yet; collection remains Claude-owned.
