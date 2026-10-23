# B0′ projection-waived knowledge probe

Prepared October 15, 2026, 18:25 UTC. **Separately labelled user-reported waiver; original qualification remains FAIL.** Claude's 18:18 shared handoff reports that the user chose “Submit anyway” at about 17:20. Codex did not independently observe that direct message. This authorization covers one B0′ knowledge probe, not a final candidate or a general gate relaxation. Claude owns push, visible acceptance and submission; Codex has launched none.

The waiver `b0prime-projection-20261015` is bound to failed receipt SHA **`ef9838a483d02c9e24e7a70553871e4a1d30541949a8f3d4525f0903344abbfe`** from [r2's failed rehearsal](b0prime-drfull-k62-result-20261015.md). It waives only the estimated nine-hour and 8.3h projection bounds. The conservative estimate stays **33,645.493 seconds / 9.346h** in the receipt; the actual nine-hour alarm and elapsed-time checks remain enforced. Claude's historical-runtime and timeout-probability estimates are owner-reported, not independently validated guarantees.

The separate builder clones the frozen saved validators into an isolated namespace, accepts the exact failed status instead of PASS, and removes only projected upper bounds. It replays all four ledgers, source/checkpoint/input/state/reload/sentinel/injection/resource/S6/CSV/projection checks. It does not mutate validators, inference code or failed evidence. The ordinary entry builder still rejects the failed rehearsal. The waiver validator is excluded from the inference closure.

Final entry: `outputs/kaggle/b0prime-drfull-k62-probe-waiver-20261015/`; **BOUND_PROJECTION_WAIVED_PROBE_BUILD**, qualification_pass=false, 564,046 notebook bytes, 234 production sources. Pure image-only configuration, private/offline 2×T4, original Docker/assets/dr-full version1, title equals id. Production sources are byte-identical to r2; three rehearsal wrappers and all private evaluation references are removed.

| Artifact | SHA256 |
| --- | --- |
| notebook.ipynb | `c29bd07424d8528430c17d444c728af4860e6ac0e487829b21480f7616c4989e` |
| kernel-metadata.json | `3f060d6b56f5049d82fe13be847270b60f84884e5074ac3a759b0699ffc39e55` |
| payload | `e9f0a9cf49ff80b890d390162d98c2a38c4fb142f125cf8f86c3b7eeca5b3108` |
| MANIFEST.json | `45d166caa86dd5aab29625dee5ff6ab6e0096219080ca66421abd96a48d9dd33` |

Acceptance: **21 targeted tests PASS/zero skips**, including seven new waiver tests; bare Python seven tests OK/five optional-array skips. Mutated receipt, missing/wrong waiver, state/input/RSS and actual elapsed≥9h are rejected. Real saved replay passes; ordinary qualification still rejects. Entry test verifies unchanged production bytes, stripped fixtures and the exact 32,400-second alarm. [Tests](../../outputs/analysis/b0prime-20261015/waiver-tests.log), [bare tests](../../outputs/analysis/b0prime-20261015/waiver-dependency-free-tests.log), [static packet proof](../../outputs/analysis/b0prime-20261015/waived-entry-static-verification.json). No new model/device or inference change; prior real-reader/CPU evidence is reused. No full-suite or new CUDA run is claimed.

After actual job/quota/submission reconciliation, Claude claims this one entry:

```bash
kaggle kernels push -p ~/Dev/rsna-knee-abnormality-detection/outputs/kaggle/b0prime-drfull-k62-probe-waiver-20261015
```

No `--timeout`. Post accepted version/source/settings and preserve visible receipts/four NPZ/CSV/logs under `outputs/kaggle/b0prime-drfull-k62-probe-waiver-20261015-visible/s6-recovery-entry/` with the existing filtered collector. Before any separately owned submission, compare visible raw old/S6 probabilities and fingerprints to the r2 same-study outputs at their existing tolerances, and independently recompute the unchanged 70:30 blend over the actual visible population. A pending or failed visible run cannot be submitted. No duplicate collector or automatic final selection.

Stop/recovery: no job launched by Codex; Claude posts actual job/version/log paths after claiming it. Preserve this immutable packet and the original failed r2 result. Entry runs end naturally; an actual timeout or other failure stays failed. The existing campaign heartbeat remains active.

## Visible result and submission

Claude pushed version1 and submitted **57240823** at18:33:31 UTC. Parent independently matched accepted notebook/payload/settings to this sealed packet, then replayed the collected visible output:177.097s,3 studies, root plus four workers PASS. All four NPZ ledgers and genuine coverage passed unchanged merge checks. Raw dr-full/S6 probabilities and input fingerprints equal r2 on these studies (max delta0); the recomputed70:30 rank blend equals the official-schema CSV exactly (max delta0), finite in[0,1]. CSV SHA `2a33f3946c1c26b79d65db403d7aa9423edf44517554e5a2e48c733a056db440`.

[Accepted-source proof](../../outputs/analysis/b0prime-20261015/waived-accepted-source-verification.json), [visible replay](../../outputs/analysis/b0prime-20261015/waived-visible-acceptance.json), [official CSV snapshot](../../outputs/analysis/b0prime-20261015/official-after-waived-probe.json). Official status **PENDING at18:35:19 UTC**, no score; A′57229227 also PENDING. This establishes visible mechanics only. Original qualification remains FAIL; no hidden completion/generalization/final-selection claim. Claude owns terminal monitoring, no duplicate collection or submission.

The recipe pre-registers displayed-score reading against B0's0.940:≥0.942 information gain;0.940–0.941 neutral;≤0.939 degradation; timeout failed probe. Interpret this as an exposed ensemble intervention, not clean held-out causal evidence.

## Terminal competitive result, October 16

Official CSV at 01:43 UTC confirms **B0′57240823 COMPLETE / public 0.944**, versus **B0 57119585 COMPLETE / 0.940**. [Independent snapshot](../../outputs/analysis/local64-20261015/official-submissions-20261016-0143.json). The displayed +0.004 exceeds the registered ≥0.942 heuristic. It supports testing another checkpoint replacement; it does not identify a labels-specific mechanism or establish a two-standard-deviation effect without hidden paired-score variance. Completion preserves the original failed conservative projection and its named waiver; no new qualification PASS or final-selection action follows.

Claude reports completion inside the actual 9 h limit; an exact hidden timing receipt has not been collected by Codex. The next scoped proposal is [na-full in A′'s R384 position](na-full-aprime-swap-scope-20261016.md), with a fixed one-view K62 frame and fresh qualification gates.
