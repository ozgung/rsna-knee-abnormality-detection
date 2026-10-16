# V2-all solo entry scope

Fixed October 16, 2026, 09:20 UTC, before any V2-all competitive score. Claude requested the solo frame after relaying the user's post-hoc approval for all-data training. Codex scopes and accepts integration; Claude owns substantial implementation, remote execution and submissions.

The primary probe replaces only the S6fall checkpoint in the archived publisher solo pipeline. Compare against publisher-reported **0.950**, subject to Claude's pending official provenance check, using the agreed tie-or-better candidate rule. A displayed tie is not evidence of improvement. A subsequent A′ replacement at compact share0.5 is a separate qualified entry; it is not part of this solo probe. Preserve the original S6 fold0 BLOCKED verdict and post-hoc authorization as separate facts.

## Frozen source and inference

Archive: `outputs/research/public-0957-20261015/kernels/nartaa_rsna-knee-s6fall-solo/rsna-knee-s6fall-solo.py`, SHA256 **6e87350d5ae092b7a889546ca13517303523bc469f7e611ffe945913434d22fe**. This is the captured source, not a claim about the latest server version or our reproduction of its score. Its executable weights are A=0/B=1 despite older blend wording in the header.

- **B:** the new single V2-all ConvNeXt-small/spatial-attention checkpoint. Preserve pinned `compact_infer.py`34f433fe…, source tree, SKM-TEA anatomy5904dd16…, size320/crop288, v4 views,3 slices/3.5mm, eval stride1, chunk192 and workers2 per GPU. Use **fp16 on T4**; existing bf16/A100 gold checks do not qualify that backend.
- **A:** preserve the embedded publisher CoAtNet source exactly, SHA256 **bcd864cbf349882e876959b49cb348cbb89b4c62fc204d4e4261049c5e3a79c5**, with its original checkpoint/input/TTA behavior. A runs on the complete inference population even though its ordinary weight is zero. Do not substitute na-full, R224, R7 or a cached CSV for A.
- The no-failure output is the per-target average-tie percentile rank of B over the **entire** inference population, then the publisher's float32 submission serialization in official ID/target order. No weights or thresholds are selected after seeing the score.

The “solo” name describes the normal score contribution, not a single-model operational dependency. A's OAI-associated checkpoint remains required for recovery and sanity checks; this is not an OAI-free entry. V2's label-teacher exposure also remains explicit. The intervention changes checkpoint training/targets and population relative to public S6fall; any score difference is not isolated labels causation.

## Exact recovery behavior

Preserve the captured controller functions, not a newly invented mean-probability fallback:

1. Retry B's failed studies once when the existing controller permits it. A process/setup/source/model failure, missing or inconsistent receipt, nonfinite scored output, exceeded cap or absent retry time is a hard error. A deadline/timed_out result is a hard error in both passes; it never qualifies for imputation.
2. For B's final test stage with population N≥100, at most `max(3, ceil(0.0025*N))` unresolved rows are permitted after retry. Otherwise the cap is zero, including the39-study healthy rehearsal. At N=1300 the cap is4. A's own recorded internal fallback cap is `max(3, ceil(0.01*N))` (13 at1300), not permission to fail A as a component.
3. If B rows remain missing within cap, fill those B rows with the per-target arithmetic mean of successful B **probabilities**, then rank B across all N. Replace only the failed rows' final ranks by A's ranks computed across all N. Never rank just the failed subset; do not emit the fill probabilities directly.
4. Preserve the publisher's A/B rank-correlation check on successful rows: when at least100 survive, all correlations finite, mean≥0.60 and minimum≥0.40, using the exact original floating-point comparisons. This inference gate is separate from the failed fold0 training-comparison sanity gate.
5. Failure deletes the submission candidate; no A-only or partially written file may remain. Healthy real qualification requires genuine predictions on all39 from both arms; exercise retry/caps/deadline failures separately with synthetic fault injection.

[Exact pure-function synthetic replay](../../outputs/analysis/s6v2-solo-scope-20261016/scope-function-acceptance.json), [replay script](../../outputs/analysis/s6v2-solo-scope-20261016/audit_scope.py). Checks cover average ties against SciPy, cap boundaries, full-population mean-fill/rank-A recovery and correlation gates. They do not exercise GPU, raw reader, live retries or deadline handling.

## Export and packet gates

The expected private export is `ozgung/knee-s6v2-v2all-20261016`, file `E_prior_s42_v2all.pt`. **That filename does not match the pinned loader's `E_prior_s*_f*.pt` glob.** At packet preparation, create a hash-verified runtime alias such as `E_prior_s42_fall_v2.pt`, retaining identical bytes and an explicit source-file/alias mapping. Use a dedicated model directory with exactly that one checkpoint. Do not modify the inference loader or accidentally co-load public S6fall. [Discovery check](../../outputs/analysis/s6v2-solo-scope-20261016/filename-discovery.json).

Before sealing: bind the actual immutable dataset version, checkpoint full hash/size, strict restricted CPU load, canonical targets, constructor/full-state pins, training revision/config/seed16epochs, all-data membership4345/holdout0 and gold/quarantine exclusions. Verify actual completed export/runtime receipts; a proposed slug is insufficient. The existing export script provides SHA256SUMS/run_config/membership, not the publisher's DONE/code-manifest layout. Adapt the preparation layer to genuine new export receipts; never copy an old DONE assertion or manufacture a training-completion claim. Keep public code extras in their separately pinned source mount. No weights, reports, labels or raw images in committed code or staged inference inputs.

Seal a fresh private/offline2×T4 rehearsal, using the existing frozen39 image registry, exact final model/source/dependency identities, both-board sentinels/strict reload, raw-input and probability checks, complete genuine ledgers and full-population rank/schema replay. Test missing/wrong source/checkpoint/target/assignment, retry recovery, over-cap failures and first/retry deadlines before entry admission. Record time, peak memory/disk and both arms' production/fixed/retry costs. A's full-population execution is part of runtime; do not estimate this as S6-only.

There is **no qualified runtime estimate yet** for this new checkpoint/package. Retain ordinary fresh8.3h qualification and actual9h platform checks; earlier waivers do not transfer. The captured solo controller also has its own8h guard and10min backstop plus stage guards; preserve/measure those as the inherited inference contract, not a new training budget. If that contract needs changing, record a separate prospective variant before a score. No self-imposed training stop or duplicate job follows from this scope.

Only after exact model/package qualification may Claude build the image-only entry and execute the authorized competitive workflow. Freeze artifact identities, attach code/dependencies/weights offline, validate finite[0,1] exact ID/target coverage, and preserve failures. Codex has scoped/tested functions only; no entry packet, GPU run or submission was created.
