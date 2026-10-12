# S6 capped-recovery protocol — October 12, 2026

Frozen before stress metrics are inspected. Source baseline: committed strict S6 entry version2 at96dbcea, failed competition submission57086107. Best actual public remains0.950. This is a new candidate, not a retrospective qualification of the submitted method.

## Evidence and hypothesis

Official SDK at02:34UTC confirms S6 and full384 generic unhandled rerun errors, zero output bytes and no score. Neither hidden traceback nor exact execution phase is available. PENDING includes queue time; failure timing does not establish startup versus mid-run failure. Remaining S6 observer gates reject native recovered/localizer errors and skipped series, unlike source-supported continuation. Resource slope and one-study extremes are under independent review.

Hypothesis: retaining successful image-only predictions with a small, explicit recovery population can preserve the observed S6 gain while tolerating data-local unusable input. It will not fix global schema, dependency, state or device failures.

## Frozen candidate

Keep successful inference, weights, windows, FP16, source six-decimal S6 values and fixed70:30 blend. Replace an entire missing12-target arm row with that arm's target-wise mean over genuine successful studies across both shards. Quantize successful S6 values and its means to the source six-decimal representation. Require successes in each arm. Cap the union of studies missing either arm at `floor(0.03*N)` using actual test count. Fill first, then rank every actual test row within each arm using average percentile ranks; blend70:30. Record explicit genuine/recovered status per study and arm.

Retain global source/state/schema/path checks and platform/resource failures. No arbitrary worker-death swallowing or deadline substitution. Data-local exception classes and actual cleanup behavior must be enumerated and tested before runtime acceptance. Accepting upstream recovered diagnostics requires a valid returned image input, not fabricated images.

## CPU sensitivity experiment

Use the exact saved870 original320/S6 arrays and frozen truth/component evidence from the accepted comparison. Freeze masks and hashes before scoring: seed20261012, label-blind ordinal permutation. Controlzero failures; union8 (1%) and26 (3% cap) for old-only, S6-only, coincident and disjoint failures. Disjoint cases partition each union between arms. Additionally use top26 S6 window-count studies, descending workload/canonical-index tie, with those same four patterns. Total13 scenarios; no policy, mask or weighting search after outcomes.

Acceptance per primary stress scenario: exact zero-failure equivalence; positive macro delta against intact original320; positive lower95% bound from the existing2,000-component paired bootstrap (seed42); fewer than three supported targets declining by at least0.01. Report all12 target AUCs, supports/class counts, undefined targets and explicit macro policy. This measures controlled sensitivity, not naturally failed-study accuracy or independent generalization; public-source exposure remains.

Test cap boundaries, tied means, permutations, shard partitions, both-arm failures, zero successes, nonfinite values and fatal failure isolation. Existing dependencies only; CPU first, no GPU launch until reviewed results. No self-imposed experiment-hour veto.

## Next gate and ownership

SolMedium owns implementation/evaluation in the isolated `s6-capped-recovery` worktree. Parent reviews exact numerical evidence and refreshed rules, commits, and only then proceeds to runtime packaging. A fresh actual-pinned image-only rehearsal must demonstrate healthy-study parity, isolated injected recoverable failures, subsequent-study correctness, unchanged model state, complete ranks and measured resources. Old strict receipts cannot certify this code. Parent retains launch/submission ownership; no blind resubmission.
