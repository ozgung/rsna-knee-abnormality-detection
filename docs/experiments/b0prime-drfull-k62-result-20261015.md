# B0′ K62 rehearsal: runtime failure, saved mechanics pass

**18:25 UTC authorization update:** Claude subsequently reported the user's explicit choice to submit one projection-waived knowledge probe. The [separate waived entry](b0prime-projection-waived-probe-20261015.md) preserves this failure and the actual nine-hour limit; it does not turn qualification into PASS. The no-entry state below describes the original acceptance checkpoint.

Recorded October 15, 2026, 18:15 UTC. **B0′ is not qualified; no entry packet or submission was created.** Kaggle independently reports `ozgung/b0prime-drfull-k62-qual-20261015-r2` ERROR. Its already-collected root receipt is FAIL / ValueError / `NINE_HOUR_PROJECTION_GATE`. Exact server version still awaits Claude's push receipt; the accepted source/settings match the sealed r2 packet. [Preparation and immutable hashes](b0prime-drfull-k62-probe-20261015.md).

## Result

All four workers completed PASS, covering 39 genuine rows per arm (20/19 shards). Actual rehearsal elapsed was **1,555.694 seconds / 25.93 minutes**; the failure concerns the hidden-population projection, not an actual nine-hour rehearsal overrun.

| Projection | Seconds | Hours |
| --- | ---: | ---: |
| Base scenario, no added windows/imbalance | 24,579.738 | 6.8277 |
| Conservative, 25% more S6 windows / 15% imbalance | **33,645.493** | **9.3460** |
| Required qualification maximum | 29,880 | 8.3 |
| Actual platform runtime maximum | 32,400 | 9 |

Conservative excess is **3,765.493 seconds / 62.76 minutes** over 8.3h and **1,245.493 seconds / 20.76 minutes** over nine hours. The actual hidden run has not been attempted. Sensitivity scenarios are estimates, not population bounds. The actual platform limit cannot be waived; the earlier A′ projection exception does not apply to this probe.

## Independent saved diagnostic

The ordinary saved verifier correctly rejects the terminal FAIL receipt with `PASS_EXACT_REHEARSAL_REQUIRED`. A separate read-only diagnostic observes the expected root-status and projection rejections while retaining the other checks. It independently asserts exact failed status/type/reason, version/mode/source binding and actual elapsed <9h before allowing those three diagnostic exceptions. It does not normalize or rewrite the failed receipt and cannot build an entry.

**Mechanics audit passes:** exact prepared packet/source/checkpoint bindings; all 39 image inputs and old reader counters; state/grid/reload and both-board sentinels; classified recovery injections; all four worker receipts/NPZ hashes, finite genuine outputs and resources; unchanged S6 probabilities and four-logit sigmoid consistency; recomputed 70:30 visible blend; and exact projection replay from saved measured inputs. It observed 446 `require` calls, with only three expected rejections: terminal FAIL, nine-hour projection, and 8.3-hour projection. All other observed calls passed. This is real-reader/mechanical evidence on the frozen exposed-development fixture, not qualification or competitive accuracy.

- [Diagnostic audit](../../outputs/analysis/b0prime-20261015/terminal/diagnostic-audit.json), [replay script](../../outputs/analysis/b0prime-20261015/terminal/audit_failed_rehearsal.py), [log](../../outputs/analysis/b0prime-20261015/terminal/audit.log).
- [Actual failed-receipt entry rejection](../../outputs/analysis/b0prime-20261015/terminal/entry-gate-rejection.json): builder rejects, destination remains absent.
- Original result root: `outputs/kaggle/b0prime-drfull-k62-qual-20261015-r2-result/s6-recovery-entry/`. Preserve receipts, assignment, CSV, four NPZ files and logs unchanged. Collection was already present; Codex launched no collector.

## Runtime interpretation and next action

Original B0 projected 31,277.545 seconds / 8.69h. In this fresh session, slower-shard production increased **8.94% for dr-full K62 versus original Dread** and **7.46% for unchanged S6**. Phase overhead also increased slightly. The unchanged S6 slowdown supports session-speed variation as a contributor; it does not establish the cause or isolate a checkpoint effect. Historical substitution cannot qualify the current failure. [Phase comparison](../../outputs/analysis/b0prime-20261015/terminal/phase-comparison.json).

No production repair or repeated GPU run is indicated by this audit. Claude owns any next proposal after reviewing the exact failure and runtime risk. The existing entry builder remains fail-closed. No implicit projection waiver, K reduction, retry or competitive submission is authorized by this result. Any separately authorized exception must preserve the failed evidence and distinguish its estimated risk from the actual nine-hour platform limit.

A′ submission 57229227 remains officially **PENDING at 18:15:28 UTC**, with no score. [Snapshot](../../outputs/analysis/r224-integration-20261015/aprime-20261015-1815.json). Its separate v3 rescue lane remains suspended. Best A and final-selection rules remain unchanged. No new unit-test/full-suite repeat was needed because production code did not change; targeted saved replay and the actual entry rejection were exercised.
