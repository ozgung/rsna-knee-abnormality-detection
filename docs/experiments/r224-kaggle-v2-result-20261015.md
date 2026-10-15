# R224 fresh Kaggle qualification — failed runtime gate

Updated 2026-10-15 10:00 UTC. Recipe `ws2-a-prime-r224-v2`; actual kernel `ozgung/a-prime-r224-rehearsal`, server version 2. Claude owns execution and collection. Accepted notebook and settings match the sealed qualification package; the metadata-only push2 transport copy leaves its source and checker binding unchanged.

**Qualification FAILED. No entry is authorized by this result.** The fresh conservative projection is 30,936.801 s (8.5936 h), exceeding the unchanged 29,880 s (8.3 h) gate by 1,056.801 s (17.61 min). Actual bounded rehearsal elapsed 1,985.627 s; actual elapsed time below nine hours does not establish a full-population runtime pass.

| Measured projection component | Seconds |
| --- | ---: |
| Frozen baseline, fresh session | 28,296.680 |
| Additional R224 production, conservative | 2,552.284 |
| R224 fixed overhead | 87.824 |
| Final merge | 0.014 |
| Combined | 30,936.801 |

Parent ran the unmodified full checker: it rejects `FRESH_FULL_REHEARSAL_BINDING` because the saved terminal receipt is FAIL. The actual runtime failure artifact identifies `COMBINED_PROJECTION_AND_ACTUAL_LIMIT`. No receipt, source, guard or gate was changed to obtain acceptance.

A separate **saved-mechanics diagnostic passed**, with no qualification claim: exact preparation/source binding; frozen baseline verifier; all six worker ledgers recomputed; each arm covers 39 studies with no failures; native checkpoint/immutable states/environment/physical boards; all 39 guarded volume/mask bindings; eight finite, nonnegative CPU/CUDA crop statistics and strict reload/cross-board probability sentinels; unchanged baseline ranks and hashes; three-row finite/schema-correct CSV recomputed; exact saved projection and resource replay. Native workers completed 20+19, no fp32 recovery. This is saved real-execution evidence, not a local raw-image/model rerun or competitive evaluation.

Evidence (ignored/private inputs remain uncommitted): [reproducible diagnostic](../../outputs/analysis/r224-integration-20261015/audit_kaggle_v2_failure.py), [diagnostic receipt](../../outputs/analysis/r224-integration-20261015/kaggle-v2-failure-audit.json), [full checker log](../../outputs/analysis/r224-integration-20261015/kaggle-v2-frozen-checker.log), [accepted source](../../outputs/analysis/r224-integration-20261015/accepted-kernel-v2-verification.json). Full collected root: `outputs/kaggle/ws2-r224-v2-qual-20261015-result/ws2-a-prime/`.

## Timing diagnosis and next decision

Compared with archived A v7, the same baseline's slowest old-arm production rose 110.870→116.319 s (+4.92%); R7 rose 446.987→479.062 s (+7.18%). Board 1 R7 stayed near 422.019→421.132 s while board 0 slowed. Fixed old/R7 overhead rose only about 20 s combined. The baseline projection increased 1,743.664 s. These are observations of session/board variation; they do not establish its cause or future runtime. [Aggregate comparison](../../outputs/analysis/r224-integration-20261015/kaggle-v2-baseline-timing-comparison.json).

Using the archived baseline would give **29,193.138 s (8.1092 h)**, including R224 fixed overhead and final merge. This corrects the earlier handoff's 29,105 s illustration. It remains a historical substitution, not qualification evidence. Do not replace the fresh baseline, relax the gate retrospectively or build an entry from this failure.

Claude's next task is a concise headroom proposal using these measured phase/board rates and actual remaining quota: justify a fresh unchanged-source qualification or propose a separately bound optimization and its acceptance checks; otherwise close A′. No automatic repeat loop. Existing user authorization covers justified experiments; no new approval checkpoint is inferred. Any prospective plan change must be recorded before execution, and entry building still requires genuine fresh acceptance.

Colab v2b was lost mid-A′ with no collected evidence. Owner reports token removed before inference, CLI 404/state cleared; no timing, ratio or parity verdict. No further Colab attempt is planned. Lost session 2647d8 cannot be stopped through the now-empty CLI config; owner/user can end it in Colab UI or let it expire. No session authentication reconstruction or duplicate job.
