# B0 Dread substitution: integration and submission

The fixed 70:30 recovery entry replaces its nartaa/OAI Raptor arm with Dread's CC0 v4 Raptor checkpoint under the separately qualified K62 input contract. S6 and the recovery cap retain the qualified default code. This is a no-OAI candidate; it is not evidence of a top-ten improvement.

Claude completed the offline rehearsal `ozgung/knee-s6-dread-b0-rehearsal-20261013` version1 and submitted the ordinary entry `ozgung/knee-s6-dread-b0-entry-20261013` version1 once as **57119585**, at 2026-10-12 21:47:48 UTC (October13 00:47 TRT). SDK independently confirms PENDING at22:12UTC; the best public score remains0.954 (57096002).

Rehearsal notebook SHA256: `9de1fcc114416f773f57d88d62da1b6396f5e1896737838a146d5f46bc216851`. Entry notebook SHA256: `c4dfebc6d5176172810195c20d4d3d553b2d78c4f439de22ec4061f41af69736`. The failed51-character-slug push is preserved; no training or duplicate submission follows from integration.

Claude's terminal audit reports123PASS/0FAIL: all39 fixture predictions are genuine, S6 probabilities/logits are byte-identical to the earlier qualified rehearsal, Dread visible reader counts match its K62 reference, and the fixed blend oracle is exact. The ordinary entry's visible CSV equals the rehearsal CSV. Parent review checked the integrated five source files against Claude's worktree, both immutable build receipts, notebook hashes and packaged production source hashes; all match. The original qualified default modules remain byte-identical.

Parent focused integration verification: **20 tests PASS, no errors or skips**, covering default build byte identity, B0 source/profile guards, partial-input handling, failure classification, schema, entry/rehearsal separation and builder gates. Real checkpoint CPU/T4 behavior relies on Claude's existing matched-environment receipts; the parent did not repeat CUDA inference. Parent environment uses Torch2.14, distinct from Kaggle's2.10. This is not a full-suite pass; the previously documented notebook-size failures remain outside this integration.

The39-fixture runtime projection is6.35h base and8.69h conservative against the9h limit. These are estimates, not population bounds. One larger sensitivity exceeds9h. Healthy visible mechanics do not establish hidden-run completion or generalization. Hardcoded cleanup flags are not measured deletion evidence; actual typed failure, state and next-healthy checks are separately exercised.

Evidence: ignored `outputs/analysis/b0-dread-swap-20261012/parent-integration-acceptance.json`, `parent-integration-tests.log`, `audit-b0-rehearsal-receipt.json`, push/submit logs and the exact builds/results under `outputs/kaggle/`. The implementation was uncommitted at launch with exact hashes bound; parent integration now preserves those bytes. Next: Claude monitors57119585; retain the existing best entry while the next controlled training protocol is prepared.
