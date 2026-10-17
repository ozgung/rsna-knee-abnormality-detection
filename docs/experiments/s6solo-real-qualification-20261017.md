# Matched solo GPU qualification and entry preparation — October 17

Recorded October 17, 01:43 TRT / October 16, 22:43 UTC. Both exact checkpoint/package variants pass independent replay of the frozen 39-study private/offline rehearsal on two physical Tesla T4 boards. This qualifies the recorded image-inference path and package mechanics on that fixture. It is not a predictive-performance, hidden-population or generalization result. The prospective matched competitive comparison remains unresolved.

The intervention is the V2-all checkpoint versus public S6fall with the same captured solo controller, preprocessing, full-population A recovery arm and normal A/B contribution 0/1. No training or preprocessing fit occurs here; label masks, image-only inference, official target order and grouping safeguards remain. The fixture contains 36 training studies and three public test studies, with no labels or reports read. Fixture configuration SHA256: **8941f363ab6d21fdf99c78532c9db2a8b37794417785a842e35336d4eca10695**. [Frozen scope](s6-v2all-solo-scope-20261016.md), [actual checkpoint/training exposure](s6v2-final-acceptance-20261016.md), [sealed preparation](s6solo-pair-preparation-20261016.md).

Claude owns kernel pushes and collection. It reports version 1 for both rehearsals, pushed October 16 at 21:55:24/27 UTC; official latest endpoints are COMPLETE at 22:40. Latest executable/bootstrap/settings match sealed packets; explicit immutable-version source retrieval failed and the API omits attachment version suffixes. Runtime model/code/receipt full hashes pass, but immutable server-version provenance remains separately requested.

| Arm | Actual elapsed | Conservative 1,300-study estimate | Genuine fixture rows | Raw receipt SHA256 |
| --- | --- | --- | --- | --- |
| V2-all | 473.7 s | 9951.6145 s (2.764 h) | 39 | 200a0e9163003360e25813e02d203446293cf9ac1dd43cef3c65f3ad27476d71 |
| S6fall control | 471.7 s | 9796.5224 s (2.721 h) | 39 | 51ffaf28300aca64762ae7a2a459a07e491bbb9c2422c5e068b98190b21cf5cb |

Every original qualification gate is independently recomputed and passes: actual per-board full state/buffers/constructor and A grid match their pins before/after inference, strict same-input reload is identical, both components genuinely score all 39 without fallback/retry, sentinel pixel/meta hashes agree, ledger and all-population ranks/official CSV replay exactly, and resource/timing gates pass. Both ordinary 8.3 h projection and actual elapsed <9 h pass without a waiver. The estimate retains observer cost and the original conservative work/imbalance factors; it is not a hidden-runtime guarantee.

Actual environment: Python 3.13.15, torch 2.11.0+cu128, timm 1.0.29, NumPy 2.1.3, pydicom 3.0.2, captured Docker and two T4s. Sampled GPU peaks are at most 6,909 MiB per board; host peaks are 14,914,719,744 bytes (V2) and 14,670,778,368 bytes (control), versus 33,658,318,848 total. Working-disk free minima exceed 20.93 GB. Sampling is not an absolute peak bound. State equality against CPU/historical reference pins is freshly measured on these GPUs; historical environment qualification is not inherited.

## Host replay portability correction

Initial filtered collections omitted component timing logs and sentinel input-hash JSONL. Claude fetched the six logs and eight JSONL files without rewriting source outputs; all required genuine evidence is now present. The original verifier also required exact equality of derived probability-difference diagnostics across NumPy versions; actual local recomputation differs from saved zero by at most 5.55e-17.

Commit **50e4bc4** adds a host-only verifier and routes entry construction through it. The sealed producer and all rehearsal/inference sources remain byte-identical. Every original scientific gate must independently PASS, artifacts/ledger/other fields remain exact, and only the derived sentinel max_abs_probability comparison permits finite absolute roundoff <=1e-12. The scientific 1e-3 gate is unchanged and independently enforced. Original receipt bytes/hashes are preserved; each entry binds the host policy/source hash and actual roundoff differences.

Parent 43 focused tests PASS, including exact replay, legitimate diagnostic roundoff, rejection above roundoff even below the scientific tolerance, unchanged exact diagnostics, missing input evidence and altered artifact bindings. Bare optional replay checks skip all six without ML dependencies; dependency-free smoke PASS. No inference method or model changed.

[Independent real replay](../../outputs/analysis/s6solo-pair-20261016/real-acceptance/acceptance.json), [replay code](../../outputs/analysis/s6solo-pair-20261016/real-acceptance/accept_results.py), [43 tests](../../outputs/analysis/s6solo-pair-20261016/real-acceptance/host-replay-tests.log), [entry acceptance](../../outputs/analysis/s6solo-pair-20261016/real-acceptance/entry-acceptance.json).

## Entry artifacts

Built with committed host verifier 50e4bc4. Both entries replay exactly and import from fresh isolated emitted sources. Their four inference sources are identical across arms and to the sealed rehearsals; entry mode excludes fixture and observer sources.

| Arm / root under outputs/kaggle | Notebook SHA256 | Payload SHA256 |
| --- | --- | --- |
| s6solo-v2-entry-20261016 | ed7e2c21197041961a39cd6040c4f964a9444f087fb181973f62453b8ab47f5a | 9dd406adecdc2e22e4aac54bffab1d0057ea11246d0070d6bbd808ad82429f55 |
| s6solo-s6fall-entry-20261016 | 72484024c02463d1bebf2c95a14ce7d81efefe7c766cac3cbb1d9880cd6ce37f | 6410ccf12b9c9b1525b0565a164eeedf4dbe09b16342cb18c9c4e31cbe5482f0 |

Claude retains launch, visible-result checks, collection and submission ownership. Reconcile actual quota/slots and existing human authorization before competitive execution; preparation creates no additional submission authorization. Record immutable entry versions/source and both displayed scores/fallback counts. Candidate >= control remains the frozen exposed-development heuristic; failed or undefined control is unresolved. Gold-58 selection exposure, teacher/pretrained exposure and the original fold-0 BLOCKED verdict are unchanged. No campaign entry was pushed or submitted by Codex, and no score or final-selection action is claimed.

A‴ r3 is a separate lane: official latest endpoint ERROR at 22:40, parent independently accepts its collected non-projection contracts and actual 1778.4 s, with only a 63.477 s projection excess. Original FAIL remains; a separately named receipt-bound waiver/entry implementation is with Claude under its reported user condition.


## Visible acceptance and pending competitive pair — Oct 17, 01:48 TRT / Oct 16, 22:48 UTC

Claude reports both entry kernels version 1 COMPLETE; parent accepts the collected visible trees at outputs/kaggle/s6solo-{v2,s6fall}-entry-20261016-visible/. Both PASS in entry mode, match the exact payload/rehearsal/model hashes, have three official studies and 13 columns, finite [0,1] probabilities, exact complete-population rank/CSV replay and zero A/B fallbacks. Elapsed times are 168.0 s (V2) and 172.9 s (control). The CSVs differ, demonstrating effective checkpoint binding without implying predictive superiority. Raw visible receipt hashes: df28bea683c8d49a19ac2a03bff0fe49fe6e2cba0620feb06df5468a1d1641b6 / aa6b66ce6ce57cf369ddba6d4e4c3bd0c92fc0e582d6555ad2ac15f85db38ad9.

Claude submitted V2 **57285381** and matched control **57285382** at October 16, 22:45:04/07 UTC (Oct 17, 01:45 TRT), from owner-reported version 1. Official submissions CSV confirms both **PENDING**, no displayed scores. Existing scope requires V2 >= control; failure or undefined control leaves comparison unresolved. No promotion, final selection, per-study significance or causal-label/generalization claim is made. Claude reports three slots remaining; that is owner account state, not a new parent reservation or authorization.

[Visible acceptance](../../outputs/analysis/s6solo-pair-20261016/competitive/visible-acceptance.json), [official snapshot](../../outputs/analysis/s6solo-pair-20261016/competitive/official-submissions-2247.csv), [read-only visible replay](../../outputs/analysis/s6solo-pair-20261016/competitive/accept_visible.py). Claude retains terminal/fallback/resource collection; parent has not launched or submitted anything.


### 2026-10-17 01:07 UTC — official matched competitive result: displayed tie

[Official submissions snapshot](../../outputs/analysis/s6solo-pair-20261016/competitive/official-submissions-20261017-0107.csv), [compact result receipt](../../outputs/analysis/na-full-aprime-rehearsal-20261016/r3-real/competitive-state-20261017-0107.json).

| Arm | Submission | Official status | Displayed public score |
| --- | --- | --- | --- |
| V2 all-data | 57285381 | COMPLETE | 0.950 |
| S6fall control | 57285382 | COMPLETE | 0.950 |

Both are terminal, so the registered displayed(V2) >= displayed(control) comparison resolves TRUE by a displayed tie. V2 is eligible under that compact-slot heuristic; there is no displayed improvement. Unrounded scores, per-target support/AUC, paired uncertainty and hidden fallback/resource details are unavailable in this CSV. Do not interpret the tie as equivalence, a significant difference, labels-specific causation or clean generalization. Gold58 remains selection-exposed; original fold0 BLOCKED stays unchanged. Qualification and three-row visible zero-fallback evidence remain separate from hidden-result diagnostics. The completed competitive scores do not independently certify fallback/resource behavior; Claude owns preservation of any available terminal evidence, or an explicit unavailable record. No new entry, ensemble, threshold tuning or final-selection action is implied.

A‴ 57288634 (submitted owner v1 at00:25:07 UTC, independently present in official snapshot) is still PENDING with no score. Its named projection waiver and original r3 FAIL remain. A′0.956 remains final-1 candidate under the separate A‴ comparison rule. No parent submission or duplicate collector.
