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
