# Global identity audit and sealed weak-proxy reservation

Implemented for future training; **CPU64 pilot and full audit PASS**. This changes neither the completed fresh fit nor its original split. The audit addresses missing patient evidence outside the original selected studies before any larger-data training. It does not establish independent label provenance or authorize sealed-cohort evaluation.

The private CPU notebook attaches only the competition, uses the existing pinned Docker image, and needs stdlib plus preinstalled pydicom. GPU and internet are disabled. It reads the three hash-pinned competition CSVs and the lexicographically first `.dcm` header from **every metadata series** using `stop_before_pixels=True, specific_tags=['PatientID']`. No pixel decoding/copying, report output, labels, decoder wheels or model assets are included. Missing/unreadable headers, empty/nonscalar PatientID, schema/coverage discrepancies and changed contracts fail qualification and withhold the sealed manifest. Pydicom warning text is suppressed; only its count is retained. Failure receipts contain fixed codes/types and aggregate counts, without raw exception text or tracebacks.

The builder verifies the original feature-cache SHA, original patient-hash artifact SHA, all 2,358 ordered ID/group/role records, competition metadata hashes and the reviewed fresh plan. The original roles remain 2,000 train, 300 development and 58 gold. Historical fresh `baseline_fingerprint` `19683d3463f0a299abaaafc92ef4fd88d72b25017e6a22ad085cecc96f16d09f` includes labels added to 712 shared baseline dictionaries before hashing. It is reproduced **locally only** from existing artifacts, with no changes to that plan. The new clean ordered ID/split/group digest is `47697ebc8045d178f44488f4c4c6004aed765d8bdb7b4b4d3d409a7ac2da6a82`. No label vectors are embedded in the audit notebook or identity contract.

After full coverage of 4,407 studies / 24,371 series, the audit unions exact normalized-report hashes and **all observed patient hashes**, including the original selected-series evidence, transitively. Patient normalization is the original `strip()` plus `knee-patient-v1:` SHA256 convention; report normalization is the existing NFKC/casefold/whitespace convention. Multiple nonempty patient hashes within a study are counted and retained as edges. Components touching gold or **any** original 2,300 train/development rows cannot enter the sealed cohort. Remaining components are sorted by SHA256 of `fresh-sealed-v1:20261008:` plus the canonical sorted-membership fingerprint. The shortest whole-component prefix reaching 300 studies is reserved, with overshoot recorded. The manifest preserves `original_split` and separately marks future training candidates or quarantine for links to development/gold/sealed components. No target value enters this selection. Future training must consume the validated full manifest; it must not repurpose the other original development rows.

The pilot samples 32 original nongold cohort studies and 32 outside that cohort by fixed salted ID hashes, then reads **all** their metadata series. Its 600-second hard budget includes setup. Full execution requires a successful, source-identical pilot and a projection of `pilot nonheader overhead + 1.5 × header/listing seconds × total series / measured series <= 3600 seconds`; full execution also has a 3,600-second deadline. Both use POSIX deadlines plus explicit per-header deadline checks. Startup and stratum study/series counts are recorded. No DICOM read-byte estimate is fabricated. Persistent JSON outputs are capped at 32 MiB; the local 2 GiB image-cache allowance is unchanged.

Pilot reuse validates exact sampled-series coverage, identity/receipt fingerprints, serialized identity SHA, metadata/original contract and runtime-source hashes before accepting cached header identities. Full builds embed that small reviewed private bundle so no extra attachment is needed. Both final notebook and joined code must fit the known1,000,000-byte Kaggle source ceiling or the build fails before writing files. A changed runtime requires a new pilot. This is a semantic header cache bound to the immutable competition attachment and CSV hashes, not a checksum of every raw DICOM byte. One header per series cannot establish within-series identity consistency, anonymized IDs may miss repeat patients, and normalized reports miss paraphrases.

Build the CPU pilot from the project root (build only):

```bash
.venv/bin/python scripts/build_sealed_identity.py \
  data/raw outputs/experiments/dino-frozen-20261008/artifacts \
  outputs/kaggle/fresh-cache-result/fresh_reader_cache/plan-private.json \
  outputs/kaggle/sealed-identity-pilot knee-sealed-identity-pilot-20261008 --stage pilot
```

After parent review of `sealed_identity/receipt.json` and a passing time gate, build a separate full notebook with the same inputs, a fresh destination/slug, `--stage full --pilot-dir PATH/TO/sealed_identity`. Never launch full execution on a failed gate. Both generated notebooks and all outputs must stay private and ignored. A full PASS emits `identity-private.json`, `sealed-manifest-private.json` and aggregate `receipt.json`; a pilot emits only identity data and its receipt. Any audit failure emits only an aggregate failure receipt in a fresh output directory. Existing cohort manifests are not overwritten.

Verification:12 focused tests pass, covering transitive report/patient crosslinks, immutable original roles, whole-component overshoot, label/order independence, exact metadata/header coverage, source-bound pilot reuse, privacy including parser warnings, expired/runtime/output budgets and tamper rejection. A real pydicom fixture verifies only PatientID is retained and PixelData is absent. The full ML suite passes158/158; dependency-free158 checks pass with77 optional skips, and default smoke/diff checks pass. The private pilot builder was exercised against the original local metadata/features/patient evidence/plan and its notebook compiled. These are local checks, not measured remote throughput or a sealed real cohort.

## CPU pilot and full launch,03:07UTC

Pilot `ozgung/knee-sealed-identity-pilot-20261008` v1 (sourceca7f674) PASS:
64studies/353series (32original170series;32outside183series), no parser warnings,
header/listing11.189728s, total12.897767s. Py3.12.13/pydicom3.0.2. Conservative full
projection1160.510602s including1.5margin and measured overhead1.708038s, below
3600s. Receipt fingerprint`0c667564945292ececbda57c20c4ffa2b79d3f81d21ab4994ebf7219a915d6b8`;
identitySHA`116326e7e9e5c7566cf656fd97b376eb7885c2378b7b126794bf3dfa7585c93b`.
Builder revalidated original metadata, exactpilot coverage/source/identity/receipt
and expansion gate before full launch. Pilot outputs are under
`outputs/kaggle/sealed-identity-pilot-result/sealed_identity/`.

Full `ozgung/knee-sealed-identity-full-20261008` v1 launched03:07UTC fromsource7e0f305,
internal3600/external3900s, CPU/nointernet/competitiononly. Notebook827,461bytes,
SHA`364de14e127472696a9846439f01cd0ba8507b51490bcce3e60e8adec6d381c0`.
Build directory `outputs/kaggle/sealed-identity-full/`. Runtimecontract
`63cc0d81da737a38a570e432fb56bc698cf4b3528065fd783e72f312084d1167` unchanged.
No real sealed manifest has been accepted yet; parent must verify fullPASS, exact
24,371-series/4,407-study coverage, transitive role separation and manifest equality.

## Full audit accepted after interruption,12:59UTC

All4,407studies/24,371series passed in526.300875s:24,018new header reads and353
hash-verified pilot rows, zero parser warnings. Private identity+manifest bytes
9,721,238, under32MiB. Exact local reconstruction of the source-pinned identity and
manifest, receipt/hash verification, original roles and component separation PASS.
Future roles:3,745training candidates,300development,300sealed,58gold,4quarantined
gold relatives. No original training study newly quarantined; the current512/200
roles are unchanged. Sealed selection reaches300exactly, no overshoot, within4,257
transitive components. All4,407patient hashes are unique; anonymization and one
representative header per series remain identity limitations. No model predictions
or target values were used to choose or evaluate the sealed cohort.

Frozen manifest fingerprint:
`911f1605497f5c181402e3fbf70c26d588ffb9c73d6cb2d67e45f22eec52245d`;
fileSHA`21c71745703d742585684298494a6269ae4217fadc5fc4c26f3bf0d10a289467`.
ReceiptSHA`874a94583cfd396bf44b6b15b9a6e29397c1f3014bf4f79884f3060aa4fef249`.
Private outputs: `outputs/kaggle/sealed-identity-full-result/sealed_identity/`.
Aggregate acceptance: `outputs/analysis/sealed-identity/verification.json`.
