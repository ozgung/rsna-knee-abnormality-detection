# ConvNeXt input resilience and renewed qualification

The standalone ConvNeXt and fixed blend hidden reruns failed with Kaggle's generic
“Notebook Threw Exception”. The hidden traceback is unavailable; the exact cause
remains unknown. Synthetic reproduction identified a concrete shared-wrapper
defect: it rejected an otherwise usable series whenever the public loader reported
even one unreadable directory item. The inspected public inference instead uses
the remaining volume when it has at least three slices.

The shared wrapper now follows the [public author's inference](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944)
for usable partial volumes and for selected series that fail or contain fewer
than three slices. An unusable selected series leaves its slot masked; other
valid slots run. The wrapper never retries an alternative candidate. Public
preprocessing/model source hashes, checkpoint identities, window sampling,
position calculation and three-fold probability mean are unchanged.

`decode_failures` counts unreadable directory items, so it may now be positive on
a successful run. `partial_series`, `failed_series`, `short_series`,
`zero_decoded_series`, `missing_slots`, and sanitized exception-type counts make
the behavior observable. There is no arbitrary tolerated-error percentage. An
entirely empty or zero-content study still fails explicitly with aggregate slot
diagnostics; no constant-prediction or learned all-missing-head extension was
introduced. A zero-content slot with another usable slot retains the public
source's mask semantics.

MemoryError, TimeoutError and OpenCV StsNoMem remain fatal. A temporary `_read`
guard uses a private BaseException sentinel to bypass the public loader's broad
`except Exception`, then re-raises the original fatal error. The loader is restored
in `finally`. KeyboardInterrupt and SystemExit propagate naturally. This wrapper
is sequential; no global reader patch is shared across worker threads.

## Qualification before another entry

The standalone rehearsal first exercises the production preparation functions with
synthetic volumes: healthy/missing planes, one unreadable item with a usable
volume, selected-series exception, short selected series, all-empty hard failure,
and caught memory/timeout failures. It restores both loader functions and audit
counters before reading data. No raw image mutation or label access is involved.

The real sample expands from 16 to **64 training studies**, selected by the same
fixed ID hash, plus the three visible test studies. The internal rehearsal bound
is **3,500 seconds**. This is a budget, not a measured result. Repeat inference,
serialized-input parity, resource measurements and runtime projection remain in
the receipt. Local synthetic success is not qualification for remote inference.

`convnext-partial-series-v2` and the exact entry SHA bind the new policy. Both
standalone submission packaging and blend component qualification require the
new 67-study receipt and successful stress checks. Existing receipts cannot
qualify the changed wrapper. The fixed blend must also repeat its shared-environment
visible rehearsal; its runtime fingerprint includes the new qualification code.
Standalone full-entry packaging now requires:

```bash
python scripts/build_public_reader.py SOURCE_DIR NEW_DEST SLUG --submission \
  --weight-hashes NEW_REHEARSAL/reader-weight-hashes.json \
  --rehearsal-report NEW_REHEARSAL/reader-audit.json
```

No Raptor preprocessing, metadata schema, target order, blend formula, training
recipe or selection rule changed. No job or competition entry was launched by
this implementation task.

## Local verification

Focused reader and blend checks: **24/24 PASS** with the ignored public snapshots
available. They include execution of the pinned public `load_series` function
with synthetic `_read` results, fatal inner-reader failures, exact public Dataset
array/mask/position parity for healthy and missing-plane inputs, and rejection of
the old real qualification receipts. Packaging checks compile private offline
notebooks; synthetic receipt fixtures are explicitly not real rehearsal evidence.
Independent read-only review repeated these 24 checks and found no blocker. The
full ML suite passed **135/135**, with optional source snapshots enabled. The
dependency-free interpreter ran 135 checks with 71 optional skips; both default
smoke runs passed. `git diff --check` passed. One initial full-suite invocation
used a nonexistent optional source path; correcting that test-only environment
variable produced the complete pass above.

Next: integrate the reviewed commit, run the expanded standalone offline rehearsal,
inspect aggregate input/resource diagnostics, then requalify the fixed blend if
the standalone run passes. The hidden-run cause remains unconfirmed until actual
execution evidence resolves it.
