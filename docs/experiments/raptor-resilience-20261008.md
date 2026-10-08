# Raptor production acceptance and separate decoder audit

Read-only reproduction against public snapshot `859aae12…` identified wrapper
rejections of usable inputs. Missing IOP, IPP or spacing produced 18 valid synthetic
frames in the public reader but failed the strict header reader. One unreadable
or decoded black frame produced 16 valid frames; a failed or empty selected
sagittal series plus a healthy coronal series produced 12. The prior wrapper
aborted these studies, including cases with exactly the public volume and mask.
This is a concrete robustness defect; the hidden Kaggle exception cause is still
unknown.

The new `raptor-public-partial-v2` production policy preserves the pinned public
functions and transforms only two exception handlers for fatal-resource propagation.
Public InstanceNumber
and spacing defaults, repeated short-series sampling, percentile normalization,
cropping, interpolation and masks are unchanged. Only successful pixel reads are
memoized within one study. No alternative series is tried, no model is retrained,
and no constant prediction or fp32 inference fallback is introduced.

Production validation retains shape/type/finite-pixel checks, exact mask equality
to nonzero image content, absent-slot emptiness and whole-study nonempty checks.
Masked frames and empty selected slots are accepted when the study remains usable.
The receipt records geometry/spacing defaults, header parse failures, invalid
header geometry for strict-audit eligibility, pixel errors, decoded unique images,
empty series, masked frames, partial slots and empty selected slots. These counts
are observations, not a requirement that all missing-data counts equal zero.

Temporary sequential reader guards use a private BaseException sentinel so that
MemoryError, TimeoutError and OpenCV allocation errors cannot be swallowed by
either public `except Exception`. A narrowly validated AST transform also inserts
a fatal-error check at the start of both catch bodies, covering geometry arithmetic
and list appends after reader callbacks return. Exactly one catch per function is
required; its transformed code hash and catch count are recorded in the manifest.
Removing the two inserted checks/name bindings recovers the exact original
scientific AST. The supplied public file bytes and their SHA stay unchanged.
Invalid nonfinite/non-2D pixels remain fatal.
Both the pydicom function and public reader factory are restored in `finally`.
The wrapper does not run parallel study readers.

The existing `reference_reader` and `validate_volume` remain strict. The rehearsal
compares every production volume/mask with an uncached public traversal. It also
runs the strict comparison on healthy inputs with complete geometry and selected
frames. Ineligible inputs are explicitly reported as `ineligible_partial_input`;
their strict decoder error is null, while public-source parity must still be
exact. Receipts report the strict comparison coverage separately.

## Requalification

The Raptor builder now defaults to exactly **64 ID-hash-selected training studies
plus three visible test studies**, with a hard **1,800-second** wall-clock cap.
No labels or gold membership are read for selection. It uses the pinned common
Python 3.12 / torch 2.10.0+cu128 / timm 1.0.26 image, the existing compatible cp312
codec attachment and the same plain CC0 checkpoint. No cp313 wheel or old training
notebook attachment is used.

Sixteen synthetic production-path stress cases run first, using temporary
synthetic files and datasets: healthy, three missing-metadata variants, malformed
header, unreadable frame, black frame, unreadable/empty selected series with
another valid plane, repeated short series, empty-study failure, fatal header and
pixel resource errors, geometry-arithmetic and append-region resource failures,
and invalid-pixel failure. Source-versus-production input
parity and function restoration are checked. No raw competition file is mutated.

The policy version and actual module SHA are bound into both the builder contract
and receipt. Old Raptor receipts are rejected. A fresh shared-environment blend
rehearsal is also required because its runtime qualification fingerprint changes;
it must report the new policy, exact raw reread pixel/mask parity and strict-audit
coverage. The blend formula and component weights are unchanged.

```bash
python scripts/build_reference_rehearsal.py PUBLIC_RAPTOR_SOURCE NEW_DEST SLUG \
  --profile raptor \
  --checkpoint-sha256 89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded
```

This task launches no job. Local tests exercise pinned source functions with
synthetic pixels; local OpenCV is unavailable, so both paths share the same resize
stub there. Remote stress checks use actual OpenCV. Actual visible probability
parity, expanded real-image qualification and hidden-run resolution remain
pending. Parent task owns STATUS and scheduling.

Local checks: **142/142 full ML tests passed** with all optional audited source
snapshots enabled. The dependency-free interpreter ran 142 checks with 76 optional
skips. Both default smoke commands and diff whitespace checks passed. Six changed
runtime sources compile, and the real pinned public source packages into a
284,227-byte private offline notebook without executing it. No CUDA forward pass
or real-image decoding was performed during this implementation task.
Independent review passed six Raptor and 23 reference/blend checks. It reproduced
fatal geometry errors and injected MemoryError/TimeoutError at the actual compiled
append line; both propagated with reader globals restored. No blocker remained.
