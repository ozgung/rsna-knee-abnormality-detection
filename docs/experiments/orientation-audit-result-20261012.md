# Source96 DICOM orientation audit result — 2026-10-12

The header-only audit completed successfully. The universal mirror assumption did **not** pass the frozen 99% per-plane descriptive criterion. Keep all study membership and use the prospectively declared per-study bypass for a separately identified, header-gated augmentation evaluation; no threshold relaxation or accuracy claim.

- Job: `ozgung/knee-orientation-audit-v2-20261012`, version1 / kernel138365986. Private, offline, CPU only. Accepted notebook matches the reviewed source, SHA92fb4c037b84a39f1725592acb26e3c16e47f5d172531de77cf66cd4199c92a6; contract99ad997c8e0daf288cd70f33dfb2753d36615e01f54a554869f8173b1ac72d97.
- Train: all 4,407 studies /20,792 selected series /677,292 headers. This is a superset of the 4,345-study training cache population. No labels, reports or pixels read.
- Visible test: all 3 studies /13 selected series /373 headers; all 3 route mirror_ok. This is only a smoke population.
- Runtime:1,235.846seconds (~20.6minutes), including the visible-test pass. Reported parent-process peak RSS167,120,896bytes excludes worker memory.

| Population | Eligible for preset approximate L/R transform | Plain bypass | Different transform |
| --- | ---: | ---: | ---: |
| All training studies | 4,128 (93.669%) |279 (6.331%)|0|
| Eligible cache studies |4,067/4,345|278|0|
| Fold-0 development |805/870|65|0|

| Plane | Studies with all selected series standard | Fraction | Frozen99% descriptor |
| --- | ---: | ---: | --- |
| Sagittal |4,207/4,407|95.462%|Failed|
| Coronal |4,282/4,407|97.164%|Failed|
| Axial |4,326/4,407|98.162%|Failed|

All selected series have the expected dominant mirror axis and consistent sign (sagittal normal R; coronal/axial row L). No missing/invalid headers, fallback ordering, tied/degenerate order or within-series IOP inconsistency was reported. The mirror-vector obliquity exclusions are 391 sagittal,211 coronal and81 axial series, combining to 279 distinct studies. The angular threshold is approximate alignment, not proof of an exact anatomical reflection or model label invariance.

## Decision and next checks

Preserve the fixed 10-epoch training and K16 primary export. First reproduce saved K16 probabilities on the selected checkpoint, then compare frozen K94plain. A new header-gated TTA arm may use the declared transform only for mirror_ok studies; bypass_plain and needs_transform retain identical plain predictions. All development studies stay in the evaluation, with additional eligible-subset metrics and support counts reported. The historical all-study mirror arm remains unqualified; no public-score forecast follows from geometry.

Claude owns preparation of the strict route consumer and plain-only mode in the existing secondary-inference tooling. Parent owns output acceptance, checkpoint qualification and execution; no new GPU job launched for this result.

## Evidence

Protocol: [frozen audit plan](orientation-audit-plan-20261012.md). Exact collector/source receipts: `outputs/analysis/orientation-audit-20261012/{launch-v1.json,terminal-collection-v1.json,accepted-notebook-v1.ipynb}`. Aggregate and private routing files: `outputs/kaggle/orientation-audit-v2-20261012-result-v1/orientation_audit/`. Private routing identifiers remain excluded from Git and prose. Source tests: 46 PASS / 0 skips; independent terminal validation **90 checks PASS / 0 failures**, including source/contract/job binding, artifact hashes, real CSV membership, eligible cache membership, fold-0 development coverage, routing hashes and denominator consistency. Receipt: `outputs/analysis/orientation-audit-20261012/independent-terminal-review-v1.json`.
