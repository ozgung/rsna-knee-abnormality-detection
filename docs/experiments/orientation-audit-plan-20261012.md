# Label-blind DICOM orientation audit and per-study mirror routing (2026-10-12)

Status: reviewed and tested on synthetic data; parent accepted one CPU launch at17:45UTC. **Completed: version1 / kernel138365986; independent terminal integrity 90 checks PASS.** Real selected-series headers have now been audited; the universal mirror assumption remains **unqualified** because every plane misses the frozen 99% descriptor. See [result and routing decision](orientation-audit-result-20261012.md). Code: `knee/orientation_audit.py`; tests: `tests/test_orientation_audit.py`; builder: `scripts/build_orientation_audit.py`; build: `outputs/kaggle/orientation-audit-v2-20261012/` (kernel `ozgung/knee-orientation-audit-v2-20261012`). The first build `outputs/kaggle/orientation-audit-20261012/` is superseded by v2 and left unchanged; never push it.

v2 corrections after review: (1) degenerate, unavailable or ambiguous (tied) slice order can no longer be `mirror_ok`; (2) slices are enumerated, read and keyed exactly as the pinned reader does, including its fallbacks; (3) missing or invalid PixelSpacing is `bypass_plain`; (4) any unobserved (sampled) slice makes a series `bypass_plain`; (5) the builder requires all four cache-shard contracts.

## Purpose

`knee/source96_secondary_inference.py:54` (`MIRROR_AXIS`) assumes: Sagittal slices lie along patient L/R, so reversing slice order is an L/R mirror; Coronal and Axial image width (columns, the IOP row direction) is patient L/R, so a width flip is an L/R mirror. The pinned reader applies no transpose or reorientation. This audit checks the assumption from header metadata (`ImageOrientationPatient`, `ImagePositionPatient`, `PixelSpacing`, `InstanceNumber`) and assigns every study a routing category computed from headers alone. Until it is recorded, no `k94_mirror_tta` number is interpreted (protocol: `source96-secondary-inference-protocol-20261012.md`).

The cohort fractions are **descriptive**. A 99% cohort share cannot certify a fixed transform for any individual study, so the **per-study routing governs**.

## Series-selection policy reused (not re-implemented)

The audit executes the pinned code itself: `pinned_selector` builds the qualified wide96 reader through `source_span_pilot.arm_module(codes, 'wide96')`, and `select_series` repeats the `decode` slot loop around the pinned `_pick_series_for_slot`.

| What | Where |
|---|---|
| Slot layout `SLOTS96 = Sagittal/fluid 1 x26, Sagittal/fluid 0 x22, Coronal/fluid 1 x18, Coronal/fluid 0 x12, Axial/any x18` | `knee/source_span_pilot.py:18`; arm `wide96` at `:22`; cache config `knee/source96_cache.py:21` |
| Pinned public reader, SHA `859aae12...`; extracts `_make_reader`, `_pick_series_for_slot`, `build_study` | `knee/corpus44_parity.py:20`, `:25-45` (`recipe`), `:48-54` (`reader_module`) |
| Re-parameterisation by slots, depth 96, span 2-98% | `knee/source_span_pilot.py:31-45` (`recipes`), `:62-70` (`arm_module`) |
| Selection loop: per slot `_pick_series_for_slot(rows, plane, fluid, used)`, picked series added to `used` | `knee/source_span_pilot.py:140-144` (`decode`); pinned `build_study` lines 309-313 in `outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py` |
| `_pick_series_for_slot`: first unused row of the plane whose `Fluid_Sensitive` equals the slot preference, else first unused row of the plane (Axial: no preference) | same pinned file, lines 295-301 |
| Cache call and row order: `pilot.arm_module(codes, 'wide96')`, `pilot.decode(...)`, rows appended in `train_series.csv` file order | `knee/source96_cache.py:239`, `:244`, `:286-288` |
| Reader order: unsorted `glob(<series>/*.dcm)`, full `dcmread(stop_before_pixels=True)`, stable ascending sort on `dot(IPP, row x col)` (else `InstanceNumber`; position 0.0 if anything in the per-file `try` raises, including an unparsable PixelSpacing) | pinned file lines 259-277 (glob 260, key 268, sort 275); `reader_key`, `reader_order` and `read_series` repeat it, parity-tested against the pinned `order_and_meta` built via `arm_module` |
| Slices used: `linspace(int(n*.02), max(int(n*.98)-1, ...), k).round()` | pinned file `build_study`; `used_positions` repeats it (tested against `np.linspace`) |

Binding: the builder requires the pinned source SHA, the qualified pilot receipt `48655a7e...` (reader hashes `narrow e331776e...`, `wide d45be10f...`), and all four cache-shard build contracts (reuse the same reader hashes, slots, span and the same `train_series.csv` SHA). It builds the kernel only if they agree (4 of 4 verified). Slot names: `sag0` = Sagittal fluid-pref 1 (26 slices), `sag1` = Sagittal fluid-pref 0 (22), `cor0` = Coronal fluid-pref 1 (18), `cor1` = Coronal fluid-pref 0 (12), `ax` = Axial (18). A missing second sagittal or coronal series leaves the slot empty (zero-filled by the reader).

Population: all studies in `train_series.csv` (4,407 studies, 24,371 listed series, **20,792 selected**: sag0 4,407, sag1 4,288, cor0 4,407, cor1 3,283, ax 4,407), a superset of the 4,345-study cache population. Only `train_series.csv` is opened (series metadata columns); `train.csv` (targets) is never read. Visible test: `test_series.csv`, 3 studies, 13 selected series, run as a separate tiny split (smoke only, not cohort evidence).

## Computation and frozen parameters

Per selected series, from every instance header: row and column cosines from IOP (median over valid instances), normal = row x col. For each vector: dominant LPS axis, sign (L/R, P/A, S/I) and `|dominant component|`. Class: standard if `>= cos 20 deg (0.93969)`, oblique if `>= cos 45 deg (0.70711)`, else nonstandard. Mirror vector per plane: Sagittal normal, Coronal row, Axial row. Series mirror status (first match): `unverifiable` (no instance with valid IOP) > `fail_axis` (mirror vector not dominant on x) > `fail_nonstandard` (x dominant, class nonstandard) > `geometry_flagged` (x dominant, but any of: inconsistent IOP, mixed normal, missing/invalid IOP, IPP or PixelSpacing, unreadable header, a slice the reader files by fallback, a sampled series, or an order other than `ascending`) > `ok_oblique` > `ok_standard`. Order states: `ascending` (strictly along +normal, all reader keys distinct), `ambiguous` (equal reader keys: the reader's tie order is its file enumeration order), `degenerate` (fewer than two slices or no extent along the normal), `not_ascending`, `unavailable` (no usable IPP); only `ascending` can be `mirror_ok` or `needs_transform`.

Frozen before launch (recorded in the build contract and every output; a test pins the values):

| Parameter | Value |
|---|---|
| standard / oblique class | `>= cos 20 deg` / `>= cos 45 deg` of the dominant component (`1e-9` float guard) |
| IOP consistency tolerance | any of the 6 IOP components differs from the series median by more than **1e-3** (per slice) |
| mixed-normal tolerance | instance normal differs from the series normal by more than **5 deg** or has a different signed dominant label (per slice) |
| IOP validity | unit length and row.col orthogonality within **1e-2**, else invalid (counted as `iop_invalid`, not analysed) |
| ambiguity margin (reported flag on the normal) | top two `|components|` differ by less than **0.10** |
| order tolerance | **1e-3 mm** along the normal |

**Per-series routing** (headers only): `mirror_ok` = `ok_standard`. `needs_transform` = complete, consistent geometry, mirror vector not on x, but another vector is standard-class x (a candidate flip exists on width, height or slice order under the frozen angular tolerance, but is not the assumed one and is unqualified). `bypass_plain` = everything else (oblique, nonstandard, missing, invalid, inconsistent, unverifiable, no files). **Per-study routing**: any `bypass_plain` series gives `bypass_plain`; else any `needs_transform` gives `needs_transform`; else `mirror_ok` (a study with no present series is `bypass_plain`). Sign is reported (`mirror_vector_lr_sign`) but does not change routing.

**Frozen denominators.** Study: every study in the CSV (train 4,407; test 3), each with exactly one route. Plane: studies with at least one selected series in that plane. Slot: studies whose slot has a selected series (series that exist but have no readable `.dcm` stay in the denominator as `unverifiable`). Series: selected series. Slice: header files read; "used" slices are the reader's per-slot picks (26/22/18/12/18), reported separately from all slices.

## Outputs

Kaggle `/kaggle/working/orientation_audit/`:
- `train-orientation.json`, `test-orientation.json`: **aggregate counts only, no study or series identifiers.** Per plane **and per slot (sag0, sag1, cor0, cor1, ax)**: series; mirror status counts; series route counts and the image axis for `needs_transform`; signed (L/R/P/A/S/I) and unsigned dominant-axis counts for row, col and normal; class counts per vector and per series; flag counts; order-along-normal counts; L/R sign of the mirror vector; minimum dominant component. **Slice-level** counts per plane and slot: instances with IOP inconsistency, mixed normal, missing/invalid IOP or IPP, bad pixel spacing, unreadable header, "routing flagged"; used slices and used slices flagged; series with any inconsistent slice; worst series. Study level: per-plane study counts, the three-way study routing counts, denominators, `ordered_study_ids_sha256`, `routing_array_sha256`, and a `descriptive_summary`.
- `routing-private-train.json`, `routing-private-test.json`: **private**, positional arrays in series-CSV first-seen study order (`study_ids`, `routing`, per-slot `slot_status` and `slot_route`, `slot_names`). Retrieve only into ignored `outputs/`; never quote contents in docs or handoffs. Consumers must match by study ID, check that the aggregate's `ordered_study_ids_sha256` equals the hash of the file's IDs, and fail if any evaluation study lacks an entry.

## Bounded CPU plan

Private, offline, CPU-only; `pydicom.dcmread(path, stop_before_pixels=True)` exactly as the pinned reader (no `specific_tags`, so a header the reader cannot parse is unreadable here too); 8 forked workers; no pixels decoded; memory small (one chunk of 32 studies). The audit reads **every** instance header of each selected series: the cost is modest, and instance-level inconsistency and the reader's order need all of them. `--instance-cap N` remains only as a diagnostic: any series with unobserved slices is marked `sampled` and is `bypass_plain` (never certifying), and used-slice counts are disabled. The production build uses no cap.

Estimate (not measured for this code): the earlier header inventory (`outputs/kaggle/dicom-native-resolution-20261010-result/dicom_resolution/train-headers.json`) read 819,078 headers in 4,254.05 s (192.5 headers/s, 8 threads, no failures); the visible test took 2.89 s for 557 headers. Selected series are 20,792 of 24,371 (85.3%), so about **699k header reads**, assuming selected series have the average length (unverified), giving roughly **61 minutes** at the earlier rate. That rate was for a 7-tag parse; the full-header parse used here may cost a few times more per header (unmeasured), and a process pool may offset it (Kaggle core count not checked), so plan for one to a few hours, far inside the 12 h limit. The actual 12 h platform limit is enforced in the kernel. Progress is saved every 10 chunks; there is no resume.

Build: `python scripts/build_orientation_audit.py` (done; refuses to overwrite an existing directory; v2 contract hashes in `outputs/kaggle/orientation-audit-v2-20261012/build-receipt.json`). Executed once17:46UTC after parent commitdd4c767; do not repeat:

```
/Users/ozgun/miniconda3/bin/kaggle kernels push -p outputs/kaggle/orientation-audit-v2-20261012
```

## Acceptance criteria (declared before launch)

Governing, per case:
1. **Coverage:** `COMPLETE` for both splits, no `failure.json`; train audited series equals the build-time count (20,792; the kernel checks it); routing arrays contain exactly one route per CSV study (4,407; 3) and the route counts sum to the study total. Series with no usable IOP are `unverifiable`, those with unreadable or incomplete headers are `geometry_flagged`; both are counted and routed to `bypass_plain`, never dropped.
2. **No silent exclusion:** the fixed evaluation membership (for example the 870 fold-0 studies) is kept whole. Mirror TTA is applied only to `mirror_ok` studies. `bypass_plain` studies use plain inference for that study (their mirror view equals the plain view). `needs_transform` studies are bypassed until a separately qualified transform exists; using one is a new pre-declared step with its own check on real data. Any TTA result is reported on the full membership with bypass substitution **and** on the `mirror_ok` subset, with membership counts for each.
3. **Interpretation gate (descriptive):** the mirror rule counts as "verified for the cohort" for a plane only if at least **99%** of studies with that plane present have all of its selected series `ok_standard` (per-plane denominator above, at least 100 studies). This describes the cohort; it does not clear any individual study, and the 1% remainder is routed as above, not dropped.

On failure: if any plane is below 99%, or the audit is incomplete, or the ID hash or count checks fail, the mirror stays **unverified** and no `k94_mirror_tta` number is interpreted as an L/R mirror; the rule is revised (plane-specific actual-axis transform, qualified before use) rather than the threshold relaxed. Any slice-level inconsistency, missing or invalid IOP/IPP/PixelSpacing, reader fallback, tied or degenerate order, or unobserved slice in a series sends that series, and its study, to `bypass_plain` regardless of the cohort fractions.

## Limits

Header geometry only: it does not establish knee laterality, whether labels are mirror-invariant for a given model, or that mirrored predictions help. IOP is in patient coordinates, so the check is independent of patient-position tags. The one-pixel crop offset of a mirrored width (protocol) is unchanged. The test split (3 studies) is a smoke run, not evidence.

## Verification (synthetic)

`.venv/bin/python -m unittest tests.test_orientation_audit -v`: 46 tests OK, 0 skipped (pydicom 3.0.2, NumPy 2.5.3). `python3.12` without those packages: 46 run, 14 skipped, OK. Tests cover standard sagittal/coronal/axial, sign handling, oblique, nonstandard (coronal rows along y or z fail), inconsistent IOP, missing tags, order along the normal, slice-level and used-slice counts, per-series and per-study routing, per-slot counts, frozen parameters, pinned-selector parity, an end-to-end tree of synthetic pydicom files (inline and 2-process pool), the private routing file and the full `run` path with the pinned reader. v2 additions: identical-IPP sagittal pydicom fixtures and one-slice, no-IPP and tied series do not qualify; missing, zero, negative, short or non-numeric PixelSpacing is `bypass_plain` in every plane; sampled series never certify; reader parity against the pinned `order_and_meta` for ties, IPP of the wrong length, non-numeric IPP or IOP, short IOP, absent IOP/IPP, unparsable PixelSpacing and unreadable files, under natural and reversed file enumeration; the builder rejects a missing or disagreeing cache-shard contract and an existing destination. Mutation checks (each guard removed in turn) fail the targeted tests: degenerate/unavailable order allowed (3 failures), tie ambiguity removed (2), PixelSpacing not routing (3), sampling undetected (3), sorted glob (3), reader PixelSpacing parse dropped (2), exception position changed to InstanceNumber (3), reader fallback not routing (1), transform allowed on non-ascending order (1), builder skipping a missing contract (1), unreadable-file position changed (3). A smoke execution of the generated v2 notebook on a synthetic tree (paths and pins patched) completed; the real pins, data and Kaggle environment are untested.

Parent acceptance17:45UTC: independent Sol review46tests PASS/0skip and15 bundle checks PASS, including all four cache contracts, exact embedded source/reader/contract/notebook binding and private/offline/CPU settings. Notebook SHA92fb4c037b84a39f1725592acb26e3c16e47f5d172531de77cf66cd4199c92a6; contract SHA99ad997c8e0daf288cd70f33dfb2753d36615e01f54a554869f8173b1ac72d97. Angular qualification is approximate patient-L/R alignment, not proof of an exact anatomical reflection. Actual routing/results remain pending.

Accepted Kaggle source equals reviewed notebook; server metadata confirms private/offline/CPU. Exact launch receipt: `outputs/analysis/orientation-audit-20261012/launch-v1.json`. Actual cohort results pending.
