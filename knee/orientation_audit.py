"""Label-blind DICOM orientation audit and per-study mirror routing for source96.

Header metadata only: no pixel data, no labels, no reports. Series are chosen with the pinned public
reader's own ``_pick_series_for_slot`` over ``SLOTS96`` (see ``pinned_selector``), so the audit covers the
series and plane assignment the 96-slot cache used.

Assumption under test (``knee/source96_secondary_inference.py`` ``MIRROR_AXIS``): Sagittal slices lie along
patient L/R (series normal = x axis); Coronal and Axial image width (IOP row direction) is patient L/R.
Axes are DICOM LPS: +x patient Left, +y Posterior, +z Superior.

Outputs: an aggregate file (counts only, no identifiers) and a separate private per-study routing array
(mirror_ok / bypass_plain / needs_transform) computed from headers only. The cohort fractions are
descriptive; the per-study routing is the rule that governs.
"""
from collections import Counter
import csv
import glob
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import sys
import time
import warnings

VERSION = 'orientation-audit-kernel-v2'
PLANES = ('Sagittal', 'Coronal', 'Axial')
VECTORS = ('row', 'col', 'normal')
MIRROR_VECTOR = {'Sagittal': 'normal', 'Coronal': 'row', 'Axial': 'row'}
VECTOR_IMAGE_AXIS = {'row': 'width', 'col': 'height', 'normal': 'slice_order'}
SIGNED_LABELS = {('x', 1): 'L', ('x', -1): 'R', ('y', 1): 'P', ('y', -1): 'A', ('z', 1): 'S', ('z', -1): 'I'}
AXES = ('x', 'y', 'z')
CLASSES = ('standard', 'oblique', 'nonstandard')
STATUSES = ('ok_standard', 'ok_oblique', 'geometry_flagged', 'fail_nonstandard', 'fail_axis', 'unverifiable')
ROUTES = ('mirror_ok', 'bypass_plain', 'needs_transform')
ORDER_STATES = ('ascending', 'not_ascending', 'ambiguous', 'degenerate', 'unavailable')
FLAGS = ('inconsistent_iop', 'mixed_normal', 'ambiguous_normal', 'missing_iop', 'invalid_iop', 'missing_ipp',
         'invalid_ipp', 'missing_pixel_spacing', 'unreadable_header', 'reader_fallback', 'no_files', 'sampled')
# Any of these (or an order other than 'ascending') keeps a series out of mirror_ok: the geometry contract
# is complete, consistent, fully observed and ordered unambiguously exactly as the pinned reader orders it.
FLAGS_ROUTING = ('inconsistent_iop', 'mixed_normal', 'missing_iop', 'invalid_iop', 'missing_ipp', 'invalid_ipp',
                 'missing_pixel_spacing', 'unreadable_header', 'reader_fallback', 'sampled')
# Slice (instance) level counts; the first nine make a slice "routing flagged".
SLICE_ROUTING = ('iop_inconsistent', 'normal_mixed', 'iop_missing', 'iop_invalid', 'ipp_missing', 'ipp_invalid',
                 'pixel_spacing_bad', 'unreadable', 'reader_fallback')
SLICE_COUNTS = SLICE_ROUTING + ('routing_flagged', 'used', 'used_routing_flagged')
SLOT_ABBREV = {'Sagittal': 'sag', 'Coronal': 'cor', 'Axial': 'ax'}
SERIES_FIELDS = ['StudyInstanceUID', 'SeriesInstanceUID', 'Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']
SAFE_ID = re.compile(r'[A-Za-z0-9_.-]+')
SPAN96 = (0.02, 0.98)
DESCRIPTIVE_MIN_FRACTION = 0.99
DESCRIPTIVE_MIN_STUDIES = 100


def thresholds(**overrides):
    """Declared thresholds (recorded in every output); cosines derived from the angles."""
    values = {'standard_degrees': 20.0, 'oblique_degrees': 45.0, 'iop_tolerance': 1e-3,
              'normal_tolerance_degrees': 5.0, 'ambiguity_margin': 0.10, 'iop_validity_tolerance': 1e-2,
              'order_tolerance_mm': 1e-3, 'cosine_epsilon': 1e-9}
    unknown = set(overrides) - set(values)
    if unknown:
        raise ValueError(f'Unknown threshold {sorted(unknown)}')
    values.update(overrides)
    if not 0 < values['standard_degrees'] < values['oblique_degrees'] <= 90:
        raise ValueError('Require 0 < standard_degrees < oblique_degrees <= 90')
    for key in ('iop_tolerance', 'normal_tolerance_degrees', 'ambiguity_margin', 'iop_validity_tolerance',
                'order_tolerance_mm', 'cosine_epsilon'):
        if not math.isfinite(values[key]) or values[key] < 0:
            raise ValueError(f'Invalid threshold {key}')
    values['standard_min_cosine'] = math.cos(math.radians(values['standard_degrees']))
    values['oblique_min_cosine'] = math.cos(math.radians(values['oblique_degrees']))
    return values


def slot_names(slots):
    """sag0, sag1, cor0, cor1, ax: plane abbreviation plus index within the plane (bare if one slot)."""
    totals = Counter(plane for plane, _f, _c in slots)
    seen, names = Counter(), []
    for plane, _f, _c in slots:
        names.append(SLOT_ABBREV[plane] if totals[plane] == 1 else f'{SLOT_ABBREV[plane]}{seen[plane]}')
        seen[plane] += 1
    return names


# ---- vector helpers (plain Python: no NumPy needed for the pure functions) ----

def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(v):
    n = math.sqrt(_dot(v, v))
    return None if not math.isfinite(n) or n <= 0 else tuple(x / n for x in v)


def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _angle_degrees(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(a, b)))))


def dominant_axis(v):
    """(axis, sign, |component|, second |component|) with ties broken x < y < z."""
    mags = [abs(c) for c in v]
    i = max(range(3), key=lambda k: (mags[k], -k))
    second = max(m for k, m in enumerate(mags) if k != i)
    return AXES[i], (1 if v[i] >= 0 else -1), mags[i], second


def classify_component(magnitude, th):
    eps = th['cosine_epsilon']
    if magnitude >= th['standard_min_cosine'] - eps:
        return 'standard'
    if magnitude >= th['oblique_min_cosine'] - eps:
        return 'oblique'
    return 'nonstandard'


def vector_info(v, th):
    axis, sign, mag, second = dominant_axis(v)
    return {'axis': axis, 'sign': sign, 'label': SIGNED_LABELS[(axis, sign)], 'abs': mag,
            'class': classify_component(mag, th), 'ambiguous': (mag - second) < th['ambiguity_margin']}


# ---- per-instance header extraction ----

def _floats(value, n):
    """(tuple, state): state is ok, missing (absent/empty) or invalid (wrong length/non-finite)."""
    if value is None:
        return None, 'missing'
    try:
        seq = [float(x) for x in value]
    except (TypeError, ValueError):
        return None, 'invalid'
    if not seq:
        return None, 'missing'
    if len(seq) != n or not all(math.isfinite(x) for x in seq):
        return None, 'invalid'
    return tuple(seq), 'ok'


def extract_instance(ds, th):
    """Reduce a header (pydicom Dataset or any object with the same attributes) to plain values."""
    iop, iop_state = _floats(getattr(ds, 'ImageOrientationPatient', None), 6)
    if iop is not None:
        tol = th['iop_validity_tolerance']
        r, c = iop[:3], iop[3:]
        if abs(math.sqrt(_dot(r, r)) - 1) > tol or abs(math.sqrt(_dot(c, c)) - 1) > tol or abs(_dot(r, c)) > tol:
            iop_state = 'invalid'
    ipp, ipp_state = _floats(getattr(ds, 'ImagePositionPatient', None), 3)
    spacing, spacing_state = _floats(getattr(ds, 'PixelSpacing', None), 2)
    if spacing is not None and any(x <= 0 for x in spacing):
        spacing_state = 'invalid'
    number = getattr(ds, 'InstanceNumber', None)
    try:
        number = None if number is None else float(number)
    except (TypeError, ValueError):
        number = None
    key, source = reader_key(ds)
    return {'iop': iop, 'iop_state': iop_state, 'ipp': ipp, 'ipp_state': ipp_state,
            'spacing_state': spacing_state, 'instance_number': number, 'reader_key': key, 'reader_source': source}


def reader_key(ds):
    """Sort key of the pinned reader's ``order_and_meta`` (public reader lines 261-277), same fallbacks.

    ``geometry``: dot(IPP, row x col) from the raw IOP/IPP. ``instance_number``: IOP or IPP absent, or IOP not
    of length 6. ``exception``: anything in that try block raises (IPP of the wrong length, non-numeric
    values, unparsable PixelSpacing...), so the reader files the slice at position 0.0.
    """
    try:
        iop = getattr(ds, 'ImageOrientationPatient', None)
        ipp = getattr(ds, 'ImagePositionPatient', None)
        if iop is not None and ipp is not None and len(iop) == 6:
            r, c, p = [float(x) for x in iop[:3]], [float(x) for x in iop[3:]], [float(x) for x in ipp]
            if len(p) != 3:
                raise ValueError('IPP shape')
            key, source = float(_dot(p, _cross(r, c))), 'geometry'
        else:
            key, source = float(getattr(ds, 'InstanceNumber', 0) or 0), 'instance_number'
        spacing = getattr(ds, 'PixelSpacing', None)
        if spacing is not None:
            float(spacing[0])
        return key, source
    except Exception:
        return 0.0, 'exception'


def reader_position(inst):
    """Position the pinned reader sorts this instance by (unreadable headers: 0.0)."""
    return 0.0 if 'error' in inst else inst['reader_key']


def reader_order(instances):
    """Indices in the pinned reader's order: stable ascending sort on its key over its file enumeration."""
    return sorted(range(len(instances)), key=lambda k: (reader_position(instances[k]), k))


def used_positions(n, count, span=SPAN96):
    """Indices into the reader-sorted list that the pinned ``build_study`` slices (np.linspace rounding)."""
    if n <= 0:
        return []
    if n == 1:
        return [0] * count
    lo, hi = int(n * span[0]), max(int(n * span[1]) - 1, int(n * span[0]))
    if count == 1:
        values = [float(lo)]
    else:
        step = (hi - lo) / (count - 1)
        values = [i * step + lo for i in range(count)]
        values[-1] = float(hi)
    return [min(int(round(v)), n - 1) for v in values]


# ---- per-series analysis ----

def analyze_series(instances, plane, th, pick=None, files_total=None):
    """Orientation record for one selected series. Contains no identifiers.

    ``instances`` are in the pinned reader's own file enumeration order. ``files_total`` is the number of
    ``.dcm`` files in the series; if it exceeds ``len(instances)`` the series was only sampled, so unobserved
    slices exist and it is flagged ``sampled`` and can never be mirror_ok. ``pick`` = (slice count, span) of
    the slot: when given (and the series is fully observed), slice-level counts are also reported for the
    slices the pinned reader actually uses.
    """
    if plane not in MIRROR_VECTOR:
        raise ValueError('Unknown plane')
    sampled = files_total is not None and files_total > len(instances)
    if sampled:
        pick = None
    good = [i for i in instances if 'error' not in i]
    ok = [i for i in good if i['iop_state'] == 'ok']
    # Slice level: every instance gets its own flags; nothing is summarized by a majority.
    slice_flags = [set() for _ in instances]
    for k, inst in enumerate(instances):
        f = slice_flags[k]
        if 'error' in inst:
            f.add('unreadable')
            continue
        if inst['iop_state'] != 'ok':
            f.add('iop_' + inst['iop_state'])
        if inst['ipp_state'] != 'ok':
            f.add('ipp_' + inst['ipp_state'])
        if inst['spacing_state'] != 'ok':
            f.add('pixel_spacing_bad')
        if inst['reader_source'] != 'geometry':
            f.add('reader_fallback')
    result = {'plane': plane, 'files': len(instances), 'vectors': None, 'order': 'unavailable',
              'series_class': None, 'route': 'bypass_plain', 'lr_vector': None,
              'mirror': {'vector': MIRROR_VECTOR[plane], 'axis': None, 'sign': None, 'label': None,
                         'class': None, 'status': 'unverifiable'}}
    median = normal = vectors = None
    if ok:
        median = [_median([i['iop'][k] for i in ok]) for k in range(6)]
        row, col = _unit(median[:3]), _unit(median[3:])
        normal = _unit(_cross(row, col)) if row and col else None
    if normal is not None:
        vectors = {'row': vector_info(row, th), 'col': vector_info(col, th), 'normal': vector_info(normal, th)}
        for k, inst in enumerate(instances):
            if 'error' in inst or inst['iop_state'] != 'ok':
                continue
            if any(abs(inst['iop'][c] - median[c]) > th['iop_tolerance'] for c in range(6)):
                slice_flags[k].add('iop_inconsistent')
            n_i = _unit(_cross(inst['iop'][:3], inst['iop'][3:]))
            info = dominant_axis(n_i)
            if (_angle_degrees(n_i, normal) > th['normal_tolerance_degrees']
                    or SIGNED_LABELS[(info[0], info[1])] != vectors['normal']['label']):
                slice_flags[k].add('normal_mixed')
    slices = Counter({'total': len(instances)})
    for f in slice_flags:
        for name in f:
            slices[name] += 1
        slices['routing_flagged'] += bool(f & set(SLICE_ROUTING))
    if pick is not None and instances:
        count, span = pick
        order = reader_order(instances)
        for position in used_positions(len(instances), count, span):
            slices['used'] += 1
            slices['used_routing_flagged'] += bool(slice_flags[order[position]] & set(SLICE_ROUTING))
    result['slices'] = {name: slices[name] for name in ('total',) + SLICE_COUNTS}
    flags = {name: False for name in FLAGS}
    flags.update(no_files=not instances, unreadable_header=slices['unreadable'] > 0,
                 missing_iop=slices['iop_missing'] > 0, invalid_iop=slices['iop_invalid'] > 0,
                 missing_ipp=slices['ipp_missing'] > 0, invalid_ipp=slices['ipp_invalid'] > 0,
                 missing_pixel_spacing=slices['pixel_spacing_bad'] > 0, reader_fallback=slices['reader_fallback'] > 0,
                 sampled=sampled,
                 inconsistent_iop=slices['iop_inconsistent'] > 0, mixed_normal=slices['normal_mixed'] > 0,
                 ambiguous_normal=bool(vectors and vectors['normal']['ambiguous']))
    result['flags'] = flags
    if vectors is None:
        return result
    # Reader order (stable ascending sort on its key): does it run along +normal in IPP space, and is it unique?
    keys = [reader_position(i) for i in instances]
    ordered = [instances[k] for k in reader_order(instances)
               if 'error' not in instances[k] and instances[k]['ipp_state'] == 'ok']
    if ordered:
        proj = [_dot(i['ipp'], normal) for i in ordered]
        span_mm, tol = proj[-1] - proj[0], th['order_tolerance_mm']
        if len(proj) < 2 or abs(span_mm) <= tol:
            result['order'] = 'degenerate'
        elif not (span_mm > 0 and all(b - a >= -tol for a, b in zip(proj, proj[1:]))):
            result['order'] = 'not_ascending'
        elif len(set(keys)) != len(keys):
            result['order'] = 'ambiguous'   # equal reader keys: the reader's tie order is its file enumeration order
        else:
            result['order'] = 'ascending'
    worst = max((vectors[v]['class'] for v in VECTORS), key=CLASSES.index)
    mirror_info = vectors[MIRROR_VECTOR[plane]]
    if mirror_info['axis'] != 'x':
        status = 'fail_axis'
    elif mirror_info['class'] == 'nonstandard':
        status = 'fail_nonstandard'
    elif any(flags[name] for name in FLAGS_ROUTING) or result['order'] != 'ascending':
        status = 'geometry_flagged'
    else:
        status = 'ok_standard' if mirror_info['class'] == 'standard' else 'ok_oblique'
    result.update(vectors=vectors, series_class=worst)
    result['mirror'] = {'vector': MIRROR_VECTOR[plane], 'axis': mirror_info['axis'], 'sign': mirror_info['sign'],
                        'label': mirror_info['label'], 'class': mirror_info['class'], 'status': status}
    result['route'], result['lr_vector'] = series_route(result)
    return result


def series_route(result):
    """Per-series routing from headers only.

    mirror_ok: assumed mirror axis is L/R, standard class, geometry (IOP, IPP, PixelSpacing) complete, consistent,
        fully observed and ordered unambiguously (strictly ascending along the normal, no equal reader keys).
    needs_transform: the same clean geometry, and a different vector is L/R with standard class,
        so an exact flip exists along another image axis (width / height / slice order) but is not the
        assumed one; it must be separately qualified before use.
    bypass_plain: everything else (oblique, nonstandard, missing, invalid, inconsistent, sampled, degenerate or
        ambiguous order, unverifiable).
    """
    status = result['mirror']['status']
    if status == 'ok_standard':
        return 'mirror_ok', None
    clean = (not any(result['flags'][f] for f in FLAGS_ROUTING) and result['order'] == 'ascending'
             and not result['flags']['no_files'])
    if status == 'fail_axis' and clean:
        for v in VECTORS:
            info = result['vectors'][v]
            if v != result['mirror']['vector'] and info['axis'] == 'x' and info['class'] == 'standard':
                return 'needs_transform', v
    return 'bypass_plain', None


def study_route(series_routes):
    """Study routing over present series: any bypass -> bypass_plain; else any transform -> needs_transform."""
    if not series_routes or 'bypass_plain' in series_routes:
        return 'bypass_plain'
    return 'needs_transform' if 'needs_transform' in series_routes else 'mirror_ok'


# ---- series selection: the pinned reader's own policy ----

def select_series(pick, slots, rows):
    """Slot -> series assignment exactly as ``source_span_pilot.decode`` / pinned ``build_study`` make it.

    ``pick`` is the pinned ``_pick_series_for_slot``; a series that is picked is marked used
    even if it later has no files, as in ``build_study``. Returns (slot, plane, series_uid|None).
    """
    used, chosen = set(), []
    for slot, (plane, fluid, _count) in enumerate(slots):
        row = pick(rows, plane, fluid, used)
        if row is not None:
            used.add(row['SeriesInstanceUID'])
        chosen.append((slot, plane, None if row is None else row['SeriesInstanceUID']))
    return chosen


def pinned_selector(codes):
    """Build the qualified wide96 reader module; return (pick function, slots)."""
    from knee import source_span_pilot as pilot
    module = pilot.arm_module(codes, 'wide96')
    slots = [tuple(s) for s in module.SLOTS]
    if slots != [tuple(s) for s in pilot.SLOTS96] or module.MAXS != 96 or tuple(pilot.SPANS['wide96']) != SPAN96:
        raise ValueError('Reader slot layout or span differs from the qualified wide96 reader')
    return module._pick_series_for_slot, slots


def load_series_records(path):
    """Study -> series rows in file order (same order the cache builder used); metadata only."""
    records, seen = {}, set()
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SERIES_FIELDS:
            raise ValueError('Official series schema changed')
        for row in reader:
            if (set(row) != set(SERIES_FIELDS) or any(v is None for v in row.values())
                    or row['Fluid_Sensitive'] not in ('0', '1') or row['Fat_Suppression'] not in ('0', '1')
                    or row['Anatomical_Plane'] not in PLANES):
                raise ValueError('Invalid series metadata')
            for key in ('StudyInstanceUID', 'SeriesInstanceUID'):
                if not SAFE_ID.fullmatch(row[key]) or row[key] in ('.', '..'):
                    raise ValueError('Unsafe identity')
            if row['SeriesInstanceUID'] in seen:
                raise ValueError('Duplicate series')
            seen.add(row['SeriesInstanceUID'])
            records.setdefault(row['StudyInstanceUID'], []).append(row)
    return records


# ---- header reading (driver side; pydicom imported lazily) ----

def sample_indices(n, cap):
    """All indices, or ``cap`` evenly spaced ones including first and last."""
    if cap is None or n <= cap:
        return list(range(n))
    if cap < 2:
        raise ValueError('Instance cap must be at least 2')
    return sorted({round(k * (n - 1) / (cap - 1)) for k in range(cap)})


def read_series(directory, th, cap=None):
    """(files read, instance records, total .dcm files) for one series directory, as the pinned reader reads it."""
    import pydicom
    warnings.filterwarnings('ignore', module=r'pydicom(?:\..*)?')
    files = glob.glob(f'{directory}/*.dcm')   # unsorted, as the pinned reader: ties keep this enumeration order
    chosen = [files[k] for k in sample_indices(len(files), cap)]
    instances = []
    for path in chosen:
        try:  # same call as the pinned reader (no specific_tags): a header it cannot parse is unreadable here too
            instances.append(extract_instance(pydicom.dcmread(path, stop_before_pixels=True), th))
        except Exception as error:  # the pinned reader keeps unreadable headers with position 0.0
            instances.append({'error': type(error).__name__})
    return chosen, instances, len(files)


def audit_series_dir(directory, plane, th, cap=None, pick=None):
    """Worker: read headers (no pixels) under ``directory`` and analyze. Module-level for pickling."""
    _chosen, instances, total = read_series(directory, th, cap)
    result = analyze_series(instances, plane, th, pick=pick, files_total=total)
    result['files_total'] = total
    return result


def _audit_task(task):
    return audit_series_dir(*task)


# ---- aggregation (counts only) ----

class _Group:
    """Counts for a set of series (one plane, or one slot). Every selected series counts individually."""

    def __init__(self):
        self.series = self.headers = 0
        self.status, self.route, self.transform_axis = Counter(), Counter(), Counter()
        self.dominant = {v: Counter() for v in VECTORS}
        self.signed = {v: Counter() for v in VECTORS}
        self.klass = {v: Counter() for v in VECTORS}
        self.min_abs = {}
        self.series_class, self.flags, self.order, self.lr = Counter(), Counter(), Counter(), Counter()
        self.slices = Counter()
        self.max_inconsistent = 0

    def add(self, r):
        self.series += 1
        self.headers += r['files']
        self.status[r['mirror']['status']] += 1
        self.route[r['route']] += 1
        if r['lr_vector']:
            self.transform_axis[VECTOR_IMAGE_AXIS[r['lr_vector']]] += 1
        self.order[r['order']] += 1
        for name, on in r['flags'].items():
            self.flags[name] += bool(on)
        for name, n in r['slices'].items():
            self.slices[name] += n
        self.max_inconsistent = max(self.max_inconsistent, r['slices']['iop_inconsistent'])
        if r['vectors']:
            for v in VECTORS:
                info = r['vectors'][v]
                self.dominant[v][info['axis']] += 1
                self.signed[v][info['label']] += 1
                self.klass[v][info['class']] += 1
                self.min_abs[v] = min(self.min_abs.get(v, 1.0), info['abs'])
            self.series_class[r['series_class']] += 1
            if r['mirror']['axis'] == 'x':
                self.lr[r['mirror']['label']] += 1

    def to_dict(self):
        slices = {name: self.slices[name] for name in ('total',) + SLICE_COUNTS}
        slices['series_with_any_iop_inconsistent_slice'] = self.flags['inconsistent_iop']
        slices['max_iop_inconsistent_slices_in_a_series'] = self.max_inconsistent
        return {'series': self.series, 'header_files_read': self.headers,
                'mirror_status': {k: self.status[k] for k in STATUSES},
                'series_route': {k: self.route[k] for k in ROUTES},
                'needs_transform_image_axis': {k: self.transform_axis[k] for k in VECTOR_IMAGE_AXIS.values()},
                'mirror_vector_lr_sign': {'L': self.lr['L'], 'R': self.lr['R']},
                'dominant_axis': {v: {a: self.dominant[v][a] for a in AXES} for v in VECTORS},
                'signed_axis': {v: {l: self.signed[v][l] for l in SIGNED_LABELS.values()} for v in VECTORS},
                'vector_class': {v: {k: self.klass[v][k] for k in CLASSES} for v in VECTORS},
                'series_class': {k: self.series_class[k] for k in CLASSES},
                'min_dominant_abs_component': {v: self.min_abs.get(v) for v in VECTORS},
                'flags': {k: self.flags[k] for k in FLAGS},
                'order_along_normal': {k: self.order[k] for k in ORDER_STATES},
                'slice_level': slices}


class Aggregate:
    def __init__(self, slots):
        self.slots = [tuple(s) for s in slots]
        self.names = slot_names(self.slots)
        self.studies = 0
        self.studies_with_series = 0
        self.slot_missing = Counter()
        self.slot_present = Counter()
        self.routing = Counter()
        self.plane_studies = {p: Counter() for p in PLANES}
        self.planes = {p: _Group() for p in PLANES}
        self.by_slot = [_Group() for _ in self.slots]
        self.rows = []   # (study id, route, [status|None per slot], [route|None per slot]); private output only

    def add_study(self, study_id, chosen, results):
        """``chosen``: select_series output; ``results``: slot -> series record for present series."""
        self.studies += 1
        statuses, routes = [None] * len(self.slots), [None] * len(self.slots)
        by_plane = {p: [] for p in PLANES}
        for slot, plane, uid in chosen:
            if uid is None or slot not in results:
                self.slot_missing[self.names[slot]] += 1
                continue
            r = results[slot]
            self.slot_present[self.names[slot]] += 1
            self.planes[plane].add(r)
            self.by_slot[slot].add(r)
            statuses[slot], routes[slot] = r['mirror']['status'], r['route']
            by_plane[plane].append(r['mirror']['status'])
        present = [s for s in statuses if s is not None]
        self.studies_with_series += bool(present)
        for plane, found in by_plane.items():
            if not found:
                continue
            s = self.plane_studies[plane]
            s['with_plane'] += 1
            s['all_ok_standard'] += all(x == 'ok_standard' for x in found)
            s['not_all_ok_standard'] += any(x != 'ok_standard' for x in found)
            s['any_assumption_fail'] += any(x in ('fail_axis', 'fail_nonstandard') for x in found)
            s['any_fail_axis'] += any(x == 'fail_axis' for x in found)
            s['any_unverifiable'] += any(x == 'unverifiable' for x in found)
            s['any_geometry_flagged'] += any(x == 'geometry_flagged' for x in found)
            s['any_ok_oblique'] += any(x == 'ok_oblique' for x in found)
        route = study_route([x for x in routes if x is not None])
        self.routing[route] += 1
        self.rows.append((study_id, route, statuses, routes))

    def routing_arrays(self):
        """Positional arrays in processing order; consumed privately, never merged into the aggregate."""
        return {'study_ids': [r[0] for r in self.rows], 'routing': [r[1] for r in self.rows],
                'slot_status': [r[2] for r in self.rows], 'slot_route': [r[3] for r in self.rows],
                'slot_names': self.names}

    def to_dict(self):
        planes = {}
        for plane, group in self.planes.items():
            s = self.plane_studies[plane]
            studies = {k: s[k] for k in ('with_plane', 'all_ok_standard', 'not_all_ok_standard', 'any_assumption_fail',
                                         'any_fail_axis', 'any_unverifiable', 'any_geometry_flagged', 'any_ok_oblique')}
            studies['fraction_all_ok_standard'] = (studies['all_ok_standard'] / studies['with_plane']
                                                   if studies['with_plane'] else None)
            planes[plane] = {'mirror_vector': MIRROR_VECTOR[plane], **group.to_dict(), 'studies': studies}
        slots = {}
        for i, group in enumerate(self.by_slot):
            plane, fluid, count = self.slots[i]
            name = self.names[i]
            slots[name] = {'plane': plane, 'fluid_preference': fluid, 'slices_in_slot': count,
                           'studies_with_series': self.slot_present[name], 'studies_without_series': self.slot_missing[name],
                           **group.to_dict()}
        arrays = self.routing_arrays()
        return {'studies_total': self.studies, 'studies_with_selected_series': self.studies_with_series,
                'denominators': {'studies': self.studies,
                                 'plane_studies': {p: self.plane_studies[p]['with_plane'] for p in PLANES},
                                 'slot_series': {n: self.slot_present[n] for n in self.names}},
                'study_routing': {k: self.routing[k] for k in ROUTES},
                'ordered_study_ids_sha256': hashlib.sha256('\n'.join(arrays['study_ids']).encode()).hexdigest(),
                'routing_array_sha256': hashlib.sha256(json.dumps(arrays['routing']).encode()).hexdigest(),
                'planes': planes, 'slots': slots}


def descriptive_summary(aggregate, min_fraction=DESCRIPTIVE_MIN_FRACTION, min_studies=DESCRIPTIVE_MIN_STUDIES):
    """Descriptive cohort summary only: it cannot certify a fixed transform for any individual study.

    The per-study routing arrays govern which studies may be mirrored.
    """
    per_plane = {}
    for plane, p in aggregate['planes'].items():
        s = p['studies']
        enough = s['with_plane'] >= min_studies
        per_plane[plane] = {'studies': s['with_plane'], 'all_ok_standard': s['all_ok_standard'],
                            'fraction': s['fraction_all_ok_standard'],
                            'at_or_above_threshold': (s['fraction_all_ok_standard'] >= min_fraction) if enough else None}
    total = aggregate['studies_total']
    return {'role': 'descriptive only; per-study routing governs', 'threshold_fraction': min_fraction,
            'min_studies': min_studies, 'per_plane': per_plane,
            'study_routing_fraction': {k: (v / total if total else None) for k, v in aggregate['study_routing'].items()}}


# ---- driver ----

def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + '\n')
    tmp.replace(path)


def _sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def audit_split(image_root, records, pick, slots, th, *, out, config, routing_out=None, workers=8, cap=None,
                span=SPAN96, chunk_studies=32, started=None):
    """Audit every selected series of ``records`` under ``image_root``; write ``out`` atomically.

    The aggregate ``out`` has no identifiers. ``routing_out`` (if given) receives the private per-study
    routing arrays, written once at the end.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    # fork where available (Kaggle Linux): workers need no __main__ re-import, which a notebook cannot provide.
    context = multiprocessing.get_context('fork' if 'fork' in multiprocessing.get_all_start_methods() else None)
    started = time.monotonic() if started is None else started
    image_root = Path(image_root)
    aggregate = Aggregate(slots)
    state = {'status': 'RUNNING', 'version': VERSION, 'config': config, 'series_audited': 0, 'studies_done': 0,
             'studies_planned': len(records), 'pixels_decoded': False, 'labels_read': False,
             'identifiers_in_aggregate': False, 'private_routing_file': None if routing_out is None else Path(routing_out).name}

    def save(status=None):
        if status:
            state['status'] = status
        state.update(elapsed_seconds=time.monotonic() - started,
                     peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                     aggregate=aggregate.to_dict())
        _atomic_json(out, state)

    save()
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=context) if workers > 1 else None
    try:
        studies = list(records)
        for begin in range(0, len(studies), chunk_studies):
            batch, tasks, owners = [], [], []
            for sid in studies[begin:begin + chunk_studies]:
                chosen = select_series(pick, slots, records[sid])
                batch.append((sid, chosen))
                for slot, plane, uid in chosen:
                    if uid is not None:
                        tasks.append((str(image_root / sid / uid), plane, th, cap, (slots[slot][2], tuple(span))))
                        owners.append((sid, slot))
            results = list(pool.map(_audit_task, tasks, chunksize=2)) if pool else [_audit_task(t) for t in tasks]
            by_study = {}
            for (sid, slot), r in zip(owners, results):
                by_study.setdefault(sid, {})[slot] = r
            for sid, chosen in batch:
                aggregate.add_study(sid, chosen, by_study.get(sid, {}))
            state['series_audited'] += len(tasks)
            state['studies_done'] += len(batch)
            if (begin // chunk_studies) % 10 == 0:
                save()
                print(json.dumps({'event': 'orientation_progress', 'studies': state['studies_done'],
                                  'series': state['series_audited']}), flush=True)
    except BaseException as error:
        state['failure'] = {'type': type(error).__name__}
        save('ERROR')
        raise
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    if state['studies_done'] != len(records):
        raise ValueError('Routing must cover every study')
    if routing_out is not None:
        arrays = aggregate.routing_arrays()
        _atomic_json(routing_out, {'version': VERSION, 'private': True, 'split': config.get('split'),
                                   'order': 'series CSV first-seen study order; arrays are positional',
                                   'routes': list(ROUTES), **arrays})
    save('COMPLETE' if state['series_audited'] else 'ERROR')
    return state


def _exact_root(candidates, files, dirs):
    matches = [r for r in candidates if all((r / f).is_file() for f in files) and all((r / d).is_dir() for d in dirs)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one competition mount, found {len(matches)}')
    return matches[0]


def run(inputs, out, contract, codes, *, started):
    """Kaggle entry: pinned selector, visible test then train population; aggregate + private routing files."""
    from knee.fresh_reader_setup import time_limit
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    th = contract['thresholds']
    contract_sha256 = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'),
                                                allow_nan=False).encode()).hexdigest()
    try:
        with time_limit(max(1, 43200 - (time.monotonic() - started)), 'actual platform12h limit'):
            if th != thresholds(**{k: v for k, v in th.items() if not k.endswith('_cosine')}):
                raise ValueError('Threshold record inconsistent')
            if {k: hashlib.sha256(v.encode()).hexdigest() for k, v in codes.items()} != contract['reader_hashes']:
                raise ValueError('Reader source differs from pinned contract')
            pick, slots = pinned_selector(codes)
            if [list(s) for s in slots] != contract['slots'] or tuple(contract['span']) != SPAN96:
                raise ValueError('Slot layout differs from contract')
            root = _exact_root([Path(inputs) / 'competitions/rsna-knee-abnormality-detection',
                                Path(inputs) / 'rsna-knee-abnormality-detection'],
                               ['train_series.csv', 'test_series.csv'], ['train_series', 'test_series'])
            if _sha256_file(root / 'train_series.csv') != contract['train_series_csv_sha256']:
                raise ValueError('train_series.csv differs from pinned contract')
            import importlib.metadata
            report = {}
            for split in ('test', 'train'):  # tiny split first: fails fast on environment problems
                csv_path = root / f'{split}_series.csv'
                records = load_series_records(csv_path)
                config = {'split': split, 'contract_sha256': contract_sha256, 'thresholds': th,
                          'slots': contract['slots'], 'slot_names': slot_names(slots), 'span': contract['span'],
                          'reader_hashes': contract['reader_hashes'], 'series_csv_sha256': _sha256_file(csv_path),
                          'workers': contract['workers'], 'instance_cap': contract['instance_cap'],
                          'code_revision': contract['code_revision'],
                          'population': f'all studies in {split}_series.csv (superset of the 4345-study cache population)',
                          'denominators': contract['denominators'], 'python': sys.version.split()[0],
                          'pydicom': importlib.metadata.version('pydicom'),
                          'descriptive_threshold': {'fraction': DESCRIPTIVE_MIN_FRACTION, 'min_studies': DESCRIPTIVE_MIN_STUDIES}}
                target = out / f'{split}-orientation.json'
                state = audit_split(root / f'{split}_series', records, pick, slots, th, out=target, config=config,
                                    routing_out=out / f'routing-private-{split}.json', workers=contract['workers'],
                                    cap=contract['instance_cap'], span=contract['span'], started=started)
                expected = contract.get('expected_selected_series', {}).get(split)
                if expected is not None and state['series_audited'] != expected:
                    raise ValueError('Selected-series count differs from the build-time count')
                state['descriptive_summary'] = descriptive_summary(state['aggregate'])
                _atomic_json(target, state)
                report[split] = state['status']
    except BaseException as error:
        _atomic_json(out / 'failure.json', {'status': 'ERROR', 'type': type(error).__name__,
                                            'identifiers_in_aggregate': False})
        raise RuntimeError('Orientation audit failed; sanitized aggregate saved') from None
    return {'status': 'COMPLETE' if set(report.values()) == {'COMPLETE'} else 'ERROR', 'splits': report,
            'elapsed_seconds': time.monotonic() - started}
