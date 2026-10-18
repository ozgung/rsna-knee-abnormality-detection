"""Secondary frozen-checkpoint inference for the own224-d96 top-3 SWA model.

One checkpoint, one pass over the same studies, separate prediction sets. Three modes (the CLI requires an explicit one):
  plain_only                  k16_plain (parity with the primary export) and k94_plain. No mirror, no routing artifacts
                              are read, so this mode runs even when the orientation-audit artifacts are not available.
  header_gated_approx_lr_tta  k16_plain, k94_plain and k94_header_gated_approx_lr_tta: only studies the label-blind
                              DICOM header audit routed to mirror_ok get the approximate L/R mirror (equal 50:50 mean of
                              plain and mirrored K94); every other study carries its plain K94 row byte-for-byte. The full
                              fixed development membership is kept, nothing is dropped, nothing is tuned.
  ungated_mirror_tta          LEGACY, UNGATED: k16_plain, k94_plain and k94_mirror_tta, the mirror applied to every study
                              regardless of its headers. Kept computable for historical comparison only; its receipts
                              keep their original semantics and are not reinterpreted by the audit.
The mirror is the anatomical L/R mirror under the ASSUMED standard orientation. The header audit shows that 93.67% of
the training studies are routed mirror_ok, but the per-plane all-standard shares (sagittal 95.46%, coronal 97.16%, axial
98.16%) FAIL the frozen 99% cohort descriptor, so the mirror is NOT universally verified (mirror_orientation_verified and
universal_geometry_verified stay false), and consistent geometry alone does not establish label invariance or any
accuracy benefit.

All outputs are SECONDARY diagnostics. The frozen K16 primary endpoint governs promotion;
nothing here selects epochs, weights, targets or a blend, and nothing here trains.
Models and torch are imported lazily so the dependency-free audit path is unchanged.
"""
import argparse
import ast
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import resource
import sys
import time
import types

from knee import public_training_smoke as smoke, source96_training as guard, source_span_pilot as pilot
from knee.device import select_device

VERSION = 'source96-secondary-inference-v1'
ARCH = guard.ARCH
TARGETS = list(smoke.TARGETS)
RES = 224
PARITY_ATOL = 1e-4        # same-hardware FP16 tolerance already used by the own224 K16 parity check
WINDOW_COUNTS = (16, 94)  # K16 primary; K94 = every interior triplet of a 96-slot stack
ARMS = ('k16_plain', 'k94_plain', 'k94_mirror_tta')   # the LEGACY ungated arm set, unchanged
GATED_ARM = 'k94_header_gated_approx_lr_tta'
MODE_PLAIN, MODE_GATED, MODE_UNGATED = 'plain_only', 'header_gated_approx_lr_tta', 'ungated_mirror_tta'
MODE_ARMS = {MODE_PLAIN: ('k16_plain', 'k94_plain'), MODE_GATED: ('k16_plain', 'k94_plain', GATED_ARM), MODE_UNGATED: ARMS}
VERSION_ROUTED = 'source96-secondary-inference-v2'   # plain_only and header-gated receipts; the legacy mode keeps VERSION
ROLE = ('Secondary diagnostics only: the frozen K16 primary endpoint governs promotion. '
        'No selection, blending or training uses these outputs.')

# Slot layout comes from the single source of truth that built the 96-slot cache.
SLOT_PLANES = tuple(plane for plane, _fluid, _count in pilot.SLOTS96)
SLOT_FLUID = tuple(fluid for _plane, fluid, _count in pilot.SLOTS96)
SLOT_SIZES = tuple(count for _plane, _fluid, count in pilot.SLOTS96)
DEPTH = sum(SLOT_SIZES)
SLOT_BOUNDS = tuple((sum(SLOT_SIZES[:i]), sum(SLOT_SIZES[:i + 1])) for i in range(len(SLOT_SIZES)))
# ASSUMED (unverified) anatomical left/right axis per plane under the pinned reader and standard IOP (see
# docs/experiments/source96-secondary-inference-protocol-20261012.md for the derivation and what it rests on):
#   Sagittal: the SLICE axis is taken to be the patient L/R axis (series are sorted by IPP . (row x col)),
#             the in-plane axes A/P and S/I.  Mirror = reverse the slice order of the slot.
#   Coronal / Axial: slices stack along A/P or S/I; the image COLUMN axis (array width) is taken to be
#             patient L/R.  Mirror = flip the width axis; slice order stays.
# No DICOM header is available to the cache, so none of this has been measured.
MIRROR_AXIS = {'Sagittal': 'slice_order', 'Coronal': 'width', 'Axial': 'width'}
MIRROR_CONVENTION = ('uint8 volume and mask mirrored before windowing/normalisation; per sagittal slot the slice order is '
                     'reversed, per coronal/axial slot the image width is flipped; window centres are re-planned on the '
                     'mirrored mask; masked slices move with their data and stay masked.')
MIRROR_ORIENTATION_VERIFIED = False
UNIVERSAL_GEOMETRY_VERIFIED = False   # per-study routing is not cohort-wide verification; the 99% descriptor failed
MIRROR_ORIENTATION_ASSUMPTION = (
    'Anatomical L/R mirror under the assumed standard orientation (unverified): sagittal slice axis = patient L/R, '
    'coronal and axial image width = patient L/R. The cache stores no ImageOrientationPatient and no real DICOM was '
    'inspected, so oblique or nonstandard series are not excluded.')
ORIENTATION_AUDIT_REQUIRED = (
    'Before any k94_mirror_tta number is interpreted: run a label-blind header audit over the training series that tallies, '
    'per plane, the dominant axis (argmax of the absolute direction cosine) of the IOP row direction, the IOP column '
    'direction and their cross-product normal, with oblique (no clearly dominant axis) and nonstandard cases counted '
    'separately, and stratify or exclude studies whose series do not match the assumed axes.')
GATED_NOTE = ('Header-gated approximate L/R mirror: only studies whose selected series all passed the label-blind header audit '
              '(route mirror_ok) are mirrored. Consistent geometry alone does not establish label invariance or any accuracy benefit; '
              'the per-plane all-standard cohort shares fail the frozen 99% descriptor, so the mirror is not universally verified.')
if DEPTH != 96 or len(SLOT_SIZES) != 5 or set(SLOT_PLANES) - set(MIRROR_AXIS):
    raise ValueError('SLOTS96 layout changed; re-derive the per-plane mirror before use')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def slot_layout():
    return [{'slot': i, 'plane': SLOT_PLANES[i], 'fluid': SLOT_FLUID[i], 'start': lo, 'stop': hi,
             'mirror': MIRROR_AXIS[SLOT_PLANES[i]]} for i, (lo, hi) in enumerate(SLOT_BOUNDS)]


# --------------------------------------------------------------------------- L/R mirror (orientation assumed, unverified)
def mirror_volume(volume):
    """L/R mirror of a (96,H,W) uint8 stack under the assumed standard orientation (UNVERIFIED).

    Returns a new array and never mutates the cache.  The rule below is an anatomical L/R mirror only if the
    orientation assumption in MIRROR_ORIENTATION_ASSUMPTION holds; no header audit has confirmed it.

    Why a per-slot rule rather than one horizontal flip or one whole-stack reversal:
    * Under the assumption, sagittal slices are perpendicular to the patient L/R axis, so a left/right
      reflection does not change the content of any sagittal slice (its axes are A/P and S/I); it only
      reverses the order in which the slices lie along L/R.  Flipping the width of a sagittal image would
      be an A/P flip, which is not an L/R mirror.
    * The two sagittal slots are two different series (fluid-sensitive and not).  Each series is
      reversed within its own slot; reversing all 48 sagittal positions at once would swap the two
      series into each other's slots.
    * Under the assumption, coronal and axial slices contain the L/R axis along their columns, so the
      reflection flips the width; their stacking order (A/P or S/I) is unchanged.
    * Because the whole volume is transformed first and windows are cut afterwards, a sagittal window
      centred at c in the mirrored study is the reversed triplet centred at the reflected slice.  A
      window-local channel reversal at the SAME centre, as in the public Raptor notebook, equals that in a
      full-valid synthetic reflected-centre comparison for every window inside one series (the four K94
      windows that straddle a sagittal series boundary, centres 25, 26, 47, 48, differ by construction).
      That comparison says nothing about partial masks: with missing or partial planes the mirrored mask
      re-plans the centres, so the sampled bags of the two constructions are not claimed to match.
    Done on uint8, before the per-channel ImageNet normalisation: mirror the stack, then normalise.
    """
    import numpy as np
    v = np.asarray(volume)
    require(v.ndim == 3 and v.shape[0] == DEPTH and v.dtype == np.uint8, 'Mirror requires a (96,H,W) uint8 volume')
    out = np.empty_like(v)
    for (lo, hi), plane in zip(SLOT_BOUNDS, SLOT_PLANES):
        out[lo:hi] = v[lo:hi][::-1] if MIRROR_AXIS[plane] == 'slice_order' else v[lo:hi, :, ::-1]
    return out


def mirror_mask(mask):
    """Mask that follows the data: slice-order planes are permuted like the slices, width planes are unchanged.

    A masked or missing slice therefore stays masked at its new position and no valid slice is moved to a
    masked index (or the reverse), so the pixel/mask consistency invariant still holds after mirroring.
    """
    import numpy as np
    m = np.asarray(mask)
    require(m.shape == (DEPTH,) and np.isin(m, [0, 1]).all(), 'Mirror requires a binary (96,) mask')
    out = m.copy()
    for (lo, hi), plane in zip(SLOT_BOUNDS, SLOT_PLANES):
        if MIRROR_AXIS[plane] == 'slice_order':
            out[lo:hi] = m[lo:hi][::-1]
    return out


def mirror_study(volume, mask):
    return mirror_volume(volume), mirror_mask(mask)


def validate_study(volume, mask):
    """Same invariants as the cache loader: geometry, binary mask, pixel/mask consistency, not empty."""
    import numpy as np
    v, m = np.asarray(volume), np.asarray(mask)
    require(v.ndim == 3 and v.shape[0] == DEPTH and v.dtype == np.uint8, 'Volume geometry changed')
    require(m.shape == (DEPTH,) and np.isin(m, [0, 1]).all(), 'Mask geometry changed')
    require(np.array_equal(m.astype(bool), v.reshape(DEPTH, -1).any(axis=1)), 'Pixel/mask consistency failed')
    require(bool(m.any()), 'Whole-empty study cannot be scored')


class MappedCorpus:
    """Row-indexed view (vols or masks) that applies a transform on access; the source stays untouched."""

    def __init__(self, source, transform):
        self.source, self.transform = source, transform

    @property
    def shape(self):
        return tuple(self.source.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.transform(self.source[row])


# --------------------------------------------------------------------------- window centre planning
def eval_centres(mask, k):
    """Independent restatement of the pinned eval policy (StudyWindows._centers with train=False).

    Valid-range min/max, both triplet neighbours inside the valid range, evenly spaced with
    linspace(...).round(), duplicates kept; fewer than three valid slices falls back to slices 0..2.
    """
    import numpy as np
    valid = np.flatnonzero(np.asarray(mask) > 0)
    if len(valid) < 3:
        valid = np.arange(min(3, DEPTH))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, DEPTH - 2))]
    return [cs[i] for i in np.linspace(0, len(cs) - 1, k).round().astype(int)]


def center_plan(mask, sampler, k):
    """Centres from the pinned sampler (cross-checked against eval_centres) plus coverage flags."""
    import numpy as np
    require(k in WINDOW_COUNTS, 'Only K16 and K94 are supported')
    require(not sampler.train, 'Deterministic eval sampler required')
    m = np.asarray(mask)
    require(m.shape == (DEPTH,) and np.isin(m, [0, 1]).all(), 'Binary96 mask required')
    valid = np.flatnonzero(m > 0)
    fallback = len(valid) < 3
    if fallback:
        valid = np.arange(3)
    sampler._D = DEPTH
    centers = [max(1, min(int(c), DEPTH - 2)) for c in sampler._centers(valid, k)]
    require(len(centers) == k and centers == eval_centres(m, k), 'Pinned sampler centre policy differs')
    slot = np.repeat(np.arange(len(SLOT_SIZES)), SLOT_SIZES)
    return centers, {'windows': k, 'unique_centers': len(set(centers)), 'duplicate_centers': k - len(set(centers)),
                     'masked_triplets': sum(not bool(m[c - 1:c + 2].all()) for c in centers),
                     'boundary_triplets': sum(len(set(slot[c - 1:c + 2])) > 1 for c in centers),
                     'fallback_studies': int(fallback)}


# --------------------------------------------------------------------------- pinned model code
DEFINITION_NAMES = ('StudyWindows', 'build_backbone', 'RaptorClassifier')


class _LazyTimm:
    """Import timm only when a backbone is actually requested."""

    def __getattr__(self, name):
        return getattr(importlib.import_module('timm'), name)


def load_definitions(source, expected_sha256=smoke.SOURCE_SHA, timm_module=None):
    """Execute only StudyWindows, build_backbone and RaptorClassifier from byte-pinned trainer source.

    The original public trainer and the source96 patched trainer carry the same scientific definitions;
    pass the trainer file's own pinned digest. Never imports or runs the trainer's main().
    """
    require(smoke.digest(source.encode()) == expected_sha256, 'Trainer source differs from its pinned digest')
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in DEFINITION_NAMES]
    require(sorted(n.name for n in nodes) == sorted(DEFINITION_NAMES), 'Trainer definition inventory differs')
    ns = {'np': np, 'torch': torch, 'nn': nn, 'F': F, 'random': random, 'Dataset': torch.utils.data.Dataset,
          'timm': timm_module if timm_module is not None else _LazyTimm(),
          '_roi_square_box': None, '_open_corpus': lambda: None}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-secondary-definitions>', 'exec'), ns)
    return types.SimpleNamespace(**{name: ns[name] for name in DEFINITION_NAMES}, source_sha256=expected_sha256)


def make_sampler(definitions, ids, id2row, vols, masks, k, res):
    """Pinned StudyWindows over the given (possibly mirrored) corpus view; same call as the primary export."""
    import numpy as np
    labels = {sid: np.zeros(12, np.float32) for sid in ids}
    sampler = definitions.StudyWindows('', list(ids), id2row, labels, res, k, False, aug=False, norm='imagenet', roi=False)
    sampler.vols, sampler.masks = vols, masks
    return sampler


def checked_windows(x, k, res):
    require(tuple(x.shape) == (k, 3, res, res) and bool(x.isfinite().all()), 'Window schema')
    return x


def verify_state(state, res=RES):
    import torch
    require(isinstance(state, dict) and state.get('arch') == ARCH and state.get('res') == res and state.get('lab') == TARGETS,
            'Frozen checkpoint architecture/resolution/target identity differs')
    epochs = state.get('swa_over')
    require(isinstance(epochs, list) and len(epochs) == 3 and state.get('epoch') == epochs, 'Top-3 SWA identity required')
    weights = state.get('model')
    require(bool(weights) and all(torch.is_tensor(v) and bool(torch.isfinite(v).all()) for v in weights.values()), 'Finite checkpoint state')


def bind_checkpoint(work, fold, checkpoint, res=RES):
    """Accept only the unchanged qualified primary-K16 SWA checkpoint, via the existing secondary export guard."""
    import torch
    path = Path(checkpoint)
    require(path.is_file() and not path.is_symlink(), 'Frozen checkpoint file required')
    sha = smoke.file_sha(path)
    state = torch.load(path, map_location='cpu', weights_only=True)
    verify_state(state, res)
    epochs = state['swa_over']
    try:
        guard.export_guard(work, fold, path, 'swa_topk', epochs, False, True)
        primary = json.loads((Path(work) / f'run_fold{fold}.json').read_bytes())
    except (OSError, KeyError, TypeError, AttributeError):
        raise ValueError('K94 requires unchanged qualified primaryK16 checkpoint: primary record unreadable or incomplete') from None
    binding = {'checkpoint_file': path.name, 'checkpoint_sha256': sha, 'checkpoint_kind': 'swa_topk', 'selected_epochs': epochs,
               'primary_record_checkpoint_sha256': primary['checkpoint_sha256'], 'primary_record_k_eval': primary['export']['k_eval'],
               'export_guard': 'secondary=True accepted: unchanged primary K16 checkpoint, no epoch selection'}
    return state, binding


def build_model(definitions, state, device):
    backbone = definitions.build_backbone(state['arch'], pretrained=False)
    model = definitions.RaptorClassifier(backbone, F_dim=backbone.num_features).to(device)
    model.load_state_dict(state['model'], strict=True)
    return model.eval()


def predict(model, x, device, fp16):
    """Primary-export arithmetic: FP16 autocast on CUDA only, sigmoid of float32 logits, no gradients."""
    import torch
    with torch.no_grad():
        with torch.autocast(device, dtype=torch.float16, enabled=bool(fp16) and device == 'cuda'):
            return torch.sigmoid(model(x.to(device)).float()).cpu().numpy()


# --------------------------------------------------------------------------- outputs
def probabilities(values, rows):
    import numpy as np
    a = np.asarray(values)
    require(a.shape == (rows, 12) and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all(), 'Probability schema/range')
    return a


def save_json(path, value):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def write_csv(path, ids, values):
    p = Path(path)
    tmp = p.with_suffix('.tmp')
    with tmp.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['StudyInstanceUID', *TARGETS])
        for sid, row in zip(ids, values):
            writer.writerow([sid, *[repr(float(v)) for v in row]])
    tmp.replace(p)


def write_npz(path, ids, values, sha, **views):
    import numpy as np
    p = Path(path)
    tmp = p.with_name(p.stem + '.tmp.npz')
    np.savez(tmp, ids=np.asarray(ids), probabilities=values, targets=np.asarray(TARGETS), checkpoint_sha256=np.asarray(sha), **views)
    tmp.replace(p)


PRIMARY_OOF_SETS = ('val', 'gold')  # the only set labels the pinned train_knee.py export writes


def saved_primary(path, ids):
    """Primary-export OOF rows (set=val), aligned to ids, validated before any comparison.

    Rejects: a wrong header, malformed rows (wrong field count, non-numeric or empty values), duplicate
    validation rows (a dict would silently collapse them), validation IDs that differ from ids in either
    direction, probabilities that are non-finite or outside [0, 1], and any set label other than the primary export's
    'val' and 'gold'.  Only 'gold' rows are skipped, after the same row-shape validation.
    Error messages carry counts and line numbers only, never study identifiers.
    """
    import numpy as np
    ids = [str(i) for i in ids]
    require(ids and len(ids) == len(set(ids)), 'Primary OOF comparison needs non-empty unique evaluation IDs')
    rows = {}
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ['StudyInstanceUID', *TARGETS, 'set'], 'Primary OOF schema')
        for r in reader:
            line = reader.line_num
            require(None not in r and all(v is not None for v in r.values()), f'Primary OOF malformed row at line {line}')
            require(r['set'] in PRIMARY_OOF_SETS, f'Primary OOF unknown set label at line {line}')
            if r['set'] != 'val':
                continue
            require(r['StudyInstanceUID'] not in rows, f'Primary OOF duplicate validation row at line {line}')
            try:
                values = [float(r[t]) for t in TARGETS]
            except ValueError:
                raise ValueError(f'Primary OOF malformed value at line {line}') from None
            require(all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values),
                    f'Primary OOF probability not finite within [0, 1] at line {line}')
            rows[r['StudyInstanceUID']] = values
    missing, unexpected = len(set(ids) - set(rows)), len(set(rows) - set(ids))
    require(not missing and not unexpected and len(rows) == len(ids),
            f'Primary OOF must cover exactly the evaluation studies (missing {missing}, unexpected {unexpected})')
    return np.asarray([rows[i] for i in ids], dtype=np.float64)


def parity_report(current, saved, atol, enforce):
    import numpy as np
    error = float(np.max(np.abs(np.asarray(current, dtype=np.float64) - saved)))
    return {'max_absolute_error': error, 'absolute_tolerance': atol, 'within_tolerance': error <= atol, 'enforced': bool(enforce),
            'reference': 'primary export oof (K16, same checkpoint)',
            'note': 'Same checkpoint, device, FP16, window path and batch grouping as the primary export; '
                    'only nondeterministic FP16 reduction differences are expected.'}


def development_ids(ids, folds, excluded, fold):
    """Held-out fold members in cache order, matching the primary export's val_ids construction."""
    return [u for u in ids if u not in excluded and folds.get(u, -1) == fold]


# --------------------------------------------------------------------------- pinned orientation-audit artifacts
ROUTES = ('mirror_ok', 'bypass_plain', 'needs_transform')   # same vocabulary as knee.orientation_audit.ROUTES
ELIGIBLE_ROUTE = 'mirror_ok'
DESCRIPTOR_MIN_FRACTION = 0.99                               # frozen cohort descriptor; never relaxed here
ROUTING_PLANES = ('Sagittal', 'Coronal', 'Axial')
# Immutable pins of the label-blind header audit (orientation-audit-kernel-v2, train split, 4407 studies / 20792 series).
# File hashes are of the exact bytes; the two array hashes and the canonical contract hash are the audit's own
# integrity fields (see knee/orientation_audit.py). Any other file or any edit of these files is refused.
ORIENTATION_PINS = {
    'orientation_version': 'orientation-audit-kernel-v2', 'split': 'train', 'studies': 4407,
    'aggregate_file': 'train-orientation.json',
    'aggregate_file_sha256': 'bb44eedc9c487aee653b0cea7e4e93256e05e65d68fe8fccdcbc039965ded92a',
    'routing_file': 'routing-private-train.json',
    'routing_file_sha256': 'b66fad82f5a03c432c491c0b7640ecb100d30fc7c09fd59fa53de573076447ee',
    'contract_file': 'build-contract.json',
    'contract_file_sha256': '6fafc543a74a5ada62383b2b7884e16a48f799af35433b64b117e536ff41de18',
    'contract_canonical_sha256': '99ad997c8e0daf288cd70f33dfb2753d36615e01f54a554869f8173b1ac72d97',
    'ordered_study_ids_sha256': '21c1944bd15c3397290f0816de614ad4153f62e84c4bfb0e4d6147ac72084af8',
    'routing_array_sha256': '9068bae1a386be883eb4f5398f80db915a78c170a1e16ffdf37226d69011fa5f'}
_VERIFIED = object()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_pinned_file(path, name, expected_sha256):
    """Read the file once, hash the exact bytes against the pin, and return those same bytes (no check/use gap)."""
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), f'Orientation {name} file required')
    require(p.name == name, f'Orientation file name differs from the pinned {name}')
    data = p.read_bytes()
    require(sha256_bytes(data) == expected_sha256, f'Orientation {name} file hash differs from its pin')
    return data


def parse_json(data, what):
    try:
        return json.loads(data)
    except ValueError:
        raise ValueError(f'Orientation {what} is not valid JSON') from None


def route_hashes(route_by_id):
    """(ordered-ID hash, route-array hash, route counts) recomputed from a route map, exactly as the audit computed them."""
    ids, routes = list(route_by_id), list(route_by_id.values())
    return sha256_bytes('\n'.join(ids).encode()), sha256_bytes(json.dumps(routes).encode()), {r: routes.count(r) for r in ROUTES}


class VerifiedRouting:
    """Routes of an orientation audit whose files, hashes and contract were all verified; only load_routing_artifacts builds one.

    Immutable: the route map is a read-only view over a private copy, every attribute is fixed at construction (slots plus
    blocked __setattr__/__delattr__), and the nested counts, descriptor, pins and receipt are held as frozen JSON text and
    handed out as fresh copies.  The content binding is also re-checked from the payload at construction and at every use
    (revalidate): the recomputed ID-order and route-array hashes and the route counts must equal the stored values and the
    pins, so a payload that was swapped behind the guards, or forged with the original hashes, is refused.
    """
    __slots__ = ('route_by_id', 'ordered_study_ids_sha256', 'routing_array_sha256', '_counts', '_descriptor', '_pins', '_receipt')

    def __init__(self, token, route_by_id, order_ids_sha256, routing_array_sha256, counts, descriptor, pins, receipt):
        require(token is _VERIFIED, 'VerifiedRouting is created only by load_routing_artifacts')
        values = (types.MappingProxyType({str(k): str(v) for k, v in dict(route_by_id).items()}), order_ids_sha256, routing_array_sha256,
                  json.dumps(counts, sort_keys=True), json.dumps(descriptor, sort_keys=True), json.dumps(dict(pins), sort_keys=True),
                  json.dumps(receipt, sort_keys=True))
        for name, value in zip(self.__slots__, values):
            object.__setattr__(self, name, value)
        self.revalidate()

    def __setattr__(self, name, value):
        raise AttributeError('VerifiedRouting is immutable')

    def __delattr__(self, name):
        raise AttributeError('VerifiedRouting is immutable')

    @property
    def population_route_counts(self):
        return json.loads(self._counts)

    @property
    def descriptor(self):
        return json.loads(self._descriptor)

    @property
    def pins(self):
        return json.loads(self._pins)

    @property
    def receipt(self):
        return json.loads(self._receipt)

    def revalidate(self):
        """Recompute the content binding from the payload and compare it with the stored values and the pins."""
        ids_hash, routing_hash, counts = route_hashes(self.route_by_id)
        pins = self.pins
        require(all(r in ROUTES for r in self.route_by_id.values()), 'Unknown route in the verified routing')
        require(ids_hash == self.ordered_study_ids_sha256 == pins['ordered_study_ids_sha256'], 'Verified routing ID order no longer matches its hash')
        require(routing_hash == self.routing_array_sha256 == pins['routing_array_sha256'], 'Verified routing routes no longer match their hash')
        require(counts == self.population_route_counts, 'Verified routing counts no longer match its routes')

    def __len__(self):
        return len(self.route_by_id)


def descriptor_summary(summary):
    """Recompute the frozen 99% per-plane cohort descriptor from the audit's counts; refuse an inconsistent or relaxed one."""
    require(isinstance(summary, dict) and summary.get('threshold_fraction') == DESCRIPTOR_MIN_FRACTION,
            'Cohort descriptor threshold differs from the frozen 99%')
    planes = summary.get('per_plane')
    require(isinstance(planes, dict) and set(planes) == set(ROUTING_PLANES), 'Cohort descriptor planes differ')
    out = {}
    for plane in ROUTING_PLANES:
        row = planes[plane]
        studies, ok = row.get('studies'), row.get('all_ok_standard')
        require(type(studies) is int and type(ok) is int and 0 < studies and 0 <= ok <= studies, 'Cohort descriptor counts invalid')
        share = ok / studies
        require(abs(share - row.get('fraction', -1.0)) <= 1e-12, 'Cohort descriptor share differs from its counts')
        passes = share >= DESCRIPTOR_MIN_FRACTION
        flag = row.get('at_or_above_threshold')
        require(flag is passes or (flag is None and studies < summary.get('min_studies', 0)), 'Cohort descriptor pass flag differs from its counts')
        out[plane] = {'all_standard_share': share, 'all_ok_standard': ok, 'studies': studies, 'passes_99_percent': passes}
    return {'threshold_fraction': DESCRIPTOR_MIN_FRACTION, 'per_plane': out,
            'all_planes_pass': all(v['passes_99_percent'] for v in out.values()), 'role': 'descriptive only; per-study routing governs'}


def load_routing_artifacts(aggregate, routing, contract, pins=ORIENTATION_PINS):
    """Strict consumption of the pinned header-audit artifacts; fail closed, never print study identifiers.

    Checks, in order: the three file hashes against the pins (one read each), the canonical contract hash both against the
    pin and against the hash the aggregate refers to, the aggregate's integrity fields (complete, label-blind,
    no identifiers), the private routing file structure (known routes only, unique IDs, equal-length arrays), the recomputed
    ordered_study_ids_sha256 and routing_array_sha256 against both the aggregate and the pins (an altered ID order or route
    changes them), and the aggregate's route counts against the recomputed ones. Returns a VerifiedRouting.
    """
    pins = dict(pins)
    contract_bytes = read_pinned_file(contract, pins['contract_file'], pins['contract_file_sha256'])
    aggregate_bytes = read_pinned_file(aggregate, pins['aggregate_file'], pins['aggregate_file_sha256'])
    routing_bytes = read_pinned_file(routing, pins['routing_file'], pins['routing_file_sha256'])
    contract_doc = parse_json(contract_bytes, 'contract')
    agg_doc, route_doc = parse_json(aggregate_bytes, 'aggregate'), parse_json(routing_bytes, 'routing file')
    require(isinstance(contract_doc, dict) and isinstance(agg_doc, dict) and isinstance(route_doc, dict), 'Orientation artifacts must be objects')
    canonical_contract = sha256_bytes(json.dumps(contract_doc, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    config, body = agg_doc.get('config'), agg_doc.get('aggregate')
    require(isinstance(config, dict) and isinstance(body, dict), 'Orientation aggregate structure differs')
    require(canonical_contract == pins['contract_canonical_sha256'], 'Contract hash differs from its pin')
    require(config.get('contract_sha256') == canonical_contract, 'Aggregate refers to a different contract')
    require(contract_doc.get('version') == pins['orientation_version'] == agg_doc.get('version') == route_doc.get('version'),
            'Orientation version differs')
    require(config.get('split') == pins['split'] == route_doc.get('split'), 'Orientation split differs')
    require(config.get('series_csv_sha256') == contract_doc.get('train_series_csv_sha256'), 'Series table differs from the contract')
    require(agg_doc.get('status') == 'COMPLETE' and agg_doc.get('studies_done') == agg_doc.get('studies_planned') == pins['studies'],
            'Orientation audit is not complete')
    require(agg_doc.get('labels_read') is False and agg_doc.get('pixels_decoded') is False and agg_doc.get('identifiers_in_aggregate') is False,
            'Orientation audit is not label-blind and identifier-free')
    require(agg_doc.get('private_routing_file') == pins['routing_file'] and route_doc.get('private') is True, 'Routing file is not the audit private file')
    require(route_doc.get('routes') == list(ROUTES), 'Route vocabulary differs')
    ids, routes = route_doc.get('study_ids'), route_doc.get('routing')
    require(isinstance(ids, list) and isinstance(routes, list) and len(ids) == len(routes) == pins['studies'] == body.get('studies_total'),
            'Routing arrays must be equal length and cover every audited study')
    require(all(type(i) is str and i for i in ids), 'Routing study IDs must be non-empty strings')
    require(len(set(ids)) == len(ids), 'Duplicate study IDs in the routing file')
    unknown = sum(r not in ROUTES for r in routes)
    require(not unknown, f'Unknown route strings in the routing file ({unknown})')
    for key in ('slot_route', 'slot_status'):
        require(isinstance(route_doc.get(key), list) and len(route_doc[key]) == len(ids), 'Per-slot routing arrays differ in length')
    ids_hash = sha256_bytes('\n'.join(ids).encode())
    routing_hash = sha256_bytes(json.dumps(routes).encode())
    require(ids_hash == body.get('ordered_study_ids_sha256') == pins['ordered_study_ids_sha256'], 'Study ID order differs from its hash')
    require(routing_hash == body.get('routing_array_sha256') == pins['routing_array_sha256'], 'Route array differs from its hash')
    counts = {r: routes.count(r) for r in ROUTES}
    require(counts == body.get('study_routing'), 'Route counts differ from the aggregate')
    descriptor = descriptor_summary(agg_doc.get('descriptive_summary'))
    receipt = {'orientation_version': pins['orientation_version'], 'split': pins['split'], 'studies': len(ids),
               'file_sha256': {pins['aggregate_file']: pins['aggregate_file_sha256'], pins['routing_file']: pins['routing_file_sha256'],
                               pins['contract_file']: pins['contract_file_sha256']},
               'contract_canonical_sha256': canonical_contract, 'ordered_study_ids_sha256': ids_hash, 'routing_array_sha256': routing_hash,
               'pins_are_frozen_default': pins == ORIENTATION_PINS}
    return VerifiedRouting(_VERIFIED, dict(zip(ids, routes)), ids_hash, routing_hash, counts, descriptor, pins, receipt)


def join_routes(ids, verified):
    """Explicit, checked join of routes to the evaluation study IDs by ID (never by position); missing routes are refused."""
    require(isinstance(verified, VerifiedRouting), 'Routes must come from load_routing_artifacts')
    verified.revalidate()           # use-time content binding: the payload must still hash to the pinned values
    ids = [str(i) for i in ids]
    require(ids and len(ids) == len(set(ids)), 'Unique non-empty evaluation IDs required for the route join')
    missing = sum(i not in verified.route_by_id for i in ids)
    require(not missing, f'Missing routes for {missing} evaluation studies')
    routes = [verified.route_by_id[i] for i in ids]
    require(all(r in ROUTES for r in routes), 'Unknown route for an evaluation study')
    return routes


def membership(ids, routes):
    """Full-population and eligible (mirror_ok) subset counts and order-sensitive hashes, plus route counts. No identifiers."""
    eligible = [i for i, r in zip(ids, routes) if r == ELIGIBLE_ROUTE]
    return {'full_population': {'studies': len(ids), 'ids_sha256': smoke.digest(smoke.canonical(list(ids)))},
            'eligible_subset': {'rule': f'route == {ELIGIBLE_ROUTE}', 'studies': len(eligible), 'ids_sha256': smoke.digest(smoke.canonical(eligible))},
            'route_counts': {r: routes.count(r) for r in ROUTES}}


# --------------------------------------------------------------------------- descriptive plain vs gated comparison (computes only)
def auc_binary(y, p):
    """Rank-based (Mann-Whitney, ties averaged) AUC; None when a class is absent. Pure NumPy, nothing is selected."""
    import numpy as np
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if not pos or not neg:
        return None
    order = np.argsort(p, kind='mergesort')
    sp = p[order]
    ranks = np.empty(len(p), np.float64)
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def compare_plain_vs_gated(labels, label_mask, plain, gated, eligible):
    """Per-target AUC and support, full population vs eligible subset, plain K94 vs gated TTA. Computes only.

    labels/label_mask (n,12): 0/1 labels and 1 where the target is defined for that study (supplied later by the caller);
    plain/gated (n,12): probabilities in [0, 1]; eligible (n,) bool. Inputs are never mutated. gated must equal plain
    byte-for-byte on every non-eligible row. Differences are descriptive: nothing is selected, tuned or thresholded, and
    the macro is the unweighted mean over targets whose AUC is defined (undefined targets are listed, not imputed).
    """
    import numpy as np
    y, m, a, b = np.asarray(labels), np.asarray(label_mask), np.asarray(plain), np.asarray(gated)
    e = np.asarray(eligible)
    n = len(e)
    require(e.dtype == np.bool_ and e.shape == (n,), 'eligible must be a boolean vector')
    require(y.shape == m.shape == a.shape == b.shape == (n, 12), 'Labels, masks and predictions must be (n, 12)')
    require(np.isin(y, [0, 1]).all() and np.isin(m, [0, 1]).all(), 'Labels and masks must be binary')
    for arr in (a, b):
        require(np.isfinite(arr).all() and ((arr >= 0) & (arr <= 1)).all(), 'Predictions must be finite probabilities')
    require(a.dtype == b.dtype and a[~e].tobytes() == b[~e].tobytes(), 'Gated predictions must equal plain predictions on non-eligible rows')
    result = {'policy': 'descriptive only; per-target AUC; macro = unweighted mean over targets with both classes; no selection or tuning',
              'targets': list(TARGETS)}
    for name, rows in (('full_population', np.ones(n, bool)), ('eligible_subset', e)):
        arms = {}
        for arm, pred in (('plain', a), ('gated', b)):
            per_target = {}
            for t, target in enumerate(TARGETS):
                use = rows & (m[:, t] == 1)
                yt = y[use, t]
                per_target[target] = {'auc': auc_binary(yt, pred[use, t]), 'support': int(use.sum()),
                                      'positives': int((yt == 1).sum()), 'negatives': int((yt == 0).sum())}
            defined = [v['auc'] for v in per_target.values() if v['auc'] is not None]
            arms[arm] = {'per_target': per_target, 'macro_auc': float(np.mean(defined)) if defined else None,
                         'defined_targets': len(defined), 'undefined_targets': [k for k, v in per_target.items() if v['auc'] is None]}
        delta = {k: (None if arms['gated']['per_target'][k]['auc'] is None or arms['plain']['per_target'][k]['auc'] is None
                     else arms['gated']['per_target'][k]['auc'] - arms['plain']['per_target'][k]['auc']) for k in TARGETS}
        result[name] = {'studies': int(rows.sum()), **arms, 'gated_minus_plain_auc': delta}
    return result


# --------------------------------------------------------------------------- driver
def run_secondary(*, work, fold, checkpoint, definitions, vols, masks, ids, id2row, out, device, fp16=True, res=RES,
                  batch_studies=4, k94_win_chunk=16, cpu_threads=1, saved_oof=None, parity_atol=PARITY_ATOL,
                  enforce_parity=True, report_every=8, clock=time.monotonic, mode=MODE_PLAIN, routing=None):
    """Produce the outputs of one mode for ids from one frozen checkpoint; write private arrays plus an aggregate receipt.

    mode: plain_only (k16_plain, k94_plain; routing must be None and is never read), header_gated_approx_lr_tta (adds
    k94_header_gated_approx_lr_tta; routing must be a VerifiedRouting) or ungated_mirror_tta (LEGACY: adds the ungated
    k94_mirror_tta, never a default).  In the gated mode the route join by study ID
    runs before the output directory or the corpus is touched; only mirror_ok studies are mirrored, every other study keeps
    its plain K94 row byte-for-byte, and the full membership is preserved.
    The checkpoint is refused (before the corpus is touched) unless export_guard(secondary=True) accepts it;
    the CLI additionally binds it before it opens the corpus at all, and this call re-binds immediately
    before inference.  The receipt marks the mirror orientation and universal geometry as unverified.
    K16 uses the primary export's batch grouping (batch_studies studies per forward, win_chunk 0). K94 views
    run one study per forward with win_chunk=k94_win_chunk to bound memory; chunking only changes backbone
    call grouping.
    Order: (1) the complete K16 pass over the full fixed membership; (2) the K16 parity gate against the saved primary export
    (tolerance parity_atol); (3) only then K94 plain and the mirrored view; (4) all arm outputs are written together at the end.
    An enforced parity mismatch raises ValueError after writing ONLY receipt.json (status PARITY_FAIL, the parity diagnostics,
    the K16 window plan): no K94 or mirror window is planned or forwarded, and no arm file is written, K16 included.  A corrupt
    reference raises before any K94 work whether or not parity is enforced.  With enforce_parity=False the mismatch is recorded
    in the receipt and the run continues with phase (3).
    """
    import numpy as np
    import torch
    started = clock()
    require(mode in MODE_ARMS, 'Unknown mode')
    if mode == MODE_PLAIN:
        require(routing is None, 'plain_only mode must not be given routing artifacts')
    elif mode == MODE_GATED:
        require(isinstance(routing, VerifiedRouting), 'Header-gated mode requires verified routing artifacts')
    else:
        require(routing is None, 'The legacy ungated mode takes no routing artifacts')
    state, binding = bind_checkpoint(work, fold, checkpoint, res)
    require(device in ('cpu', 'mps', 'cuda'), 'Explicit device required: cpu, mps or cuda (no auto, no fallback)')
    device = select_device(device)
    ids = [str(i) for i in ids]
    require(ids and len(ids) == len(set(ids)) and set(ids) <= set(id2row), 'Unique evaluation IDs inside the corpus required')
    require(vols.shape[1] == DEPTH and masks.shape[1] == DEPTH, 'Corpus depth must be 96')
    require(type(batch_studies) is int and batch_studies >= 1 and type(k94_win_chunk) is int and k94_win_chunk >= 0, 'Invalid batching')
    n = len(ids)
    routes = join_routes(ids, routing) if mode == MODE_GATED else None
    # rows that receive the mirrored K94 view: all (legacy), only mirror_ok (gated), none (plain_only)
    elig_idx = [] if mode == MODE_PLAIN else list(range(n)) if mode == MODE_UNGATED else [j for j, r in enumerate(routes) if r == ELIGIBLE_ROUTE]
    local = {j: i for i, j in enumerate(elig_idx)}
    elig_ids = [ids[j] for j in elig_idx]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    arms = MODE_ARMS[mode]
    receipt = {'version': VERSION if mode == MODE_UNGATED else VERSION_ROUTED, 'mode': mode, 'status': 'RUNNING', 'role': ROLE,
               'optimizer_updates': 0, 'binding': binding,
               'slot_layout': slot_layout(), 'mirror_convention': MIRROR_CONVENTION, 'studies': n,
               'mirror_orientation_verified': MIRROR_ORIENTATION_VERIFIED, 'universal_geometry_verified': UNIVERSAL_GEOMETRY_VERIFIED,
               'mirror_orientation_assumption': MIRROR_ORIENTATION_ASSUMPTION,
               'ids_sha256': smoke.digest(smoke.canonical(ids)), 'fold': fold, 'completed_studies': 0,
               'settings': {'device': device, 'fp16_autocast_cuda_only': bool(fp16), 'res': res, 'batch_studies': batch_studies,
                            'k16_win_chunk': 0, 'k94_win_chunk': k94_win_chunk, 'cpu_threads': cpu_threads},
               'trainer_source_sha256': definitions.source_sha256,
               'code_sha256': {Path(m.__file__).name: smoke.file_sha(m.__file__) for m in (sys.modules[__name__], guard, pilot)}}
    if mode == MODE_UNGATED:
        receipt['ungated'] = True
        receipt['orientation_audit_required_before_tta_interpretation'] = ORIENTATION_AUDIT_REQUIRED
    elif mode == MODE_PLAIN:
        receipt['routing'] = {'used': False, 'note': 'plain_only: no mirror, no routing artifacts read'}
    else:
        receipt['routing'] = {'used': True, 'gate': f'{ELIGIBLE_ROUTE} -> equal 50:50 mean of plain and approximate-L/R-mirrored K94; '
                                                     'bypass_plain and needs_transform -> the identical plain K94 row; nothing dropped',
                              'note': GATED_NOTE, **membership(ids, routes), 'artifacts': routing.receipt,
                              'audit_population_route_counts': routing.population_route_counts, 'cohort_descriptor_99_percent': routing.descriptor}

    def write():
        receipt['elapsed_seconds'] = clock() - started
        receipt['peak_host_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        save_json(out / 'receipt.json', receipt)

    write()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(cpu_threads)
        model = build_model(definitions, state, device)
        del state
        digest_before = guard.tensor_hash(model.state_dict())
        plain_v, plain_m = vols, masks
        mirror_v, mirror_m = MappedCorpus(vols, mirror_volume), MappedCorpus(masks, mirror_mask)
        s16 = make_sampler(definitions, ids, id2row, plain_v, plain_m, 16, res)
        s94 = make_sampler(definitions, ids, id2row, plain_v, plain_m, 94, res)
        s94m = make_sampler(definitions, elig_ids, id2row, mirror_v, mirror_m, 94, res) if elig_ids else None
        views = ['k16_plain', 'k94_plain'] + (['k94_mirror_view'] if mode != MODE_PLAIN else [])
        preds = {name: [] for name in views}
        stats = {name: {} for name in preds}
        plans = {name: [] for name in preds}
        seconds = dict.fromkeys(preds, 0.0)

        def plan(name, mask, sampler, k):
            centers, flags = center_plan(mask, sampler, k)
            plans[name].append(centers)
            for key, value in flags.items():
                stats[name][key] = stats[name].get(key, 0) + value

        batches = [list(range(start, min(start + batch_studies, n))) for start in range(0, n, batch_studies)]
        # Phase 1: the full fixed-membership K16 pass (the primary export's arithmetic) and nothing else.
        model.win_chunk = 0
        for number, chunk in enumerate(batches, 1):
            for j in chunk:
                row = id2row[ids[j]]
                validate_study(vols[row], masks[row])
                plan('k16_plain', masks[row], s16, 16)
            t0 = clock()
            x16 = torch.stack([checked_windows(s16[j][0], 16, res) for j in chunk])
            preds['k16_plain'].extend(predict(model, x16, device, fp16))
            seconds['k16_plain'] += clock() - t0
            receipt['k16_completed_studies'] = chunk[-1] + 1
            if number % report_every == 0:
                write()
        p16 = probabilities(np.asarray(preds['k16_plain'], dtype=np.float32), n)
        # K16 parity gate: decided on the complete K16 output, before any K94 or mirror window is planned or forwarded and
        # before any arm output is written. A corrupt reference raises here as well (enforced or not).
        receipt['parity'] = {'k16_vs_primary': 'not requested'}
        if saved_oof is not None:
            report = parity_report(p16, saved_primary(saved_oof, ids), parity_atol, enforce_parity)
            receipt['parity'] = {'k16_vs_primary': report, 'reference_file_sha256': smoke.file_sha(saved_oof)}
            if enforce_parity and not report['within_tolerance']:
                receipt.update(status='PARITY_FAIL', outputs={}, k94_and_mirror_views_run=False,
                               window_plans={'k16_plain': {**stats['k16_plain'], 'plans_sha256': smoke.digest(smoke.canonical(plans['k16_plain']))}},
                               failure='K16 output differs from the primary export beyond tolerance; no K94 or mirror view was run and no arm output was written',
                               environment=environment(device))
                write()
                raise ValueError('K16 output differs from the primary export beyond tolerance')
        # Phase 2: K94 plain (and the mirrored view for the eligible studies) only after the gate; with enforce_parity=False a
        # mismatch is recorded in the receipt and the run continues.
        model.win_chunk = k94_win_chunk
        for chunk in batches:
            for j in chunk:
                row = id2row[ids[j]]
                plan('k94_plain', masks[row], s94, 94)
                if j in local:
                    plan('k94_mirror_view', mirror_mask(masks[row]), s94m, 94)
            for j in chunk:
                for name, sampler, index in (('k94_plain', s94, j), ('k94_mirror_view', s94m, local.get(j))):
                    if name not in preds or index is None:
                        continue
                    t0 = clock()
                    x = checked_windows(sampler[index][0], 94, res).unsqueeze(0)
                    preds[name].append(predict(model, x, device, fp16)[0])
                    seconds[name] += clock() - t0
            receipt['completed_studies'] = chunk[-1] + 1
            if (chunk[0] // batch_studies + 1) % report_every == 0:
                write()
        p94 = probabilities(np.asarray(preds['k94_plain'], dtype=np.float32), n)
        digest_after = guard.tensor_hash(model.state_dict())
        sha_after = smoke.file_sha(checkpoint)
        require(digest_after == digest_before, 'Model parameters/buffers changed')
        require(sha_after == binding['checkpoint_sha256'], 'Frozen checkpoint mutated during inference')
        binding.update(model_state_sha256=digest_before, model_state_sha256_after_run=digest_after, checkpoint_sha256_after_run=sha_after)
        sha = binding['checkpoint_sha256']
        sets = [('k16_plain', p16, 16, False, {}), ('k94_plain', p94, 94, False, {})]
        extra = {}
        if mode != MODE_PLAIN:
            pm = probabilities(np.asarray(preds['k94_mirror_view'], dtype=np.float32).reshape(-1, 12), len(elig_idx))
            tta = p94.copy()                 # every non-eligible row is the plain K94 row itself
            if elig_idx:
                tta[elig_idx] = (0.5 * (p94[elig_idx] + pm)).astype(np.float32)
            tta = probabilities(tta, n)
            bypass = [j for j in range(n) if j not in local]
            require(tta[bypass].tobytes() == p94[bypass].tobytes(), 'Bypassed studies must equal plain K94 byte-for-byte')
            if mode == MODE_UNGATED:
                sets.append(('k94_mirror_tta', tta, 94, True, {'plain_view': p94, 'mirror_view': pm,
                                                                'mirror_orientation_verified': np.asarray(MIRROR_ORIENTATION_VERIFIED)}))
                extra = {'aggregation': 'mean of k94_plain and mirrored-view probabilities, equal weights', 'ungated': True,
                         'mirror_orientation_verified': MIRROR_ORIENTATION_VERIFIED}
            else:
                eligible = np.zeros(n, bool)
                eligible[elig_idx] = True
                sets.append((GATED_ARM, tta, 94, True, {'plain_view': p94, 'mirror_view_eligible': pm, 'eligible': eligible,
                                                         'routes': np.asarray(routes), 'mirror_orientation_verified': np.asarray(MIRROR_ORIENTATION_VERIFIED),
                                                         'universal_geometry_verified': np.asarray(UNIVERSAL_GEOMETRY_VERIFIED)}))
                extra = {'aggregation': 'mirror_ok studies: equal 50:50 mean of plain and approximate-L/R-mirrored K94; every other study: '
                                        'its plain K94 row, byte-identical to k94_plain',
                         'mirrored_studies': len(elig_idx), 'plain_passthrough_studies': len(bypass),
                         'bypass_rows_byte_identical_to_k94_plain': True, 'full_membership_preserved': True,
                         'mirror_orientation_verified': MIRROR_ORIENTATION_VERIFIED,
                         'universal_geometry_verified': UNIVERSAL_GEOMETRY_VERIFIED}
        outputs = {}
        for arm, values, k, mirrored, arrays in sets:
            write_npz(out / f'{arm}-private.npz', ids, values, sha, **arrays)
            write_csv(out / f'{arm}.csv', ids, values)
            outputs[arm] = {'k': k, 'mirror_tta': mirrored, 'checkpoint_sha256': sha, 'rows': n,
                            'npz_sha256': smoke.file_sha(out / f'{arm}-private.npz'), 'csv_sha256': smoke.file_sha(out / f'{arm}.csv'),
                            'seconds': seconds['k94_plain'] + seconds['k94_mirror_view'] if mirrored else seconds[arm]}
        if extra:
            outputs[arms[-1]].update(extra)
        require(tuple(outputs) == arms, 'Output arm set differs from the mode')
        require({o['checkpoint_sha256'] for o in outputs.values()} == {sha}, 'Outputs must share one checkpoint')
        binding['all_outputs_same_checkpoint'] = True
        receipt.update(outputs=outputs, window_plans={name: {**stats[name], 'plans_sha256': smoke.digest(smoke.canonical(plans[name]))} for name in preds})
        receipt['status'] = 'PASS'
        receipt['environment'] = environment(device)
        write()
        return receipt
    except BaseException as error:
        if receipt['status'] == 'RUNNING':
            receipt.update(status='FAIL', failure_type=type(error).__name__)
            write()
        raise
    finally:
        torch.set_num_threads(previous_threads)


def environment(device):
    import numpy
    import torch
    packages = {'torch': torch.__version__, 'numpy': numpy.__version__}
    try:
        packages['timm'] = importlib.metadata.version('timm')
    except importlib.metadata.PackageNotFoundError:
        packages['timm'] = None
    result = {'python': sys.version.split()[0], 'packages': packages, 'device': device}
    if device == 'cuda':
        result.update(device_name=torch.cuda.get_device_name(0), cuda=torch.version.cuda, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work', type=Path, required=True, help='trainer work dir: swa checkpoint, run_fold0.json, raptor_ft_fold0.json, folds_v0.json, excluded_ids.json, oof_fold0.csv')
    parser.add_argument('--fold', type=int, default=0, choices=[0])
    parser.add_argument('--trainer-source', type=Path, required=True)
    parser.add_argument('--trainer-sha256', default=smoke.SOURCE_SHA, help='pinned digest of the trainer source (package manifest patched_sha256)')
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], required=True)
    parser.add_argument('--k94-win-chunk', type=int, default=16)
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument('--plain-only', action='store_true', help='K16 parity and K94 plain only: no mirror, no routing artifacts are read')
    kind.add_argument('--header-gated-tta', action='store_true',
                      help='add k94_header_gated_approx_lr_tta: mirror only studies the pinned header audit routed to mirror_ok')
    kind.add_argument('--ungated-mirror-tta', action='store_true',
                      help='LEGACY, UNGATED: mirror every study regardless of headers (k94_mirror_tta); historical comparison only')
    parser.add_argument('--orientation-aggregate', type=Path, help='pinned train-orientation.json (header-gated mode only)')
    parser.add_argument('--orientation-routing', type=Path, help='pinned routing-private-train.json (header-gated mode only)')
    parser.add_argument('--orientation-contract', type=Path, help='pinned build-contract.json (header-gated mode only)')
    parser.add_argument('--skip-parity', action='store_true', help='do not compare K16 with the primary export oof')
    parser.add_argument('--no-enforce-parity', action='store_true', help='record the parity error without failing the run')
    args = parser.parse_args(argv)
    mode = MODE_PLAIN if args.plain_only else MODE_GATED if args.header_gated_tta else MODE_UNGATED
    artifact_paths = (args.orientation_aggregate, args.orientation_routing, args.orientation_contract)
    if mode == MODE_GATED and any(p is None for p in artifact_paths):
        parser.error('--header-gated-tta requires --orientation-aggregate, --orientation-routing and --orientation-contract')
    if mode != MODE_GATED and any(p is not None for p in artifact_paths):
        parser.error('orientation artifacts are accepted only with --header-gated-tta')
    work = args.work
    checkpoint = work / f'raptor_ft_fold{args.fold}_swa.pt'
    # Guard first: bind and qualify the frozen primary (export_guard(secondary=True) plus checkpoint hash) and check the
    # pinned trainer definitions BEFORE anything opens the corpus. guard.staged_corpus() and
    # guard.verify_export_inputs() (which opens it) come only afterwards. run_secondary re-binds just before inference.
    bind_checkpoint(work, args.fold, checkpoint)
    # Routing artifacts need no corpus: verify them (hashes, contract, ID order) now; the by-ID join happens inside run_secondary.
    routing = load_routing_artifacts(*artifact_paths) if mode == MODE_GATED else None
    definitions = load_definitions(args.trainer_source.read_text(), args.trainer_sha256)
    vols, masks, all_ids = guard.staged_corpus()
    all_ids = [str(i) for i in all_ids]
    folds = json.loads((work / 'folds_v0.json').read_bytes())['folds']
    excluded = set(json.loads((work / 'excluded_ids.json').read_bytes())['quarantined_study_ids'])
    dev = development_ids(all_ids, folds, excluded, args.fold)
    guard.verify_export_inputs(args.labels, args.truth, work / 'folds_v0.json', work / 'excluded_ids.json', dev)
    receipt = run_secondary(work=work, fold=args.fold, checkpoint=checkpoint, definitions=definitions,
                            vols=vols, masks=masks, ids=dev, id2row={u: i for i, u in enumerate(all_ids)}, out=args.out, device=args.device,
                            k94_win_chunk=args.k94_win_chunk, saved_oof=None if args.skip_parity else work / f'oof_fold{args.fold}.csv',
                            enforce_parity=not args.no_enforce_parity, mode=mode, routing=routing)
    keys = ('status', 'mode', 'studies', 'mirror_orientation_verified', 'universal_geometry_verified', 'outputs', 'parity', 'elapsed_seconds')
    print(json.dumps({k: receipt[k] for k in keys}, indent=2))


if __name__ == '__main__':
    main()
