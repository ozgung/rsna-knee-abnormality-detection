"""Secondary frozen-checkpoint inference for the own224-d96 top-3 SWA model.

One checkpoint, one pass over the same studies, three separate prediction sets:
  k16_plain       K16, no mirror. Same arithmetic as the primary export (parity check).
  k94_plain       K94, no mirror.
  k94_mirror_tta  K94 with L/R mirror TTA: mean of plain and mirrored K94. The mirror is the anatomical L/R
                  mirror under the ASSUMED standard orientation; that assumption is UNVERIFIED (no label-blind
                  IOP orientation audit has been run), so no TTA result may be interpreted before that audit.

All three are SECONDARY diagnostics. The frozen K16 primary endpoint governs promotion;
nothing here selects epochs, weights, targets or a blend, and nothing here trains.
Models and torch are imported lazily so the dependency-free audit path is unchanged.
"""
import argparse
import ast
import csv
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
ARMS = ('k16_plain', 'k94_plain', 'k94_mirror_tta')
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
MIRROR_ORIENTATION_ASSUMPTION = (
    'Anatomical L/R mirror under the assumed standard orientation (unverified): sagittal slice axis = patient L/R, '
    'coronal and axial image width = patient L/R. The cache stores no ImageOrientationPatient and no real DICOM was '
    'inspected, so oblique or nonstandard series are not excluded.')
ORIENTATION_AUDIT_REQUIRED = (
    'Before any k94_mirror_tta number is interpreted: run a label-blind header audit over the training series that tallies, '
    'per plane, the dominant axis (argmax of the absolute direction cosine) of the IOP row direction, the IOP column '
    'direction and their cross-product normal, with oblique (no clearly dominant axis) and nonstandard cases counted '
    'separately, and stratify or exclude studies whose series do not match the assumed axes.')
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


# --------------------------------------------------------------------------- driver
def run_secondary(*, work, fold, checkpoint, definitions, vols, masks, ids, id2row, out, device, fp16=True, res=RES,
                  batch_studies=4, k94_win_chunk=16, cpu_threads=1, saved_oof=None, parity_atol=PARITY_ATOL,
                  enforce_parity=True, report_every=8, clock=time.monotonic):
    """Produce the three outputs for ids from one frozen checkpoint; write private arrays plus an aggregate receipt.

    The checkpoint is refused (before the corpus is touched) unless export_guard(secondary=True) accepts it;
    the CLI additionally binds it before it opens the corpus at all, and this call re-binds immediately
    before inference.  The receipt marks the mirror orientation as unverified.
    K16 uses the primary export's batch grouping (batch_studies studies per forward, win_chunk 0). K94 views
    run one study per forward with win_chunk=k94_win_chunk to bound memory; chunking only changes backbone
    call grouping. Raises ValueError after saving everything when a requested K16 parity check fails.
    """
    import numpy as np
    import torch
    started = clock()
    state, binding = bind_checkpoint(work, fold, checkpoint, res)
    require(device in ('cpu', 'mps', 'cuda'), 'Explicit device required: cpu, mps or cuda (no auto, no fallback)')
    device = select_device(device)
    ids = [str(i) for i in ids]
    require(ids and len(ids) == len(set(ids)) and set(ids) <= set(id2row), 'Unique evaluation IDs inside the corpus required')
    require(vols.shape[1] == DEPTH and masks.shape[1] == DEPTH, 'Corpus depth must be 96')
    require(type(batch_studies) is int and batch_studies >= 1 and type(k94_win_chunk) is int and k94_win_chunk >= 0, 'Invalid batching')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    n = len(ids)
    receipt = {'version': VERSION, 'status': 'RUNNING', 'role': ROLE, 'optimizer_updates': 0, 'binding': binding,
               'slot_layout': slot_layout(), 'mirror_convention': MIRROR_CONVENTION, 'studies': n,
               'mirror_orientation_verified': MIRROR_ORIENTATION_VERIFIED,
               'mirror_orientation_assumption': MIRROR_ORIENTATION_ASSUMPTION,
               'orientation_audit_required_before_tta_interpretation': ORIENTATION_AUDIT_REQUIRED,
               'ids_sha256': smoke.digest(smoke.canonical(ids)), 'fold': fold, 'completed_studies': 0,
               'settings': {'device': device, 'fp16_autocast_cuda_only': bool(fp16), 'res': res, 'batch_studies': batch_studies,
                            'k16_win_chunk': 0, 'k94_win_chunk': k94_win_chunk, 'cpu_threads': cpu_threads},
               'trainer_source_sha256': definitions.source_sha256,
               'code_sha256': {Path(m.__file__).name: smoke.file_sha(m.__file__) for m in (sys.modules[__name__], guard, pilot)}}

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
        s94m = make_sampler(definitions, ids, id2row, mirror_v, mirror_m, 94, res)
        preds = {'k16_plain': [], 'k94_plain': [], 'k94_mirror_view': []}
        stats = {name: {} for name in preds}
        plans = {name: [] for name in preds}
        seconds = dict.fromkeys(preds, 0.0)

        def plan(name, mask, sampler, k):
            centers, flags = center_plan(mask, sampler, k)
            plans[name].append(centers)
            for key, value in flags.items():
                stats[name][key] = stats[name].get(key, 0) + value

        for start in range(0, n, batch_studies):
            chunk = list(range(start, min(start + batch_studies, n)))
            for j in chunk:
                row = id2row[ids[j]]
                validate_study(vols[row], masks[row])
                plan('k16_plain', masks[row], s16, 16)
                plan('k94_plain', masks[row], s94, 94)
                plan('k94_mirror_view', mirror_mask(masks[row]), s94m, 94)
            t0 = clock()
            x16 = torch.stack([checked_windows(s16[j][0], 16, res) for j in chunk])
            model.win_chunk = 0
            preds['k16_plain'].extend(predict(model, x16, device, fp16))
            seconds['k16_plain'] += clock() - t0
            model.win_chunk = k94_win_chunk
            for j in chunk:
                for name, sampler in (('k94_plain', s94), ('k94_mirror_view', s94m)):
                    t0 = clock()
                    x = checked_windows(sampler[j][0], 94, res).unsqueeze(0)
                    preds[name].append(predict(model, x, device, fp16)[0])
                    seconds[name] += clock() - t0
            receipt['completed_studies'] = chunk[-1] + 1
            if (start // batch_studies + 1) % report_every == 0:
                write()
        p16, p94, pm = (probabilities(np.asarray(preds[name], dtype=np.float32), n) for name in preds)
        tta = probabilities((0.5 * (p94 + pm)).astype(np.float32), n)
        digest_after = guard.tensor_hash(model.state_dict())
        sha_after = smoke.file_sha(checkpoint)
        require(digest_after == digest_before, 'Model parameters/buffers changed')
        require(sha_after == binding['checkpoint_sha256'], 'Frozen checkpoint mutated during inference')
        binding.update(model_state_sha256=digest_before, model_state_sha256_after_run=digest_after, checkpoint_sha256_after_run=sha_after)
        sha = binding['checkpoint_sha256']
        outputs = {}
        for arm, values, k, mirrored, views in (('k16_plain', p16, 16, False, {}), ('k94_plain', p94, 94, False, {}),
                                                ('k94_mirror_tta', tta, 94, True, {'plain_view': p94, 'mirror_view': pm,
                                                                    'mirror_orientation_verified': np.asarray(MIRROR_ORIENTATION_VERIFIED)})):
            write_npz(out / f'{arm}-private.npz', ids, values, sha, **views)
            write_csv(out / f'{arm}.csv', ids, values)
            outputs[arm] = {'k': k, 'mirror_tta': mirrored, 'checkpoint_sha256': sha, 'rows': n,
                            'npz_sha256': smoke.file_sha(out / f'{arm}-private.npz'), 'csv_sha256': smoke.file_sha(out / f'{arm}.csv'),
                            'seconds': seconds['k94_plain'] + seconds['k94_mirror_view'] if mirrored else seconds[arm]}
        outputs['k94_mirror_tta']['aggregation'] = 'mean of k94_plain and mirrored-view probabilities, equal weights'
        outputs['k94_mirror_tta']['mirror_orientation_verified'] = MIRROR_ORIENTATION_VERIFIED
        require({o['checkpoint_sha256'] for o in outputs.values()} == {sha}, 'Outputs must share one checkpoint')
        binding['all_outputs_same_checkpoint'] = True
        receipt.update(outputs=outputs, window_plans={name: {**stats[name], 'plans_sha256': smoke.digest(smoke.canonical(plans[name]))} for name in preds})
        receipt['parity'] = {'k16_vs_primary': 'not requested'}
        failure = None
        if saved_oof is not None:
            report = parity_report(p16, saved_primary(saved_oof, ids), parity_atol, enforce_parity)
            receipt['parity'] = {'k16_vs_primary': report, 'reference_file_sha256': smoke.file_sha(saved_oof)}
            if enforce_parity and not report['within_tolerance']:
                failure = 'K16 output differs from the primary export beyond tolerance'
        receipt['status'] = 'PARITY_FAIL' if failure else 'PASS'
        receipt['environment'] = environment(device)
        write()
        if failure:
            raise ValueError(failure)
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
    parser.add_argument('--skip-parity', action='store_true', help='do not compare K16 with the primary export oof')
    parser.add_argument('--no-enforce-parity', action='store_true', help='record the parity error without failing the run')
    args = parser.parse_args(argv)
    work = args.work
    checkpoint = work / f'raptor_ft_fold{args.fold}_swa.pt'
    # Guard first: bind and qualify the frozen primary (export_guard(secondary=True) plus checkpoint hash) and check the
    # pinned trainer definitions BEFORE anything opens the corpus. guard.staged_corpus() and
    # guard.verify_export_inputs() (which opens it) come only afterwards. run_secondary re-binds just before inference.
    bind_checkpoint(work, args.fold, checkpoint)
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
                            enforce_parity=not args.no_enforce_parity)
    print(json.dumps({k: receipt[k] for k in ('status', 'studies', 'mirror_orientation_verified', 'outputs', 'parity', 'elapsed_seconds')}, indent=2))


if __name__ == '__main__':
    main()
