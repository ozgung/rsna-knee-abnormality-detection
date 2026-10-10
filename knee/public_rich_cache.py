"""CPU-only exact public rich-input mechanics pilot; private index, aggregate receipt."""
from collections import Counter
import csv
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import sys
import time

from knee.reference import bounded_ids, canonical_bytes, sha256

SEED = 20261010
COUNT = 32
SERIES_FIELDS = ['StudyInstanceUID', 'SeriesInstanceUID', 'Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']


def select_inputs(root, targets, exclusions, count=COUNT):
    """Use target-cell presence only for gold membership, never numeric outcomes."""
    root = Path(root)
    with (root / 'train_series.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SERIES_FIELDS:
            raise ValueError('Official series schema changed')
        records, series = {}, set()
        for row in reader:
            if (set(row) != set(SERIES_FIELDS) or any(v is None for v in row.values()) or
                    row['Fluid_Sensitive'] not in ('0', '1') or row['Fat_Suppression'] not in ('0', '1') or
                    row['Anatomical_Plane'] not in ('Sagittal', 'Coronal', 'Axial')):
                raise ValueError('Invalid series metadata')
            for key in ('StudyInstanceUID', 'SeriesInstanceUID'):
                bounded_ids([row[key]], count=1)
            if row['SeriesInstanceUID'] in series:
                raise ValueError('Duplicate series')
            series.add(row['SeriesInstanceUID'])
            records.setdefault(row['StudyInstanceUID'], []).append(row)
    gold, official = set(), set()
    with (root / 'train.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        if not set(['StudyInstanceUID', *targets]) <= set(reader.fieldnames or []):
            raise ValueError('Official target schema changed')
        for row in reader:
            sid = row['StudyInstanceUID']
            bounded_ids([sid], count=1)
            if sid in official: raise ValueError('Duplicate official study')
            official.add(sid)
            present = [row[name] is not None and bool(row[name].strip()) for name in targets]
            if any(present) and not all(present): raise ValueError('Partial gold membership ambiguous')
            if all(present): gold.add(sid)
            # Report text and target magnitudes are never retained or interpreted.
    excluded = exclusions.get('quarantined_study_ids')
    if not isinstance(excluded, list) or len(excluded) != 4:
        raise ValueError('Exact four-study quarantine manifest required')
    bounded_ids(excluded, count=4)
    if (not set(excluded) <= official or set(excluded) & gold or
            set(records) != official or len(gold) != 58):
        raise ValueError('Official membership/exclusion coverage changed')
    ids = bounded_ids(sorted(official - gold - set(excluded)), count=count, seed=SEED)
    return ids, records, {'official_studies': len(official), 'gold_excluded': len(gold),
                          'quarantine_excluded': len(excluded)}


def verify_cache(public, volume, mask, path):
    import numpy as np
    import torch
    if volume.dtype != np.uint8 or volume.shape != (96, 384, 384) or mask.shape != (96,):
        raise ValueError('Rich volume geometry changed')
    if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError('Invalid or empty mask')
    if not np.array_equal(mask.astype(bool), volume.reshape(96, -1).any(axis=1)):
        raise ValueError('Pixel/mask consistency failed')
    path = Path(path)
    np.save(path.with_suffix('.volume.npy'), volume, allow_pickle=False)
    np.save(path.with_suffix('.mask.npy'), mask, allow_pickle=False)
    cached = np.load(path.with_suffix('.volume.npy'), allow_pickle=False)
    cached_mask = np.load(path.with_suffix('.mask.npy'), allow_pickle=False)
    if volume.tobytes() != cached.tobytes() or mask.dtype != cached_mask.dtype or mask.tobytes() != cached_mask.tobytes():
        raise ValueError('Cache byte parity failed')
    for mirror in (False, True):
        a = public.eval_windows(volume, mask, 94, 320, mirror=mirror)
        b = public.eval_windows(cached, cached_mask, 94, 320, mirror=mirror)
        if tuple(a.shape) != (94, 3, 320, 320) or not torch.isfinite(a).all() or not torch.equal(a, b):
            raise ValueError('Central320 CPU window parity failed')
        del a, b
    return {'volume_sha256': sha256(volume.tobytes()), 'mask_sha256': sha256(mask.tobytes()),
            'volume_dtype': str(volume.dtype), 'mask_dtype': str(mask.dtype)}


def run(public_source, decoder_source, manifest, root, out, exclusions, exclusion_sha256):
    import numpy as np
    import torch
    from knee.reference import load_definitions
    from knee.mirror_input import prepare_study, stress_inputs, policy_contract
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def version(package):
        try: return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: return None
    report = {'status': 'RUNNING', 'purpose': 'input mechanics only', 'models_loaded': 0,
              'source_sha256': manifest['source_sha256'], 'manifest_sha256': sha256(canonical_bytes(manifest)),
              'exclusion_sha256': exclusion_sha256, 'input_policy': policy_contract(),
              'source_bundle_sha256': __import__('os').environ.get('KNEE_SOURCE_SHA256'),
              'source_revision': __import__('os').environ.get('KNEE_CODE_REVISION'),
              'sampling_seed': SEED, 'requested_studies': COUNT, 'completed_studies': 0,
              'byte_parity': True, 'window_parity': True, 'repeat_decodes': 0,
              'storage_bytes384': 0, 'decode_seconds': 0.0, 'cache_window_seconds': 0.0, 'repeat_seconds': 0.0, 'central320_raw_storage_potential_bytes': COUNT * 96 * 320 * 320,
              'free_disk_before': shutil.disk_usage(out).free,
              'environment': {'python': sys.version, 'packages': {p: version(p) for p in
                 ('torch', 'numpy', 'pydicom', 'opencv-python', 'opencv-python-headless')}}}
    counters = Counter(); private = []
    def save():
        report.update(elapsed_seconds=time.monotonic()-started, counters=dict(counters),
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024))
        (out/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    save()
    try:
        torch.set_num_threads(4)
        public, decoder = load_definitions(public_source, decoder_source, manifest)
        ids, records, membership = select_inputs(root, public.LAB, exclusions)
        report.update(membership=membership, ordered_id_set_sha256=sha256(canonical_bytes(ids)),
                      series_metadata_sha256=sha256((Path(root)/'train_series.csv').read_bytes()))
        raw = Path(root)/'train_series'
        if not raw.is_dir() or any(not (raw/sid).is_dir() for sid in ids):
            raise ValueError('Selected raw directories missing')
        required = COUNT * (96 * 384 * 384 + 4096) + 512 * 1024 * 1024
        if shutil.disk_usage(out).free < required: raise OSError('Insufficient cache disk')
        report['synthetic_stress_checks'] = stress_inputs(public, decoder)
        for ordinal, sid in enumerate(ids):
            before = time.monotonic()
            volume, mask, counts = prepare_study(public, decoder, sid, records, raw)
            if counts.get('decoded_unique_slices', 0) <= 0: raise ValueError('No decode evidence')
            report['decode_seconds'] += time.monotonic() - before
            counters.update(counts)
            before = time.monotonic()
            hashes = verify_cache(public, volume, mask, out/f'{ordinal:03d}')
            report['cache_window_seconds'] += time.monotonic() - before
            if ordinal < 2:
                before = time.monotonic()
                repeated, repeated_mask, repeated_counts = prepare_study(public, decoder, sid, records, raw)
                if (repeated_counts.get('decoded_unique_slices', 0) <= 0 or not np.array_equal(volume, repeated) or
                        not np.array_equal(mask, repeated_mask)):
                    raise ValueError('Repeated decode parity failed')
                report['repeat_seconds'] += time.monotonic() - before
                report['repeat_decodes'] += 1
                del repeated, repeated_mask
            private.append({'StudyInstanceUID': sid, 'ordinal': ordinal, **hashes})
            report['completed_studies'] += 1
            report['storage_bytes384'] += sum((out/f'{ordinal:03d}.{kind}.npy').stat().st_size for kind in ('volume','mask'))
            del volume, mask
            save()
        (out/'private-index.json').write_text(json.dumps(private, indent=2)+'\n')
        report['private_index_sha256'] = sha256((out/'private-index.json').read_bytes())
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='FAIL', failure_type=type(error).__name__)
        # Never export exception text: public decoder paths may contain IDs.
        save()
        raise RuntimeError('Rich-cache pilot failed; inspect aggregate receipt') from None
    finally:
        report['free_disk_after'] = shutil.disk_usage(out).free
        save()
    return report
