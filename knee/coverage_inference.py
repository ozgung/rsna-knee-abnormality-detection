"""Image-only TRAIN04 EMA inference and raw/cache rehearsal; no fitting."""
import ast
from collections import Counter
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time

from knee.coverage_cache import VERSION, read_volumes, sample_windows, validate_volumes
from knee.fresh_reader import SOURCE_HASHES, check_sources, fingerprint, load_source, write_json
from knee.reference_blend import read_test_contract, write_probabilities
from knee.validation import file_sha256

POLICY = 'train04-coverage256-partial-v1'
TRAIN_COUNT = 64
PLATFORM_SECONDS = 9 * 3600
MODEL_RECIPE = {'backbone': 'convnext_tiny.fb_in22k_ft_in1k', 'image_size': 256,
                'slots': 6, 'eval_windows': 12, 'tta': 'none', 'weights': 'selected_ema'}
SERIES_COLUMNS = ['StudyInstanceUID', 'SeriesInstanceUID', 'Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']
CODECS = {
 'pylibjpeg-2.1.0-py3-none-any.whl': '25df9496a69e64e98c887fddee12a1271e275b5f74ba804f9bf98a08bb80993e',
 'pylibjpeg_libjpeg-2.4.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl': '01d950ef496476a9223e4966376cb88098fcf5c55a12a21f722d7b5f84daae43',
 'pylibjpeg_openjpeg-2.6.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl': 'aa6b08327be6f6daf0814a61fd7410f53a08e8e0b35774c634d9907402ff4f1c'}
STRESS_CASES = {'healthy_cache_parity', 'partial_unchanged', 'exception_masked', 'short_masked',
                'black_masked', 'all_empty_fails', 'memory_fatal', 'timeout_fatal', 'opencv_memory_fatal'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_manifest(manifest):
    require(manifest.get('version') == 1 and manifest.get('recipe') == MODEL_RECIPE and
            manifest.get('cache_version') == VERSION and manifest.get('external') == SOURCE_HASHES and
            manifest.get('fingerprint') == fingerprint({k: v for k, v in manifest.items() if k != 'fingerprint'}),
            'Own inference asset contract differs')
    require(len(manifest.get('targets', [])) == 12 and len(set(manifest['targets'])) == 12 and
            manifest.get('variant') in ('v1', 'consensus') and type(manifest.get('selected_epoch')) is int and
            1 <= manifest['selected_epoch'] <= 12, 'Own selected checkpoint metadata differs')
    for key in ('weights_sha256', 'training_checkpoint_sha256', 'training_contract_fingerprint',
                'plan_fingerprint', 'index_sha256', 'run_sha256', 'report_sha256', 'training_build_sha256',
                'training_notebook_sha256', 'training_source_sha256'):
        require(digest(manifest.get(key)), 'Missing own checkpoint provenance digest')
    require(isinstance(manifest.get('training_environment'), dict), 'Training environment missing')
    revision = manifest.get('training_revision', '')
    require(isinstance(revision, str) and len(revision) == 40 and all(c in '0123456789abcdef' for c in revision),
            'Training revision missing')


def fatal(error):
    import cv2
    return isinstance(error, (MemoryError, TimeoutError)) or (isinstance(error, cv2.error) and error.code == -4)


def protect_reader(prep, audit):
    """Prefix exactly the one public per-file catch; all scientific AST is intact.

    Protect the whole catch, including slices.extend and asynchronous deadlines,
    rather than merely wrapping _read. Original source file bytes remain pinned.
    """
    tree = ast.parse(inspect.getsource(prep.load_series))
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    require(len(handlers) == 1 and isinstance(handlers[0].type, ast.Name) and
            handlers[0].type.id == 'Exception' and handlers[0].name is None,
            'Public load_series exception structure changed')
    handler = handlers[0]; handler.name = '_coverage_error'
    handler.body.insert(0, ast.parse('if _coverage_fatal(_coverage_error):\n    raise').body[0])
    def counted_fatal(error):
        if fatal(error):
            return True
        return False
    namespace = dict(vars(prep), _coverage_fatal=counted_fatal)
    exec(compile(ast.fix_missing_locations(tree), '<coverage-protected-preprocess>', 'exec'), namespace)
    return namespace['load_series']


def resize_volume(volume):
    import cv2
    import numpy as np
    require(volume.dtype == np.uint8 and volume.ndim == 3 and volume.shape[1:] == (384, 384) and
            len(volume) <= 64, 'Public volume dtype/shape changed')
    return np.stack([cv2.resize(v, (256, 256), interpolation=cv2.INTER_AREA) for v in volume])


def prepare_volumes(series_root, study, slots, reader, planes, audit):
    import numpy as np
    require(len(slots) == 6, 'Six source slots required')
    volumes = []
    for slot, candidates in enumerate(slots):
        if not candidates:
            audit['missing_slots'] += 1
            volumes.append(np.empty((0, 256, 256), np.uint8)); continue
        audit['selected_series'] += 1
        try:
            volume, info = reader(str(Path(series_root) / study / candidates[0]), planes[slot % 3])
            require(type(info.get('n_bad')) is int and info['n_bad'] >= 0, 'Invalid decode counter')
            audit['unreadable_files'] += info['n_bad']
            if info['n_bad']: audit['partial_series'] += 1
            if len(volume) < 3:
                audit['short_series'] += 1
                volumes.append(np.empty((0, 256, 256), np.uint8)); continue
            volume = resize_volume(volume)
            nonzero = volume.any((1, 2))
            audit['black_frames'] += int((~nonzero).sum())
            if nonzero.sum() < 3:
                audit['empty_series'] += 1
                volumes.append(np.empty((0, 256, 256), np.uint8)); continue
            # Retain every frame, including black endpoints: no ordering/math change.
            volumes.append(volume)
        except Exception as exc:
            if fatal(exc):
                audit['resource_errors'] += 1
                raise
            audit['failed_series'] += 1
            volumes.append(np.empty((0, 256, 256), np.uint8))
    if not any(len(v) for v in volumes):
        audit['empty_studies'] += 1
        raise RuntimeError('Whole study empty after selected-slot masking; no prediction fallback')
    validate_volumes(volumes)
    return volumes


def eval_arrays(volumes):
    return sample_windows(volumes, 12, train=False, rng=None)


def series_table(root, split, ids, code, audit):
    import pandas as pd
    wanted = set(ids); rows = []; seen = set()
    with (Path(root) / (split + '_series.csv')).open(newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == SERIES_COLUMNS, 'Official series schema differs')
        for row in reader:
            if row['StudyInstanceUID'] not in wanted: continue
            uid, sid = row['StudyInstanceUID'], row['SeriesInstanceUID']
            require(all(v and v not in ('.', '..') and '/' not in v and '\\' not in v for v in (uid, sid)) and
                    sid not in seen and row['Fat_Suppression'] in ('0', '1') and
                    row['Anatomical_Plane'] in code.PLANES, 'Invalid/duplicate series metadata')
            seen.add(sid); rows.append(row)
    require({r['StudyInstanceUID'] for r in rows} == wanted, 'Image series membership differs')
    frame = pd.DataFrame(rows, columns=SERIES_COLUMNS).sort_values('SeriesInstanceUID')
    frame['Fat_Suppression'] = frame['Fat_Suppression'].astype(int)
    sizes = {}
    for row in frame.itertuples():
        folder = Path(root) / (split + '_series') / row.StudyInstanceUID / row.SeriesInstanceUID
        if folder.is_dir(): sizes[row.SeriesInstanceUID] = min(sum(p.is_file() for p in folder.iterdir()), 64)
        else:
            sizes[row.SeriesInstanceUID] = 0; audit['missing_series_directories'] += 1
    return code.build_study_table(frame, sizes)


def load_model(assets, sources, manifest):
    import torch
    validate_manifest(manifest); check_sources(sources)
    require(file_sha256(Path(assets) / 'weights.pt') == manifest['weights_sha256'], 'Own weights hash differs')
    checkpoint = torch.load(Path(assets) / 'weights.pt', map_location='cpu', weights_only=True)
    require(set(checkpoint) == {'model'} and checkpoint['model'], 'Weights-only asset schema differs')
    require(all(torch.is_tensor(v) and torch.isfinite(v).all() for v in checkpoint['model'].values()), 'Invalid EMA tensors')
    code = load_source(sources, 'knee')
    require(code.LABELS == manifest['targets'], 'Model/asset target order differs')
    model = code.KneeNet(MODEL_RECIPE['backbone'], pretrained=False, img_size=256)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model, code


def predict(model, arrays):
    import numpy as np
    import torch
    with torch.inference_mode():
        x, mask, pos = [torch.from_numpy(v).unsqueeze(0).cuda() for v in arrays]
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x, mask, pos, res=256, train=False)
        p = logits.float().sigmoid()[0].cpu().numpy()
    require(p.shape == (12,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid image-only probabilities')
    return p


def stress_inputs():
    import cv2
    import numpy as np
    checks = {}; slots = [['first', 'never-retry'], ['second'], [], [], [], []]
    volume = np.stack([np.full((384, 384), z + 1, np.uint8) for z in range(8)])
    def reader(path, plane):
        require(not path.endswith('never-retry'), 'Unexpected candidate retry')
        return volume, {'n_bad': 0}
    def prepared(callback, table=slots):
        return eval_arrays(prepare_volumes(Path('synthetic'), 'study', table, callback,
                                          ['Sagittal', 'Coronal', 'Axial'], Counter()))
    healthy = prepared(reader)
    expected = eval_arrays([resize_volume(volume), resize_volume(volume)] + [np.empty((0,256,256), np.uint8)] * 4)
    require(all(np.array_equal(a,b) for a,b in zip(healthy, expected)), 'Healthy cache parity failed')
    checks['healthy_cache_parity'] = 'PASS'
    partial = prepared(lambda path, plane: (volume, {'n_bad': 1}))
    require(all(np.array_equal(a,b) for a,b in zip(healthy, partial)), 'Partial volume changed')
    checks['partial_unchanged'] = 'PASS'
    missing = prepared(reader, [[], ['second'], [], [], [], []])
    for mode in ('exception', 'short', 'black'):
        def broken(path, plane):
            if path.endswith('first'):
                if mode == 'exception': raise ValueError('synthetic unreadable series')
                return (volume[:2] if mode == 'short' else np.zeros_like(volume)), {'n_bad': 0}
            return reader(path, plane)
        actual = prepared(broken)
        require(all(np.array_equal(a,b) for a,b in zip(missing, actual)), 'Unusable selected slot not masked')
        checks[mode + '_masked'] = 'PASS'
    try: prepared(reader, [[] for _ in range(6)])
    except RuntimeError: checks['all_empty_fails'] = 'PASS'
    no_memory = cv2.error('synthetic no memory'); no_memory.code = -4
    for error, name in ((MemoryError(), 'memory_fatal'), (TimeoutError(), 'timeout_fatal'),
                        (no_memory, 'opencv_memory_fatal')):
        def exhausted(*args): raise error
        try: prepared(exhausted)
        except type(error): checks[name] = 'PASS'
    require(set(checks) == STRESS_CASES, 'Incomplete production stress checks')
    return {'status': 'PASS', 'uses_images': False, 'uses_labels': False, 'cases': checks}


def require_rehearsal(report, qualification):
    require(report.get('status') == 'PASS' and report.get('rehearsal') is True and
            report.get('qualification_fingerprint') == qualification and report.get('policy') == POLICY and
            report.get('train_studies') == TRAIN_COUNT and report.get('test_studies') == 3 and
            report.get('strict_cache_comparisons') == TRAIN_COUNT and
            report.get('input_max_abs_error') == 0 and report.get('empty_studies') == 0 and
            report.get('resource_errors') == 0 and report.get('prediction_fallbacks') == 0 and digest(report.get('sample_fingerprint')),
            'Matching expanded own-model rehearsal required')
    require(all(type(report.get(k)) is int and report[k] >= 0 for k in ('selected_series', 'missing_slots',
            'unreadable_files', 'partial_series', 'short_series', 'black_frames', 'empty_series', 'failed_series',
            'missing_series_directories')), 'Aggregate input counters missing')
    require(report.get('stress_checks') == {'status': 'PASS', 'uses_images': False, 'uses_labels': False,
            'cases': dict.fromkeys(STRESS_CASES, 'PASS')}, 'Stress qualification missing')
    require(all(isinstance(report.get(k), (int, float)) and math.isfinite(report[k]) and 0 <= report[k] <= 1e-6
                for k in ('repeat_max_abs_error', 'cache_prediction_max_abs_error')) and
            0 < report.get('elapsed_seconds', math.inf) <= PLATFORM_SECONDS, 'Prediction/runtime qualification failed')


def run(root, assets, sources, contract, out, *, cache=None):
    import importlib.metadata
    import cv2
    import numpy as np
    import torch
    started = time.monotonic(); out = Path(out); out.mkdir(parents=True, exist_ok=False)
    audit = Counter({k: 0 for k in ('selected_series', 'missing_slots', 'unreadable_files', 'partial_series',
        'short_series', 'black_frames', 'empty_series', 'failed_series', 'resource_errors', 'empty_studies',
        'missing_series_directories')})
    report = {'status': 'FAIL', 'policy': POLICY, 'rehearsal': contract['rehearsal'],
              'qualification_fingerprint': contract['qualification_fingerprint'], 'train_studies': 0,
              'test_studies': 0, 'strict_cache_comparisons': 0, 'input_max_abs_error': 0,
              'repeat_max_abs_error': 0., 'cache_prediction_max_abs_error': 0.}
    try:
        require(torch.cuda.is_available(), 'CUDA required; no fallback')
        torch.manual_seed(42); torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False; torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
        manifest = json.loads((Path(assets) / 'manifest.json').read_text())
        require(manifest == contract['manifest'], 'Attached own manifest differs')
        model, code = load_model(assets, sources, manifest)
        model = model.cuda().to(memory_format=torch.channels_last).eval().requires_grad_(False)
        prep = load_source(sources, 'preprocess'); reader = protect_reader(prep, audit)
        columns, ids, _ = read_test_contract(root, rehearsal=contract['rehearsal'])
        require(columns[1:] == manifest['targets'], 'Official target order differs from own model')
        table = series_table(root, 'test', ids, code, audit)
        tasks = [('test', uid, table.get(uid, [[] for _ in range(6)])) for uid in ids]
        index = None
        if contract['rehearsal']:
            require(cache is not None, 'Expanded rehearsal requires exact training cache')
            cache = Path(cache)
            require(file_sha256(cache / 'index-private.json') == manifest['index_sha256'], 'Rehearsal cache index differs')
            index = json.loads((cache / 'index-private.json').read_text())
            selected = sorted(index, key=lambda uid: hashlib.sha256(('train04-inference-rehearsal-v1:' + uid).encode()).hexdigest())[:TRAIN_COUNT]
            require(len(selected) == TRAIN_COUNT and not set(selected) & set(ids), 'Rehearsal cohort coverage differs')
            table = series_table(root, 'train', selected, code, audit)
            tasks += [('train', uid, table.get(uid, [[] for _ in range(6)])) for uid in selected]
            report['sample_fingerprint'] = fingerprint(selected)
            report['stress_checks'] = stress_inputs()
        durations = []; predictions = []
        for split, uid, slots in tasks:
            tick = time.monotonic(); before = audit.copy()
            volumes = prepare_volumes(Path(root) / (split + '_series'), uid, slots, reader, code.PLANES, audit)
            arrays = eval_arrays(volumes); prediction = predict(model, arrays)
            if contract['rehearsal']:
                rerun = prepare_volumes(Path(root) / (split + '_series'), uid, slots, reader, code.PLANES, Counter())
                repeated = predict(model, eval_arrays(rerun))
                report['repeat_max_abs_error'] = max(report['repeat_max_abs_error'], float(np.max(abs(prediction-repeated))))
                if split == 'train':
                    item = index[uid]; filename = hashlib.sha256(uid.encode()).hexdigest() + '.npz'
                    require(item['file'] == filename and file_sha256(cache / 'arrays' / filename) == item['sha256'], 'Rehearsal cached pixels differ')
                    require(item['series'] == [s[0] if s else None for s in slots], 'Raw/cache selected series differs')
                    require(not any(audit[k] != before[k] for k in ('unreadable_files', 'failed_series', 'short_series', 'empty_series')),
                            'Strict healthy cache comparison ineligible')
                    cached = read_volumes(cache / 'arrays' / filename)
                    require(all(np.array_equal(a,b) for a,b in zip(volumes,cached)), 'Raw/cached variable volume pixels differ')
                    cached_arrays = eval_arrays(cached)
                    require(all(np.array_equal(a,b) for a,b in zip(arrays,cached_arrays)), 'Raw/cache triplets masks or positions differ')
                    cached_prediction = predict(model, cached_arrays)
                    report['cache_prediction_max_abs_error'] = max(report['cache_prediction_max_abs_error'], float(np.max(abs(prediction-cached_prediction))))
                    report['strict_cache_comparisons'] += 1
            durations.append(time.monotonic()-tick)
            report[split + '_studies'] += 1
            if split == 'test': predictions.append(prediction)
            if len(durations) % 8 == 0: print(json.dumps({'event': 'coverage_inference', 'studies': len(durations)}), flush=True)
        require(report['repeat_max_abs_error'] <= 1e-6 and report['cache_prediction_max_abs_error'] <= 1e-6, 'Own model repeat/cache prediction mismatch')
        report['submission_sha256'] = write_probabilities(out / 'submission.csv', columns, ids, predictions)
        report.update(status='PASS', prediction_fallbacks=0, study_seconds_mean=float(np.mean(durations)),
            estimated_1300_seconds_p95=float(np.quantile(durations, .95)*1300),
            estimate_note='Rehearsal includes independent rereads, repeat predictions and healthy cache checks; not hidden-test time.',
            versions={k: importlib.metadata.version(k) for k in ('torch','timm','numpy','pydicom')} | {'opencv': cv2.__version__},
            cuda_devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except BaseException as exc:
        report['failure_type'] = type(exc).__name__
        raise
    finally:
        report.update(dict(audit)); report['elapsed_seconds'] = time.monotonic()-started
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1024 if sys.platform.startswith('linux') else 1)
        report['cuda_peak_bytes'] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        report['disk_free_bytes'] = shutil.disk_usage(out).free
        report['output_bytes'] = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        write_json(out / 'rehearsal.json', report)
    return report
