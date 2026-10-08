"""Bundled entry; public source is pinned and supplied by the builder."""
import gc
import hashlib
import importlib.util
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
t0 = time.monotonic()
roots = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
         Path('/kaggle/input/rsna-knee-abnormality-detection')]
root = next((p for p in roots if (p/'test.csv').is_file()), None)
assets = next((p for p in [Path('/kaggle/input/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader'),
                          Path('/kaggle/input/rsna-knee-2-5d-convnext-reader')]
               if (p/'cnxt_v0_fold0.pt').is_file()), None)
if root is None or assets is None:
    raise RuntimeError('Expected competition/public reader mounts missing')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', '--find-links', str(assets),
                'pylibjpeg', 'pylibjpeg-libjpeg', 'pylibjpeg-openjpeg'], check=True)
import numpy as np
import pandas as pd
import torch
import cv2
if not torch.cuda.is_available():
    raise RuntimeError('This rehearsal requires CUDA; no CPU fallback')
torch.manual_seed(20261008)
torch.backends.cudnn.benchmark = False

def load_source(name, filename):
    spec = importlib.util.spec_from_file_location(name, source_dir / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

prep = load_source('audited_preprocess', 'preprocess.py')
model_code = load_source('audited_model', 'knee.py')
labels = list(pd.read_csv(root/'sample_submission.csv', nrows=0).columns)[1:]
assert labels == model_code.LABELS
checkpoints = [assets/f'cnxt_v0_fold{i}.pt' for i in range(3)]
hashes = {p.name: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for p in checkpoints}
if EXPECTED_WEIGHTS and hashes != EXPECTED_WEIGHTS:
    raise RuntimeError('Checkpoint identity changed after rehearsal')
Path('/kaggle/working/reader-weight-hashes.json').write_text(json.dumps(hashes, indent=2))
models, model_args = [], []
for path in checkpoints:
    ck = torch.load(path, map_location='cpu', weights_only=True)
    args = ck['args']; model_args.append(args)
    m = model_code.KneeNet(args['backbone'].split('.')[0], pretrained=False, img_size=args['res'])
    m.load_state_dict(ck['model'], strict=True)
    models.append((m.cuda().to(memory_format=torch.channels_last).eval(), args['res']))
    del ck
gc.collect()

audit = {'code_revision': CODE_REVISION, 'source_hashes': SOURCE_HASHES, 'weight_hashes': hashes,
         'entry_sha256': ENTRY_SHA256, 'input_policy_version': 'convnext-partial-series-v2',
         'training_args': model_args, 'rehearsal': REHEARSAL, 'studies': 0, 'selected_series': 0,
         'missing_slots': 0, 'decode_failures': 0, 'zero_studies': 0, 'repeat_max_abs_error': None,
         'partial_series': 0, 'failed_series': 0, 'short_series': 0, 'zero_decoded_series': 0,
         'resource_errors': 0, 'series_error_types': {},
         'cache_parity_max_abs_error': None, 'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
         'versions': {p: importlib.metadata.version(p) for p in ['torch','timm','numpy','pandas','pydicom','pylibjpeg']}}
audit['versions']['opencv'] = cv2.__version__

def count_event(name, amount=1):
    audit[name] = audit.get(name, 0) + amount

def fatal_series_error(exc):
    return isinstance(exc, (MemoryError, TimeoutError)) or (isinstance(exc, cv2.error) and exc.code == -4)

def load_selected_series(path):
    # The public per-file reader catches Exception. Preserve resource/deadline
    # failures even when swallowed, without changing the preprocessing source.
    original_read = prep._read
    fatal_failure = None
    class FatalRead(BaseException):
        pass
    def guarded_read(*args, **kwargs):
        nonlocal fatal_failure
        try:
            return original_read(*args, **kwargs)
        except Exception as exc:
            if fatal_series_error(exc):
                fatal_failure = exc
                # Bypass the public per-file except Exception immediately.
                raise FatalRead from None
            raise
    prep._read = guarded_read
    try:
        result = prep.load_series(path)
    except FatalRead:
        count_event('resource_errors')
        raise fatal_failure from None
    except Exception as exc:
        if fatal_failure is not None or fatal_series_error(exc):
            count_event('resource_errors')
            raise fatal_failure if fatal_failure is not None else exc
        raise
    finally:
        prep._read = original_read
    if fatal_failure is not None:
        count_event('resource_errors')
        raise fatal_failure
    return result

def prepare_study(study, split, table):
    slots = table.get(study, [[] for _ in range(model_code.N_SLOTS)])
    x = np.zeros((6, 12, 3, prep.SIZE, prep.SIZE), np.uint8)
    mask, pos = np.zeros(6, bool), np.zeros((6,12), np.float32)
    missing, failed, short = 0, 0, 0
    for slot, candidates in enumerate(slots):
        if not candidates:
            count_event('missing_slots'); missing += 1
            continue
        count_event('selected_series')
        try:
            volume, info = load_selected_series(str(root/f'{split}_series'/study/candidates[0]))
        except Exception as exc:
            if fatal_series_error(exc):
                raise
            # Match public DicomStudyDataset: mask an unusable selected series.
            # Do not switch to another candidate or replace it with predictions.
            count_event('failed_series'); failed += 1
            errors = audit.setdefault('series_error_types', {})
            errors[type(exc).__name__] = errors.get(type(exc).__name__, 0) + 1
            continue
        count_event('decode_failures', int(info['n_bad']))
        if len(volume) < 3:
            count_event('short_series'); short += 1
            continue
        if volume.ndim != 3 or volume.shape[1:] != (prep.SIZE, prep.SIZE) or volume.dtype != np.uint8:
            raise ValueError('Public volume shape/type changed')
        if info['n_bad']:
            count_event('partial_series')
        if not np.any(volume):
            count_event('zero_decoded_series')
        centers = model_code.window_centres(len(volume), 12, False)
        x[slot] = np.stack([volume[np.clip([c-1,c,c+1],0,len(volume)-1)] for c in centers])
        mask[slot] = True
        pos[slot] = centers / max(len(volume)-1,1)
    if not mask.any() or not np.any(x):
        count_event('zero_studies')
        raise RuntimeError(f'Study has no usable image content: missing_slots={missing}, failed_series={failed}, short_series={short}; no fallback')
    return x, mask, pos

def stress_inputs():
    """Exercise the production loader with synthetic volumes; restore all state."""
    original_loader, original_read = prep.load_series, prep._read
    original_audit = json.loads(json.dumps(audit))
    table = {'synthetic': [['first', 'never-retry'], ['second'], [], [], [], []]}
    absent = {'synthetic': [[], ['second'], [], [], [], []]}
    volume = np.arange(8 * prep.SIZE * prep.SIZE, dtype=np.uint32).reshape(8, prep.SIZE, prep.SIZE).astype(np.uint8)
    checks = {}
    def healthy(path):
        if str(path).endswith('never-retry'):
            raise AssertionError('Unexpected candidate retry')
        return volume, {'n_bad': 0}
    def equal(first, second):
        return all(np.array_equal(a, b) for a, b in zip(first, second))
    try:
        prep.load_series = healthy
        full = prepare_study('synthetic', 'test', table)
        missing = prepare_study('synthetic', 'test', absent)
        assert full[1].tolist() == [True, True, False, False, False, False]
        assert missing[1].tolist() == [False, True, False, False, False, False]
        checks['healthy_missing_planes'] = 'PASS'
        prep.load_series = lambda path: (volume, {'n_bad': 1 if str(path).endswith('first') else 0})
        assert equal(full, prepare_study('synthetic', 'test', table))
        checks['usable_partial_volume'] = 'PASS'
        for mode in ('exception', 'short'):
            def injected(path):
                if str(path).endswith('first'):
                    if mode == 'exception':
                        raise ValueError('synthetic unreadable selected series')
                    return volume[:2], {'n_bad': 0}
                return healthy(path)
            prep.load_series = injected
            assert equal(missing, prepare_study('synthetic', 'test', table))
            checks[mode + '_masked_without_retry'] = 'PASS'
        prep.load_series = healthy
        try:
            prepare_study('synthetic', 'test', {'synthetic': [[] for _ in range(6)]})
        except RuntimeError:
            checks['all_empty_fails'] = 'PASS'
        else:
            raise AssertionError('Empty study was accepted')
        def public_catches_error(path):
            try:
                prep._read(path)
            except Exception:
                pass
            return volume, {'n_bad': 1}
        for error_type, case in ((MemoryError, 'swallowed_memory_still_fatal'),
                                 (TimeoutError, 'swallowed_timeout_still_fatal')):
            def exhausted(*args, **kwargs):
                raise error_type('synthetic fatal failure')
            prep._read, prep.load_series = exhausted, public_catches_error
            try:
                prepare_study('synthetic', 'test', table)
            except error_type:
                checks[case] = 'PASS'
            else:
                raise AssertionError('Fatal failure was masked')
        return {'status': 'PASS', 'cases': checks, 'uses_raw_data': False, 'uses_labels': False}
    finally:
        prep.load_series, prep._read = original_loader, original_read
        audit.clear(); audit.update(original_audit)

@torch.inference_mode()
def predict(arrays):
    x, mask, pos = [torch.from_numpy(a).unsqueeze(0).cuda() for a in arrays]
    with torch.autocast('cuda', dtype=torch.float16):
        p = torch.stack([m(x, mask, pos, res=res).float().sigmoid() for m,res in models]).mean(0)
    p = p[0].cpu().numpy()
    if p.shape != (12,) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise RuntimeError('Invalid predictions')
    return p

def run(split, ids):
    meta = pd.read_csv(root/f'{split}_series.csv', dtype={'StudyInstanceUID': str,'SeriesInstanceUID':str})
    meta = meta[meta.StudyInstanceUID.isin(set(ids))]
    sizes = {}
    for row in meta.itertuples():
        d = root/f'{split}_series'/row.StudyInstanceUID/row.SeriesInstanceUID
        sizes[row.SeriesInstanceUID] = min(len(list(d.iterdir())), prep.MAX_SLICES) if d.is_dir() else 0
    table = model_code.build_study_table(meta, sizes)
    predictions=[]; durations=[]
    for index, study in enumerate(ids):
        start=time.monotonic(); arrays=prepare_study(study,split,table); p=predict(arrays)
        if REHEARSAL and index == 0:
            q=predict(arrays)
            repeat=float(np.max(np.abs(p-q)))
            if repeat > 1e-6: raise RuntimeError('Repeated inference differs')
            audit['repeat_max_abs_error']=max(audit['repeat_max_abs_error'] or 0,repeat)
            cache=Path('/kaggle/working/reader-rehearsal-arrays.npz')
            np.savez(cache, x=arrays[0], mask=arrays[1], pos=arrays[2])
            with np.load(cache) as z: q=predict((z['x'],z['mask'],z['pos']))
            parity=float(np.max(np.abs(p-q))); cache.unlink()
            if parity > 1e-6: raise RuntimeError('Persisted input parity failed')
            audit['cache_parity_max_abs_error']=max(audit['cache_parity_max_abs_error'] or 0,parity)
        predictions.append(p); durations.append(time.monotonic()-start); audit['studies']+=1
        if (index+1)%8==0: print(f'{split}: {index+1}/{len(ids)} studies, {time.monotonic()-t0:.1f}s',flush=True)
        if time.monotonic()-t0 > (3500 if REHEARSAL else 28000): raise TimeoutError('Bounded runtime exhausted')
    return pd.DataFrame(predictions,columns=labels,index=pd.Index(ids,name='StudyInstanceUID')),durations

try:
    if REHEARSAL:
        audit['stress_checks'] = stress_inputs()
        train_ids=pd.read_csv(root/'train_series.csv',usecols=['StudyInstanceUID'],dtype=str).StudyInstanceUID.unique().tolist()
        chosen=sorted(train_ids,key=lambda x:hashlib.sha256(('reader-audit20261008'+x).encode()).hexdigest())[:64]
        if len(chosen) != 64: raise ValueError('Expanded rehearsal requires 64 label-blind training studies')
        audit['train_studies_requested'] = len(chosen)
        _, train_times=run('train',chosen)
        audit['sample_fingerprint']=hashlib.sha256('\n'.join(chosen).encode()).hexdigest()
        audit['train_study_seconds_mean']=float(np.mean(train_times))
        audit['estimated_1300_studies_seconds']=float(np.mean(train_times)*1300)
    ids=pd.read_csv(root/'test.csv',usecols=['StudyInstanceUID'],dtype=str).StudyInstanceUID.tolist()
    if not ids or len(ids)!=len(set(ids)): raise RuntimeError('Missing/duplicate test IDs')
    pred,test_times=run('test',ids)
    if pred.index.tolist()!=ids or pred.columns.tolist()!=labels: raise RuntimeError('Schema or ID coverage differs')
    pred.to_csv('/kaggle/working/submission.csv')
    audit.update(status='PASS',test_studies=len(ids),test_seconds=sum(test_times),elapsed_seconds=time.monotonic()-t0,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
except Exception as exc:
    audit.update(status='FAIL', failure_type=type(exc).__name__, elapsed_seconds=time.monotonic()-t0)
    raise
finally:
    Path('/kaggle/working/reader-audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({k:v for k,v in audit.items() if k not in ['training_args','weight_hashes','source_hashes']},indent=2))
