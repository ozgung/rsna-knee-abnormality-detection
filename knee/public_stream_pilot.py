"""CPU-only exact-byte stream qualification. No labels, models or CUDA use."""
from collections import Counter
import csv
import json
from pathlib import Path
import random
import resource
import sys
import time

from knee.reference import canonical_bytes, sha256
from knee.public_adaptation_pilot import cache_contract, file_sha, INDEX_SHA, IDS_SHA
from knee.public_study_stream import StudyLRU, StudyStream

STAGES = frozenset({'binding', 'metadata', 'definitions', 'stream', 'restart', 'cache', 'failure', 'rng'})


def safe_failure(stage, error):
    # Never export exception messages, paths, study IDs, or traceback.
    return {'failure_stage': stage if stage in STAGES else 'unknown',
            'failure_kind': 'INPUT_OR_PARITY_CHECK_FAILED'}


def sample_indices(mask, study, epoch=0):
    """Exact adaptation center policy; ID/epoch-local RNG, no global RNG access."""
    import numpy as np
    valid = np.flatnonzero(mask > 0)
    if not len(valid): raise ValueError('Empty mask')
    lo, hi = int(valid.min()), int(valid.max())
    centers = list(range(lo + 1, hi))
    if not centers: centers = [max(1, min((lo + hi)//2, len(mask)-2))]
    seed = int(sha256(f'42:{epoch}:{study}'.encode())[:16], 16)
    selected = np.random.default_rng(seed).choice(centers, 12, replace=len(centers)<12)
    return np.stack([selected-1, selected, selected+1], 1)


def window_pair(public, volume, mask, study):
    """Crop only after native quantization; compare both indexing arithmetic paths."""
    import numpy as np
    import torch
    indices = sample_indices(mask, study)
    cropped = volume[:, 32:352, 32:352]
    native = torch.from_numpy(np.ascontiguousarray(volume[indices][:,:,32:352,32:352])).float()/255.0
    crop = torch.from_numpy(np.ascontiguousarray(cropped[indices])).float()/255.0
    return ((native-public._MEAN)/public._STD)[None], ((crop-public._MEAN)/public._STD)[None]


def require_byte_parity(actual, expected):
    if len(actual)!=2 or len(expected)!=2 or any(
            a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes()
            for a,b in zip(actual,expected)):
        raise ValueError('Raw array schema or bytes differ')


def rng_snapshot():
    import numpy as np
    import torch
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def require_rng(before):
    import numpy as np
    import torch
    after = rng_snapshot()
    if (before[0] != after[0] or before[1][0] != after[1][0] or
            not np.array_equal(before[1][1], after[1][1]) or before[1][2:] != after[1][2:] or
            not torch.equal(before[2], after[2])):
        raise ValueError('Global RNG changed')


def run(public_source, decoder_source, manifest, cache, root, out, contract):
    import numpy as np
    import torch
    from knee.reference import load_definitions, bounded_ids
    from knee.mirror_input import prepare_study, policy_contract
    cache, root, out = Path(cache), Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'status':'RUNNING', 'purpose':'CPU input mechanics only', 'models_loaded':0,
              'completed_studies':0, 'contract':contract, 'prefetch_limit':4, 'lru_capacity':2,
              'source_revision':__import__('os').environ.get('KNEE_CODE_REVISION'),
              'source_bundle_sha256':__import__('os').environ.get('KNEE_SOURCE_SHA256'),
              'decode_calls':0, 'decode_seconds':0.0, 'window_seconds':0.0,
              'consumer_wait_seconds':0.0, 'observed_queue_peak':0,
              'environment':{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
                             'pydicom':__import__('pydicom').__version__,'cv2':__import__('cv2').__version__}}
    counts = Counter()
    stage = 'binding'
    def save():
        report.update(elapsed_seconds=time.monotonic()-started, counters=dict(counts),
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        (out/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        report['environment'].update(pydicom=__import__('pydicom').__version__,cv2=__import__('cv2').__version__)
        receipt = json.loads((cache/'receipt.json').read_text())
        if cache_contract(receipt)!=contract or file_sha(cache/'private-index.json')!=INDEX_SHA:
            raise ValueError('Cache contract changed')
        if (manifest['source_sha256']!=contract['source_sha256'] or
                sha256(canonical_bytes(manifest))!=contract['manifest_sha256'] or
                policy_contract()!=receipt['input_policy']): raise ValueError('Reader contract changed')
        index = json.loads((cache/'private-index.json').read_text())
        ids = [r['StudyInstanceUID'] for r in index]
        bounded_ids(ids,count=32)
        if len(ids)!=32 or sha256(canonical_bytes(ids))!=IDS_SHA or [r['ordinal'] for r in index]!=list(range(32)):
            raise ValueError('Membership changed')
        stage = 'metadata'
        if file_sha(root/'train_series.csv')!=contract['series_metadata_sha256']:
            raise ValueError('Series metadata changed')
        records = {}
        with (root/'train_series.csv').open(newline='') as handle:
            for row in csv.DictReader(handle): records.setdefault(row['StudyInstanceUID'],[]).append(row)
        if not (root/'train_series').is_dir() or any(not records.get(s) or not (root/'train_series'/s).is_dir() for s in ids):
            raise ValueError('Raw study layout missing')
        stage = 'definitions'
        public, decoder = load_definitions(public_source,decoder_source,manifest)
        torch.set_num_threads(4)
        before = rng_snapshot()
        key = sha256(canonical_bytes({'cache':contract,'schema':'uint8:96x384x384/mask96','reader_policy':policy_contract()}))
        def cached(i):
            volume = np.load(cache/f'{i:03d}.volume.npy',allow_pickle=False)
            mask = np.load(cache/f'{i:03d}.mask.npy',allow_pickle=False)
            row = index[i]
            if (volume.shape!=(96,384,384) or volume.dtype!=np.uint8 or mask.shape!=(96,) or
                    str(mask.dtype)!=row['mask_dtype'] or sha256(volume.tobytes())!=row['volume_sha256'] or
                    sha256(mask.tobytes())!=row['mask_sha256'] or not np.isin(mask,[0,1]).all() or not mask.any() or
                    not np.array_equal(mask.astype(bool),volume.reshape(96,-1).any(axis=1))):
                raise ValueError('Cache integrity failed')
            return volume,mask
        def read(i):
            t = time.monotonic()
            volume,mask,c = prepare_study(public,decoder,ids[i],records,root/'train_series')
            if c.get('decoded_unique_slices',0)<=0 or c.get('resource_errors',0): raise ValueError('No eligible decode')
            counts.update(c)
            report['decode_calls']+=1; report['decode_seconds']+=time.monotonic()-t
            return volume,mask
        def compare(i,value):
            volume,mask = value
            cv,cm = cached(i)
            require_byte_parity((volume,mask),(cv,cm))
            t=time.monotonic()
            a,b=window_pair(public,volume,mask,ids[i]); c,d=window_pair(public,cv,cm,ids[i])
            if a.shape!=(1,12,3,320,320) or not torch.isfinite(a).all() or not all(torch.equal(a,x) for x in (b,c,d)):
                raise ValueError('Sampled windows differ')
            report['window_seconds']+=time.monotonic()-t
        stage = 'stream'
        with StudyStream(range(32),read,contract=key,close_timeout=30.0) as stream:
            for i in range(32):
                t=time.monotonic(); value=next(stream)
                report['consumer_wait_seconds']+=time.monotonic()-t
                with stream._condition: report['observed_queue_peak']=max(report['observed_queue_peak'],len(stream._buffer))
                compare(i,value); stream.acknowledge(); report['completed_studies']+=1
            if stream.consumed_cursor!=32: raise ValueError('Consumption coverage failed')
        stage='restart'
        with StudyStream(range(32),read,contract=key,close_timeout=30.0) as stream:
            compare(0,next(stream)); stream.acknowledge()
            compare(1,next(stream))  # Delivered but intentionally not acknowledged.
            with stream._condition:
                stream._condition.wait_for(lambda: len(stream._buffer)==4 or stream._done)
                if stream._error is not None: raise stream._error
                report['observed_queue_peak']=max(report['observed_queue_peak'],len(stream._buffer))
                cursor=stream.consumed_cursor
                fetched=stream.fetched_cursor
        with StudyStream(range(32),read,contract=key,consumed_cursor=cursor,close_timeout=30.0) as stream:
            compare(1,next(stream)); stream.acknowledge()
        if cursor!=1 or fetched!=6: raise ValueError('Cursor replay failed')
        report.update(restart_consumed_cursor=cursor,restart_fetched_cursor=fetched,restart_discarded_deliveries=fetched-cursor,restart_exact=True,
                      backpressure_full_queue_observed=True)
        stage='cache'
        lru=StudyLRU(2); start_calls=report['decode_calls']
        a=lru.read(key,0,read); compare(0,a)
        if lru.read(key,0,read) is not a: raise ValueError('Cache hit identity failed')
        lru.read(key,1,read); lru.read(key,2,read)
        b=lru.read(key,0,read); compare(0,b)
        if b is a or report['decode_calls']-start_calls!=4: raise ValueError('Eviction failed')
        report.update(forced_cache_hits=1,forced_cache_misses=4,forced_evictions=2)
        stage='failure'
        def fail(i): raise RuntimeError('controlled synthetic producer failure')
        caught=False
        with StudyStream([0],fail,contract=key,close_timeout=30.0) as stream:
            try: next(stream)
            except RuntimeError: caught=True
            if stream.consumed_cursor!=0: raise ValueError('Failed read consumed')
        if not caught: raise ValueError('Failure not propagated')
        report['producer_failure_propagated']=True
        stage='rng'; require_rng(before)
        report.update(status='PASS',byte_parity=True,sampled_window_parity=True,central_crop_after_quantization=True,rng_unchanged=True)
    except Exception as error:
        report.update(status='FAIL',**safe_failure(stage,error)); save(); raise RuntimeError('CPU stream qualification failed') from None
    save(); return report
