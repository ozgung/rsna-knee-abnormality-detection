"""Disposable CUDA mechanics qualification of the prospective public adaptation."""
import hashlib
import json
from pathlib import Path
import random
import time

from knee.reference import canonical_bytes, sha256

CACHE_SOURCE = 'ozgung/knee-public-rich-cache-pilot-20261010'
INDEX_SHA = 'a2e6c241d30f0992e7ddfdb44b1db2881191fb72cbf885a6ee62e8d4d31e94a0'
IDS_SHA = '77ceb3b6fd1b16f1ce2694f6af3170297df56429cd58c5ac1cfae55c8dbb4359'
CACHE_REVISION = '227c0010d72547deb57919cc3756711e44671e14'
EXCLUSION_SHA = '44834d7638eb60142a668a7597bd3093fe569e13b6f77e60a9c185a4ac3206f1'
COUNTERS_SHA = 'b2446ea4d06a81219df72b3342530f25cd77b45b8d8d95794175ebc729c92507'
CACHE_BUNDLE_SHA = '3b8d91eb4aa258de2898dc304ba727dfe14ca3a80bfb3a7aacff39767f0e12e5'
LABEL_SHA = 'd2cdb8a37e0c4c491758d17c96e3bd4081f06b257c7849608df88590368e94e2'
MASK_SHA = 'a134cfce2f751f90f4219d791a563ee677b6223ac28f32e456bb3759c1d3fba9'
STATE_ATOL = 1e-6
STATE_RTOL = 1e-5


SAFE_CHECKS = frozenset({'LABEL_BYTES', 'LABEL_TARGET_SCHEMA', 'MASK_TARGET_SCHEMA',
    'LABEL_DUPLICATE_IDS', 'LABEL_SELECTED_COVERAGE', 'MASK_INVALID_VALUES',
    'LABEL_OBSERVED_TARGET_INVALID', 'LABEL_NONCONTRIBUTING_STUDIES', 'LABEL_CONTRIBUTOR_COUNT'})
SAFE_STAGES = frozenset({'storage', 'cache_contract', 'cache_binding', 'source_lineage',
    'reader_policy', 'cache_index', 'public_definitions', 'label_preflight',
    'official_metadata', 'model_load', 'frozen_parity', 'optimizer_setup',
    'full_updates', 'tail_reference', 'resume_continuation', 'reload_predictions'})


class PilotCheckError(ValueError):
    """Only reviewed enum checks and aggregate numeric counts can leave runtime."""
    def __init__(self, check, counts=None):
        if check not in SAFE_CHECKS: raise ValueError('Unreviewed diagnostic enum')
        self.check = check
        self.counts = counts or {}
        if (not set(self.counts) <= {'selected_studies', 'noncontributing_studies', 'observed_cells', 'unknown_cells'} or
                any(type(v) is not int or v < 0 for v in self.counts.values())):
            raise ValueError('Unsafe diagnostic aggregate')
        super().__init__(check)


def failure_diagnostic(stage, error):
    result = {'failure_stage': stage if stage in SAFE_STAGES else 'unclassified',
              'failure_check': error.check if isinstance(error, PilotCheckError) else 'UNCLASSIFIED_EXCEPTION'}
    if isinstance(error, PilotCheckError): result['failure_counts'] = error.counts
    return result


def file_sha(path):
    with Path(path).open('rb') as handle: return hashlib.file_digest(handle, 'sha256').hexdigest()


def cache_contract(receipt):
    if (receipt.get('status') != 'PASS' or receipt.get('completed_studies') != 32 or
            receipt.get('requested_studies') != 32 or
            receipt.get('source_revision') != CACHE_REVISION or
            receipt.get('exclusion_sha256') != EXCLUSION_SHA or
            receipt.get('source_bundle_sha256') != CACHE_BUNDLE_SHA or
            sha256(canonical_bytes(receipt.get('counters'))) != COUNTERS_SHA or receipt.get('byte_parity') is not True or
            receipt.get('window_parity') is not True or receipt.get('repeat_decodes') != 2 or
            receipt.get('private_index_sha256') != INDEX_SHA or receipt.get('ordered_id_set_sha256') != IDS_SHA or
            receipt.get('counters', {}).get('zero_studies') != 0 or
            receipt.get('counters', {}).get('resource_errors') != 0):
        raise ValueError('Qualified exact cache required')
    return {'receipt_sha256': sha256(canonical_bytes(receipt)), 'index_sha256': INDEX_SHA,
            'ordered_ids_sha256': IDS_SHA, 'source_sha256': receipt['source_sha256'],
            'manifest_sha256': receipt['manifest_sha256'], 'series_metadata_sha256': receipt['series_metadata_sha256'],
            'cache_source_revision': CACHE_REVISION, 'exclusion_sha256': EXCLUSION_SHA,
            'cache_bundle_sha256': CACHE_BUNDLE_SHA, 'cache_counters': receipt['counters']}


def optimizer_groups(model):
    """Include every trainable parameter once, excluding norm/bias from decay."""
    import torch
    norm_ids = {id(p) for module in model.modules() if isinstance(module,
        (torch.nn.modules.batchnorm._BatchNorm, torch.nn.LayerNorm, torch.nn.GroupNorm))
        for p in module.parameters(recurse=False)}
    groups = {(family, decay): [] for family in ('encoder', 'head') for decay in (True, False)}
    expected = []
    for name, param in model.named_parameters():
        if not param.requires_grad: raise ValueError('Unexpected frozen parameter')
        expected.append(id(param))
        family = 'encoder' if name.startswith('backbone.') else 'head'
        decay = param.ndim > 1 and id(param) not in norm_ids and not name.endswith('.bias')
        groups[family, decay].append(param)
    if any(not params for params in groups.values()): raise ValueError('Incomplete encoder/head groups')
    actual = [id(p) for params in groups.values() for p in params]
    if sorted(actual) != sorted(expected) or len(set(actual)) != len(actual):
        raise ValueError('Optimizer parameter coverage differs')
    return [{'params': params, 'name': family + ('_decay' if decay else '_no_decay'),
             'base_lr': 1e-6 if family == 'encoder' else 1e-5,
             'lr': 1e-6 if family == 'encoder' else 1e-5,
             'weight_decay': .01 if decay else 0.0} for (family, decay), params in groups.items()]


def training_mode(model):
    import torch
    model.train()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.eval()
        if name.startswith('att.') and isinstance(module, torch.nn.Dropout) and module.p != .2:
            raise ValueError('Protocol head dropout differs')


def bn_state(model):
    import torch
    return {name: value.detach().cpu().clone() for name, value in model.named_buffers()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}


def require_bn(model, expected):
    import torch
    actual = bn_state(model)
    if actual.keys() != expected.keys() or any(not torch.equal(actual[k], v) for k,v in expected.items()):
        raise ValueError('BN buffers drifted')


def masked_loss(logits, targets, mask):
    import torch
    if logits.shape != targets.shape or mask.shape != targets.shape or not torch.isfinite(logits).all():
        raise ValueError('Loss shape/finiteness failed')
    if not torch.isfinite(targets).all() or not ((targets>=0)&(targets<=1)).all() or not ((mask==0)|(mask==1)).all():
        raise ValueError('Invalid soft targets/masks')
    denominator = mask.sum()
    if denominator <= 0: raise ValueError('Noncontributing study')
    return (torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets, reduction='none') * mask).sum()/denominator


def sampled_windows(public, volume, mask, sid, epoch):
    import numpy as np
    import torch
    valid = np.flatnonzero(mask > 0)
    if not len(valid): raise ValueError('Empty training image')
    lo,hi=int(valid.min()),int(valid.max())
    centers=list(range(lo+1,hi))
    if not centers: centers=[max(1,min((lo+hi)//2,len(mask)-2))]
    seed=int(sha256(f'42:{epoch}:{sid}'.encode())[:16],16)
    rng=np.random.default_rng(seed)
    selected=rng.choice(centers,12,replace=len(centers)<12)
    indices=np.stack([selected-1,selected,selected+1],1)
    tri=torch.from_numpy(np.ascontiguousarray(volume[indices,:, :][:,:,32:352,32:352])).float()/255.0
    return ((tri-public._MEAN)/public._STD)[None].to('cuda')


def checkpoint_forward(model, windows):
    """Inherited backbone/head, chunk4; nonreentrant checkpoint preserves RNG."""
    import torch
    from torch.utils.checkpoint import checkpoint
    flat=windows.flatten(0,1)
    features=torch.cat([checkpoint(model.backbone, chunk, use_reentrant=False, preserve_rng_state=True)
                        for chunk in flat.split(4)],dim=0)
    return model.head(features.view(windows.shape[0],windows.shape[1],-1))


def rng_state():
    import numpy as np
    import torch
    state=np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], torch.tensor(state[1].astype('int64')), *state[2:]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    npstate=state['numpy']; np.random.set_state((npstate[0],npstate[1].numpy().astype('uint32'),*npstate[2:]))
    torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def save_state(path, model, optimizer, sampler, updates, contract):
    import torch
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':rng_state(),
                'sampler':sampler,'updates':updates,'contract':contract},path)


def restore_state(path, model, optimizer, contract):
    import torch
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['contract'] != contract: raise ValueError('Resume contract differs')
    model.load_state_dict(state['model'],strict=True); optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'])
    return state['sampler'],state['updates']


def assert_state_close(a,b, *, atol=STATE_ATOL, rtol=STATE_RTOL):
    import torch
    if torch.is_tensor(a):
        if a.shape != b.shape or a.dtype != b.dtype or not torch.allclose(a.cpu(),b.cpu(),atol=atol,rtol=rtol):
            raise ValueError('Saved-state continuation differs')
        return float((a.cpu()-b.cpu()).abs().max()) if a.is_floating_point() and a.numel() else 0.0
    if isinstance(a,dict):
        if a.keys()!=b.keys(): raise ValueError('State keys differ')
        return max([assert_state_close(a[k],b[k],atol=atol,rtol=rtol) for k in a]+[0.0])
    if isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b): raise ValueError('State sequence differs')
        return max([assert_state_close(x,y,atol=atol,rtol=rtol) for x,y in zip(a,b)]+[0.0])
    if a!=b: raise ValueError('State scalar differs')
    return 0.0


def tensor_max_error(a, b):
    """Measured same-shape tensor difference, recorded before strict comparison."""
    import torch
    if torch.is_tensor(a):
        if not a.is_floating_point() or not a.numel(): return 0.0
        return float((a.cpu()-b.cpu()).abs().max())
    if isinstance(a,dict): return max([tensor_max_error(a[k],b[k]) for k in a]+[0.0])
    if isinstance(a,(list,tuple)): return max([tensor_max_error(x,y) for x,y in zip(a,b)]+[0.0])
    return 0.0


def apply_update(model, optimizer, updates):
    import torch
    for group in optimizer.param_groups: group['lr']=group['base_lr']*min((updates+1)/20,1.0)
    counts={family:0 for family in ('encoder','head')}
    for name,param in model.named_parameters():
        if param.grad is None or not torch.isfinite(param.grad).all(): raise ValueError('Missing/nonfinite gradient')
        counts['encoder' if name.startswith('backbone.') else 'head']+=int(torch.count_nonzero(param.grad))
    if any(v==0 for v in counts.values()): raise ValueError('Zero group gradient')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()): raise ValueError('Nonfinite updated parameters')
    for group in optimizer.param_groups:
        if any(int(optimizer.state[p]['step']) != updates+1 for p in group['params']):
            raise ValueError('AdamW update counter differs')
    return counts,float(norm)


def load_labels(label_path, mask_path, ids, targets):
    import numpy as np
    import pyarrow.parquet as pq
    import re
    if file_sha(label_path)!=LABEL_SHA or file_sha(mask_path)!=MASK_SHA:
        raise PilotCheckError('LABEL_BYTES')
    normalize=lambda s: re.sub('[^a-z0-9]','',s.lower())
    label_names=pq.read_schema(label_path).names
    if not set(['StudyInstanceUID',*targets])<=set(label_names): raise PilotCheckError('LABEL_TARGET_SCHEMA')
    names=pq.read_schema(mask_path).names
    mapping={normalize(c[2:]):c for c in names if c.startswith('m_')}
    mask_columns=[mapping.get(normalize(t)) for t in targets]
    if None in mask_columns or len(set(mask_columns))!=12: raise PilotCheckError('MASK_TARGET_SCHEMA')
    # Only ID/target/mask columns, never reports or raw y_ outcomes.
    labels=pq.read_table(label_path,columns=['StudyInstanceUID',*targets]).to_pydict()
    masks=pq.read_table(mask_path,columns=['StudyInstanceUID',*mask_columns]).to_pydict()
    def rows(table,columns):
        keys=table['StudyInstanceUID']
        if len(set(keys))!=len(keys): raise PilotCheckError('LABEL_DUPLICATE_IDS')
        positions={str(key):i for i,key in enumerate(keys)}
        if not set(ids)<=positions.keys(): raise PilotCheckError('LABEL_SELECTED_COVERAGE')
        return {sid: np.asarray([table[c][positions[sid]] for c in columns],dtype=np.float32) for sid in ids}
    ys=rows(labels,targets); ms=rows(masks,mask_columns)
    for sid in ids:
        y,m=ys[sid],ms[sid]
        if not np.isfinite(m).all() or not np.isin(m,[0,1]).all(): raise PilotCheckError('MASK_INVALID_VALUES')
        if not np.isfinite(y[m>0]).all() or not ((y[m>0]>=0)&(y[m>0]<=1)).all(): raise PilotCheckError('LABEL_OBSERVED_TARGET_INVALID')
        ys[sid]=np.where(m>0,y,0).astype(np.float32)
    return ys,ms


def contributor_ordinals(ids, masks):
    """Valid all-unknown studies contribute no loss and are skipped before shuffle."""
    return [i for i, sid in enumerate(ids) if any(masks[sid])]


def run(public_source, decoder_source, manifest, checkpoint, cache, labels, masks, root, out, contract):
    import csv
    import importlib.metadata
    import numpy as np
    import os
    import resource
    import shutil
    import torch
    from knee.reference import load_definitions,load_model,bounded_ids
    from knee.mirror_input import prepare_study,policy_contract
    if not torch.cuda.is_available(): raise RuntimeError('AdamW qualification requires CUDA')
    if importlib.metadata.version('torch')!='2.10.0+cu128' or importlib.metadata.version('timm')!='1.0.26':
        raise ValueError('Pinned CUDA environment changed')
    out=Path(out); out.mkdir(parents=True,exist_ok=False); cache=Path(cache); root=Path(root)
    started=time.monotonic(); temporary=[out/'resume.pt',out/'reference.pt',out/'continued.pt']
    report={'status':'RUNNING','purpose':'disposable AdamW mechanics only; no validation metrics',
            'contract':contract,'checkpoint_sha256':manifest['checkpoint_sha256'],
            'labels_sha256':LABEL_SHA,'masks_sha256':MASK_SHA,
            'mask_policy':'explicit original m_ observation masks; prospective correction to finite-only TrackA loader','source_revision':os.environ.get('KNEE_CODE_REVISION'),
            'bundle_sha256':os.environ.get('KNEE_SOURCE_SHA256'),'resume_state_atol':0.0,'resume_state_rtol':0.0,
            'tail_gradient_atol':STATE_ATOL,'tail_gradient_rtol':STATE_RTOL,
            'environment':{'torch':torch.__version__,'timm':importlib.metadata.version('timm'),
                           'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)},
            'seed':42,'windows_per_study':12,'window_chunk':4,'checkpointing':'nonreentrant; preserve_rng_state',
            'precision':'FP32','cache_storage_bytes':452996096,'warmup_successful_updates':20,'accumulation':16,'tail_denominator':3,
            'updates':0,'contributing_studies':0,'reference_study_passes':3,'resumed_study_passes':3,
            'center_policy':'Inherited min/max valid-mask span, public fallback center; cross-slot and partial-mask windows preserved; no replacement if at least12 centers',
            'cache_load_seconds':0.0,'training_window_seconds':0.0,'forward_backward_seconds':0.0,
            'free_disk_before':shutil.disk_usage(out).free}
    def save():
        report.update(elapsed_seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (out/'qualification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    stage='storage'
    try:
        if shutil.disk_usage(out).free < 4*1024**3: raise OSError('Insufficient disposable state storage')
        stage='cache_contract'
        observed=cache_contract(json.loads((cache/'receipt.json').read_text()))
        stage='cache_binding'
        if observed!=contract or file_sha(cache/'private-index.json')!=INDEX_SHA: raise ValueError('Cache binding changed')
        stage='source_lineage'
        if manifest['source_sha256']!=contract['source_sha256'] or sha256(canonical_bytes(manifest))!=contract['manifest_sha256']:
            raise ValueError('Public source/cache lineage differs')
        stage='reader_policy'
        if policy_contract()!=json.loads((cache/'receipt.json').read_text())['input_policy']:
            raise ValueError('Protected reader policy changed')
        stage='cache_index'
        index=json.loads((cache/'private-index.json').read_text())
        ids=[r['StudyInstanceUID'] for r in index]
        bounded_ids(ids,count=32)
        if len(ids)!=32 or sha256(canonical_bytes(ids))!=IDS_SHA or [r['ordinal'] for r in index]!=list(range(32)):
            raise ValueError('Cache index membership/order differs')
        stage='public_definitions'
        public,decoder=load_definitions(public_source,decoder_source,manifest)
        def cached(i):
            row=index[i]
            volume=np.load(cache/f'{i:03d}.volume.npy',allow_pickle=False)
            mask=np.load(cache/f'{i:03d}.mask.npy',allow_pickle=False)
            if (volume.shape!=(96,384,384) or volume.dtype!=np.uint8 or mask.shape!=(96,) or
                    str(mask.dtype)!=row['mask_dtype'] or sha256(volume.tobytes())!=row['volume_sha256'] or
                    sha256(mask.tobytes())!=row['mask_sha256'] or not np.isin(mask,[0,1]).all() or not mask.any() or
                    not np.array_equal(mask.astype(bool),volume.reshape(96,-1).any(axis=1))):
                raise ValueError('Cache array integrity failed')
            return volume,mask
        stage='label_preflight'
        ys,ms=load_labels(labels,masks,ids,public.LAB)
        report['observed_cells']=int(sum(ms[s].sum() for s in ids))
        contributors=contributor_ordinals(ids,ms)
        skipped=[i for i in range(32) if i not in contributors]
        if len(contributors)!=30 or len(skipped)!=2: raise PilotCheckError('LABEL_CONTRIBUTOR_COUNT')
        report.update(requested_studies=32,contributing_unique_studies=30,skipped_unknown_studies=2,
                      group_sizes=[16,14],contributor_ids_sha256=sha256(canonical_bytes([ids[i] for i in contributors])),
                      skipped_ids_sha256=sha256(canonical_bytes([ids[i] for i in skipped])))
        report['targets']=public.LAB
        stage='official_metadata'
        if file_sha(root/'train_series.csv')!=contract['series_metadata_sha256']: raise ValueError('Official metadata changed')
        records={}
        with (root/'train_series.csv').open(newline='') as handle:
            for row in csv.DictReader(handle): records.setdefault(row['StudyInstanceUID'],[]).append(row)
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        stage='model_load'
        model=load_model(public,checkpoint,'cuda',manifest)
        if any(p.dtype != torch.float32 for p in model.parameters()) or torch.is_autocast_enabled():
            raise ValueError('Protocol requires actual FP32 parameters/computation')
        torch.cuda.reset_peak_memory_stats()
        parity_start=time.monotonic(); frozen_errors=[]; window_errors=[]
        stage='frozen_parity'
        for i in (0,1):
            raw,rawmask,counts=prepare_study(public,decoder,ids[i],records,root/'train_series')
            volume,mask=cached(i)
            if counts.get('decoded_unique_slices',0)<=0 or not np.array_equal(raw,volume) or not np.array_equal(rawmask,mask):
                raise ValueError('Fresh raw/cache bytes differ')
            raw_windows=[public.gpu_windows(raw,rawmask,94,320,'cuda',mirror=m) for m in (False,True)]
            cached_windows=[public.gpu_windows(volume,mask,94,320,'cuda',mirror=m) for m in (False,True)]
            if any(not torch.equal(a,b) for a,b in zip(raw_windows,cached_windows)): raise ValueError('Raw/cache GPU windows differ')
            a=public.infer_pair_probs(model,*raw_windows,'cuda'); b=public.infer_pair_probs(model,*cached_windows,'cuda')
            error=float(np.max(np.abs(a-b)))
            if not np.isfinite(a).all() or not np.isfinite(b).all() or error>1e-7: raise ValueError('Frozen prediction parity failed')
            frozen_errors.append(error); window_errors.append(0.0)
            del raw,rawmask,volume,mask,raw_windows,cached_windows
        report.update(frozen_parity_studies=2,frozen_prediction_max_error=max(frozen_errors),
                      gpu_window_max_error=max(window_errors),raw_frozen_parity_seconds=time.monotonic()-parity_start)
        stage='optimizer_setup'
        optimizer=torch.optim.AdamW(optimizer_groups(model),betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
        report['optimizer_groups']=[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups]
        original_bn=bn_state(model)
        if not original_bn: raise ValueError('Expected public BN buffers absent')
        probes={family:next((name,p.detach().cpu().clone()) for name,p in model.named_parameters()
                     if (name.startswith('backbone.'))==(family=='encoder')) for family in ('encoder','head')}
        def order(epoch): return np.random.default_rng(42+epoch).permutation(contributors).tolist()
        sampler={'epoch':0,'position':0,'order':order(0)}
        report['sampler_order_sha256']=sha256(canonical_bytes(sampler['order']))
        def accumulate(chosen,epoch,denominator):
            training_mode(model); optimizer.zero_grad(set_to_none=True)
            for i in chosen:
                tick=time.monotonic()
                volume,mask=cached(i)
                report['cache_load_seconds']+=time.monotonic()-tick
                tick=time.monotonic()
                windows=sampled_windows(public,volume,mask,ids[i],epoch)
                torch.cuda.synchronize()
                report['training_window_seconds']+=time.monotonic()-tick
                tick=time.monotonic()
                logits=checkpoint_forward(model,windows)
                y=torch.from_numpy(ys[ids[i]])[None].to('cuda'); m=torch.from_numpy(ms[ids[i]])[None].to('cuda')
                (masked_loss(logits,y,m)/denominator).backward()
                torch.cuda.synchronize()
                report['forward_backward_seconds']+=time.monotonic()-tick
                del windows,volume,mask,logits
            require_bn(model,original_bn)
        train_start=time.monotonic()
        stage='full_updates'
        for update in range(2):
            chosen=sampler['order'][sampler['position']:sampler['position']+16]
            if len(chosen)!=[16,14][update]: raise ValueError('Accumulation membership differs')
            accumulate(chosen,0,len(chosen))
            counts,norm=apply_update(model,optimizer,update)
            sampler['position']+=len(chosen); report['updates']+=1; report['contributing_studies']+=len(chosen)
            report['last_gradient_counts']=counts; report['last_unclipped_grad_norm']=norm; save()
        if sampler['position']!=30: raise ValueError('Contributor epoch coverage differs')
        sampler={'epoch':1,'position':0,'order':order(1)}
        report['next_epoch_sampler_order_sha256']=sha256(canonical_bytes(sampler['order']))
        save_state(temporary[0],model,optimizer,sampler,2,contract)
        chosen=sampler['order'][:3]
        accumulate(chosen,1,3)
        actual_grad={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
        apply_update(model,optimizer,2); sampler['position']=3
        save_state(temporary[1],model,optimizer,sampler,3,contract)
        # Reopen saved full state; reference accumulates unscaled study losses,
        # then divides by actual3. This would detect an accidental /16 tail.
        stage='tail_reference'
        restored,updates=restore_state(temporary[0],model,optimizer,contract)
        if restored!={'epoch':1,'position':0,'order':order(1)} or updates!=2: raise ValueError('Sampler/counter resume differs')
        accumulate(restored['order'][:3],restored['epoch'],1)
        tail_error=0.0
        for name,p in model.named_parameters():
            p.grad.div_(3)
            reference=p.grad.detach().cpu(); actual=actual_grad[name]
            if not torch.allclose(actual,reference,atol=STATE_ATOL,rtol=STATE_RTOL): raise ValueError('Tail denominator reference failed')
            tail_error=max(tail_error,float((actual-reference).abs().max()))
        del actual_grad
        stage='resume_continuation'
        # Resume proof uses IDENTICAL /3 arithmetic, separately from reordered
        # tail reference. Deterministic model/optimizer/RNG/sampler must be exact.
        restored,updates=restore_state(temporary[0],model,optimizer,contract)
        accumulate(restored['order'][:3],restored['epoch'],3)
        apply_update(model,optimizer,updates); restored['position']=3
        save_state(temporary[2],model,optimizer,restored,3,contract)
        a=torch.load(temporary[1],map_location='cpu',weights_only=True)
        b=torch.load(temporary[2],map_location='cpu',weights_only=True)
        model_error=tensor_max_error(a['model'],b['model'])
        report['measured_resume_model_max_error']=model_error
        report['measured_resume_optimizer_max_error']=tensor_max_error(a['optimizer'],b['optimizer'])
        model_error=assert_state_close(a['model'],b['model'],atol=0,rtol=0)
        optimizer_error=assert_state_close(a['optimizer'],b['optimizer'],atol=0,rtol=0)
        assert_state_close(a['rng'],b['rng'],atol=0,rtol=0)
        assert_state_close(a['sampler'],b['sampler'],atol=0,rtol=0)
        if a['updates']!=b['updates'] or b['updates']!=3: raise ValueError('Resume update count differs')
        del a,b
        stage='reload_predictions'
        volume,mask=cached(0)
        probe_windows=sampled_windows(public,volume,mask,ids[0],1)
        model.eval()
        first=torch.load(temporary[1],map_location='cpu',weights_only=True)
        model.load_state_dict(first['model'],strict=True); del first
        with torch.no_grad(): expected=checkpoint_forward(model,probe_windows).detach().cpu()
        second=torch.load(temporary[2],map_location='cpu',weights_only=True)
        model.load_state_dict(second['model'],strict=True); del second
        with torch.no_grad(): actual=checkpoint_forward(model,probe_windows).detach().cpu()
        prediction_error=assert_state_close(expected,actual,atol=0,rtol=0)
        del volume,mask,probe_windows
        require_bn(model,original_bn)
        changes={family:float((dict(model.named_parameters())[name].detach().cpu()-before).abs().max())
                 for family,(name,before) in probes.items()}
        if any(v<=0 or not np.isfinite(v) for v in changes.values()): raise ValueError('Group parameter update absent')
        report.update(updates=3,contributing_studies=33,tail_gradient_max_error=tail_error,
                      resume_model_max_error=model_error,resume_optimizer_max_error=optimizer_error,
                      resume_rng_exact=True,resume_sampler_exact=True,bn_buffers_unchanged=True,
                      save_reload_logit_max_error=prediction_error,
                      group_parameter_max_changes=changes,training_seconds=time.monotonic()-train_start,
                      final_lrs={g['name']:g['lr'] for g in optimizer.param_groups},status='PASS')
        save()
    except Exception as error:
        report.update(status='FAIL',failure_type=type(error).__name__,**failure_diagnostic(stage,error))
        save()
        raise RuntimeError('AdamW qualification failed; inspect sanitized aggregate receipt') from None
    finally:
        for path in temporary: path.unlink(missing_ok=True)
        report.update(disposable_states_deleted=all(not p.exists() for p in temporary),free_disk_after=shutil.disk_usage(out).free)
        save()
    return report
