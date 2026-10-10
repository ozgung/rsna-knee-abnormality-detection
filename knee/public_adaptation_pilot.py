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
STREAM_RECEIPT_SHA = '17e8e3aeb7bcfab660e7fe9b6804a7bd513886fceb451a80ed3eadff3fc5fb64'
STREAM_REVISION = '429997b7aa8bda1607e0dc6d5974ea1ca042508a'
STREAM_BUNDLE_SHA = '017a63d249cb8d1aa969652f2645328dbc1f18b526627a871f558be2deead1a6'


SAFE_CHECKS = frozenset({'LABEL_BYTES', 'LABEL_TARGET_SCHEMA', 'MASK_TARGET_SCHEMA',
    'LABEL_DUPLICATE_IDS', 'LABEL_SELECTED_COVERAGE', 'MASK_INVALID_VALUES',
    'LABEL_OBSERVED_TARGET_INVALID', 'LABEL_NONCONTRIBUTING_STUDIES', 'LABEL_CONTRIBUTOR_COUNT', 'BN_BUFFERS_CHANGED', 'RESUME_CONTRACT_MISMATCH',
    'RESUME_SAMPLER_MISMATCH', 'RESUME_UPDATE_COUNTER_MISMATCH', 'RESUME_MODEL_MISMATCH',
    'RESUME_OPTIMIZER_MISMATCH', 'RESUME_RNG_MISMATCH', 'RESUME_PREDICTION_MISMATCH',
    'TAIL_GRADIENT_MISMATCH', 'TAIL_GRADIENT_INVALID', 'RESUME_NOT_QUALIFIED', 'TAIL_FORWARD_MISMATCH', 'GROUP_COUNT_MISMATCH',
    'ARTIFACT_BINDING_MISMATCH', 'ARTIFACT_STATE_MISMATCH', 'STREAM_INPUT_MISMATCH',
    'STREAM_CURSOR_MISMATCH', 'TAIL_GROSS_DISCREPANCY'})
SAFE_STAGES = frozenset({'storage', 'cache_contract', 'cache_binding', 'source_lineage',
    'reader_policy', 'cache_index', 'public_definitions', 'label_preflight',
    'official_metadata', 'model_load', 'frozen_parity', 'optimizer_setup',
    'full_updates', 'tail_reference', 'resume_continuation', 'reload_predictions',
    'baseline_tail_update', 'resume_restore', 'tail_reference_restore', 'tail_gradient_comparison',
    'stream_binding', 'artifact_restore', 'artifact_export', 'artifact_expectation'})


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
        raise PilotCheckError('BN_BUFFERS_CHANGED')


def masked_loss(logits, targets, mask):
    import torch
    if logits.shape != targets.shape or mask.shape != targets.shape or not torch.isfinite(logits).all():
        raise ValueError('Loss shape/finiteness failed')
    if not torch.isfinite(targets).all() or not ((targets>=0)&(targets<=1)).all() or not ((mask==0)|(mask==1)).all():
        raise ValueError('Invalid soft targets/masks')
    denominator = mask.sum()
    if denominator <= 0: raise ValueError('Noncontributing study')
    return (torch.nn.functional.binary_cross_entropy_with_logits(logits if logits.dtype==torch.float64 else logits.float(), targets, reduction='none') * mask).sum()/denominator



def study_mean_backward(loss, actual_count):
    """Production operation: one observed-cell study mean / actual group count."""
    import torch
    if type(actual_count) is not int or not 1 <= actual_count <= 16:
        raise PilotCheckError('GROUP_COUNT_MISMATCH')
    if loss.ndim != 0 or not torch.isfinite(loss): raise PilotCheckError('TAIL_GRADIENT_INVALID')
    (loss / actual_count).backward()


def require_group_counts(group_count, denominator, processed):
    if (any(type(v) is not int for v in (group_count,denominator,processed)) or
            not 1 <= group_count <= 16 or group_count != denominator or processed != group_count):
        raise PilotCheckError('GROUP_COUNT_MISMATCH')


def require_exact_gradients(actual, reference):
    """Exact preclip tensor gate, independent of old elementwise tolerances."""
    import torch
    if actual.keys()!=reference.keys(): raise PilotCheckError('TAIL_GRADIENT_INVALID')
    for name,a in actual.items():
        b=reference[name]
        if (not torch.is_tensor(a) or not torch.is_tensor(b) or a.shape!=b.shape or a.dtype!=b.dtype or
                not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise PilotCheckError('TAIL_GRADIENT_INVALID')
        if not torch.equal(a.cpu(),b.cpu()): raise PilotCheckError('TAIL_GRADIENT_MISMATCH')


def stream_contract(receipt, contract):
    if (sha256(canonical_bytes(receipt))!=STREAM_RECEIPT_SHA or receipt.get('contract')!=contract or
            receipt.get('source_revision')!=STREAM_REVISION or receipt.get('source_bundle_sha256')!=STREAM_BUNDLE_SHA or
            receipt.get('status')!='PASS' or receipt.get('completed_studies')!=32 or
            not all(receipt.get(k) is True for k in ('byte_parity','sampled_window_parity','restart_exact',
                'rng_unchanged','producer_failure_propagated','central_crop_after_quantization'))):
        raise PilotCheckError('ARTIFACT_BINDING_MISMATCH')
    return {'receipt_sha256':STREAM_RECEIPT_SHA,'source_revision':STREAM_REVISION,'bundle_sha256':STREAM_BUNDLE_SHA}



def decode_metrics(records):
    """Snapshot immutable completed producer records; caller owns report writes."""
    completed=tuple(records)
    counters={}
    for _,_,items in completed:
        for key,value in items:counters[key]=counters.get(key,0)+value
    return {'decode_calls':len(completed),'decode_seconds':sum(end-start for start,end,_ in completed),
            'stream_counters':counters}


def scientific_config():
    return {'precision':'FP32','seed':42,'windows':12,'chunk':4,'checkpointing':'nonreentrant-preserve-rng',
            'encoder_lr':1e-6,'head_lr':1e-5,'warmup':20,'decay':.01,'betas':[.9,.999],'eps':1e-8,
            'clip_norm':1.0,'dropout':.2,'bn_stats':'frozen','groups':[16,14],'next_tail':3,
            'scaling':'per-study-observed-cell-mean / actual-group-count','labels_sha256':LABEL_SHA,
            'masks_sha256':MASK_SHA,'prefetch':4,'lru':0,'decoder_cleanup_timeout_seconds':30.0,'reordered_relative_budget':1e-5}


def recovery_binding(contract, stream):
    import os
    return {'cache':contract,'stream':stream,'scientific_config':scientific_config(),
            'source_revision':os.environ.get('KNEE_SCIENTIFIC_REVISION',os.environ.get('KNEE_CODE_REVISION')),
            'bundle_sha256':os.environ.get('KNEE_SOURCE_SHA256')}


def validate_recovery_manifest(manifest, binding):
    import re
    fields={'schema','purpose','binding','checkpoint_file','checkpoint_bytes','checkpoint_sha256',
            'updates_before','updates_after','consumed_before','consumed_after','expected','status'}
    if (not isinstance(manifest,dict) or set(manifest)!=fields or manifest.get('schema')!=1 or manifest.get('purpose')!='disposable external recovery qualification' or
            manifest.get('status')!='PASS' or manifest.get('binding')!=binding or manifest.get('checkpoint_file')!='recovery.pt' or
            type(manifest.get('checkpoint_bytes')) is not int or manifest['checkpoint_bytes']<=0 or
            [manifest.get(k) for k in ('updates_before','updates_after','consumed_before','consumed_after')]!=[2,3,0,3] or
            not isinstance(manifest.get('expected'),dict) or set(manifest.get('expected',{}))!={'model','optimizer','rng','sampler','gradients','prediction','forward'} or
            any(not isinstance(x,str) or not re.fullmatch('[0-9a-f]{64}',x)
                for x in [manifest.get('checkpoint_sha256'),*manifest.get('expected',{}).values()])):
        raise PilotCheckError('ARTIFACT_BINDING_MISMATCH')
    return manifest


def verify_recovery_artifact(root, expected_manifest, binding):
    root=Path(root)
    validate_recovery_manifest(expected_manifest,binding)
    actual=json.loads((root/'recovery-manifest.json').read_text())
    if actual!=expected_manifest: raise PilotCheckError('ARTIFACT_BINDING_MISMATCH')
    checkpoint=root/'recovery.pt'
    if checkpoint.stat().st_size!=actual['checkpoint_bytes'] or file_sha(checkpoint)!=actual['checkpoint_sha256']:
        raise PilotCheckError('ARTIFACT_BINDING_MISMATCH')
    return checkpoint


def expected_fingerprints(state, gradients, prediction, forward):
    return {**{k:state_fingerprint(state[k]) for k in ('model','optimizer','rng','sampler')},
            'gradients':state_fingerprint(gradients),'prediction':state_fingerprint(prediction),
            'forward':state_fingerprint(forward)}


def verify_expected_fingerprints(actual, expected):
    if actual!=expected: raise PilotCheckError('ARTIFACT_STATE_MISMATCH')


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
    if state['contract'] != contract: raise PilotCheckError('RESUME_CONTRACT_MISMATCH')
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


def require_exact_state(actual, expected, check):
    """Keep exact checks separately attributable without exposing exception text."""
    try: return assert_state_close(actual, expected, atol=0, rtol=0)
    except ValueError: raise PilotCheckError(check) from None


def gradient_comparison(actual, reference):
    """Complete preclip FP64 aggregate reductions; no gradient arrays exported."""
    import math
    import torch
    if actual.keys() != reference.keys(): raise PilotCheckError('TAIL_GRADIENT_INVALID')
    def empty():
        return dict(error2=0.0,actual2=0.0,reference2=0.0,dot=0.0,max_abs_error=0.0,
                    max_relative_error_nonzero_reference=0.0,mismatched_elements=0,
                    mismatched_parameters=0,parameters_compared=0,elements_compared=0,
                    zero_reference_elements=0,zero_reference_nonzero_actual=0)
    def merge(total,local):
        for key,value in local.items():
            if key.startswith('max_'): total[key]=max(total[key],value)
            else: total[key]+=value
    def finish(stats):
        x=dict(stats); actual_norm=math.sqrt(x.pop('actual2')); reference_norm=math.sqrt(x.pop('reference2'))
        error=math.sqrt(x.pop('error2')); dot=x.pop('dot')
        x.update(actual_norm=actual_norm,reference_norm=reference_norm,l2_error=error,
                 relative_l2_error=error/reference_norm if reference_norm else (0.0 if not error else None),
                 norm_ratio=actual_norm/reference_norm if reference_norm else None,
                 cosine_similarity=dot/(actual_norm*reference_norm) if actual_norm and reference_norm else None,
                 fitted_scale=dot/reference_norm**2 if reference_norm else None)
        return x
    totals=empty(); families={f:empty() for f in ('encoder','head')};worst=None;worst_stats=None
    for name,a in actual.items():
        b=reference[name]
        if (not torch.is_tensor(a) or not torch.is_tensor(b) or a.shape!=b.shape or a.dtype!=b.dtype or
                not torch.isfinite(a).all() or not torch.isfinite(b).all()):
            raise PilotCheckError('TAIL_GRADIENT_INVALID')
        a=a.detach().cpu(); b=b.detach().cpu(); x=a.double(); y=b.double(); d=x-y
        local=empty(); zero=y==0; nonzero=~zero
        local.update(error2=float(d.square().sum()),actual2=float(x.square().sum()),
                     reference2=float(y.square().sum()),dot=float((x*y).sum()),
                     max_abs_error=float(d.abs().max()) if a.numel() else 0.0,
                     max_relative_error_nonzero_reference=float((d[nonzero].abs()/y[nonzero].abs()).max()) if nonzero.any() else 0.0,
                     mismatched_elements=int((~torch.isclose(a,b,atol=STATE_ATOL,rtol=STATE_RTOL)).sum()),
                     parameters_compared=1,elements_compared=a.numel(),zero_reference_elements=int(zero.sum()),
                     zero_reference_nonzero_actual=int((zero&(x!=0)).sum()))
        local['mismatched_parameters']=int(local['mismatched_elements']>0)
        if worst is None or local['max_abs_error']>worst_stats['max_abs_error']: worst=name;worst_stats=local
        merge(totals,local);merge(families['encoder' if name.startswith('backbone.') else 'head'],local)
    return {**finish(totals),'worst_parameter':worst,'worst_tensor':finish(worst_stats) if worst_stats else {},
            'per_family':{name:finish(v) for name,v in families.items()},'atol':STATE_ATOL,'rtol':STATE_RTOL}


def evaluate_tail_reference(report, actual, reference):
    """Exact replay is mandatory first; preserve all aggregates before rejection."""
    if report.get('exact_resume_status')!='PASS' or report.get('save_reload_status')!='PASS':
        raise PilotCheckError('RESUME_NOT_QUALIFIED')
    comparison=gradient_comparison(actual,reference)
    report['tail_gradient_comparison']=comparison
    forward=report.get('reordered_reference_forward')
    if forward is not None:
        if not forward['rng_exact']: raise PilotCheckError('RESUME_RNG_MISMATCH')
        if not all(forward[k] for k in ('input_exact','logits_exact','raw_loss_exact')):
            raise PilotCheckError('TAIL_FORWARD_MISMATCH')
    import math
    for metrics in [comparison,*comparison['per_family'].values()]:
        values=[metrics['relative_l2_error'],metrics['fitted_scale']]
        if (any(v is None or not math.isfinite(v) for v in values) or
                metrics['reference_norm']<=0 or values[0]>1e-5 or abs(values[1]-1)>1e-5):
            raise PilotCheckError('TAIL_GROSS_DISCREPANCY')
    return comparison


def state_fingerprint(state):
    """Aggregate signature for saved RNG/forward tensors; values stay in memory."""
    import torch
    digest=hashlib.sha256()
    def token(value):
        encoded=canonical_bytes(value);digest.update(len(encoded).to_bytes(8,'big'));digest.update(encoded)
    def add(value):
        if torch.is_tensor(value):
            v=value.detach().cpu().contiguous()
            token(['tensor',str(v.dtype),list(v.shape)])
            raw=v.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8,'big'));digest.update(raw)
        elif isinstance(value,dict):
            token(['dict',len(value)])
            for key in sorted(value,key=lambda k:(type(k).__name__,canonical_bytes(k))):
                token([type(key).__name__,key]);add(value[key])
        elif isinstance(value,(list,tuple)):
            token([type(value).__name__,len(value)])
            for item in value: add(item)
        else: token([type(value).__name__,value])
    add(state);return digest.hexdigest()


def forward_comparison(actual, reference):
    """Aggregate three-tail input/RNG/logit/raw-loss replay evidence only."""
    import torch
    if len(actual)!=3 or len(reference)!=3: raise PilotCheckError('TAIL_FORWARD_MISMATCH')
    a_logits=torch.cat([v['logits'].flatten() for v in actual]);b_logits=torch.cat([v['logits'].flatten() for v in reference])
    a_loss=torch.stack([v['loss'] for v in actual]);b_loss=torch.stack([v['loss'] for v in reference])
    result={}
    for name in ('input_sha256','rng_sha256'):
        a=sha256(canonical_bytes([v[name] for v in actual]));b=sha256(canonical_bytes([v[name] for v in reference]))
        result[name.replace('_sha256','_fingerprint')+'_actual']=a
        result[name.replace('_sha256','_fingerprint')+'_reference']=b
        result[name.replace('_sha256','_exact')]=a==b
    result.update(logits_fingerprint_actual=state_fingerprint(a_logits),logits_fingerprint_reference=state_fingerprint(b_logits),
                  raw_loss_fingerprint_actual=state_fingerprint(a_loss),raw_loss_fingerprint_reference=state_fingerprint(b_loss),
                  logits_exact=torch.equal(a_logits,b_logits),raw_loss_exact=torch.equal(a_loss,b_loss),
                  logits_max_abs_error=float((a_logits-b_logits).abs().max()),raw_loss_max_abs_error=float((a_loss-b_loss).abs().max()))
    return result


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


def run(public_source, decoder_source, manifest, checkpoint, cache, labels, masks, root, out, contract, *,
        streamed=False, qualified_stream=None, recovery_root=None, expected_manifest=None, export_recovery=False):
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
    decode_records=[]; gpu_intervals=[]
    report={'status':'RUNNING','purpose':'disposable AdamW mechanics only; no validation metrics',
            'contract':contract,'checkpoint_sha256':manifest['checkpoint_sha256'],
            'labels_sha256':LABEL_SHA,'masks_sha256':MASK_SHA,
            'mask_policy':'explicit original m_ observation masks; prospective correction to finite-only TrackA loader','source_revision':os.environ.get('KNEE_CODE_REVISION'),
            'bundle_sha256':os.environ.get('KNEE_SOURCE_SHA256'),'resume_state_atol':0.0,'resume_state_rtol':0.0,
            'tail_gradient_atol':STATE_ATOL,'tail_gradient_rtol':STATE_RTOL,
            'elementwise_reordered_tolerance_role':'informational only; exact inline gate is zero',
            'reordered_global_and_family_l2_scale_budget':1e-5,'exact_preclip_gradient_tolerance':0.0,
            'environment':{'torch':torch.__version__,'timm':importlib.metadata.version('timm'),
                           'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)},
            'seed':42,'windows_per_study':12,'window_chunk':4,'checkpointing':'nonreentrant; preserve_rng_state',
            'precision':'FP32','cache_storage_bytes':452996096,'warmup_successful_updates':20,'accumulation':16,'tail_denominator':3,
            'updates':0,'contributing_studies':0,'reference_study_passes':3,'resumed_study_passes':3,
            'center_policy':'Inherited min/max valid-mask span, public fallback center; cross-slot and partial-mask windows preserved; no replacement if at least12 centers',
            'cache_load_seconds':0.0,'training_window_seconds':0.0,'forward_backward_seconds':0.0,'actual_forward_backward_passes':0,
            'free_disk_before':shutil.disk_usage(out).free}
    def save():
        if streamed:report.update(decode_metrics(decode_records))
        free_now=shutil.disk_usage(out).free
        report['minimum_free_disk_bytes']=min(report.get('minimum_free_disk_bytes',free_now),free_now)
        report.update(elapsed_seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        (out/'qualification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    active_stream=None
    state_contract=contract
    stage='storage'
    try:
        if streamed:
            if (recovery_root is None)!=(expected_manifest is None):raise PilotCheckError('ARTIFACT_BINDING_MISMATCH')
            stage='stream_binding'
            if qualified_stream!={'receipt_sha256':STREAM_RECEIPT_SHA,'source_revision':STREAM_REVISION,'bundle_sha256':STREAM_BUNDLE_SHA}:
                raise PilotCheckError('ARTIFACT_BINDING_MISMATCH')
            state_contract=recovery_binding(contract,qualified_stream)
            report.update(source_revision=state_contract['source_revision'],packaging_revision=os.environ.get('KNEE_CODE_REVISION'))
            report.update(input_mode='protected raw stream',qualified_cpu_stream=qualified_stream,scientific_config=scientific_config(),
                          prefetch_limit=4,lru_capacity=0,decode_calls=0,decode_seconds=0.0,consumer_wait_seconds=0.0,observed_queue_peak=0,stream_counters={})
        elif recovery_root is not None or export_recovery:
            raise PilotCheckError('ARTIFACT_BINDING_MISMATCH')
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
        if not (root/'train_series').is_dir() or any(not records.get(sid) or not (root/'train_series'/sid).is_dir() for sid in ids):
            raise PilotCheckError('STREAM_INPUT_MISMATCH')
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
        def raw_read(i):
            tick=time.monotonic()
            volume,mask,c=prepare_study(public,decoder,ids[i],records,root/'train_series')
            row=index[i]
            if (c.get('decoded_unique_slices',0)<=0 or c.get('resource_errors',0) or
                    volume.shape!=(96,384,384) or volume.dtype!=np.uint8 or mask.shape!=(96,) or
                    str(mask.dtype)!=row['mask_dtype'] or sha256(volume.tobytes())!=row['volume_sha256'] or
                    sha256(mask.tobytes())!=row['mask_sha256']):
                raise PilotCheckError('STREAM_INPUT_MISMATCH')
            end=time.monotonic();decode_records.append((tick,end,tuple(sorted(c.items()))))
            return volume,mask
        def start_stream(state):
            from knee.public_study_stream import StudyStream
            return StudyStream(state['order'],raw_read,contract=sha256(canonical_bytes(state_contract)),
                               consumed_cursor=state['position'],prefetch=4,close_timeout=30.0)
        def close_stream():
            nonlocal active_stream
            if active_stream is not None:
                with active_stream._condition:
                    report['observed_queue_peak']=max(report['observed_queue_peak'],len(active_stream._buffer))
                    report['last_fetched_cursor']=active_stream.fetched_cursor
                    report['last_consumed_cursor']=active_stream.consumed_cursor
                active_stream.close();active_stream=None
        def accumulate(chosen,epoch,denominator,capture=None,operation='helper',stream=None):
            training_mode(model); optimizer.zero_grad(set_to_none=True)
            if operation!='reordered':require_group_counts(len(chosen),denominator,len(chosen))
            processed=0
            for i in chosen:
                tick=time.monotonic()
                if stream is None:
                    volume,mask=cached(i);report['cache_load_seconds']+=time.monotonic()-tick
                else:
                    volume,mask=next(stream);report['consumer_wait_seconds']+=time.monotonic()-tick
                    with stream._condition:report['observed_queue_peak']=max(report['observed_queue_peak'],len(stream._buffer))
                tick=time.monotonic()
                windows=sampled_windows(public,volume,mask,ids[i],epoch)
                torch.cuda.synchronize()
                report['training_window_seconds']+=time.monotonic()-tick
                tick=time.monotonic()
                before_rng=state_fingerprint(rng_state()) if capture is not None else None
                logits=checkpoint_forward(model,windows)
                y=torch.from_numpy(ys[ids[i]])[None].to('cuda'); m=torch.from_numpy(ms[ids[i]])[None].to('cuda')
                loss=masked_loss(logits,y,m)
                if capture is not None:
                    capture.append({'input_sha256':state_fingerprint(windows), 'rng_sha256':before_rng,
                                    'logits':logits.detach().cpu().clone(),'loss':loss.detach().cpu().clone()})
                if operation=='helper':study_mean_backward(loss,len(chosen))
                elif operation=='inline':(loss/len(chosen)).backward()
                elif operation=='reordered':loss.backward()
                else:raise PilotCheckError('GROUP_COUNT_MISMATCH')
                torch.cuda.synchronize();end=time.monotonic();gpu_intervals.append((tick,end))
                report['forward_backward_seconds']+=end-tick
                report['actual_forward_backward_passes']+=1;processed+=1
                if stream is not None:stream.acknowledge()
                del windows,volume,mask,logits
            if operation!='reordered':require_group_counts(len(chosen),denominator,processed)
            require_bn(model,original_bn)
        train_start=time.monotonic()
        if recovery_root is None:
            if streamed:active_stream=start_stream(sampler)
            stage='full_updates'
            report['actual_group_denominators']=[]
            for update in range(2):
                chosen=sampler['order'][sampler['position']:sampler['position']+16]
                if len(chosen)!=[16,14][update]: raise PilotCheckError('GROUP_COUNT_MISMATCH')
                accumulate(chosen,0,len(chosen),stream=active_stream)
                if active_stream is not None and active_stream.consumed_cursor!=sampler['position']+len(chosen):
                    raise PilotCheckError('STREAM_CURSOR_MISMATCH')
                counts,norm=apply_update(model,optimizer,update)
                sampler['position']+=len(chosen); report['updates']+=1; report['contributing_studies']+=len(chosen)
                report['actual_group_denominators'].append(len(chosen))
                report['last_gradient_counts']=counts; report['last_unclipped_grad_norm']=norm; save()
            close_stream()
            if sampler['position']!=30: raise PilotCheckError('STREAM_CURSOR_MISMATCH')
            sampler={'epoch':1,'position':0,'order':order(1)}
        else:
            stage='artifact_restore'
            recovery=verify_recovery_artifact(recovery_root,expected_manifest,state_contract)
            sampler,updates=restore_state(recovery,model,optimizer,state_contract)
            if sampler!={'epoch':1,'position':0,'order':order(1)}:raise PilotCheckError('RESUME_SAMPLER_MISMATCH')
            if updates!=2:raise PilotCheckError('RESUME_UPDATE_COUNTER_MISMATCH')
            require_bn(model,original_bn)
            report.update(updates=2,contributing_studies=0,actual_group_denominators=[],external_artifact_bytes_verified=True,
                          reference_study_passes=0,resumed_study_passes=0)
        report['next_epoch_sampler_order_sha256']=sha256(canonical_bytes(sampler['order']))
        save_state(temporary[0],model,optimizer,sampler,2,state_contract)
        chosen=sampler['order'][:3]
        stage='baseline_tail_update'
        baseline_forward=[]
        if streamed:active_stream=start_stream(sampler)
        accumulate(chosen,1,3,capture=baseline_forward,stream=active_stream)
        if active_stream is not None and active_stream.consumed_cursor!=3:raise PilotCheckError('STREAM_CURSOR_MISMATCH')
        close_stream()
        actual_grad={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
        apply_update(model,optimizer,2); sampler['position']=3
        report.update(updates=3,contributing_studies=33 if recovery_root is None else 3)
        report['actual_group_denominators'].append(3)
        save_state(temporary[1],model,optimizer,sampler,3,state_contract)
        if recovery_root is not None:
            stage='artifact_expectation'
            state=torch.load(temporary[1],map_location='cpu',weights_only=True)
            reloaded_sampler,reloaded_updates=restore_state(temporary[1],model,optimizer,state_contract)
            if reloaded_sampler!={'epoch':1,'position':3,'order':order(1)}:raise PilotCheckError('RESUME_SAMPLER_MISMATCH')
            if reloaded_updates!=3:raise PilotCheckError('RESUME_UPDATE_COUNTER_MISMATCH')
            require_bn(model,original_bn)
            volume,mask=cached(0);probe_windows=sampled_windows(public,volume,mask,ids[0],1)
            model.eval()
            with torch.no_grad():prediction=checkpoint_forward(model,probe_windows).detach().cpu()
            observed_expected=expected_fingerprints(state,actual_grad,prediction,baseline_forward)
            verify_expected_fingerprints(observed_expected,expected_manifest['expected'])
            report.update(external_artifact_recovery_status='PASS',external_expected_fingerprints=observed_expected,
                          status='PASS',bn_buffers_unchanged=True,exact_preclip_gradients=True,
                          qualified_optimizer_updates=1,actual_optimizer_step_calls=1,external_next_study_forward_backward_passes=3,training_seconds=time.monotonic()-train_start,
                          exact_resume_status='PASS',save_reload_status='PASS',updates=3,contributing_studies=3)
            require_bn(model,original_bn);save()
            return report
        # Qualify identical arithmetic BEFORE independently reordered scaling.
        stage='resume_restore'
        restored,updates=restore_state(temporary[0],model,optimizer,state_contract)
        if restored!={'epoch':1,'position':0,'order':order(1)}: raise PilotCheckError('RESUME_SAMPLER_MISMATCH')
        if updates!=2: raise PilotCheckError('RESUME_UPDATE_COUNTER_MISMATCH')
        require_bn(model,original_bn)
        stage='resume_continuation'
        replay_forward=[]
        accumulate(restored['order'][:3],restored['epoch'],3,capture=replay_forward,operation='inline')
        report['identical_replay_forward']=forward_comparison(baseline_forward,replay_forward)
        if not report['identical_replay_forward']['rng_exact']: raise PilotCheckError('RESUME_RNG_MISMATCH')
        if not all(report['identical_replay_forward'][k] for k in ('input_exact','logits_exact','raw_loss_exact')):
            raise PilotCheckError('TAIL_FORWARD_MISMATCH')
        replay_grad={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()}
        report['identical_preclip_gradient_max_error']=tensor_max_error(actual_grad,replay_grad)
        require_exact_gradients(actual_grad,replay_grad);del replay_grad
        report['independent_inline_gradients_exact']=True
        report['stream_rng_unchanged_via_exact_cached_replay']=bool(streamed)
        apply_update(model,optimizer,updates); restored['position']=3
        save_state(temporary[2],model,optimizer,restored,3,state_contract)
        a=torch.load(temporary[1],map_location='cpu',weights_only=True)
        b=torch.load(temporary[2],map_location='cpu',weights_only=True)
        report['measured_resume_model_max_error']=tensor_max_error(a['model'],b['model'])
        report['measured_resume_optimizer_max_error']=tensor_max_error(a['optimizer'],b['optimizer'])
        model_error=require_exact_state(a['model'],b['model'],'RESUME_MODEL_MISMATCH')
        optimizer_error=require_exact_state(a['optimizer'],b['optimizer'],'RESUME_OPTIMIZER_MISMATCH')
        require_exact_state(a['rng'],b['rng'],'RESUME_RNG_MISMATCH')
        require_exact_state(a['sampler'],b['sampler'],'RESUME_SAMPLER_MISMATCH')
        if a['updates']!=b['updates'] or b['updates']!=3: raise PilotCheckError('RESUME_UPDATE_COUNTER_MISMATCH')
        del a,b
        report.update(exact_resume_status='PASS',resume_model_max_error=model_error,
                      resume_optimizer_max_error=optimizer_error,resume_rng_exact=True,resume_sampler_exact=True)
        save()
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
        prediction_error=require_exact_state(expected,actual,'RESUME_PREDICTION_MISMATCH')
        del volume,mask,probe_windows
        require_bn(model,original_bn)
        report.update(save_reload_logit_max_error=prediction_error,save_reload_status='PASS',bn_buffers_unchanged=True)
        save()
        changes={family:float((dict(model.named_parameters())[name].detach().cpu()-before).abs().max())
                 for family,(name,before) in probes.items()}
        if any(v<=0 or not np.isfinite(v) for v in changes.values()): raise ValueError('Group parameter update absent')
        report['group_parameter_max_changes']=changes
        save()
        qualified_state=torch.load(temporary[2],map_location='cpu',weights_only=True)
        expected_recovery=expected_fingerprints(qualified_state,actual_grad,actual,baseline_forward)
        del qualified_state
        # Reordered scaling is informational, with prospectively fixed global/family safeguard.
        # Restore the SAME pre-tail model/RNG; do not compare different updates.
        stage='tail_reference_restore'
        restored,updates=restore_state(temporary[0],model,optimizer,state_contract)
        if restored!={'epoch':1,'position':0,'order':order(1)}: raise PilotCheckError('RESUME_SAMPLER_MISMATCH')
        if updates!=2: raise PilotCheckError('RESUME_UPDATE_COUNTER_MISMATCH')
        require_bn(model,original_bn)
        stage='tail_reference'
        reference_forward=[]
        accumulate(restored['order'][:3],restored['epoch'],1,capture=reference_forward,operation='reordered')
        report['reordered_reference_forward']=forward_comparison(baseline_forward,reference_forward)
        reference_grad={}
        for name,p in model.named_parameters():
            if p.grad is None: raise PilotCheckError('TAIL_GRADIENT_INVALID')
            p.grad.div_(3)
            reference_grad[name]=p.grad.detach().cpu()
        stage='tail_gradient_comparison'
        comparison=evaluate_tail_reference(report,actual_grad,reference_grad)
        save()  # Outer failure handler also persists complete aggregates on mismatch.
        tail_error=comparison['max_abs_error']
        del actual_grad,reference_grad
        if not all(report['reordered_reference_forward'][name] for name in ('input_exact','rng_exact','logits_exact','raw_loss_exact')):
            raise PilotCheckError('TAIL_FORWARD_MISMATCH')
        # Parameter-update probes refer to the qualified continuation, not the
        # pre-tail state temporarily restored for the independent gradient check.
        restored,updates=restore_state(temporary[2],model,optimizer,state_contract)
        require_bn(model,original_bn)
        report.update(updates=3,contributing_studies=33,tail_gradient_max_error=tail_error,qualified_optimizer_updates=3,actual_optimizer_step_calls=4,
                      resume_model_max_error=model_error,resume_optimizer_max_error=optimizer_error,
                      resume_rng_exact=True,resume_sampler_exact=True,bn_buffers_unchanged=True,
                      save_reload_logit_max_error=prediction_error,
                      group_parameter_max_changes=changes,training_seconds=time.monotonic()-train_start,
                      final_lrs={g['name']:g['lr'] for g in optimizer.param_groups},status='PASS')
        if export_recovery:
            stage='artifact_export'
            recovery_manifest={'schema':1,'purpose':'disposable external recovery qualification','status':'PASS',
                'binding':state_contract,'checkpoint_file':'recovery.pt','checkpoint_bytes':temporary[0].stat().st_size,
                'checkpoint_sha256':file_sha(temporary[0]),'updates_before':2,'updates_after':3,
                'consumed_before':0,'consumed_after':3,'expected':expected_recovery}
            validate_recovery_manifest(recovery_manifest,state_contract)
            temporary[0].replace(out/'recovery.pt')
            (out/'recovery-manifest.json').write_text(json.dumps(recovery_manifest,indent=2)+'\n')
            report.update(recovery_artifact_exported=True,recovery_manifest_sha256=sha256(canonical_bytes(recovery_manifest)),
                          recovery_checkpoint_sha256=recovery_manifest['checkpoint_sha256'],recovery_checkpoint_bytes=recovery_manifest['checkpoint_bytes'])
        save()
    except Exception as error:
        report.update(status='FAIL',failure_type=type(error).__name__,**failure_diagnostic(stage,error))
        save()
        raise RuntimeError('AdamW qualification failed; inspect sanitized aggregate receipt') from None
    finally:
        try:
            if active_stream is not None:close_stream()
        except Exception:
            if report['status']=='PASS':report.update(status='FAIL',failure_stage='stream_binding',failure_check='STREAM_CURSOR_MISMATCH')
            report['stream_cleanup_failed']=True
        report['measured_decode_gpu_overlap_seconds']=sum(max(0,min(b,d)-max(a,c)) for a,b,_ in tuple(decode_records) for c,d in gpu_intervals)
        report['decode_seconds_include_integrity_hash']=True
        for path in temporary: path.unlink(missing_ok=True)
        if report['status']!='PASS':
            (out/'recovery.pt').unlink(missing_ok=True);(out/'recovery-manifest.json').unlink(missing_ok=True)
        report.update(temporary_duplicates_deleted=all(not p.exists() for p in temporary),
                      disposable_states_deleted=all(not p.exists() for p in temporary) and not (out/'recovery.pt').exists(),
                      retained_private_recovery_checkpoint=(out/'recovery.pt').exists(),free_disk_after=shutil.disk_usage(out).free)
        save()
        if report.get('stream_cleanup_failed'):raise RuntimeError('Stream cleanup failed; inspect sanitized aggregate receipt') from None
    return report
