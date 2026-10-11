"""Fixed two-epoch public-state adaptation. Optional ML imports stay inside calls.

Private study identities are input/checkpoint data, never aggregate receipt fields.
No intermediate development scoring, outcome selection, or time-budget stop exists.
"""
import json
import os
from pathlib import Path
import re
import signal
import shutil
import time
from knee.reference import canonical_bytes, sha256
from knee.public_adaptation_pilot import (LABEL_SHA, MASK_SHA, file_sha, scientific_config,
    validate_recovery_manifest, load_labels, masked_loss, study_mean_backward,
    optimizer_groups, training_mode, bn_state, require_bn, apply_update,
    sampled_windows, checkpoint_forward, rng_state, restore_rng, state_fingerprint)

SCIENTIFIC_REVISION = '881faedd5ec8c52f8f259550adfd73377f33f438'
SCIENTIFIC_BUNDLE = '8453a365818c6a1f4eac904f56f778c615d6824acd29e88c3b414d24a3e0cb90'
INITIAL_SHA = '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
FOLD_SHA = '12bebc897a549d660cf5e6ab597c9a8d8f84dc887b4abd4c99b67fb4a25cb3ae'
EXCLUDED_SHA = '44834d7638eb60142a668a7597bd3093fe569e13b6f77e60a9c185a4ac3206f1'


def require(condition, check):
    if not condition:
        raise ValueError(check)


def fingerprint(value):
    return sha256(canonical_bytes(value))


def fit_config():
    config = scientific_config()
    for key in ('groups', 'next_tail', 'reordered_relative_budget'):
        del config[key]
    return {**config, 'epochs': 2, 'fold': 0, 'accumulation': 16,
            'contributors': 3455, 'updates_per_epoch': 216, 'updates': 432,
            'tail': 15, 'endpoint': 'FP32 mean of two epoch-end parameters',
            'evaluation': 'original binary y_/m_; K94 anatomical mirror',
            'bootstrap': {'draws': 2000, 'seed': 42, 'paired': True}}


def qualification_gate(stage1, stage2, manifest, contract, stream):
    """Synthetic checks do not replace parent-pinned actual CUDA receipts."""
    binding = {'cache': contract, 'stream': stream, 'scientific_config': scientific_config(),
               'source_revision': SCIENTIFIC_REVISION, 'bundle_sha256': SCIENTIFIC_BUNDLE}
    validate_recovery_manifest(manifest, binding)
    for report in (stage1, stage2):
        require(report.get('status') == 'PASS' and report.get('contract') == contract and
                report.get('bundle_sha256') == SCIENTIFIC_BUNDLE and
                report.get('source_revision') == SCIENTIFIC_REVISION and
                report.get('checkpoint_sha256') == INITIAL_SHA and
                report.get('labels_sha256') == LABEL_SHA and report.get('masks_sha256') == MASK_SHA and
                report.get('bn_buffers_unchanged') is True and report.get('updates') == 3 and
                report.get('exact_resume_status') == 'PASS' and
                report.get('scientific_config') == scientific_config() and
                report.get('qualified_cpu_stream') == stream and
                report.get('input_mode') == 'protected raw stream', 'QUALIFICATION_BINDING')
    require(stage1.get('actual_group_denominators') == [16, 14, 3] and
            stage1.get('independent_inline_gradients_exact') is True and
            stage1.get('stream_rng_unchanged_via_exact_cached_replay') is True and
            stage1.get('recovery_artifact_exported') is True and
            stage1.get('recovery_manifest_sha256') == fingerprint(manifest) and
            stage1.get('recovery_checkpoint_sha256') == manifest['checkpoint_sha256'] and
            stage1.get('recovery_checkpoint_bytes') == manifest['checkpoint_bytes'] and
            stage1.get('save_reload_logit_max_error') == 0 and
            stage1.get('resume_model_max_error') == 0 and
            stage1.get('resume_optimizer_max_error') == 0, 'STAGE1_NOT_QUALIFIED')
    require(stage2.get('external_artifact_recovery_status') == 'PASS' and
            stage2.get('external_artifact_bytes_verified') is True and
            stage2.get('external_expected_fingerprints') == manifest['expected'] and
            stage2.get('exact_preclip_gradients') is True and
            stage2.get('qualified_optimizer_updates') == 1 and
            stage2.get('actual_optimizer_step_calls') == 1 and
            stage2.get('external_next_study_forward_backward_passes') == 3 and
            stage2.get('save_reload_status') == 'PASS', 'STAGE2_NOT_QUALIFIED')
    return {'stage1': fingerprint(stage1), 'stage2': fingerprint(stage2),
            'recovery_manifest': fingerprint(manifest), 'scientific_binding': binding}


def epoch_order(contributors, epoch):
    import numpy as np
    require(epoch in (0, 1), 'EPOCH_RANGE')
    return np.random.default_rng(42 + epoch).permutation(contributors).tolist()


def validate_cursor(epoch, cursor, updates, orders, completed):
    require(type(epoch) is int and epoch in (0, 1, 2) and type(cursor) is int and
            type(updates) is int, 'RESUME_CURSOR')
    n = len(orders[0]); per_epoch = (n + 15) // 16
    require(len(orders) == 2 and len(orders[1]) == n and n > 0, 'RESUME_ORDER')
    if epoch == 2:
        require(cursor == 0 and updates == 2 * per_epoch and completed == 2, 'RESUME_CURSOR')
    else:
        require(0 <= cursor < n and cursor % 16 == 0 and
                updates == epoch * per_epoch + cursor // 16 and completed == epoch, 'RESUME_CURSOR')


def atomic_save(path, value, writer=None):
    """Replace a complete fsynced file; retain the previous boundary on failure."""
    if writer is None:
        import torch
        writer = torch.save
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('wb') as handle:
            writer(value, handle); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        tmp.unlink(missing_ok=True)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def average_endpoint(states, parameter_names, original_bn):
    import torch
    require(len(states) == 2 and states[0].keys() == states[1].keys(), 'ENDPOINT_STATES')
    names = set(parameter_names)
    require(names <= states[0].keys() and original_bn.keys() <= states[0].keys(), 'ENDPOINT_SCHEMA')
    result = {}
    for name, a in states[0].items():
        b = states[1][name]
        require(a.shape == b.shape and a.dtype == b.dtype, 'ENDPOINT_SCHEMA')
        if name in original_bn:
            require(torch.equal(a, original_bn[name]) and torch.equal(b, original_bn[name]), 'BN_DRIFT')
            result[name] = original_bn[name].clone()
        elif name in names:
            require(a.dtype == torch.float32 and torch.isfinite(a).all() and torch.isfinite(b).all(), 'ENDPOINT_PRECISION')
            result[name] = (a / 2 + b / 2).float()
        else:
            # There is no prospective policy for averaging arbitrary model buffers.
            require(torch.equal(a, b), 'BUFFER_DRIFT')
            result[name] = a.clone()
    return result


class PlatformInterruption:
    """Actual SIGTERM/SIGINT request: finish current optimizer group and save."""
    def __init__(self):
        self.requested = False
        self.signum = None
        self.previous = {}
    def notify(self, signum, frame=None):
        self.requested = True; self.signum = signum
    def __enter__(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            self.previous[signum] = signal.signal(signum, self.notify)
        return self
    def __exit__(self, *args):
        for signum, handler in self.previous.items(): signal.signal(signum, handler)


def train_epochs(model, optimizer, contributors, loss_for, binding, checkpoint, *,
                 stream_factory=None, interruption=None, resume=False, on_boundary=None, timings=None, gpu_intervals=None):
    """Shared fixed recipe engine; toy tests exercise external boundary recovery."""
    import torch
    timings = timings if timings is not None else {}
    for key in ('consumer_wait_seconds','forward_backward_seconds','checkpoint_seconds','update_seconds'):
        timings.setdefault(key,0.0)
    timings.setdefault('actual_forward_backward_passes',0)
    timings.setdefault('observed_queue_peak',0)
    cuda=next(model.parameters()).device.type=='cuda'
    orders = [epoch_order(contributors, e) for e in (0, 1)]
    original_bn = bn_state(model)
    epoch = cursor = updates = 0; epoch_states = []
    if resume:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        require(set(state) == {'schema','binding','orders','epoch','cursor','updates','model','optimizer',
                              'rng','epoch_states','original_bn','fingerprints'}, 'RESUME_SCHEMA')
        require(state['schema'] == 1 and state['binding'] == binding and state['orders'] == orders,
                'RESUME_BINDING')
        for key in ('model','optimizer','rng','epoch_states','original_bn'):
            require(state_fingerprint(state[key]) == state['fingerprints'][key], 'RESUME_PAYLOAD')
        require(state_fingerprint(original_bn) == state_fingerprint(state['original_bn']), 'RESUME_INITIAL_BN')
        epoch, cursor, updates = state['epoch'], state['cursor'], state['updates']
        epoch_states = state['epoch_states']
        validate_cursor(epoch, cursor, updates, orders, len(epoch_states))
        model.load_state_dict(state['model'], strict=True); optimizer.load_state_dict(state['optimizer'])
        require_bn(model, original_bn)
        for group in optimizer.param_groups:
            require(all((int(optimizer.state[p]['step']) == updates if updates else not optimizer.state[p]) for p in group['params']), 'RESUME_OPTIMIZER_COUNTER')
        restore_rng(state['rng'])
    def save():
        payload = {'schema':1, 'binding':binding, 'orders':orders, 'epoch':epoch, 'cursor':cursor,
                   'updates':updates, 'model':cpu_state(model), 'optimizer':optimizer.state_dict(),
                   'rng':rng_state(), 'epoch_states':epoch_states, 'original_bn':original_bn}
        payload['fingerprints'] = {k:state_fingerprint(payload[k]) for k in
                                  ('model','optimizer','rng','epoch_states','original_bn')}
        tick=time.monotonic(); atomic_save(checkpoint, payload)
        timings['checkpoint_seconds']+=time.monotonic()-tick
    # An initial boundary permits restart even if decoding fails before update one.
    if not resume: save()
    while epoch < 2:
        stream = stream_factory(orders[epoch], cursor) if stream_factory else None
        try:
            while cursor < len(contributors):
                chosen = orders[epoch][cursor:cursor+16]
                training_mode(model); optimizer.zero_grad(set_to_none=True)
                for sid in chosen:
                    tick=time.monotonic(); value = next(stream) if stream is not None else None
                    timings['consumer_wait_seconds']+=time.monotonic()-tick
                    if stream is not None:
                        with stream._condition: timings['observed_queue_peak']=max(timings['observed_queue_peak'],len(stream._buffer))
                    tick=time.monotonic()
                    study_mean_backward(loss_for(model, sid, epoch, value), len(chosen))
                    if cuda: torch.cuda.synchronize()
                    end=time.monotonic(); timings['forward_backward_seconds']+=end-tick
                    timings['actual_forward_backward_passes']+=1
                    if gpu_intervals is not None: gpu_intervals.append((tick,end))
                    if stream is not None: stream.acknowledge()
                tick=time.monotonic(); apply_update(model, optimizer, updates)
                if cuda: torch.cuda.synchronize()
                timings['update_seconds']+=time.monotonic()-tick
                updates += 1; cursor += len(chosen)
                require_bn(model, original_bn)
                if stream is not None: require(stream.consumed_cursor == cursor, 'STREAM_CURSOR')
                if cursor == len(contributors):
                    epoch_states.append(cpu_state(model)); epoch += 1; cursor = 0
                validate_cursor(epoch, cursor, updates, orders, len(epoch_states))
                save()
                if on_boundary: on_boundary(epoch, cursor, updates)
                if interruption is not None and interruption.requested:
                    return {'status':'INTERRUPTED', 'epoch':epoch,'cursor':cursor,'updates':updates}
                if cursor == 0: break
        finally:
            if stream is not None: stream.close()
    return {'status':'COMPLETE', 'updates':updates,
            'endpoint':average_endpoint(epoch_states, dict(model.named_parameters()), original_bn)}


def recover_completed_endpoint(model, contributors, binding, checkpoint, endpoint_path):
    """Verify a finished training artifact and its fixed endpoint without optimization."""
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(set(state) == {'schema','binding','orders','epoch','cursor','updates','model','optimizer',
                          'rng','epoch_states','original_bn','fingerprints'}, 'RESUME_SCHEMA')
    orders = [epoch_order(contributors, e) for e in (0, 1)]
    require(state['schema'] == 1 and state['binding'] == binding and state['orders'] == orders,
            'RESUME_BINDING')
    for key in ('model','optimizer','rng','epoch_states','original_bn'):
        require(state_fingerprint(state[key]) == state['fingerprints'][key], 'RESUME_PAYLOAD')
    validate_cursor(state['epoch'],state['cursor'],state['updates'],orders,len(state['epoch_states']))
    require(state['epoch'] == 2 and state['cursor'] == 0, 'RECOVERY_NOT_COMPLETED')
    require(state_fingerprint(bn_state(model)) == state_fingerprint(state['original_bn']), 'RESUME_INITIAL_BN')
    require(len(state['optimizer']['state']) == len(list(model.parameters())) and
            all(int(value['step']) == state['updates'] for value in state['optimizer']['state'].values()),
            'RESUME_OPTIMIZER_COUNTER')
    model.load_state_dict(state['model'],strict=True)
    require_bn(model,state['original_bn'])
    require(state_fingerprint(state['model']) == state_fingerprint(state['epoch_states'][-1]),
            'RECOVERY_FINAL_MODEL')
    expected = average_endpoint(state['epoch_states'],dict(model.named_parameters()),state['original_bn'])
    attached = torch.load(endpoint_path,map_location='cpu',weights_only=True)
    require(state_fingerprint(expected) == state_fingerprint(attached), 'RECOVERY_ENDPOINT')
    require(all(not value.is_floating_point() or torch.isfinite(value).all() for value in attached.values()),
            'RECOVERY_ENDPOINT_FINITE')
    model.load_state_dict(attached,strict=True)
    return {'status':'COMPLETE','updates':state['updates'],'endpoint':attached}


TRUTH_POLICY = 'original-observed-y-float32-ge-0.5-mask-unchanged-v1'

def load_binary_truth(path, ids, targets):
    import numpy as np
    import pyarrow.parquet as pq
    require(file_sha(path) == MASK_SHA, 'TRUTH_BYTES')
    norm = lambda s: re.sub('[^a-z0-9]', '', s.lower())
    schema = pq.read_schema(path).names
    columns = []
    for prefix in ('y_', 'm_'):
        mapping = {norm(c[2:]):c for c in schema if c.startswith(prefix)}
        selected = [mapping.get(norm(t)) for t in targets]
        require(None not in selected and len(set(selected)) == len(targets), 'TRUTH_SCHEMA')
        columns += selected
    table = pq.read_table(path, columns=['StudyInstanceUID',*columns]).to_pydict()
    keys = table['StudyInstanceUID']; pos = {str(s):i for i,s in enumerate(keys)}
    require(len(pos) == len(keys) and set(ids) <= pos.keys(), 'TRUTH_COVERAGE')
    values = np.asarray([[table[c][pos[s]] for c in columns] for s in ids], dtype=np.float64)
    y, m = np.split(values, 2, axis=1)
    require(np.isfinite(m).all() and np.isin(m,[0,1]).all() and
            np.isfinite(y[m>0]).all() and ((y[m>0]>=0)&(y[m>0]<=1)).all(), 'TRUTH_BINARY')
    # Historical original-label evaluator: float32 conversion followed by >=0.5.
    # Unknown cells are masked before conversion; they cannot become negatives in support.
    binary=(np.where(m>0,y,0).astype(np.float32)>=np.float32(.5)).astype(np.float64)
    return np.where(m>0,binary,0), m


def metric_report(y, masks, probabilities, targets):
    import numpy as np
    from knee.validation import _auc
    y, masks, p = map(np.asarray, (y,masks,probabilities))
    require(y.shape == masks.shape == p.shape and y.ndim == 2 and y.shape[1] == len(targets) and
            np.isin(masks,[0,1]).all() and np.isin(y[masks>0],[0,1]).all() and
            np.isfinite(p).all() and ((p>=0)&(p<=1)).all(), 'METRIC_INPUT')
    result = {}; defined=[]
    for j,t in enumerate(targets):
        known=masks[:,j]>0; positives=int(y[known,j].sum()); negatives=int(known.sum())-positives
        auc=_auc(list(zip(y[known,j].tolist(),p[known,j].tolist()))) if positives and negatives else None
        result[t]={'auc':auc,'positive':positives,'negative':negatives,'observed':int(known.sum()),
                   'missing':int((~known).sum()),'undefined':auc is None}
        if auc is not None: defined.append(auc)
    return {'targets':result,'macro_auc':sum(defined)/len(defined) if defined else None,
            'macro_policy':'unweighted arithmetic mean over targets with both observed classes',
            'defined_targets':len(defined)}


def paired_bootstrap(y, masks, control, endpoint, groups, *, draws=2000, seed=42, targets=None):
    """Whole-component multiplicities, paired predictions, fixed original defined set."""
    import numpy as np
    y,m,c,e=map(np.asarray,(y,masks,control,endpoint)); groups=np.asarray(groups)
    require(y.shape==m.shape==c.shape==e.shape and len(groups)==len(y) and len(y)>0,
            'BOOTSTRAP_COVERAGE')
    names=list(targets) if targets is not None else [str(j) for j in range(y.shape[1])]
    require(len(names)==y.shape[1] and len(set(names))==len(names),'BOOTSTRAP_TARGETS')
    base=metric_report(y,m,c,names); metric_report(y,m,e,names)
    defined=[j for j in range(y.shape[1]) if not base['targets'][names[j]]['undefined']]
    require(defined and type(draws) is int and draws>0, 'BOOTSTRAP_UNDEFINED')
    unique,inverse=np.unique(groups,return_inverse=True)
    require(all(str(g).strip() for g in unique), 'BOOTSTRAP_GROUP')
    cluster=np.random.default_rng(seed).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=draws)
    counts=cluster[:,inverse]
    def weighted_auc(p,j):
        known=np.flatnonzero(m[:,j]>0); order=known[np.argsort(p[known,j],kind='stable')]
        starts=np.r_[0,np.flatnonzero(np.diff(p[order,j])!=0)+1]
        positive=y[order,j]==1
        pos=np.add.reduceat(counts[:,order]*positive,starts,axis=1)
        neg=np.add.reduceat(counts[:,order]*~positive,starts,axis=1)
        denom=pos.sum(1)*neg.sum(1)
        with np.errstate(invalid='ignore',divide='ignore'):
            return (pos*(np.cumsum(neg,axis=1)-neg+neg/2)).sum(1)/denom
    differences=np.stack([weighted_auc(e,j)-weighted_auc(c,j) for j in defined],axis=1)
    valid=np.isfinite(differences).all(1)
    delta=differences[valid].mean(1)
    invalid_causes={names[j]:int((~np.isfinite(differences[:,k])).sum()) for k,j in enumerate(defined)}
    return {'draws':draws,'valid_draws':int(valid.sum()),'undefined_draws':int((~valid).sum()),
            'seed':seed,'clusters':len(unique),'cluster_size_min':int(np.bincount(inverse).min()),
            'cluster_size_max':int(np.bincount(inverse).max()),
            'ci95':np.quantile(delta,[.025,.975]).tolist() if len(delta) else None,
            'invalid_target_causes':invalid_causes, 'original_defined_target_indices':defined,'original_defined_targets':[names[j] for j in defined],
            'uncertainty_status':'no finite CI' if not len(delta) else 'limited support' if valid.sum()<.9*draws else 'finite CI',
            'policy':'paired whole-cluster resampling; require all originally defined targets per retained draw'}


def report_components(rows, train, dev):
    from knee.validation import _report_key, _identity_columns
    indexed={r['StudyInstanceUID']:r for r in rows}
    require(len(indexed)==len(rows) and set(train+dev)<=indexed.keys(), 'GROUP_COVERAGE')
    patient_columns=_identity_columns(rows[0].keys()); require(len(patient_columns)<=1,'GROUP_IDENTITY')
    parent={s:s for s in indexed}; seen={}
    def find(s):
        while parent[s]!=s: parent[s]=parent[parent[s]]; s=parent[s]
        return s
    for sid,row in indexed.items():
        values=[('report',_report_key(row.get('Report','')))]
        if patient_columns: values.append(('patient',row.get(patient_columns[0],'').strip()))
        for kind,value in values:
            if value:
                key=(kind,value)
                if key in seen: parent[find(sid)]=find(seen[key])
                else: seen[key]=sid
    require(not ({find(s) for s in train}&{find(s) for s in dev}), 'GROUP_SPLIT_LEAKAGE')
    groups=[fingerprint(['private-component',find(s)]) for s in dev]
    return groups, {'grouping':'patient/report components' if patient_columns else 'exact normalized-report components',
                    'missing_patient_identity':sum(not indexed[s].get(patient_columns[0],'').strip() for s in train+dev) if patient_columns else len(train)+len(dev),
                    'limitations':'Near-duplicate reports and repeat patients without available identity are not detected; public checkpoint training exposure makes this development evidence.'}


def preflight_inputs(root, label_path, mask_path, folds, excluded, preflight, targets):
    import csv
    import numpy as np
    import pyarrow.parquet as pq
    require(fingerprint(folds)==preflight['fingerprints']['fold_mapping'], 'FOLD_MAPPING')
    with (Path(root)/'train.csv').open(newline='') as handle: rows=list(csv.DictReader(handle))
    ids=[r['StudyInstanceUID'] for r in rows]; require(len(set(ids))==len(ids),'OFFICIAL_DUPLICATES')
    gold={r['StudyInstanceUID'] for r in rows if all(r[t].strip() for t in targets)}
    label_ids=set(pq.read_table(label_path,columns=['StudyInstanceUID']).column(0).to_pylist())
    eligible=sorted(set(ids)&label_ids-gold-set(excluded))
    require(set(eligible)<=folds.keys(),'FOLD_COVERAGE')
    train=[s for s in eligible if folds[s]!=0]; dev=[s for s in eligible if folds[s]==0]
    ys,ms=load_labels(label_path,mask_path,eligible,targets)
    contributors=[s for s in train if ms[s].any()]; skipped=[s for s in train if not ms[s].any()]
    actual={k:fingerprint(v) for k,v in [('eligible',eligible),('train',train),('dev',dev),('contributors',contributors),('skipped',skipped)]}
    actual.update({f'epoch{e}_order':fingerprint(epoch_order(contributors,e)) for e in (0,1)})
    actual['fold_mapping']=fingerprint(folds)
    actual['selected_masks_in_target_order']=fingerprint([[s,ms[s].tolist()] for s in eligible])
    require(actual==preflight['fingerprints'] and (len(eligible),len(train),len(dev),len(contributors),len(skipped))==(4345,3475,870,3455,20), 'PREFLIGHT_MEMBERSHIP')
    require(targets==preflight['targets'] and preflight['status']=='AGGREGATE_PREFLIGHT_PASS' and
            all(preflight['checks'].values()),'PREFLIGHT_CONTRACT')
    for j,t in enumerate(targets):
        require(preflight['support'][t]=={'train_observed':int(sum(ms[s][j] for s in train)),
            'contributor_observed':int(sum(ms[s][j] for s in contributors)),
            'dev_observed':int(sum(ms[s][j] for s in dev))},'PREFLIGHT_SUPPORT')
    groups,grouping=report_components(rows,train,dev)
    return train,dev,contributors,ys,ms,groups,grouping


def run(public_source, decoder_source, manifest, checkpoint, labels, masks, root, out,
        folds, excluded, preflight, gate, binding, *, resume=None, evaluation_recovery=None):
    """Remote CUDA entry point; failed receipts never include exception text/IDs."""
    out=Path(out); out.mkdir(parents=True,exist_ok=resume is not None)
    started=time.monotonic(); stage='environment'; decode_records=[]; gpu_intervals=[]
    report={'status':'RUNNING','config':fit_config(),'binding':binding,'updates':0,
            'generalization_evidence':False,'free_disk_before':shutil.disk_usage(out).free,
            'timings':{'consumer_wait_seconds':0.0,'forward_backward_seconds':0.0}}
    def receipt():
        from knee.public_adaptation_pilot import decode_metrics
        available=shutil.disk_usage(out).free
        report.update(free_disk_now=available,
                      minimum_free_disk_bytes=min(report.get('minimum_free_disk_bytes',available),available))
        report.update(elapsed_seconds=time.monotonic()-started,**decode_metrics(decode_records),
            measured_decode_gpu_overlap_seconds=sum(max(0,min(b,d)-max(a,c)) for a,b,_ in tuple(decode_records) for c,d in gpu_intervals),
            timing_policy='decoder and CUDA intervals overlap; durations are not additive')
        atomic_save(out/'receipt.json',report,lambda value,h:h.write((json.dumps(value,indent=2,allow_nan=False)+'\n').encode()))
    receipt()
    try:
        import csv
        import importlib.metadata
        import numpy as np
        import resource
        import torch
        from knee.reference import load_definitions, load_model
        from knee.mirror_input import prepare_study, policy_contract
        from knee.public_study_stream import StudyStream
        report['environment']={'python':__import__('sys').version,'torch':torch.__version__,
                               'timm':importlib.metadata.version('timm'),'numpy':np.__version__,
                               'pyarrow':importlib.metadata.version('pyarrow'),'cuda':torch.version.cuda}
        require(__import__('sys').version_info[:2]==(3,12) and torch.cuda.is_available() and
                torch.__version__=='2.10.0+cu128' and report['environment']['timm']=='1.0.26','CUDA_ENVIRONMENT')
        report['environment']['device']=torch.cuda.get_device_name(0)
        stage='qualification'
        require(qualification_gate(gate['stage1'],gate['stage2'],gate['manifest'],gate['contract'],gate['stream'])==binding['qualification'],'QUALIFICATION_GATE')
        require(binding['config']==fit_config() and binding['preflight']==fingerprint(preflight) and
                (evaluation_recovery['operational_bundle_sha256'] if evaluation_recovery else binding['source_bundle_sha256'])==os.environ.get('KNEE_SOURCE_SHA256') and
                binding['public_manifest']==fingerprint(manifest) and manifest['checkpoint_sha256']==INITIAL_SHA and
                binding['input_policy']==policy_contract() and binding['excluded_content_sha256']==fingerprint(excluded),'FIT_BINDING')
        require(resume is None or Path(resume)==out/'resume.pt','RESUME_LOCATION')
        stage='inputs'; root=Path(root)
        require(file_sha(root/'train.csv')==binding['official_train_sha256'],'OFFICIAL_BYTES')
        series_sha=preflight['source_sha256']['data/raw/train_series.csv']
        require(file_sha(root/'train_series.csv')==series_sha,'SERIES_BYTES')
        public,decoder=load_definitions(public_source,decoder_source,manifest)
        train,dev,contributors,ys,ms,groups,grouping=preflight_inputs(root,labels,masks,folds,excluded,preflight,public.LAB)
        report.update(population=preflight['population'],support=preflight['support'],grouping=grouping)
        records={}
        with (root/'train_series.csv').open(newline='') as handle:
            for row in csv.DictReader(handle): records.setdefault(row['StudyInstanceUID'],[]).append(row)
        require(all(records.get(s) and (root/'train_series'/s).is_dir() for s in train+dev),'IMAGE_COVERAGE')
        stage='initial_state'; __import__('random').seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.set_num_threads(4); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        model=load_model(public,checkpoint,'cuda',manifest)
        require(all(p.dtype==torch.float32 for p in model.parameters()) and not torch.is_autocast_enabled(),'FP32_REQUIRED')
        optimizer=None if evaluation_recovery else torch.optim.AdamW(optimizer_groups(model),betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
        torch.cuda.reset_peak_memory_stats()
        def read(sid):
            tick=time.monotonic()
            try:
                volume,mask,counts=prepare_study(public,decoder,sid,records,root/'train_series')
            except BaseException as error:
                counts=getattr(error,'mirror_input_counts',{})
                decode_records.append((tick,time.monotonic(),tuple(sorted((k,v) for k,v in counts.items() if isinstance(v,int)))))
                raise
            decode_records.append((tick,time.monotonic(),tuple(sorted(counts.items()))))
            require(counts.get('decoded_unique_slices',0)>0 and not any(counts.get(k,0) for k in
                ('resource_errors','invalid_pixels','pixel_failures','fast_bad_headers','header_parse_failures')),'DECODE_FAILURE')
            require(volume.shape==(96,384,384) and volume.dtype==np.uint8 and mask.shape==(96,) and
                    np.isin(mask,[0,1]).all() and mask.any(),'INPUT_SCHEMA')
            return volume,mask
        def stream_factory(order,cursor):
            return StudyStream(order,read,contract=fingerprint(binding),consumed_cursor=cursor,prefetch=4,close_timeout=30)
        def loss_for(current,sid,epoch,value):
            windows=sampled_windows(public,*value,sid,epoch)
            logits=checkpoint_forward(current,windows)
            loss=masked_loss(logits,torch.from_numpy(ys[sid])[None].to('cuda'),torch.from_numpy(ms[sid])[None].to('cuda'))
            return loss
        def boundary(epoch,cursor,updates):
            torch.cuda.synchronize(); report.update(epoch=epoch,consumed_cursor=cursor,updates=updates,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                checkpoint_sha256=file_sha(out/'resume.pt'),checkpoint_bytes=(out/'resume.pt').stat().st_size)
            receipt()
        stage='training'
        train_started=time.monotonic()
        if evaluation_recovery:
            stage='evaluation_recovery'
            prior=evaluation_recovery['prior']
            require(resume is not None and prior['binding']==binding and prior['status']=='FAIL' and
                    prior['failure_stage']=='evaluation' and prior['updates']==432 and
                    prior['epoch']==2 and prior['consumed_cursor']==0, 'RECOVERY_RECEIPT')
            require(file_sha(resume)==prior['checkpoint_sha256'] and
                    Path(resume).stat().st_size==prior['checkpoint_bytes'] and
                    (evaluation_recovery['endpoint_sha256'] is None or
                     file_sha(out/'endpoint.pt')==evaluation_recovery['endpoint_sha256']), 'RECOVERY_BYTES')
            result=recover_completed_endpoint(model,contributors,binding,resume,out/'endpoint.pt')
            report.update(evaluation_recovery=evaluation_recovery,epoch=2,consumed_cursor=0,
                          optimizer_updates_this_run=0,training_seconds=prior['training_seconds'],
                          endpoint_sha256=file_sha(out/'endpoint.pt'),endpoint_provenance='exact all-tensor fixed average oracle from SHA-pinned final resume')
        else:
            with PlatformInterruption() as stop:
                result=train_epochs(model,optimizer,contributors,loss_for,binding,out/'resume.pt',
                    stream_factory=stream_factory,interruption=stop,resume=resume is not None,on_boundary=boundary,
                    timings=report['timings'],gpu_intervals=gpu_intervals)
            report['training_seconds']=time.monotonic()-train_started
        report['updates']=result['updates']
        if result['status']=='INTERRUPTED':
            report.update(status='INTERRUPTED',checkpoint_sha256=file_sha(out/'resume.pt'),
                          checkpoint_bytes=(out/'resume.pt').stat().st_size); receipt(); return report
        require(result['updates']==432,'UPDATE_TOTAL')
        endpoint=result['endpoint']
        if not evaluation_recovery: atomic_save(out/'endpoint.pt',endpoint)
        # Strict reload and exact equality before any endpoint/control metric.
        loaded=torch.load(out/'endpoint.pt',map_location='cpu',weights_only=True)
        require(state_fingerprint(loaded)==state_fingerprint(endpoint),'ENDPOINT_RELOAD')
        model.load_state_dict(loaded,strict=True); model.eval(); del optimizer,result,loaded
        stage='evaluation'; evaluation_started=time.monotonic()
        # Fail closed before decoding/inference when the original truth contract is invalid.
        report['truth_policy']=TRUTH_POLICY
        y,m=load_binary_truth(masks,dev,public.LAB)
        train_y,train_m=load_binary_truth(masks,train,public.LAB)
        endpoint_predictions=[]; probe=None; endpoint_probe=None
        with StudyStream(dev,read,contract=fingerprint(binding),prefetch=4,close_timeout=30) as stream:
            for sid in dev:
                volume,mask=next(stream)
                windows=[public.gpu_windows(volume,mask,94,320,'cuda',mirror=m) for m in (False,True)]
                p=np.asarray(public.infer_pair_probs(model,*windows,'cuda'))
                require(p.shape==(12,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'PREDICTION_SCHEMA')
                endpoint_predictions.append(p)
                if probe is None: probe=windows; endpoint_probe=p.copy()
                stream.acknowledge()
        # A strict second reload must reproduce an actual image-only probability.
        model.load_state_dict(torch.load(out/'endpoint.pt',map_location='cpu',weights_only=True),strict=True); model.eval()
        require(np.array_equal(endpoint_probe,np.asarray(public.infer_pair_probs(model,*probe,'cuda'))),'ENDPOINT_PREDICTION_RELOAD')
        del model,endpoint,probe; torch.cuda.empty_cache()
        control=load_model(public,checkpoint,'cuda',manifest); control.eval(); control_predictions=[]
        with StudyStream(dev,read,contract=fingerprint(binding),prefetch=4,close_timeout=30) as stream:
            for sid in dev:
                volume,mask=next(stream)
                windows=[public.gpu_windows(volume,mask,94,320,'cuda',mirror=m) for m in (False,True)]
                control_predictions.append(public.infer_pair_probs(control,*windows,'cuda')); stream.acknowledge()
        prevalence=np.divide((train_y*train_m).sum(0),train_m.sum(0),out=np.full(12,np.nan),where=train_m.sum(0)>0)
        require(np.isfinite(prevalence).all(),'PREVALENCE_UNDEFINED')
        c,e=np.asarray(control_predictions),np.asarray(endpoint_predictions)
        atomic_save(out/'predictions-private.npz',None,lambda _,h:np.savez(h,control=c,endpoint=e))  # Private order bound by folds/preflight.
        prevalence_fit={t:{'probability':float(prevalence[j]),'positive':int((train_y[:,j]*train_m[:,j]).sum()),
            'negative':int(((1-train_y[:,j])*train_m[:,j]).sum()),'observed':int(train_m[:,j].sum()),
            'missing':int(len(train)-train_m[:,j].sum())} for j,t in enumerate(public.LAB)}
        metrics={name:metric_report(y,m,p,public.LAB) for name,p in
                 [('control',c),('endpoint',e),('train_prevalence',np.broadcast_to(prevalence,y.shape))]}
        bootstrap=paired_bootstrap(y,m,c,e,groups,targets=public.LAB)
        delta=metrics['endpoint']['macro_auc']-metrics['control']['macro_auc']
        damage=[t for t in public.LAB if metrics['control']['targets'][t]['auc'] is not None and
                metrics['endpoint']['targets'][t]['auc'] is not None and
                metrics['endpoint']['targets'][t]['positive']>=20 and metrics['endpoint']['targets'][t]['negative']>=20 and
                metrics['endpoint']['targets'][t]['auc']-metrics['control']['targets'][t]['auc']<=-.01]
        triage='reject' if (bootstrap['ci95'] is not None and bootstrap['ci95'][1]<0) or len(damage)>=3 else 'unpromising; do not expand' if delta<=0 else 'inconclusive; no expansion' if bootstrap['ci95'] is None else 'eligible for separately rehearsed candidate consideration'
        report.update(status='PASS',evaluation_seconds=time.monotonic()-evaluation_started,metrics=metrics,train_prevalence_fit=prevalence_fit,paired_bootstrap=bootstrap,macro_delta=delta,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                      target_damage=damage,triage=triage,endpoint_sha256=file_sha(out/'endpoint.pt'),
                      resume_sha256=file_sha(out/'resume.pt'),endpoint_tensor_sha256=state_fingerprint(torch.load(out/'endpoint.pt',map_location='cpu',weights_only=True)),
                      evaluated_studies=len(dev),inference_contract='exact inherited K94 anatomical-mirror image-only',
                      intermediate_metrics_opened=False,checkpoint_selection=False)
    except BaseException as error:
        report.update(status='FAIL',failure_stage=stage,failure_type=type(error).__name__)
        if isinstance(error,ValueError) and str(error).isupper() and re.fullmatch('[A-Z_]+',str(error)):
            report['failure_check']=str(error)
        if (out/'resume.pt').is_file():
            report.update(checkpoint_sha256=file_sha(out/'resume.pt'),checkpoint_bytes=(out/'resume.pt').stat().st_size)
        receipt()
        raise RuntimeError('Fixed adaptation failed; inspect sanitized receipt; preserve last boundary') from None
    receipt(); return report
