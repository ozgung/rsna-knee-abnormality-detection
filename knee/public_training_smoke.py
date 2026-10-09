"""Gold-free adapter around hash-pinned public Raptor training definitions.

This module imports no ML dependencies until used. The public main is never executed.
"""
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
import resource
from pathlib import Path
import random
import re
import sys
import time

VERSION = 'public-raptor-training-smoke-v1'
SOURCE_SHA = 'd39966791ef1c48d9317656e8346c60cae1c064192c87b3f350e298ec4c567e8'
MANIFEST_SHA = '21c71745703d742585684298494a6269ae4217fadc5fc4c26f3bf0d10a289467'
LABEL_SHA = 'd2cdb8a37e0c4c491758d17c96e3bd4081f06b257c7849608df88590368e94e2'
IDENTITY_KERNEL = 'ozgung/knee-sealed-identity-full-20261008'
ASSETS = 'nartaa/rsna-knee-hpo-assets'
CORPORA = ('dreaddevelopment/knee-raptor-corpus', 'dreaddevelopment/knee-raptor-corpus-ext')
PINS = {'all_ids.npy': 'ce332c0a673abde31856db1b3b86fba9e0a2f73d38fff23c7b02c51b869f3c30',
        'all_masks.npy': '1e16d5f93fe0b779ddf5bd3aa5beba01ca216373a633f517f74e3b5e94cb8fac',
        'extra_ids.npy': '65a32fe80ad4cffd20b847b3c36bda7b0c33178fff652d8828a4b0d4e1ccf9bb',
        'extra_masks.npy': 'e1650f959055e12332fbfe5d4a035c8b95c3a22a1ec556ac324e968984dc6c40'}
TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA',
           'PF OA', 'Effusion', 'Synovitis', "Baker's", 'Contusion', 'Fracture']
DEFINITIONS = ('_TwoPartVols', 'StudyWindows', 'collate', 'build_backbone', 'RaptorClassifier')
SETTINGS = {'seed': 42, 'studies': 40, 'epochs': 2, 'batch_size': 8, 'accum': 1,
            'arch': 'convnext_tiny.fb_in22k_ft_in1k', 'resolution': 256, 'k': 8, 'k_eval': 16,
            'norm': 'imagenet', 'roi': False, 'workers': 0, 'win_chunk': 0,
            'backbone_lr': 3e-5, 'head_lr': 1e-3, 'weight_decay': .02,
            'clip_norm': 3., 'pct_start': .15, 'precision': 'float16',
            'objective': 'public_dense_positive_weight_BCE', 'selection': 'none',
            'scheduler_steps': 10, 'notebook_seconds': 43200}
PACKAGES = {'torch': '2.10.0+cu128', 'torchvision': '0.25.0+cu128', 'timm': '1.0.26',
            'numpy': '2.0.2', 'pandas': '2.3.3'}
SALT = 'public-raptor-smoke40-20261009-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require_file(path, expected):
    if file_sha(path) != expected:
        raise ValueError('Pinned input bytes differ')


def cohort_records(records):
    """Project identity roles only; reject conflicts before selecting forty IDs."""
    if len(records) != 4407:
        raise ValueError('Identity coverage differs')
    role_counts = Counter(r['future_role'] for r in records)
    if role_counts != {'training_candidate': 3745, 'development': 300, 'sealed': 300, 'gold': 58, 'quarantined': 4}:
        raise ValueError('Frozen role counts differ')
    ids = [r['uid'] for r in records]
    if len(set(ids)) != len(ids) or any(not isinstance(u, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', u) for u in ids):
        raise ValueError('Invalid identity coverage')
    groups = defaultdict(set)
    original = []
    for r in records:
        if not isinstance(r.get('component'), str) or not r['component']:
            raise ValueError('Component identity missing')
        groups[r['component']].add(r['future_role'])
        if r['original_split'] == 'train':
            if r['future_role'] != 'training_candidate':
                raise ValueError('Original training includes excluded role')
            original.append(r['uid'])
        if r['original_split'] == 'weak_validation' and r['future_role'] != 'development':
            raise ValueError('Original development role changed')
    # Gold/related component is intentionally quarantined as a unit from fitting.
    for roles in groups.values():
        if len(roles) > 1 and roles != {'gold', 'quarantined'}:
            raise ValueError('Frozen component crosses permitted/reserved roles')
    if len(original) != 2000 or sum(r['original_split'] == 'weak_validation' for r in records) != 300:
        raise ValueError('Original train/development membership differs')
    original.sort(key=lambda u: (digest(f'{SALT}:{u}'.encode()), u))
    selected = original[:SETTINGS['studies']]
    return selected, {'manifest_sha256': MANIFEST_SHA, 'salt': SALT, 'studies': len(selected),
        'original_training_studies': len(original), 'role_counts': dict(role_counts),
        'ordered_selected_ids_sha256': digest(canonical(selected)),
        'original_training_ids_sha256': digest(canonical(sorted(original)))}


def membership(path):
    require_file(path, MANIFEST_SHA)
    document = json.loads(Path(path).read_bytes())
    return cohort_records(document['records'])


def public_definitions(source, *, corpus=None):
    """Execute only the five reviewed definitions, never imports or public main."""
    if digest(source.encode()) != SOURCE_SHA:
        raise ValueError('Public training source changed')
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    import timm
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in DEFINITIONS]
    if [n.name for n in nodes] != list(DEFINITIONS):
        raise ValueError('Public definition inventory differs')
    ns = {'np': np, 'torch': torch, 'nn': nn, 'F': F, 'timm': timm, 'random': random,
          'Dataset': torch.utils.data.Dataset, '_open_corpus': lambda: corpus}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned-public-training>', 'exec'), ns)
    return ns


def selected_labels(path, selected):
    require_file(path, LABEL_SHA)
    import numpy as np
    import pandas as pd
    identities = pd.read_parquet(path, columns=['StudyInstanceUID'])
    if len(identities) != 4349 or identities.StudyInstanceUID.isna().any() or identities.StudyInstanceUID.astype(str).duplicated().any():
        raise ValueError('Public nongold label coverage differs')
    table = pd.read_parquet(path, columns=['StudyInstanceUID', *TARGETS],
                            filters=[('StudyInstanceUID', 'in', selected)])
    table['StudyInstanceUID'] = table.StudyInstanceUID.astype(str)
    if len(table) != len(selected):
        raise ValueError('Selected target-row coverage differs')
    table = table.set_index('StudyInstanceUID')
    if not set(selected) <= set(table.index):
        raise ValueError('Selected training labels missing')
    values = table.loc[selected, TARGETS].to_numpy(dtype=np.float32)
    if values.shape != (len(selected), 12) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('Invalid selected soft probabilities')
    return {u: values[i] for i, u in enumerate(selected)}, values


def group_name(name):
    if name.startswith('backbone.'):
        return 'encoder'
    return 'head_bias' if name.endswith('.bias') or name == 'clsb' else 'nonbias_head'


def gradient_stats(model):
    import torch
    result = {}
    for group in ('encoder', 'nonbias_head', 'head_bias'):
        grads = [p.grad.detach() for n, p in model.named_parameters() if group_name(n) == group and p.grad is not None]
        finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads]))) if finite else None
        result[group] = {'tensors': len(grads), 'finite': finite, 'norm': norm}
    return result


def update(model, optimizer, scaler, scheduler, loss):
    """One microbatch/step. AMP overflow changes scaler only, never scheduler."""
    import torch
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient = gradient_stats(model)
    if any(v['tensors'] == 0 for v in gradient.values()):
        raise RuntimeError('A trainable parameter group received no gradient')
    finite = all(v['finite'] for v in gradient.values())
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['clip_norm'])) if finite else None
    if norm is not None and not math.isfinite(norm):
        raise FloatingPointError('Nonfinite aggregate gradient norm')
    if not finite and not scaler.is_enabled():
        raise FloatingPointError('Nonfinite unscaled gradient')
    before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    accepted = float(scaler.get_scale()) >= before
    if accepted and not finite:
        raise FloatingPointError('Nonfinite gradient was accepted')
    if accepted and scheduler.last_epoch + 1 < SETTINGS['scheduler_steps']:
        scheduler.step()  # Same finite-update schedule boundary as published code.
    optimizer.zero_grad(set_to_none=True)
    return {'accepted': accepted, 'gradient': gradient, 'preclip_norm': norm,
            'clipped': bool(norm is not None and norm > SETTINGS['clip_norm']),
            'scale_before': before, 'scale_after': float(scaler.get_scale())}


def tensor_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(canonical([name, list(tensor.shape), str(tensor.dtype)]))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def movement(model, initial):
    result = dict.fromkeys(('encoder', 'nonbias_head', 'head_bias'), 0.)
    for name, p in model.named_parameters():
        delta = float((p.detach().cpu() - initial[name]).abs().max())
        result[group_name(name)] = max(result[group_name(name)], delta)
    return result


def metrics(probabilities, values, weights):
    import numpy as np
    raw = probabilities.astype(np.float64)
    y = values.astype(np.float64)
    if raw.shape != y.shape or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise FloatingPointError('Invalid training predictions')
    p = np.clip(raw, 1e-7, 1-1e-7)
    bce = -(y*np.log(p)+(1-y)*np.log1p(-p)).mean(0)
    weighted = -(weights*y*np.log(p)+(1-y)*np.log1p(-p)).mean(0)
    from sklearn.metrics import roc_auc_score
    per = {}
    aucs = []
    for j, label in enumerate(TARGETS):
        hard = (y[:, j] >= .5).astype(int)
        auc = float(roc_auc_score(hard, p[:, j])) if len(set(hard)) == 2 else None
        if auc is not None:
            aucs.append(auc)
        per[label] = {'soft_mean': float(y[:,j].mean()), 'positive_at_half': int(hard.sum()),
            'negative_at_half': int(len(hard)-hard.sum()), 'bce': float(bce[j]),
            'weighted_bce': float(weighted[j]), 'auc_vs_thresholded_soft': auc}
    return {'macro_bce': float(bce.mean()), 'macro_weighted_bce': float(weighted.mean()),
        'mean_prediction_entropy': float(-(p*np.log(p)+(1-p)*np.log1p(-p)).mean()),
        'macro_auc_defined_targets': float(np.mean(aucs)) if aucs else None,
        'defined_auc_targets': len(aucs), 'auc_policy': 'mean of defined train-only thresholded-soft AUCs; never validation',
        'per_target': per}


def require_contract(contract):
    if (contract.get('version') != VERSION or contract.get('settings') != SETTINGS or
            contract.get('source_sha256') != SOURCE_SHA or contract.get('manifest_sha256') != MANIFEST_SHA or
            contract.get('label_sha256') != LABEL_SHA or contract.get('corpus_pins') != PINS or
            contract.get('packages') != PACKAGES or contract.get('internet') is not True or
            contract.get('identity_kernel') != IDENTITY_KERNEL or contract.get('gold_evaluation') is not False or
            contract.get('validation_selection') is not False):
        raise ValueError('Exact smoke contract required')


def run(inputs, out, contract, *, started):
    """Only original-training images/labels are indexed; all outputs are aggregates."""
    require_contract(contract)
    from knee.fresh_reader_setup import dataset_root, notebook_root, time_limit
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    report = {'version': VERSION, 'status': 'running', 'phase': 'setup',
        'contract_sha256': digest(canonical(contract)), 'accepted_updates': 0, 'amp_skips': 0,
        'attempted_updates': 0, 'epochs': [], 'settings': SETTINGS,
        'predictive_quality': 'unmeasured; train-only wiring smoke', 'fitted_weights_exported': False}
    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['peak_host_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        tmp = out/'smoke-report.json.tmp'; tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        tmp.replace(out/'smoke-report.json')
    save()
    try:
        with time_limit(SETTINGS['notebook_seconds']-(time.monotonic()-started), 'actual Kaggle12h platform limit'):
            import numpy as np
            import torch
            if sys.version_info[:2] != (3,12) or not torch.cuda.is_available():
                raise RuntimeError('Python3.12 CUDA required; no CPU fallback')
            versions = {p: importlib.metadata.version(p) for p in PACKAGES}
            if versions != PACKAGES:
                raise ValueError('Pinned ML environment differs')
            report['environment'] = {'packages': versions, 'python': sys.version.split()[0],
                'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                'gpu': torch.cuda.get_device_name(0), 'pyarrow': importlib.metadata.version('pyarrow'),
                'scikit_learn': importlib.metadata.version('scikit-learn')}
            torch.set_num_threads(4)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.deterministic = False
            report['backend'] = {'torch_threads': 4, 'cudnn_benchmark': False, 'matmul_tf32': False,
                'cudnn_tf32': True, 'cudnn_deterministic': False, 'device': 'cuda:0'}
            random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
            identity = notebook_root(inputs, IDENTITY_KERNEL, 'sealed_identity', ['sealed-manifest-private.json'])
            selected, cohort = membership(identity/'sealed-manifest-private.json')
            if cohort != contract['cohort']:
                raise ValueError('Mounted cohort differs')
            report['cohort'] = cohort; report['phase'] = 'corpus'; save()
            assets = dataset_root(inputs, ASSETS, ['train_knee.py', 'labels_v1_blend.parquet'])
            require_file(assets/'train_knee.py', SOURCE_SHA)
            roots = [dataset_root(inputs, ref, names) for ref,names in zip(CORPORA,
                     [('all_vols.npy','all_ids.npy','all_masks.npy'), ('extra_vols.npy','extra_ids.npy','extra_masks.npy')])]
            for root, prefix in zip(roots, ['all','extra']):
                for suffix in ['ids','masks']:
                    name = f'{prefix}_{suffix}.npy'; require_file(root/name, PINS[name])
            ids = np.concatenate([np.load(root/f'{prefix}_ids.npy', allow_pickle=False).astype(str)
                                  for root,prefix in zip(roots,['all','extra'])])
            if len(ids) != 4407 or len(set(ids)) != 4407 or not set(selected) <= set(ids):
                raise ValueError('Corpus identity coverage differs')
            id2row = {u:i for i,u in enumerate(ids)}
            vols = [np.load(root/f'{prefix}_vols.npy', mmap_mode='r', allow_pickle=False) for root,prefix in zip(roots,['all','extra'])]
            for vol, size in zip(vols,[3200,1207]):
                if vol.shape != (size,44,336,336) or vol.dtype != np.uint8:
                    raise ValueError('Public corpus shape/type differs')
            masks = np.concatenate([np.load(root/f'{prefix}_masks.npy', allow_pickle=False) for root,prefix in zip(roots,['all','extra'])])
            if masks.shape != (4407,44) or not np.isin(masks,[0,1]).all():
                raise ValueError('Corpus mask schema differs')
            source = (assets/'train_knee.py').read_text()
            public = public_definitions(source)
            volume = public['_TwoPartVols'](*vols)
            public['_open_corpus'] = lambda: (volume, masks)
            pixels = hashlib.sha256()
            for uid in selected:
                row = id2row[uid]; v = volume[row]
                if not masks[row].any() or v[~masks[row].astype(bool)].any():
                    raise ValueError('Selected corpus pixels/mask differ or study empty')
                pixels.update(v.tobytes())
            report['selected_corpus_sha256_first_observed'] = pixels.hexdigest()
            labels, values = selected_labels(assets/'labels_v1_blend.parquet', selected)
            report['selected_targets_sha256'] = digest(values.tobytes())
            if report['selected_targets_sha256'] != contract['selected_targets_sha256']:
                raise ValueError('Selected label content/order differs')
            prev = np.clip(values.mean(0), .03, .7); weights = np.clip((1-prev)/prev,1,10)
            report['positive_weights'] = weights.tolist()
            raw_prev = values.mean(0)
            report['training_prior'] = metrics(np.tile(raw_prev,(40,1)), values, weights)
            optimum = weights*raw_prev/(1-raw_prev+weights*raw_prev)
            report['weighted_training_prior'] = metrics(np.tile(optimum,(40,1)), values, weights)
            common = dict(root='', ids=selected, id2row=id2row, labels=labels, res=256, norm='imagenet')
            train = public['StudyWindows'](**common,k=8,train=True)
            valid = public['StudyWindows'](**common,k=16,train=False)
            tl = torch.utils.data.DataLoader(train,batch_size=8,shuffle=True,num_workers=0,drop_last=True,collate_fn=public['collate'])
            vl = torch.utils.data.DataLoader(valid,batch_size=4,shuffle=False,num_workers=0,collate_fn=public['collate'])
            report['phase'] = 'generic_initialization'; save()
            import timm
            config = timm.get_pretrained_cfg(SETTINGS['arch'])
            if config.hf_hub_id != 'timm/convnext_tiny.fb_in22k_ft_in1k':
                raise ValueError('Expected generic ImageNet pretrained source')
            backbone = public['build_backbone'](SETTINGS['arch'],pretrained=True)
            report['generic_pretrained_config'] = {k:v for k,v in backbone.pretrained_cfg.items() if k in ('architecture','tag','hf_hub_id','url','input_size','mean','std')}
            model = public['RaptorClassifier'](backbone,F_dim=backbone.num_features).to('cuda:0')
            initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            report['initial_model_tensor_sha256'] = tensor_digest(initial)
            report['generic_asset_hashes'] = {str(p.relative_to(Path('/tmp/public-training-assets'))):file_sha(p)
                for p in Path('/tmp/public-training-assets').rglob('*') if p.is_file() and p.suffix in ('.safetensors','.bin','.pth','.pt')}
            if not report['generic_asset_hashes']:
                raise ValueError('Generic ImageNet asset provenance missing')
            optimizer = torch.optim.AdamW([{'params':[p for n,p in model.named_parameters() if n.startswith('backbone.')],'lr':3e-5},
                {'params':[p for n,p in model.named_parameters() if not n.startswith('backbone.')],'lr':1e-3}],weight_decay=.02)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=[3e-5,1e-3],total_steps=10,pct_start=.15)
            scaler = torch.amp.GradScaler('cuda',enabled=True)
            lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights,device='cuda:0'))
            def evaluate():
                model.eval(); predictions = []
                with torch.no_grad():
                    for x,_ in vl:
                        with torch.autocast('cuda',dtype=torch.float16):
                            logits = model(x.to('cuda:0'))
                        if not bool(torch.isfinite(logits).all()):
                            raise FloatingPointError('Nonfinite evaluation logits')
                        predictions.append(torch.sigmoid(logits.float()).cpu().numpy())
                return np.concatenate(predictions)
            report['initial_train_metrics'] = metrics(evaluate(),values,weights)
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(2):
                report['phase'] = 'training'; model.train(); optimizer.zero_grad(set_to_none=True)
                losses = []; steps = []
                for x,y in tl:
                    if not bool(torch.isfinite(x).all()):
                        raise FloatingPointError('Nonfinite input windows')
                    with torch.autocast('cuda',dtype=torch.float16):
                        logits = model(x.to('cuda:0')); loss = lossf(logits.float(),y.to('cuda:0'))
                    event = update(model,optimizer,scaler,scheduler,loss)
                    report['attempted_updates'] += 1
                    report['accepted_updates'] += int(event['accepted']); report['amp_skips'] += int(not event['accepted'])
                    losses.append(float(loss.detach())); steps.append(event)
                report['phase'] = 'train_evaluation'
                epoch_result = {'epoch':epoch+1,'mean_train_loss':sum(losses)/len(losses), 'steps':steps,
                    'clipped_fraction_of_accepted':sum(s['clipped'] for s in steps)/max(1,sum(s['accepted'] for s in steps)),
                    'train_metrics':metrics(evaluate(),values,weights), 'parameter_movement':movement(model,initial),
                    'scheduler_last_epoch':scheduler.last_epoch, 'scaler_scale':float(scaler.get_scale())}
                report['epochs'].append(epoch_result); save()
                print(json.dumps({'event':'epoch_complete','epoch':epoch+1,'accepted_updates':report['accepted_updates'],
                    'amp_skips':report['amp_skips'],'loss':epoch_result['mean_train_loss'],
                    'train_bce':epoch_result['train_metrics']['macro_bce']}),flush=True)
            report['phase'] = 'reload_check'
            before = evaluate()
            temp = Path('/tmp/public-training-smoke-checkpoint.pt')
            try:
                torch.save(model.state_dict(),temp)
                reloaded = torch.load(temp,map_location='cpu',weights_only=True)
                model.load_state_dict(reloaded,strict=True)
                error = float(np.abs(before-evaluate()).max())
                if error != 0:
                    raise ValueError('Same-process exact checkpoint reload differs')
                report['checkpoint_reload_max_error'] = error
            finally:
                temp.unlink(missing_ok=True)
            moves = movement(model,initial)
            if report['accepted_updates'] < 1 or any(moves[g] <= 0 for g in ('encoder','nonbias_head')):
                raise ValueError('Accepted encoder/head learning updates not demonstrated')
            report.update(status='PASS',phase='complete',final_parameter_movement=moves,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            save()
            require_result(report, contract)
    except BaseException as error:
        report['status'] = 'FAIL'
        report['error_class'] = type(error).__name__ if type(error).__module__ in ('builtins','torch','torch.cuda') else 'ExternalError'
        save()
        raise RuntimeError('Public training smoke failed; inspect sanitized aggregate receipt') from None
    return report


def require_result(report, contract):
    """Acceptance means training wiring, never predictive model quality."""
    require_contract(contract)
    epochs = report.get('epochs', [])
    if (report.get('status') != 'PASS' or report.get('phase') != 'complete' or report.get('version') != VERSION or
            report.get('contract_sha256') != digest(canonical(contract)) or report.get('cohort') != contract['cohort'] or
            report.get('settings') != SETTINGS or report.get('fitted_weights_exported') is not False or
            report.get('environment', {}).get('packages') != PACKAGES or
            report.get('attempted_updates') != 10 or not 1 <= report.get('accepted_updates', 0) <= 10 or
            report.get('accepted_updates', 0) + report.get('amp_skips', 0) != 10 or
            len(epochs) != 2 or [e.get('epoch') for e in epochs] != [1, 2] or
            report.get('checkpoint_reload_max_error') != 0 or
            report.get('selected_targets_sha256') != contract['selected_targets_sha256']):
        raise ValueError('Complete exact training smoke evidence required')
    steps = [step for epoch in epochs for step in epoch['steps']]
    if len(steps) != 10 or sum(s['accepted'] for s in steps) != report['accepted_updates']:
        raise ValueError('Accepted update accounting differs')
    for step in steps:
        if step['accepted'] and (not math.isfinite(step['preclip_norm']) or
                any(not v['finite'] or not math.isfinite(v['norm']) for v in step['gradient'].values())):
            raise ValueError('Accepted update gradients invalid')
    for group in ('encoder', 'nonbias_head'):
        moved = report.get('final_parameter_movement', {}).get(group, 0)
        if not math.isfinite(moved) or moved <= 0:
            raise ValueError('Encoder/head movement not demonstrated')
    for key in ('initial_model_tensor_sha256', 'selected_corpus_sha256_first_observed'):
        if not re.fullmatch('[0-9a-f]{64}', report.get(key, '')):
            raise ValueError('Observed model/corpus evidence missing')
    assets = report.get('generic_asset_hashes', {})
    if not assets or any(not re.fullmatch('[0-9a-f]{64}', h) for h in assets.values()):
        raise ValueError('Generic initialization artifact hashes missing')
    if not 0 < report.get('elapsed_seconds', float('inf')) <= SETTINGS['notebook_seconds']:
        raise ValueError('Actual platform limit exceeded')
    for key in ('peak_host_rss_bytes', 'peak_cuda_allocated_bytes', 'peak_cuda_reserved_bytes'):
        if not isinstance(report.get(key), int) or report[key] <= 0:
            raise ValueError('Runtime memory evidence missing')
    for measurement in [report['initial_train_metrics'], report['training_prior'], report['weighted_training_prior'],
                        *[e['train_metrics'] for e in epochs]]:
        if set(measurement['per_target']) != set(TARGETS) or not math.isfinite(measurement['macro_bce']):
            raise ValueError('Training-only metric coverage differs')
        if any(t['positive_at_half'] + t['negative_at_half'] != 40 for t in measurement['per_target'].values()):
            raise ValueError('Training-only metric support differs')
