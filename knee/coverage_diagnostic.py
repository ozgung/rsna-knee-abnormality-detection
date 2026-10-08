"""Disposable TRAIN04 training-only diagnostic; never writes model weights."""
import copy
import json
import math
from pathlib import Path
import time


def parameter_group(name):
    if name.startswith('enc.'):
        return 'encoder'
    if name != 'cls_b' and not name.endswith('.bias'):
        return 'nonbias_head'
    return 'head_bias'


def gradient_summary(model):
    import torch
    result={}
    for group in ('encoder','nonbias_head','head_bias'):
        params=[p for n,p in model.named_parameters() if parameter_group(n)==group]
        grads=[p.grad.detach().float() for p in params if p.grad is not None]
        finite=all(bool(torch.isfinite(g).all()) for g in grads)
        result[group]={'parameters':len(params),'with_gradient':len(grads),'finite':finite,
            'norm':math.sqrt(sum(float(g.square().sum()) for g in grads)) if finite else None}
    return result


def movement(model, initial):
    result={}
    for group in ('encoder','nonbias_head','head_bias'):
        maximum=total=base=0.
        for name,param in model.named_parameters():
            if parameter_group(name)!=group:continue
            old=initial[name].float();delta=param.detach().cpu().float()-old
            maximum=max(maximum,float(delta.abs().max()));total+=float(delta.square().sum())
            base+=float(old.square().sum())
        result[group]={'max_abs':maximum,'l2':math.sqrt(total),
                       'relative_l2':math.sqrt(total/base) if base else None}
    return result


def update(model,ema,optimizer,scaler,scheduler,loss,accepted):
    """Post-forward boundary: empty/overflow never advance optimizer/scheduler/EMA.

    GradScaler alone may lower its scale on overflow, as in the original recipe.
    """
    import torch
    if loss is None:return {'kind':'empty','accepted':accepted}
    if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite diagnostic loss')
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    grads=gradient_summary(model)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
    before=scaler.get_scale();scaler.step(optimizer);scaler.update()
    overflow=scaler.get_scale()<before
    if not overflow:
        if not all(g['finite'] for g in grads.values()):raise ValueError('Accepted nonfinite gradients')
        scheduler.step();accepted+=1;decay=min(.998,(1+accepted)/(10+accepted))
        with torch.no_grad():
            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-decay)
            for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)
    return {'kind':'overflow' if overflow else 'accepted','accepted':accepted,
            'loss':float(loss.detach()),'gradient':grads,
            'preclip_norm':float(norm) if bool(torch.isfinite(norm)) else None,
            'clipped':bool(norm>3),'scale_before':before,'scale_after':scaler.get_scale()}


def run(cache,sources,initial,out,contract,*,deadline):
    import numpy as np
    import torch
    from knee.coverage_cache import sample_windows,verify_index
    from knee.coverage_training import CONFIG,INITIAL_ENCODER_SHA256,VolumeCache,supervision,weighted_terms
    from knee.fresh_reader import (environment,fingerprint,load_source,macro_loss,restore_rng,
        tensor_state_sha256,training_diagnostic_indices,validate_plan,write_json)
    from knee.fresh_reader_setup import progress
    from knee.validation import file_sha256
    start=time.monotonic();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','purpose':'trainability only; no development metrics or reusable fitted weights',
            'contract':contract,'phases':[],'fitted_weights_saved':False}
    def save():
        report['elapsed_seconds']=time.monotonic()-start
        write_json(out/'diagnostic-report.json',report)
    def check_time():
        if time.monotonic()>=deadline:raise TimeoutError('Platform session limit')
    try:
        if not torch.cuda.is_available():raise RuntimeError('Diagnostic requires CUDA')
        env=environment();report['environment']=env
        if {k:env[k] for k in ('python','versions','device','cuda_devices')}!=contract['reference_environment']:
            raise ValueError('Original runtime environment differs')
        cache=Path(cache);plan=json.loads((cache/'plan-private.json').read_text());validate_plan(plan)
        if plan['fingerprint']!=contract['plan_fingerprint']:raise ValueError('Plan differs')
        index=json.loads((cache/'index-private.json').read_text());verify_index(cache,plan,index,complete=True)
        if file_sha256(initial)!=contract['initial_sha256']:raise ValueError('Initial artifact differs')
        init=torch.load(initial,map_location='cpu',weights_only=False)
        if init['config']!=CONFIG or init['plan_fingerprint']!=plan['fingerprint']:
            raise ValueError('Initialization configuration differs')
        if init['initial_encoder_sha256']!=INITIAL_ENCODER_SHA256:raise ValueError('Initial encoder receipt differs')
        code=load_source(sources,'knee');rows=plan['records'];y,w=supervision(plan)
        train=[i for i,r in enumerate(rows) if r['split']=='train']
        chosen=training_diagnostic_indices(rows,32)
        if len(chosen)!=32 or any(rows[i]['split']!='train' for i in chosen):raise ValueError('Training selection differs')
        prior=np.clip((y[train]*w[train]).sum(0)/w[train].sum(0),1e-4,1-1e-4)
        report.update(selection_fingerprint=fingerprint([rows[i]['uid'] for i in chosen]),
                      training_studies=32,targets=plan['targets'],selection_policy='fixed ID hash, independent of labels')
        volumes=VolumeCache(cache,index)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
        def batch(indices,augmented):
            samples=[sample_windows(volumes.get(rows[i]['uid']),8,train=augmented,rng=np.random) for i in indices]
            x,m,p=[torch.from_numpy(np.stack([s[j] for s in samples])).cuda() for j in range(3)]
            return x,m,p,torch.tensor(y[indices],device='cuda'),torch.tensor(w[indices],device='cuda')
        @torch.inference_mode()
        def evaluate(model,indices):
            model.eval();nums=torch.zeros(12);dens=torch.zeros(12)
            for offset in range(0,len(indices),2):
                check_time();x,m,p,yy,ww=batch(indices[offset:offset+2],False)
                with torch.autocast('cuda',dtype=torch.float16):z=model(x,m,p,res=256,train=False)
                n,d=weighted_terms(z,yy,ww);nums+=n.cpu();dens+=d.cpu()
            if not torch.isfinite(nums).all() or not (dens>0).any():raise ValueError('Invalid training evaluation')
            return {'macro_bce':float(macro_loss(nums,dens)),
                    'per_target_bce':[float(n/d) if d>0 else None for n,d in zip(nums,dens)],
                    'confidence_support':dens.tolist(),'undefined_targets':[t for t,d in zip(plan['targets'],dens) if d<=0]}
        yy=torch.tensor(y[chosen]);ww=torch.tensor(w[chosen]);den=ww.sum(0);known=den>0
        n,d=weighted_terms(torch.tensor(np.log(prior/(1-prior))).expand_as(yy),yy,ww)
        entropy=-(yy.clamp_min(1e-30).log()*yy+(1-yy).clamp_min(1e-30).log()*(1-yy))
        floors=(entropy*ww).sum(0)
        report['comparators']={'evaluation_cohort':'fixed32_overfit',
            'prior_source':'all 2000 training rows only','prior_macro_bce':float(macro_loss(n,d)),
            'prior_per_target_bce':[float(a/b) if b>0 else None for a,b in zip(n,d)],
            'entropy_floor_macro_bce':float((floors[known]/den[known]).mean()),
            'entropy_floor_per_target':[float(a/b) if b>0 else None for a,b in zip(floors,den)]}
        for phase,target_updates in [('normal_augmented_batch',1),('fixed32_overfit',64)]:
            check_time()
            model=code.KneeNet(CONFIG['backbone'],pretrained=False,img_size=256)
            model.load_state_dict(init['model'],strict=True);model=model.cuda().to(memory_format=torch.channels_last)
            if tensor_state_sha256(model.enc.state_dict())!=INITIAL_ENCODER_SHA256:raise ValueError('Initial encoder tensors differ')
            ema=copy.deepcopy(model).eval().requires_grad_(False)
            optimizer=torch.optim.AdamW([{'params':model.enc.parameters(),'lr':1e-4},
                {'params':[p for n,p in model.named_parameters() if not n.startswith('enc.')],'lr':5e-4}],weight_decay=.02)
            scaler=torch.amp.GradScaler('cuda');scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,12000)
            restore_rng(init['rng'])
            normal=np.random.permutation(train).tolist()[:2] if phase=='normal_augmented_batch' else None
            indices=normal if normal is not None else chosen
            row={'phase':phase,'accepted':0,'attempted':0,'empty_skips':0,'amp_skips':0,'steps':[],
                 'initial_online':evaluate(model,indices),'initial_ema':evaluate(ema,indices)}
            report['phases'].append(row);save();consecutive=0
            while row['accepted']<target_updates:
                check_time();model.train()
                offset=(row['attempted']%16)*2
                ids=normal if normal is not None else chosen[offset:offset+2]
                optimizer.zero_grad(set_to_none=True);x,m,p,yy,ww=batch(ids,normal is not None)
                with torch.autocast('cuda',dtype=torch.float16):n,d=weighted_terms(model(x,m,p,res=256,train=normal is not None),yy,ww)
                if not torch.isfinite(n).all():raise ValueError('Nonfinite diagnostic terms')
                loss=macro_loss(n,d) if (d>0).any() else None
                result=update(model,ema,optimizer,scaler,scheduler,loss,row['accepted'])
                row['attempted']+=1;row['accepted']=result['accepted'];row['steps'].append(result)
                row['empty_skips']+=int(result['kind']=='empty');row['amp_skips']+=int(result['kind']=='overflow')
                consecutive=consecutive+1 if result['kind']=='overflow' else 0
                if consecutive>5:raise ValueError('More than five consecutive AMP overflows')
                if normal is not None and result['kind']=='empty':raise ValueError('Normal diagnostic batch has no supervision')
                if row['attempted']%16==0 or row['accepted']==target_updates:
                    row['online']=evaluate(model,indices);row['ema']=evaluate(ema,indices)
                    row['movement']=movement(model,init['model']);row['ema_movement']=movement(ema,init['model'])
                    save();progress('train04_diagnostic_progress',phase=phase,accepted=row['accepted'],
                                    attempted=row['attempted'],online=row['online'],ema=row['ema'],movement=row['movement'])
            row['scheduler_steps']=scheduler.last_epoch
            del model,ema,optimizer,scaler,scheduler,x,m,p,yy,ww,n,d,loss
        report['status']='PASS';report['learning_claim']='Inspect train BCE/entropy excess changes; execution PASS is not generalization'
        report['cuda_peak_bytes']=torch.cuda.max_memory_allocated();save();return report
    except BaseException as exc:
        report['status']='FAIL';report['failure_type']=type(exc).__name__;save();raise
