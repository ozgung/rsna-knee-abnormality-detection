"""TRAIN04 fixed twelve-epoch image training; heavy imports are runtime-only."""
from collections import OrderedDict
import copy
import csv
import json
import math
from pathlib import Path
import random
import time

from knee.coverage_cache import VERSION, read_volumes, sample_windows, verify_index
from knee.fresh_reader import (environment, fingerprint, load_source, macro_loss, per_target_bce,
    restore_rng, rng_state, save_checkpoint, tensor_state_sha256, training_diagnostic_indices,
    validate_plan, weak_validation_evidence, write_json)
from knee.fresh_reader_setup import progress, time_limit
from knee.validation import file_sha256, evaluate_predictions
from knee.coverage_objective import (BATCH_OBJECTIVE, GLOBAL_OBJECTIVE, EMPTY_POLICY,
    REFERENCE_SOURCES, training_mean_weights, training_loss, require_objective_invariants,
    copy_initial, load_exact_initial, StepTelemetry, parameter_probes, probe_displacements)

CONFIG = dict(seed=42, backbone='convnext_tiny.fb_in22k_ft_in1k', image_size=256,
              train_windows=8, eval_windows=12, batch_size=2, epochs=12,
              lr_encoder=1e-4, lr_head=5e-4, weight_decay=.02, ema=.998,
              seconds=12*3600, host_cache_bytes=256*1024**2)
# Kaggle's GPU session limit, not an experiment spending budget. Timing estimates
# are informational; reserve two minutes only to save state before termination.
CHECKPOINT_SECONDS = 120
INITIAL_ENCODER_SHA256='355cb8e236d6793f7ec440278ad14a59a5484663a99f9ed54d8b6ea105b8bf66'
LABEL_SOURCES = {
 'steven': ('llm_labels_v4_blend.csv','eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4'),
 'pilkwang': ('report_labels_v2.csv','6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920'),
 'lixin': ('labels_llm_gpt56sol.csv','a79b05e609336811d51c97f9a48f00ea8004244a1a80fe886504a62bb39867ae'),
 'qwen': ('qwen_knee_weak_labels.csv','26adba0228422b3fa2b77e2a9e4771741b144c59f5a59e23e6c58dadaf19e043')}


def supervision(plan, *, source_paths=None, with_audit=False):
    import numpy as np
    original=np.asarray([r['labels'] for r in plan['records']],np.float32)
    targets=original.copy();weights=2*np.abs(original-.5)
    audit={}
    if source_paths is not None:
        if set(source_paths)!=set(LABEL_SOURCES):raise ValueError('Exact four label sources required')
        lookup={r['uid']:i for i,r in enumerate(plan['records']) if r['split']=='train'}
        tables=[]
        for key,(filename,digest) in LABEL_SOURCES.items():
            path=Path(source_paths[key])
            if path.name!=filename or file_sha256(path)!=digest:raise ValueError('Alternate source identity changed')
            values=np.full(original.shape,np.nan,np.float64);seen=set();failed=0
            with path.open(newline='') as handle:
                reader=csv.DictReader(handle)
                columns=[t+'__label' for t in plan['targets']] if key=='qwen' else plan['targets']
                if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames) or not set(columns+['StudyInstanceUID'])<=set(reader.fieldnames):
                    raise ValueError('Alternate target schema differs')
                for row in reader:
                    uid=row['StudyInstanceUID']
                    if uid not in lookup:continue
                    if uid in seen:raise ValueError('Duplicate alternate training row')
                    seen.add(uid)
                    if 'status' in row and row['status']!='ok':
                        if row['status'] not in ('error','missing','parse_error','failed'):
                            raise ValueError('Unrecognized alternate source status')
                        failed+=1;continue
                    for j,col in enumerate(columns):
                        raw=row[col].strip()
                        if raw.lower() not in ('','nan','none','null'):
                            value=float(raw)
                            if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid alternate target')
                            values[lookup[uid],j]=value
            if seen!=set(lookup):raise ValueError('Alternate training coverage differs')
            audit[key]={'matched_training_rows':len(seen),'failed_rows_excluded':failed}
            tables.append(values)
        stack=np.stack(tables);count=np.isfinite(stack).sum(0)
        tr=list(lookup.values())
        if np.any(count[tr]==0):raise ValueError('Missing consensus evidence')
        consensus=np.clip(np.nansum(stack[:,tr],axis=0)/count[tr],.01,.99).astype(np.float32)
        targets[tr]=np.where(weights[tr]>0,consensus,.5)
    if not np.isfinite(targets).all():raise ValueError('Invalid supervision')
    return (targets,weights,audit) if with_audit else (targets,weights)


def weighted_terms(logits, targets, weights):
    import torch
    if logits.shape!=targets.shape or targets.shape!=weights.shape:
        raise ValueError('Supervision shape mismatch')
    return (torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),targets.float(),reduction='none')*weights).sum(0),weights.sum(0)


def timing_projection(train_batch, eval_batch, elapsed, *, diagnostic_batch=None, train_count=2000, dev_count=200, diagnostic_count=64, remaining_seconds=None):
    diagnostic_batch=eval_batch if diagnostic_batch is None else diagnostic_batch
    if any(not math.isfinite(x) or x<=0 for x in (train_batch,eval_batch,diagnostic_batch)) or not math.isfinite(elapsed) or elapsed<0:
        raise ValueError('Invalid throughput measurements')
    per_epoch=1.5*(math.ceil(train_count/2)*train_batch+math.ceil(dev_count/2)*eval_batch+math.ceil(diagnostic_count/2)*diagnostic_batch)
    remaining=CONFIG['seconds']-elapsed-CHECKPOINT_SECONDS if remaining_seconds is None else remaining_seconds
    return dict(passed=CONFIG['epochs']*per_epoch<=remaining,epochs=CONFIG['epochs'],
                enforcement='informational_only',
                estimated_epoch_seconds_with_margin=per_epoch,remaining_seconds=remaining,
                train_batch_seconds=train_batch,eval_batch_seconds=eval_batch,diagnostic_batch_seconds=diagnostic_batch,margin=1.5)


def validate_resume_state(state, contract, elapsed_before):
    if state['contract']!=contract or state['epoch']>=CONFIG['epochs'] or state['elapsed_seconds']>elapsed_before:
        raise ValueError('Resume state/horizon differs')
    # The previous session's ETA is evidence, never permission to resume.


class VolumeCache:
    def __init__(self,root,index,limit=CONFIG['host_cache_bytes']):
        self.root,self.index,self.limit=Path(root),index,limit
        self.items=OrderedDict();self.bytes=0

    def get(self,uid):
        if uid in self.items:
            self.items.move_to_end(uid);return self.items[uid]
        value=read_volumes(self.root/'arrays'/self.index[uid]['file'])
        size=sum(v.nbytes for v in value)
        while self.items and self.bytes+size>self.limit:
            _,old=self.items.popitem(last=False);self.bytes-=sum(v.nbytes for v in old)
        if size<=self.limit:self.items[uid]=value;self.bytes+=size
        return value


def runtime_sources():
    root=Path(__file__).resolve().parents[1]
    return {name:file_sha256(root/name) for name in ('knee/coverage_training.py','knee/coverage_cache.py',
        'knee/fresh_reader.py','knee/fresh_reader_setup.py','knee/validation.py','knee/coverage_objective.py')}


def comparison_sources():
    root=Path(__file__).resolve().parents[1]
    return {name:file_sha256(root/name) for name in
            (*REFERENCE_SOURCES,'knee/coverage_training.py','knee/coverage_objective.py')}


def require_pair_invariants(current,reference,*,environment=True):
    keys=['config','plan_fingerprint','runner_sha256','source_hashes','runtime_sources',
          'index_sha256','complete_sha256','weights_fingerprint']
    if environment:keys.append('runtime_environment')
    if reference.get('variant')!='v1' or any(k not in current or current[k]!=reference.get(k) for k in keys):
        raise ValueError('Controlled target comparison changed a non-label invariant')


def _train(cache, sources, out, *, source_paths=None, initial=None, resume=None,reference_contract=None,
           session_deadline=None,objective=BATCH_OBJECTIVE,reference_initial_sha256=None):
    import numpy as np
    import torch
    started=time.monotonic();elapsed_before=0.
    if objective not in (BATCH_OBJECTIVE,GLOBAL_OBJECTIVE):raise ValueError('Unknown TRAIN04 objective')
    if objective==GLOBAL_OBJECTIVE and (source_paths is not None or initial is None or reference_contract is None):
        raise ValueError('Global objective requires original v1 initialization and contract')
    session_deadline=started+CONFIG['seconds'] if session_deadline is None else session_deadline
    if not math.isfinite(session_deadline) or session_deadline<=started:
        raise ValueError('Invalid platform session deadline')
    def session_remaining():return session_deadline-time.monotonic()-CHECKPOINT_SECONDS
    cache,out=Path(cache),Path(out)
    if not torch.cuda.is_available():raise RuntimeError('TRAIN04 requires CUDA; no fallback')
    plan=json.loads((cache/'plan-private.json').read_text());validate_plan(plan)
    complete=json.loads((cache/'cache-report.json').read_text())
    if (plan['version']!=VERSION or complete['status']!='PASS' or complete['stage']!='complete' or
            complete['plan_fingerprint']!=plan['fingerprint'] or complete['cached_studies']!=2200):
        raise ValueError('Complete matching coverage cache required')
    labels,weights,label_audit=supervision(plan,source_paths=source_paths,with_audit=True)
    variant='consensus' if source_paths else 'v1'
    if variant=='consensus' and initial is None:raise ValueError('Consensus requires exact reference initialization')
    env=environment()
    contract={'version':'train04-v1','variant':variant,'config':CONFIG,'plan_fingerprint':plan['fingerprint'],
              'runner_sha256':file_sha256(__file__),'source_hashes':plan['source_hashes'],
              'runtime_sources':runtime_sources(),'runtime_environment':{k:env[k] for k in ('python','versions','device','cuda_devices')},
              'index_sha256':file_sha256(cache/'index-private.json'),'complete_sha256':file_sha256(cache/'cache-report.json'),
              'labels_fingerprint':fingerprint(labels.tolist()),'weights_fingerprint':fingerprint(weights.tolist()),
              'label_sources':LABEL_SOURCES if source_paths else {},'label_source_audit':label_audit,
              'initial_file_sha256':file_sha256(initial) if initial else None}
    train_indices=[i for i,r in enumerate(plan['records']) if r['split']=='train']
    mean_weights=training_mean_weights(weights,train_indices)
    contract.update(objective=objective,empty_batch_policy=EMPTY_POLICY,
                    train_mean_weights=mean_weights,train_mean_weights_fingerprint=fingerprint(mean_weights),
                    comparison_sources=comparison_sources(),
                    reference_contract_fingerprint=reference_contract.get('fingerprint') if reference_contract else None)
    contract['fingerprint']=fingerprint(contract)
    if objective==GLOBAL_OBJECTIVE:
        require_objective_invariants(contract,reference_contract)
        if not reference_initial_sha256 or contract['initial_file_sha256']!=reference_initial_sha256:
            raise ValueError('Global objective initialization artifact differs')
    elif variant=='consensus':
        if reference_contract is None:raise ValueError('Controlled comparison requires reference contract')
        require_pair_invariants(contract,reference_contract)
    elif reference_contract is not None:raise ValueError('Unexpected reference contract for original targets')
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'preparation-report.json',{'status':'RUNNING','contract':contract})
    state=None
    if resume:
        previous=json.loads((Path(resume).parent/'training-report.json').read_text())
        if previous['contract']!=contract:raise ValueError('Resume contract changed')
        elapsed_before=previous['elapsed_seconds']
        if not math.isfinite(elapsed_before) or elapsed_before<0:
            raise ValueError('Invalid resume elapsed time')
        # Only our private full-state checkpoint is loaded; never public pickle weights.
        state=torch.load(resume,map_location='cpu',weights_only=False)
        validate_resume_state(state,contract,elapsed_before)
    def elapsed():return elapsed_before+time.monotonic()-started
    def deadline():
        if session_remaining()<=0:raise TimeoutError('Kaggle session ending; save checkpoint')
    rows=plan['records'];tr=[i for i,r in enumerate(rows) if r['split']=='train'];va=[i for i,r in enumerate(rows) if r['split']=='weak_validation']
    if (len(tr),len(va))!=(2000,200):raise ValueError('Training cohort differs')
    diagnostic=training_diagnostic_indices(rows)
    index=json.loads((cache/'index-private.json').read_text())
    progress('train04_cache_verify_begin',studies=len(rows))
    with time_limit(session_remaining(),'Kaggle session: preparation'):
        verify_index(cache,plan,index,complete=True)
    progress('train04_cache_verify_complete',seconds=elapsed())
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    original=np.asarray([r['labels'] for r in rows],np.float32)
    prior=np.clip((labels[tr]*weights[tr]).sum(0)/weights[tr].sum(0),1e-4,1-1e-4)
    original_prior=np.clip((original[tr]*weights[tr]).sum(0)/weights[tr].sum(0),1e-4,1-1e-4)
    if not np.isfinite(prior).all() or np.any(weights[tr].sum(0)<=0):raise ValueError('Training support missing')
    with time_limit(session_remaining(),'Kaggle session: initialization'):
        code=load_source(sources,'knee')
        model=code.KneeNet(CONFIG['backbone'],pretrained=initial is None and state is None,img_size=256)
        if initial:
            init=load_exact_initial(model,initial,expected_sha256=contract['initial_file_sha256'],
                                    plan_fingerprint=plan['fingerprint'],config=CONFIG)
        else:
            model.cls_b.data.copy_(torch.tensor(np.log(original_prior/(1-original_prior))))
        model=model.cuda().to(memory_format=torch.channels_last)
        init_hash=tensor_state_sha256(model.enc.state_dict()) if state is None else state['initial_encoder_sha256']
        if init_hash!=INITIAL_ENCODER_SHA256:raise ValueError('Generic ImageNet encoder changed after CUDA probe')
        if initial and init_hash!=init['initial_encoder_sha256']:raise ValueError('Initial encoder tensors differ')
        if state is None:
            if objective==GLOBAL_OBJECTIVE:
                copy_initial(initial,out/'initial.pt',reference_initial_sha256)
            else:
                save_checkpoint(out/'initial.pt',dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                     config=CONFIG,plan_fingerprint=plan['fingerprint'],initial_encoder_sha256=init_hash,rng=rng_state()))
        probes=parameter_probes(model)
        ema=copy.deepcopy(model).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW([{'params':model.enc.parameters(),'lr':CONFIG['lr_encoder']},
          {'params':[p for n,p in model.named_parameters() if not n.startswith('enc.')],'lr':CONFIG['lr_head']}],weight_decay=.02)
        scaler=torch.amp.GradScaler('cuda')
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,12*1000)
    mean_weights_tensor=torch.tensor(mean_weights,device='cuda',dtype=torch.float32)
    volumes=VolumeCache(cache,index)
    def batch(indices,training=False):
        samples=[sample_windows(volumes.get(rows[i]['uid']),8 if training else 12,train=training,rng=np.random) for i in indices]
        x,m,p=[torch.from_numpy(np.stack([s[j] for s in samples])).cuda() for j in range(3)]
        return x,m,p,torch.tensor(labels[indices],device='cuda'),torch.tensor(weights[indices],device='cuda')
    @torch.inference_mode()
    def evaluate(current,indices,*,require_all=True):
        current.eval();nums=torch.zeros(12);dens=torch.zeros(12);preds=[];logits=[]
        for offset in range(0,len(indices),2):
            deadline();x,m,p,y,w=batch(indices[offset:offset+2])
            with torch.autocast('cuda',dtype=torch.float16):z=current(x,m,p,res=256,train=False)
            n,d=weighted_terms(z,y,w);nums+=n.cpu();dens+=d.cpu()
            preds.extend(z.float().sigmoid().cpu().tolist());logits.extend(z.float().cpu().tolist())
        if not torch.isfinite(nums).all() or not (dens>0).any() or (require_all and not (dens>0).all()):
            raise ValueError('Evaluation nonfinite or unsupported')
        truth={str(i):[None if y==.5 else float(y>.5) for y in rows[i]['labels']] for i in indices}
        auc=evaluate_predictions(truth,{str(i):p for i,p in zip(indices,preds)},plan['targets'])
        return {'weak_bce':float(macro_loss(nums,dens)),'per_target_bce':per_target_bce(nums,dens),
                'confidence_support':dens.tolist(),'thresholded_weak_auc':auc},preds,logits
    initial_output_sha=file_sha256(out/'initial.pt') if state is None else state['initial_output_sha256']
    best_checkpoint_sha=state['best_checkpoint_sha256'] if state else None
    best_evidence_sha=state['best_evidence_sha256'] if state else None
    if state:
        model.load_state_dict(state['model']);ema.load_state_dict(state['ema']);optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler']);scheduler.load_state_dict(state['scheduler']);restore_rng(state['rng'])
        probes=state['parameter_probes']
        gate=state['gate'];shutil_best=Path(resume).parent/'best.pt'
        import shutil
        initial_previous=Path(resume).parent/'initial.pt'
        if file_sha256(initial_previous)!=initial_output_sha:raise ValueError('Resume initialization artifact changed')
        shutil.copy2(initial_previous,out/'initial.pt')
        if state['best']<math.inf:
            evidence_previous=Path(resume).parent/'best-weak-validation-private.json'
            if file_sha256(shutil_best)!=best_checkpoint_sha or file_sha256(evidence_previous)!=best_evidence_sha:
                raise ValueError('Resume selected checkpoint/evidence identity changed')
            shutil.copy2(shutil_best,out/'best.pt')
            shutil.copy2(evidence_previous,out/'best-weak-validation-private.json')
    else:
        saved_rng=rng_state();model.train();tick=time.monotonic()
        try:
            for b in range(64):
                deadline()
                if b==16:torch.cuda.synchronize();tick=time.monotonic()
                optimizer.zero_grad(set_to_none=True);x,m,p,y,w=batch(tr[2*b:2*b+2],True)
                with torch.autocast('cuda',dtype=torch.float16):n,d=weighted_terms(model(x,m,p,res=256,train=True),y,w)
                loss=training_loss(n,d,objective=objective,batch_size=len(y),mean_weights=mean_weights_tensor)
                if loss is not None:scaler.scale(loss).backward()
            torch.cuda.synchronize();train_seconds=(time.monotonic()-tick)/48
            tick=time.monotonic();evaluate(ema,va[:16],require_all=False);torch.cuda.synchronize();eval_warmup_seconds=time.monotonic()-tick
            tick=time.monotonic();evaluate(ema,va[:16],require_all=False);torch.cuda.synchronize();eval_seconds=(time.monotonic()-tick)/8
            tick=time.monotonic();evaluate(ema,diagnostic[:16],require_all=False);torch.cuda.synchronize();diagnostic_warmup_seconds=time.monotonic()-tick
            tick=time.monotonic();evaluate(ema,diagnostic[:16],require_all=False);torch.cuda.synchronize();diagnostic_seconds=(time.monotonic()-tick)/8
        finally:
            optimizer.zero_grad(set_to_none=True);restore_rng(saved_rng)
        gate=timing_projection(train_seconds,eval_seconds,elapsed(),diagnostic_batch=diagnostic_seconds,
                               remaining_seconds=session_remaining())
        gate.update(validation_warmup_seconds=eval_warmup_seconds,diagnostic_warmup_seconds=diagnostic_warmup_seconds,
                    eval_warmup_batches=8,eval_timed_batches=8,diagnostic_warmup_batches=8,diagnostic_timed_batches=8)
        write_json(out/'gpu-gate.json',gate);progress('train04_gpu_gate',**gate)
        # Keep measured ETA in receipts; it must not veto the authorized fit.
    run={'contract':contract,'environment':environment(),'initial_encoder_sha256':init_hash,
         'initial_model_sha256':file_sha256(out/'initial.pt') if (out/'initial.pt').exists() else None,
         'pretrained_cfg':model.enc.pretrained_cfg,'training_prior':prior.tolist(),
         'training_confidence_support':weights[tr].sum(0).tolist(),'gate':gate}
    write_json(out/'run-contract.json',run)
    vy=torch.tensor(labels[va]);vw=torch.tensor(weights[va]);n,d=weighted_terms(torch.tensor(np.log(prior/(1-prior))).expand_as(vy),vy,vw)
    prevalence=float(macro_loss(n,d))
    epoch=state['epoch'] if state else 0;cursor=state['cursor'] if state else 0
    order=state['order'] if state else None;history=state['history'] if state else []
    best=state['best'] if state else math.inf;step=state['step'] if state else 0
    overflow_total=state['overflow_total'] if state else 0;consecutive=state['consecutive'] if state else 0
    empty_skips=state['zero_supervision_skips'] if state else 0
    telemetry=StepTelemetry('cuda',state['telemetry'] if state else None)
    status='RUNNING'
    def save_last():
        save_checkpoint(out/'last.pt',dict(contract=contract,model=model.state_dict(),ema=ema.state_dict(),
          optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),rng=rng_state(),
          epoch=epoch,cursor=cursor,order=order,history=history,best=best,step=step,gate=gate,
          initial_encoder_sha256=init_hash,initial_output_sha256=initial_output_sha,
          best_checkpoint_sha256=best_checkpoint_sha,best_evidence_sha256=best_evidence_sha,
          overflow_total=overflow_total,consecutive=consecutive,elapsed_seconds=elapsed(),
          zero_supervision_skips=empty_skips,telemetry=telemetry.values,parameter_probes=probes))
    try:
        with time_limit(session_remaining(),'Kaggle session: fit'):
            while epoch<12:
                if order is None:order=np.random.permutation(tr).tolist()
                model.train();progress('train04_epoch_begin',epoch=epoch+1)
                while cursor<len(order):
                    deadline();x,m,p,y,w=batch(order[cursor:cursor+2],True);optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.float16):n,d=weighted_terms(model(x,m,p,res=256,train=True),y,w)
                    loss=training_loss(n,d,objective=objective,batch_size=len(y),mean_weights=mean_weights_tensor)
                    if loss is None:
                        cursor+=len(y);empty_skips+=1
                        progress('train04_empty_batch',epoch=epoch+1,studies=cursor,zero_supervision_skips=empty_skips)
                        continue
                    scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    preclip_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
                    previous_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();cursor+=2
                    overflow=scaler.get_scale()<previous_scale
                    telemetry.add(loss,preclip_norm,overflow)
                    overflow_total+=int(overflow);consecutive=consecutive+1 if overflow else 0
                    if consecutive>5:raise ValueError('More than five consecutive AMP overflows')
                    if not overflow:
                        scheduler.step();step+=1;decay=min(.998,(1+step)/(10+step))
                        with torch.no_grad():
                            for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-decay)
                            for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)
                    if step==1 and not overflow:progress('train04_parameter_probes',updates=step,probes=probe_displacements(model,probes))
                    if cursor%128==0:progress('train04_steps',epoch=epoch+1,studies=cursor,updates=step,overflows=overflow_total,
                        zero_supervision_skips=empty_skips,cumulative_training_telemetry=telemetry.report(),seconds=elapsed())
                diag,_,_=evaluate(ema,diagnostic,require_all=False);metrics,preds,logits=evaluate(ema,va)
                history.append({'epoch':epoch+1,**metrics,'training_diagnostic':diag,'elapsed_seconds':elapsed(),
                    'cumulative_training_telemetry':telemetry.report(),'parameter_probe_changes':probe_displacements(model,probes)})
                if metrics['weak_bce']<best:
                    evidence=weak_validation_evidence(plan,va,preds,logits=logits,epoch=epoch+1,weak_bce=metrics['weak_bce'])
                    save_checkpoint(out/'best.pt',dict(contract=contract,model=ema.state_dict(),epoch=epoch+1,
                        targets=plan['targets'],weak_bce=metrics['weak_bce'],initial_encoder_sha256=init_hash,
                        weak_validation_evidence=evidence))
                    write_json(out/'best-weak-validation-private.json',{'best_checkpoint_sha256':file_sha256(out/'best.pt'),'evidence':evidence,'contract':contract})
                    best_checkpoint_sha=file_sha256(out/'best.pt')
                    best_evidence_sha=file_sha256(out/'best-weak-validation-private.json')
                    best=metrics['weak_bce']
                epoch+=1;cursor=0;order=None;save_last()
                progress('train04_epoch_complete',epoch=epoch,weak_bce=metrics['weak_bce'],
                    thresholded_weak_auc=metrics['thresholded_weak_auc'],per_target_bce=metrics['per_target_bce'],
                    confidence_support=metrics['confidence_support'],training_diagnostic=diag,
                    cumulative_training_telemetry=telemetry.report(),seconds=elapsed())
            status='PASS'
    except TimeoutError:
        status='PLATFORM_SESSION_STOP';save_last()
    except BaseException:
        status='FAIL';save_last();raise
    finally:
        write_json(out/'training-report.json',dict(contract=contract,status=status,history=history,
          completed_epochs=epoch,selected_bce=best if math.isfinite(best) else None,
          prevalence_weak_bce=prevalence,elapsed_seconds=elapsed(),cuda_peak_bytes=torch.cuda.max_memory_allocated(),
          successful_updates=step,amp_overflow_skips=overflow_total,initial_encoder_sha256=init_hash,
          zero_supervision_skips=empty_skips,cumulative_training_telemetry=telemetry.report()))
    return status


def train(cache,sources,out,**kwargs):
    """Retain a sanitized receipt even if preparation fails before the fit loop."""
    started=time.monotonic();out=Path(out)
    if out.exists():raise ValueError('Training output already exists')
    try:
        return _train(cache,sources,out,**kwargs)
    except BaseException as exc:
        out.mkdir(parents=True,exist_ok=True)
        path=out/'training-report.json'
        if not path.exists():
            prep=out/'preparation-report.json'
            write_json(path,{'status':'FAIL','phase':'preparation','failure_type':type(exc).__name__,
                'contract':json.loads(prep.read_text())['contract'] if prep.exists() else None,
                'current_attempt_seconds':time.monotonic()-started,'completed_epochs':0,
                'runner_sha256':file_sha256(__file__)})
        raise
