"""One frozen-checkpoint full384 coverage ablation; protected central320 unchanged."""
from collections import Counter
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import time
from knee.reference import canonical_bytes,sha256,validate_predictions
from knee.public_adaptation_fit import (require,fingerprint,atomic_save,preflight_inputs,
    load_binary_truth,metric_report,paired_bootstrap,PlatformInterruption)
from knee.public_adaptation_pilot import file_sha,state_fingerprint

SAMPLE_SHA='1e4cf23b048d1ced8c960ee8035abbfe1f43aba4992feb6c4d42d4949748c414'
PREDICTION_ATOL=1e-6
ARCH='coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
GRID_NAMES=frozenset(f'backbone.stages.{stage}.blocks.{block}.attn.rel_pos.{kind}'
    for stage,n in ((2,14),(3,2)) for block in range(n)
    for kind in ('relative_position_index','rel_coords_log'))


def config():
    return {'control_res':320,'candidate_res':384,'windows':94,'chunk':16,'precision':'CUDA fp16',
            'tta':'equal original/anatomical-mirror probabilities within each arm','optimizer_updates':0,
            'bootstrap_draws':2000,'bootstrap_seed':42,'prediction_atol':PREDICTION_ATOL,
            'input_policy':'identical decoded public384; candidate fullfield; control central320; no resize'}


def centers(public,mask):
    import numpy as np
    return np.array([max(1,min(c,94)) for c in public._eval_centers(mask,96,94)],dtype=np.int64)


def fullfield_windows(public,volume,mask,device='cpu',*,mirror=False):
    """Explicit384 adapter; exact CPU normalization LUT and inherited uint8 mirror."""
    import numpy as np
    import torch
    require(volume.shape==(96,384,384) and volume.dtype==np.uint8 and mask.shape==(96,) and
            np.isin(mask,[0,1]).all() and mask.any(),'FULLFIELD_INPUT')
    cs=centers(public,mask)
    tri=torch.from_numpy(np.ascontiguousarray(volume[np.stack([cs-1,cs,cs+1],1)])).to(device)
    if mirror: tri=public._mirror_tri(tri,cs,96)
    require(tuple(tri.shape)==(94,3,384,384),'FULLFIELD_GEOMETRY')
    v=torch.from_numpy(np.arange(256,dtype=np.uint8).astype(np.float32)/255)
    lut=torch.stack([v,v,v],0).view(3,256,1)
    lut=((lut-public._MEAN.view(3,1,1))/public._STD.view(3,1,1)).view(3,256).contiguous().to(device)
    return torch.stack([lut[c][tri[:,c].long()] for c in range(3)],1)


def qualify_windows(public,volume,mask,device='cuda'):
    import torch
    hashes={}
    for mirror in (False,True):
        cpu=fullfield_windows(public,volume,mask,mirror=mirror)
        gpu=fullfield_windows(public,volume,mask,device,mirror=mirror)
        control_cpu=public.eval_windows(volume,mask,94,320,mirror=mirror)
        control_gpu=public.gpu_windows(volume,mask,94,320,device,mirror=mirror)
        require(torch.equal(cpu,gpu.cpu()),'FULLFIELD_CPU_GPU')
        require(torch.equal(control_cpu,control_gpu.cpu()),'CONTROL_CPU_GPU')
        require(torch.equal(cpu[:,:,32:352,32:352],control_cpu),'CENTRAL_SUBARRAY')
        require(torch.equal(cpu,fullfield_windows(public,volume,mask,mirror=mirror)),'WINDOW_REPEAT')
        hashes['mirrored' if mirror else 'normal']=state_fingerprint(cpu)
        del cpu,gpu,control_cpu,control_gpu
    return hashes


def nonpersistent(model):
    return {f'{prefix}.{name}'.lstrip('.') for prefix,module in model.named_modules()
            for name in module._non_persistent_buffers_set if module._buffers.get(name) is not None}


def snapshot(model):
    return {'state':state_fingerprint(model.state_dict()),
            'buffers':state_fingerprint(dict(model.named_buffers()))}


def require_immutable(model,before):
    require(all(not m.training for m in model.modules()),'EVAL_STATE')
    require(snapshot(model)==before,'PARAMETER_OR_BUFFER_MUTATION')


def validate_grid_pair(control,candidate,oracles):
    """Only prospective32 constructor-derived nonpersistent grids may differ."""
    import torch
    a,b=control.state_dict(),candidate.state_dict()
    require(a.keys()==b.keys() and all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a),'STRICT_STATE')
    ba,bb=dict(control.named_buffers()),dict(candidate.named_buffers())
    require(ba.keys()==bb.keys(),'BUFFER_KEYS')
    differences={k for k in ba if not torch.equal(ba[k].cpu(),bb[k].cpu())}
    require(differences==GRID_NAMES,'GRID_ALLOWLIST')
    require(GRID_NAMES<=nonpersistent(control) and GRID_NAMES<=nonpersistent(candidate),'GRID_PERSISTENCE')
    for model,res in ((control,320),(candidate,384)):
        buffers=dict(model.named_buffers()); oracle=oracles[res]
        require(set(oracle)==GRID_NAMES,'ORACLE_KEYS')
        require(all(torch.equal(buffers[k].cpu(),oracle[k]) for k in GRID_NAMES),'CONSTRUCTOR_GRID_ORACLE')
    return {'state_tensors':len(a),'state_sha256':state_fingerprint(a),'allowed_nonpersistent_differences':32,
            'constructor_oracle_sha256':{str(r):state_fingerprint(o) for r,o in oracles.items()}}


def make_candidate(public,control):
    import timm
    import torch
    require(importlib.metadata.version('timm')=='1.0.26','TIMM_VERSION')
    oracles={}
    for res in (320,384):
        oracle=timm.create_model(ARCH,pretrained=False,num_classes=0,in_chans=3,img_size=res,global_pool='avg')
        oracles[res]={f'backbone.{k}':v.detach().cpu().clone() for k,v in oracle.named_buffers() if f'backbone.{k}' in GRID_NAMES}
        require(len(oracle.state_dict())==478,'BACKBONE_STATE_KEYS')
        del oracle
    backbone=timm.create_model(ARCH,pretrained=False,num_classes=0,in_chans=3,img_size=384,global_pool='avg')
    candidate=public.RaptorClassifier(backbone,F_dim=backbone.num_features)
    candidate.load_state_dict({k:v.detach().cpu() for k,v in control.state_dict().items()},strict=True)
    candidate.eval().to(next(control.parameters()).device)
    grid=validate_grid_pair(control,candidate,oracles)
    return candidate,grid


def prediction_error(actual,expected,check):
    import numpy as np
    a,b=map(np.asarray,(actual,expected))
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),check+'_SCHEMA')
    maximum=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))
    if maximum>PREDICTION_ATOL:
        error=ValueError(check);error.prediction_max_abs=maximum
        raise error
    return maximum


def infer_candidate(public,model,plain,mirrored,device='cuda'):
    require(tuple(plain.shape)==tuple(mirrored.shape)==(94,3,384,384),'CANDIDATE_WINDOWS')
    require(str(device).startswith('cuda') and all(not m.training for m in model.modules()),'CANDIDATE_CUDA_EVAL')
    p=0.5*(public.infer_probs(model,plain,device)+public.infer_probs(model,mirrored,device))
    return validate_predictions(p)


def paired_inference(public,control,candidate,volume,mask):
    import numpy as np
    before=sha256(volume.tobytes())
    x=[public.gpu_windows(volume,mask,94,320,'cuda',mirror=m) for m in (False,True)]
    control_p=validate_predictions(public.infer_pair_probs(control,*x,'cuda')).astype(np.float32)
    del x
    x=[fullfield_windows(public,volume,mask,'cuda',mirror=m) for m in (False,True)]
    candidate_p=np.asarray(infer_candidate(public,candidate,*x),dtype=np.float32)
    del x
    require(before==sha256(volume.tobytes()),'VOLUME_IMMUTABILITY')
    return control_p,candidate_p


def decision(metrics,bootstrap):
    c,e=metrics['control'],metrics['candidate']; delta=e['macro_auc']-c['macro_auc']
    damage=[t for t,a in e['targets'].items() if a['auc'] is not None and c['targets'][t]['auc'] is not None and
            a['positive']>=20 and a['negative']>=20 and a['auc']-c['targets'][t]['auc']<=-.01]
    ci=bootstrap['ci95']
    triage='reject; method closed' if delta<=0 or len(damage)>=3 else 'inconclusive; no expansion'
    if delta>0 and len(damage)<3 and ci is not None and ci[0]>0:
        triage='eligible for fixed second-fold replication and image-only rehearsal'
    return {'macro_delta':delta,'target_damage':damage,'triage':triage}


def analyze(y,m,train_y,train_m,control,candidate,groups,targets):
    import numpy as np
    import torch
    prevalence=np.divide((train_y*train_m).sum(0),train_m.sum(0),out=np.full(len(targets),np.nan),where=train_m.sum(0)>0)
    require(np.isfinite(prevalence).all(),'PREVALENCE_DEFINED')
    metrics={name:metric_report(y,m,p,targets) for name,p in [('control',control),('candidate',candidate),('train_prevalence',np.broadcast_to(prevalence,y.shape))]}
    bootstrap=paired_bootstrap(y,m,control,candidate,groups,draws=2000,seed=42,targets=targets)
    return {'metrics':metrics,'paired_bootstrap':bootstrap,**decision(metrics,bootstrap),
            'truth_array_sha256':state_fingerprint({'y':torch.from_numpy(y),'mask':torch.from_numpy(m)}),
            'train_prevalence_fit':{t:{'probability':float(prevalence[j]),'observed':int(train_m[:,j].sum()),
                'positive':int((train_y[:,j]*train_m[:,j]).sum()),'negative':int(((1-train_y[:,j])*train_m[:,j]).sum())} for j,t in enumerate(targets)}}


def run(public_source,decoder_source,manifest,checkpoint,labels,masks,root,out,
        folds,excluded,preflight,sample,binding,saved_control):
    import numpy as np
    import torch
    from knee.reference import load_definitions,load_model
    from knee.mirror_input import prepare_study,stress_inputs
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();stage='setup'
    report={'status':'RUNNING','config':config(),'binding':binding,'optimizer_updates':0,'training':False,
            'generalization_evidence':False,'qualification_completed':0,'evaluated_studies':0,'free_disk_before':shutil.disk_usage(out).free,
            'qualification':[],'decode_counters':{},'max_prediction_errors':{},'intermediate_metrics_opened':False}
    def save():
        report.update(elapsed_seconds=time.monotonic()-started,free_disk_now=shutil.disk_usage(out).free,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        if torch.cuda.is_available(): report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        atomic_save(out/'receipt.json',report,lambda value,h:h.write((json.dumps(value,indent=2,allow_nan=False)+'\n').encode()))
    save()
    try:
        require(sys.version_info[:2]==(3,12) and torch.cuda.is_available() and torch.__version__=='2.10.0+cu128','CUDA_ENVIRONMENT')
        require(binding['config']==config() and binding['preflight_sha256']==fingerprint(preflight) and
                binding['source_bundle_sha256']==os.environ.get('KNEE_SOURCE_SHA256') and
                binding['public_manifest_sha256']==fingerprint(manifest),'RUN_BINDING')
        require(file_sha(root/'train.csv')==binding['official_train_sha256'] and
                file_sha(root/'train_series.csv')==preflight['source_sha256']['data/raw/train_series.csv'],'OFFICIAL_INPUTS')
        require(fingerprint(sample)==SAMPLE_SHA==binding['sample_sha256'],'SAMPLE_BINDING')
        require(file_sha(labels)==binding['label_sha256'] and file_sha(masks)==binding['truth_sha256'],'LABEL_TRUTH_BINDING')
        report['environment']={'python':sys.version,'torch':torch.__version__,'timm':importlib.metadata.version('timm'),
            'numpy':np.__version__,'pyarrow':importlib.metadata.version('pyarrow'),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)}
        torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
        public,decoder=load_definitions(public_source,decoder_source,manifest)
        stage='sample_membership'
        require(len(sample)==32 and len(set(sample))==32 and all(s in folds and folds[s]!=0 and s not in excluded for s in sample),'SAMPLE_TRAIN_ONLY')
        # Presence-only gold protection before mechanics; report values, weak-label
        # magnitudes and report-component text are not read by this stage.
        with (root/'train.csv').open(newline='') as handle:
            gold={row['StudyInstanceUID'] for row in csv.DictReader(handle) if all(row[t].strip() for t in public.LAB)}
        require(not set(sample)&gold,'SAMPLE_GOLD_PROTECTION')
        require(saved_control.shape==(870,12) and saved_control.dtype==np.float32 and np.isfinite(saved_control).all(), 'SAVED_CONTROL_SCHEMA')
        require(sha256(saved_control.tobytes())==binding['saved_control_array_sha256'],'SAVED_CONTROL_BINDING')
        atomic_save(out/'population-private.json',{'qualification_study_ids':sample},
                    lambda value,h:h.write((json.dumps(value)+'\n').encode()))
        report['synthetic_decoder_stress']=stress_inputs(public,decoder)
        records={}
        with (root/'train_series.csv').open(newline='') as handle:
            for row in csv.DictReader(handle): records.setdefault(row['StudyInstanceUID'],[]).append(row)
        stage='models';control=load_model(public,checkpoint,'cuda',manifest);candidate,grid=make_candidate(public,control)
        report['state_grid_gate']=grid;before=[snapshot(m) for m in (control,candidate)]
        serialized=out/'frozen-state.pt';atomic_save(serialized,{k:v.detach().cpu() for k,v in control.state_dict().items()})
        report['serialized_state_sha256']=file_sha(serialized)
        torch.cuda.reset_peak_memory_stats();counters=Counter();prediction_errors=Counter()
        def read(sid,reference=False):
            v,m,counts=prepare_study(public,decoder,sid,records,root/'train_series',reference=reference)
            counters.update({k:v for k,v in counts.items() if isinstance(v,int)})
            require(v.shape==(96,384,384) and v.dtype==np.uint8 and m.shape==(96,) and m.any(),'DECODE_SCHEMA')
            return v,m
        stage='qualification';qualification_started=time.monotonic();qualification_pair_seconds=[]
        with PlatformInterruption() as stop:
            for ordinal,sid in enumerate(sample):
                v,m=read(sid);rv,rm=read(sid,True)
                require(np.array_equal(v,rv) and np.array_equal(m,rm),'REFERENCE_FAST_PIXELS_MASK')
                del rv,rm
                hashes=qualify_windows(public,v,m)
                torch.cuda.synchronize();tick=time.monotonic()
                p=paired_inference(public,control,candidate,v,m)
                torch.cuda.synchronize();qualification_pair_seconds.append(time.monotonic()-tick)
                repeat=paired_inference(public,control,candidate,v,m)
                for arm,a,b in zip(('control','candidate'),p,repeat):
                    err=prediction_error(a,b,'QUALIFICATION_REPEAT_'+arm.upper());prediction_errors[arm+'_repeat']=max(prediction_errors[arm+'_repeat'],err)
                # Every32 image probe has strict serialized reload, with its own constructor grid retained.
                state=torch.load(serialized,map_location='cpu',weights_only=True)
                for model in (control,candidate):model.load_state_dict(state,strict=True);model.eval()
                del state
                reload=paired_inference(public,control,candidate,v,m)
                for arm,a,b in zip(('control','candidate'),p,reload):
                    err=prediction_error(a,b,'QUALIFICATION_RELOAD_'+arm.upper());prediction_errors[arm+'_reload']=max(prediction_errors[arm+'_reload'],err)
                for model,snap in zip((control,candidate),before):require_immutable(model,snap)
                report['qualification'].append({'ordinal':ordinal,'volume_sha256':sha256(v.tobytes()),'mask_sha256':sha256(m.tobytes()),'windows':hashes})
                report['qualification_completed']+=1;report['decode_counters']=dict(counters);report['max_prediction_errors']=dict(prediction_errors);save()
                del v,m,p,repeat,reload
                if stop.requested:report.update(status='INTERRUPTED',failure_stage=stage,platform_signal=stop.signum);save();return report
            report['qualification_seconds']=time.monotonic()-qualification_started
            report['estimated_evaluation_seconds']=sum(qualification_pair_seconds)/32*870
            report['runtime_estimate_policy']='measured initial paired CUDA forwards on32 probes; excludes decoding/setup/metrics'
            report['mechanical_gate']='PASS';save()
            stage='input_preflight'
            train,dev,_,_,_,groups,grouping=preflight_inputs(root,labels,masks,folds,excluded,preflight,public.LAB)
            require(set(sample)<=set(train) and not set(sample)&set(dev),'QUALIFIED_MEMBERSHIP')
            require(fingerprint(dev)==binding['dev_order_sha256'] and len(dev)==870,'DEV_ORDER')
            report['grouping']=grouping
            atomic_save(out/'population-private.json',{'qualification_study_ids':sample,'development_study_ids':dev,'groups':groups},
                        lambda value,h:h.write((json.dumps(value)+'\n').encode()))
            stage='evaluation';evaluation_started=time.monotonic();pred_c=[];pred_e=[]
            for ordinal,sid in enumerate(dev):
                v,m=read(sid);c,e=paired_inference(public,control,candidate,v,m)
                error=prediction_error(c,saved_control[ordinal],'RECOVERY_CONTROL_PARITY')
                prediction_errors['saved_control']=max(prediction_errors['saved_control'],error)
                pred_c.append(c);pred_e.append(e)
                # Save every consumed boundary before any metrics or platform stop.
                atomic_save(out/'predictions-private.npz',None,lambda _,h:np.savez(h,study_ids=np.asarray(dev[:ordinal+1]),
                    control=np.asarray(pred_c,dtype=np.float32),candidate=np.asarray(pred_e,dtype=np.float32)))
                report['evaluated_studies']=ordinal+1;report['max_prediction_errors']=dict(prediction_errors);report['decode_counters']=dict(counters);save()
                del v,m
                if stop.requested:report.update(status='INTERRUPTED',failure_stage=stage,platform_signal=stop.signum);save();return report
        for model,snap in zip((control,candidate),before):require_immutable(model,snap)
        report['final_state_and_all_buffers_immutable']=True;report['evaluation_seconds']=time.monotonic()-evaluation_started
        report['predictions_sha256']=file_sha(out/'predictions-private.npz');save()
        stage='analysis';y,m=load_binary_truth(masks,dev,public.LAB);ty,tm=load_binary_truth(masks,train,public.LAB)
        report.update(status='PASS',**analyze(y,m,ty,tm,np.asarray(pred_c),np.asarray(pred_e),groups,public.LAB))
        serialized.unlink();report['storage_bytes']=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    except BaseException as error:
        report.update(status='FAIL',failure_stage=stage,failure_type=type(error).__name__)
        if hasattr(error,'prediction_max_abs'):report['failure_prediction_max_abs']=error.prediction_max_abs
        save();raise RuntimeError('Fullfield ablation failed; inspect sanitized receipt') from None
    finally:save()
    return report
