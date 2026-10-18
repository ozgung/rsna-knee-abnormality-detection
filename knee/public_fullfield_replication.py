"""Fixed fold1 replication of qualified full384 versus original central320.

This module does not call or modify the fold0 runner/preflight. It reuses its
qualified image/grid/state/analysis primitives with an independently bound866
population and same-job original-source reference on32 training probes.
"""
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
from knee.public_adaptation_fit import (require,fingerprint,atomic_save,report_components,
    load_binary_truth,PlatformInterruption)
from knee.public_adaptation_pilot import file_sha,load_labels
from knee.public_fullfield import (config as inherited_config,qualify_windows,make_candidate,
    snapshot,require_immutable,paired_inference,prediction_error,analyze,decision)

HOLDOUT=1
ELIGIBLE=4345
DEV_COUNT=866
TRAIN_COUNT=3479
SAMPLE_PREFIX='fullfield-fold1-mechanics-v1:'
FOLD0_RECEIPT_SHA='6213938a94b39e08cbe17e5fb42e0b3dcb0e8dc49354a09ed73151049eea517c'
FOLD0_NOTEBOOK_SHA='553760d78cc45dae5a240bfdd819f126a976da8e34efb69163381e3d297fda43'


def config():
    return {**inherited_config(),'holdout_fold':HOLDOUT,'development_studies':DEV_COUNT,
            'mechanics_sample_policy':SAMPLE_PREFIX+'UID; SHA256 ascending; first32 training-only nongold/nonquarantine',
            'control_provenance':'same-job original-source central320 parity on all32 mechanics probes; no historical fold1 saved control'}


def membership(rows,label_ids,folds,excluded,targets):
    """Exact inherited eligibility, with explicit fold1 and866 rather than fold0."""
    ids=[r['StudyInstanceUID'] for r in rows]
    require(len(ids)==4407 and len(set(ids))==len(ids),'OFFICIAL_MEMBERSHIP')
    require(all(t in rows[0] for t in targets) and len(targets)==12 and len(set(targets))==12,'TARGET_SCHEMA')
    gold={r['StudyInstanceUID'] for r in rows if all(r[t].strip() for t in targets)}
    require(len(gold)==58,'GOLD_MEMBERSHIP')
    eligible=sorted(set(ids)&set(label_ids)-gold-set(excluded))
    require(len(eligible)==ELIGIBLE and set(eligible)<=folds.keys(),'ELIGIBLE_MEMBERSHIP')
    require(all(type(folds[s]) is int and 0<=folds[s]<5 for s in eligible),'FOLD_VALUES')
    train=[s for s in eligible if folds[s]!=HOLDOUT];dev=[s for s in eligible if folds[s]==HOLDOUT]
    require((len(train),len(dev))==(TRAIN_COUNT,DEV_COUNT),'FOLD1_MEMBERSHIP')
    require(not set(train)&set(dev) and not set(eligible)&gold,'POPULATION_LEAKAGE')
    return eligible,train,dev,gold


def freeze_sample(rows,folds,excluded,targets):
    """Only presence of official gold fields; no weak-label values or scores."""
    ids=[r['StudyInstanceUID'] for r in rows];require(len(set(ids))==len(ids),'OFFICIAL_DUPLICATES')
    gold={r['StudyInstanceUID'] for r in rows if all(r[t].strip() for t in targets)}
    pool=set(ids)&folds.keys()-gold-set(excluded)
    pool={s for s in pool if type(folds[s]) is int and 0<=folds[s]<5 and folds[s]!=HOLDOUT}
    require(len(pool)>=32,'SAMPLE_POOL')
    return sorted(pool,key=lambda s:(sha256((SAMPLE_PREFIX+s).encode()),s))[:32]


def construct_preflight(rows,label_ids,folds,excluded,targets,masks):
    eligible,train,dev,_=membership(rows,label_ids,folds,excluded,targets)
    require(set(masks)==set(eligible),'MASK_MEMBERSHIP')
    require(all(len(masks[s])==12 and all(v in (0,1) for v in masks[s]) for s in eligible),'MASK_SCHEMA')
    selected=[[s,masks[s].tolist()] for s in eligible]
    groups,grouping=report_components(rows,train,dev)
    support={t:{'train_observed':int(sum(masks[s][j] for s in train)),
                'dev_observed':int(sum(masks[s][j] for s in dev))} for j,t in enumerate(targets)}
    pre={'status':'FOLD1_PREFLIGHT_PASS','holdout_fold':HOLDOUT,'targets':list(targets),
         'counts':{'eligible':len(eligible),'train':len(train),'development':len(dev),'gold':58},
         'fingerprints':{'eligible':fingerprint(eligible),'train':fingerprint(train),'dev':fingerprint(dev),
                         'fold_mapping':fingerprint(folds),'selected_masks_in_target_order':fingerprint(selected),'groups':fingerprint(groups)},
         'support':support,'grouping':grouping}
    return pre,train,dev,groups


def preflight_inputs(root,labels,masks,folds,excluded,expected,targets):
    import pyarrow.parquet as pq
    with (Path(root)/'train.csv').open(newline='') as h:rows=list(csv.DictReader(h))
    table=pq.read_table(labels,columns=['StudyInstanceUID']);label_ids=table.column(0).to_pylist()
    require(len(label_ids)==len(set(label_ids)),'LABEL_DUPLICATE_IDS')
    eligible,_,_,_=membership(rows,label_ids,folds,excluded,targets)
    _,ms=load_labels(labels,masks,eligible,targets)
    actual,train,dev,groups=construct_preflight(rows,label_ids,folds,excluded,targets,ms)
    require(actual==expected,'FOLD1_PREFLIGHT_BINDING')
    return train,dev,groups


def validate_fold0(receipt,manifest):
    require(receipt['status']=='PASS' and receipt['optimizer_updates']==0 and receipt['qualification_completed']==32 and
            receipt['evaluated_studies']==870 and receipt['mechanical_gate']=='PASS' and
            receipt['final_state_and_all_buffers_immutable'] is True,'FOLD0_RUNTIME_QUALIFICATION')
    require(receipt['binding']['public_manifest_sha256']==fingerprint(manifest) and
            receipt['binding']['config']==inherited_config(),'FOLD0_CONTRACT')
    require(receipt['metrics']['control']['defined_targets']==receipt['metrics']['candidate']['defined_targets']==12,'FOLD0_DEFINED_TARGETS')
    require(receipt['macro_delta']>0 and receipt['paired_bootstrap']['ci95'][0]>0 and len(receipt['target_damage'])<3,'FOLD0_REPLICATION_GATE')


def reference_control(public,model,volume,mask):
    """Untouched original CPU eval_windows and infer_pair_probs, same fp16 model."""
    plain=public.eval_windows(volume,mask,94,320,mirror=False)
    mirrored=public.eval_windows(volume,mask,94,320,mirror=True)
    return validate_predictions(public.infer_pair_probs(model,plain,mirrored,'cuda'))


def save_predictions(path,ids,control,candidate):
    import numpy as np
    c,e=np.asarray(control,dtype=np.float32),np.asarray(candidate,dtype=np.float32)
    require(c.shape==e.shape==(len(ids),12) and len(set(ids))==len(ids) and len(ids)>0,'PREDICTION_COVERAGE')
    require(np.isfinite(c).all() and np.isfinite(e).all() and ((c>=0)&(c<=1)).all() and ((e>=0)&(e<=1)).all(),'PREDICTION_VALUES')
    atomic_save(path,None,lambda _,h:np.savez(h,study_ids=np.asarray(ids),control=c,candidate=e))


def replication_analysis(y,m,ty,tm,control,candidate,groups,targets,preflight):
    require(len(targets)==12 and y.shape==m.shape==(DEV_COUNT,12),'ANALYSIS866_SCHEMA')
    require(all(int(m[:,j].sum())==preflight['support'][t]['dev_observed'] for j,t in enumerate(targets)),'ANALYSIS_SUPPORT')
    result=analyze(y,m,ty,tm,control,candidate,groups,targets)
    require(result['metrics']['control']['defined_targets']==result['metrics']['candidate']['defined_targets']==12,'ALL12_TARGETS_REQUIRED')
    # A replication pass advances only to a fixed image-only offline rehearsal.
    if result['macro_delta']>0 and len(result['target_damage'])<3 and result['paired_bootstrap']['ci95'] is not None and result['paired_bootstrap']['ci95'][0]>0:
        result['triage']='eligible for image-only offline rehearsal'
    else:result['triage']='replication closed; no expansion'
    result['generalization_evidence']=False
    return result


def run(public_source,decoder_source,manifest,checkpoint,labels,masks,root,out,folds,excluded,preflight,sample,binding,fold0):
    import numpy as np
    import torch
    from knee.reference import load_definitions,load_model
    from knee.mirror_input import prepare_study,stress_inputs
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();stage='setup'
    report={'status':'RUNNING','config':config(),'binding':binding,'optimizer_updates':0,'training':False,'holdout_fold':1,
            'generalization_evidence':False,'qualification_completed':0,'evaluated_studies':0,'qualification':[],
            'decode_counters':{},'max_prediction_errors':{},'intermediate_metrics_opened':False,'historical_fold1_control':'unavailable; not fabricated'}
    def save():
        report.update(elapsed_seconds=time.monotonic()-started,free_disk_now=shutil.disk_usage(out).free,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        if torch.cuda.is_available():report['peak_cuda_bytes']=torch.cuda.max_memory_allocated()
        atomic_save(out/'receipt.json',report,lambda v,h:h.write((json.dumps(v,indent=2,allow_nan=False)+'\n').encode()))
    save()
    try:
        require(sys.version_info[:2]==(3,12) and torch.cuda.is_available() and torch.__version__=='2.10.0+cu128','CUDA_ENVIRONMENT')
        require(binding['config']==config() and fingerprint(preflight)==binding['preflight_sha256'] and
                binding['source_bundle_sha256']==os.environ.get('KNEE_SOURCE_SHA256') and
                binding['public_manifest_sha256']==fingerprint(manifest),'RUN_BINDING')
        require(file_sha(root/'train.csv')==binding['official_train_sha256'] and
                file_sha(root/'train_series.csv')==binding['series_sha256'] and
                file_sha(root/'sample_submission.csv')==binding['official_schema_sha256'],'OFFICIAL_INPUTS')
        require(file_sha(labels)==binding['label_sha256'] and file_sha(masks)==binding['truth_sha256'],'LABEL_TRUTH_BINDING')
        require(fingerprint(folds)==binding['fold_mapping_sha256'] and fingerprint(excluded)==binding['exclusions_sha256'],'MEMBERSHIP_BINDING')
        require(fingerprint(fold0)==binding['fold0_receipt_fingerprint'],'FOLD0_RECEIPT_BINDING');validate_fold0(fold0,manifest)
        public,decoder=load_definitions(public_source,decoder_source,manifest)
        with (root/'sample_submission.csv').open(newline='') as h:header=next(csv.reader(h))
        require(header==['StudyInstanceUID']+public.LAB,'OFFICIAL_SCHEMA')
        with (root/'train.csv').open(newline='') as h:rows=list(csv.DictReader(h))
        require(sample==freeze_sample(rows,folds,excluded,public.LAB) and fingerprint(sample)==binding['sample_sha256'],'FROZEN32_TRAIN_ONLY')
        atomic_save(out/'population-private.json',{'qualification_study_ids':sample},lambda v,h:h.write(json.dumps(v).encode()))
        report['environment']={'python':sys.version,'torch':torch.__version__,'timm':importlib.metadata.version('timm'),
            'numpy':np.__version__,'pyarrow':importlib.metadata.version('pyarrow'),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)}
        torch.set_num_threads(4);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
        report['synthetic_decoder_stress']=stress_inputs(public,decoder)
        records={}
        with (root/'train_series.csv').open(newline='') as h:
            for row in csv.DictReader(h):records.setdefault(row['StudyInstanceUID'],[]).append(row)
        stage='models';control=load_model(public,checkpoint,'cuda',manifest);candidate,grid=make_candidate(public,control)
        report['state_grid_gate']=grid;before=[snapshot(m) for m in (control,candidate)]
        serialized=out/'frozen-state.pt';atomic_save(serialized,{k:v.detach().cpu() for k,v in control.state_dict().items()})
        report['serialized_state_sha256']=file_sha(serialized)
        torch.cuda.reset_peak_memory_stats();counters=Counter();errors=Counter()
        def read(sid,reference=False):
            v,m,counts=prepare_study(public,decoder,sid,records,root/'train_series',reference=reference)
            counters.update({k:v for k,v in counts.items() if isinstance(v,int)})
            require(v.shape==(96,384,384) and v.dtype==np.uint8 and m.shape==(96,) and m.any(),'DECODE_SCHEMA')
            return v,m
        stage='qualification';qual_started=time.monotonic();pair_seconds=[];qual_control=[];qual_candidate=[];qual_reference=[]
        with PlatformInterruption() as stop:
            for ordinal,sid in enumerate(sample):
                v,m=read(sid);rv,rm=read(sid,True)
                require(np.array_equal(v,rv) and np.array_equal(m,rm),'REFERENCE_FAST_PIXELS_MASK')
                hashes=qualify_windows(public,v,m)
                original=reference_control(public,control,rv,rm);del rv,rm
                torch.cuda.synchronize();tick=time.monotonic();pair=paired_inference(public,control,candidate,v,m)
                torch.cuda.synchronize();pair_seconds.append(time.monotonic()-tick)
                err=prediction_error(pair[0],original,'ORIGINAL_REFERENCE320_PARITY');errors['original_reference320']=max(errors['original_reference320'],err)
                repeat=paired_inference(public,control,candidate,v,m)
                for arm,a,b in zip(('control','candidate'),pair,repeat):
                    err=prediction_error(a,b,'QUALIFICATION_REPEAT_'+arm.upper());errors[arm+'_repeat']=max(errors[arm+'_repeat'],err)
                state=torch.load(serialized,map_location='cpu',weights_only=True)
                for model in (control,candidate):model.load_state_dict(state,strict=True);model.eval()
                del state
                reload=paired_inference(public,control,candidate,v,m)
                for arm,a,b in zip(('control','candidate'),pair,reload):
                    err=prediction_error(a,b,'QUALIFICATION_RELOAD_'+arm.upper());errors[arm+'_reload']=max(errors[arm+'_reload'],err)
                for model,snap in zip((control,candidate),before):require_immutable(model,snap)
                qual_control.append(pair[0]);qual_candidate.append(pair[1]);qual_reference.append(original)
                save_predictions(out/'qualification-predictions-private.npz',sample[:ordinal+1],qual_control,qual_candidate)
                atomic_save(out/'qualification-reference-private.npz',None,lambda _,h:np.savez(h,study_ids=np.asarray(sample[:ordinal+1]),reference320=np.asarray(qual_reference,dtype=np.float32)))
                report['qualification'].append({'ordinal':ordinal,'volume_sha256':sha256(v.tobytes()),'mask_sha256':sha256(m.tobytes()),'windows':hashes})
                report['qualification_completed']=ordinal+1;report['decode_counters']=dict(counters);report['max_prediction_errors']=dict(errors);save()
                del v,m,pair,repeat,reload,original
                if stop.requested:report.update(status='INTERRUPTED',failure_stage=stage,platform_signal=stop.signum);save();return report
            report['qualification_seconds']=time.monotonic()-qual_started;report['estimated_evaluation_seconds']=sum(pair_seconds)/32*DEV_COUNT
            report['runtime_estimate_policy']='measured initial paired CUDA forwards on32 probes; excludes decode/setup/metrics';report['mechanical_gate']='PASS';save()
            stage='input_preflight';train,dev,groups=preflight_inputs(root,labels,masks,folds,excluded,preflight,public.LAB)
            require(set(sample)<=set(train) and not set(sample)&set(dev) and fingerprint(dev)==binding['dev_order_sha256'],'QUALIFIED_MEMBERSHIP')
            report['grouping']=preflight['grouping']
            atomic_save(out/'population-private.json',{'qualification_study_ids':sample,'development_study_ids':dev,'groups':groups},lambda v,h:h.write(json.dumps(v).encode()))
            stage='evaluation';evaluation_started=time.monotonic();pred_c=[];pred_e=[]
            for ordinal,sid in enumerate(dev):
                v,m=read(sid);c,e=paired_inference(public,control,candidate,v,m);pred_c.append(c);pred_e.append(e)
                save_predictions(out/'predictions-private.npz',dev[:ordinal+1],pred_c,pred_e)
                report['evaluated_studies']=ordinal+1;report['decode_counters']=dict(counters);save();del v,m
                if stop.requested:report.update(status='INTERRUPTED',failure_stage=stage,platform_signal=stop.signum);save();return report
        for model,snap in zip((control,candidate),before):require_immutable(model,snap)
        report['final_state_and_all_buffers_immutable']=True;report['evaluation_seconds']=time.monotonic()-evaluation_started
        report['predictions_sha256']=file_sha(out/'predictions-private.npz');save()
        stage='analysis'
        # Re-read completed atomic arrays; metrics consume the preserved artifact.
        with np.load(out/'predictions-private.npz',allow_pickle=False) as archive:
            require(archive['study_ids'].tolist()==dev,'SAVED_PREDICTION_ORDER');c,e=archive['control'].copy(),archive['candidate'].copy()
        y,m=load_binary_truth(masks,dev,public.LAB);ty,tm=load_binary_truth(masks,train,public.LAB)
        report.update(status='PASS',**replication_analysis(y,m,ty,tm,c,e,groups,public.LAB,preflight))
        serialized.unlink();report['storage_bytes']=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    except BaseException as error:
        report.update(status='FAIL',failure_stage=stage,failure_type=type(error).__name__)
        if hasattr(error,'prediction_max_abs'):report['failure_prediction_max_abs']=error.prediction_max_abs
        save();raise RuntimeError('Fullfield fold1 replication failed; inspect sanitized receipt') from None
    finally:save()
    return report
