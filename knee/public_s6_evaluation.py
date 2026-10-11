"""One fixed870 image-only S6/old-central320 rank comparison; no training.

S6 executes the qualified released source unchanged in a fresh process. Native
scratch is removed by CStudies.__getitem__ at every study boundary. The staged
root contains ID-only membership, selected metadata and raw-image symlinks.
Complete raw predictions are preserved before any truth or scoring is opened.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from knee.public_s6_qualification import (LABELS,RECIPE,UID,SERIES_COLUMNS,TRAIN_SHA,
    canonical,digest,file_sha,require,save,verify_sources,verify_assets,verify_staged,synthetic_source_checks)
from knee.public_adaptation_fit import (fingerprint,atomic_save,PlatformInterruption,preflight_inputs,
    load_binary_truth,metric_report,paired_bootstrap)
from knee.public_adaptation_pilot import state_fingerprint

COUNT=870


def environment(torch):
    import importlib.metadata
    versions={}
    for name in ('numpy','pandas','timm','pydicom','scipy','pyarrow','setuptools'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {'pid':os.getpid(),'python':sys.version,'torch':torch.__version__,'cpu_count':os.cpu_count(),
            'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),
            'cuda_device_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'selected_device':'cuda:0','multi_gpu':False,'packages':versions}


def run_child(args,env):
    """Forward platform stops; allow the child to checkpoint a study boundary."""
    import signal
    with PlatformInterruption() as stop:
        child=subprocess.Popen(args,env=env)
        forwarded=False
        while child.poll() is None:
            if stop.requested and not forwarded:
                child.send_signal(stop.signum or signal.SIGTERM);forwarded=True
            time.sleep(.5)
        if stop.requested:raise KeyboardInterrupt('Platform interruption; no automatic resume')
        if child.returncode:raise subprocess.CalledProcessError(child.returncode,args)
CONTROL_SENTINELS=(0,1,868,869)
PILOT_NOTEBOOK_SHA='0214bfac025f9b26415f544aaca7b70e3291d978a364891eef3ee5189fb6d8a4'
PILOT_RECEIPT_SHA='26a7b4797530c850d1facaac79b7409ce721678f7981b8fc8a3915724810fefd'
COVERAGE_SHA='8b4e791432f0f39ae72fd02e924018fa4fe8853d916b1797c671e30a73182166'
CONTROL_RECEIPT_SHA='6213938a94b39e08cbe17e5fb42e0b3dcb0e8dc49354a09ed73151049eea517c'
CONTROL_PREDICTIONS_SHA='eb005cb64349ca7e65ffa6d1fd7a51a7f5bd06d8460596f453c66eb2866e910b'
CONTROL_POPULATION_SHA='c1c4bfecefb9b3527b7fbae6d23c3665fb1f83ac3b14d3e4858f8a2f443cabfd'


def config():
    return {'studies':COUNT,'s6':RECIPE,'control':'original central320 K94 anatomical-mirror; full384 excluded',
            'blend':[.7,.3],'rank':'pandas rank(method=average,pct=True) on all870 rows per target',
            's6_quantization':'source CSV %.6f parsed as float64','optimizer_updates':0,'bootstrap_draws':2000,'bootstrap_seed':42,
            'sentinels':list(CONTROL_SENTINELS),'prediction_atol':1e-6,'resume':'disabled; preserve failure and diagnose'}


def qualification_gate(pilot,coverage):
    require(pilot['source_state']==pilot['mechanics']==pilot['prediction']=='PASS' and pilot['completed_studies']==32 and
            pilot['optimizer_updates']==0 and pilot['zero_imputed'] and pilot['zero_failed'] and pilot['fp32_recoveries']==0 and
            pilot['recipe']==RECIPE,'PILOT_MECHANICAL_GATE')
    require(coverage['verdict']=='PASS_GROSS_VISIBLE_UNION_SCREEN' and coverage['development_evaluation_planning_allowed'] is True,'PILOT_ANATOMICAL_GATE')


def validate_population(ids,folds,excluded,groups,preflight):
    require(len(ids)==COUNT and len(set(ids))==COUNT and len(groups)==COUNT,'POPULATION870_COVERAGE')
    require(all(s in folds and folds[s]==0 and s not in excluded for s in ids),'ORIGINAL_FOLD0_MEMBERSHIP')
    require(fingerprint(ids)==preflight['fingerprints']['dev'] and fingerprint(folds)==preflight['fingerprints']['fold_mapping'],'ORIGINAL_POPULATION_BINDING')
    require(preflight['targets']==LABELS and all(str(g).strip() for g in groups),'TARGET_GROUP_SCHEMA')


def stage_studies(raw,staged,ids,series_sha):
    """Generic874 IDs; no native cache allocation, report exposure or target read."""
    raw,staged=Path(raw),Path(staged)
    require(not staged.exists() and len(ids)==len(set(ids)) and all(s and '/' not in s and '\\' not in s and s not in ('.','..') for s in ids),'FRESH_STAGE_MEMBERSHIP')
    require(file_sha(raw/'train.csv')==TRAIN_SHA and file_sha(raw/'train_series.csv')==series_sha,'OFFICIAL_STAGING_BYTES')
    wanted=set(ids);rows=[]
    with (raw/'train_series.csv').open(newline='') as h:
        r=csv.DictReader(h);require(set(SERIES_COLUMNS)<=set(r.fieldnames),'SERIES_SCHEMA')
        rows=[{k:x[k] for k in SERIES_COLUMNS} for x in r if x[UID] in wanted]
    pairs=[(r[UID],r['SeriesInstanceUID']) for r in rows]
    require({u for u,_ in pairs}==wanted and len(pairs)==len(set(pairs)),'STAGED_SERIES_COVERAGE')
    require(all(s and '/' not in s and '\\' not in s and s not in ('.','..') for _,s in pairs),'SERIES_PATH_SCHEMA')
    staged.mkdir(parents=True);(staged/'train_series').mkdir()
    with (staged/'train.csv').open('w',newline='') as h:
        w=csv.writer(h);w.writerow([UID]);w.writerows([[s] for s in ids])
    with (staged/'train_series.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=SERIES_COLUMNS);w.writeheader();w.writerows(rows)
    inventory=[]
    for u,s in pairs:
        source=raw/'train_series'/u/s;require(source.is_dir() and source.resolve().is_relative_to((raw/'train_series'/u).resolve()) and any(source.glob('*.dcm')),'SELECTED_IMAGE_PATH')
        p=staged/'train_series'/u/s;p.parent.mkdir(exist_ok=True);p.symlink_to(source.resolve(),target_is_directory=True);inventory.append([u,s,str(source.resolve())])
    result={'train_sha256':file_sha(staged/'train.csv'),'series_sha256':file_sha(staged/'train_series.csv'),'membership_sha256':digest(ids),
            'series_membership_sha256':digest(pairs),'image_paths_sha256':digest(inventory),'studies':len(ids),'series':len(rows)}
    verify_staged(staged,ids,result);return result


def array_sha(a):
    import numpy as np
    a=np.ascontiguousarray(a);h=hashlib.sha256(canonical([str(a.dtype),list(a.shape)]));h.update(a.tobytes());return h.hexdigest()


def model_digest(models):
    """Byte-identical to accepted pilot: all named parameters and buffers."""
    h=hashlib.sha256()
    for index,m in enumerate(models):
        tensors={'parameter/'+n:t for n,t in m.named_parameters()};tensors.update({'buffer/'+n:t for n,t in m.named_buffers()})
        for n,t in sorted(tensors.items()):
            a=t.detach().cpu().contiguous().numpy();h.update(canonical([index,n,str(a.dtype),list(a.shape)]));h.update(a.tobytes())
    return h.hexdigest()


def save_predictions(path,ids,logits,probabilities):
    import numpy as np
    z,p=np.asarray(logits,dtype=np.float32),np.asarray(probabilities,dtype=np.float64)
    require(len(ids)>0 and len(ids)==len(set(ids)) and z.shape==(len(ids),4,12) and p.shape==(len(ids),12),'PREDICTION_LEDGER_SCHEMA')
    require(np.isfinite(z).all() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'PREDICTION_LEDGER_VALUES')
    atomic_save(path,None,lambda _,h:np.savez(h,study_ids=np.asarray(ids),logits=z,probabilities=p))


def fixed_ranks(control,s6):
    """Quantize S6 exactly like source CSV; never inspect labels/masks here."""
    import numpy as np
    import pandas as pd
    c,s=np.asarray(control),np.asarray(s6)
    require(c.shape==s.shape==(COUNT,12) and np.isfinite(c).all() and np.isfinite(s).all(),'COMPLETE870_ARMS_REQUIRED')
    require(((c>=0)&(c<=1)).all() and ((s>=0)&(s<=1)).all(),'ARM_PROBABILITIES')
    quantized=np.array([[float('%.6f'%float(v)) for v in row] for row in s],dtype=np.float64)
    cr=pd.DataFrame(c).rank(method='average',pct=True).to_numpy()
    sr=pd.DataFrame(quantized).rank(method='average',pct=True).to_numpy()
    return quantized,cr,sr,.7*cr+.3*sr


def analyze(y,m,ty,tm,control,s6,blend,groups,preflight):
    import numpy as np
    require(y.shape==m.shape==(COUNT,12) and preflight['targets']==LABELS,'TRUTH870_SCHEMA')
    require(all(int(m[:,j].sum())==preflight['support'][t]['dev_observed'] for j,t in enumerate(LABELS)),'ORIGINAL_SUPPORT_REQUIRED')
    prevalence=np.divide((ty*tm).sum(0),tm.sum(0),out=np.full(12,np.nan),where=tm.sum(0)>0);require(np.isfinite(prevalence).all(),'TRAIN_PREVALENCE')
    metrics={name:metric_report(y,m,p,LABELS) for name,p in [('control',control),('fixed_blend',blend),('s6_descriptive',s6),('train_prevalence',np.broadcast_to(prevalence,y.shape))]}
    require(metrics['control']['defined_targets']==metrics['fixed_blend']['defined_targets']==12,'ALL12_DEFINED_REQUIRED')
    bootstrap=paired_bootstrap(y,m,control,blend,groups,draws=2000,seed=42,targets=LABELS)
    delta=metrics['fixed_blend']['macro_auc']-metrics['control']['macro_auc']
    damage=[t for t,a in metrics['fixed_blend']['targets'].items() if a['positive']>=20 and a['negative']>=20 and a['auc']-metrics['control']['targets'][t]['auc']<=-.01]
    passing=delta>0 and len(damage)<3 and bootstrap['ci95'] is not None and bootstrap['ci95'][0]>0
    return {'metrics':metrics,'paired_bootstrap':bootstrap,'macro_delta':delta,'target_damage':damage,
            'triage':'eligible for separate image-only rehearsal' if passing else 'fixed improvement probe closed; no search',
            'generalization_evidence':False,'prevalence_fit':'training-fold observed truth only','truth_sha256':fingerprint({'y':y.tolist(),'m':m.tolist()})}


def control_worker(config_path):
    import numpy as np
    import torch
    from knee.reference import load_definitions,load_model
    from knee.mirror_input import prepare_study
    from knee.public_fullfield import GRID_NAMES,prediction_error,snapshot,require_immutable
    cfg=json.loads(Path(config_path).read_bytes());out=Path(cfg['out'])/'control';out.mkdir();r={'status':'RUNNING','completed':0,'records':[],'phase':'source_state'};save(out/'receipt.json',r)
    try:
        r['environment']=environment(torch);save(out/'receipt.json',r)
        require(sys.version_info[:2]==(3,12) and r['environment']['packages']['timm']=='1.0.26' and torch.cuda.is_available() and torch.cuda.get_device_name(0)=='Tesla T4' and torch.__version__=='2.10.0+cu128','CONTROL_CUDA_ENVIRONMENT')
        public,decoder=load_definitions(cfg['public_source'],cfg['decoder_source'],cfg['public_manifest'])
        model=load_model(public,cfg['control_checkpoint'],'cuda',cfg['public_manifest']);before=snapshot(model)
        require(state_fingerprint(model.state_dict())==cfg['control_state_sha256'],'ORIGINAL_CONTROL_STATE')
        grid={k:v.detach().cpu().clone() for k,v in model.named_buffers() if k in GRID_NAMES}
        require(state_fingerprint(grid)==cfg['control_grid_sha256'],'ORIGINAL_CONTROL_GRID')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
        records={}
        with (Path(cfg['raw'])/'train_series.csv').open(newline='') as h:
            for row in csv.DictReader(h):records.setdefault(row[UID],[]).append(row)
        saved=np.load(cfg['saved_control'],allow_pickle=False);state_path=out/'frozen-control.pt';atomic_save(state_path,{k:v.detach().cpu() for k,v in model.state_dict().items()})
        with PlatformInterruption() as stop:
            for ordinal in CONTROL_SENTINELS:
                sid=cfg['dev'][ordinal];v,m,_=prepare_study(public,decoder,sid,records,Path(cfg['raw'])/'train_series',reference=True)
                fv,fm,_=prepare_study(public,decoder,sid,records,Path(cfg['raw'])/'train_series')
                require(np.array_equal(v,fv) and np.array_equal(m,fm),'CONTROL_REFERENCE_INPUT')
                inputs=[public.eval_windows(v,m,94,320,mirror=b) for b in (False,True)]
                p=public.infer_pair_probs(model,*inputs,'cuda');err=prediction_error(p,saved[ordinal],'SAVED_OLD320_PARITY')
                model.load_state_dict(torch.load(state_path,map_location='cpu',weights_only=True),strict=True);model.eval()
                again=public.infer_pair_probs(model,*inputs,'cuda');reload=prediction_error(p,again,'OLD320_RELOAD')
                require_immutable(model,before);r['records'].append({'ordinal':ordinal,'volume_sha256':sha256_bytes(v),'mask_sha256':sha256_bytes(m),'saved_max_abs':err,'reload_max_abs':reload});r['completed']+=1;save(out/'receipt.json',r)
                if stop.requested:r.update(status='INTERRUPTED',platform_signal=stop.signum);save(out/'receipt.json',r);return
        state_path.unlink();r.update(status='PASS',state_before=before,state_after=snapshot(model));save(out/'receipt.json',r)
    except BaseException as e:
        save(out/'failure-private.json',{'type':type(e).__name__,'detail':str(e)});r.update(status='FAIL',failure_type=type(e).__name__);save(out/'receipt.json',r);raise RuntimeError('Control sentinel gate failed') from None


def sha256_bytes(a):return hashlib.sha256(a.tobytes()).hexdigest()


def s6_worker(config_path):
    import numpy as np
    import torch
    import resource
    import setuptools
    import distutils.util
    cfg=json.loads(Path(config_path).read_bytes());out=Path(cfg['out'])/'s6';out.mkdir();start=time.perf_counter()
    r={'status':'RUNNING','phase':'source_state','completed':0,'optimizer_updates':0,'records':[]};save(out/'receipt.json',r)
    try:
        r['environment']=environment(torch);save(out/'receipt.json',r)
        verify_sources(cfg['code'],cfg['source_pins']);verify_assets(cfg['compact'],cfg['anatomy'],cfg['manifest'],cfg['pilot_binding'])
        verify_staged(cfg['staged'],cfg['staged_ids'],cfg['stage_binding'])
        require(sys.version_info[:2]==(3,12) and r['environment']['packages']['timm']=='1.0.26' and torch.cuda.is_available() and torch.cuda.get_device_name(0)=='Tesla T4' and torch.__version__=='2.10.0+cu128','S6_CUDA_ENVIRONMENT')
        require('monai' not in sys.modules,'FRESH_MONAI_NAMESPACE');sys.path.insert(0,cfg['code']);os.environ['RSNA_CODE_ROOT']=cfg['code']
        import compact_infer as CI
        import compact_train as CT
        require(CT.LABELS==LABELS,'S6_TARGET_ORDER');r['synthetic_source_checks']=synthetic_source_checks()
        torch.manual_seed(42);np.random.seed(42);torch.backends.cudnn.benchmark=False
        spec=json.loads((Path(cfg['compact'])/'state_spec.json').read_bytes());models=[]
        for name in cfg['manifest']['models']:
            sd=torch.load(Path(cfg['compact'])/name,map_location='cpu',weights_only=True)
            kw={'spatial':any(k=='sp_q' or k.endswith('.sp_q') for k in sd),'extent':any(k.split('.')[-1] in ('ext_u','ext_view','ext_gamma') for k in sd)}
            model=CT.Net('convnext_small',None,freeze_blocks=16,crop=352,state_spec=spec,in_chans=3,**kw);model.load_state_dict(sd,strict=True)
            models.append(model.eval().requires_grad_(False).to('cuda').to(memory_format=torch.channels_last));del sd
        pilot_ids=cfg['pilot_sentinels'];all_ids=pilot_ids+cfg['dev']
        ds=CI.CStudies(cfg['staged'],'train',all_ids,out/'scratch',cfg['code'],384,1,'v4',3,cfg['anatomy'],3.5,str(Path(cfg['compact'])/'side_cls.pt'),2.,0.,20.,extent_meta=any(m.extent for m in models));ds._setup()
        import monai
        r['environment']['vendored_monai']={'version':monai.__version__,'path':str(Path(monai.__file__).resolve())}
        require(monai.__version__=='0.6.0' and Path(monai.__file__).resolve().is_relative_to(Path(cfg['code'])/'training/vendor_runtime/monai060'),'VENDORED_MONAI')
        states=models+[ds.bank.model,ds.side_net];initial=model_digest(states);require(initial==cfg['pilot_state_sha256'],'QUALIFIED_S6_STATE');r['state_before']=initial
        def finite_aux(module,inputs,output):require(torch.isfinite(output).all().item(),'NONFINITE_AUXILIARY')
        ds.bank.model.register_forward_hook(finite_aux);ds.side_net.register_forward_hook(finite_aux)
        capture={};original_build=ds.builder.build;original_rows=ds.window_rows;original_record=ds.bank.record
        def observe_record(*a,**kw):
            record=original_record(*a,**kw);capture['anchors']=record;require(not any(str(w).startswith('error:') for w in record['why']),'SWALLOWED_LOCALIZER_ERROR');return record
        def observe_build(uid):
            item=original_build(uid);capture['item']=item
            study=json.loads((Path(ds.builder.cache)/'studies'/(hashlib.sha256(uid.encode()).hexdigest()[:24]+'.json')).read_bytes())
            require(not study.get('errors',[]) and all(not s.get('errors') for s in study['series']) and not item['skipped_series'],'SWALLOWED_DECODE_OR_SERIES_ERROR')
            capture['scratch_bytes']=sum(p.stat().st_size for p in Path(ds.builder.cache).rglob('*') if p.is_file());return item
        def observe_rows(*a,**kw):
            rows=original_rows(*a,**kw);capture['rows']=rows;return rows
        ds.bank.record=observe_record;ds.builder.build=observe_build;ds.window_rows=observe_rows
        logits=[];probs=[]
        with PlatformInterruption() as stop:
            for i,sid in enumerate(all_ids):
                capture.clear();r['phase']='pilot_sentinel' if i<len(pilot_ids) else 'evaluation';r['current_ordinal']=i-len(pilot_ids);save(out/'receipt.json',r);item=ds[i]
                if item['error'] is not None:
                    save(out/'current-study-error-private.json',{'ordinal':r['current_ordinal'],'study':sid,'source_error':item['error']})
                require(item['error'] is None and len(item['images'])>0,'FAILED_STUDY')
                a=item['images'].numpy();meta=item['meta'].numpy();require(a.dtype==np.uint8 and a.shape[1:]==(3,384,384),'S6_INPUT_SCHEMA')
                fp={'render_windows':array_sha(a),'metadata':array_sha(meta),'retained_crop':array_sha(a[:,:,16:368,16:368]),'mask':digest([True]*len(a)),
                    'pixel_rows':array_sha(capture['item']['px'].numpy()),'row_metadata':array_sha(capture['item']['meta'].numpy()),'window_blocks':digest(capture['item']['blocks']),'window_rows':digest(capture['rows'])}
                rawmeta=capture['item']['meta'].numpy()
                require(all(len(set(rawmeta[list(t),0]))==1 and len(set(rawmeta[list(t),3]))==1 for t in [row[2:] for row in capture['rows']]),'TRIPLET_SERIES_VIEW')
                t=time.perf_counter();img=item['images'].to('cuda')[None,:,:,16:368,16:368];mt=item['meta'].to('cuda')[None];mask=torch.ones((1,len(a)),dtype=torch.bool,device='cuda')
                require(all(not module.training for model in states for module in model.modules()),'S6_EVAL_STATE')
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):zs=torch.stack([m(img,mt,mask,chunk=16).float()[0] for m in models])
                require(torch.isfinite(zs).all().item(),'NONFINITE_FP16_NO_RESCUE');z=zs.cpu().numpy();p=torch.sigmoid(zs.mean(0)).double().cpu().numpy()
                oracle=torch.sigmoid(torch.from_numpy(z).mean(0)).double().numpy();err=float(np.max(np.abs(p-oracle)));require(err<=1e-6,'MEAN_SIGMOID_RECONSTRUCTION')
                torch.cuda.synchronize();elapsed=time.perf_counter()-t
                require(not any((out/'scratch').glob('s_*')),'PER_STUDY_SCRATCH_NOT_REMOVED')
                row={'ordinal':i-len(pilot_ids),'windows':len(a),'input':fp,'preprocess_seconds':item['seconds'],'four_model_seconds':elapsed,'mean_sigmoid_error':err,
                     'anchor_method':capture['anchors']['method'],'anchor_reasons':capture['anchors']['why'],'side':item.get('side'),'native_scratch_bytes':capture['scratch_bytes'],
                     'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                if i<len(pilot_ids):
                    expected=cfg['pilot_expected'][i];require(fp==expected['input'],'PILOT_INPUT_WINDOW_PARITY')
                    row['pilot_probability_error']=float(np.max(np.abs(p-np.array(expected['probabilities']))));require(row['pilot_probability_error']<=1e-6,'PILOT_PREDICTION_PARITY')
                    r.setdefault('pilot_sentinels',[]).append(row)
                else:
                    logits.append(z);probs.append(p);save_predictions(out/'predictions-private.npz',cfg['dev'][:len(probs)],logits,probs)
                    r['records'].append(row);r['completed']=len(probs)
                r['elapsed_seconds']=time.perf_counter()-start;save(out/'receipt.json',r)
                del img,mt,mask,zs,item,a,meta,capture['item']
                if stop.requested:r.update(status='INTERRUPTED',platform_signal=stop.signum);save(out/'receipt.json',r);return
        require(len(probs)==COUNT,'COMPLETE870_REQUIRED');r['state_after']=model_digest(states);require(r['state_after']==initial,'S6_PARAMETER_BUFFER_MUTATION')
        r.update(status='PASS',zero_imputed=True,zero_failed=True,fp32_recoveries=0);save(out/'receipt.json',r)
    except BaseException as e:
        save(out/'failure-private.json',{'type':type(e).__name__,'detail':str(e)});r.update(status='FAIL',failure_type=type(e).__name__,elapsed_seconds=time.perf_counter()-start);save(out/'receipt.json',r);raise RuntimeError('S6 evaluation failed; preserve partial ledger') from None


def run(config_path):
    import numpy as np
    cfg=json.loads(Path(config_path).read_bytes());out=Path(cfg['out']);out.mkdir(exist_ok=False);started=time.perf_counter()
    r={'status':'RUNNING','phase':'gates','binding':cfg['binding'],'config':config(),'completed_studies':0,'optimizer_updates':0,'intermediate_metrics_opened':False}
    save(out/'receipt.json',r)
    try:
        qualification_gate(cfg['pilot_receipt'],cfg['coverage']);validate_population(cfg['dev'],cfg['folds'],cfg['excluded'],cfg['groups'],cfg['preflight'])
        require(digest(cfg['binding'])==cfg['binding_sha256'],'CONFIG_BINDING');require(file_sha(cfg['labels'])==cfg['binding']['label_sha256'] and file_sha(cfg['masks'])==cfg['binding']['truth_sha256'],'LABEL_TRUTH_BYTES')
        require(file_sha(cfg['raw']+'/sample_submission.csv')==cfg['binding']['schema_sha256'],'OFFICIAL_SCHEMA_BYTES')
        require(file_sha(cfg['raw']+'/train.csv')==cfg['binding']['official_train_sha256'] and file_sha(cfg['raw']+'/train_series.csv')==cfg['binding']['series_sha256'],'OFFICIAL_INPUT_BYTES')
        verify_sources(cfg['code'],cfg['source_pins']);verify_assets(cfg['compact'],cfg['anatomy'],cfg['manifest'],cfg['pilot_binding'])
        require(array_sha(np.load(cfg['saved_control'],allow_pickle=False))==cfg['binding']['saved_control_array_sha256'],'SAVED_CONTROL_BYTES')
        env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1])}
        r['phase']='control_sentinels';save(out/'receipt.json',r)
        run_child([sys.executable,'-m','knee.public_s6_evaluation','control',str(config_path)],env)
        require(json.loads((out/'control/receipt.json').read_bytes())['status']=='PASS','CONTROL_SENTINEL_FAILURE')
        r['phase']='stage_images_only';save(out/'receipt.json',r)
        cfg['staged_ids']=cfg['pilot_sentinels']+cfg['dev'];cfg['stage_binding']=stage_studies(cfg['raw'],cfg['staged'],cfg['staged_ids'],cfg['binding']['series_sha256']);save(config_path,cfg)
        r['phase']='s6_stream';save(out/'receipt.json',r)
        run_child([sys.executable,'-m','knee.public_s6_evaluation','s6',str(config_path)],env)
        worker=json.loads((out/'s6/receipt.json').read_bytes());require(worker['status']=='PASS' and worker['completed']==COUNT,'COMPLETE_GENUINE870_GATE')
        with np.load(out/'s6/predictions-private.npz',allow_pickle=False) as ar:
            require(ar['study_ids'].tolist()==cfg['dev'],'S6_SAVED_ID_ORDER');s6=ar['probabilities'].copy();logits=ar['logits'].copy()
        require(logits.shape==(COUNT,4,12) and np.isfinite(logits).all(),'FOUR_FOLD_LOGITS_GATE')
        c=np.load(cfg['saved_control'],allow_pickle=False);q,cr,sr,blend=fixed_ranks(c,s6)
        atomic_save(out/'comparison-private.npz',None,lambda _,h:np.savez(h,study_ids=np.asarray(cfg['dev']),control=c,s6_raw=s6,s6_quantized=q,control_rank=cr,s6_rank=sr,fixed_blend=blend,logits=logits))
        r['completed_studies']=COUNT;r['prediction_artifacts_sha256']={n:file_sha(out/n) for n in ['comparison-private.npz','s6/predictions-private.npz']};r['phase']='truth_analysis';save(out/'receipt.json',r)
        train,dev,_,_,_,groups,grouping=preflight_inputs(cfg['raw'],cfg['labels'],cfg['masks'],cfg['folds'],cfg['excluded'],cfg['preflight'],LABELS)
        require(dev==cfg['dev'] and groups==cfg['groups'],'ORIGINAL_TRUTH_GROUP_BINDING')
        y,m=load_binary_truth(cfg['masks'],dev,LABELS);ty,tm=load_binary_truth(cfg['masks'],train,LABELS)
        result=analyze(y,m,ty,tm,c,q,blend,groups,cfg['preflight'])
        r.update(status='PASS',phase='complete',**result,elapsed_seconds=time.perf_counter()-started,zero_imputed=True,zero_failed=True,fp32_recoveries=0,
                 environment=worker['environment'],state_before=worker['state_before'],state_after=worker['state_after'],grouping=grouping,windows=sum(x['windows'] for x in worker['records']),
                 peak_cuda_bytes=max(x['peak_cuda_bytes'] for x in worker['records']),peak_native_scratch_bytes=max(x['native_scratch_bytes'] for x in worker['records']),
                 unknowns=['population crop anatomy beyond qualified32','field-center/side-flip/missing-plane cohort strata','chunk192 equivalence','hidden-test/offline rehearsal runtime'])
        save(out/'receipt.json',r);return r
    except BaseException as e:
        for phase in ('control','s6'):
            receipt=out/phase/'receipt.json'
            if receipt.is_file():
                child=json.loads(receipt.read_bytes());r.setdefault('worker_progress',{})[phase]={k:child[k] for k in ('status','phase','completed','current_ordinal','failure_type','environment') if k in child}
                if phase=='s6':r['completed_studies']=child.get('completed',0)
        save(out/'failure-private.json',{'type':type(e).__name__,'detail':str(e)});r.update(status='FAIL',failure_type=type(e).__name__,elapsed_seconds=time.perf_counter()-started);save(out/'receipt.json',r);raise RuntimeError('Fixed S6 comparison failed; no automatic resume') from None


if __name__=='__main__':
    if sys.argv[1]=='control':control_worker(sys.argv[2])
    elif sys.argv[1]=='s6':s6_worker(sys.argv[2])
    elif sys.argv[1]=='run':run(sys.argv[2])
    else:raise SystemExit('Expected control, s6 or run')
