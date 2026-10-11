"""Complete-study dual-T4 image-only fixed70 old320/30 S6 entry.

Torch is imported only in fresh board-isolated workers. Atomic ledgers remain
on failure; no partial merge, missing fill, shard ranks or automatic resume.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from knee.public_s6_qualification import (LABELS,RECIPE,UID,SERIES_COLUMNS,canonical,digest,file_sha,require,save,verify_sources,verify_assets,verify_staged,synthetic_source_checks)
from knee.public_s6_evaluation import array_sha,model_digest,environment,save_predictions
from knee.public_adaptation_fit import PlatformInterruption,atomic_save
from knee.public_adaptation_pilot import state_fingerprint
from knee.public_fullfield_entry import safe_id,read_series,test_contract,write_csv

VERSION='s6-dual-image-only-v1'
METHOD={'old':'central320 K94 equal original/mirror probabilities','s6':RECIPE,'blend':[.7,.3],
        'rank':'pandas average pct over all actual test rows','s6_quantization':'source Python %.6f float64','assignment':'canonical ordinal modulo2'}
MAX_SECONDS=9*3600
SENTINELS=[0,1,20,9]


def probabilities(value,n):
    import numpy as np
    p=np.asarray(value);require(p.shape==(n,12) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'FINITE_COMPLETE_PROBABILITIES');return p


def quantize(value):
    import numpy as np
    return np.array([[float('%.6f'%float(v)) for v in row] for row in value],dtype=np.float64)


def blend(control,s6):
    import pandas as pd
    require(len(control)>0,'EMPTY_RANK_POPULATION');c=probabilities(control,len(control));s=quantize(probabilities(s6,len(control)))
    return .7*pd.DataFrame(c).rank(method='average',pct=True).to_numpy()+.3*pd.DataFrame(s).rank(method='average',pct=True).to_numpy()


def assignments(ids):
    require(len(ids)>=2 and len(ids)==len(set(ids)) and all(safe_id(s) for s in ids),'CANONICAL_IDS')
    return [[s for i,s in enumerate(ids) if i%2==board] for board in range(2)]


def stage(staged,ids,records,image_paths):
    """CStudies gets an ID-only train alias; original train.csv is never read."""
    staged=Path(staged);require(not staged.exists() and set(ids)==set(records)==set(image_paths),'FRESH_EXACT_STAGE')
    staged.mkdir(parents=True);(staged/'train_series').mkdir()
    with (staged/'train.csv').open('w',newline='') as h:
        w=csv.writer(h);w.writerow([UID]);w.writerows([[s] for s in ids])
    pairs=[];inventory=[]
    with (staged/'train_series.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=SERIES_COLUMNS);w.writeheader()
        for sid in ids:
            require(safe_id(sid),'UNSAFE_ID')
            for row in records[sid]:
                series=row['SeriesInstanceUID'];require(safe_id(series) and row[UID]==sid,'STAGE_METADATA')
                source=Path(image_paths[sid])/sid/series
                require(source.is_dir() and source.resolve().is_relative_to((Path(image_paths[sid])/sid).resolve()) and any(source.glob('*.dcm')),'IMAGE_PATH')
                p=staged/'train_series'/sid/series;p.parent.mkdir(exist_ok=True);p.symlink_to(source.resolve(),target_is_directory=True)
                w.writerow({k:row[k] for k in SERIES_COLUMNS});pairs.append((sid,series));inventory.append([sid,series,str(source.resolve())])
    require(len(pairs)==len(set(pairs)),'DUPLICATE_STAGE_SERIES')
    r={'train_sha256':file_sha(staged/'train.csv'),'series_sha256':file_sha(staged/'train_series.csv'),'image_paths_sha256':digest(inventory),
       'membership_sha256':digest(ids),'series_membership_sha256':digest(pairs),'studies':len(ids),'series':len(pairs)}
    verify_staged(staged,ids,r);return r


def old_ledger(path,ids,p):
    import numpy as np
    p=probabilities(p,len(ids));require(len(ids)==len(set(ids)),'DUPLICATE_LEDGER_ID')
    atomic_save(path,None,lambda _,h:np.savez(h,study_ids=np.asarray(ids),probabilities=p))


def merge(ids,shards,arm):
    import numpy as np
    expected=assignments(ids);by_id={};require(len(shards)==2,'TWO_SHARDS_REQUIRED')
    for board,(r,path) in enumerate(shards):
        require(r.get('status')=='PASS' and r.get('board')==board and r.get('arm')==arm and r.get('assigned')==expected[board] and r.get('completed')==len(expected[board]),'WORKER_TERMINAL_GATE')
        require(r.get('state_before')==r.get('state_after') and r.get('zero_failed') is True and r.get('zero_imputed') is True and r.get('fp32_recoveries')==0,'WORKER_STATE_FAILURE_GATE')
        require(r.get('ledger_sha256')==file_sha(path),'ATOMIC_LEDGER_BINDING')
        with np.load(path,allow_pickle=False) as a:
            got=a['study_ids'].tolist();p=probabilities(a['probabilities'],len(got));require(got==expected[board],'SHARD_ID_ORDER_COVERAGE')
            if arm=='s6':require(a['logits'].shape==(len(got),4,12) and np.isfinite(a['logits']).all(),'FOUR_READER_LOGITS_REQUIRED')
            for sid,row in zip(got,p):require(sid not in by_id,'DUPLICATE_MERGE_ID');by_id[sid]=row
    require(set(by_id)==set(ids),'COMPLETE_ARMS_REQUIRED');return np.array([by_id[s] for s in ids])


def physical_uuid(properties,expected):
    """Compare the exact16 CUDA UUID bytes with the NVIDIA inventory UUID.

    PyTorch2.10 exposes CUuuid.bytes; nvidia-smi includes GPU-. Compare
    bytes so display formatting cannot replace the physical-board check.
    """
    import uuid
    require(isinstance(expected,str) and expected.startswith('GPU-'),'NVIDIA_UUID_SCHEMA')
    raw=bytes(properties.uuid.bytes);require(len(raw)==16,'CUDA_UUID16_BYTES')
    require(uuid.UUID(bytes=raw)==uuid.UUID(expected[4:]),'PHYSICAL_UUID_BINDING')
    return {'physical_uuid':expected,'torch_uuid_raw':str(properties.uuid),'uuid_bytes_hex':raw.hex()}


def board_environment(torch,cfg):
    import importlib.metadata
    r=environment(torch);r['physical_board']=cfg['board'];r['cuda_visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES')
    require(r['cuda_visible_devices']==str(cfg['board']) and torch.cuda.device_count()==1 and torch.cuda.get_device_name(0)=='Tesla T4','ISOLATED_T4_REQUIRED')
    require(list(sys.version_info[:2])==[3,12] and torch.__version__==cfg['environment']['torch'],'PINNED_PYTHON_TORCH')
    for name,pin in cfg['environment']['packages'].items():require(importlib.metadata.version(name)==pin,'PINNED_PACKAGE_'+name)
    r.update(physical_uuid(torch.cuda.get_device_properties(0),cfg['physical']['uuid']))
    r['torch_threads']=torch.get_num_threads();r['torch_interop_threads']=torch.get_num_interop_threads();return r


def probe_gate(sid,arm,p,fp,cfg):
    import numpy as np
    expected=cfg.get('probes',{}).get(sid)
    if expected is None:return None
    e=expected.get(arm)
    if e is None:return None
    require(all(fp.get(k)==v for k,v in e['input'].items()),'QUALIFIED_'+arm+'_INPUT')
    if e['probabilities'] is None:return None
    err=float(np.max(np.abs(p-np.array(e['probabilities']))));require(err<=1e-6,'QUALIFIED_'+arm+'_PROBABILITY')
    if arm=='s6':require(np.array_equal(quantize([p]),quantize([e['probabilities']])),'EXACT_SOURCE_SIX_DECIMALS')
    return err


def window_audit(plain,*,full=None,reference=None):
    import torch
    require(len(plain)==2 and all(x.shape[-2:]==(320,320) for x in plain),'OLD320_WINDOW_SCHEMA')
    if reference is not None:require(all(torch.equal(a,b) for a,b in zip(plain,reference)),'OLD320_FRESH_REFERENCE_WINDOWS')
    result={'old320_windows':{k:state_fingerprint(x) for k,x in zip(('normal','mirrored'),plain)}}
    if full is not None:
        require(len(full)==2 and all(x.shape[-2:]==(384,384) for x in full),'QUALIFIED_FULL384_WINDOW_SCHEMA')
        require(all(torch.equal(f[:,:,32:352,32:352],c) for f,c in zip(full,plain)),'QUALIFIED_CENTRAL320_SUBARRAY')
        result['qualified_full384_windows']={k:state_fingerprint(x) for k,x in zip(('normal','mirrored'),full)}
    return result


def old_worker(config_path):
    import numpy as np
    import torch
    import resource
    from knee.reference import load_definitions,load_model
    from knee.mirror_input import prepare_study
    from knee.public_fullfield import GRID_NAMES,snapshot,require_immutable,fullfield_windows
    cfg=json.loads(Path(config_path).read_bytes());out=Path(cfg['out'])/'old';out.mkdir();start=time.monotonic()
    r={'status':'RUNNING','board':cfg['board'],'arm':'old','assigned':cfg['dev'],'completed':0,'records':[],'sentinels':[]};save(out/'receipt.json',r)
    try:
        r['environment']=board_environment(torch,cfg);verify_staged(cfg['staged'],cfg['staged_ids'],cfg['stage_binding'])
        public,decoder=load_definitions(cfg['public_source'],cfg['decoder_source'],cfg['public_manifest'])
        model=load_model(public,cfg['control_checkpoint'],'cuda',cfg['public_manifest']);before=snapshot(model);r['state_before']=before
        require(before['state']==cfg['control_state_sha256'],'QUALIFIED_OLD_STATE')
        grid={k:v.detach().cpu().clone() for k,v in model.named_buffers() if k in GRID_NAMES};r['constructor_grid_sha256']=state_fingerprint(grid);require(r['constructor_grid_sha256']==cfg['control_grid_sha256'],'OLD_CONSTRUCTOR_GRID')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
        records=read_series(Path(cfg['staged'])/'train_series.csv');frozen=out/'frozen-state.pt';atomic_save(frozen,{k:v.detach().cpu() for k,v in model.state_dict().items()})
        def infer(sid):
            t=time.monotonic();v,m,counters=prepare_study(public,decoder,sid,records,Path(cfg['staged'])/'train_series')
            require(all(counters.get(k,0)==0 for k in ('resource_errors','invalid_pixels','pixel_failures','fast_bad_headers','header_parse_failures')),'OLD_DECODE_FAILURE')
            reference_seconds=0.
            if sid in cfg.get('probes',{}):
                reference_start=time.monotonic()
                rv,rm,rc=prepare_study(public,decoder,sid,records,Path(cfg['staged'])/'train_series',reference=True)
                require(np.array_equal(v,rv) and np.array_equal(m,rm),'OLD_ORIGINAL_SOURCE_REFERENCE_INPUT')
                require(all(rc.get(k,0)==0 for k in ('resource_errors','invalid_pixels','pixel_failures','fast_bad_headers','header_parse_failures')),'OLD_REFERENCE_DECODE_FAILURE')
                refinputs=[public.eval_windows(rv,rm,94,320,mirror=b) for b in (False,True)]
                refp=np.asarray(public.infer_pair_probs(model,*refinputs,'cuda'),dtype=np.float32);torch.cuda.synchronize();reference_seconds=time.monotonic()-reference_start
            inputs=[public.eval_windows(v,m,94,320,mirror=b) for b in (False,True)];p=np.asarray(public.infer_pair_probs(model,*inputs,'cuda'),dtype=np.float32);probabilities([p],1)
            if sid in cfg.get('probes',{}):require(float(np.max(np.abs(p-refp)))<=1e-6,'OLD_ORIGINAL_SOURCE_REFERENCE_PROBABILITY')
            fp={'volume_sha256':hashlib.sha256(v.tobytes()).hexdigest(),'mask_sha256':hashlib.sha256(m.tobytes()).hexdigest(),**window_audit(inputs,reference=refinputs if sid in cfg.get('probes',{}) else None)}
            expected_old=cfg.get('probes',{}).get(sid,{}).get('old',{})
            if 'qualified_full384_windows' in expected_old.get('input',{}):
                audit_start=time.monotonic();full=[fullfield_windows(public,v,m,mirror=b) for b in (False,True)]
                fp.update(window_audit(inputs,full=full,reference=refinputs));del full;reference_seconds+=time.monotonic()-audit_start
            torch.cuda.synchronize();return p,fp,{'seconds':time.monotonic()-t,'production_seconds':time.monotonic()-t-reference_seconds,'reference_seconds':reference_seconds,'decode_counters':counters}
        ps=[]
        with PlatformInterruption() as stop:
            for sid in cfg.get('old_sentinels',[]):
                p,fp,d=infer(sid);probe_gate(sid,'old',p,fp,cfg)
                model.load_state_dict(torch.load(frozen,map_location='cpu',weights_only=True),strict=True);model.eval();require_immutable(model,before)
                q,g,_=infer(sid);probe_gate(sid,'old',q,g,cfg)
                require(g==fp and float(np.max(np.abs(p-q)))<=1e-6,'OLD_FRESH_DECODE_RELOAD')
                r['sentinels'].append({'id':sid,'input':fp,'probabilities':p.tolist(),'reloaded_probabilities':q.tolist(),'strict_reload':True});save(out/'receipt.json',r)
                if stop.requested:raise InterruptedError('Interrupted sentinel')
            for sid in cfg['dev']:
                r['current_id']=sid;save(out/'receipt.json',r);p,fp,d=infer(sid);err=probe_gate(sid,'old',p,fp,cfg)
                if sid in cfg.get('probes',{}):
                    model.load_state_dict(torch.load(frozen,map_location='cpu',weights_only=True),strict=True);model.eval();require_immutable(model,before)
                    q,g,_=infer(sid);probe_gate(sid,'old',q,g,cfg);require(fp==g and float(np.max(np.abs(p-q)))<=1e-6,'OLD32_REPEAT_RELOAD')
                    d.update(source_reference_parity=True,strict_reload=True,reload_max_abs=float(np.max(np.abs(p-q))))
                ps.append(p)
                old_ledger(out/'predictions.npz',cfg['dev'][:len(ps)],ps);r['records'].append({'id':sid,'input':fp,'qualified_error':err,**d});r['completed']=len(ps);save(out/'receipt.json',r)
                if stop.requested:raise InterruptedError('Interrupted study boundary')
        require_immutable(model,before);r.update(status='PASS',state_after=snapshot(model),zero_failed=True,zero_imputed=True,fp32_recoveries=0,
            ledger_sha256=file_sha(out/'predictions.npz'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        frozen.unlink()
    except BaseException as e:
        save(out/'failure-private.json',{'type':type(e).__name__,'detail':str(e)});r.update(status='INTERRUPTED' if isinstance(e,(InterruptedError,KeyboardInterrupt)) else 'FAIL',failure_type=type(e).__name__);raise
    finally:r['elapsed_seconds']=time.monotonic()-start;save(out/'receipt.json',r)


def s6_worker(config_path):
    import numpy as np
    import torch
    import resource
    import setuptools
    import distutils.util
    cfg=json.loads(Path(config_path).read_bytes());out=Path(cfg['out'])/'s6';out.mkdir();start=time.perf_counter()
    r={'status':'RUNNING','phase':'source_state','board':cfg['board'],'arm':'s6','assigned':cfg['dev'],'completed':0,'optimizer_updates':0,'records':[]};save(out/'receipt.json',r)
    try:
        r['environment']=board_environment(torch,cfg);save(out/'receipt.json',r)
        verify_sources(cfg['code'],cfg['source_pins']);verify_assets(cfg['compact'],cfg['anatomy'],cfg['manifest'],cfg['asset_binding'])
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
        pilot_ids=cfg.get('s6_sentinels',[])*2;all_ids=pilot_ids+cfg['dev']
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
                if i<len(pilot_ids):
                    require(model_digest(states)==initial,'S6_STATE_BEFORE_RELOAD')
                    for m,name in zip(models,cfg['manifest']['models']):
                        m.load_state_dict(torch.load(Path(cfg['compact'])/name,map_location='cpu',weights_only=True),strict=True);m.eval()
                    CI.FORCED_SIDE.clear()
                    ds=CI.CStudies(cfg['staged'],'train',all_ids,out/'scratch',cfg['code'],384,1,'v4',3,cfg['anatomy'],3.5,str(Path(cfg['compact'])/'side_cls.pt'),2.,0.,20.,extent_meta=any(m.extent for m in models));ds._setup()
                    states=models+[ds.bank.model,ds.side_net];require(model_digest(states)==initial,'S6_STRICT_RELOAD_STATE')
                    ds.bank.model.register_forward_hook(finite_aux);ds.side_net.register_forward_hook(finite_aux)
                    original_build=ds.builder.build;original_rows=ds.window_rows;original_record=ds.bank.record
                    ds.bank.record=observe_record;ds.builder.build=observe_build;ds.window_rows=observe_rows
                study_start=time.perf_counter();capture.clear();r['phase']='pilot_sentinel' if i<len(pilot_ids) else 'evaluation';r['current_ordinal']=i-len(pilot_ids);save(out/'receipt.json',r);item=ds[i]
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
                     'anchor_method':capture['anchors']['method'],'anchor_reasons':capture['anchors']['why'],'side':item.get('side'),'native_scratch_bytes':capture['scratch_bytes'],'field_center_fallbacks':sum(x=='field_center_fallback' for x in capture['anchors']['method']),
                     'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                expected_mechanics=cfg.get('probes',{}).get(sid,{}).get('s6',{}).get('mechanics')
                if expected_mechanics is not None:require(all(row.get(k)==v for k,v in expected_mechanics.items()),'QUALIFIED_S6_ANCHOR_FALLBACK_SEMANTICS')
                if i<len(pilot_ids):
                    expected=cfg['probes'][sid]['s6'];require(fp==expected['input'],'PILOT_INPUT_WINDOW_PARITY')
                    row['pilot_probability_error']=float(np.max(np.abs(p-np.array(expected['probabilities']))));require(row['pilot_probability_error']<=1e-6,'PILOT_PREDICTION_PARITY')
                    probe_gate(sid,'s6',p,fp,cfg)
                    row.update(id=sid,probabilities=p.tolist(),strict_reload=True,logits=z.tolist())
                    r.setdefault('sentinel_repeats',[]).append(row)
                    if i>=len(pilot_ids)//2:
                        prior=r['sentinel_repeats'][i-len(pilot_ids)//2]
                        require(prior['input']==fp and float(np.max(np.abs(p-np.array(prior['probabilities']))))<=1e-6,'S6_BOTH_FRESH_RELOAD_PARITY')
                        row['reloaded_probabilities']=p.tolist();row['reloaded_logits']=z.tolist();row['logits']=prior['logits'];row['probabilities']=prior['probabilities'];r.setdefault('sentinels',[]).append(row)
                else:
                    row.update(id=sid,qualified_error=probe_gate(sid,'s6',p,fp,cfg));logits.append(z);probs.append(p);save_predictions(out/'predictions.npz',cfg['dev'][:len(probs)],logits,probs)
                    r['records'].append(row);r['completed']=len(probs)
                row['production_seconds']=time.perf_counter()-study_start
                r['elapsed_seconds']=time.perf_counter()-start;save(out/'receipt.json',r)
                del img,mt,mask,zs,item,a,meta,capture['item']
                if stop.requested:raise InterruptedError('Platform interruption at study boundary')
        require(len(probs)==len(cfg['dev']),'COMPLETE_ASSIGNED_REQUIRED');r['state_after']=model_digest(states);require(r['state_after']==initial,'S6_PARAMETER_BUFFER_MUTATION')
        r.update(status='PASS',zero_imputed=True,zero_failed=True,fp32_recoveries=0,ledger_sha256=file_sha(out/'predictions.npz'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved());save(out/'receipt.json',r)
    except BaseException as e:
        save(out/'failure-private.json',{'type':type(e).__name__,'detail':str(e)});r.update(status='INTERRUPTED' if isinstance(e,(InterruptedError,KeyboardInterrupt)) else 'FAIL',failure_type=type(e).__name__,elapsed_seconds=time.perf_counter()-start);save(out/'receipt.json',r);raise RuntimeError('S6 evaluation failed; preserve partial ledger') from None


def resource_snapshot(out):
    """Sample aggregate parent/descendant RSS without initializing CUDA."""
    processes={}
    for p in Path('/proc').glob('[0-9]*/status'):
        try:
            f=dict(line.split(':',1) for line in p.read_text().splitlines() if ':' in line)
            processes[int(p.parent.name)]=(int(f.get('PPid','0')),int(f.get('VmRSS','0 kB').split()[0])*1024)
        except (OSError,ValueError):pass
    descendants={os.getpid()};changed=True
    while changed:
        added={pid for pid,(ppid,_) in processes.items() if ppid in descendants};changed=not added<=descendants;descendants|=added
    return {'host_rss_bytes':sum(processes.get(pid,(0,0))[1] for pid in descendants),
            'output_disk_bytes':sum(p.stat().st_size for p in Path(out).rglob('*') if p.is_file()),'disk_free_bytes':shutil.disk_usage(out).free}


def worker_environment(board):
    project=str(Path(__file__).resolve().parents[1])
    return {**os.environ,'CUDA_VISIBLE_DEVICES':str(board),'PYTHONDONTWRITEBYTECODE':'1',
            'PYTHONPATH':project+(os.pathsep+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else ''),'KNEE_ENTRY_PROJECT':project}


def launch(cfg,out,arm):
    import signal
    children=[];start=time.monotonic();high={'host_rss_bytes':0,'output_disk_bytes':0};min_free=shutil.disk_usage(out).free
    try:
        with PlatformInterruption() as stop:
            for board in range(2):
                folder=out/f'board{board}-{arm}';folder.mkdir()
                c={**cfg,'board':board,'arm':arm,'dev':cfg['assignment'][board],'physical':cfg['boards'][board],'out':str(folder)}
                path=out/f'board{board}-{arm}-config.json';save(path,c)
                env=worker_environment(board)
                log=(out/f'board{board}-{arm}.log').open('xb')
                child=subprocess.Popen([sys.executable,'-m','knee.public_s6_entry','--worker',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT);log.close();children.append(child)
            while any(c.poll() is None for c in children):
                s=resource_snapshot(out)
                for k in high:high[k]=max(high[k],s[k])
                min_free=min(min_free,s['disk_free_bytes'])
                if stop.requested or any(c.poll() not in (None,0) for c in children):
                    for c in children:
                        if c.poll() is None:c.send_signal(stop.signum or signal.SIGTERM)
                    if stop.requested:raise InterruptedError('Platform interruption')
                    raise RuntimeError('Worker failure; no partial merge')
                time.sleep(.5)
            require(all(c.returncode==0 for c in children),'WORKER_EXIT_FAILURE')
        reports=[json.loads((out/f'board{b}-{arm}'/arm/'receipt.json').read_bytes()) for b in range(2)]
        return reports,{'wall_seconds':time.monotonic()-start,'sample_interval_seconds':.5,'peaks':high,'minimum_disk_free_bytes':min_free}
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=45)
            except subprocess.TimeoutExpired:c.kill();c.wait()


def projection(cfg,phases,reports,setup,merge_seconds):
    n=sum(len(r['records']) for r in reports['old']);windows=sum(x['windows'] for r in reports['s6'] for x in r['records'])
    d=cfg['development_windows'];require(len(d)==870 and all(w>0 for w in d),'FROZEN870_WINDOW_DISTRIBUTION');require(n>0 and windows>0,'MEASURED_THROUGHPUT')
    production={};overhead={}
    for arm in ('old','s6'):
        production[arm]=max(sum(x['production_seconds'] for x in r['records']) for r in reports[arm])
        require(0<production[arm]<=phases[arm]['wall_seconds'],'MEASURED_PHASE_PRODUCTION')
        # Keep all observed setup, parity, sentinel, hash/load/final-state and
        # scheduling overhead once. Production excludes rehearsal-only repeats
        # and reference decoder/forward passes, absent from the pure entry.
        overhead[arm]=phases[arm]['wall_seconds']-production[arm]
    mean=sum(d)/870;scenarios=[]
    for multiplier in (1.,1.25,1.5):
        for imbalance in (1.,1.15,1.3):
            seconds=setup+merge_seconds+sum(overhead.values())+production['old']*1300/n*imbalance+production['s6']*(1300*mean*multiplier/windows)*imbalance
            scenarios.append({'hidden_window_multiplier':multiplier,'shard_imbalance':imbalance,'seconds':seconds})
    return {'population':1300,'window_distribution':{'count':870,'total':sum(d),'mean':mean,'max':max(d)},'scenarios':scenarios,
        'measured_production_wall_seconds':production,'measured_fixed_phase_overhead_seconds':overhead,
        'conservative_seconds':next(x['seconds'] for x in scenarios if x['hidden_window_multiplier']==1.25 and x['shard_imbalance']==1.15),
        'policy':'Slower observed shard production; all measured phase overhead once, including extra qualification work; 25% more windows and15% additional imbalance. Sensitivities are not hidden-population bounds; growing ledger I/O and hidden tails remain unknown.'}


def require_rehearsal(r,binding,csv_path):
    require(binding.get('version')==VERSION and binding.get('method')==METHOD,'EXACT_FROZEN_METHOD')
    require(r.get('status')=='PASS' and r.get('version')==VERSION and r.get('mode')=='rehearsal' and r.get('binding')==binding,'PASS_EXACT_REHEARSAL_REQUIRED')
    require(r.get('completed')==39 and r.get('completed_visible')==3 and r.get('completed_probes')==36 and r.get('completed_s6_probes')==32 and r.get('completed_old_historical')==4 and r.get('all_input_state_probability_gates') is True and r.get('zero_failed') is True and r.get('zero_imputed') is True,'REHEARSAL_COMPLETE_GATES')
    require(len(r.get('boards',[]))==2 and len({x['uuid'] for x in r['boards']})==2 and all(x['name']=='Tesla T4' for x in r['boards']),'TWO_PHYSICAL_T4_GATES')
    require(0<r.get('projection',{}).get('conservative_seconds',math.inf)<MAX_SECONDS and 0<r.get('elapsed_seconds',math.inf)<MAX_SECONDS,'NINE_HOUR_PROJECTION_GATE')
    require(r.get('csv_sha256')==file_sha(csv_path),'REHEARSAL_CSV_BINDING')
    with Path(csv_path).open(newline='') as h:rows=list(csv.reader(h))
    require(rows[0]==[UID]+LABELS and len(rows)==4 and [x[0] for x in rows[1:]]==r['visible_ids'],'VISIBLE_CSV_COVERAGE')
    probabilities([[float(x) for x in row[1:]] for row in rows[1:]],3)
    for arm in ('old','s6'):
        require(arm in r.get('phases',{}) and r['phases'][arm]['wall_seconds']>0 and r['phases'][arm]['peaks']['host_rss_bytes']>0 and r['phases'][arm]['peaks']['output_disk_bytes']>0 and r['phases'][arm]['minimum_disk_free_bytes']>0,'CONCURRENT_RESOURCE_MEASUREMENT')
        rr=r.get('workers',{}).get(arm,[]);require(len(rr)==2,'REHEARSAL_WORKERS')
        for board,x in enumerate(rr):
            require(x['status']=='PASS' and x['board']==board and x['state_before']==x['state_after'] and x['completed']==len(x['assigned']) and x['peak_cuda_bytes']>0 and x['peak_rss_bytes']>0 and x['zero_failed'] is True and x['zero_imputed'] is True and x['fp32_recoveries']==0,'REHEARSAL_WORKER_STATE_RESOURCE')
            require((x['state_before']['state']==binding['qualified_state']['old_state'] and x.get('constructor_grid_sha256')==binding['qualified_state']['old_grid']) if arm=='old' else x['state_before']==binding['qualified_state']['s6_state'],'QUALIFIED_COMPLETE_STATE_BINDING')
            require(x['environment']['torch']==binding['environment']['torch'] and x['environment']['packages']==binding['environment']['packages'],'QUALIFIED_ENVIRONMENT_BINDING')
            require(x['environment']['physical_uuid']==r['boards'][board]['uuid'] and x['environment']['cuda_device_count']==1,'ISOLATED_BOARD_RUNTIME')
            require(len(x['sentinels'])==4 and [s['id'] for s in x['sentinels']]==r['sentinel_ids'][arm] and all(s['strict_reload'] for s in x['sentinels']),'BOTH_BOARD_SENTINELS')
            for s in x['sentinels']:
                require(all(s['input'].get(k)==v for k,v in binding['sentinel_inputs'][arm][s['id']].items()),'BOUND_SENTINEL_INPUTS')
                import numpy as np
                p=np.array(s['probabilities']);q=np.array(s['reloaded_probabilities']);expected=np.array(binding['sentinel_probabilities'][arm][s['id']])
                require(max(float(np.max(np.abs(p-expected))),float(np.max(np.abs(q-expected))),float(np.max(np.abs(p-q))))<=1e-6,'SENTINEL_EXACT_REFERENCE_GATE')
                if arm=='s6':
                    require(np.array_equal(quantize([p]),quantize([expected])) and np.array_equal(quantize([q]),quantize([expected])),'SENTINEL_EXACT_ROUNDING_GATE')
                    for field,target in [('logits',p),('reloaded_logits',q)]:
                        z=np.asarray(s[field],dtype=np.float32);require(z.shape==(4,12) and np.isfinite(z).all(),'SENTINEL_RAW_FOUR_LOGITS')
                        mean=z.mean(0).astype(np.float64);sigmoid=np.exp(-np.logaddexp(0.,-mean))
                        require(float(np.max(np.abs(sigmoid-target)))<=1e-6,'SENTINEL_MEAN_LOGIT_SIGMOID')
        for a,b in zip(rr[0]['sentinels'],rr[1]['sentinels']):
            require(a['input']==b['input'] and max(abs(x-y) for x,y in zip(a['probabilities'],b['probabilities']))<=1e-6,'CROSS_BOARD_SENTINEL_PARITY')

    # All32 are assigned once, while each arm's four historical sentinels were
    # independently decoded/reloaded on both boards outside assigned ledgers.
    oldrows={row['id']:row for worker in r['workers']['old'] for row in worker['records']}
    s6rows={row['id']:row for worker in r['workers']['s6'] for row in worker['records']}
    require(len(oldrows)==len(s6rows)==39 and set(oldrows)==set(s6rows),'COMPLETE39_MECHANICS')
    require(set(binding['s6_probe_ids']).isdisjoint(binding['old_historical_ids']) and not set(binding['s6_probe_ids']+binding['old_historical_ids'])&set(r['visible_ids']),'DISJOINT39_MECHANICS')
    for sid in binding['s6_probe_ids']:
        require(oldrows[sid].get('source_reference_parity') is True and oldrows[sid].get('strict_reload') is True and 0<=oldrows[sid].get('reload_max_abs',math.inf)<=1e-6,'ALL32_OLD_SOURCE_REPEAT_RELOAD')
        require(s6rows[sid].get('qualified_error') is not None and 0<=s6rows[sid]['qualified_error']<=1e-6,'ALL32_S6_PRIOR_PREDICTION')


def run(root,cfg,out,*,started):
    """Shared complete two-arm core for rehearsal and hidden-test replacement."""
    out=Path(out);out.mkdir();csv_created=False;r={'status':'RUNNING','version':VERSION,'mode':cfg['mode'],'binding':cfg['binding']};save(out/'receipt.json',r)
    try:
        columns,visible,records=test_contract(root,LABELS,visible=cfg['mode']=='rehearsal');paths={s:str(Path(root)/'test_series') for s in visible};ids=list(visible)
        if cfg['mode']=='rehearsal':
            probes=cfg['probes'];require(len(probes)==36 and not set(probes)&set(ids),'FROZEN32_DISTINCT_PROBES');training=read_series(Path(root)/'train_series.csv')
            for sid in probes:require(sid in training,'PROBE_METADATA');records[sid]=training[sid];paths[sid]=str(Path(root)/'train_series')
            require(len(cfg['probe_order'])==36 and set(cfg['probe_order'])==set(probes),'FROZEN36_CANONICAL_ORDER');ids=cfg['probe_order']+ids
        cfg={**cfg,'staged_ids':ids,'assignment':assignments(ids)};save(out/'assignment.json',{'ids':ids,'assignment':cfg['assignment'],'policy':METHOD['assignment']})
        staged=out/'images-only';cfg.update(staged=str(staged),stage_binding=stage(staged,ids,records,paths))
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True);boards=[]
        for line in inventory.splitlines():
            index,uuid,name=[x.strip() for x in line.split(',')];boards.append({'index':int(index),'uuid':uuid,'name':name})
        require([b['index'] for b in boards]==[0,1] and len({b['uuid'] for b in boards})==2 and all(b['name']=='Tesla T4' for b in boards),'EXACT_TWO_T4_INVENTORY');cfg['boards']=boards
        r.update(boards=boards,visible_ids=visible,setup_seconds=time.monotonic()-started);save(out/'receipt.json',r);phases={};reports={};arms={}
        for arm in ('old','s6'):
            reports[arm],phases[arm]=launch(cfg,out,arm);arms[arm]=merge(ids,[(x,out/f'board{i}-{arm}'/arm/'predictions.npz') for i,x in enumerate(reports[arm])],arm)
            r.update(phases=phases,workers=reports);save(out/'receipt.json',r)
        merge_start=time.monotonic();index={s:i for i,s in enumerate(ids)};indices=[index[s] for s in visible]
        prediction=blend(arms['old'][indices],arms['s6'][indices]);csv_path=out/'visible-predictions.csv' if cfg['mode']=='rehearsal' else Path('/kaggle/working/submission.csv')
        require(time.monotonic()-started<MAX_SECONDS,'ACTUAL_NINE_HOUR_LIMIT_BEFORE_CSV')
        sha=write_csv(csv_path,columns,visible,prediction);csv_created=True
        r.update(status='PASS',completed=len(ids),completed_visible=len(visible),completed_probes=len(ids)-len(visible),completed_s6_probes=32 if cfg['mode']=='rehearsal' else 0,completed_old_historical=4 if cfg['mode']=='rehearsal' else 0,zero_failed=True,zero_imputed=True,csv_sha256=sha,csv_path=str(csv_path),
            all_input_state_probability_gates=True,sentinel_ids={'old':cfg.get('old_sentinels',[]),'s6':cfg.get('s6_sentinels',[])},elapsed_seconds=time.monotonic()-started,
            projection=projection(cfg,phases,reports,r['setup_seconds'],time.monotonic()-merge_start) if cfg['mode']=='rehearsal' else None)
        require(r['elapsed_seconds']<MAX_SECONDS,'ACTUAL_NINE_HOUR_LIMIT')
        if cfg['mode']=='rehearsal':require_rehearsal(r,cfg['binding'],csv_path)
        save(out/'receipt.json',r);return r
    except BaseException as e:
        if csv_created and cfg['mode']=='entry' and csv_path.exists():csv_path.replace(out/'rejected-submission.csv')
        r.update(status='INTERRUPTED' if isinstance(e,(KeyboardInterrupt,InterruptedError,TimeoutError)) else 'FAIL',failure_type=type(e).__name__,elapsed_seconds=time.monotonic()-started)
        save(out/'failure-private.json',{'type':type(e).__name__,'detail':str(e)});save(out/'receipt.json',r);raise


if __name__=='__main__':
    import argparse
    require(Path(__file__).resolve()==(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_s6_entry.py').resolve(),'EXACT_FRESH_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args();cfg=json.loads(Path(a.worker).read_bytes())
    (old_worker if cfg['arm']=='old' else s6_worker)(a.worker)
