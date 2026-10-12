"""Complete-study dual-T4 image-only fixed70 old320/30 S6 entry.

Torch is imported only in fresh board-isolated workers. Atomic ledgers remain
on failure. Only classified data-local rows may receive capped global arm means;
no partial merge, shard ranks or automatic resume.
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
from knee.public_s6_qualification import (LABELS,RECIPE,UID,SERIES_COLUMNS,canonical,digest,file_sha,require,save,verify_sources,verify_assets,synthetic_source_checks)
from knee.public_s6_evaluation import array_sha,model_digest,environment,save_predictions
from knee.public_adaptation_fit import PlatformInterruption,atomic_save
from knee.public_adaptation_pilot import state_fingerprint
from knee.public_fullfield_entry import safe_id,read_series,test_contract,write_csv

VERSION='s6-dual-capped-recovery-v1'
INPUT_POLICY={'version':'source-partial-input-v1','staging':'all metadata, exact confined source aliases, bound present/empty/absent inventory','recovered_diagnostics':['pixel_failures','fast_bad_headers','header_parse_failures'],'fatal_counters':['resource_errors','invalid_pixels'],'prediction_recovery':'global genuine arm means; union floor3% cap; whole actual population ranks', 'failure_classification':'classified-data-local-v1'}
METHOD={'old':'central320 K94 equal original/mirror probabilities','s6':RECIPE,'blend':[.7,.3],
        'rank':'pandas average pct over all actual test rows','s6_quantization':'source Python %.6f float64','assignment':'canonical ordinal modulo2'}
from knee.public_s6_recovery import recover, RECIPE as RECOVERY_RECIPE
from knee.public_s6_recovery_runtime import (DataLocalFailure, FatalSourceError, cleanup_study, install_source_guards, install_old_guards, save_ledger)
METHOD['recovery_recipe']=RECOVERY_RECIPE
MAX_SECONDS=9*3600
SENTINELS=[0,1,20,9]
rehearsal_hook=None


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


def source_availability(source,root):
    root=Path(root).resolve();source=Path(source)
    require(source.parent.resolve().is_relative_to(root) and source.resolve().is_relative_to(source.parent.resolve()),'IMAGE_PATH_ESCAPE')
    require(not source.parent.exists() or source.parent.is_dir(),'IMAGE_STUDY_NOT_DIRECTORY')
    require(not source.is_symlink() or source.exists(),'DANGLING_SOURCE_ALIAS')
    if source.exists():
        require(source.is_dir(),'IMAGE_PATH_NOT_DIRECTORY')
        for image in source.glob('*.dcm'):
            require(image.is_file() and image.resolve().is_relative_to(source.resolve()),'DICOM_PATH_ESCAPE')
        return 'present' if any(source.glob('*.dcm')) else 'empty'
    return 'absent'


def verify_staged(staged,ids,receipt):
    """Entry-local verifier preserves source-supported missing/empty series."""
    staged=Path(staged)
    require(set(p.name for p in staged.iterdir())=={'train.csv','train_series.csv','train_series'},'UNEXPECTED_STAGE_TABLE')
    require(not (staged/'train_series').is_symlink(),'STAGE_ROOT_ALIAS')
    for name in ('train.csv','train_series.csv'):
        require(not (staged/name).is_symlink() and file_sha(staged/name)==receipt['train_sha256' if name=='train.csv' else 'series_sha256'],'STAGED_BYTES_CHANGED')
    with (staged/'train.csv').open(newline='') as h:
        r=csv.DictReader(h);require(r.fieldnames==[UID] and [x[UID] for x in r]==ids,'STAGED_ID_ONLY_ORDER')
    with (staged/'train_series.csv').open(newline='') as h:
        r=csv.DictReader(h);require(r.fieldnames==SERIES_COLUMNS,'STAGED_SERIES_SCHEMA');rows=list(r)
    pairs=[(x[UID],x['SeriesInstanceUID']) for x in rows]
    require({u for u,_ in pairs}==set(ids) and len(pairs)==len(set(pairs)) and all(safe_id(u) and safe_id(s) for u,s in pairs),'STAGED_SERIES_MEMBERSHIP')
    require(digest(ids)==receipt['membership_sha256'] and digest(pairs)==receipt['series_membership_sha256'],'STAGED_MEMBERSHIP_BINDING')
    images=staged/'train_series';require({p.name for p in images.iterdir()}==set(ids),'STAGED_STUDY_MEMBERSHIP')
    for p in images.iterdir():require(p.is_dir() and not p.is_symlink(),'STAGED_STUDY_ALIAS')
    require({(p.parent.name,p.name) for p in images.glob('*/*')}==set(pairs),'EXPOSED_IMAGE_MEMBERSHIP')
    inventory=receipt['availability_inventory'];require(digest(inventory)==receipt['availability_sha256'] and len(inventory)==len(pairs),'AVAILABILITY_BINDING')
    paths=[]
    for (u,s),entry in zip(pairs,inventory):
        require(entry[:2]==[u,s] and len(entry)==5,'AVAILABILITY_ORDER')
        source,state,root=entry[2:];target=Path(root)/u/s;alias=images/u/s
        require(str(target.resolve())==source and alias.is_symlink() and str(alias.readlink())==source and str(alias.resolve())==source,'EXACT_SOURCE_ALIAS')
        require(source_availability(target,root)==state,'SOURCE_AVAILABILITY_CHANGED');paths.append([u,s,source])
    require(digest(paths)==receipt['image_paths_sha256'] and receipt['studies']==len(ids) and receipt['series']==len(pairs),'SOURCE_INVENTORY_BINDING')


def stage(staged,ids,records,image_paths):
    """ID-only alias, every metadata row, no images fabricated or removed."""
    staged=Path(staged);require(not staged.exists() and set(ids)==set(records)==set(image_paths),'FRESH_EXACT_STAGE')
    staged.mkdir(parents=True);(staged/'train_series').mkdir()
    with (staged/'train.csv').open('w',newline='') as h:
        w=csv.writer(h);w.writerow([UID]);w.writerows([[s] for s in ids])
    pairs=[];inventory=[];availability=[]
    with (staged/'train_series.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=SERIES_COLUMNS);w.writeheader()
        for sid in ids:
            require(safe_id(sid),'UNSAFE_ID')
            for row in records[sid]:
                series=row['SeriesInstanceUID'];require(safe_id(series) and row[UID]==sid,'STAGE_METADATA')
                root=Path(image_paths[sid]).resolve();source=root/sid/series;state=source_availability(source,root)
                p=staged/'train_series'/sid/series;p.parent.mkdir(exist_ok=True);p.symlink_to(source.resolve(),target_is_directory=True)
                w.writerow({k:row[k] for k in SERIES_COLUMNS});pairs.append((sid,series));inventory.append([sid,series,str(source.resolve())]);availability.append([sid,series,str(source.resolve()),state,str(root)])
    require(len(pairs)==len(set(pairs)),'DUPLICATE_STAGE_SERIES')
    r={'train_sha256':file_sha(staged/'train.csv'),'series_sha256':file_sha(staged/'train_series.csv'),'image_paths_sha256':digest(inventory),
       'membership_sha256':digest(ids),'series_membership_sha256':digest(pairs),'studies':len(ids),'series':len(pairs),
       'availability_inventory':availability,'availability_sha256':digest(availability)}
    verify_staged(staged,ids,r);return r


def old_decode_gate(counters):
    # prepare_study already runs the unchanged source volume/mask/empty gate.
    require(all(counters.get(k,0)==0 for k in INPUT_POLICY['fatal_counters']),'OLD_DECODE_FAILURE')


def mirror_stress(public,decoder):
    from unittest.mock import patch
    from knee import mirror_input as MI
    diagnostics=[];prepare=MI.prepare_study
    def observe(*a,**kw):
        value=prepare(*a,**kw);old_decode_gate(value[2])
        diagnostics.append({'reference':kw.get('reference',False),'counters':dict(value[2])})
        return value
    with patch.object(MI,'prepare_study',side_effect=observe):result=MI.stress_inputs(public,decoder)
    require(result.get('status')=='PASS' and result.get('cases')==dict.fromkeys(MI.STRESS_CASES,'PASS') and result.get('uses_raw_data') is False and result.get('uses_labels') is False,'ACTUAL_MIRROR_STRESS_REQUIRED')
    return {**result,'case_count':len(result['cases']),'successful_decode_count':len(diagnostics),'decode_diagnostics':diagnostics}


def old_ledger(path,ids,p):
    import numpy as np
    p=probabilities(p,len(ids));require(len(ids)==len(set(ids)),'DUPLICATE_LEDGER_ID')
    atomic_save(path,None,lambda _,h:np.savez(h,study_ids=np.asarray(ids),probabilities=p))


def merge(ids,shards,arm):
    import numpy as np
    expected=assignments(ids);by_id={};missing={};require(len(shards)==2,'TWO_SHARDS_REQUIRED')
    for board,(r,path) in enumerate(shards):
        require(r.get('version')==VERSION and r.get('status')=='PASS' and r.get('board')==board and r.get('arm')==arm and r.get('assigned')==expected[board] and r.get('completed')==len(expected[board]),'WORKER_TERMINAL_GATE')
        require(r.get('state_before')==r.get('state_after') and r.get('zero_imputed') is True and r.get('fp32_recoveries')==0,'WORKER_STATE_FAILURE_GATE')
        require(r.get('ledger_sha256')==file_sha(path),'ATOMIC_LEDGER_BINDING')
        with np.load(path,allow_pickle=False) as a:
            got=a['study_ids'].tolist();statuses=a['statuses'].tolist();p=a['probabilities']
            require(got==expected[board] and len(statuses)==len(got) and p.shape==(len(got),12),'SHARD_ID_ORDER_COVERAGE')
            good=np.asarray(statuses)=='genuine';require(set(statuses)<={'genuine','data_local_failure'},'EXPLICIT_STATUS_SCHEMA')
            probabilities(p[good],int(good.sum()));require(np.isnan(p[~good]).all(),'NO_WORKER_IMPUTATION')
            rows=r['records'];require([x['id'] for x in rows]==got and [x['status'] for x in rows]==statuses,'LEDGER_RECORD_STATUS_BINDING')
            require(r['genuine']==int(good.sum()) and r['failed']==int((~good).sum()),'WORKER_STATUS_COUNTS')
            require(all(x.get('cleanup',{}).get('scratch_removed') is True and x.get('cleanup',{}).get('side_override_removed') is True and x.get('state_healthy') is True for x in rows),'PER_STUDY_CLEANUP_STATE')
            if arm=='s6':
                z=a['logits'];require(z.shape==(len(got),4,12) and np.isfinite(z[good]).all() and np.isnan(z[~good]).all(),'FOUR_READER_GENUINE_LOGITS_REQUIRED')
            for sid,row,status in zip(got,p,statuses):
                require(sid not in by_id,'DUPLICATE_MERGE_ID');by_id[sid]=row;missing[sid]=status=='data_local_failure'
    require(set(by_id)==set(ids),'COMPLETE_STATUS_ARMS_REQUIRED')
    return np.array([by_id[s] for s in ids]),np.array([missing[s] for s in ids],dtype=bool)


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
    r={'status':'RUNNING','version':VERSION,'board':cfg['board'],'arm':'old','assigned':cfg['dev'],'completed':0,'records':[],'sentinels':[]};save(out/'receipt.json',r)
    try:
        r['environment']=board_environment(torch,cfg);verify_staged(cfg['staged'],cfg['staged_ids'],cfg['stage_binding'])
        public,decoder=load_definitions(cfg['public_source'],cfg['decoder_source'],cfg['public_manifest'])
        if cfg['mode']=='rehearsal' and cfg['board']==0:r['partial_input_stress']=dict(mirror_stress(public,decoder),guarded=False,policy='baseline_source_before_recovery_guards');save(out/'receipt.json',r)
        r['source_guard_receipt']=install_old_guards(public,decoder)
        if cfg['mode']=='rehearsal' and cfg['board']==0:
            from knee.public_s6_recovery_rehearsal import guarded_mirror_partial
            r['guarded_partial_input_stress']=guarded_mirror_partial(public,decoder);save(out/'receipt.json',r)
        model=load_model(public,cfg['control_checkpoint'],'cuda',cfg['public_manifest']);before=snapshot(model);r['state_before']=before
        require(before['state']==cfg['control_state_sha256'],'QUALIFIED_OLD_STATE')
        grid={k:v.detach().cpu().clone() for k,v in model.named_buffers() if k in GRID_NAMES};r['constructor_grid_sha256']=state_fingerprint(grid);require(r['constructor_grid_sha256']==cfg['control_grid_sha256'],'OLD_CONSTRUCTOR_GRID')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
        records=read_series(Path(cfg['staged'])/'train_series.csv');frozen=out/'frozen-state.pt';atomic_save(frozen,{k:v.detach().cpu() for k,v in model.state_dict().items()})
        def infer(sid):
            t=time.monotonic()
            v,m,counters=prepare_study(public,decoder,sid,records,Path(cfg['staged'])/'train_series')
            old_decode_gate(counters)
            reference_seconds=0.
            if sid in cfg.get('probes',{}):
                reference_start=time.monotonic()
                rv,rm,rc=prepare_study(public,decoder,sid,records,Path(cfg['staged'])/'train_series',reference=True)
                require(np.array_equal(v,rv) and np.array_equal(m,rm),'OLD_ORIGINAL_SOURCE_REFERENCE_INPUT')
                old_decode_gate(rc)
                refinputs=[public.eval_windows(rv,rm,94,320,mirror=b) for b in (False,True)]
                refp=np.asarray(public.infer_pair_probs(model,*refinputs,'cuda'),dtype=np.float32);torch.cuda.synchronize();reference_seconds=time.monotonic()-reference_start
            inputs=[public.eval_windows(v,m,94,320,mirror=b) for b in (False,True)];p=np.asarray(public.infer_pair_probs(model,*inputs,'cuda'),dtype=np.float32);probabilities([p],1)
            if sid in cfg.get('probes',{}):require(float(np.max(np.abs(p-refp)))<=1e-6,'OLD_ORIGINAL_SOURCE_REFERENCE_PROBABILITY')
            fp={'volume_sha256':hashlib.sha256(v.tobytes()).hexdigest(),'mask_sha256':hashlib.sha256(m.tobytes()).hexdigest(),**window_audit(inputs,reference=refinputs if sid in cfg.get('probes',{}) else None)}
            expected_old=cfg.get('probes',{}).get(sid,{}).get('old',{})
            if 'qualified_full384_windows' in expected_old.get('input',{}):
                audit_start=time.monotonic();full=[fullfield_windows(public,v,m,mirror=b) for b in (False,True)]
                fp.update(window_audit(inputs,full=full,reference=refinputs));del full;reference_seconds+=time.monotonic()-audit_start
            torch.cuda.synchronize();return p,fp,{'seconds':time.monotonic()-t,'production_seconds':time.monotonic()-t-reference_seconds,'reference_seconds':reference_seconds,'decode_counters':counters,'reference_decode_counters':rc if sid in cfg.get('probes',{}) else None}
        ps=[];statuses=[]
        with PlatformInterruption() as stop:
            for sid in cfg.get('old_sentinels',[]):
                p,fp,d=infer(sid);probe_gate(sid,'old',p,fp,cfg)
                model.load_state_dict(torch.load(frozen,map_location='cpu',weights_only=True),strict=True);model.eval();require_immutable(model,before)
                q,g,_=infer(sid);probe_gate(sid,'old',q,g,cfg)
                require(g==fp and float(np.max(np.abs(p-q)))<=1e-6,'OLD_FRESH_DECODE_RELOAD')
                r['sentinels'].append({'id':sid,'input':fp,'probabilities':p.tolist(),'reloaded_probabilities':q.tolist(),'strict_reload':True});save(out/'receipt.json',r)
                if stop.requested:raise InterruptedError('Interrupted sentinel')
            if cfg['mode']=='rehearsal':
                require(rehearsal_hook is not None,'FRESH_INJECTION_EXTENSION_REQUIRED')
                r['recovery_injection']=rehearsal_hook('old',cfg,{'infer':infer,'sentinels':r['sentinels'],'state_check':lambda:require_immutable(model,before)});save(out/'receipt.json',r)
            for sid in cfg['dev']:
                study_started=time.monotonic()
                try:
                    r['current_id']=sid;save(out/'receipt.json',r);p,fp,d=infer(sid);err=probe_gate(sid,'old',p,fp,cfg)
                    if sid in cfg.get('probes',{}):
                        model.load_state_dict(torch.load(frozen,map_location='cpu',weights_only=True),strict=True);model.eval();require_immutable(model,before)
                        q,g,_=infer(sid);probe_gate(sid,'old',q,g,cfg);require(fp==g and float(np.max(np.abs(p-q)))<=1e-6,'OLD32_REPEAT_RELOAD')
                        d.update(source_reference_parity=True,strict_reload=True,reload_max_abs=float(np.max(np.abs(p-q))))
                    ps.append(p);statuses.append('genuine');row={'id':sid,'status':'genuine','input':fp,'qualified_error':err,**d}
                except DataLocalFailure as e:
                    require(cfg['mode']!='rehearsal','HEALTHY39_MUST_BE_GENUINE')
                    ps.append(np.full(12,np.nan));statuses.append('data_local_failure')
                    row={'id':sid,'status':'data_local_failure','phase':'old_input','failure_code':str(e),'production_seconds':time.monotonic()-study_started}
                    save(out/('failure-'+str(len(ps)-1)+'-private.json'),{'id':sid,'type':type(e).__name__,'detail':e.detail,'site':e.site,'phase':'old_input'})
                finally:
                    state_started=time.monotonic()
                    if sys.exc_info()[0] is None:require_immutable(model,before)
                    state_seconds=time.monotonic()-state_started
                require(statuses.count('data_local_failure')<=math.floor(3*len(cfg['staged_ids'])/100),'GLOBAL_UNION_CAP_ALREADY_EXCEEDED')
                row['production_seconds']+=state_seconds
                row.update(cleanup={'scratch_removed':True,'side_override_removed':True},state_healthy=True)
                r['records'].append(row);r['completed']=len(ps)
                save_ledger(out/'predictions.npz',cfg['dev'][:len(ps)],ps,statuses);save(out/'receipt.json',r)
                if stop.requested:raise InterruptedError('Interrupted study boundary')
        require_immutable(model,before);r.update(status='PASS',state_after=snapshot(model),zero_failed=all(s=='genuine' for s in statuses),genuine=statuses.count('genuine'),failed=statuses.count('data_local_failure'),zero_imputed=True,fp32_recoveries=0,
            ledger_sha256=file_sha(out/'predictions.npz'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        frozen.unlink()
    except BaseException as e:
        actual=e.error if isinstance(e,FatalSourceError) else e
        save(out/'failure-private.json',{'type':type(actual).__name__,'detail':str(actual),'boundary_type':type(e).__name__});r.update(status='INTERRUPTED' if isinstance(actual,(InterruptedError,KeyboardInterrupt,TimeoutError)) else 'FAIL',failure_type=type(actual).__name__)
        if actual is not e:raise actual from e
        raise
    finally:r['elapsed_seconds']=time.monotonic()-start;save(out/'receipt.json',r)


def reset_native_renderer(cfg):
    """Restore the pinned original module before its one-time lean patch.

    Fresh sentinel datasets must not patch a previously patched prepare.
    The exact scientific source remains unchanged and its guard runs again.
    """
    import importlib
    native=importlib.import_module('training.native_cache')
    path=(Path(cfg['code'])/'training/native_cache.py').resolve()
    require(Path(native.__file__).resolve()==path,'EXACT_NATIVE_SOURCE_PATH')
    require(file_sha(path)==cfg['source_pins']['training/native_cache.py'],'EXACT_NATIVE_SOURCE_BYTES')
    return importlib.reload(native)


def native_availability_stress(cfg,*,guarded=False):
    """Actual pinned lean prepare and OneStudySource; synthetic images only."""
    import contextlib
    import io
    import tempfile
    import numpy as np
    import pydicom
    import probe_infer as PI
    native=reset_native_renderer(cfg);previous=PI.OneStudySource.current;checks={}
    try:
        native=PI.install_native(lean=True)
        if guarded:
            import compact_infer as CI
            install_source_guards(CI,None,cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'direct';root.mkdir();uid='synthetic';series='usable'
            rows=[dict(zip(SERIES_COLUMNS,[uid,series,'1','0','Sagittal'])),dict(zip(SERIES_COLUMNS,[uid,'unavailable','1','0','Coronal']))]
            with (root/'train.csv').open('w',newline='') as h:w=csv.writer(h);w.writerow([UID]);w.writerow([uid])
            with (root/'train_series.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=SERIES_COLUMNS);w.writeheader();w.writerows(rows)
            images=root/'train_series'/uid/series;images.mkdir(parents=True)
            for i in range(4):
                meta=pydicom.Dataset();meta.TransferSyntaxUID=pydicom.uid.ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID=pydicom.uid.MRImageStorage;meta.MediaStorageSOPInstanceUID='1.2.826.0.1.3680043.8.498.'+str(i+1)
                ds=pydicom.FileDataset(None,{},file_meta=meta,preamble=bytes(128));ds.SOPClassUID=meta.MediaStorageSOPClassUID;ds.SOPInstanceUID=meta.MediaStorageSOPInstanceUID
                ds.Rows=32;ds.Columns=40;ds.SamplesPerPixel=1;ds.BitsAllocated=16;ds.BitsStored=12;ds.HighBit=11;ds.PixelRepresentation=0
                ds.PhotometricInterpretation='MONOCHROME2';ds.ImageOrientationPatient=[0,1,0,0,0,1];ds.ImagePositionPatient=[i,0,0]
                ds.PixelSpacing=[.5,.5];ds.InstanceNumber=i;ds.ImageLaterality='L'
                ds.PixelData=((np.arange(1280,dtype=np.uint16).reshape(32,40)+i*11)%2000).tobytes()
                pydicom.dcmwrite(images/f'{i}.dcm',ds,enforce_file_format=True)
            PI.OneStudySource.current=uid
            for case in ('absent','empty'):
                if case=='empty':(root/'train_series'/uid/'unavailable').mkdir()
                alias=Path(tmp)/('stage-'+case);binding=stage(alias,[uid],{uid:rows},{uid:root/'train_series'});verify_staged(alias,[uid],binding)
                with contextlib.redirect_stdout(io.StringIO()):
                    direct=native.prepare(root,Path(tmp)/('direct-out-'+case),split='train',study_ids=[uid],source_directory=True)
                    staged=native.prepare(alias,Path(tmp)/('stage-out-'+case),split='train',study_ids=[uid],source_directory=True)
                require(direct['failed_studies']==staged['failed_studies']==[],'NATIVE_USABLE_COMPANION')
                a=direct['studies'][0];b=staged['studies'][0]
                require(len(a['series'])==len(b['series'])==1 and a['errors']==b['errors'],'NATIVE_AVAILABILITY_SOURCE_EQUAL')
                x=a['series'][0];y=b['series'][0];v=np.load(x['path'],allow_pickle=False);w=np.load(y['path'],allow_pickle=False)
                require(np.array_equal(v,w) and {k:z for k,z in x.items() if k!='path'}=={k:z for k,z in y.items() if k!='path'},'NATIVE_ARRAY_METADATA_EQUAL')
                checks[case]={'status':'PASS','usable_series':1,'slices':len(v),'array_sha256':array_sha(v),'metadata_sha256':digest({k:z for k,z in x.items() if k!='path'})}
    finally:
        PI.OneStudySource.current=previous;reset_native_renderer(cfg)
    return {'status':'PASS','case_count':2,'cases':checks,'uses_raw_data':False,'uses_labels':False,'reader':'pinned probe_infer.OneStudySource and install_native(lean=True)','guarded':guarded,'policy':'classified-data-local-v1' if guarded else 'baseline_source_before_recovery_guards'}


def s6_worker(config_path):
    import numpy as np
    import torch
    import resource
    import setuptools
    import distutils.util
    cfg=json.loads(Path(config_path).read_bytes());out=Path(cfg['out'])/'s6';out.mkdir();start=time.perf_counter()
    r={'status':'RUNNING','version':VERSION,'phase':'source_state','board':cfg['board'],'arm':'s6','assigned':cfg['dev'],'completed':0,'optimizer_updates':0,'records':[]};save(out/'receipt.json',r)
    try:
        r['environment']=board_environment(torch,cfg);save(out/'receipt.json',r)
        verify_sources(cfg['code'],cfg['source_pins']);verify_assets(cfg['compact'],cfg['anatomy'],cfg['manifest'],cfg['asset_binding'])
        verify_staged(cfg['staged'],cfg['staged_ids'],cfg['stage_binding'])
        require(sys.version_info[:2]==(3,12) and r['environment']['packages']['timm']=='1.0.26' and torch.cuda.is_available() and torch.cuda.get_device_name(0)=='Tesla T4' and torch.__version__=='2.10.0+cu128','S6_CUDA_ENVIRONMENT')
        require('monai' not in sys.modules,'FRESH_MONAI_NAMESPACE');sys.path.insert(0,cfg['code']);os.environ['RSNA_CODE_ROOT']=cfg['code']
        import compact_infer as CI
        import compact_train as CT
        if cfg['mode']=='rehearsal' and cfg['board']==0:r['native_availability_stress']=native_availability_stress(cfg);r['guarded_native_availability_stress']=native_availability_stress(cfg,guarded=True);save(out/'receipt.json',r)
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
            record=original_record(*a,**kw);capture['anchors']=record;return record
        def observe_build(uid):
            item=original_build(uid);capture['item']=item
            study=json.loads((Path(ds.builder.cache)/'studies'/(hashlib.sha256(uid.encode()).hexdigest()[:24]+'.json')).read_bytes())
            capture['source_diagnostics']={'study_errors':study.get('errors',[]),'series_errors':[s.get('errors',[]) for s in study['series']],'skipped_series':item['skipped_series']}
            capture['scratch_bytes']=sum(p.stat().st_size for p in Path(ds.builder.cache).rglob('*') if p.is_file());return item
        def observe_rows(*a,**kw):
            rows=original_rows(*a,**kw);capture['rows']=rows;return rows
        ds.bank.record=observe_record;ds.builder.build=observe_build;ds.window_rows=observe_rows
        r['source_guard_receipt']=install_source_guards(CI,ds,cfg)
        original_record=ds.bank.__class__.record.__get__(ds.bank)
        original_build=ds.builder.__class__.build.__get__(ds.builder)
        ds.bank.record=observe_record;ds.builder.build=observe_build
        if cfg['mode']=='rehearsal':
            require(rehearsal_hook is not None,'FRESH_INJECTION_EXTENSION_REQUIRED')
            r['recovery_injection']=rehearsal_hook('s6',cfg,{'CI':CI,'ds':ds,'probes':cfg['probes'],'state_check':lambda:require(model_digest(states)==initial,'INJECTION_STATE_PARITY')});save(out/'receipt.json',r)
        logits=[];probs=[];statuses=[]
        with PlatformInterruption() as stop:
            for i,sid in enumerate(all_ids):
                if i<len(pilot_ids):
                    require(model_digest(states)==initial,'S6_STATE_BEFORE_RELOAD')
                    for m,name in zip(models,cfg['manifest']['models']):
                        m.load_state_dict(torch.load(Path(cfg['compact'])/name,map_location='cpu',weights_only=True),strict=True);m.eval()
                    CI.FORCED_SIDE.clear()
                    reset_native_renderer(cfg)
                    ds=CI.CStudies(cfg['staged'],'train',all_ids,out/'scratch',cfg['code'],384,1,'v4',3,cfg['anatomy'],3.5,str(Path(cfg['compact'])/'side_cls.pt'),2.,0.,20.,extent_meta=any(m.extent for m in models));ds._setup()
                    states=models+[ds.bank.model,ds.side_net];require(model_digest(states)==initial,'S6_STRICT_RELOAD_STATE')
                    ds.bank.model.register_forward_hook(finite_aux);ds.side_net.register_forward_hook(finite_aux)
                    original_build=ds.builder.build;original_rows=ds.window_rows;original_record=ds.bank.record
                    r['source_guard_receipt']=install_source_guards(CI,ds,cfg)
                    original_record=ds.bank.__class__.record.__get__(ds.bank)
                    original_build=ds.builder.__class__.build.__get__(ds.builder)
                    ds.bank.record=observe_record;ds.builder.build=observe_build;ds.window_rows=observe_rows
                try:
                    study_start=time.perf_counter();capture.clear();r['phase']='pilot_sentinel' if i<len(pilot_ids) else 'evaluation';r['current_ordinal']=i-len(pilot_ids);save(out/'receipt.json',r)
                    item=ds[i]
                    if item['error'] is not None:
                        save(out/'current-study-error-private.json',{'ordinal':r['current_ordinal'],'study':sid,'source_error':item['error']})
                    require(item['error'] is None and len(item['images'])>0,'UNCLASSIFIED_FAILED_STUDY')
                    a=item['images'].numpy();meta=item['meta'].numpy();require(a.dtype==np.uint8 and a.shape[1:]==(3,384,384) and np.isfinite(meta).all(),'S6_INPUT_SCHEMA')
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
                        row.update(id=sid,status='genuine',source_diagnostics=capture.get('source_diagnostics',{}),qualified_error=probe_gate(sid,'s6',p,fp,cfg));logits.append(z);probs.append(p);statuses.append('genuine')
                        r['completed']=len(probs)
                    row['production_seconds']=time.perf_counter()-study_start
                    del img,mt,mask,zs,item,a,meta
    
                except DataLocalFailure as e:
                    require(i>=len(pilot_ids) and cfg['mode']!='rehearsal','HEALTHY39_MUST_BE_GENUINE')
                    probs.append(np.full(12,np.nan));logits.append(np.full((4,12),np.nan,dtype=np.float32));statuses.append('data_local_failure')
                    row={'id':sid,'status':'data_local_failure','phase':'s6_input','failure_code':str(e),'production_seconds':time.perf_counter()-study_start}
                    r['completed']=len(probs)
                    save(out/('failure-'+str(len(probs)-1)+'-private.json'),{'id':sid,'type':type(e).__name__,'detail':e.detail,'site':e.site,'phase':'s6_input'})
                finally:
                    clean=cleanup_study(out/'scratch',sid,CI.FORCED_SIDE);capture.clear()
                    if sys.exc_info()[0] is None:require(model_digest(states)==initial,'S6_STATE_AFTER_EACH_STUDY')
                require(statuses.count('data_local_failure')<=math.floor(3*len(cfg['staged_ids'])/100),'GLOBAL_UNION_CAP_ALREADY_EXCEEDED')
                row.update(cleanup=clean,state_healthy=True)
                row['production_seconds']=time.perf_counter()-study_start
                if i>=len(pilot_ids):
                    r['records'].append(row);save_ledger(out/'predictions.npz',cfg['dev'][:len(probs)],probs,statuses,logits=logits)
                r['elapsed_seconds']=time.perf_counter()-start;save(out/'receipt.json',r)
                if stop.requested:raise InterruptedError('Platform interruption at study boundary')
        require(len(probs)==len(cfg['dev']),'COMPLETE_ASSIGNED_REQUIRED');r['state_after']=model_digest(states);require(r['state_after']==initial,'S6_PARAMETER_BUFFER_MUTATION')
        r.update(status='PASS',genuine=statuses.count('genuine'),failed=statuses.count('data_local_failure'),zero_imputed=True,zero_failed=all(s=='genuine' for s in statuses),fp32_recoveries=0,ledger_sha256=file_sha(out/'predictions.npz'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved());save(out/'receipt.json',r)
    except BaseException as e:
        actual=e.error if isinstance(e,FatalSourceError) else e
        save(out/'failure-private.json',{'type':type(actual).__name__,'detail':str(actual),'boundary_type':type(e).__name__});r.update(status='INTERRUPTED' if isinstance(actual,(InterruptedError,KeyboardInterrupt,TimeoutError)) else 'FAIL',failure_type=type(actual).__name__,elapsed_seconds=time.perf_counter()-start);save(out/'receipt.json',r)
        if actual is not e:raise actual from e
        raise


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
                child=subprocess.Popen([sys.executable,'-m',('knee.public_s6_recovery_rehearsal' if cfg['mode']=='rehearsal' else 'knee.public_s6_recovery_entry'),'--worker',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT);log.close();children.append(child)
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
    require(binding.get('version')==VERSION and binding.get('method')==METHOD and binding.get('input_policy')==INPUT_POLICY,'EXACT_FROZEN_METHOD')
    require(r.get('status')=='PASS' and r.get('version')==VERSION and r.get('mode')=='rehearsal' and r.get('binding')==binding,'PASS_EXACT_REHEARSAL_REQUIRED')
    require(r.get('completed')==39 and r.get('completed_visible')==3 and r.get('completed_probes')==36 and r.get('completed_s6_probes')==32 and r.get('completed_old_historical')==4 and r.get('all_input_state_probability_gates') is True and r.get('zero_failed') is True and r.get('zero_imputed') is True,'REHEARSAL_COMPLETE_GATES')
    require(len(r.get('boards',[]))==2 and len({x['uuid'] for x in r['boards']})==2 and all(x['name']=='Tesla T4' for x in r['boards']),'TWO_PHYSICAL_T4_GATES')
    require(0<r.get('projection',{}).get('conservative_seconds',math.inf)<MAX_SECONDS and 0<r.get('elapsed_seconds',math.inf)<MAX_SECONDS,'NINE_HOUR_PROJECTION_GATE')
    require(r.get('csv_sha256')==file_sha(csv_path),'REHEARSAL_CSV_BINDING')
    with Path(csv_path).open(newline='') as h:rows=list(csv.reader(h))
    require(rows[0]==[UID]+LABELS and len(rows)==4 and [x[0] for x in rows[1:]]==r['visible_ids'],'VISIBLE_CSV_COVERAGE')
    probabilities([[float(x) for x in row[1:]] for row in rows[1:]],3)
    from knee.mirror_input import STRESS_CASES
    stress=r.get('workers',{}).get('old',[{}])[0].get('partial_input_stress',{})
    require(stress.get('status')=='PASS' and stress.get('cases')==dict.fromkeys(STRESS_CASES,'PASS') and stress.get('case_count')==17 and stress.get('uses_raw_data') is False and stress.get('uses_labels') is False,'REHEARSAL_PARTIAL_INPUT_STRESS')
    guarded=r.get('workers',{}).get('old',[{}])[0].get('guarded_partial_input_stress',{})
    require(guarded.get('status')=='PASS' and guarded.get('guarded') is True and guarded.get('case_count')==10 and set(guarded.get('cases',{}))=={'healthy','missing_iop','missing_ipp','missing_spacing','empty_series','missing_series','bad_header','bad_pixel','black_frame','short_series'},'GUARDED_MIRROR_PARTIAL_TEN')
    guarded_native=r.get('workers',{}).get('s6',[{}])[0].get('guarded_native_availability_stress',{})
    require(guarded_native.get('status')=='PASS' and guarded_native.get('guarded') is True and guarded_native.get('case_count')==2 and set(guarded_native.get('cases',{}))=={'absent','empty'},'GUARDED_NATIVE_PARTIAL_TWO')
    native=r.get('workers',{}).get('s6',[{}])[0].get('native_availability_stress',{})
    require(native.get('status')=='PASS' and native.get('case_count')==2 and native.get('uses_raw_data') is False and native.get('uses_labels') is False and set(native.get('cases',{}))=={'absent','empty'} and all(x.get('status')=='PASS' and x.get('usable_series')==1 and x.get('slices',0)>0 for x in native['cases'].values()),'REHEARSAL_NATIVE_AVAILABILITY_STRESS')
    for arm in ('old','s6'):
        require(arm in r.get('phases',{}) and r['phases'][arm]['wall_seconds']>0 and r['phases'][arm]['peaks']['host_rss_bytes']>0 and r['phases'][arm]['peaks']['output_disk_bytes']>0 and r['phases'][arm]['minimum_disk_free_bytes']>0,'CONCURRENT_RESOURCE_MEASUREMENT')
        rr=r.get('workers',{}).get(arm,[]);require(len(rr)==2,'REHEARSAL_WORKERS')
        require(all(x.get('recovery_injection',{}).get('status')=='PASS' and x['recovery_injection'].get('failure_status')=='data_local_failure' and x['recovery_injection'].get('injected_rows_genuine') is False and x['recovery_injection'].get('state_healthy') is True for x in rr),'ACTUAL_CLASSIFIED_RECOVERY_INJECTIONS')
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
        staged=out/'images-only';cfg.update(staged=str(staged),stage_binding=stage(staged,ids,records,paths));r['stage_binding']=cfg['stage_binding'];save(out/'receipt.json',r)
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True);boards=[]
        for line in inventory.splitlines():
            index,uuid,name=[x.strip() for x in line.split(',')];boards.append({'index':int(index),'uuid':uuid,'name':name})
        require([b['index'] for b in boards]==[0,1] and len({b['uuid'] for b in boards})==2 and all(b['name']=='Tesla T4' for b in boards),'EXACT_TWO_T4_INVENTORY');cfg['boards']=boards
        r.update(boards=boards,visible_ids=visible,setup_seconds=time.monotonic()-started);save(out/'receipt.json',r);phases={};reports={};arms={}
        for arm in ('old','s6'):
            reports[arm],phases[arm]=launch(cfg,out,arm);arms[arm]=merge(ids,[(x,out/f'board{i}-{arm}'/arm/'predictions.npz') for i,x in enumerate(reports[arm])],arm)
            r.update(phases=phases,workers=reports);save(out/'receipt.json',r)
        merge_start=time.monotonic();index={s:i for i,s in enumerate(ids)};indices=[index[s] for s in visible]
        recovery=recover(arms['old'][0][indices],arms['s6'][0][indices],arms['old'][1][indices],arms['s6'][1][indices])
        prediction=recovery['blend'];r['recovery']={k:v for k,v in recovery.items() if k not in ('control','s6','control_rank','s6_rank','blend')}
        csv_path=out/'visible-predictions.csv' if cfg['mode']=='rehearsal' else Path('/kaggle/working/submission.csv')
        require(time.monotonic()-started<MAX_SECONDS,'ACTUAL_NINE_HOUR_LIMIT_BEFORE_CSV')
        sha=write_csv(csv_path,columns,visible,prediction);csv_created=True
        r.update(status='PASS',completed=len(ids),completed_visible=len(visible),completed_probes=len(ids)-len(visible),completed_s6_probes=32 if cfg['mode']=='rehearsal' else 0,completed_old_historical=4 if cfg['mode']=='rehearsal' else 0,zero_failed=recovery['union_recovered']==0,zero_imputed=recovery['union_recovered']==0,csv_sha256=sha,csv_path=str(csv_path),
            all_required_recipe_gates=True,all_input_state_probability_gates=recovery['union_recovered']==0,sentinel_ids={'old':cfg.get('old_sentinels',[]),'s6':cfg.get('s6_sentinels',[])},elapsed_seconds=time.monotonic()-started,
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
    require(Path(__file__).resolve()==(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_s6_recovery_entry.py').resolve(),'EXACT_FRESH_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args();cfg=json.loads(Path(a.worker).read_bytes())
    (old_worker if cfg['arm']=='old' else s6_worker)(a.worker)
