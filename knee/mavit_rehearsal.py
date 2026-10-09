"""Separate image-only MaViT67 qualification; no fusion, fitting or submission."""
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import re
import shutil
import sys
import time
import warnings

from knee import mavit_checkpoint as checkpoint, mavit_input as inputs
from knee.reference import bounded_ids, validate_predictions
from knee.reference_blend import read_test_contract, write_probabilities

VERSION='mavit-image67-v1'
CHECKPOINT_SHA256='7acd2a990a89c7a7d73ec0488badc5058a61b0c77142742c532ebc196749bdbc'
CPU_REPORT_SHA256='8d694907b0cf70504da2b33644db1ea3a18a057365987bfa59b4f8ac55a2d1e6'
STATE_SCHEMA='84594142eaf7e23138196a5c442171a9d005b2db2d249fedebe31b6d8f70ceb0'
MAX_SECONDS=32400
FLAGS={'cudnn_benchmark':False,'cuda_matmul_allow_tf32':False,'cudnn_deterministic':False,'cudnn_allow_tf32':True,'torch_threads':4}
PACKAGES={'torch':'2.10.0+cu128','timm':'1.0.26','einops':'0.8.2','numpy':'2.0.2','torchvision':'0.25.0+cu128',
          'pydicom':'3.0.2','pandas':'2.3.3'}
SERIES_COLUMNS=['StudyInstanceUID','SeriesInstanceUID','Fluid_Sensitive','Fat_Suppression','Anatomical_Plane']
STRESS_CASES={'healthy_public','missing_view','short_mask_no_retry','shape_tie_mask','unsafe_pixels_mask','black_view_mask_preserved',
              'whole_black_fatal','unknown_exception_fatal','memory_fatal','timeout_fatal','descriptor_fatal',
              'missing_geometry_defaults','pixel_decode_mask','inner_program_error_fatal','multiframe_mask','whole_empty_fatal'}


def cpu_evidence(path,source_dir,root):
    raw=Path(path).read_bytes()
    if checkpoint.digest(raw)!=CPU_REPORT_SHA256:raise ValueError('Authentic reviewed CPU checkpoint receipt required')
    r=json.loads(raw);_,proof=checkpoint.source_contract(source_dir)
    if (r['status']!='PASS' or r['source_contract']!=proof or r['checkpoint']['sha256']!=CHECKPOINT_SHA256 or
            r['state_compatibility']['actual_schema_sha256']!=STATE_SCHEMA or not r['state_compatibility']['strict_load']):
        raise ValueError('CPU architecture qualification differs')
    for n,d in r['runtime_sha256'].items():
        if checkpoint.digest((Path(root)/n).read_bytes())!=d:raise ValueError('CPU-audited model helper changed')
    return r


def read_series(root,split):
    rows=[];seen=set()
    with (Path(root)/f'{split}_series.csv').open(newline='') as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=SERIES_COLUMNS:raise ValueError('Official MaViT series schema differs')
        for r in reader:
            if set(r)!=set(SERIES_COLUMNS) or any(r[k] is None for k in SERIES_COLUMNS):raise ValueError('Malformed metadata row')
            for k in SERIES_COLUMNS[:2]:bounded_ids([r[k]],count=1)
            if r['SeriesInstanceUID'] in seen:raise ValueError('Duplicate series metadata')
            seen.add(r['SeriesInstanceUID'])
            if r['Fluid_Sensitive'] not in ('0','1') or r['Fat_Suppression']!=r['Fluid_Sensitive'] or r['Anatomical_Plane'] not in ('Sagittal','Axial','Coronal'):
                raise ValueError('Unsupported official MaViT view key')
            r['Fluid_Sensitive']=int(r['Fluid_Sensitive']);r['Fat_Suppression']=int(r['Fat_Suppression']);rows.append(r)
    by_study={}
    for r in rows:by_study.setdefault(r['StudyInstanceUID'],[]).append(r)
    return by_study


def load_model(source_dir):
    import torch
    path=Path(source_dir)/checkpoint.CHECKPOINT
    with path.open('rb') as h:
        if hashlib.file_digest(h,'sha256').hexdigest()!=CHECKPOINT_SHA256:raise ValueError('Pinned MaViT checkpoint changed')
    source,_=checkpoint.source_contract(source_dir);models=[];audit={}
    def factory():
        m=checkpoint.instantiate(source);models.append(m);return m
    checkpoint.inspect_checkpoint(path,factory,audit)
    if audit['state_compatibility']['actual_schema_sha256']!=STATE_SCHEMA:raise ValueError('CPU-audited state schema changed')
    return models.pop().to('cuda').eval(),audit


def predict(model,arrays):
    import numpy as np
    import torch
    if model.training or any(m.training for m in model.modules()):raise ValueError('MaViT must be eval')
    volume,mask,positions=arrays
    x=torch.from_numpy(volume).unsqueeze(0).byte().to('cuda')
    m=torch.from_numpy(mask).unsqueeze(0).bool().to('cuda')
    p=torch.from_numpy(positions).unsqueeze(0).float().to('cuda')
    with torch.no_grad(),torch.amp.autocast('cuda',dtype=torch.float16):
        logits=model.infer(x,m,p)
        if logits.shape!=(1,12) or not torch.isfinite(logits).all():raise ValueError('MaViT logits shape/finite failure')
        probability=torch.sigmoid(logits)
    values=probability.detach().float().cpu().numpy()[0]
    return validate_predictions(values)


def stress_inputs(source):
    """Tiny simulated series through actual source selection/sampling/guards, no raw mutation."""
    import numpy as np
    from unittest.mock import patch
    prod=inputs.definitions(source,production=True);plain=inputs.definitions(source,production=False)
    readers=[m.make_knee_dataset_reader(out_size=8,num_slice=8,io_threads=1) for m in (prod,plain)]
    rows=[dict(StudyInstanceUID='synthetic',SeriesInstanceUID=s,Anatomical_Plane=p,Fluid_Sensitive=0,Fat_Suppression=0)
          for s,p in [('bad','Sagittal'),('never-retry','Sagittal'),('good','Axial')]]
    mode=['healthy'];arr=np.arange(64,dtype=np.float32).reshape(8,8)
    def files(pattern):
        if 'never-retry' in pattern:raise AssertionError('Alternative series retry')
        return [pattern.replace('*.dcm',f'{i}.dcm') for i in range(2 if mode[0]=='short' and '/bad/' in pattern else 4)]
    def one(path):return path.encode()
    def px(data,info):
        path=data.decode();i=int(Path(path).stem)
        if '/bad/' in path:
            if mode[0]=='short' and i>=2:return np.ones((9,9),np.float32),None
            if mode[0]=='tie':return np.ones((8+i%2,8+i%2),np.float32),None
            if mode[0]=='multiframe':return np.ones((2,8,8),np.float32),None
            if mode[0]=='unsafe':return np.full((8,8),np.nan),None
            if mode[0]=='black':return np.ones((8,8),np.float32),None
            if mode[0]=='unknown':raise AssertionError('Unexpected source bug')
            if mode[0]=='memory':raise MemoryError('synthetic')
            if mode[0]=='timeout':raise TimeoutError('synthetic')
            if mode[0]=='descriptor':raise OSError(__import__('errno').EMFILE,'synthetic')
        if mode[0]=='allblack':return np.ones((8,8),np.float32),None
        return arr+i,None
    checks={}
    with patch.object(prod.glob,'glob',side_effect=files):
        for mod,reader in zip((prod,plain),readers):
            mod._read_whole=one;mod._scan_full=lambda _:None
            reader.ctx.hdr_rec=lambda f,data,info:(int(Path(f).stem),f,.5,True);reader.ctx.px=px
            mod._knee_header_values=lambda *args:dict(image_laterality='L',laterality='',iop=[1,0,0,0,1,0],ipp=[0,0,0],ps=[.5,.5],rows=8,cols=8)
        def get(which=0):return inputs.prepare((prod,plain)[which],readers[which],'synthetic',rows,'synthetic',production=which==0)
        a,_=get();b,_=get(1)
        if not all(np.array_equal(x,y) for x,y in zip(a,b)):raise ValueError('Synthetic public MaViT parity differs')
        checks['healthy_public']='PASS'
        if a[1].tolist()!=[True,False,True,False,False,False]:raise ValueError('Missing-view public mask differs')
        checks['missing_view']='PASS'
        # Unique modal count2 -> short; tie is independently classified.
        for name,label in [('short','short_mask_no_retry'),('tie','shape_tie_mask'),('unsafe','unsafe_pixels_mask'),('multiframe','multiframe_mask')]:
            mode[0]=name;x,c=get()
            if x[1].tolist()!=[False,False,True,False,False,False] or c['masked_views']!=1:raise ValueError('Malformed selected view not masked')
            checks[label]='PASS'
        mode[0]='black';x,c=get()
        if not x[1][0] or x[0][0].any() or c['black_views']!=1:raise ValueError('Source black-view mask changed')
        checks['black_view_mask_preserved']='PASS'
        mode[0]='healthy'
        missing=dict(image_laterality='',laterality='',iop=None,ipp=None,ps=None,rows=8,cols=8)
        with patch.object(prod,'_knee_header_values',return_value=missing),patch.object(plain,'_knee_header_values',return_value=missing):
            x,c=get();y,_=get(1)
            if not all(np.array_equal(a,b) for a,b in zip(x,y)) or c['laterality_unknown_views']!=2 or c['header_geometry_default_frames']!=8:
                raise ValueError('Source missing geometry defaults changed')
            checks['missing_geometry_defaults']='PASS'
        original_px=readers[0].ctx.px
        def inner(data,info):
            if '/bad/' in data.decode():return prod._Ctx.px(readers[0].ctx,data,None)
            return original_px(data,info)
        with patch.object(readers[0].ctx,'px',side_effect=inner),patch.object(readers[0].ctx,'px_b',side_effect=ValueError('Synthetic corrupt pixel')):
            x,c=get()
            if x[1][0] or not x[1][2] or c['pixel_decode_failures']!=1 or c['masked_views']!=1:raise ValueError('Corrupt selected view not masked')
            checks['pixel_decode_mask']='PASS'
        with patch.object(readers[0].ctx,'px',side_effect=inner),patch.object(readers[0].ctx,'px_b',side_effect=AssertionError('Unknown source error')):
            try:get()
            except AssertionError:checks['inner_program_error_fatal']='PASS'
            else:raise ValueError('Unknown program error swallowed')
        with patch.object(prod.glob,'glob',return_value=[]):
            try:get()
            except ValueError as error:
                if getattr(error,'mavit_counts',{}).get('whole_empty_studies')!=1:raise
                checks['whole_empty_fatal']='PASS'
            else:raise ValueError('Whole-empty study accepted')
        for name,label,error in [('allblack','whole_black_fatal',ValueError),('unknown','unknown_exception_fatal',AssertionError),
                                 ('memory','memory_fatal',MemoryError),('timeout','timeout_fatal',TimeoutError),('descriptor','descriptor_fatal',OSError)]:
            mode[0]=name
            try:get()
            except error:checks[label]='PASS'
            else:raise ValueError('Required failure did not propagate')
    if set(checks)!=STRESS_CASES:raise ValueError('Synthetic MaViT checks incomplete')
    return {'status':'PASS','cases':checks,'uses_labels':False,'uses_raw_data':False}


def require_rehearsal(contract,report,*,parent_complete=True):
    if (contract.get('version')!=VERSION or contract.get('checkpoint_sha256')!=CHECKPOINT_SHA256 or
            contract.get('cpu_report_sha256')!=CPU_REPORT_SHA256 or contract.get('model_state_schema_sha256')!=STATE_SCHEMA or
            contract.get('max_seconds')!=MAX_SECONDS or contract.get('private') is not True or contract.get('internet') is not False or
            report.get('parent_complete') is not parent_complete or report.get('status')!='PASS' or report.get('contract_sha256')!=checkpoint.digest(checkpoint.canonical(contract)) or
            report.get('version')!=VERSION or report.get('studies')!=67 or report.get('test_studies')!=3 or report.get('train_studies')!=64 or
            report.get('checkpoint_sha256')!=CHECKPOINT_SHA256 or report.get('prediction_fallbacks')!=0 or
            report.get('input_counts',{}).get('resource_errors')!=0 or report.get('input_counts',{}).get('whole_empty_studies')!=0 or
            report.get('stress',{}).get('cases')!=dict.fromkeys(STRESS_CASES,'PASS') or report.get('backend_flags')!=FLAGS or
            report.get('tensor_max_error')!=0 or not 0<=report.get('probability_public_max_error',float('inf'))<=1e-6 or
            not 0<=report.get('repeat_max_error',float('inf'))<=1e-6 or
            not math.isfinite(report.get('elapsed_seconds',float('inf'))) or not 0<report['elapsed_seconds']<=MAX_SECONDS):
        raise ValueError('Complete exact MaViT67 qualification required')
    policy=contract.get('input_policy',{})
    if (policy.get('version')!=inputs.VERSION or policy.get('sha256')!=checkpoint.digest(Path(inputs.__file__).read_bytes()) or
            policy.get('source_sha256')!=checkpoint.SOURCE_HASHES['my_fastread_knee.py']):
        raise ValueError('Current reviewed MaViT input policy required')
    counts=report.get('input_counts',{})
    if set(counts)!=set(inputs.COUNTERS) or any(type(n) is not int or n<0 for n in counts.values()):raise ValueError('Input accounting incomplete')
    state=report.get('state_compatibility',{})
    if state.get('actual_schema_sha256')!=STATE_SCHEMA or state.get('strict_load') is not True or state.get('loaded_tensors_equal') is not True:
        raise ValueError('Strict audited model compatibility missing')
    if report.get('stress',{}).get('status')!='PASS' or report['stress'].get('uses_labels') is not False or report['stress'].get('uses_raw_data') is not False:
        raise ValueError('Production stress qualification incomplete')
    for key in ('visible_csv_sha256','prediction_order_sha256'):
        if not re.fullmatch('[0-9a-f]{64}',report.get(key,'')):raise ValueError('Prediction evidence missing')
    coverage=report.get('strict_coverage',{})
    if not set(coverage)<={'PASS','ineligible_malformed_input'} or any(type(n) is not int or n<0 for n in coverage.values()) or coverage.get('PASS',0)<1 or sum(coverage.values())!=67:raise ValueError('Healthy public comparison coverage missing')
    sampling=report.get('sampling',{})
    if (sampling.get('seed')!=20261008 or sampling.get('uses_labels') is not False or
            sampling.get('selection')!='SHA256(seed:StudyInstanceUID) order' or not re.fullmatch('[0-9a-f]{64}',sampling.get('ordered_ids_sha256',''))):
        raise ValueError('Fixed label-blind cohort evidence missing')
    env=report.get('environment',{})
    if (env.get('python_major_minor')!=[3,12] or any(env.get('packages',{}).get(p)!=v for p,v in PACKAGES.items()) or
            not isinstance(env.get('packages',{}).get('scipy'),str) or not env['packages']['scipy'] or
            report.get('codec_hashes')!=contract.get('codecs')):raise ValueError('MaViT dependency evidence differs')
    for key in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_host_rss_bytes','disk_free_bytes'):
        if not isinstance(report.get(key),(int,float)) or report[key]<=0 or not math.isfinite(report[key]):raise ValueError('Actual resource evidence missing')


def rehearse(root,assets,contract,out,*,started):
    import numpy as np
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','version':VERSION,'contract_sha256':checkpoint.digest(checkpoint.canonical(contract)),
      'parent_complete':False,'source_revision':contract['source_revision'],'studies':0,'test_studies':3,'train_studies':64,'prediction_fallbacks':0,
      'input_counts':dict.fromkeys(inputs.COUNTERS,0),'training_lineage':'unknown','predictive_quality':'unmeasured',
      'prospective_reason':'distinct encoder is a potential complement; no accuracy or independence claim'}
    def save():
        report['elapsed_seconds']=time.monotonic()-started
        (out/'mavit-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        warnings.filterwarnings('ignore',module=r'pydicom(?:\..*)?')
        if sys.version_info[:2]!=(3,12) or not torch.cuda.is_available():raise RuntimeError('Pinned Python3.12/CUDA required')
        versions={p:importlib.metadata.version(p) for p in (*PACKAGES,'scipy')}
        if any(versions[p]!=v for p,v in PACKAGES.items()):raise ValueError('Pinned model/decoder environment differs')
        torch.set_num_threads(FLAGS['torch_threads']);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.deterministic=False;torch.backends.cudnn.allow_tf32=True
        report['backend_flags']={'cudnn_benchmark':torch.backends.cudnn.benchmark,'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
          'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'torch_threads':torch.get_num_threads()}
        report['environment']={'python_major_minor':list(sys.version_info[:2]),'packages':versions,'gpu':torch.cuda.get_device_name(0),
          'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'scipy_policy':'first observed under pinned image; bind this exact version for future production'}
        source=(Path(assets)/'my_fastread_knee.py').read_text()
        if inputs.policy_contract(source)!=contract['input_policy']:raise ValueError('Input policy/source changed')
        prod=inputs.definitions(source,production=True);plain=inputs.definitions(source,production=False)
        args={k:inputs.RECIPE[k] for k in ('out_size','num_slice','percentile','crop_mm','io_threads','crop_only_if_nonsquare')}
        readers=[m.make_knee_dataset_reader(**args) for m in (prod,plain)]
        columns,ids,_=read_test_contract(root,rehearsal=True)
        if columns!=['StudyInstanceUID',*prod.KNEE_LABELS]:raise ValueError('Official target order differs')
        records={split:read_series(root,split) for split in ('test','train')}
        if set(records['test'])!=set(ids):raise ValueError('Exact official test metadata coverage differs')
        training=bounded_ids(list(records['train']),count=64,seed=20261008);jobs=[('test',sid) for sid in ids]+[('train',sid) for sid in training]
        report['sampling']={'seed':20261008,'uses_labels':False,'selection':'SHA256(seed:StudyInstanceUID) order',
                            'ordered_ids_sha256':checkpoint.digest(checkpoint.canonical(jobs))}
        report['stress']=stress_inputs(source)
        model,audit=load_model(assets);report['checkpoint_sha256']=CHECKPOINT_SHA256;report['state_compatibility']=audit['state_compatibility']
        torch.cuda.reset_peak_memory_stats();report['setup_seconds']=time.monotonic()-started
        values=[];counts=Counter();coverage=Counter();production_times=[];public_times=[];repeat_times=[];public_error=repeat_error=0.
        for index,(split,sid) in enumerate(jobs):
            if time.monotonic()-started>=MAX_SECONDS:raise TimeoutError('MaViT actual9h deadline exhausted')
            before=time.monotonic();arrays,event=inputs.prepare(prod,readers[0],sid,records[split][sid],Path(root)/f'{split}_series',production=True)
            probability=predict(model,arrays);torch.cuda.synchronize();production_times.append(time.monotonic()-before);counts.update(event)
            if event['masked_views']==0:
                before=time.monotonic();expected,_=inputs.prepare(plain,readers[1],sid,records[split][sid],Path(root)/f'{split}_series',production=False)
                if not all(np.array_equal(a,b) for a,b in zip(arrays,expected)):raise ValueError('Healthy public MaViT tensors differ')
                reference=predict(model,expected);public_error=max(public_error,float(np.max(np.abs(probability-reference))))
                if public_error>1e-6:raise ValueError('Same-model public probabilities differ')
                public_times.append(time.monotonic()-before);coverage['PASS']+=1;del expected
            else:coverage['ineligible_malformed_input']+=1
            before=time.monotonic();again=predict(model,arrays);repeat_error=max(repeat_error,float(np.max(np.abs(probability-again))))
            if repeat_error>1e-6:raise ValueError('MaViT repeat probabilities differ')
            repeat_times.append(time.monotonic()-before);values.append(probability);del arrays
            report.update(studies=index+1,input_counts=dict(counts),strict_coverage=dict(coverage),tensor_max_error=0,
                          probability_public_max_error=public_error,repeat_max_error=repeat_error)
            save();print(json.dumps({'event':'mavit_studies','completed':index+1,'total':67,'elapsed_seconds':report['elapsed_seconds']}),flush=True)
        values=np.stack(values);validate_predictions(values,rows=67)
        report.update(status='PASS',visible_csv_sha256=write_probabilities(out/'visible-predictions.csv',columns,ids,values[:3]),
            prediction_order_sha256=checkpoint.digest(checkpoint.canonical(values.tolist())),
            production_seconds=sum(production_times),public_comparison_seconds=sum(public_times),repeat_seconds=sum(repeat_times),
            production_mean_seconds=float(np.mean(production_times)),production_p95_seconds=float(np.percentile(production_times,95)),
            estimated_1300_seconds=report['setup_seconds']+1300*float(np.percentile(production_times,95)),runtime_estimate_policy='informational_only',
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,disk_free_bytes=shutil.disk_usage(out).free,
            output_disk_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),codec_hashes=contract['codecs'])
        save();require_rehearsal(contract,report,parent_complete=False);return report
    except Exception as error:
        if hasattr(error,'mavit_counts'):report['failure_input_counts']=error.mavit_counts
        report.update(status='FAIL',failure_type=type(error).__name__);save()
        raise RuntimeError('MaViT qualification failed; inspect aggregate receipt') from None


if __name__=='__main__':
    import argparse
    import signal
    parser=argparse.ArgumentParser();parser.add_argument('--payload',required=True);parser.add_argument('--deadline',type=float,required=True)
    a=parser.parse_args();remaining=a.deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('MaViT startup exhausted actual platform deadline')
    def expired(signum,frame):raise TimeoutError('MaViT actual9h deadline exhausted')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
    try:
        p=json.loads(Path(a.payload).read_text());c=p['contract']
        for name,sha in c['runtime_sources'].items():
            if checkpoint.digest((Path(p['project'])/name).read_bytes())!=sha:raise ValueError('Rehearsal runtime source changed')
        rehearse(p['root'],p['assets'],c,p['output'],started=a.deadline-MAX_SECONDS)
    except BaseException as error:
        print(json.dumps({'event':'mavit_rehearsal_failed','failure_type':type(error).__name__}),flush=True);sys.exit(1)
