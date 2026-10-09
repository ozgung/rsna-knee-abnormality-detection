"""Train-membership-only MaViT raw-image diagnostic; no probability/array export."""
from collections import Counter
import csv
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
from knee import mavit_checkpoint as cp, mavit_input as inputs, mavit_rehearsal as reference
from knee.mavit_submission import qualification as production_qualification

VERSION='mavit-train2000-diagnostic-v1'
REHEARSAL_REPORT_SHA256='e51e51273867c13885d4d6c84fcf824a417857cc654bcf21e8e7d770a3e0c364'
MAX_SECONDS=32400
ATOL=1e-6
INTERVAL=32
KERNEL='ozgung/knee-train04-cache-full-20261008'
PLAN_SHA256='5d05d478dcb63a90a322671f3553cd70c09319ac6685b9059e2dbeb6dec4a3cc'
PLAN_FINGERPRINT='3f4f0230aff05c8ade9b78ff6e898571839349e3cca41a86c6ded491c7408a32'
SALT='mavit-broad-train2000-v1:20261009'
TRAIN_COUNT=2000
DEV_COUNT=200
SERIES_COUNT=11145
PHASES={'setup','cohort','metadata','model','decode','inference','repeat','complete'}


def membership(path):
    """Hash the original file, then project ONLY uid/split; no labels/group access."""
    raw=Path(path).read_bytes()
    if cp.digest(raw)!=PLAN_SHA256:raise ValueError('Pinned original membership file differs')
    document=json.loads(raw)
    if document.get('fingerprint')!=PLAN_FINGERPRINT:raise ValueError('Original plan identity differs')
    records=document.get('records')
    if not isinstance(records,list) or len(records)!=TRAIN_COUNT+DEV_COUNT:raise ValueError('Original membership coverage differs')
    projected=[]
    for row in records:
        if not isinstance(row,dict):raise ValueError('Original membership schema differs')
        uid,split=row.get('uid'),row.get('split')
        if (not isinstance(uid,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',uid) or uid in ('.','..') or
                split not in ('train','weak_validation')):raise ValueError('Invalid original membership')
        projected.append((uid,split))
    del records,document,raw
    train=[uid for uid,split in projected if split=='train'];dev=[uid for uid,split in projected if split=='weak_validation']
    if (len(train)!=TRAIN_COUNT or len(dev)!=DEV_COUNT or len(set(train+dev))!=TRAIN_COUNT+DEV_COUNT):
        raise ValueError('Original train/development membership overlaps or differs')
    train.sort(key=lambda uid:(cp.digest(f'{SALT}:{uid}'.encode()),uid));del projected,dev
    return train,{'kernel':KERNEL,'plan_sha256':PLAN_SHA256,'plan_fingerprint':PLAN_FINGERPRINT,'salt':SALT,
                  'train_studies':TRAIN_COUNT,'excluded_development_studies':DEV_COUNT,'series':SERIES_COUNT,
                  'ordered_ids_sha256':cp.digest(cp.canonical(train)),'fields_used':['uid','split']}


def qualification(contract):
    q=production_qualification(contract.get('production_contract',{}));cohort=contract.get('cohort',{})
    if (contract.get('version')!=VERSION or contract.get('max_seconds')!=MAX_SECONDS or contract.get('repeat_atol')!=ATOL or
            contract.get('progress_interval')!=INTERVAL or contract.get('prediction_exports') is not False or
            not contract['production_contract']['visible'] or q.get('rehearsal_report_sha256')!=REHEARSAL_REPORT_SHA256 or q['rehearsal_contract']['input_policy']['version']!='mavit-active-reader-v3' or
            cohort!={'kernel':KERNEL,'plan_sha256':PLAN_SHA256,'plan_fingerprint':PLAN_FINGERPRINT,'salt':SALT,
                     'train_studies':TRAIN_COUNT,'excluded_development_studies':DEV_COUNT,'series':SERIES_COUNT,
                     'ordered_ids_sha256':cohort.get('ordered_ids_sha256'),'fields_used':['uid','split']} or
            not re.fullmatch('[0-9a-f]{64}',cohort.get('ordered_ids_sha256',''))):
        raise ValueError('Exact qualified v3 training diagnostic contract required')
    return q


def safe_failure(error):
    """No exception text, paths, frame locals, identifiers or arbitrary function names."""
    modules={'builtins','pydicom.errors','knee.mavit_input','torch','torch.cuda','cv2','numpy.linalg'}
    name=type(error).__name__
    if type(error).__module__ not in modules or not re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,63}',name):name='UnknownError'
    files={str(Path(__file__)):'diagnostic',str(Path(inputs.__file__)):'input_policy',str(Path(reference.__file__)):'inference',
           '<audited-mavit-active-reader>':'public_reader'}
    site=None;trace=error.__traceback__
    while trace:
        code=trace.tb_frame.f_code
        if code.co_filename in files and re.fullmatch('[A-Za-z_][A-Za-z0-9_]{0,63}',code.co_name):
            site={'source':files[code.co_filename],'function':code.co_name,'line':trace.tb_lineno}
        trace=trace.tb_next
    return {'error_class':name,'source_site':site}


def rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)


def require_result(report,contract,*,parent_complete=True):
    q=qualification(contract);counts=report.get('input_counts',{})
    if (report.get('status')!='PASS' or report.get('parent_complete') is not parent_complete or report.get('version')!=VERSION or
            report.get('contract_sha256')!=cp.digest(cp.canonical(contract)) or report.get('cohort')!=contract['cohort'] or
            report.get('phase')!='complete' or report.get('completed_studies')!=TRAIN_COUNT or report.get('ordinal')!=TRAIN_COUNT or
            report.get('finite_prediction_calls')!=TRAIN_COUNT*2 or report.get('prediction_exports') is not False or
            report.get('prediction_fallbacks')!=0 or report.get('backend_flags')!=reference.FLAGS or
            report.get('environment',{}).get('python_major_minor')!=[3,12] or report.get('environment',{}).get('packages')!=q['packages'] or
            report.get('checkpoint_sha256')!=reference.CHECKPOINT_SHA256 or
            report.get('state_compatibility',{}).get('strict_load') is not True or
            report.get('state_compatibility',{}).get('loaded_tensors_equal') is not True or
            report.get('state_compatibility',{}).get('actual_schema_sha256')!=reference.STATE_SCHEMA or
            report.get('codec_hashes')!=q['rehearsal_contract']['codecs'] or
            set(counts)!=set(inputs.COUNTERS) or any(type(v) is not int or v<0 for v in counts.values()) or
            counts.get('resource_errors')!=0 or counts.get('whole_empty_studies')!=0 or
            not 0<=report.get('repeat_max_error',float('inf'))<=ATOL or
            not 0<report.get('elapsed_seconds',float('inf'))<=MAX_SECONDS or
            not TRAIN_COUNT<=report.get('present_views_total',0)<=6*TRAIN_COUNT):
        raise ValueError('Complete exact train2000 diagnostic evidence required')
    for key in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_host_rss_bytes','disk_free_bytes'):
        if not isinstance(report.get(key),(int,float)) or not math.isfinite(report[key]) or report[key]<=0:
            raise ValueError('Actual diagnostic resource evidence missing')


def run(root,assets,plan_path,contract,out,*,started):
    """One process, same production calls; stop the first failure without exporting values."""
    import numpy as np
    import torch
    q=qualification(contract);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','parent_complete':False,'version':VERSION,'contract_sha256':cp.digest(cp.canonical(contract)),
        'phase':'setup','ordinal':0,'completed_studies':0,'finite_prediction_calls':0,'present_views_total':0,
        'prediction_exports':False,'prediction_fallbacks':0,'input_counts':dict.fromkeys(inputs.COUNTERS,0),'repeat_max_error':0.,
        'training_lineage':'unknown','interpretation':'raw-image robustness diagnostic; no predictive-quality measurement'}
    def save():
        report.update(elapsed_seconds=time.monotonic()-started,peak_host_rss_bytes=rss_bytes())
        (out/'diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def resources():
        report.update(peak_host_rss_bytes=rss_bytes(),cuda_allocated_bytes=torch.cuda.memory_allocated(),
            cuda_reserved_bytes=torch.cuda.memory_reserved(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
    def phase(value,ordinal=None):
        if time.monotonic()-started>=MAX_SECONDS:raise TimeoutError('Actual9h diagnostic deadline exhausted')
        report['phase']=value
        if ordinal is not None:report['ordinal']=ordinal
    save()
    try:
        if sys.version_info[:2]!=(3,12) or not torch.cuda.is_available():raise RuntimeError('Pinned Python3.12/CUDA required; no fallback')
        versions={p:importlib.metadata.version(p) for p in q['packages']}
        if versions!=q['packages']:raise ValueError('Qualified MaViT environment changed, including SciPy')
        warnings.filterwarnings('ignore',module=r'pydicom(?:\..*)?')
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.deterministic=False;torch.backends.cudnn.allow_tf32=True
        flags={'cudnn_benchmark':torch.backends.cudnn.benchmark,'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
               'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'torch_threads':torch.get_num_threads()}
        if flags!=reference.FLAGS:raise ValueError('Qualified MaViT backend flags differ')
        report.update(backend_flags=flags,environment={'python_major_minor':list(sys.version_info[:2]),'packages':versions,
                      'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()})
        source=(Path(assets)/'my_fastread_knee.py').read_text()
        if inputs.policy_contract(source)!=q['rehearsal_contract']['input_policy']:raise ValueError('Qualified reader source/policy changed')
        _,model_source=cp.source_contract(assets)
        if model_source!=q['rehearsal_contract']['cpu_source_contract']:raise ValueError('CPU-audited model definitions changed')
        public=inputs.definitions(source,production=True)
        reader=public.make_knee_dataset_reader(**{k:inputs.RECIPE[k] for k in ('out_size','num_slice','percentile','crop_mm','io_threads','crop_only_if_nonsquare')})
        phase('cohort');ids,cohort=membership(plan_path)
        if cohort!=contract['cohort']:raise ValueError('Mounted diagnostic membership differs')
        report['cohort']=cohort
        phase('metadata');all_rows=reference.read_series(root,'train')
        if not set(ids)<=all_rows.keys():raise ValueError('Training image metadata coverage incomplete')
        rows={uid:all_rows[uid] for uid in ids};del all_rows
        if sum(map(len,rows.values()))!=SERIES_COUNT:raise ValueError('Pinned training series coverage differs')
        with (Path(root)/'sample_submission.csv').open(newline='') as handle:columns=next(csv.reader(handle))
        if columns!=['StudyInstanceUID',*public.KNEE_LABELS] or columns[1:]!=q['targets']:raise ValueError('Official target order differs')
        phase('model');model,audit=reference.load_model(assets)
        report.update(checkpoint_sha256=reference.CHECKPOINT_SHA256,state_compatibility=audit['state_compatibility'],codec_hashes=q['rehearsal_contract']['codecs'])
        torch.cuda.reset_peak_memory_stats();report['setup_seconds']=time.monotonic()-started;save()
        counts=Counter(dict.fromkeys(inputs.COUNTERS,0));production_seconds=0.;repeat_seconds=0.
        for ordinal,uid in enumerate(ids,1):
            phase('decode',ordinal);report['current_study']=None;report['current_repeat_max_error']=None;before=time.monotonic()
            arrays,event=inputs.prepare(public,reader,uid,rows[uid],Path(root)/'train_series',production=True)
            counts.update(event);report['input_counts']=dict(counts);report['present_views_total']+=int(arrays[1].sum())
            report['current_study']={'volume_shape':list(arrays[0].shape),'volume_dtype':str(arrays[0].dtype),
                'mask_shape':list(arrays[1].shape),'present_views':int(arrays[1].sum()),
                'position_shape':list(arrays[2].shape),'position_dtype':str(arrays[2].dtype)}
            phase('inference');probability=reference.predict(model,arrays);report['finite_prediction_calls']+=1
            production_seconds+=time.monotonic()-before
            phase('repeat');before=time.monotonic();again=reference.predict(model,arrays);report['finite_prediction_calls']+=1
            repeat_seconds+=time.monotonic()-before;error=float(np.max(np.abs(probability-again)))
            report['current_repeat_max_error']=error if math.isfinite(error) else None
            if math.isfinite(error):report['repeat_max_error']=max(report['repeat_max_error'],error)
            if not math.isfinite(error) or error>ATOL:raise ValueError('MaViT within-run repeat exceeds1e-6')
            report.update(completed_studies=ordinal,repeat_max_error=max(report['repeat_max_error'],error),
                production_seconds=production_seconds,repeat_seconds=repeat_seconds)
            del arrays,probability,again
            if ordinal%INTERVAL==0 or ordinal==len(ids):
                resources()
                save();print(json.dumps({'event':'mavit_train_diagnostic_progress','completed':ordinal,'total':len(ids),
                    'elapsed_seconds':report['elapsed_seconds'],'peak_host_rss_bytes':report['peak_host_rss_bytes'],
                    'input_counts':report['input_counts'],'repeat_max_error':report['repeat_max_error']}),flush=True)
        phase('complete');report.update(status='PASS',disk_free_bytes=shutil.disk_usage(out).free);save()
        require_result(report,contract,parent_complete=False);return report
    except BaseException as error:
        if hasattr(error,'mavit_counts'):report['failure_input_counts']=error.mavit_counts
        report.update(status='FAIL',failure=safe_failure(error))
        try:resources()
        except BaseException:report['resource_snapshot_failed']=True  # Keep the original failure, even if CUDA reporting also fails.
        save();raise


if __name__=='__main__':
    import argparse
    import signal
    parser=argparse.ArgumentParser();parser.add_argument('--payload',required=True);parser.add_argument('--deadline',type=float,required=True);args=parser.parse_args()
    def expired(signum,frame):raise TimeoutError('MaViT diagnostic actual9h deadline exhausted')
    try:
        remaining=args.deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('MaViT diagnostic startup deadline exhausted')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
        p=json.loads(Path(args.payload).read_text());q=qualification(p['contract'])
        for name,sha in q['runtime_sources'].items():
            if cp.digest((Path(p['project'])/name).read_bytes())!=sha:raise ValueError('Qualified diagnostic runtime changed')
        run(p['root'],p['assets'],p['plan'],p['contract'],p['output'],started=args.deadline-MAX_SECONDS)
    except BaseException as error:
        print(json.dumps({'event':'mavit_train_diagnostic_failed',**safe_failure(error)}),flush=True);sys.exit(1)
