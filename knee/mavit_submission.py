"""Standalone MaViT orchestration; all scientific functions come from the67 archive."""
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
from knee.reference_blend import read_test_contract,write_probabilities

VERSION='mavit-standalone-v1'
ATOL=1e-6
MAX_SECONDS=reference.MAX_SECONDS


def read_predictions(path):
    with Path(path).open(newline='') as handle:
        reader=csv.reader(handle);columns=next(reader);rows=list(reader)
    if len(columns)!=13 or columns[0]!='StudyInstanceUID' or len(set(columns))!=13 or not rows or any(len(row)!=13 for row in rows):
        raise ValueError('MaViT prediction schema invalid')
    ids=[row[0] for row in rows]
    if len(set(ids))!=len(ids) or any(not x or x in ('.','..') or '/' in x or '\\' in x for x in ids):raise ValueError('MaViT prediction IDs invalid')
    values=[[float(v) for v in row[1:]] for row in rows]
    reference.validate_predictions(values[0] if len(rows)==1 else values,rows=len(rows))
    return columns,ids,values


def compare_csv(expected,actual):
    a,b=read_predictions(expected),read_predictions(actual)
    if a[:2]!=b[:2] or len(a[1])!=3:raise ValueError('Visible MaViT IDs/target order differ')
    error=max(abs(x-y) for ar,br in zip(a[2],b[2]) for x,y in zip(ar,br))
    if error>ATOL:raise ValueError('Visible MaViT cross-run probabilities exceed1e-6')
    return error


def qualification(contract):
    q=contract.get('qualification',{});original=q.get('rehearsal_contract',{});policy=original.get('input_policy',{})
    if (contract.get('version')!=VERSION or contract.get('max_seconds')!=MAX_SECONDS or type(contract.get('visible')) is not bool or
            contract.get('private') is not True or contract.get('internet') is not False or
            contract.get('qualification_sha256')!=cp.digest(cp.canonical(q)) or q.get('version')!=VERSION or q.get('cross_run_atol')!=ATOL or
            q.get('repeat_atol')!=ATOL or original.get('checkpoint_sha256')!=reference.CHECKPOINT_SHA256 or
            original.get('model_state_schema_sha256')!=reference.STATE_SCHEMA or original.get('cpu_report_sha256')!=reference.CPU_REPORT_SHA256 or
            policy.get('version')!=inputs.VERSION or policy.get('sha256')!=cp.digest(Path(inputs.__file__).read_bytes()) or
            q.get('backend_flags')!=reference.FLAGS or q.get('packages',{}).keys()!=set(reference.PACKAGES)|{'scipy'} or
            any(q['packages'][p]!=v for p,v in reference.PACKAGES.items()) or not isinstance(q['packages']['scipy'],str) or not q['packages']['scipy']):
        raise ValueError('Qualified immutable MaViT production contract required')
    evidence=contract.get('visible_evidence')
    if contract['visible']:
        if evidence is not None:raise ValueError('Visible run cannot inherit full-entry evidence')
    elif (not isinstance(evidence,dict) or evidence.get('absolute_tolerance')!=ATOL or
          not 0<=evidence.get('immutable67_max_error',float('inf'))<=ATOL or
          any(not re.fullmatch('[0-9a-f]{64}',evidence.get(k,'')) for k in ('report_sha256','csv_sha256'))):
        raise ValueError('Full entry requires reviewed visible qualification')
    return q


def require_result(report,contract,csv_path,*,parent_complete=True):
    q=qualification(contract);columns,ids,_=read_predictions(csv_path);counts=report.get('input_counts',{});n=len(ids)
    if (report.get('status')!='PASS' or report.get('parent_complete') is not parent_complete or report.get('version')!=VERSION or
            report.get('qualification_sha256')!=contract['qualification_sha256'] or report.get('contract_sha256')!=cp.digest(cp.canonical(contract)) or
            report.get('visible') is not contract['visible'] or (contract['visible'] and n!=3) or report.get('test_studies')!=n or
            report.get('completed_studies')!=n or report.get('prediction_fallbacks')!=0 or report.get('backend_flags')!=reference.FLAGS or
            report.get('checkpoint_sha256')!=reference.CHECKPOINT_SHA256 or report.get('state_compatibility',{}).get('strict_load') is not True or
            report.get('state_compatibility',{}).get('loaded_tensors_equal') is not True or
            report.get('state_compatibility',{}).get('actual_schema_sha256')!=reference.STATE_SCHEMA or
            report.get('environment',{}).get('python_major_minor')!=[3,12] or report.get('environment',{}).get('packages')!=q['packages'] or
            report.get('codec_hashes')!=q['rehearsal_contract']['codecs'] or
            set(counts)!=set(inputs.COUNTERS) or any(type(v) is not int or v<0 for v in counts.values()) or
            counts.get('resource_errors')!=0 or counts.get('whole_empty_studies')!=0 or
            not 0<=report.get('repeat_max_error',float('inf'))<=ATOL or
            not 0<report.get('elapsed_seconds',float('inf'))<=MAX_SECONDS or
            report.get('csv_sha256')!=cp.digest(Path(csv_path).read_bytes()) or columns[1:]!=q['targets'] or
            report.get('ordered_ids_sha256')!=cp.digest(cp.canonical(ids)) or report.get('target_order_sha256')!=cp.digest(cp.canonical(columns[1:]))):
        raise ValueError('Complete exact MaViT production evidence required')
    for key in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_host_rss_bytes','disk_free_bytes','output_disk_bytes'):
        if not isinstance(report.get(key),(int,float)) or not math.isfinite(report[key]) or report[key]<=0:raise ValueError('Actual MaViT resource evidence missing')


def finish(out,contract,*,started,target=None,owned_outputs=None):
    """Only parent completion releases CSV; failure removes only our exclusive output."""
    out=Path(out);audit=out/'mavit-audit.json';report=json.loads(audit.read_text())
    owned=set() if owned_outputs is None else owned_outputs;created=None
    try:
        report['elapsed_seconds']=time.monotonic()-started
        csv_path=out/('visible-predictions.csv' if contract['visible'] else 'submission.csv')
        require_result(report,contract,csv_path,parent_complete=False)
        if not contract['visible']:
            if target is None:raise ValueError('Final output path required')
            output=Path(target)
            with csv_path.open('rb') as src,output.open('xb') as dst:
                created=output;owned.add(output);shutil.copyfileobj(src,dst)
            if cp.digest(output.read_bytes())!=report['csv_sha256']:raise ValueError('Final MaViT CSV copy differs')
        elif target is not None:raise ValueError('Visible rehearsal must not create submission.csv')
        report.update(parent_complete=True,elapsed_seconds=time.monotonic()-started)
        require_result(report,contract,csv_path);audit.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
    except BaseException:
        if created is not None:
            created.unlink(missing_ok=True);owned.discard(created)
        raise


def run(root,assets,contract,out,*,started):
    import numpy as np
    import torch
    q=qualification(contract);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','parent_complete':False,'version':VERSION,'visible':contract['visible'],
      'qualification_sha256':contract['qualification_sha256'],'contract_sha256':cp.digest(cp.canonical(contract)),
      'completed_studies':0,'prediction_fallbacks':0,'input_counts':dict.fromkeys(inputs.COUNTERS,0),'repeat_max_error':0.,
      'training_lineage':'unknown','predictive_quality':'unmeasured'}
    def save():
        report['elapsed_seconds']=time.monotonic()-started
        (out/'mavit-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        if sys.version_info[:2]!=(3,12) or not torch.cuda.is_available():raise RuntimeError('Pinned Python3.12/CUDA required; no fallback')
        versions={p:importlib.metadata.version(p) for p in q['packages']}
        if versions!=q['packages']:raise ValueError('Qualified MaViT environment changed, including SciPy')
        warnings.filterwarnings('ignore',module=r'pydicom(?:\..*)?')
        torch.set_num_threads(reference.FLAGS['torch_threads']);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.deterministic=False;torch.backends.cudnn.allow_tf32=True
        report['backend_flags']={'cudnn_benchmark':torch.backends.cudnn.benchmark,'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
          'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'torch_threads':torch.get_num_threads()}
        if report['backend_flags']!=reference.FLAGS:raise ValueError('Qualified backend flags differ')
        report['environment']={'python_major_minor':list(sys.version_info[:2]),'packages':versions,'gpu':torch.cuda.get_device_name(0),
                              'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
        source=(Path(assets)/'my_fastread_knee.py').read_text()
        if inputs.policy_contract(source)!=q['rehearsal_contract']['input_policy']:raise ValueError('Qualified reader source/policy changed')
        _,model_source=cp.source_contract(assets)
        if model_source!=q['rehearsal_contract']['cpu_source_contract']:raise ValueError('CPU-audited model definitions changed')
        public=inputs.definitions(source,production=True)
        reader=public.make_knee_dataset_reader(**{k:inputs.RECIPE[k] for k in ('out_size','num_slice','percentile','crop_mm','io_threads','crop_only_if_nonsquare')})
        columns,ids,_=read_test_contract(root,rehearsal=contract['visible']);records=reference.read_series(root,'test')
        if set(records)!=set(ids) or columns!=['StudyInstanceUID',*public.KNEE_LABELS] or columns[1:]!=q['targets']:
            raise ValueError('Official test schema/coverage differs')
        report.update(test_studies=len(ids),ordered_ids_sha256=cp.digest(cp.canonical(ids)),target_order_sha256=cp.digest(cp.canonical(columns[1:])))
        model,audit=reference.load_model(assets);report.update(checkpoint_sha256=reference.CHECKPOINT_SHA256,state_compatibility=audit['state_compatibility'])
        torch.cuda.reset_peak_memory_stats();report['setup_seconds']=time.monotonic()-started
        values=[];counts=Counter();times=[];repeat_times=[]
        for index,sid in enumerate(ids):
            if time.monotonic()-started>=MAX_SECONDS:raise TimeoutError('Actual9h MaViT runtime exhausted')
            before=time.monotonic();arrays,event=inputs.prepare(public,reader,sid,records[sid],Path(root)/'test_series',production=True)
            probability=reference.predict(model,arrays);torch.cuda.synchronize();times.append(time.monotonic()-before)
            before=time.monotonic();again=reference.predict(model,arrays);repeat_times.append(time.monotonic()-before)
            error=float(np.max(np.abs(probability-again)))
            if error>ATOL:raise ValueError('MaViT within-run repeat exceeds1e-6')
            values.append(probability);counts.update(event);report.update(completed_studies=index+1,input_counts=dict(counts),repeat_max_error=max(report['repeat_max_error'],error))
            del arrays;save();print(json.dumps({'event':'mavit_test_studies','completed':index+1,'total':len(ids),'elapsed_seconds':report['elapsed_seconds']}),flush=True)
        values=np.stack(values);reference.validate_predictions(values[0] if len(ids)==1 else values,rows=len(ids))
        filename='visible-predictions.csv' if contract['visible'] else 'submission.csv'
        report.update(status='PASS',csv_sha256=write_probabilities(out/filename,columns,ids,values),codec_hashes=q['rehearsal_contract']['codecs'],
            production_seconds=sum(times),repeat_seconds=sum(repeat_times),production_p95_seconds=float(np.percentile(times,95)),
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,disk_free_bytes=shutil.disk_usage(out).free,
            output_disk_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()))
        save();require_result(report,contract,out/filename,parent_complete=False);return report
    except Exception as error:
        if hasattr(error,'mavit_counts'):report['failure_input_counts']=error.mavit_counts
        report.update(status='FAIL',failure_type=type(error).__name__);save();raise RuntimeError('MaViT production failed; aggregate receipt only') from None


if __name__=='__main__':
    import argparse
    import signal
    parser=argparse.ArgumentParser();parser.add_argument('--payload',required=True);parser.add_argument('--deadline',type=float,required=True);a=parser.parse_args()
    remaining=a.deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('MaViT startup exhausted platform deadline')
    def expired(signum,frame):raise TimeoutError('MaViT actual9h deadline exhausted')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
    try:
        p=json.loads(Path(a.payload).read_text());q=qualification(p['contract'])
        for name,sha in q['runtime_sources'].items():
            if cp.digest((Path(p['project'])/name).read_bytes())!=sha:raise ValueError('MaViT qualified source changed')
        run(p['root'],p['assets'],p['contract'],p['output'],started=a.deadline-MAX_SECONDS)
    except BaseException as error:
        print(json.dumps({'event':'mavit_production_failed','failure_type':type(error).__name__}),flush=True);sys.exit(1)
