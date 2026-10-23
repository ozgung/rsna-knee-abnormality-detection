"""One bounded TRAIN04 run sequence. No scheduling, purchases or submissions.

Wait for existing CPU/probe jobs, verify receipts, run one reference and at most
one gated target comparison. State/logs remain ignored; source drift stops work.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
KAGGLE='~/miniconda3/bin/kaggle'
PYTHON=ROOT/'.venv/bin/python'
KAGGLE_PYTHON='~/miniconda3/bin/python'
CACHE='ozgung/knee-train04-cache-full-20261008'
PROBE='ozgung/knee-train04-coverage-probe-warm-20261008'
REFERENCE='ozgung/knee-train04-reference-20261009'
ALTERNATE='ozgung/knee-train04-consensus-20261009'
NOTEBOOK_SECONDS=12*3600  # Kaggle's actual GPU-session limit.
PLAN='3f4f0230aff05c8ade9b78ff6e898571839349e3cca41a86c6ded491c7408a32'
FILES=('knee/coverage_training.py','knee/coverage_cache.py','knee/fresh_reader.py',
       'knee/fresh_reader_setup.py','knee/validation.py','knee/features.py','knee/experiment.py',
       'scripts/build_coverage_training.py','scripts/run_train04_sequence.py','scripts/probe_coverage_entry.py',
       'scripts/download_kaggle_reports.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_pins():return {n:sha(ROOT/n) for n in FILES}


def require_pins(expected):
    if expected!=source_pins():raise ValueError('Reviewed source files changed; sequence stopped')


def kernel_state(text):
    match=re.search(r'KernelWorkerStatus\.([A-Z_]+)',text)
    if not match:raise ValueError('Unrecognized kernel status; no speculative retry')
    return match.group(1)


def gpu_remaining(rows):
    row=next(r for r in rows if r['resource']=='GPU')
    match=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)h',row['remaining'])
    if not match:raise ValueError('Unrecognized GPU quota')
    return float(match.group(1))


def require_training_complete(report):
    if report['status']!='PASS' or report['completed_epochs']!=12:
        raise ValueError('Fit did not complete the declared horizon')
    if not math.isfinite(report['elapsed_seconds']) or report['elapsed_seconds']<=0:
        raise ValueError('Invalid fit runtime')


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();pins=source_pins()
    state={'status':'RUNNING','started_utc':now(),'pid':os.getpid(),'source_pins':pins,'jobs':{},
           'authorization':'User removed self-imposed budgets; fixed TRAIN04 pair; no competition submissions',
           'time_policy':'platform_limit_only','poll_seconds':60,'stop_file':str(out/'STOP')}
    def save(event,**extra):
        state.update(event=event,updated_utc=now(),elapsed_seconds=time.monotonic()-started,**extra)
        tmp=out/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(out/'state.json')
        print(json.dumps({'event':event,'utc':state['updated_utc'],**extra}),flush=True)
    def guard():
        require_pins(pins)
        if (out/'STOP').exists():raise InterruptedError('Explicit sequence stop file present')
    def command(args,name,timeout=180):
        guard()
        result=subprocess.run(args,cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
        (out/(name+'.log')).write_text(result.stdout)
        if result.returncode:raise RuntimeError('Command failed: '+name+'; no automatic retry')
        return result.stdout
    def wait(kernel,label):
        previous=None
        while True:
            status=kernel_state(command([KAGGLE,'kernels','status',kernel],label+'-status'))
            state['jobs'][label]={'kernel':kernel,'status':status}
            if status!=previous:save(label+'_status',kernel=kernel,kernel_status=status);previous=status
            if status=='COMPLETE':return
            if status not in ('QUEUED','RUNNING'):raise RuntimeError(label+' terminal state '+status)
            time.sleep(60)
    def download(kernel,label):
        destination=ROOT/'outputs/kaggle'/(label+'-result')
        if kernel==CACHE:
            required=['coverage_cache/'+n for n in ('plan-private.json','cache-report.json','index-private.json')]
        elif kernel==PROBE:
            required=['coverage_probe/probe-report.json']
        else:
            required=['coverage_training/'+n for n in ('training-report.json','run-contract.json',
                      'best-weak-validation-private.json','gpu-gate.json')]
        command([KAGGLE_PYTHON,'scripts/download_kaggle_reports.py',kernel,str(destination),
                 '--required',*required],label+'-download',timeout=300)
        return destination
    def quota(label):
        rows=json.loads(command([KAGGLE,'quota','--format','json'],label+'-quota'))
        remaining=gpu_remaining(rows)
        if remaining<=0:raise RuntimeError('No Kaggle GPU quota available')
        save(label+'_quota_pass',remaining_gpu_hours=remaining)
    def train(label,kernel,cache_dir,reference_dir=None):
        guard();quota(label)
        destination=ROOT/'outputs/kaggle'/label
        if destination.exists():raise ValueError('Training package already exists; refuse duplicate launch')
        args=[str(PYTHON),'scripts/build_coverage_training.py','outputs/research/2026-10-08/convnext-audit',
              str(destination),kernel.split('/')[1],str(cache_dir),'--cache-kernel',CACHE]
        if reference_dir:args+=['--reference-result',str(reference_dir),'--reference-kernel',REFERENCE]
        command(args,label+'-build')
        metadata=json.loads((destination/'kernel-metadata.json').read_text())
        if (metadata['id']!=kernel or not metadata['is_private'] or not metadata['enable_gpu'] or
                metadata['competition_sources'] or metadata['model_sources']):
            raise ValueError('Unexpected training upload scope')
        save(label+'_push_start',next_kernel=kernel)
        # Unknown push outcomes stop the sequence; never blindly push again.
        response=command([KAGGLE,'kernels','push','-p',str(destination),'--timeout',str(NOTEBOOK_SECONDS)],label+'-push')
        if 'Kernel version 1 successfully pushed' not in response:
            raise ValueError('Expected a new v1 kernel; inspect push outcome before continuing')
        wait(kernel,label)
        result=download(kernel,label)/'coverage_training'
        report=json.loads((result/'training-report.json').read_text())
        require_training_complete(report)
        save(label+'_complete',fit_elapsed_seconds=report['elapsed_seconds'],selected_bce=report['selected_bce'])
        return result
    save('sequence_started')
    try:
        wait(PROBE,'train04-coverage-probe')
        probe_dir=download(PROBE,'train04-coverage-probe-warm')
        probe=json.loads((probe_dir/'coverage_probe/probe-report.json').read_text())
        if (probe['status']!='PASS' or probe['optimizer_updates']!=0 or probe['fitted_weights_saved'] or
                not math.isfinite(probe['elapsed_seconds']) or probe['elapsed_seconds']<=0):
            raise ValueError('No-update CUDA probe failed; revise prospectively before fitting')
        save('probe_verified',probe_seconds=probe['elapsed_seconds'])
        wait(CACHE,'train04-cache-full-v2')
        cache_dir=download(CACHE,'train04-cache-full-v2')/'coverage_cache'
        plan=json.loads((cache_dir/'plan-private.json').read_text())
        report=json.loads((cache_dir/'cache-report.json').read_text())
        if (plan['fingerprint']!=PLAN or report['plan_fingerprint']!=PLAN or report['status']!='PASS' or
                report['stage']!='complete' or not report['gate']['passed'] or report['failures'] or
                report['cached_studies']!=2200 or report['cache_bytes']>17*1024**3 or
                not report['expansion_disk_gate_passed']):
            raise ValueError('Full CPU cache failed its actual qualification')
        save('full_cache_verified',cache_bytes=report['cache_bytes'])
        reference=train('train04-reference-20261009',REFERENCE,cache_dir)
        # Builder checks reference metrics and exact non-label invariants before any alternate upload.
        alternate=train('train04-consensus-20261009',ALTERNATE,cache_dir,reference)
        save('pair_complete',status='COMPLETE',reference_result=str(reference),alternate_result=str(alternate))
    except BaseException as exc:
        save('sequence_stopped',status='STOPPED',failure_type=type(exc).__name__,reason=str(exc))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output')
    args=parser.parse_args();run(args.output)
