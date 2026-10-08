"""Package the fixed TRAIN04 pair after actual cache qualification; no launch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from knee.coverage_training import CONFIG,runtime_sources,require_pair_invariants,comparison_sources,supervision
from knee.coverage_objective import (BATCH_OBJECTIVE,GLOBAL_OBJECTIVE,validate_reference_prefix,
                                     require_objective_invariants)
from knee.fresh_reader import SOURCE_HASHES,check_sources,validate_plan
from knee.fresh_reader_setup import reference_parts
from knee.validation import file_sha256
from build_fresh_reader import DOCKER


def fingerprint_weights(plan):
    from knee.coverage_training import supervision
    from knee.fresh_reader import fingerprint
    return fingerprint(supervision(plan)[1].tolist())


def reference_gate(report):
    if report.get('status')!='PASS' or report.get('completed_epochs')!=12 or report['contract']['variant']!='v1':
        raise ValueError('Completed original-target reference required')
    history=report['history']
    if [r['epoch'] for r in history]!=list(range(1,13)):raise ValueError('Reference history incomplete')
    selected=min(history,key=lambda r:r['weak_bce'])
    auc=selected['thresholded_weak_auc']
    if (auc['undefined_targets'] or auc['macro_all_targets_auc'] is None or auc['macro_all_targets_auc']<.82 or
            report['prevalence_weak_bce']-selected['weak_bce']<.05):
        raise ValueError('Reference fails prospective label-comparison gate')
    return selected


def build(sources,destination,slug,cache_receipts,*,cache_kernel,reference_result=None,reference_kernel=None,
          objective=BATCH_OBJECTIVE):
    if objective not in (BATCH_OBJECTIVE,GLOBAL_OBJECTIVE):raise ValueError('Unknown TRAIN04 objective')
    if objective==GLOBAL_OBJECTIVE and not reference_result:raise ValueError('Original reference initialization required')
    check_sources(sources);reference_parts(cache_kernel,owner='ozgung')
    receipts=Path(cache_receipts)
    plan=json.loads((receipts/'plan-private.json').read_text());validate_plan(plan)
    report=json.loads((receipts/'cache-report.json').read_text())
    index=json.loads((receipts/'index-private.json').read_text())
    if (report['status']!='PASS' or report['stage']!='complete' or report['cached_studies']!=2200 or
            report['plan_fingerprint']!=plan['fingerprint'] or not report['gate']['passed'] or
            set(index)!={r['uid'] for r in plan['records']}):
        raise ValueError('Successful complete2200 cache required')
    reference_pins=None;reference_contract=None;reference_prefix=None
    if (reference_result is None)!=(reference_kernel is None):raise ValueError('Reference files and kernel required together')
    if reference_result:
        reference_parts(reference_kernel,owner='ozgung');ref=Path(reference_result)
        r=json.loads((ref/'training-report.json').read_text())
        run=json.loads((ref/'run-contract.json').read_text())
        if objective==GLOBAL_OBJECTIVE:
            reference_prefix=validate_reference_prefix(r,run)
        else:
            reference_gate(r)
        if run['contract']!=r['contract'] or r['contract']['plan_fingerprint']!=plan['fingerprint']:
            raise ValueError('Reference plan/run identity differs')
        reference_contract=r['contract']
        current={'config':CONFIG,'plan_fingerprint':plan['fingerprint'],
                 'runner_sha256':file_sha256(ROOT/'knee/coverage_training.py'),
                 'source_hashes':SOURCE_HASHES,'runtime_sources':runtime_sources(),
                 'index_sha256':file_sha256(receipts/'index-private.json'),
                 'complete_sha256':file_sha256(receipts/'cache-report.json'),
                 'weights_fingerprint':fingerprint_weights(plan)}
        if objective==GLOBAL_OBJECTIVE:
            from knee.fresh_reader import fingerprint
            current.update(objective=objective,variant='v1',comparison_sources=comparison_sources(),
                           labels_fingerprint=fingerprint(supervision(plan)[0].tolist()))
            require_objective_invariants(current,reference_contract,environment=False)
        else:
            require_pair_invariants(current,reference_contract,environment=False)
        reference_pins={'training-report.json':file_sha256(ref/'training-report.json'),
                        'run-contract.json':file_sha256(ref/'run-contract.json'),
                        'initial.pt':run['initial_model_sha256']}
        if any(not isinstance(x,str) or len(x)!=64 for x in reference_pins.values()):raise ValueError('Reference artifact hashes missing')
    names=['knee/__init__.py','knee/validation.py','knee/features.py','knee/experiment.py','knee/fresh_reader.py',
           'knee/fresh_reader_setup.py','knee/coverage_cache.py','knee/coverage_training.py','knee/coverage_objective.py']
    own={n:(ROOT/n).read_text() for n in names};external={n:(Path(sources)/n).read_text() for n in SOURCE_HASHES}
    for n,s in {**own,**external}.items():compile(s,n,'exec')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    pins={n:file_sha256(receipts/n) for n in ('plan-private.json','cache-report.json','index-private.json')}
    contract={'revision':revision,'cache_kernel':cache_kernel,'cache_pins':pins,'reference_kernel':reference_kernel,
              'reference_pins':reference_pins,'reference_contract':reference_contract,
              'reference_prefix':reference_prefix,'objective':objective,
              'config':CONFIG,'sources':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in own.items()},
              'builder_sha256':file_sha256(__file__),'external':SOURCE_HASHES,
              'notebook_seconds':CONFIG['seconds'],'time_policy':'platform_limit_only'}
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    license_text=(ROOT/'docs/licenses/fresh_reader_Apache-2.0.txt').read_text()
    entry=f'''import os,sys,time,json,hashlib
from pathlib import Path
STARTED=time.monotonic()
CODE={own!r}
EXTERNAL={external!r}
CONTRACT={contract!r}
base=Path('/kaggle/working/coverage_training_code')
for name,content in CODE.items():
    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
sources=base/'external';sources.mkdir()
(sources/'LICENSE').write_text({license_text!r})
for name,content in EXTERNAL.items():(sources/name).write_text(content)
sys.path.insert(0,str(base))
os.environ['KNEE_CODE_REVISION']={revision!r}
os.environ['KNEE_SOURCE_SHA256']={digest!r}
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
os.environ['HF_HUB_ETAG_TIMEOUT']='30'
from knee.coverage_training import train,LABEL_SOURCES
from knee.fresh_reader_setup import notebook_root,dataset_root,time_limit
from knee.validation import file_sha256
from knee.fresh_reader import write_json
try:
    with time_limit(CONTRACT['notebook_seconds']-(time.monotonic()-STARTED),'Kaggle GPU session'):
        inputs=Path('/kaggle/input')
        cache=notebook_root(inputs,{cache_kernel!r},'coverage_cache',list(CONTRACT['cache_pins']))
        for name,digest in CONTRACT['cache_pins'].items():
            if file_sha256(cache/name)!=digest:raise ValueError('Cache attachment receipt changed')
        initial=None;label_paths=None
        if CONTRACT['reference_kernel']:
            ref=notebook_root(inputs,CONTRACT['reference_kernel'],'coverage_training',list(CONTRACT['reference_pins']))
            for name,digest in CONTRACT['reference_pins'].items():
                if file_sha256(ref/name)!=digest:raise ValueError('Reference artifact changed')
            initial=ref/'initial.pt'
            if CONTRACT['objective']!={GLOBAL_OBJECTIVE!r}:
                refs={{'steven':'stevenleehans/rsna-knee-llm-report-labels','pilkwang':'pilkwang/rsna-knee-llm-labels','lixin':'lixin73/rsna-knee-llm-report-labels-sol56','qwen':'laymond/rsna-knee-abnormality-qwen3-8b-weak-labels'}}
                label_paths={{k:dataset_root(inputs,refs[k],[filename])/filename for k,(filename,digest) in LABEL_SOURCES.items()}}
        result=train(cache,sources,'/kaggle/working/coverage_training',source_paths=label_paths,initial=initial,
                     reference_contract=CONTRACT['reference_contract'],session_deadline=STARTED+CONTRACT['notebook_seconds'],
                     objective=CONTRACT['objective'],
                     reference_initial_sha256=CONTRACT['reference_pins']['initial.pt'] if CONTRACT['reference_pins'] else None)
        print(json.dumps({{'status':result,'elapsed_seconds':time.monotonic()-STARTED}}),flush=True)
except BaseException as exc:
    write_json('/kaggle/working/train04-failure.json',{{'failure_type':type(exc).__name__,'elapsed_seconds':time.monotonic()-STARTED,'contract':CONTRACT}})
    raise RuntimeError('TRAIN04 failed; inspect sanitized receipts') from None
'''
    compile(entry,'train04-entry','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},
        'cells':[{'cell_type':'markdown','id':'description','metadata':{},'source':['# TRAIN04 fixed coverage and supervision experiment\nPrivate training. Generic ConvNeXt initialization;2,000train/200development; no gold/sealed evaluation. Original unknown masks preserved.\n']},
                 {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':entry.splitlines(True)}]}
    meta={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
          'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':not bool(reference_kernel),
          'dataset_sources':['stevenleehans/rsna-knee-llm-report-labels','pilkwang/rsna-knee-llm-labels','lixin73/rsna-knee-llm-report-labels-sol56','laymond/rsna-knee-abnormality-qwen3-8b-weak-labels'] if reference_kernel and objective!=GLOBAL_OBJECTIVE else [],
          'kernel_sources':[cache_kernel]+([reference_kernel] if reference_kernel else []),
          'competition_sources':[],'model_sources':[],'docker_image':DOCKER,'machine_shape':'NvidiaTeslaT4'}
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('build-contract.json',contract)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    if (out/'notebook.ipynb').stat().st_size>900000:raise ValueError('Notebook too large')
    return contract


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('sources','destination','slug','cache_receipts'):p.add_argument(name)
    p.add_argument('--cache-kernel',required=True);p.add_argument('--reference-result');p.add_argument('--reference-kernel')
    p.add_argument('--objective',choices=(BATCH_OBJECTIVE,GLOBAL_OBJECTIVE),default=BATCH_OBJECTIVE)
    a=p.parse_args();build(a.sources,a.destination,a.slug,a.cache_receipts,cache_kernel=a.cache_kernel,
                           reference_result=a.reference_result,reference_kernel=a.reference_kernel,objective=a.objective)
