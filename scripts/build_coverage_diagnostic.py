"""Build only: disposable exact-init TRAIN04 training diagnostic."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from knee.coverage_objective import (REFERENCE_BUILD,REFERENCE_REVISION,REFERENCE_RUNNER,
                                    REFERENCE_SOURCES,validate_reference_prefix)
from knee.fresh_reader import SOURCE_HASHES,validate_plan
from knee.fresh_reader_setup import reference_parts
from knee.validation import file_sha256
from build_fresh_reader import DOCKER


def archived_sources(notebook):
    bindings={}
    for cell in json.loads(Path(notebook).read_text())['cells']:
        if cell['cell_type']!='code':continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                name=node.targets[0].id
                if name in ('CODE','EXTERNAL','CONTRACT'):
                    if name in bindings:raise ValueError('Duplicate archived binding')
                    bindings[name]=ast.literal_eval(node.value)
    expected=REFERENCE_SOURCES|{'knee/coverage_training.py':REFERENCE_RUNNER}
    for key,pins in [('CODE',expected),('EXTERNAL',SOURCE_HASHES)]:
        source=bindings[key]
        if set(source)!=set(pins) or any(hashlib.sha256(source[n].encode()).hexdigest()!=h for n,h in pins.items()):
            raise ValueError('Immutable archived source differs')
    contract=bindings['CONTRACT']
    if (contract['revision']!=REFERENCE_REVISION or
            hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()!=REFERENCE_BUILD):
        raise ValueError('Immutable archived build differs')
    return bindings


def build(archive,cache_receipts,reference_result,destination,slug,*,cache_kernel,reference_kernel,reference_dataset=None):
    reference_parts(cache_kernel,owner='ozgung');reference_parts(reference_kernel,owner='ozgung')
    if reference_dataset:reference_parts(reference_dataset,owner='ozgung')
    if not slug or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in slug):raise ValueError('Invalid slug')
    bundle=archived_sources(archive);cache=Path(cache_receipts);ref=Path(reference_result)
    report=json.loads((ref/'training-report.json').read_text());run=json.loads((ref/'run-contract.json').read_text())
    prefix=validate_reference_prefix(report,run)
    plan=json.loads((cache/'plan-private.json').read_text());validate_plan(plan)
    done=json.loads((cache/'cache-report.json').read_text());index=json.loads((cache/'index-private.json').read_text())
    if (done['status']!='PASS' or done['stage']!='complete' or done['cached_studies']!=2200 or
        done['plan_fingerprint']!=plan['fingerprint'] or set(index)!={r['uid'] for r in plan['records']} or
        run['contract']['plan_fingerprint']!=plan['fingerprint']):raise ValueError('Cache/plan differs')
    pins={n:file_sha256(cache/n) for n in ('plan-private.json','cache-report.json','index-private.json')}
    if pins!=bundle['CONTRACT']['cache_pins'] or cache_kernel!=bundle['CONTRACT']['cache_kernel']:
        raise ValueError('Original cache identity differs')
    if file_sha256(ref/'initial.pt')!=run['initial_model_sha256']:raise ValueError('Original initialization differs')
    own=bundle['CODE']|{'knee/coverage_diagnostic.py':(ROOT/'knee/coverage_diagnostic.py').read_text()}
    external=bundle['EXTERNAL']
    for n,s in {**own,**external}.items():compile(s,n,'exec')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    contract={'purpose':'disposable TRAIN04 trainability diagnostic','revision':revision,
        'original_revision':REFERENCE_REVISION,'original_build':REFERENCE_BUILD,'reference_prefix':prefix,
        'sources':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in own.items()},'external':SOURCE_HASHES,
        'builder_sha256':file_sha256(__file__),'archive_sha256':file_sha256(archive),
        'cache_kernel':cache_kernel,'cache_pins':pins,'reference_kernel':reference_kernel,
        'reference_dataset':reference_dataset,
        'reference_pins':{n:file_sha256(ref/n) for n in ('training-report.json','run-contract.json','initial.pt')},
        'initial_sha256':run['initial_model_sha256'],'reference_environment':run['contract']['runtime_environment'],
        'plan_fingerprint':plan['fingerprint'],'seed':42,'notebook_seconds':43200,
        'time_policy':'platform_limit_only','fitted_weights_saved':False,
        'diagnostic':{'normal_accepted_updates':1,'fixed_training_studies':32,'fixed_accepted_updates':64,
            'objective':'original batch macro v1','scheduler_horizon':12000,'fixed_windows':8,
            'fixed_augmentation':False,'head_dropout':'original train mode','selection':'fixed train-only ID hash',
            'phase_reset':'exact initial model and saved RNG; fresh optimizer/scaler/scheduler/EMA'}}
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    license_text=(ROOT/'docs/licenses/fresh_reader_Apache-2.0.txt').read_text()
    entry=f'''import os,sys,time,json,hashlib
from pathlib import Path
STARTED=time.monotonic()
CODE={own!r}
EXTERNAL={external!r}
CONTRACT={contract!r}
base=Path('/kaggle/working/coverage_diagnostic_code')
for name,content in CODE.items():
    if hashlib.sha256(content.encode()).hexdigest()!=CONTRACT['sources'][name]:raise ValueError('Packaged source differs')
    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
sources=base/'external';sources.mkdir()
(sources/'LICENSE').write_text({license_text!r})
for name,content in EXTERNAL.items():
    if hashlib.sha256(content.encode()).hexdigest()!=CONTRACT['external'][name]:raise ValueError('External source differs')
    (sources/name).write_text(content)
sys.path.insert(0,str(base))
os.environ['KNEE_CODE_REVISION']={revision!r}
os.environ['KNEE_SOURCE_SHA256']={digest!r}
from knee.coverage_diagnostic import run
from knee.fresh_reader_setup import dataset_root,notebook_root,time_limit
from knee.validation import file_sha256
from knee.fresh_reader import write_json
try:
    with time_limit(CONTRACT['notebook_seconds']-(time.monotonic()-STARTED),'Kaggle GPU session'):
        inputs=Path('/kaggle/input')
        cache=notebook_root(inputs,CONTRACT['cache_kernel'],'coverage_cache',list(CONTRACT['cache_pins']))
        if CONTRACT['reference_dataset']:
            ref=dataset_root(inputs,CONTRACT['reference_dataset'],list(CONTRACT['reference_pins']))
        else:
            ref=notebook_root(inputs,CONTRACT['reference_kernel'],'coverage_training',list(CONTRACT['reference_pins']))
        for root,pins in [(cache,CONTRACT['cache_pins']),(ref,CONTRACT['reference_pins'])]:
            for name,digest in pins.items():
                if file_sha256(root/name)!=digest:raise ValueError('Attachment identity differs')
        result=run(cache,sources,ref/'initial.pt','/kaggle/working/coverage_diagnostic',CONTRACT,
                   deadline=STARTED+CONTRACT['notebook_seconds'])
        print(json.dumps({{'status':result['status'],'elapsed_seconds':time.monotonic()-STARTED}}),flush=True)
except BaseException as exc:
    write_json('/kaggle/working/train04-diagnostic-failure.json',{{'failure_type':type(exc).__name__,
               'elapsed_seconds':time.monotonic()-STARTED,'contract':CONTRACT}})
    raise RuntimeError('TRAIN04 diagnostic failed; inspect sanitized receipts') from None
'''
    compile(entry,'diagnostic-entry','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'}},'cells':[
        {'cell_type':'markdown','id':'description','metadata':{},'source':['# Disposable TRAIN04 trainability diagnostic\nTraining only. No fitted weights saved or reused.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':entry.splitlines(True)}]}
    meta={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python',
          'kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
          'dataset_sources':[reference_dataset] if reference_dataset else [],
          'kernel_sources':[cache_kernel]+([] if reference_dataset else [reference_kernel]),'competition_sources':[],
          'model_sources':[],'docker_image':DOCKER,'machine_shape':'NvidiaTeslaT4'}
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('build-contract.json',contract)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    return contract


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','cache_receipts','reference_result','destination','slug'):p.add_argument(name)
    p.add_argument('--cache-kernel',required=True);p.add_argument('--reference-kernel',required=True)
    p.add_argument('--reference-dataset')
    a=p.parse_args();build(a.archive,a.cache_receipts,a.reference_result,a.destination,a.slug,
        cache_kernel=a.cache_kernel,reference_kernel=a.reference_kernel,reference_dataset=a.reference_dataset)
