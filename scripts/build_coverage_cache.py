"""Build a private CPU-only TRAIN04 pilot/full cache; never launch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from knee.coverage_cache import CACHE_CONFIG, VERSION
from knee.fresh_reader import SOURCE_HASHES, check_sources
from knee.fresh_reader_setup import reference_parts
from build_fresh_reader import DOCKER


def build(source_dir, destination, slug, expected_plan, *, pilot_kernel=None):
    check_sources(source_dir)
    if len(expected_plan)!=64 or any(c not in '0123456789abcdef' for c in expected_plan):
        raise ValueError('Expected locally verified plan hash')
    if pilot_kernel:
        reference_parts(pilot_kernel, owner='ozgung')
    names = ['knee/__init__.py','knee/validation.py','knee/features.py','knee/experiment.py',
             'knee/fresh_reader.py','knee/fresh_reader_setup.py','knee/coverage_cache.py']
    own = {n:(ROOT/n).read_text() for n in names}
    external = {n:(Path(source_dir)/n).read_text() for n in SOURCE_HASHES}
    for n,s in {**own,**external}.items():compile(s,n,'exec')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    config = {'version':VERSION,'expected_plan':expected_plan,'pilot_kernel':pilot_kernel,
              'revision':revision,'own_sources':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in own.items()},
              'external_sources':SOURCE_HASHES,'cache':CACHE_CONFIG,
              'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    seconds = CACHE_CONFIG['full_seconds' if pilot_kernel else 'pilot_seconds']
    license_text = (ROOT/'docs/licenses/fresh_reader_Apache-2.0.txt').read_text()
    entry = f'''import os, sys, json, subprocess, time
from pathlib import Path
STARTED=time.monotonic()
CODE={own!r}
EXTERNAL={external!r}
CONTRACT={config!r}
base=Path('/kaggle/working/coverage_code')
for name,content in CODE.items():
    path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(content)
sources=base/'external';sources.mkdir()
(sources/'LICENSE').write_text({license_text!r})
for name,content in EXTERNAL.items():(sources/name).write_text(content)
sys.path.insert(0,str(base))
os.environ['KNEE_CODE_REVISION']={revision!r}
os.environ['KNEE_SOURCE_SHA256']={fingerprint!r}
from knee.coverage_cache import make_coverage_plan, build_cache
from knee.fresh_reader_setup import exact_root, dataset_root, notebook_root, time_limit
from knee.fresh_reader import write_json
try:
    with time_limit({seconds+300}, 'complete CPU notebook'):
        with time_limit(300,'CPU setup'):
            inputs=Path('/kaggle/input')
            data=exact_root([inputs/'competitions/rsna-knee-abnormality-detection',inputs/'rsna-knee-abnormality-detection'],['train.csv','train_series.csv'])
            assets=dataset_root(inputs,'goodpjw2008/rsna-knee-2-5d-convnext-reader',['pylibjpeg-2.1.0-py3-none-any.whl'])
            weak=dataset_root(inputs,'stevenleehans/rsna-knee-llm-report-labels',['llm_labels_full.csv'])/'llm_labels_full.csv'
            original=notebook_root(inputs,'ozgung/knee-fresh-cache-20261008','fresh_reader_cache',['plan-private.json'])/'plan-private.json'
            identity=notebook_root(inputs,'ozgung/knee-sealed-identity-full-20261008','sealed_identity',['sealed-manifest-private.json'])/'sealed-manifest-private.json'
            subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--find-links',str(assets),'pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg'],check=True,timeout=120)
            previous=notebook_root(inputs,{pilot_kernel!r},'coverage_cache',['cache-report.json','plan-private.json']) if {pilot_kernel is not None!r} else None
            plan=make_coverage_plan(data,weak,original,identity)
            if plan['fingerprint']!={expected_plan!r}:raise ValueError('Remote plan differs from local verified plan')
        report=build_cache(data,sources,plan,'/kaggle/working/coverage_cache',previous=previous)
        print(json.dumps(report),flush=True)
except BaseException as exc:
    write_json('/kaggle/working/coverage-failure.json',{{'failure_type':type(exc).__name__,'elapsed_seconds':time.monotonic()-STARTED,'contract':CONTRACT}})
    raise RuntimeError('CPU coverage preparation failed; inspect aggregate receipt') from None
'''
    compile(entry,'coverage-cache-entry','exec')
    notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},
      'cells':[{'cell_type':'markdown','id':'description','metadata':{},'source':['# TRAIN04 bounded CPU coverage cache\nPrivate image preprocessing. No fit or gold/sealed metrics. Apache-2.0 reader geometry; frozen original membership.\n']},
               {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':entry.splitlines(True)}]}
    meta={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb',
          'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,
          'dataset_sources':['goodpjw2008/rsna-knee-2-5d-convnext-reader','stevenleehans/rsna-knee-llm-report-labels'],
          'competition_sources':['rsna-knee-abnormality-detection'],
          'kernel_sources':['ozgung/knee-fresh-cache-20261008','ozgung/knee-sealed-identity-full-20261008']+([pilot_kernel] if pilot_kernel else []),
          'model_sources':[],'docker_image':DOCKER}
    out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    for name,value in [('notebook.ipynb',notebook),('kernel-metadata.json',meta),('build-contract.json',config)]:
        (out/name).write_text(json.dumps(value,indent=2)+'\n')
    if (out/'notebook.ipynb').stat().st_size>900000:raise ValueError('Notebook too large')
    return config


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source_dir','destination','slug','expected_plan'):p.add_argument(name)
    p.add_argument('--pilot-kernel')
    a=p.parse_args();build(a.source_dir,a.destination,a.slug,a.expected_plan,pilot_kernel=a.pilot_kernel)
