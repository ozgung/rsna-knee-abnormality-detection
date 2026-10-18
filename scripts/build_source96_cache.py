"""Build-only private CPU shard of the qualified96wide336 image cache."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from knee import source96_cache as cache,source_span_pilot as pilot,public_training_smoke as smoke
from knee.reference_blend import DOCKER_IMAGE,CONVNEXT_DATASET
from knee.coverage_inference import CODECS

PILOT_RECEIPT_SHA='48655a7ef2863e2038beb9d7247f91afb7f9aba1c8153813b0cdaf03fb400e45'
FILES=('knee/__init__.py','knee/source96_cache.py','knee/source_span_pilot.py','knee/corpus44_parity.py',
       'knee/public_training_smoke.py','knee/public_rich_cache.py','knee/reference.py','knee/fresh_reader_setup.py','knee/raptor_input.py')


def build(destination,slug,*,evidence_root,public_source,training_source,shard_index,shard_count=4,resume_kernel=None,resume_dataset=None):
    import numpy as np
    out=Path(destination);evidence=Path(evidence_root)
    if out.exists():raise ValueError('Fresh immutable build directory required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug):raise ValueError('Invalid notebook slug')
    if type(shard_index) is not int or type(shard_count) is not int or not 0<=shard_index<shard_count<=64:
        raise ValueError('Invalid shard index/count')
    if resume_kernel and resume_dataset:raise ValueError('Select one resume transport')
    for ref in (resume_kernel,resume_dataset):
        if ref and not re.fullmatch('ozgung/[a-z0-9_-]+',ref):raise ValueError('Resume must be an explicit owned private attachment')
    def supplied(path):
        path=Path(path);return path if path.is_absolute() else evidence/path
    receipt_path=evidence/'outputs/kaggle/source-span-pilot-20261012-result-v1/source_span_pilot/receipt.json'
    smoke.require_file(receipt_path,PILOT_RECEIPT_SHA);qualification=json.loads(receipt_path.read_bytes())
    if qualification['status']!='PASS' or not qualification['exact_control_equal'] or qualification['completed_studies']!=32 or qualification['config']!=pilot.CONFIG:
        raise ValueError('Qualified real32 pilot differs')
    codes=pilot.recipes(supplied(public_source).read_text());sampler=pilot.sampler_source(supplied(training_source).read_text())
    if {k:smoke.digest(v.encode()) for k,v in codes.items()}!=qualification['input_pins']['readers'] or smoke.digest(sampler.encode())!=qualification['input_pins']['sampler_sha256']:
        raise ValueError('Reader/sampler differs from qualified real pilot')
    identity=evidence/'outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json'
    smoke.require_file(identity,smoke.MANIFEST_SHA);identity_document=json.loads(identity.read_bytes())
    historical=[]
    for sub,prefix in [('corpus','all'),('corpus-ext','extra')]:
        path=evidence/f'outputs/analysis/public-train-recipe-20261009/{sub}/{prefix}_ids.npy'
        smoke.require_file(path,smoke.PINS[prefix+'_ids.npy']);historical.extend(np.load(path,allow_pickle=False).astype(str).tolist())
    folds_path=evidence/'outputs/kaggle/trackA-coatnet224-fold01-20261009/folds_v0.json'
    folds_sha=smoke.file_sha(folds_path);folds=json.loads(folds_path.read_bytes())['folds']
    raw=evidence/'data/raw';ids,summary=cache.membership(historical,raw/'train.csv',folds,identity_document)
    shards=cache.make_shards(ids,shard_count)
    partitions=[{'studies':len(members),'ordinals_sha256':smoke.digest(smoke.canonical(members)),
                 'ordered_ids_sha256':smoke.digest(smoke.canonical([ids[n] for n in members]))} for members in shards]
    sources={name:(ROOT/name).read_text() for name in FILES}
    for name,source in sources.items():
        compile(source,name,'exec')
        if name in qualification['source_hashes'] and smoke.digest(source.encode())!=qualification['source_hashes'][name]:
            raise ValueError('Reused helper differs from qualified real32 pilot')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    contract={'version':cache.VERSION,'config':cache.CONFIG,'membership':summary,'shard_index':shard_index,'shard_count':shard_count,
              'partitions':partitions,'reader_hashes':{k:smoke.digest(v.encode()) for k,v in codes.items()},'sampler_sha256':smoke.digest(sampler.encode()),
              'code_revision':revision,'working_source_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).strip(),
              'sources':{k:smoke.digest(v.encode()) for k,v in sources.items()},
              'input_pins':{'competition':{n:smoke.file_sha(raw/n) for n in ('train.csv','train_series.csv')},
                            'historical_ids':{n:v for n,v in smoke.PINS.items() if n.endswith('_ids.npy')},
                            'folds_sha256':folds_sha,'identity_sha256':smoke.MANIFEST_SHA,'codecs':CODECS,'qualification_receipt_sha256':PILOT_RECEIPT_SHA},
              'qualified_environment':qualification['environment'],
              'policy':'Private cache only; faithful96wide336; no labels/report text/outcomes/models/training; fail on any study failure',
              'resume_policy':'Exact same contract, aligned membership, file/raw hashes; external transport does not change scientific contract',
              'storage_upper_bound_bytes':cache.disk_required(len(shards[shard_index])),
              'runtime_limit':'Actual CPU platform12h only;44.536s/32 warm decode is not a reliable full-corpus bound'}
    code=("import time,sys,json,hashlib,subprocess\nfrom pathlib import Path\nSTARTED=time.monotonic()\n"
          f'SOURCES={sources!r}\nCONTRACT={contract!r}\nREADERS={codes!r}\nSAMPLER={sampler!r}\n'
          "base=Path('/kaggle/working/cache96_code')\n"
          "for name,source in SOURCES.items():\n"
          "    if hashlib.sha256(source.encode()).hexdigest()!=CONTRACT['sources'][name]:raise ValueError('Packaged source differs')\n"
          "    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
          "sys.path.insert(0,str(base))\nfrom knee.fresh_reader_setup import dataset_root\nfrom knee.public_training_smoke import require_file\n"
          f"codecs=dataset_root('/kaggle/input',{CONVNEXT_DATASET!r},list(CONTRACT['input_pins']['codecs']))\n"
          "wheels=[]\nfor name,sha in CONTRACT['input_pins']['codecs'].items():\n"
          "    require_file(codecs/name,sha);wheels.append(str(codecs/name))\n"
          "if sys.version_info[:2]!=(3,12):raise ValueError('Pinned Python environment changed')\n"
          "subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*wheels],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
          "from knee.source96_cache import run\n"
          "r=run('/kaggle/input','/kaggle/working/source96_cache',CONTRACT,READERS,SAMPLER,CONTRACT['input_pins']['folds_sha256'],"
          f"started=STARTED,resume_kernel={resume_kernel!r},resume_dataset={resume_dataset!r})\n"
          "print(json.dumps({'event':'source96_cache_done','status':r['status'],'studies':r['completed_studies'],'bytes':r['bytes']}))\n")
    compile(code,'source96-cache-notebook','exec')
    if any(r['uid'] in code for r in identity_document['records']):raise ValueError('Private study identity embedded in source')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':['# Private96wide336 CPU cache shard\n',
            f'Shard{shard_index}/{shard_count}: {len(shards[shard_index])} of4345 nongold/nonquarantine studies in historical corpus order.\n',
            'Qualified wide96 reader;336px/140mm; slots26/22/18/12/18,2–98%. One-study streaming, atomic compressed files, private aligned index.\n',
            'Aggregate receipt has no IDs/report text; raw cache artifacts and index MUST remain private. No public dataset creation, model or training.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    blob=(json.dumps(nb,indent=2)+'\n').encode()
    if len(blob)>900000:raise ValueError('Notebook source limit')
    metadata={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
              'is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
              'dataset_sources':[*smoke.CORPORA,smoke.ASSETS,CONVNEXT_DATASET]+([resume_dataset] if resume_dataset else []),
              'kernel_sources':[smoke.IDENTITY_KERNEL]+([resume_kernel] if resume_kernel else []),'model_sources':[],'docker_image':DOCKER_IMAGE}
    receipt={'notebook_sha256':smoke.digest(blob),'notebook_bytes':len(blob),'contract_sha256':smoke.digest(smoke.canonical(contract)),
             'bundle_sha256':smoke.digest(smoke.canonical(sources)),'code_revision':revision,'build_only':True,
             'shard_index':shard_index,'shard_count':shard_count,'studies':len(shards[shard_index]),'required_free_disk_upper_bound_bytes':contract['storage_upper_bound_bytes']}
    out.mkdir(parents=True);(out/'notebook.ipynb').write_bytes(blob)
    for name,value in [('kernel-metadata.json',metadata),('build-contract.json',contract),('build-receipt.json',receipt)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('slug')
    for name in ('evidence-root','public-source','training-source'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--shard-index',type=int,required=True);p.add_argument('--shard-count',type=int,default=4)
    group=p.add_mutually_exclusive_group();group.add_argument('--resume-kernel');group.add_argument('--resume-dataset')
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
