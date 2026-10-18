"""Build-only private CPU image-only corpus44 parity hypothesis diagnostic."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from knee import corpus44_parity as parity
from knee import public_training_smoke as smoke
from knee.reference_blend import DOCKER_IMAGE
from knee.coverage_inference import CODECS
FILES=('knee/__init__.py','knee/corpus44_parity.py','knee/public_training_smoke.py','knee/fresh_reader_setup.py','knee/raptor_input.py')


def build(destination,slug,*,public_source,manifest,smoke_contract):
    out=Path(destination)
    if out.exists():raise ValueError('New immutable build directory required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug):raise ValueError('Invalid slug')
    source=Path(public_source).read_text();code=parity.recipe(source)
    selected,cohort=smoke.membership(manifest)
    previous=json.loads(Path(smoke_contract).read_text())
    smoke.require_contract(previous)
    if previous['cohort']!=cohort:raise ValueError('Completed smoke cohort differs')
    sources={name:(ROOT/name).read_text() for name in FILES}
    for name,value in sources.items():compile(value,name,'exec')
    contract={'version':1,'recipe_status':'hypothesis','source_sha256':parity.SOURCE_SHA,
        'reader_sha256':smoke.digest(code.encode()),'substitutions':json.loads(json.dumps(parity.SUBSTITUTIONS)),
        'smoke_cohort':cohort,'selected5_sha256':smoke.digest(smoke.canonical(selected[:5])),
        'studies':5,'device':'cpu','codecs':CODECS,'identity_kernel':smoke.IDENTITY_KERNEL,'corpus_pins':smoke.PINS,
        'selection':'first5 of exact pinned40 smoke order','exact_acceptance':'all25 slot uint8 hashes and masks equal; no tolerance',
        'sources':{k:smoke.digest(v.encode()) for k,v in sources.items()},
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'labels_read':False,'images_exported':False,'scientific_limits':['Original44 builder unavailable; span/count substitutions test an explicit hypothesis',
            'Published series preference and percentile/crop rounding remain archived-source assumptions',
            'Five-study exact match would establish bounded observed parity, never entire-corpus equivalence']}
    entry=("import time,sys,hashlib\nfrom pathlib import Path\nSTARTED=time.monotonic()\n"
        f"SOURCES={sources!r}\nCONTRACT={contract!r}\nREADER={code!r}\n"
        "base=Path('/kaggle/working/parity_code')\n"
        "for name,source in SOURCES.items():\n"
        "    if hashlib.sha256(source.encode()).hexdigest()!=CONTRACT['sources'][name]:raise ValueError('Packaged code differs')\n"
        "    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
        "sys.path.insert(0,str(base))\nfrom knee.corpus44_parity import run\n"
        "run('/kaggle/input','/kaggle/working/corpus44_parity',CONTRACT,READER,started=STARTED)\n")
    compile(entry,'corpus44-parity-notebook','exec')
    if any(uid in entry for uid in selected):raise ValueError('Private ID embedded')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':['# Image-only corpus44 parity hypothesis\n',
            'Five deterministic studies from the exact40-study smoke. CPU; no models, labels, gold scores or training.\n',
            'Pinned archived DICOM reader; explicitly substitute44 slots12/10/8/6/8 and15–85% span. Original construction code unavailable.\n',
            'Compare each slot with mounted CC0 corpus by exact uint8 hash, masks and pixel errors. Mismatch is evidence; no tolerance tuning.\n',
            'Aggregate report only. No IDs, MRI, report text or exception messages exported. Offline hash-pinned codecs.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':entry.splitlines(True)}]}
    blob=(json.dumps(notebook,indent=2)+'\n').encode()
    if len(blob)>900000:raise ValueError('Notebook source limit')
    metadata={'id':'ozgung/'+slug,'title':slug.replace('-', ' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,
        'competition_sources':['rsna-knee-abnormality-detection'],
        'dataset_sources':[*smoke.CORPORA,'goodpjw2008/rsna-knee-2-5d-convnext-reader'],
        'kernel_sources':[smoke.IDENTITY_KERNEL],'model_sources':[],'docker_image':DOCKER_IMAGE}
    receipt={'notebook_sha256':smoke.digest(blob),'notebook_bytes':len(blob),'contract_sha256':smoke.digest(smoke.canonical(contract)),
        'reader_sha256':contract['reader_sha256'],'build_only':True}
    out.mkdir(parents=True)
    (out/'notebook.ipynb').write_bytes(blob)
    for name,value in [('kernel-metadata.json',metadata),('build-contract.json',contract),('build-receipt.json',receipt)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('slug')
    for name in ('public-source','manifest','smoke-contract'):p.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
