"""Build-only private40-study gold-free public training smoke. Never launches."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee import public_training_smoke as smoke
from knee.reference_blend import DOCKER_IMAGE

FILES = ('knee/__init__.py', 'knee/public_training_smoke.py', 'knee/fresh_reader_setup.py')


def build(destination, slug, *, public_source, manifest, label_table, corpus_metadata):
    out = Path(destination)
    if out.exists():
        raise ValueError('New immutable build directory required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', slug):
        raise ValueError('Invalid private notebook slug')
    smoke.require_file(public_source, smoke.SOURCE_SHA)
    selected, cohort = smoke.membership(manifest)
    _, values = smoke.selected_labels(label_table, selected)
    # Small local public metadata only; never open a volume array during build.
    import numpy as np
    meta = Path(corpus_metadata)
    paths = {name: meta / ('corpus' if name.startswith('all_') else 'corpus-ext') / name for name in smoke.PINS}
    for name, path in paths.items():
        smoke.require_file(path, smoke.PINS[name])
    ids = np.concatenate([np.load(paths[name], allow_pickle=False).astype(str) for name in ('all_ids.npy','extra_ids.npy')])
    if len(ids) != 4407 or len(set(ids)) != 4407 or not set(selected) <= set(ids):
        raise ValueError('Selected cohort missing from public corpus')
    sources = {name:(ROOT/name).read_text() for name in FILES}
    for name, source in sources.items():
        compile(source,name,'exec')
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    contract = {'version':smoke.VERSION,'revision':revision,'settings':smoke.SETTINGS,
        'source_sha256':smoke.SOURCE_SHA,'manifest_sha256':smoke.MANIFEST_SHA,'label_sha256':smoke.LABEL_SHA,
        'corpus_pins':smoke.PINS,'cohort':cohort,'selected_targets_sha256':smoke.digest(values.tobytes()),
        'identity_kernel':smoke.IDENTITY_KERNEL,'packages':smoke.PACKAGES,'internet':True,
        'sources':{name:smoke.digest(source.encode()) for name,source in sources.items()},
        'builder_sha256':smoke.file_sha(__file__),
        'scientific_definitions':list(smoke.DEFINITIONS),'gold_evaluation':False,'validation_selection':False,
        'initialization':'timm generic ImageNet22k/1k only; downloaded asset and initial tensor hashes first observed remotely',
        'corpus_volume_binding':'published mounted arrays with exact shape/type; selected40 pixel hash first observed remotely',
        'limitations':['public sampler crosses series boundaries/masked gaps; retained unchanged',
            'dense silent-cell probabilities are model-prior supervision, not observed negatives',
            'public labels were compared upstream on gold; no independent clinical validation',
            'training-only smoke; no predictive-quality, DICOM parity or submission qualification']}
    smoke.require_contract(contract)
    code = ("import os,time,json,hashlib,sys\nfrom pathlib import Path\nSTARTED=time.monotonic()\n"
        "os.environ['HF_HOME']='/tmp/public-training-assets/huggingface'\n"
        "os.environ['HF_HUB_CACHE']='/tmp/public-training-assets/huggingface/hub'\n"
        "os.environ['HUGGINGFACE_HUB_CACHE']=os.environ['HF_HUB_CACHE']\n"
        "os.environ['TORCH_HOME']='/tmp/public-training-assets/torch'\n"
        f"SOURCES={sources!r}\nCONTRACT={contract!r}\n"
        "base=Path('/kaggle/working/public_training_code')\n"
        "for name,source in SOURCES.items():\n"
        "    if hashlib.sha256(source.encode()).hexdigest()!=CONTRACT['sources'][name]:raise ValueError('Packaged source differs')\n"
        "    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
        "sys.path.insert(0,str(base))\n"
        "from knee.public_training_smoke import run\n"
        "run('/kaggle/input','/kaggle/working/public_training_smoke',CONTRACT,started=STARTED)\n")
    compile(code,'public-training-smoke-notebook','exec')
    # Explicit privacy regression: identities and numeric labels must not be embedded.
    if any(uid in code for uid in selected):
        raise ValueError('Private identity embedded in notebook')
    notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':[
            '# Public Raptor recipe: training-only smoke\n',
            'Forty permitted original-training studies; two epochs; no gold/development/sealed metrics or checkpoint selection.\n',
            '[nartaa training source](https://www.kaggle.com/datasets/nartaa/rsna-knee-hpo-assets), adapted from dreaddevelopment, CC0 declared. ',
            'Public MIL/sampling/optimizer recipe retained; hash-pinned definitions only, never the gold-selecting public main.\n',
            'Generic ImageNet ConvNeXt-T256; k8/batch8/FP16 on CUDA. Internet is for generic pretrained assets only; no report text or images sent to services.\n',
            'Published44slice corpus is mounted remotely. Known cross-series/masked windows retained; no DICOM/inference parity claim. ',
            'All measurements are training-only and cannot establish generalization. Fitted weights are not exported.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    data = (json.dumps(notebook,indent=2)+'\n').encode()
    if len(data) > 900000:
        raise ValueError('Notebook source limit exceeded')
    metadata = {'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python',
        'kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,
        'competition_sources':[],'dataset_sources':[smoke.ASSETS,*smoke.CORPORA],
        'kernel_sources':[smoke.IDENTITY_KERNEL],'model_sources':[],'docker_image':DOCKER_IMAGE,'machine_shape':'NvidiaTeslaT4'}
    receipt = {'notebook_sha256':smoke.digest(data),'notebook_bytes':len(data),
        'contract_sha256':smoke.digest(smoke.canonical(contract)),
        'source_bundle_sha256':smoke.digest(smoke.canonical(sources)),'revision':revision}
    out.mkdir(parents=True,exist_ok=False)
    (out/'notebook.ipynb').write_bytes(data)
    for name,value in [('kernel-metadata.json',metadata),('build-contract.json',contract),('build-receipt.json',receipt)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path);parser.add_argument('slug')
    for name in ('public-source','manifest','label-table','corpus-metadata'):
        parser.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(build(**vars(parser.parse_args())),indent=2))
