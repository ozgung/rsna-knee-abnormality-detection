"""Build a private offline CPU audit only; never launch, download or infer images."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from knee.mavit_checkpoint import VERSION,DATASET,CHECKPOINT,CHECKPOINT_BYTES,PACKAGES,source_contract,canonical,digest
from knee.reference_blend import DOCKER_IMAGE

BUNDLE=('knee/__init__.py','knee/mavit_checkpoint.py','knee/fresh_reader_setup.py')


def build(destination,slug,*,source_dir):
    destination=Path(destination)
    if destination.exists():raise ValueError('Use a fresh immutable build destination')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug):raise ValueError('Invalid private notebook slug')
    _,proof=source_contract(source_dir)
    own={n:(ROOT/n).read_text() for n in BUNDLE};revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    contract={'version':VERSION,'source_contract':proof,'dataset':DATASET,'checkpoint_file':CHECKPOINT,
      'checkpoint_expected_bytes':CHECKPOINT_BYTES,'checkpoint_expected_sha256':None,'checkpoint_sha256_policy':'first observation only',
      'environment':{'python':[3,12],'packages':PACKAGES,'docker_image':DOCKER_IMAGE},
      'runtime_sha256':{n:digest(s.encode()) for n,s in own.items()},'builder_sha256':digest(Path(__file__).read_bytes()),
      'source_revision':revision,'device':'cpu','weights_only':True,'strict':True,'private':True,'internet':False,
      'training_lineage':'unknown','image_or_training_execution':False}
    code=("import time\nAUDIT_STARTED=time.monotonic()\nimport os,sys,json\nfrom pathlib import Path\n"
        "os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')\n"
        "project=Path('/kaggle/working/project')\n"
        f'sources={own!r}\n'
        "for name,content in sources.items():\n    p=project/name\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_text(content)\n"
        "sys.path.insert(0,str(project))\nfrom knee.mavit_checkpoint import run,digest\nfrom knee.fresh_reader_setup import dataset_root\n"
        f'CONTRACT={contract!r}\n'
        "for name,expected in CONTRACT['runtime_sha256'].items():\n    if digest((project/name).read_bytes())!=expected:raise ValueError('Audit runtime source changed')\n"
        "root=dataset_root('/kaggle/input',CONTRACT['dataset'],[CONTRACT['checkpoint_file'],*CONTRACT['source_contract']['source_sha256']])\n"
        "run(root,'/kaggle/working/mavit-checkpoint-audit',CONTRACT,started=AUDIT_STARTED)\n")
    compile(code,'<mavit-cpu-audit>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
      'language_info':{'name':'python','version':'3.12'}},'cells':[
      {'cell_type':'markdown','id':'purpose','metadata':{},'source':[
       '# MaViT checkpoint-only CPU audit\n',
       '[hengck23 source/model dataset](https://www.kaggle.com/datasets/hengck23/hengck23-rnsa-knee), publisher-declared MIT. Architecture credits: [MALA](https://github.com/qhfan/MALA); study head public reader attribution preserved in pinned source.\n',
       'No competition attachment, images, reports, labels, forward passes or training. Dataset contains unused gold CSVs; they are never opened.\n',
       'Safe weights-only CPU load, full tensor finiteness, unchanged architecture definitions, strict key/shape/dtype equality. Checkpoint SHA is first observed here; training lineage remains unknown. No unsafe pickle fallback or pretrained download.\n']},
      {'cell_type':'code','id':'audit','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    raw=(json.dumps(nb,indent=2)+'\n').encode()
    if len(raw)>900000:raise ValueError('Audit notebook source too large')
    meta={'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
      'is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,'competition_sources':[],
      'dataset_sources':[DATASET],'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE}
    receipt={'notebook_sha256':digest(raw),'notebook_bytes':len(raw),'contract_sha256':digest(canonical(contract)),
             'bundle_sha256':digest(canonical(own)),'source_revision':revision}
    destination.mkdir(parents=True,exist_ok=False);(destination/'notebook.ipynb').write_bytes(raw)
    for n,v in [('kernel-metadata.json',meta),('audit-contract.json',contract),('build-receipt.json',receipt)]:
        (destination/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('slug')
    p.add_argument('--source-dir',required=True,type=Path)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
