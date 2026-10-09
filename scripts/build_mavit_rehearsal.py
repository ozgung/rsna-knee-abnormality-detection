"""Build only: CPU-qualified MaViT -> separate private image-only67-study rehearsal."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from knee import mavit_checkpoint as checkpoint,mavit_input as inputs,mavit_rehearsal as rehearsal
from knee.reference_blend import CONVNEXT_DATASET,DOCKER_IMAGE
from knee.coverage_inference import CODECS

BUNDLE=('knee/__init__.py','knee/mavit_checkpoint.py','knee/mavit_input.py','knee/mavit_rehearsal.py','knee/fresh_reader_setup.py',
        'knee/reference.py','knee/raptor_input.py','knee/reference_blend.py','knee/public_reader_contract.py','scripts/rehearse_mavit_entry.py')


def build(destination,slug,*,source_dir,cpu_report):
    destination=Path(destination)
    if destination.exists():raise ValueError('New immutable rehearsal destination required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug):raise ValueError('Invalid private notebook slug')
    receipt=rehearsal.cpu_evidence(cpu_report,source_dir,ROOT)
    source=(Path(source_dir)/'my_fastread_knee.py').read_text();policy=inputs.policy_contract(source)
    own={n:(ROOT/n).read_text() for n in BUNDLE};revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    contract={'version':rehearsal.VERSION,'input_policy':policy,'checkpoint_sha256':rehearsal.CHECKPOINT_SHA256,
        'cpu_report_sha256':rehearsal.CPU_REPORT_SHA256,'cpu_source_contract':receipt['source_contract'],
        'model_state_schema_sha256':rehearsal.STATE_SCHEMA,'codecs':CODECS,'backend_flags':rehearsal.FLAGS,
        'packages':rehearsal.PACKAGES,'scipy_version_policy':'first observation under fixed image; future production must pin receipt version',
        'model_inference':'batch1 eval model.infer; CUDA float16 autocast; sigmoid inside autocast then float32 CPU; no TTA',
        'sampling':{'training_studies':64,'visible_studies':3,'seed':20261008,'uses_labels':False},
        'runtime_sources':{n:checkpoint.digest(s.encode()) for n,s in own.items()},'source_revision':revision,
        'builder_sha256':checkpoint.digest(Path(__file__).read_bytes()),'max_seconds':32400,'private':True,'internet':False,
        'training_lineage':'unknown','quality_and_independence':'unverified; execution qualification only'}
    code=("import time,signal\nMAVIT_STARTED=time.monotonic()\n"
          "def _mavit_expired(signum,frame):\n    raise TimeoutError('MaViT actual9h deadline exhausted')\n"
          "signal.signal(signal.SIGALRM,_mavit_expired)\nsignal.setitimer(signal.ITIMER_REAL,32400)\n"
          "import os,sys\nfrom pathlib import Path\nproject=Path('/kaggle/working/project')\n"
          f'sources={own!r}\n'
          "for name,content in sources.items():\n    p=project/name\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_text(content)\n"
          "sys.path.insert(0,str(project))\n"
          f'MAVIT_CONTRACT={contract!r}\n'
          "exec(compile(sources['scripts/rehearse_mavit_entry.py'],'<mavit-rehearsal-entry>','exec'))\n")
    compile(code,'<mavit-image-rehearsal>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
      'language_info':{'name':'python','version':'3.12'}},'cells':[
       {'cell_type':'markdown','id':'attribution','metadata':{},'source':[
        '# Independent MaViT image-only qualification\n',
        '[hengck23 source/checkpoint](https://www.kaggle.com/datasets/hengck23/hengck23-rnsa-knee), MIT declared. Encoder [MALA](https://github.com/qhfan/MALA); active decoder attributes nartaa. Training lineage and predictive quality remain unknown.\n',
        'Fixed64 ID-selected training images plus3 visible test images, no labels/reports/gold CSVs, fitting, fusion or submission. Healthy exact public-reader tensors and same-model probabilities; explicitly counted malformed-view masking.\n',
        'Six slots, first metadata series,8 central triplets, full-series1/99 normalization, right-knee normalization, uint8 scipy resize288, batch1 CUDA FP16 model.infer. No alternate-series, precision or device fallback. Actual9h setup-inclusive deadline; ETA informational.\n']},
       {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    raw=(json.dumps(nb,indent=2)+'\n').encode()
    if len(raw)>900000 or len(code.encode())>900000:raise ValueError('Rehearsal notebook too large')
    meta={'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
      'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
      'dataset_sources':[checkpoint.DATASET,CONVNEXT_DATASET],'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE}
    r={'notebook_sha256':checkpoint.digest(raw),'notebook_bytes':len(raw),'contract_sha256':checkpoint.digest(checkpoint.canonical(contract)),
       'source_bundle_sha256':checkpoint.digest(checkpoint.canonical(own)),'source_revision':revision}
    destination.mkdir(parents=True,exist_ok=False);(destination/'notebook.ipynb').write_bytes(raw)
    for name,value in [('kernel-metadata.json',meta),('mavit-contract.json',contract),('build-receipt.json',r)]:
        (destination/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('slug')
    p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--cpu-report',type=Path,required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
