"""Build only a frozen Versia ConvNeXt16 component rehearsal; never launch or fuse."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee.reference import canonical_bytes, sha256
from knee.reference_blend import CONVNEXT_DATASET, DOCKER_IMAGE
from knee.coverage_inference import CODECS
from knee.versia import audit_sources, extract, protect, policy_contract, RECIPE, SOURCE_HASHES, SPEC_SHA256, WEIGHTS

BUNDLE=('knee/__init__.py','knee/versia.py','knee/reference.py','knee/raptor_input.py',
        'knee/reference_blend.py','knee/public_reader_contract.py','knee/fresh_reader_setup.py',
        'scripts/versia_rehearsal_entry.py')


def build_versia(destination,slug,*,spec_dir):
    root=Path(__file__).resolve().parents[1];destination=Path(destination)
    if destination.exists():raise ValueError('Use a new destination to preserve immutable rehearsal builds')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug):raise ValueError('Invalid notebook slug')
    sources,spec=audit_sources(spec_dir);extracted=extract(sources)
    policies={name:sha256(protect(extracted[name]).encode()) for name in ('published-preprocess.py','published-infer.py')}
    own={name:(root/name).read_text() for name in BUNDLE}
    contract={'recipe':RECIPE,'policy':policy_contract(),'spec_sha256':SPEC_SHA256,'sources':SOURCE_HASHES,'weights':WEIGHTS,
        'source_url':spec['source_url'],'notebook_sha256':spec['notebook_sha256'],'frozen_at_utc':spec['frozen_at_utc'],
        'target_order':spec['target_order'],'exception_policy_sha256':policies,'codecs':CODECS,
        'private':True,'internet':False,'runtime_sources':{n:sha256(v.encode()) for n,v in own.items()},
        'builder_sha256':sha256(Path(__file__).read_bytes())}
    digest=sha256(canonical_bytes(own));revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    code=("import time,signal\nVERSIA_STARTED=time.monotonic()\n"
          "def _versia_timeout(signum,frame):\n    raise TimeoutError('Versia actual nine-hour platform limit exhausted')\n"
          "signal.signal(signal.SIGALRM,_versia_timeout)\nsignal.setitimer(signal.ITIMER_REAL,32400)\n"
          "import os,sys\nfrom pathlib import Path\nproject=Path('/kaggle/working/project')\n"
          f'sources={own!r}\n'
          "for name,content in sources.items():\n    p=project/name\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_text(content)\n"
          "sys.path.insert(0,str(project))\n"
          f"os.environ['KNEE_SOURCE_SHA256']={digest!r}\nos.environ['KNEE_CODE_REVISION']={revision!r}\n"
          f'VERSIA_SOURCES={sources!r}\nVERSIA_CONTRACT={contract!r}\n'
          "exec(compile(sources['scripts/versia_rehearsal_entry.py'],'<versia-rehearsal-entry>','exec'))\n")
    compile(code,'<versia-rehearsal>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
         'language_info':{'name':'python','version':'3.12'}},'cells':[
       {'cell_type':'markdown','id':'attribution','metadata':{},'source':[
         '# Frozen Versia ConvNeXt16 component qualification\n',
         '[Published Versia snapshot](https://www.kaggle.com/code/evgendvorkin/rsna-versia-5), by evgendvorkin; ConvNeXt source/weights by goodpjw2008 (Apache-2.0).\n',
         'Private offline 64 ID-selected training studies plus 3 visible test studies; image-only, no labels or fitting.\n',
         'Published16 windows, paired fat/nonfat slot substitution, three pinned folds, batch4/workers4. Exact public healthy tensors; partial-input policy/resource failures reported explicitly. No mirror, fusion or submission.\n',
         f'Frozen specification SHA256 `{SPEC_SHA256}`. Actual9-hour ceiling; ETA informational.\n']},
       {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    raw=(json.dumps(nb,indent=2)+'\n').encode()
    if len(raw)>900000 or len(code.encode())>900000:raise ValueError('Versia notebook exceeds source-size limit')
    meta={'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
          'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
          'dataset_sources':[CONVNEXT_DATASET],'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE}
    receipt={'source_bundle_sha256':digest,'contract_sha256':sha256(canonical_bytes(contract)),'notebook_sha256':sha256(raw),
             'notebook_bytes':len(raw),'source_revision':revision,'spec_sha256':SPEC_SHA256}
    destination.mkdir(parents=True,exist_ok=False);(destination/'notebook.ipynb').write_bytes(raw)
    for name,value in [('kernel-metadata.json',meta),('versia-contract.json',contract),('build-receipt.json',receipt)]:
        (destination/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('slug')
    p.add_argument('--spec-dir',required=True,type=Path)
    print(json.dumps(build_versia(**vars(p.parse_args())),indent=2))
