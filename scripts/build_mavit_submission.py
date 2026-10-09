"""Build only: authentic MaViT67 archive -> visible production -> full-test entry."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import build_mavit_rehearsal as rehearsal_builder
from knee import mavit_checkpoint as cp,mavit_rehearsal as reference
from knee.mavit_submission import VERSION,ATOL,MAX_SECONDS,compare_csv,read_predictions,require_result
from knee.reference_blend import CONVNEXT_DATASET,DOCKER_IMAGE

NEW_RUNTIME=('knee/mavit_submission.py','scripts/mavit_submission_entry.py')


def archived_build(directory):
    directory=Path(directory)
    raw={n:(directory/n).read_bytes() for n in ('notebook.ipynb','mavit-contract.json','build-receipt.json','kernel-metadata.json')}
    nb,c,b,meta=(json.loads(raw[n]) for n in raw);cells=[x for x in nb['cells'] if x['cell_type']=='code']
    if len(cells)!=1:raise ValueError('One exact archived rehearsal executable required')
    code=''.join(cells[0]['source']);values={}
    for node in ast.parse(code).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('sources','MAVIT_CONTRACT'):
            key=node.targets[0].id
            if key in values:raise ValueError('Duplicate archived source/contract literal')
            values[key]=ast.literal_eval(node.value)
    files=values.get('sources',{})
    if (set(files)!=set(rehearsal_builder.BUNDLE) or any(not isinstance(v,str) for v in files.values()) or values.get('MAVIT_CONTRACT')!=c or
            b.get('notebook_sha256')!=cp.digest(raw['notebook.ipynb']) or b.get('notebook_bytes')!=len(raw['notebook.ipynb']) or
            b.get('contract_sha256')!=cp.digest(cp.canonical(c)) or b.get('source_bundle_sha256')!=cp.digest(cp.canonical(files)) or
            b.get('source_revision')!=c.get('source_revision') or not re.fullmatch('[0-9a-f]{40}',c.get('source_revision','')) or
            c.get('builder_sha256')!=cp.digest(Path(rehearsal_builder.__file__).read_bytes()) or
            c.get('runtime_sources')!={n:cp.digest(v.encode()) for n,v in files.items()}):
        raise ValueError('Archived MaViT build/source/contract integrity differs')
    expected=("import time,signal\nMAVIT_STARTED=time.monotonic()\n"
          "def _mavit_expired(signum,frame):\n    raise TimeoutError('MaViT actual9h deadline exhausted')\n"
          "signal.signal(signal.SIGALRM,_mavit_expired)\nsignal.setitimer(signal.ITIMER_REAL,32400)\n"
          "import os,sys\nfrom pathlib import Path\nproject=Path('/kaggle/working/project')\n"
          f'sources={files!r}\n'
          "for name,content in sources.items():\n    p=project/name\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_text(content)\n"
          "sys.path.insert(0,str(project))\n"
          f'MAVIT_CONTRACT={c!r}\n'
          "exec(compile(sources['scripts/rehearse_mavit_entry.py'],'<mavit-rehearsal-entry>','exec'))\n")
    if code!=expected:raise ValueError('Archived MaViT executable differs from exact source reconstruction')
    for name,value in files.items():
        if (ROOT/name).read_text()!=value:raise ValueError('Scientific dependency changed after MaViT67; new qualification required')
    if (meta.get('is_private') is not True or meta.get('enable_internet') is not False or meta.get('enable_gpu') is not True or
            meta.get('enable_tpu') is not False or meta.get('docker_image')!=DOCKER_IMAGE or
            meta.get('dataset_sources')!=[cp.DATASET,CONVNEXT_DATASET] or meta.get('competition_sources')!=['rsna-knee-abnormality-detection'] or
            meta.get('kernel_sources')!=[] or meta.get('model_sources')!=[]):
        raise ValueError('Archived MaViT offline environment differs')
    return files,c,{n:cp.digest(v) for n,v in raw.items()}


def build(destination,slug,*,rehearsal_build,rehearsal_report,rehearsal_csv,submission=False,visible_report=None,visible_csv=None):
    destination=Path(destination)
    if destination.exists():raise ValueError('New immutable standalone build directory required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug):raise ValueError('Invalid private notebook slug')
    files,original,archive=archived_build(rehearsal_build)
    raw=Path(rehearsal_report).read_bytes();report=json.loads(raw);reference.require_rehearsal(original,report)
    if report.get('source_revision')!=original['source_revision'] or report.get('visible_csv_sha256')!=cp.digest(Path(rehearsal_csv).read_bytes()):
        raise ValueError('Authentic MaViT67 receipt/output must match archived build')
    compare_csv(rehearsal_csv,rehearsal_csv);columns,_,_=read_predictions(rehearsal_csv)
    sources=dict(files);sources.update({n:(ROOT/n).read_text() for n in NEW_RUNTIME})
    q={'version':VERSION,'rehearsal_contract':original,'archive_sha256':archive,'rehearsal_report_sha256':cp.digest(raw),
       'rehearsal_csv_sha256':report['visible_csv_sha256'],'targets':columns[1:],
       'packages':report['environment']['packages'],'backend_flags':report['backend_flags'],'cross_run_atol':ATOL,'repeat_atol':ATOL,
       'runtime_sources':{n:cp.digest(v.encode()) for n,v in sources.items()},'builder_sha256':cp.digest(Path(__file__).read_bytes()),
       'training_lineage':'unknown','quality_and_independence':'unverified; standalone qualification only'}
    contract={'version':VERSION,'qualification':q,'qualification_sha256':cp.digest(cp.canonical(q)),
              'visible':not submission,'max_seconds':MAX_SECONDS,'private':True,'internet':False,'visible_evidence':None}
    if submission:
        if visible_report is None or visible_csv is None:raise ValueError('Full MaViT entry requires passing visible report/CSV')
        visible_contract=dict(contract,visible=True);vr=Path(visible_report).read_bytes();visible=json.loads(vr)
        require_result(visible,visible_contract,visible_csv)
        error=compare_csv(rehearsal_csv,visible_csv)
        contract['visible_evidence']={'report_sha256':cp.digest(vr),'csv_sha256':cp.digest(Path(visible_csv).read_bytes()),
                                     'immutable67_max_error':error,'absolute_tolerance':ATOL}
    elif visible_report is not None or visible_csv is not None:raise ValueError('Visible evidence is only for full entry builds')
    code=("import time,signal\nMAVIT_STARTED=time.monotonic()\n"
          "def _mavit_expired(signum,frame):\n    raise TimeoutError('MaViT actual9h deadline exhausted')\n"
          f"signal.signal(signal.SIGALRM,_mavit_expired)\nsignal.setitimer(signal.ITIMER_REAL,{MAX_SECONDS})\n"
          "import os,sys\nfrom pathlib import Path\nproject=Path('/kaggle/working/project')\n"
          f'sources={sources!r}\n'
          "for name,content in sources.items():\n    p=project/name\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_text(content)\n"
          "sys.path.insert(0,str(project))\n"
          f'MAVIT_CONTRACT={contract!r}\n'
          "exec(compile(sources['scripts/mavit_submission_entry.py'],'<mavit-submission-entry>','exec'))\n")
    compile(code,'<mavit-standalone>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'cells':[
      {'cell_type':'markdown','id':'attribution','metadata':{},'source':[
       '# Standalone MaViT image-only inference\n',
       '[hengck23 public model and reader](https://www.kaggle.com/datasets/hengck23/hengck23-rnsa-knee), MIT declared; MALA encoder, active decoder attributes nartaa. Training lineage/independence unknown.\n',
       'Exact qualified288/6slots/8triplets, batch1 CUDA FP16 model.infer, raw sigmoid probabilities, no TTA/fusion/labels/report input. Fixed reader malformed-input policy; no alternative-series/prediction/precision/device fallback.\n',
       f'Immutable67 receipt `{q["rehearsal_report_sha256"]}`. SciPy `{q["packages"]["scipy"]}` pinned from that execution. Cross-run and repeat absolute tolerance1e-6; actual9h setup-inclusive limit.\n']},
      {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    notebook=(json.dumps(nb,indent=2)+'\n').encode()
    if len(notebook)>900000 or len(code.encode())>900000:raise ValueError('Standalone notebook exceeds source limit')
    metadata={'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
       'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
       'dataset_sources':[cp.DATASET,CONVNEXT_DATASET],'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE}
    receipt={'notebook_sha256':cp.digest(notebook),'notebook_bytes':len(notebook),'qualification_sha256':contract['qualification_sha256'],
             'contract_sha256':cp.digest(cp.canonical(contract)),'source_bundle_sha256':cp.digest(cp.canonical(sources)),
             'build_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    destination.mkdir(parents=True,exist_ok=False);(destination/'notebook.ipynb').write_bytes(notebook)
    for name,value in [('kernel-metadata.json',metadata),('mavit-contract.json',contract),('build-receipt.json',receipt)]:
        (destination/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path);parser.add_argument('slug')
    for option in ('rehearsal-build','rehearsal-report','rehearsal-csv'):parser.add_argument('--'+option,type=Path,required=True)
    parser.add_argument('--submission',action='store_true');parser.add_argument('--visible-report',type=Path);parser.add_argument('--visible-csv',type=Path)
    print(json.dumps(build(**vars(parser.parse_args())),indent=2))
