"""Build only: authentic v3 MaViT67 -> raw MRI diagnostic on fixed train2000 membership."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import build_mavit_submission as archived
from knee import mavit_checkpoint as cp,mavit_rehearsal as reference,mavit_diagnostic as diagnostic
from knee.mavit_submission import VERSION as PRODUCTION_VERSION,read_predictions
from knee.reference_blend import CONVNEXT_DATASET,DOCKER_IMAGE

NEW_RUNTIME=('knee/mavit_submission.py','knee/mavit_diagnostic.py','scripts/mavit_diagnostic_entry.py')


def build(destination,slug,*,rehearsal_build,rehearsal_report,rehearsal_csv,plan):
    destination=Path(destination)
    if destination.exists():raise ValueError('New immutable diagnostic build required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug):raise ValueError('Invalid private notebook slug')
    files,original,archive=archived.archived_build(rehearsal_build)
    raw=Path(rehearsal_report).read_bytes()
    if cp.digest(raw)!=diagnostic.REHEARSAL_REPORT_SHA256:raise ValueError('Authentic reviewed v3 MaViT67 receipt required')
    report=json.loads(raw);reference.require_rehearsal(original,report)
    if report.get('source_revision')!=original['source_revision'] or report.get('visible_csv_sha256')!=cp.digest(Path(rehearsal_csv).read_bytes()):
        raise ValueError('MaViT67 receipt/CSV must match immutable archive')
    columns,visible_ids,_=read_predictions(rehearsal_csv)
    if len(visible_ids)!=3:raise ValueError('Exact visible MaViT67 coverage required')
    ids,cohort=diagnostic.membership(plan);del ids,visible_ids
    sources=dict(files);sources.update({name:(ROOT/name).read_text() for name in NEW_RUNTIME})
    q={'version':PRODUCTION_VERSION,'rehearsal_contract':original,'archive_sha256':archive,'rehearsal_report_sha256':cp.digest(raw),
       'rehearsal_csv_sha256':report['visible_csv_sha256'],'targets':columns[1:],'packages':report['environment']['packages'],
       'backend_flags':report['backend_flags'],'cross_run_atol':diagnostic.ATOL,'repeat_atol':diagnostic.ATOL,
       'runtime_sources':{name:cp.digest(value.encode()) for name,value in sources.items()},'builder_sha256':cp.digest(Path(__file__).read_bytes()),
       'training_lineage':'unknown','quality_and_independence':'unverified; robustness diagnostic only'}
    production={'version':PRODUCTION_VERSION,'qualification':q,'qualification_sha256':cp.digest(cp.canonical(q)),
                'visible':True,'max_seconds':diagnostic.MAX_SECONDS,'private':True,'internet':False,'visible_evidence':None}
    contract={'version':diagnostic.VERSION,'production_contract':production,'cohort':cohort,'max_seconds':diagnostic.MAX_SECONDS,
              'repeat_atol':diagnostic.ATOL,'progress_interval':diagnostic.INTERVAL,'prediction_exports':False}
    diagnostic.qualification(contract)
    code=("import time,signal\nMAVIT_STARTED=time.monotonic()\n"
          "def _mavit_expired(signum,frame):\n    raise TimeoutError('MaViT diagnostic actual9h deadline exhausted')\n"
          f"signal.signal(signal.SIGALRM,_mavit_expired)\nsignal.setitimer(signal.ITIMER_REAL,{diagnostic.MAX_SECONDS})\n"
          "import os,sys\nfrom pathlib import Path\nproject=Path('/kaggle/working/project')\n"
          f'sources={sources!r}\n'
          "for name,content in sources.items():\n    p=project/name\n    p.parent.mkdir(parents=True,exist_ok=True)\n    p.write_text(content)\n"
          "sys.path.insert(0,str(project))\n"
          f'MAVIT_CONTRACT={contract!r}\n'
          "exec(compile(sources['scripts/mavit_diagnostic_entry.py'],'<mavit-diagnostic-entry>','exec'))\n")
    compile(code,'<mavit-train-diagnostic>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.12'}},'cells':[
        {'cell_type':'markdown','id':'attribution','metadata':{},'source':[
            '# MaViT raw-image train2000 robustness diagnostic\n',
            '[hengck23 public model and reader](https://www.kaggle.com/datasets/hengck23/hengck23-rnsa-knee), MIT declared; training lineage/independence unknown.\n',
            'Exact v3-qualified6slot/8triplet288 reader, batch1 CUDA FP16 model.infer, unchanged sigmoid and1e-6 repeat guard. No alternate-series, precision or device retry.\n',
            'Only uid/split membership from the byte-pinned prior TRAIN04 plan is used. No labels, reports, development/sealed/gold images, cached arrays, metrics, predictions or checkpoint exports. Attached original plan contains unused weak labels.\n',
            'Stops at first failure with sanitized phase/ordinal/class/source function and aggregate counts/resources. Actual9h setup-inclusive limit; this cannot establish hidden-test robustness or predictive quality.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    notebook=(json.dumps(nb,indent=2)+'\n').encode()
    if len(notebook)>900000 or len(code.encode())>900000:raise ValueError('Diagnostic notebook exceeds source limit')
    metadata={'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
        'dataset_sources':[cp.DATASET,CONVNEXT_DATASET],'kernel_sources':[diagnostic.KERNEL],'model_sources':[],'docker_image':DOCKER_IMAGE}
    receipt={'notebook_sha256':cp.digest(notebook),'notebook_bytes':len(notebook),'contract_sha256':cp.digest(cp.canonical(contract)),
        'source_bundle_sha256':cp.digest(cp.canonical(sources)),'rehearsal_report_sha256':cp.digest(raw),
        'build_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    destination.mkdir(parents=True,exist_ok=False);(destination/'notebook.ipynb').write_bytes(notebook)
    for name,value in [('kernel-metadata.json',metadata),('mavit-contract.json',contract),('build-receipt.json',receipt)]:
        (destination/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('destination',type=Path);parser.add_argument('slug')
    for option in ('rehearsal-build','rehearsal-report','rehearsal-csv','plan'):parser.add_argument('--'+option,type=Path,required=True)
    print(json.dumps(build(**vars(parser.parse_args())),indent=2))
