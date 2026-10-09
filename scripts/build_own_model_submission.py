"""Build-only immutable offline own-model entry, gated on five-fold rehearsal."""
import argparse,ast,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from knee import own_model_submission as final,public_training_smoke as smoke
from knee.reference_blend import DOCKER_IMAGE
from knee.coverage_inference import CODECS
FILES=('knee/__init__.py','knee/own_model_rehearsal.py','knee/corpus44_parity.py','knee/public_training_smoke.py',
       'knee/raptor_input.py','knee/reference.py','knee/fresh_reader_setup.py')


def archived(build):
    build=Path(build);names=('notebook.ipynb','build-contract.json','build-receipt.json','kernel-metadata.json')
    raw={name:(build/name).read_bytes() for name in names};nb,c,b,meta=[json.loads(raw[n]) for n in names]
    cells=[v for v in nb['cells'] if v['cell_type']=='code']
    if len(cells)!=1:raise ValueError('One archived executable required')
    code=''.join(cells[0]['source']);values={}
    for node in ast.parse(code).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('SOURCES','CONTRACT','READER','MODEL'):
            name=node.targets[0].id
            if name in values:raise ValueError('Duplicate archive assignment')
            values[name]=ast.literal_eval(node.value)
    if set(values)!={'SOURCES','CONTRACT','READER','MODEL'}:raise ValueError('Archived literals absent')
    sources,reader,model=values['SOURCES'],values['READER'],values['MODEL']
    if (values['CONTRACT']!=c or b.get('notebook_sha256')!=smoke.digest(raw['notebook.ipynb']) or
        b.get('notebook_bytes')!=len(raw['notebook.ipynb']) or b.get('contract_sha256')!=smoke.digest(smoke.canonical(c)) or
        b.get('candidate_folds')!=list(range(5)) or b.get('complete_family') is not True or
        set(sources)!=set(FILES) or set(c['sources'])!=set(FILES) or c['codecs']!=CODECS):
        raise ValueError('Archived bundle/contract binding differs')
    for name in FILES:
        if sources[name]!=(ROOT/name).read_text() or smoke.digest(sources[name].encode())!=c['sources'][name]:
            raise ValueError('Qualified scientific dependency changed; new rehearsal required')
    if smoke.digest(reader.encode())!=c['reader_sha256'] or smoke.digest(model.encode())!=c['model_definitions_sha256']:
        raise ValueError('Frozen scientific definitions differ')
    expected=("import time,sys,hashlib\nfrom pathlib import Path\nSTARTED=time.monotonic()\n"
        f"SOURCES={sources!r}\nCONTRACT={c!r}\nREADER={reader!r}\nMODEL={model!r}\n"
        "base=Path('/kaggle/working/own_model_code')\nfor name,source in SOURCES.items():\n"
        "    if hashlib.sha256(source.encode()).hexdigest()!=CONTRACT['sources'][name]:raise ValueError('Packaged sources differ')\n"
        "    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
        "sys.path.insert(0,str(base))\nfrom knee.own_model_rehearsal import run\n"
        "run('/kaggle/input','/kaggle/working/own_model_rehearsal',CONTRACT,READER,MODEL,started=STARTED)\n")
    if code!=expected:raise ValueError('Archived executable differs from exact reconstruction')
    expected_kernels=[smoke.IDENTITY_KERNEL,*final.KERNELS]
    if (meta.get('enable_internet') is not False or meta.get('enable_gpu') is not True or meta.get('is_private') is not True or
        meta.get('dataset_sources')!=[*smoke.CORPORA,'goodpjw2008/rsna-knee-2-5d-convnext-reader'] or
        meta.get('kernel_sources')!=expected_kernels or meta.get('competition_sources')!=['rsna-knee-abnormality-detection'] or
        meta.get('model_sources')!=[] or meta.get('docker_image')!=DOCKER_IMAGE):raise ValueError('Offline rehearsal environment differs')
    return sources,reader,model,c,{name:smoke.digest(value) for name,value in raw.items()}


def build(destination,slug,*,rehearsal_build,rehearsal_report,rehearsal_csv):
    out=Path(destination)
    if out.exists():raise ValueError('New immutable build directory required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug):raise ValueError('Invalid slug')
    # Reject subsets and non-PASS evidence before parsing/packaging any model code.
    original=json.loads((Path(rehearsal_build)/'build-contract.json').read_text())
    raw_report=Path(rehearsal_report).read_bytes();report=json.loads(raw_report)
    final.require_qualification(original,report)
    smoke.require_file(rehearsal_csv,report['visible_csv_sha256'])
    with Path(rehearsal_csv).open(newline='') as f:
        import csv
        rows=list(csv.reader(f))
    if len(rows)!=4:raise ValueError('Exact three-study rehearsal CSV required')
    ids=[r[0] for r in rows[1:]]
    if len(set(ids))!=3:raise ValueError('Unique visible coverage required')
    final.validate_csv(rehearsal_csv,ids)
    sources,reader,model,original,archive_hashes=archived(rehearsal_build)
    # Retain only validated aggregate evidence: no arbitrary report fields or test IDs.
    evidence={k:report[k] for k in ('status','contract_sha256','qualified_folds','train_studies','test_studies','settings','labels_read',
                                   'stress_checks','environment','elapsed_seconds','visible_csv_sha256')}
    evidence['training_parity']=[{'slots':[{'exact_hash_match':s['exact_hash_match'],'mask_match':s['mask_match']} for s in r['slots']],
                                 'windows_exact':r['windows_exact']} for r in report['training_parity']]
    keys=('fold','checkpoint_sha256','strict_load','repeat_max_error','train_probability_max_error','missing_plane_finite','cuda_peak_bytes')
    evidence['models']=[{k:r[k] for k in keys} for r in report['models']]
    # Packages/environment and stress summaries originate from the pinned runtime.
    evidence['environment']={'packages':{k:report['environment']['packages'][k] for k in ('torch','timm')}}
    evidence['stress_checks']={'cases':report['stress_checks']['cases']}
    final.require_qualification(original,evidence)
    for name in ('knee/reference_blend.py','knee/public_reader_contract.py','knee/own_model_submission.py'):
        sources[name]=(ROOT/name).read_text()
    for name,source in sources.items():compile(source,name,'exec')
    contract={'version':1,'rehearsal_contract':original,'rehearsal_report':evidence,
              'rehearsal_report_canonical_sha256':smoke.digest(smoke.canonical(evidence)),
              'rehearsal_report_sha256':smoke.digest(raw_report),'archive_sha256':archive_hashes,
              'sources':{name:smoke.digest(source.encode()) for name,source in sources.items()},
              'visible_ids_sha256':smoke.digest(smoke.canonical(ids)),
              'max_seconds':final.MAX_SECONDS,'builder_sha256':smoke.file_sha(__file__),
              'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    entry=("import time,signal\nSTARTED=time.monotonic()\n"
           "def _expired(signum,frame):raise TimeoutError('Actual nine-hour inference deadline exceeded')\n"
           f"signal.signal(signal.SIGALRM,_expired)\nsignal.setitimer(signal.ITIMER_REAL,{final.MAX_SECONDS})\n"
           "import sys,hashlib\nfrom pathlib import Path\n"
           f"SOURCES={sources!r}\nCONTRACT={contract!r}\nREADER={reader!r}\nMODEL={model!r}\n"
           "base=Path('/kaggle/working/own_model_code')\nfor name,source in SOURCES.items():\n"
           "    if hashlib.sha256(source.encode()).hexdigest()!=CONTRACT['sources'][name]:raise ValueError('Packaged sources differ')\n"
           "    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
           "sys.path.insert(0,str(base))\nfrom knee.own_model_submission import run\n"
           "run('/kaggle/input','/kaggle/working/own_model_submission',CONTRACT,READER,MODEL,started=STARTED)\n"
           "import shutil\nshutil.copyfile('/kaggle/working/own_model_submission/submission.csv','/kaggle/working/submission.csv')\n")
    compile(entry,'own-submission-notebook','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':['# Own five-fold ConvNeXt image-only inference\n','Frozen five-fold SWA, CUDAFP16,256,k16,ImageNet; equal probability mean, no TTA.\n','Test-only streaming. Actual nine-hour deadline; no network, training corpus, labels or private identity.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':entry.splitlines(True)}]}
    blob=(json.dumps(nb,indent=2)+'\n').encode()
    if len(blob)>900000:raise ValueError('Notebook source limit')
    meta={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
          'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
          'dataset_sources':['goodpjw2008/rsna-knee-2-5d-convnext-reader'],'kernel_sources':final.KERNELS,'model_sources':[],
          'docker_image':DOCKER_IMAGE,'machine_shape':'NvidiaTeslaT4'}
    receipt={'build_only':True,'notebook_sha256':smoke.digest(blob),'notebook_bytes':len(blob),
             'contract_sha256':smoke.digest(smoke.canonical(contract)),'qualified_folds':list(range(5)),'production_cuda_tested':False}
    out.mkdir(parents=True);(out/'notebook.ipynb').write_bytes(blob)
    for name,value in [('kernel-metadata.json',meta),('build-contract.json',contract),('build-receipt.json',receipt)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('slug')
    for name in ('rehearsal-build','rehearsal-report','rehearsal-csv'):p.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
