"""Separate frozen CoAtNet224 build-only rehearsal; subset wiring proof, no launch.

Copied from build_own_model_rehearsal at56516d0 so qualified ConvNeXt archives
stay unchanged. This family accepts only native224, original labels and fixed
ten-epoch report-selected SWA. It adds no inference training/label reads.
"""
import argparse,ast,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from knee import coatnet224_rehearsal as own
from knee import corpus44_parity as parity
from knee import public_training_smoke as smoke
from knee import raptor_input
from knee.reference_blend import DOCKER_IMAGE
from knee.coverage_inference import CODECS
FILES=('knee/__init__.py','knee/coatnet224_rehearsal.py','knee/corpus44_parity.py','knee/public_training_smoke.py',
       'knee/raptor_input.py','knee/reference.py','knee/fresh_reader_setup.py')


def build(destination,slug,*,public_model_source,public_reader_source,manifest,smoke_contract,parity_report,parity_review,result,folds):
    kernels=own.KERNELS
    out=Path(destination)
    if out.exists():raise ValueError('New immutable build directory required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug):raise ValueError('Invalid slug')
    selected,cohort=smoke.membership(manifest);old=json.loads(Path(smoke_contract).read_text());smoke.require_contract(old)
    if cohort!=old['cohort']:raise ValueError('Smoke cohort differs')
    review=json.loads(Path(parity_review).read_text());p=json.loads(Path(parity_report).read_text())
    if (smoke.file_sha(parity_report)!=review['report_sha256'] or review['status']!='PASS_BOUNDED_PIXEL_PARITY' or
        p['status']!='EXACT_MATCH' or len(p['studies'])!=5 or
        not all(s['exact_hash_match'] and s['mask_match'] for r in p['studies'] for s in r['slots'])):
        raise ValueError('Completed raw/corpus parity evidence required')
    reader=parity.recipe(Path(public_reader_source).read_text());model=own.model_definitions(Path(public_model_source).read_text())
    if smoke.digest(reader.encode())!='e331776e19800ef584c653cb9bb90698ff708027b0f9f483ce7afb18338a2f35':
        raise ValueError('Frozen real-qualified reader differs')
    protected,protected_meta=raptor_input.transform_exception_handlers(reader)
    folds=[int(f) for f in folds.split(',')]
    if not folds or folds!=sorted(set(folds)) or any(f not in range(5) for f in folds):raise ValueError('Explicit orderedfold set required')
    result=Path(result);members=[]
    for fold in folds:
        runfile=f'run_fold{fold}.json';record=json.loads((result/runfile).read_text())
        name=record['checkpoint_file'];path=result/name
        if Path(name).name!=name:raise ValueError('Checkpoint filename unsafe')
        member={'fold':fold,'kernel':kernels[0 if fold<2 else (1 if fold<4 else 2)],
            'run_file':runfile,'run_sha256':smoke.file_sha(result/runfile),'checkpoint_file':name,
            'checkpoint_sha256':smoke.file_sha(path),'selected_epochs':record['selected_epochs']}
        own.verify_run(record,member)
        patched=Path(public_model_source).with_name('train_knee_patched.py')
        smoke.require_file(patched,record['train_knee_patched_sha256'])
        for filename,expected in [('train_knee.patch',own.PATCH_SHA),('tracka_entry.py',own.ENTRY_SHA),
                                  ('folds_v0.json',own.POLICY['folds_sha256']),('excluded_ids.json',own.POLICY['excluded_ids_sha256'])]:
            smoke.require_file(Path(public_model_source).with_name(filename),expected)
        nodes=[n for n in ast.parse(patched.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in
               ('StudyWindows','build_backbone','RaptorClassifier')]
        if ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n'!=model:
            raise ValueError('Patched fit model/window definitions differ from frozen inference source')
        members.append(member)
    sources={name:(ROOT/name).read_text() for name in FILES}
    for name,source in sources.items():compile(source,name,'exec')
    contract={'version':1,'settings':own.SETTINGS,'members':members,'complete_family':folds==list(range(5)),
        'smoke_cohort':cohort,'selected5_sha256':smoke.digest(smoke.canonical(selected[:5])),
        'parity_report_sha256':review['report_sha256'],'parity_kernel':review['kernel'],
        'reader_sha256':smoke.digest(reader.encode()),'protected_reader_sha256':protected_meta['sha256'],
        'model_source_sha256':smoke.SOURCE_SHA,'model_definitions_sha256':smoke.digest(model.encode()),
        'sources':{name:smoke.digest(source.encode()) for name,source in sources.items()},'codecs':CODECS,
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'limits':['five training studies and three visible test studies; no hidden test qualification',
            'public mask/window gap and series boundary behavior retained; noTTA',
            'subsetfold mean is a wiring proof; production family requires allfive qualifiedfolds']}
    contract.update(own.POLICY)
    own.require_contract(contract)
    entry=("import time,sys,hashlib\nfrom pathlib import Path\nSTARTED=time.monotonic()\n"
        f"SOURCES={sources!r}\nCONTRACT={contract!r}\nREADER={reader!r}\nMODEL={model!r}\n"
        "base=Path('/kaggle/working/coatnet224_code')\nfor name,source in SOURCES.items():\n"
        "    if hashlib.sha256(source.encode()).hexdigest()!=CONTRACT['sources'][name]:raise ValueError('Packaged sources differ')\n"
        "    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
        "sys.path.insert(0,str(base))\nfrom knee.coatnet224_rehearsal import run\n"
        "run('/kaggle/input','/kaggle/working/coatnet224_rehearsal',CONTRACT,READER,MODEL,started=STARTED)\n")
    compile(entry,'own-rehearsal-notebook','exec')
    if any(u in entry for u in selected):raise ValueError('Private ID embedded')
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':['# CoAtNet224 image-only rehearsal\n',
          f'Explicit qualified candidatefolds{folds}; strictSWA load, CUDAFP16,224,k_eval16,ImageNet.\n',
          'Five frozen training inputs compared with corpus at pixel/mask/window/probability levels; three visible studies and exact repeat.\n',
          'No training labels/reports/gold scores. Equal probability mean; noTTA or tuned weights. Frozen44 reader with existing fatal guards.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':entry.splitlines(True)}]}
    blob=(json.dumps(notebook,indent=2)+'\n').encode()
    if len(blob)>900000:raise ValueError('Notebook source limit')
    meta={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,
        'competition_sources':['rsna-knee-abnormality-detection'],
        'dataset_sources':[*smoke.CORPORA,'goodpjw2008/rsna-knee-2-5d-convnext-reader'],
        'kernel_sources':list(dict.fromkeys([smoke.IDENTITY_KERNEL,*[m['kernel'] for m in members]])),
        'model_sources':[],'docker_image':DOCKER_IMAGE,'machine_shape':'NvidiaTeslaT4'}
    receipt={'notebook_sha256':smoke.digest(blob),'notebook_bytes':len(blob),'contract_sha256':smoke.digest(smoke.canonical(contract)),
             'candidate_folds':folds,'complete_family':contract['complete_family'],'build_only':True}
    out.mkdir(parents=True)
    (out/'notebook.ipynb').write_bytes(blob)
    for name,value in [('kernel-metadata.json',meta),('build-contract.json',contract),('build-receipt.json',receipt)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);p.add_argument('slug')
    for name in ('public-model-source','public-reader-source','manifest','smoke-contract','parity-report','parity-review','result'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--folds',required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
