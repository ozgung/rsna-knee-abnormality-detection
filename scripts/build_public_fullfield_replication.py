"""Build the fixed fold1 replication privately/offline; never launch a job."""
import argparse
import ast
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee.reference import audit_source,canonical_bytes,sha256,WEIGHT_DATASET
from knee.reference_blend import CONVNEXT_DATASET,DOCKER_IMAGE
from knee.coverage_inference import CODECS
from knee.public_adaptation_fit import FOLD_SHA,EXCLUDED_SHA,require,fingerprint
from knee.public_adaptation_pilot import LABEL_SHA,MASK_SHA,file_sha,load_labels
from knee.public_fullfield_replication import (config,freeze_sample,membership,construct_preflight,
    validate_fold0,FOLD0_RECEIPT_SHA,FOLD0_NOTEBOOK_SHA)
from scripts.build_public_fullfield import notebook_literals,SOURCES as QUALIFIED_SOURCES,REFERENCE_NOTEBOOK_SHA

SOURCES=QUALIFIED_SOURCES+('knee/public_fullfield_replication.py',)


def pinned(path,pin):
    require(file_sha(path)==pin,'IMMUTABLE_UPSTREAM_INPUT')
    return Path(path).read_bytes()


def qualified_sources(path):
    nb=json.loads(pinned(Path(path)/'notebook.ipynb',FOLD0_NOTEBOOK_SHA))
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    tables={}
    for n in ast.parse(code).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('sources','expected_hashes','expected_bundle'):
            tables[n.targets[0].id]=ast.literal_eval(n.value)
    require(set(tables)=={'sources','expected_hashes','expected_bundle'},'QUALIFIED_SOURCE_TABLES')
    ss=tables['sources'];require(sha256(canonical_bytes(ss))==tables['expected_bundle'],'QUALIFIED_BUNDLE')
    require({n:sha256(v.encode()) for n,v in ss.items()}==tables['expected_hashes'],'QUALIFIED_SOURCE_HASHES')
    return ss,tables['expected_bundle']


def build(source_path,reference_notebook,fold0_build,fold0_receipt,label_path,mask_path,data_root,destination,slug):
    import pyarrow.parquet as pq
    root=Path(__file__).resolve().parents[1];data=Path(data_root).resolve();dest=Path(destination)
    require(not dest.exists(),'FRESH_IMMUTABLE_BUILD')
    require(dest.resolve().is_relative_to(root/'outputs'),'PRIVATE_IGNORED_OUTPUTS')
    require(re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug),'SLUG')
    original=notebook_literals(reference_notebook);old=original['BINDING']
    require(old['fold_file_sha256']==FOLD_SHA and old['excluded_file_sha256']==EXCLUDED_SHA,'ORIGINAL_MEMBERSHIP_PINS')
    public,decoder,manifest=audit_source(Path(source_path).read_text(),profile='mirror')
    require((public,decoder,manifest)==(original['PUBLIC_SOURCE'],original['DECODER_SOURCE'],original['MANIFEST']),'ORIGINAL_PUBLIC_CONTRACT')
    upstream=json.loads(pinned(fold0_receipt,FOLD0_RECEIPT_SHA));validate_fold0(upstream,manifest)
    qualified,upstream_bundle=qualified_sources(fold0_build)
    require(upstream['binding']['source_bundle_sha256']==upstream_bundle,'FOLD0_RUNTIME_SOURCE_BINDING')
    sources={n:(root/n).read_bytes().decode() for n in SOURCES}
    require(all(sources[n]==qualified[n] for n in QUALIFIED_SOURCES),'QUALIFIED_PRIMITIVE_BYTES_CHANGED')
    require(file_sha(data/'raw/train.csv')==old['official_train_sha256'],'OFFICIAL_TRAIN_BINDING')
    require(file_sha(data/'raw/train_series.csv')==original['PREFLIGHT']['source_sha256']['data/raw/train_series.csv'],'OFFICIAL_SERIES_BINDING')
    require(file_sha(label_path)==LABEL_SHA and file_sha(mask_path)==MASK_SHA,'LABEL_TRUTH_BYTES')
    with (data/'raw/sample_submission.csv').open(newline='') as h:header=next(csv.reader(h))
    targets=original['PREFLIGHT']['targets'];require(header==['StudyInstanceUID']+targets,'OFFICIAL_TARGET_SCHEMA')
    with (data/'raw/train.csv').open(newline='') as h:rows=list(csv.DictReader(h))
    # Freeze from official presence-only membership before weak-label values/pixels.
    folds,excluded=original['FOLDS'],original['EXCLUDED'];sample=freeze_sample(rows,folds,excluded,targets)
    label_ids=pq.read_table(label_path,columns=['StudyInstanceUID']).column(0).to_pylist()
    require(len(label_ids)==len(set(label_ids)),'LABEL_ID_DUPLICATES')
    eligible,train,dev,_=membership(rows,label_ids,folds,excluded,targets)
    require(set(sample)<=set(train),'SAMPLE_TRAIN_ELIGIBILITY')
    _,masks=load_labels(label_path,mask_path,eligible,targets)
    pre,train,dev,groups=construct_preflight(rows,label_ids,folds,excluded,targets,masks)
    bundle=sha256(canonical_bytes(sources));revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    binding={'config':config(),'source_bundle_sha256':bundle,'public_manifest_sha256':fingerprint(manifest),'preflight_sha256':fingerprint(pre),
             'dev_order_sha256':fingerprint(dev),'fold_mapping_sha256':fingerprint(folds),'fold_file_sha256':FOLD_SHA,'exclusion_file_sha256':EXCLUDED_SHA,
             'exclusions_sha256':fingerprint(excluded),'official_train_sha256':old['official_train_sha256'],'series_sha256':file_sha(data/'raw/train_series.csv'),
             'official_schema_sha256':file_sha(data/'raw/sample_submission.csv'),'label_sha256':LABEL_SHA,'truth_sha256':MASK_SHA,
             'sample_sha256':fingerprint(sample),'sample_policy':config()['mechanics_sample_policy'],'groups_sha256':fingerprint(groups),
             'reference_notebook_sha256':REFERENCE_NOTEBOOK_SHA,'fold0_receipt_sha256':FOLD0_RECEIPT_SHA,'fold0_receipt_fingerprint':fingerprint(upstream),
             'fold0_notebook_sha256':FOLD0_NOTEBOOK_SHA,'qualified_primitives_bundle_sha256':upstream_bundle,'source_revision':revision,
             'source_state':'exact bundled working files; may include uncommitted replication implementation','historical_fold1_control':'unavailable; same-job reference qualification required'}
    entry=f'PUBLIC_SOURCE={public!r}\nDECODER_SOURCE={decoder!r}\nMANIFEST={manifest!r}\nPREFLIGHT={pre!r}\nFOLDS={folds!r}\nEXCLUDED={excluded!r}\nSAMPLE={sample!r}\nBINDING={binding!r}\nFOLD0={upstream!r}\nEXPECTED_CODECS={CODECS!r}\n'+'''
import importlib.metadata,subprocess
from knee.fresh_reader_setup import dataset_root
from knee.reference_blend import CONVNEXT_DATASET,select_codec_wheels
from knee.reference import WEIGHT_DATASET
from knee.runtime import competition_root
from knee.public_adaptation_pilot import file_sha
from knee.public_fullfield_replication import run
out=Path('/kaggle/working/public-fullfield-replication')
try:
    if sys.version_info[:2]!=(3,12) or importlib.metadata.version('torch')!='2.10.0+cu128' or importlib.metadata.version('timm')!='1.0.26':raise ValueError('Pinned CUDA environment differs')
    codec_root=dataset_root('/kaggle/input',CONVNEXT_DATASET,['pylibjpeg-2.1.0-py3-none-any.whl'])
    wheels=select_codec_wheels(codec_root)
    if {p.name:file_sha(p) for p in wheels}!=EXPECTED_CODECS:raise ValueError('Codec hashes differ')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)],check=True)
    weights=dataset_root('/kaggle/input',WEIGHT_DATASET,[MANIFEST['checkpoint_file']])
    labels=dataset_root('/kaggle/input','nartaa/rsna-knee-hpo-assets',['labels_v1_blend.parquet','train_labels_v1.parquet'])
    report=run(PUBLIC_SOURCE,DECODER_SOURCE,MANIFEST,weights/MANIFEST['checkpoint_file'],labels/'labels_v1_blend.parquet',labels/'train_labels_v1.parquet',competition_root(),out,FOLDS,EXCLUDED,PREFLIGHT,SAMPLE,BINDING,FOLD0)
    print(json.dumps({'status':report['status'],'optimizer_updates':0,'holdout_fold':1,'evaluated_studies':report['evaluated_studies']}),flush=True)
except BaseException as error:
    if not (out/'receipt.json').exists():
        out.mkdir(parents=True,exist_ok=True)
        (out/'receipt.json').write_text(json.dumps({'status':'FAIL','failure_stage':'offline_setup','failure_type':type(error).__name__,'binding':BINDING},indent=2)+chr(10))
    raise RuntimeError('Fullfield fold1 replication failed; inspect sanitized receipt') from None
'''
    hashes={n:sha256(v.encode()) for n,v in sources.items()}
    code="import os,sys,json,hashlib\nfrom pathlib import Path\nos.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')\n"+f'sources={sources!r}\nexpected_hashes={hashes!r}\nexpected_bundle={bundle!r}\n'+'''
if {n:hashlib.sha256(v.encode()).hexdigest() for n,v in sources.items()}!=expected_hashes:raise ValueError('Source bundle tampered')
if hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=expected_bundle:raise ValueError('Bundle digest differs')
project=Path('/kaggle/working/project')
for name,content in sources.items():
    p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode())
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected_hashes[name]:raise ValueError('Runtime source bytes changed')
sys.path.insert(0,str(project))
os.environ['KNEE_SOURCE_SHA256']=expected_bundle
'''+entry
    compile(code,'<public-fullfield-replication>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':['# Fixed fold1 public full384 replication\n','32 frozen training-only mechanics probes then conditional866 paired development studies. Zero optimizer updates; unchanged source/checkpoint/K94/mirror/grid adapters. No gold metrics, blending or search. Historical fold1 control unavailable; original reference qualified in this job.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    meta={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,
          'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
          'dataset_sources':[WEIGHT_DATASET,'nartaa/rsna-knee-hpo-assets',CONVNEXT_DATASET],'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE}
    raw=(json.dumps(nb,indent=2)+'\n').encode();require(len(raw)<900000,'NOTEBOOK_SIZE')
    dest.mkdir(parents=True);(dest/'notebook.ipynb').write_bytes(raw);(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (dest/'sample-private.json').write_text(json.dumps({'StudyInstanceUID':sample,'policy':config()['mechanics_sample_policy'],'sample_sha256':fingerprint(sample)},indent=2)+'\n')
    (dest/'population-private.json').write_text(json.dumps({'train':train,'development':dev,'groups':groups,'preflight':pre},indent=2)+'\n')
    receipt={'binding':binding,'sources':hashes,'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json','sample-private.json','population-private.json')},
             'notebook_bytes':len(raw),'status':'BUILT_UNLAUNCHED','development_studies':866,'mechanics_studies':32}
    (dest/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def verify_build(destination):
    dest=Path(destination);receipt=json.loads((dest/'build-receipt.json').read_bytes())
    for name,pin in receipt['files'].items():require(file_sha(dest/name)==pin,'IMMUTABLE_BUILD_BYTES')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source_path','reference_notebook','fold0_build','fold0_receipt','label_path','mask_path','data_root','destination','slug'):p.add_argument(name)
    r=build(**vars(p.parse_args()));print(json.dumps({'status':r['status'],'binding':r['binding'],'notebook_bytes':r['notebook_bytes']}))
