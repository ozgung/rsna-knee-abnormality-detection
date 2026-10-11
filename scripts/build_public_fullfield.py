"""Prepare an immutable private offline full384 ablation; never launch a job."""
import argparse
import ast
import base64
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee.reference import audit_source,canonical_bytes,sha256,WEIGHT_DATASET
from knee.reference_blend import CONVNEXT_DATASET,DOCKER_IMAGE
from knee.coverage_inference import CODECS
from knee.public_adaptation_fit import FOLD_SHA,EXCLUDED_SHA
from knee.public_adaptation_pilot import LABEL_SHA,MASK_SHA,file_sha
from knee.public_fullfield import config,SAMPLE_SHA

REFERENCE_NOTEBOOK_SHA='a1f16d3880d74577f30fcafebb91c43095247b7ac841d3a05b65035f9f259e56'
RECOVERY_RECEIPT_SHA='0b153e3495ead9690d4eeb1c64e1743b9d01f81f92037ef82d80461b07c3fdbc'
RECOVERY_PREDICTIONS_SHA='167e50c5b6f3b0987696008f5c540e608520fd34b9867281fc793b874c1cef69'
SAMPLE_FILE_SHA='9efa4a3d548b3e408b482303e8047f957e69e31768124ad8947108d9feed92ec'
LABEL_DATASET='nartaa/rsna-knee-hpo-assets'
SOURCES=('knee/__init__.py','knee/reference.py','knee/reference_blend.py','knee/mirror_input.py',
    'knee/raptor_input.py','knee/public_reader_contract.py','knee/fresh_reader_setup.py','knee/runtime.py',
    'knee/public_adaptation_pilot.py','knee/public_study_stream.py','knee/validation.py','knee/public_adaptation_fit.py','knee/public_fullfield.py')


def pinned(path,pin):
    if file_sha(path)!=pin:raise ValueError('Parent-reviewed input bytes changed')
    return Path(path).read_bytes()


def notebook_literals(path):
    nb=json.loads(pinned(path,REFERENCE_NOTEBOOK_SHA))
    code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    values={}
    for n in ast.parse(code).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('PUBLIC_SOURCE','DECODER_SOURCE','MANIFEST','PREFLIGHT','BINDING','FOLDS','EXCLUDED'):
            values[n.targets[0].id]=ast.literal_eval(n.value)
    if set(values)!= {'PUBLIC_SOURCE','DECODER_SOURCE','MANIFEST','PREFLIGHT','BINDING','FOLDS','EXCLUDED'}:raise ValueError('Reviewed notebook literal inventory differs')
    return values


def build(source_path,reference_notebook,sample_path,recovery_receipt,recovery_predictions,
          data_root,destination,slug):
    import numpy as np
    dest=Path(destination);root=Path(__file__).resolve().parents[1];data=Path(data_root)
    if dest.exists():raise ValueError('Fresh immutable build required')
    # Shared root outputs are also authorized, but verify ignore status there.
    if 'outputs' not in dest.resolve().parts:raise ValueError('Private builds belong in ignored outputs')
    ignored=subprocess.run(['git','check-ignore','--quiet',str(dest)],cwd=root,capture_output=True)
    if ignored.returncode!=0 and not dest.resolve().is_relative_to(root/'outputs'):
        raise ValueError('Private build is not covered by repository outputs ignore')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug):raise ValueError('Invalid slug')
    original=notebook_literals(reference_notebook);pre=original['PREFLIGHT'];old_binding=original['BINDING']
    public,decoder,manifest=audit_source(Path(source_path).read_text(),profile='mirror')
    if (public,decoder,manifest)!=(original['PUBLIC_SOURCE'],original['DECODER_SOURCE'],original['MANIFEST']):raise ValueError('Original audited public contract differs')
    if old_binding['fold_file_sha256']!=FOLD_SHA or old_binding['excluded_file_sha256']!=EXCLUDED_SHA:raise ValueError('Original membership pins differ')
    for relative,pin in pre['source_sha256'].items():
        if relative.startswith('data/') and file_sha(data/relative.removeprefix('data/'))!=pin:raise ValueError('Actual input file binding differs')
    if file_sha(data/'raw/train.csv')!=old_binding['official_train_sha256']:raise ValueError('Official training table differs')
    sample=json.loads(pinned(sample_path,SAMPLE_FILE_SHA))['StudyInstanceUID']
    if len(sample)!=32 or sha256(canonical_bytes(sample))!=SAMPLE_SHA or any(original['FOLDS'][s]==0 or s in original['EXCLUDED'] for s in sample):raise ValueError('Frozen training sample differs')
    recovery=json.loads(pinned(recovery_receipt,RECOVERY_RECEIPT_SHA))
    if recovery['status']!='PASS' or recovery['evaluated_studies']!=870 or recovery['optimizer_updates_this_run']!=0 or recovery['binding']!=old_binding:raise ValueError('Completed recovery provenance differs')
    pinned(recovery_predictions,RECOVERY_PREDICTIONS_SHA)
    with np.load(recovery_predictions,allow_pickle=False) as archive:control=np.ascontiguousarray(archive['control'])
    if control.shape!=(870,12) or control.dtype!=np.float32 or not np.isfinite(control).all():raise ValueError('Saved control array schema differs')
    sources={name:(root/name).read_text() for name in SOURCES};digest=sha256(canonical_bytes(sources))
    binding={'config':config(),'source_bundle_sha256':digest,'public_manifest_sha256':sha256(canonical_bytes(manifest)),
        'preflight_sha256':sha256(canonical_bytes(pre)),'dev_order_sha256':pre['fingerprints']['dev'],
        'fold_mapping_sha256':pre['fingerprints']['fold_mapping'],'fold_file_sha256':FOLD_SHA,'exclusion_file_sha256':EXCLUDED_SHA,
        'official_train_sha256':old_binding['official_train_sha256'],'label_sha256':LABEL_SHA,'truth_sha256':MASK_SHA,
        'sample_sha256':SAMPLE_SHA,'sample_file_sha256':SAMPLE_FILE_SHA,'reference_notebook_sha256':REFERENCE_NOTEBOOK_SHA,
        'saved_control_receipt_sha256':RECOVERY_RECEIPT_SHA,'saved_control_npz_sha256':RECOVERY_PREDICTIONS_SHA,
        'saved_control_array_sha256':sha256(control.tobytes()),'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'source_state':'exact bundled working files; may be uncommitted'}
    array_b64=base64.b64encode(control.tobytes()).decode()
    entry=f'PUBLIC_SOURCE={public!r}\nDECODER_SOURCE={decoder!r}\nMANIFEST={manifest!r}\nPREFLIGHT={pre!r}\nFOLDS={original["FOLDS"]!r}\nEXCLUDED={original["EXCLUDED"]!r}\nSAMPLE={sample!r}\nBINDING={binding!r}\nCONTROL_B64={array_b64!r}\nEXPECTED_CODECS={CODECS!r}\n'+'''
import base64,importlib.metadata,subprocess,numpy as np
from knee.fresh_reader_setup import dataset_root
from knee.reference_blend import CONVNEXT_DATASET,select_codec_wheels
from knee.reference import WEIGHT_DATASET
from knee.runtime import competition_root
from knee.public_adaptation_pilot import file_sha
from knee.public_fullfield import run
out=Path('/kaggle/working/public-fullfield')
try:
    if sys.version_info[:2]!=(3,12) or importlib.metadata.version('torch')!='2.10.0+cu128' or importlib.metadata.version('timm')!='1.0.26':raise ValueError('Pinned CUDA environment differs')
    codec_root=dataset_root('/kaggle/input',CONVNEXT_DATASET,['pylibjpeg-2.1.0-py3-none-any.whl'])
    wheels=select_codec_wheels(codec_root)
    if {p.name:file_sha(p) for p in wheels}!=EXPECTED_CODECS:raise ValueError('Codec hashes differ')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)],check=True)
    weights=dataset_root('/kaggle/input',WEIGHT_DATASET,[MANIFEST['checkpoint_file']])
    labels=dataset_root('/kaggle/input','nartaa/rsna-knee-hpo-assets',['labels_v1_blend.parquet','train_labels_v1.parquet'])
    saved=np.frombuffer(base64.b64decode(CONTROL_B64,validate=True),dtype=np.float32).reshape(870,12)
    report=run(PUBLIC_SOURCE,DECODER_SOURCE,MANIFEST,weights/MANIFEST['checkpoint_file'],labels/'labels_v1_blend.parquet',labels/'train_labels_v1.parquet',competition_root(),out,FOLDS,EXCLUDED,PREFLIGHT,SAMPLE,BINDING,saved)
    print(json.dumps({'status':report['status'],'optimizer_updates':0,'evaluated_studies':report['evaluated_studies']}),flush=True)
except BaseException as error:
    if not (out/'receipt.json').exists():
        out.mkdir(parents=True,exist_ok=True)
        (out/'receipt.json').write_text(json.dumps({'status':'FAIL','failure_stage':'offline_setup','failure_type':type(error).__name__,'binding':BINDING},indent=2)+'\n')
    raise RuntimeError('Fullfield ablation failed; inspect sanitized receipt') from None
'''
    hashes={n:sha256(v.encode()) for n,v in sources.items()}
    code="import os,sys,json,hashlib\nfrom pathlib import Path\nos.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')\n"+f'sources={sources!r}\nexpected_hashes={hashes!r}\nexpected_bundle={digest!r}\n'+'''
if {n:hashlib.sha256(v.encode()).hexdigest() for n,v in sources.items()}!=expected_hashes:raise ValueError('Source bundle tampered')
if hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=expected_bundle:raise ValueError('Bundle digest differs')
project=Path('/kaggle/working/project')
for name,content in sources.items():
    p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(content)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected_hashes[name]:raise ValueError('Runtime source bytes changed')
sys.path.insert(0,str(project))
os.environ['KNEE_SOURCE_SHA256']=expected_bundle
'''+entry
    compile(code,'<public-fullfield>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':['# Frozen public full384 versus central320\n',f'{manifest["attribution"]}. {manifest["source_url"]}\n','Exactly32 frozen training-only mechanics probes, conditional870 paired report development studies. Original checkpoint, strict state; constructor-derived spatial grids differ. No optimizer, gold scoring, fit, resize search or cross-arm blend. Public exposure persists.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    meta={'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
        'is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
        'dataset_sources':[WEIGHT_DATASET,LABEL_DATASET,CONVNEXT_DATASET],'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE}
    encoded=json.dumps(nb,indent=2)+'\n'
    if len(encoded.encode())>900000:raise ValueError('Notebook source exceeds900k')
    dest.mkdir(parents=True)
    (dest/'notebook.ipynb').write_text(encoded);(dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    receipt={'binding':binding,'sources':hashes,'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},'notebook_bytes':len(encoded.encode())}
    (dest/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def verify_build(destination):
    dest=Path(destination);receipt=json.loads((dest/'build-receipt.json').read_text())
    for name,pin in receipt['files'].items():
        if file_sha(dest/name)!=pin:raise ValueError('Immutable build file changed')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source_path','reference_notebook','sample_path','recovery_receipt','recovery_predictions','data_root','destination','slug'):p.add_argument(name)
    receipt=build(**vars(p.parse_args()));print(json.dumps(receipt,indent=2))
