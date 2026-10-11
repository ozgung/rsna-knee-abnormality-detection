"""Build, never launch, a private CPU/offline immutable geometry qualification."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee.reference import audit_source,canonical_bytes,sha256,bounded_ids
from knee.reference_blend import CONVNEXT_DATASET,DOCKER_IMAGE
from knee.coverage_inference import CODECS
from knee.native_crop_qualification import FOLD_SHA,EXCLUDED_SHA,SEED,COUNT

SOURCES=('knee/__init__.py','knee/reference.py','knee/reference_blend.py','knee/mirror_input.py',
         'knee/raptor_input.py','knee/mirror_submission.py','knee/public_reader_contract.py',
         'knee/fresh_reader_setup.py','knee/runtime.py','knee/public_rich_cache.py','knee/native_crop_qualification.py')


def pinned_json(path,pin):
    raw=Path(path).read_bytes()
    if sha256(raw)!=pin: raise ValueError('Pinned private membership bytes changed')
    return json.loads(raw)


def build(source_path,destination,slug,folds_path,exclusion_path,sample_path,train_csv):
    root=Path(__file__).resolve().parents[1]; destination=Path(destination)
    if not destination.resolve().is_relative_to(root/'outputs'): raise ValueError('Private build must be under ignored outputs')
    if destination.exists(): raise ValueError('Fresh immutable build required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug): raise ValueError('Invalid slug')
    folds=pinned_json(folds_path,FOLD_SHA)['folds']; exclusions=pinned_json(exclusion_path,EXCLUDED_SHA)
    sample=json.loads(Path(sample_path).read_text())
    ids=sample.get('StudyInstanceUID',sample.get('study_ids',sample.get('selected_study_ids',sample.get('ids'))))
    if not isinstance(ids,list) or len(ids)!=COUNT: raise ValueError('Exactly32 frozen study IDs required')
    if ids!=bounded_ids(ids,count=COUNT,seed=SEED): raise ValueError('Frozen sample hash order differs')
    if any(folds.get(s)==0 or s not in folds or s in exclusions['quarantined_study_ids'] for s in ids): raise ValueError('Sample is not training-only')
    train_sha=sha256(Path(train_csv).read_bytes())
    selected_sha=sha256(canonical_bytes(ids))
    if selected_sha!='1e4cf23b048d1ced8c960ee8035abbfe1f43aba4992feb6c4d42d4949748c414': raise ValueError('Parent-frozen sample differs')
    if train_sha!='8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33': raise ValueError('Pinned official CSV differs')
    if sample.get('selected_id_sha256',selected_sha)!=selected_sha: raise ValueError('Sample receipt digest differs')
    if sample.get('official_train_sha256',train_sha)!=train_sha: raise ValueError('Sample official source differs')
    public,decoder,manifest=audit_source(Path(source_path).read_text(),profile='mirror')
    sources={name:(root/name).read_text() for name in SOURCES}
    hashes={name:sha256(content.encode()) for name,content in sources.items()}
    binding={'fold_file_sha256':FOLD_SHA,'exclusion_file_sha256':EXCLUDED_SHA,'fold_mapping_sha256':sha256(canonical_bytes(folds)),
             'selected_order_sha256':sha256(canonical_bytes(ids)),'train_csv_sha256':sha256(Path(train_csv).read_bytes()),
             'source_bundle_sha256':sha256(canonical_bytes(sources)),
             'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
             'working_sources':'exact bundled bytes; may include uncommitted implementation'}
    entry=f'PUBLIC={public!r}\nDECODER={decoder!r}\nMANIFEST={manifest!r}\nEXCLUSIONS={exclusions!r}\nFOLDS={folds!r}\nBINDING={binding!r}\nEXPECTED_CODECS={CODECS!r}\n'+'''
from knee.fresh_reader_setup import dataset_root
from knee.reference_blend import CONVNEXT_DATASET,select_codec_wheels
from knee.runtime import competition_root
from knee.native_crop_qualification import run
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
if sys.version_info[:2]!=(3,12): raise ValueError('Pinned Python version changed')
codec_root=dataset_root('/kaggle/input',CONVNEXT_DATASET,['pylibjpeg-2.1.0-py3-none-any.whl'])
wheels=select_codec_wheels(codec_root)
if {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels}!=EXPECTED_CODECS: raise ValueError('Codec bytes changed')
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)],check=True)
report=run(PUBLIC,DECODER,MANIFEST,competition_root(),'/kaggle/working/native-crop-qualification',EXCLUSIONS,FOLDS,BINDING)
print(json.dumps({'status':report['status'],'completed_studies':report['completed_studies'],'anatomical_acceptance':'UNVERIFIED'}),flush=True)
'''
    code='import os,sys,json,hashlib,subprocess\nfrom pathlib import Path\n'+f'sources={sources!r}\nexpected_hashes={hashes!r}\nexpected_bundle={binding["source_bundle_sha256"]!r}\n'+'''
if {n:hashlib.sha256(v.encode()).hexdigest() for n,v in sources.items()}!=expected_hashes: raise ValueError('Bundle source tampered')
actual=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
if actual!=expected_bundle: raise ValueError('Bundle digest tampered')
project=Path('/kaggle/working/project')
for name,content in sources.items():
    path=project/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(content)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hashes[name]: raise ValueError('Runtime file changed')
sys.path.insert(0,str(project))
'''+entry
    compile(code,'<qualification-notebook>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','metadata':{},'id':'scope','source':['# Centered native90 CPU qualification\n','32 frozen fold0-training nongold/nonquarantine studies. No model, labels, training, network or prediction claim. Geometric center prior; anatomical review remains pending. Private outputs only.\n',f'Public source: {manifest["source_url"]}; {manifest["attribution"]}.\n']},
        {'cell_type':'code','metadata':{},'id':'run','execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    meta={'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
          'is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
          'dataset_sources':[CONVNEXT_DATASET],'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE}
    if len(json.dumps(nb,indent=2).encode()) > 900_000:
        raise ValueError('Notebook exceeds conservative source limit')
    destination.mkdir(parents=True)
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta)]: (destination/name).write_text(json.dumps(value,indent=2)+'\n')
    receipt={'binding':binding,'source_manifest':manifest,'sources':hashes,'files':{n:sha256((destination/n).read_bytes()) for n in ('notebook.ipynb','kernel-metadata.json')}}
    (destination/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def verify_build(destination):
    destination=Path(destination); receipt=json.loads((destination/'build-receipt.json').read_text())
    for name,pin in receipt['files'].items():
        if sha256((destination/name).read_bytes())!=pin: raise ValueError('Immutable build file changed')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source_path','destination','slug','folds_path','exclusion_path','sample_path','train_csv'): p.add_argument(name)
    args=p.parse_args(); receipt=build(**vars(args)); print(json.dumps(receipt,indent=2))
