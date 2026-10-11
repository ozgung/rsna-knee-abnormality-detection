"""Freeze the single accepted870 S6 rank blend; prepare only, never launch."""
import argparse
import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path
import re
import subprocess
import sys
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee.public_s6_evaluation import (PILOT_NOTEBOOK_SHA,PILOT_RECEIPT_SHA,COVERAGE_SHA,
    CONTROL_RECEIPT_SHA,CONTROL_PREDICTIONS_SHA,CONTROL_POPULATION_SHA,config,qualification_gate,
    validate_population,array_sha)
from knee.public_s6_qualification import canonical,digest,file_sha,require
from knee.public_adaptation_pilot import LABEL_SHA,MASK_SHA
from knee.public_adaptation_fit import FOLD_SHA,EXCLUDED_SHA
from knee.reference import WEIGHT_DATASET
from knee.reference_blend import DOCKER_IMAGE,CONVNEXT_DATASET
from scripts.build_public_fullfield import notebook_literals,SOURCES,LABEL_DATASET

FULLFIELD_NOTEBOOK_SHA='553760d78cc45dae5a240bfdd819f126a976da8e34efb69163381e3d297fda43'
DATASETS=['heliosli/rsna-knee-compact-v1','heliosli/rsna-orthofoundation-fold0-epoch7-20261004',CONVNEXT_DATASET,WEIGHT_DATASET,LABEL_DATASET]


def pinned(path,pin):
    require(file_sha(path)==pin,'REVIEWED_ARTIFACT_BYTES');return Path(path).read_bytes()


def literals(code,names):
    result={}
    for n in ast.parse(code).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names:
            result[n.targets[0].id]=ast.literal_eval(n.value)
    require(set(result)==set(names),'REVIEWED_LITERAL_INVENTORY');return result


def notebook_code(path,pin):
    nb=json.loads(pinned(path,pin));return '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')


def accepted_pilot(path):
    code=notebook_code(path,PILOT_NOTEBOOK_SHA)
    values=literals(code,['encoded','require_digest','pins','manifest','binding','ids','codecs'])
    sources=json.loads(zlib.decompress(base64.b64decode(values.pop('encoded'),validate=True)))
    require(digest(sources)==values['require_digest']==values['binding']['source_bundle_sha256'],'QUALIFIED_S6_SOURCE_BUNDLE')
    require(all(hashlib.sha256(sources[n].encode()).hexdigest()==p for n,p in values['pins'].items()),'QUALIFIED_S6_SOURCE_HASHES')
    functions={n.name:ast.get_source_segment(code,n) for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
    return sources,values,functions


def build(pilot_build,pilot_result,coverage_path,control_result,reference_notebook,data_root,destination,slug):
    import numpy as np
    dest=Path(destination);root=Path(__file__).resolve().parents[1];data=Path(data_root)
    require(not dest.exists() and dest.resolve().is_relative_to(root/'outputs'),'FRESH_IGNORED_PRIVATE_BUILD')
    require(re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug) is not None,'PRIVATE_SLUG')
    sources,pilot,functions=accepted_pilot(Path(pilot_build)/'notebook.ipynb')
    result=Path(pilot_result);pilot_receipt=json.loads(pinned(result/'receipt.json',PILOT_RECEIPT_SHA));coverage=json.loads(pinned(coverage_path,COVERAGE_SHA))
    qualification_gate(pilot_receipt,coverage);require(pilot_receipt['binding']==pilot['binding'],'QUALIFIED_PILOT_BINDING')
    wrapper=json.loads((result/'wrapper/receipt.json').read_bytes())
    require(wrapper['status']=='PASS' and wrapper['completed']==32 and wrapper['state_before']==wrapper['state_after'],'QUALIFIED_PILOT_STATE')
    ordinals=pilot_receipt['sentinel_ordinals'];require(ordinals==[0,1,20,9],'FROZEN_PILOT_SENTINELS')
    with np.load(result/'wrapper/predictions-private.npz',allow_pickle=False) as ar:
        require(ar['ordinals'].tolist()==list(range(32)),'QUALIFIED_PILOT_ARRAY_ORDER')
        pp=ar['probabilities'].copy()
    pilot_ids=[pilot['ids'][i] for i in ordinals]
    expected=[{'input':wrapper['records'][i]['input'],'probabilities':pp[i].tolist()} for i in ordinals]
    old=notebook_literals(reference_notebook);pre=old['PREFLIGHT'];old_binding=old['BINDING'];control_root=Path(control_result)
    control_receipt=json.loads(pinned(control_root/'receipt.json',CONTROL_RECEIPT_SHA))
    population=json.loads(pinned(control_root/'population-private.json',CONTROL_POPULATION_SHA))
    require(control_receipt['status']=='PASS' and control_receipt['evaluated_studies']==870 and control_receipt['optimizer_updates']==0 and control_receipt['final_state_and_all_buffers_immutable'],'QUALIFIED_OLD320_CONTROL')
    require(control_receipt['binding']['preflight_sha256']==digest(pre) and control_receipt['binding']['public_manifest_sha256']==digest(old['MANIFEST']),'OLD320_REFERENCE_BINDING')
    pinned(control_root/'predictions-private.npz',CONTROL_PREDICTIONS_SHA)
    with np.load(control_root/'predictions-private.npz',allow_pickle=False) as ar:
        control=ar['control'].copy();dev=ar['study_ids'].tolist()
    require(control.dtype==np.float32 and control.shape==(870,12) and np.isfinite(control).all(),'ORIGINAL_CONTROL_ARRAY_SCHEMA')
    require(hashlib.sha256(control.tobytes()).hexdigest()==control_receipt['binding']['saved_control_array_sha256'],'ORIGINAL_ADAPTATION_CONTROL_BYTES')
    require(dev==population['development_study_ids'],'ORIGINAL_CONTROL_POPULATION')
    validate_population(dev,old['FOLDS'],old['EXCLUDED'],population['groups'],pre)
    require(all(old['FOLDS'][s]!=0 and s not in old['EXCLUDED'] and s not in dev for s in pilot_ids),'TRAIN_ONLY_PILOT_SENTINELS')
    require(old_binding['fold_file_sha256']==FOLD_SHA and old_binding['excluded_file_sha256']==EXCLUDED_SHA,'ORIGINAL_MEMBERSHIP_FILES')
    for relative,pin in pre['source_sha256'].items():
        if relative.startswith('data/'):require(file_sha(data/relative.removeprefix('data/'))==pin,'PRIVATE_PREFLIGHT_INPUT_BYTES')
    require(file_sha(data/'raw/train.csv')==old_binding['official_train_sha256']==pilot['binding']['train_sha256'],'OFFICIAL_TRAIN_BYTES')
    require(file_sha(data/'raw/train_series.csv')==pilot['binding']['series_sha256'] and file_sha(data/'raw/sample_submission.csv')==pilot['binding']['schema_sha256'],'OFFICIAL_IMAGE_SCHEMA_BYTES')
    # Reuse the exact already-qualified public helper closure, not mutable imports.
    old_code=notebook_code(root/'outputs/kaggle/public-fullfield-fold0-20261011-reviewed-v2/notebook.ipynb',FULLFIELD_NOTEBOOK_SHA)
    old_sources=literals(old_code,['sources','expected_hashes','expected_bundle'])
    require(digest(old_sources['sources'])==old_sources['expected_bundle'],'QUALIFIED_PUBLIC_SOURCE_BUNDLE')
    require({n:hashlib.sha256(s.encode()).hexdigest() for n,s in old_sources['sources'].items()}==old_sources['expected_hashes'],'QUALIFIED_PUBLIC_SOURCE_BYTES')
    for name in SOURCES:
        text=old_sources['sources'][name]
        if name in sources:
            require(name=='knee/__init__.py' or sources[name]==text,'COMMON_QUALIFIED_SOURCE_CONFLICT')
            # Pilot uses an empty package initializer; preserve it byte-for-byte.
            if name=='knee/__init__.py':continue
        sources[name]=text
    sources['knee/public_s6_evaluation.py']=(root/'knee/public_s6_evaluation.py').read_bytes().decode('utf8')
    bundle=digest(sources)
    binding={'config':config(),'source_bundle_sha256':bundle,'qualified_s6_bundle_sha256':pilot['require_digest'],
        'pilot_notebook_sha256':PILOT_NOTEBOOK_SHA,'pilot_receipt_sha256':PILOT_RECEIPT_SHA,'coverage_adjudication_sha256':COVERAGE_SHA,
        'pilot_wrapper_receipt_sha256':file_sha(result/'wrapper/receipt.json'),'pilot_predictions_sha256':file_sha(result/'wrapper/predictions-private.npz'),
        'control_receipt_sha256':CONTROL_RECEIPT_SHA,'control_predictions_sha256':CONTROL_PREDICTIONS_SHA,'control_population_sha256':CONTROL_POPULATION_SHA,
        'public_source_bundle_sha256':old_sources['expected_bundle'],'public_manifest_sha256':digest(old['MANIFEST']),
        'preflight_sha256':digest(pre),'dev_order_sha256':pre['fingerprints']['dev'],'groups_sha256':digest(population['groups']),
        'fold_mapping_sha256':pre['fingerprints']['fold_mapping'],'fold_file_sha256':FOLD_SHA,'exclusion_file_sha256':EXCLUDED_SHA,
        'official_train_sha256':old_binding['official_train_sha256'],'series_sha256':pilot['binding']['series_sha256'],'schema_sha256':pilot['binding']['schema_sha256'],
        'label_sha256':LABEL_SHA,'truth_sha256':MASK_SHA,'saved_control_array_sha256':array_sha(control),
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_state':'exact bundled implementation; may be uncommitted'}
    cfg={'binding':binding,'binding_sha256':digest(binding),'source_pins':pilot['pins'],'manifest':pilot['manifest'],'pilot_binding':pilot['binding'],
         'pilot_receipt':pilot_receipt,'coverage':coverage,'pilot_sentinels':pilot_ids,'pilot_expected':expected,'pilot_state_sha256':wrapper['state_before'],
         'dev':dev,'folds':old['FOLDS'],'excluded':old['EXCLUDED'],'groups':population['groups'],'preflight':pre,
         'public_source':old['PUBLIC_SOURCE'],'decoder_source':old['DECODER_SOURCE'],'public_manifest':old['MANIFEST'],
         'control_state_sha256':control_receipt['state_grid_gate']['state_sha256'],'control_grid_sha256':control_receipt['state_grid_gate']['constructor_oracle_sha256']['320']}
    payload={'sources':sources,'cfg':cfg,'control':base64.b64encode(control.tobytes()).decode(),'codecs':pilot['codecs']}
    encoded=base64.b64encode(lzma.compress(canonical(payload),preset=9)).decode();pin=digest(payload)
    setup=functions['mounted_root']+'\n\n'+functions['record_setup_failure']
    boot="import base64,hashlib,json,os,sys,lzma,subprocess,re\nfrom pathlib import Path\n"+"os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',BUILD_MONAI='0')\n"+f'encoded={encoded!r}\nexpected_payload={pin!r}\n'+'''
payload=json.loads(lzma.decompress(base64.b64decode(encoded,validate=True)))
if hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=expected_payload:raise ValueError('Frozen evaluator payload changed')
project=Path('/kaggle/working/s6-evaluation-project');code=project/'source'
for name,content in payload['sources'].items():
    p=project/name if name.startswith('knee/') else code/name
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode())
    if hashlib.sha256(p.read_bytes()).hexdigest()!=hashlib.sha256(content.encode()).hexdigest():raise ValueError('Source staging changed bytes')
sys.path.insert(0,str(project))
from knee.public_s6_qualification import file_sha,save,require,schema_header
from knee.public_s6_evaluation import run
cfg=payload['cfg'];binding=cfg['binding'];manifest=cfg['manifest'];codecs=payload['codecs']
out=Path('/kaggle/working/s6-evaluation')
'''+setup+'''
discovery=[];phase='python_abi'
try:
    require(sys.version_info[:2]==(3,12),'Python codec ABI differs')
    phase='competition_discovery'
    raw=mounted_root('/kaggle/input','rsna-knee-abnormality-detection',['sample_submission.csv','train.csv','train_series.csv'],discovery,competition=True)
    require(file_sha(raw/'sample_submission.csv')==binding['schema_sha256'],'Official schema bytes differ');schema_header(raw/'sample_submission.csv')
    phase='codec_discovery'
    codec_root=mounted_root('/kaggle/input','goodpjw2008/rsna-knee-2-5d-convnext-reader',list(codecs),discovery)
    wheels=[]
    for name,pin in codecs.items():
        p=codec_root/name;require(file_sha(p)==pin,'Offline codec bytes differ');wheels.append(str(p))
    phase='codec_install';subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*wheels],check=True)
    phase='compact_discovery'
    compact=mounted_root('/kaggle/input','heliosli/rsna-knee-compact-v1',['manifest.json',*manifest['sha256']],discovery)
    anatomy=mounted_root('/kaggle/input','heliosli/rsna-orthofoundation-fold0-epoch7-20261004',['anatomy.pt'],discovery)/'anatomy.pt'
    weights=mounted_root('/kaggle/input','nartaa/rsna-knee-publication-swa-weights-20261007',[cfg['public_manifest']['checkpoint_file']],discovery)
    labels=mounted_root('/kaggle/input','nartaa/rsna-knee-hpo-assets',['labels_v1_blend.parquet','train_labels_v1.parquet'],discovery)
    phase='configuration'
    import numpy as np
    saved=project/'control-private.npy'
    with saved.open('wb') as h:np.save(h,np.frombuffer(base64.b64decode(payload['control'],validate=True),dtype=np.float32).reshape(870,12),allow_pickle=False)
    cfg.update(code=str(code),raw=str(raw),staged=str(project/'images-only'),compact=str(compact),anatomy=str(anatomy),out=str(out),
               saved_control=str(saved),control_checkpoint=str(weights/cfg['public_manifest']['checkpoint_file']),labels=str(labels/'labels_v1_blend.parquet'),masks=str(labels/'train_labels_v1.parquet'))
    path=project/'config-private.json';save(path,cfg)
    phase='fixed_evaluation';report=run(path)
    print(json.dumps(dict(status=report['status'],completed_studies=report['completed_studies'],macro_delta=report['macro_delta'],triage=report['triage'])),flush=True)
except BaseException as e:
    record_setup_failure(out,e,phase,discovery,binding)
    raise RuntimeError('S6 fixed evaluation failed; preserve outputs; no automatic resume') from None
'''
    compile(boot,'<s6-evaluation>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':['# Fixed870 S6 70:30 rank evaluation\n','Private offline image-only original central320 control and qualified S6; all870 rank universe; source six-decimal quantization. Frozen source/weights/folds/schema/truth; no optimizer or grid search. Separate rehearsal required after a positive no-harm result.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':boot.splitlines(True)}]}
    meta={'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,
          'enable_gpu':True,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
          'dataset_sources':DATASETS,'kernel_sources':[],'model_sources':[],'docker_image':DOCKER_IMAGE}
    encoded_nb=json.dumps(nb,indent=2)+'\n';require(len(encoded_nb.encode())<900000,f'NOTEBOOK_SOURCE_900K_CAP: notebook={len(encoded_nb.encode())} encoded={len(encoded)} source={sum(len(s.encode()) for s in sources.values())}')
    dest.mkdir(parents=True)
    for name,value in [('notebook.ipynb',nb),('kernel-metadata.json',meta),('population-private.json',{'development_study_ids':dev,'groups':population['groups'],'pilot_sentinels':pilot_ids})]:
        (dest/name).write_text(json.dumps(value,indent=2)+'\n')
    receipt={'binding':binding,'files':{n:file_sha(dest/n) for n in ['notebook.ipynb','kernel-metadata.json','population-private.json']},
             'notebook_bytes':len(encoded_nb.encode()),'compressed_payload_bytes':len(encoded),'source_files':len(sources),'source_bytes':sum(len(s.encode()) for s in sources.values()),
             'runtime_checks_exercised':False,'dataset_sources':DATASETS,'source_hashes':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}}
    (dest/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');return receipt


def verify_build(destination):
    dest=Path(destination);r=json.loads((dest/'build-receipt.json').read_bytes())
    require(all(file_sha(dest/n)==p for n,p in r['files'].items()),'IMMUTABLE_BUILD_BYTES');return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['pilot-build','pilot-result','coverage-path','control-result','reference-notebook','data-root','destination','slug']:p.add_argument('--'+name,required=True)
    a=p.parse_args();r=build(**vars(a));print(json.dumps({'notebook_bytes':r['notebook_bytes'],'runtime_checks_exercised':False}))
