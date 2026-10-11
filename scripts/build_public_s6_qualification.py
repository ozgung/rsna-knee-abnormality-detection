"""Build one immutable private/offline frozen32 S6 T4 pilot; never launch it."""
import argparse
import ast
import base64
import csv
import json
import inspect
from pathlib import Path
import re
import subprocess
import sys
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee.public_s6_qualification import (SAMPLE_SHA,TRAIN_SHA,FOLD_SHA,EXCLUDED_SHA,RECIPE,LABELS,
    checked_json,canonical,digest,file_sha,require,save,validate_membership,schema_header)

NOTEBOOK_SHA='d358d04538df4abfbdc5a69b48bd3e3491d5cd93de24d2be825f3e05e61d0af1'
MANIFEST_SHA='1c1211e64b73f1f6433ade804816572db1a2aa5422ed9c57365184bc6a9517e6'

CODECS={
 'pylibjpeg-2.1.0-py3-none-any.whl':'25df9496a69e64e98c887fddee12a1271e275b5f74ba804f9bf98a08bb80993e',
 'pylibjpeg_libjpeg-2.4.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl':'01d950ef496476a9223e4966376cb88098fcf5c55a12a21f722d7b5f84daae43',
 'pylibjpeg_openjpeg-2.6.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl':'aa6b08327be6f6daf0814a61fd7410f53a08e8e0b35774c634d9907402ff4f1c'}
DATASETS=['heliosli/rsna-knee-compact-v1','heliosli/rsna-orthofoundation-fold0-epoch7-20261004','goodpjw2008/rsna-knee-2-5d-convnext-reader']


def mounted_root(inputs, reference, required, discovery, *, competition=False):
    """Probe only two exact attachment roots; never enumerate their children."""
    parts=reference.split('/')
    require(len(parts)==(1 if competition else 2) and all(re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',p) for p in parts),'Invalid attachment reference')
    require(required and all(isinstance(n,str) and n and Path(n).name==n and n not in ('.','..') for n in required),'Invalid asset names')
    inputs=Path(inputs)
    candidates=([inputs/'competitions'/parts[0],inputs/parts[0]] if competition else
                [inputs/'datasets'/parts[0]/parts[1],inputs/parts[1]])
    probes=[{'root':str(root),'present':[n for n in required if (root/n).is_file()]} for root in candidates]
    discovery.append({'reference':reference,'required':list(required),'probes':probes})
    matches=[root for root,probe in zip(candidates,probes) if probe['present']]
    # Partial duplicate mounts also fail closed, rather than silently picking a complete one.
    require(len(matches)==1,'Expected exactly one explicit attachment mount: '+reference)
    root=matches[0]
    require(all((root/n).is_file() for n in required),'Required attachment files missing: '+reference)
    return root


def record_setup_failure(out, error, phase, discovery, binding):
    """Keep original detail privately; stdout exposes only fixed phase/type/counts."""
    import traceback
    detail=out/'setup-error-private.json'
    save(detail,dict(failure_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),phase=phase,discovery=discovery))
    if not (out/'receipt.json').exists():
        save(out/'receipt.json',dict(status='FAIL',failure_stage='offline_setup',failure_type=type(error).__name__,setup_phase=phase,
             diagnostic_file=detail.name,discovery_attempts=len(discovery),binding=binding,advancement_allowed=False))
    print(json.dumps(dict(status='FAIL',failure_stage='offline_setup',setup_phase=phase,failure_type=type(error).__name__,
                         discovery_attempts=len(discovery),diagnostic_file=detail.name,advancement_allowed=False)),flush=True)


def setup_handler_source():
    return inspect.getsource(mounted_root)+chr(10)+inspect.getsource(record_setup_failure)


def published_pins(embedded,contract):
    notebook=Path(embedded)/'rsna-knee-v2-velciraptor-dinosaur-speed.ipynb'
    require(file_sha(notebook)==NOTEBOOK_SHA,'Accepted Matterhorn notebook differs')
    found={}
    for cell in json.loads(notebook.read_bytes())['cells']:
        if cell['cell_type']!='code':continue
        source=''.join(cell['source'])
        source='\n'.join(line for line in source.splitlines() if not line.startswith(('%%','%','!')))
        try:tree=ast.parse(source)
        except SyntaxError:continue
        for node in ast.walk(tree):
            if not isinstance(node,ast.Assign):continue
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('SOURCE_SHA','CODE_SHA'):
                    require(target.id not in found,'Ambiguous published source pin table')
                    found[target.id]=ast.literal_eval(node.value)
    require(set(found)=={'SOURCE_SHA','CODE_SHA'},'Published source pins absent')
    require(all(found[name]==contract['constants'][name] for name in found),'Audit contract differs from immutable published pins')
    return found


def collect_sources(embedded,closure,closure_receipt):
    """Require collection completion before creating operational pins."""
    closure=Path(closure);embedded=Path(embedded)
    receipt=json.loads(Path(closure_receipt).read_bytes())
    require(receipt['status'] in ('complete','completed','pass','PASS'), 'Source collection incomplete')
    entries=receipt['entries']
    vendor={n:e for n,e in entries.items() if n.startswith('vendor_runtime/monai060/') and n.endswith('.py')}
    require(len(vendor)==194 and all(e['status']=='pass' for e in vendor.values()), 'Complete194-source vendor receipts required')
    inventory=json.loads((Path(closure_receipt).parent/'source-manifest.json').read_bytes())
    require(inventory['canonical_inventory_sha256']=='aa8373915909a5e62c25ae9d6fa9dfb5a04b9baa47cfb9e0dc37460c55ae0801' and digest(inventory['files'])==inventory['canonical_inventory_sha256'],'Accepted complete source inventory differs')
    require(inventory['count']==212,'Complete212 Python source files required')
    operational={e['path']:e for e in inventory['files']}
    sources={}
    for p in closure.rglob('*.py'):
        name=p.relative_to(closure).as_posix();raw=p.read_bytes()
        require(name in operational and file_sha(p)==operational[name]['sha256'] and len(raw)==operational[name]['bytes'],'Audited source operational pin differs')
        if name.startswith('vendor_runtime/'):
            require(name in vendor and file_sha(p)==vendor[name]['sha256'] and len(raw)==vendor[name]['expected_bytes'], 'Vendor bytes differ from collection')
        require(len(raw)!=64 or not re.fullmatch(rb'[a-f0-9]{64}',raw), 'Hash placeholder prohibited')
        compile(raw,str(p),'exec');sources['training/'+name]=raw.decode()
    require(len(sources)==212,'Source closure membership differs')
    require(len([n for n in sources if n.startswith('training/vendor_runtime/')])==194, 'Vendor file membership differs')
    for name in ('px_cache','compact_train','compact_infer','probe_infer','probe_dual','knee_anchor_bank_v2','side_common'):
        p=embedded/('embedded-'+name+'.py');sources[name+'.py']=p.read_bytes().decode('utf-8')
    require('training/geometry_bank.py' in sources and 'training/anatomy.py' in sources and 'training/anatomy_runtime.py' in sources,'Incomplete executed mainline closure')
    return sources


def build(embedded,closure,closure_receipt,contract_path,manifest_path,sample_path,folds_path,exclusion_path,data_root,destination,slug):
    root=Path(__file__).resolve().parents[1];dest=Path(destination);data=Path(data_root).resolve()
    require(not dest.exists(),'Fresh immutable destination required')
    require(dest.resolve().is_relative_to(data.parent.parent/'outputs') or dest.resolve().is_relative_to(root/'outputs'),'Private build must be ignored outputs')
    require(re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug),'Invalid slug')
    folds=checked_json(folds_path,FOLD_SHA)['folds'];excluded=checked_json(exclusion_path,EXCLUDED_SHA)
    sample=json.loads(Path(sample_path).read_bytes());ids=sample.get('StudyInstanceUID',sample.get('study_ids',sample.get('selected_study_ids',sample.get('ids'))))
    validate_membership(ids,folds,excluded)
    require(file_sha(data/'train.csv')==TRAIN_SHA,'Official train source changed')
    schema_header(data/'sample_submission.csv')
    contract=json.loads(Path(contract_path).read_bytes());manifest=json.loads(Path(manifest_path).read_bytes())
    require(file_sha(manifest_path)==MANIFEST_SHA==contract['manifest']['sha256'],'Reviewed manifest differs')
    pins=published_pins(embedded,contract)
    sources=collect_sources(embedded,closure,closure_receipt)
    for name,pin in pins['SOURCE_SHA'].items():
        require(__import__('hashlib').sha256(sources[name].encode()).hexdigest()==pin,'Embedded published source pin differs')
    for name,pin in pins['CODE_SHA'].items():
        require(__import__('hashlib').sha256(sources['training/'+name].encode()).hexdigest()==pin,'Mainline published source pin differs')
    hashes={n:__import__('hashlib').sha256(v.encode()).hexdigest() for n,v in sources.items()}
    sources['knee/__init__.py']=''
    sources['knee/public_s6_qualification.py']=(root/'knee/public_s6_qualification.py').read_text()
    # Source-only bundle excludes any reports, target tables, model tensors or historical endpoint.
    encoded=base64.b64encode(zlib.compress(canonical(sources),9)).decode()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    binding={'manifest_sha256':file_sha(manifest_path),'schema_sha256':file_sha(data/'sample_submission.csv'),'series_sha256':file_sha(data/'train_series.csv'),
             'frozen_order_sha256':SAMPLE_SHA,'train_sha256':TRAIN_SHA,'fold_sha256':FOLD_SHA,'exclusion_sha256':EXCLUDED_SHA,
             'source_bundle_sha256':digest(sources),'source_collection_receipt_sha256':file_sha(closure_receipt),'code_revision':revision,'working_sources':'exact bundled implementation; may be uncommitted',
             'target_mapping':{name:name for name in LABELS},'recipe':RECIPE}
    bootstrap=f'''import base64,hashlib,json,os,sys,zlib,subprocess,socket,re
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',BUILD_MONAI='0')
encoded={encoded!r}
sources=json.loads(zlib.decompress(base64.b64decode(encoded,validate=True)))
require_digest={binding['source_bundle_sha256']!r}
if hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=require_digest:raise ValueError('Bundle differs')
project=Path('/kaggle/working/s6-project');code=project/'source'
for name,content in sources.items():
    p=(project/name) if name.startswith('knee/') else (code/name)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode())
sys.path.insert(0,str(project))
from knee.public_s6_qualification import file_sha,save,run,require,schema_header
pins={hashes!r}
manifest={manifest!r}
binding={binding!r}
ids={ids!r}
codecs={CODECS!r}
out=Path('/kaggle/working/s6-qualification')
{setup_handler_source()}
discovery=[]
phase='python_abi'
def setup_phase(value):
    global phase
    phase=value
    save(out/'setup-discovery-private.json',dict(phase=phase,discovery=discovery))
    print(json.dumps(dict(event='s6_setup_phase',phase=phase,discovery_attempts=len(discovery))),flush=True)
try:
    setup_phase('python_abi')
    require(sys.version_info[:2]==(3,12),'Python codec ABI differs')
    setup_phase('competition_discovery')
    raw=mounted_root('/kaggle/input','rsna-knee-abnormality-detection',['sample_submission.csv','train.csv','train_series.csv'],discovery,competition=True)
    setup_phase('schema_hash')
    require(file_sha(raw/'sample_submission.csv')==binding['schema_sha256'],'Official schema bytes differ');schema_header(raw/'sample_submission.csv')
    setup_phase('codec_discovery')
    codec_root=mounted_root('/kaggle/input',{DATASETS[2]!r},list(codecs),discovery)
    setup_phase('codec_hash')
    wheels=[]
    for name,pin in codecs.items():
        p=codec_root/name;require(file_sha(p)==pin,'Offline codec bytes differ');wheels.append(str(p))
    setup_phase('codec_install')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*wheels],check=True)
    setup_phase('compact_discovery')
    compact=mounted_root('/kaggle/input',{DATASETS[0]!r},['manifest.json',*manifest['sha256']],discovery)
    setup_phase('anatomy_discovery')
    anatomy=mounted_root('/kaggle/input',{DATASETS[1]!r},['anatomy.pt'],discovery)/'anatomy.pt'
    setup_phase('configuration')
    config=project/'config-private.json'
    save(config,dict(code=str(code),source_pins=pins,manifest=manifest,binding=binding,ids=ids,raw=str(raw),staged=str(project/'images-only'),compact=str(compact),anatomy=str(anatomy),out=str(out)))
    setup_phase('qualification_run')
    report=run(config)
    print(json.dumps(dict(status=report['status'],completed_studies=report['completed_studies'],anatomy=report['anatomy'],advancement_allowed=False)))
except BaseException as e:
    record_setup_failure(out,e,phase,discovery,binding)
    raise RuntimeError('S6 qualification failed; preserve receipt and outputs') from None
'''
    compile(bootstrap,'<s6-bootstrap>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','id':'protocol','metadata':{},'source':['# Frozen32 image-only S6 qualification\n','Released source bytes; v4/384→352/3.5mm/all4-fold logit mean/chunk16/workers0. No training or labels. Anatomical review required before advancement.\n']},
        {'cell_type':'code','id':'qualification','metadata':{},'execution_count':None,'outputs':[],'source':bootstrap.splitlines(True)}]}
    meta={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook','is_private':True,
          'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461','enable_gpu':True,'enable_tpu':False,'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],'dataset_sources':DATASETS,'kernel_sources':[],'model_sources':[]}
    rawnb=json.dumps(nb,indent=2).encode()+b'\n';require(len(rawnb)<900000,'Source notebook exceeds900k')
    dest.mkdir(parents=True);(dest/'notebook.ipynb').write_bytes(rawnb);save(dest/'kernel-metadata.json',meta)
    receipt={'binding':binding,'source_pins':hashes,'bundle_compressed_base64_bytes':len(encoded),'notebook_bytes':len(rawnb),
             'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},'status':'BUILT_UNLAUNCHED','unknowns':['actual CUDA/model execution','actual anatomical coverage']}
    save(dest/'build-receipt.json',receipt);return receipt


def verify_build(destination):
    dest=Path(destination);r=json.loads((dest/'build-receipt.json').read_bytes())
    for n,pin in r['files'].items():require(file_sha(dest/n)==pin,'Immutable build bytes changed')
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('embedded','closure','closure_receipt','contract_path','manifest_path','sample_path','folds_path','exclusion_path','data_root','destination','slug'):p.add_argument(name)
    r=build(**vars(p.parse_args()));print(json.dumps({'status':r['status'],'notebook_bytes':r['notebook_bytes'],'binding':r['binding']}))
