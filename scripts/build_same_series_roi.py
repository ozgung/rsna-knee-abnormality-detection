"""Build, never launch, frozen32 image-only same-series sagittal ROI reader qualification."""
import argparse
import base64
import json
from pathlib import Path
import re
import subprocess
import sys
import zlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.build_public_s6_qualification import collect_sources,published_pins,MANIFEST_SHA,CODECS,mounted_root
from knee.public_s6_qualification import checked_json,FOLD_SHA,EXCLUDED_SHA,TRAIN_SHA,SAMPLE_SHA,canonical,digest,file_sha,require,save,validate_membership,schema_header
from knee.native_joint_roi_pilot import WEIGHT_SHA,WEIGHT_BYTES,WEIGHT_URL
from knee.same_series_roi import PROTOCOL


def build(embedded,closure,closure_receipt,contract_path,manifest_path,sample_path,folds_path,exclusion_path,data_root,destination,slug):
    root=Path(__file__).resolve().parents[1];dest=Path(destination);data=Path(data_root)
    require(dest.resolve().is_relative_to(root/'outputs'),'Private build must be under worktree outputs')
    require(not dest.exists(),'Fresh immutable destination required')
    require(slug != 'native-joint-roi-qualification-20261013', 'Distinct same-series kernel slug required')
    require(len(slug)<50 and bool(re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug)),'Invalid slug (must be <50 chars)')
    folds=checked_json(folds_path,FOLD_SHA)['folds'];excluded=checked_json(exclusion_path,EXCLUDED_SHA)
    sample=json.loads(Path(sample_path).read_bytes());ids=next((sample[k] for k in ('StudyInstanceUID','study_ids','selected_study_ids','ids') if k in sample),None)
    validate_membership(ids,folds,excluded)
    # Only the frozen sample's fold values are exposed to the image reader;
    # original full membership bytes remain bound above and in the receipt.
    selected_folds={uid:folds[uid] for uid in ids}
    selected_excluded={'quarantined_study_ids':[]}
    require(file_sha(data/'train.csv')==TRAIN_SHA,'Official train source changed');schema_header(data/'sample_submission.csv')
    contract=json.loads(Path(contract_path).read_bytes());manifest=checked_json(manifest_path,MANIFEST_SHA)
    require(contract['manifest']['sha256']==MANIFEST_SHA and manifest['anchor_weights_sha256']==WEIGHT_SHA,'Reviewed anchor binding differs')
    pins=published_pins(embedded,contract);sources=collect_sources(embedded,closure,closure_receipt)
    for name,pin in pins['SOURCE_SHA'].items():require(digest_text(sources[name])==pin,'Embedded source differs')
    for name,pin in pins['CODE_SHA'].items():require(digest_text(sources['training/'+name])==pin,'Mainline source differs')
    source_pins={n:digest_text(v) for n,v in sources.items()}
    for name in ('knee/__init__.py','knee/public_s6_qualification.py','knee/native_joint_roi_pilot.py','knee/same_series_roi.py'):
        sources[name]=(root/name).read_text()
    binding={'source_bundle_sha256':digest(sources),'source_collection_receipt_sha256':file_sha(closure_receipt),
             'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
             'working_sources':'exact bundled bytes; may include uncommitted implementation',
             'series_sha256':file_sha(data/'train_series.csv'),'schema_sha256':file_sha(data/'sample_submission.csv'),
             'train_sha256':TRAIN_SHA,'fold_sha256':FOLD_SHA,'exclusion_sha256':EXCLUDED_SHA,'sample_sha256':SAMPLE_SHA,
             'selected_folds_sha256':digest(selected_folds),
             'anatomy_sha256':WEIGHT_SHA,'anatomy_bytes':WEIGHT_BYTES,'anatomy_upstream':WEIGHT_URL,
             'wrapper_exposure':'candidate/segment heuristics tuned on58 gold montages; gold-exposed; no retuning; record/slab prior bypassed',
             'protocol_id':PROTOCOL,'kernel_slug':slug,'host_policy':{'device':'cpu','internet':False,'private':True},'sources':{'competition':'rsna-knee-abnormality-detection','localizer':'pinned knee_anchor_bank_v2.candidate; no record'},
             'protocol':{'study_count':32,'side_mm':100,'size':384,'control_views':24,'treatment_global_views':12,'treatment_joint_views':12,
                         'selection':'frozen training-only nongold sample; no outcome values','classifier_models':0,'training':False,
                         'crop_policy':'own-series sagittal fixed source-rounded100mm rectangle; no clipping/shrinking; exact global fallback',
                         'allocation':'original12 all-series global context +12 eligible sagittal round-robin equally spaced centers; none eligible duplicates original12 global',
                         'candidate_policy':'seven slices/4mm; original thresholds; no cross-series prior/rejection/averaging/transfer',
                         'advancement':'reader parity plus complete source-visible containment acceptance before any fit'}}
    encoded=base64.b64encode(zlib.compress(canonical(sources),9)).decode()
    import inspect
    bootstrap=f'''import base64,hashlib,json,os,sys,zlib,subprocess,re
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',BUILD_MONAI='0')
sources=json.loads(zlib.decompress(base64.b64decode({encoded!r},validate=True)))
if hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!={binding['source_bundle_sha256']!r}:raise ValueError('Source bundle differs')
project=Path('/kaggle/working/same-series-roi-project');code=project/'source'
for name,content in sources.items():
    p=(project/name) if name.startswith('knee/') else (code/name)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode())
sys.path.insert(0,str(project))
from knee.public_s6_qualification import require,file_sha
from knee.same_series_roi import run
{inspect.getsource(mounted_root)}
discovery=[]
require(sys.version_info[:2]==(3,12),'Pinned Python codec ABI differs')
raw=mounted_root('/kaggle/input','rsna-knee-abnormality-detection',['train.csv','train_series.csv','sample_submission.csv'],discovery,competition=True)
require(file_sha(raw/'sample_submission.csv')=={binding['schema_sha256']!r},'Official schema source differs')
codec=mounted_root('/kaggle/input','goodpjw2008/rsna-knee-2-5d-convnext-reader',list({CODECS!r}),discovery)
wheels=[]
for name,pin in {CODECS!r}.items():
    require(file_sha(codec/name)==pin,'Codec bytes differ');wheels.append(str(codec/name))
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*wheels],check=True)
anatomy=mounted_root('/kaggle/input','heliosli/rsna-orthofoundation-fold0-epoch7-20261004',['anatomy.pt'],discovery)/'anatomy.pt'
config=dict(code=str(code),source_pins={source_pins!r},binding={binding!r},ids={ids!r},folds={selected_folds!r},excluded={selected_excluded!r},raw=str(raw),staged=str(project/'images-only'),anatomy=str(anatomy),out='/kaggle/working/same-series-roi-qualification',device='cpu')
report=run(config)
print(json.dumps(dict(status=report['status'],completed_studies=report['completed_studies'],advancement_allowed=False)))
'''
    compile(bootstrap,'<same-series-roi-bootstrap>','exec')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'cells':[
        {'cell_type':'markdown','metadata':{},'id':'protocol','source':['# Frozen32 same-series sagittal ROI reader qualification\n','CPU/offline/image-only; gold-exposed SKM-TEA candidate path; bypasses study-level filters; no classifier or fitting. Containment remains unverified.\n']},
        {'cell_type':'code','metadata':{},'id':'qualification','execution_count':None,'outputs':[],'source':bootstrap.splitlines(True)}]}
    metadata={'id':'ozgung/'+slug,'title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb','language':'python','kernel_type':'notebook',
              'is_private':True,'enable_gpu':False,'enable_tpu':False,'enable_internet':False,
              'docker_image':'gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461',
              'competition_sources':['rsna-knee-abnormality-detection'],
              'dataset_sources':['heliosli/rsna-orthofoundation-fold0-epoch7-20261004','goodpjw2008/rsna-knee-2-5d-convnext-reader'],
              'kernel_sources':[],'model_sources':[]}
    require(len(json.dumps(nb).encode())<900000,'Source notebook exceeds conservative900k')
    dest.mkdir(parents=True);save(dest/'notebook.ipynb',nb);save(dest/'kernel-metadata.json',metadata)
    receipt={'status':'BUILT_UNLAUNCHED','protocol':PROTOCOL,'binding':binding,'source_pins':source_pins,
             'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},
             'acceptance_pending':['real native/global parity','shared adapter train/inference parity on32','strict own-series candidate/geometry binding','all source-support triplet containment review','spacing/support/fallback/runtime measurements']}
    save(dest/'build-receipt.json',receipt);return receipt


def digest_text(text):return __import__('hashlib').sha256(text.encode()).hexdigest()


def verify_build(destination):
    dest=Path(destination);receipt=json.loads((dest/'build-receipt.json').read_bytes())
    for name,pin in receipt['files'].items():require(file_sha(dest/name)==pin,'Immutable build differs')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('embedded','closure','closure_receipt','contract_path','manifest_path','sample_path','folds_path','exclusion_path','data_root','destination','slug'):p.add_argument(name)
    result=build(**vars(p.parse_args()));print(json.dumps({'status':result['status'],'binding':result['binding']}))
