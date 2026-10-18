"""Build immutable A-prime rehearsal/entry from the exact archived R7 notebook.

No public top-level code is executed. Entry generation requires fresh saved
two-T4 rehearsal ledgers, complete baseline and R224 receipts, and bound CSV.
"""
import argparse
import ast
import base64
import csv
import hashlib
import json
import lzma
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from knee import public_ws2_checkpoint as C224
from knee import public_ws2_entry as WS2
from knee import public_ws2_recovery as REC
from knee import public_r7_recovery_entry as R7
from knee.public_s6_qualification import LABELS, UID, canonical, digest, file_sha, require, save
from knee.public_fullfield_entry import test_contract

BASE_NOTEBOOK_SHA={
    'rehearsal':'712ab44fead9309f56e152e16cfc38eece9a11ca0d4729ba76c98eedcd6961ec',
    'entry':'be3e6c06bc5a9e89a999c9e6cdb5f352ce5e9916e17ae7816fad243c4e5048b9'}
BASE_BUILD_SHA={
    'rehearsal':'d820395675b7fdef5e269b5dafbd9384126b34a935f85b7758c5191e794110c2',
    'entry':'98ae6afa6bfcfceeb6414a89509ae1f073f0a8141f4cd988f25ce14c86356fad'}
BASE_PASS_SHA='e10075cf8bf53d74b5a3750e31bd81d003e7714453d4ca55047d363d96d0cc07'
PUBLIC_REFERENCE_NOTEBOOK_SHA='85c694c903ec6841488b49d5ecbe2838b8fad7db283f1f952c59f48d936e0b24'
NEW_FILES=('knee/public_ws2_checkpoint.py','knee/public_ws2_recovery.py','knee/public_ws2_entry.py')
PRIVATE_CFG=('probes','probe_order','old_sentinels','s6_sentinels','development_windows',
             'heliosli_reference','heliosli_visible_atol')
PRIVATE_BINDING=('sentinel_inputs','sentinel_probabilities','visible_ids','probe_contract_sha256',
                 's6_probe_ids','old_historical_ids')


def archived_payload(notebook,build_receipt,mode):
    require(file_sha(notebook)==BASE_NOTEBOOK_SHA[mode] and file_sha(build_receipt)==BASE_BUILD_SHA[mode],
            'EXACT_ARCHIVED_BASE_BYTES')
    build=json.loads(Path(build_receipt).read_bytes())
    require(build['files']['notebook.ipynb']==BASE_NOTEBOOK_SHA[mode] and build['mode']==mode,'BASE_BUILD_RECEIPT')
    nb=json.loads(Path(notebook).read_bytes())
    require(len(nb['cells'])==1 and nb['cells'][0]['cell_type']=='code','BASE_NOTEBOOK_CELL')
    source=''.join(nb['cells'][0]['source'])
    tree=ast.parse(source)
    literals={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
        and n.targets[0].id in ('encoded','expected_payload')}
    require(set(literals)=={'encoded','expected_payload'},'BASE_PAYLOAD_LITERALS')
    pack=json.loads(lzma.decompress(base64.b64decode(literals['encoded'],validate=True)))
    require(digest(pack)==literals['expected_payload']==build['payload_sha256'],'BASE_PAYLOAD_HASH')
    require(pack['cfg']['mode']==mode,'BASE_MODE')
    if mode=='rehearsal':
        require(pack['cfg']['binding']==build['binding'],'BASE_BINDING')
    else:
        stripped={k:v for k,v in build['binding'].items() if k not in PRIVATE_BINDING}
        private_free=pack['cfg']['binding']
        require(all(private_free.get(k)==v for k,v in stripped.items())
                and set(private_free)-set(stripped)=={'rehearsal_receipt_sha256',
                    'rehearsal_csv_sha256','qualified_r7_state'}
                and not set(private_free)&set(PRIVATE_BINDING),
                'PURE_ENTRY_BASE_BINDING')
    for name,sha in build['source_hashes'].items():
        require(hashlib.sha256(pack['sources'][name].encode()).hexdigest()==sha,'BASE_SOURCE_CLOSURE_'+name)
    return pack,source,build


def ws2_binding(rehearsal_pack,source_hashes):
    b=rehearsal_pack['cfg']['binding']
    return {'version':WS2.VERSION,'variant':'A-prime-R384-R224-R7-one-view',
        'frozen_base_notebook_sha256':BASE_NOTEBOOK_SHA['rehearsal'],
        'frozen_base_build_sha256':BASE_BUILD_SHA['rehearsal'],
        'frozen_base_passing_receipt_sha256':BASE_PASS_SHA,
        'public_reference_notebook_sha256':PUBLIC_REFERENCE_NOTEBOOK_SHA,
        'frozen_base_source_bundle_sha256':b['source_bundle_sha256'],
        'frozen_r7_binding_sha256':digest(b),
        'frozen_development_windows_sha256':digest(rehearsal_pack['cfg']['development_windows']),
        'frozen_base_method':b['method'],'frozen_base_environment':b['environment'],
        'crop224_checkpoint':{'dataset':C224.DATASET,'version':C224.DATASET_VERSION,
            'file':C224.CHECKPOINT_NAME,'sha256':C224.CHECKPOINT_SHA256,'bytes':C224.CHECKPOINT_BYTES},
        'crop224_recipe':C224.RECIPE,'rank_recipe':REC.RECIPE,
        'rank_policy':'raw R224 float32 pandas CSV, then one average percentile rank on complete mean-filled population',
        'weights':{'old320':.325,'crop224':.175,'r7':.5},
        'missing_policy':'typed whole-arm data-local only; global genuine means; union floor(.03*N)',
        'new_source_sha256':source_hashes,'uid':UID,'labels':LABELS,
        'official_columns':b['columns']}


def require_preparation_identity(prepared,expected,builder_path):
    require(prepared.get('status')=='PREPARATION_ONLY' and prepared.get('mode')=='rehearsal'
            and prepared.get('binding')==expected['cfg']['ws2_binding']
            and prepared.get('builder_sha256')==file_sha(builder_path),
            'EXACT_PREPARATION_IDENTITY')
    require(prepared.get('payload_sha256')==digest(expected)
            and prepared.get('source_hashes')=={name:hashlib.sha256(content.encode()).hexdigest()
                for name,content in expected['sources'].items()},
            'EXACT_PREPARATION_SOURCE_CLOSURE')


def verify_manifest_files(root,manifest,expected_names):
    require(set(manifest.get('files',{}))==set(expected_names),'PREPARATION_MANIFEST_COVERAGE')
    for name,pin in manifest['files'].items():
        path=Path(root)/name
        require(path.is_file() and pin=={'sha256':file_sha(path),'bytes':path.stat().st_size},
                'PREPARATION_MANIFEST_FILE_'+name)


def verify_preparation_build(build_root,archive_rehearsal,archive_rehearsal_receipt):
    """Rebuild the expected packet from pinned v7 bytes and compare every saved byte."""
    root=Path(build_root);project=Path(__file__).resolve().parents[1]
    needed=('notebook.ipynb','kernel-metadata.json','build-receipt.json','colab-payload.xz',
            'run_r224_diag.py','input-requirements.json','inputs.example.json','MANIFEST.json')
    require(all((root/name).is_file() for name in needed),'PREPARATION_BUILD_FILES')
    replay,_,_=archived_payload(archive_rehearsal,archive_rehearsal_receipt,'rehearsal')
    for name in ('knee/public_r7_recovery_entry.py','knee/public_r7_recovery.py',
                 'knee/public_s6_recovery_entry.py'):
        require(file_sha(project/name)==hashlib.sha256(replay['sources'][name].encode()).hexdigest(),
                'LOCAL_CHECKER_MATCHES_ARCHIVED_'+name)
    hashes={name:file_sha(project/name) for name in NEW_FILES}
    expected=json.loads(canonical(replay))
    expected['sources'].update({name:(project/name).read_text() for name in NEW_FILES})
    expected['cfg']['ws2_binding']=ws2_binding(replay,hashes)
    prepared=json.loads((root/'build-receipt.json').read_bytes())
    require_preparation_identity(prepared,expected,__file__)
    require(prepared.get('files')=={name:file_sha(root/name) for name in ('notebook.ipynb','kernel-metadata.json')},
            'PREPARATION_FILE_HASHES')
    notebook=json.loads((root/'notebook.ipynb').read_bytes())
    require(len(notebook.get('cells',[]))==1 and notebook['cells'][0]['cell_type']=='code'
            and ''.join(notebook['cells'][0]['source'])==bootstrap(expected),
            'EXACT_PREPARATION_BOOTSTRAP')
    require(json.loads(lzma.decompress((root/'colab-payload.xz').read_bytes()))==expected,
            'EXACT_COLAB_SOURCE_PACKET')
    require((root/'run_r224_diag.py').read_bytes()==(project/'scripts/run_public_ws2_colab_diag.py').read_bytes(),
            'EXACT_COLAB_RUNNER')
    metadata=json.loads((root/'kernel-metadata.json').read_bytes())
    require(metadata.get('is_private') is True and metadata.get('enable_internet') is False
            and metadata.get('enable_gpu') is True and metadata.get('machine_shape')=='NvidiaTeslaT4'
            and metadata.get('dataset_sources')[-1]==C224.DATASET,'PRIVATE_OFFLINE_T4_METADATA')
    requirement=json.loads((root/'input-requirements.json').read_bytes())
    registry=requirement.get('private_frozen39_registry',[])
    require(registry==[[sid,'train_series'] for sid in replay['cfg']['probe_order']]
            +[[sid,'test_series'] for sid in replay['cfg']['binding']['visible_ids']],
            'EXACT_PRIVATE39_INPUT_REGISTRY')
    manifest=json.loads((root/'MANIFEST.json').read_bytes())
    require(manifest.get('status')=='PREPARATION_ONLY' and manifest.get('payload_sha256')==digest(expected),
            'PREPARATION_MANIFEST_IDENTITY')
    verify_manifest_files(root,manifest,set(needed)-{'MANIFEST.json'})
    return prepared


def verify_fresh_rehearsal(result_root,binding):
    """Inspect actual saved ledgers and reproduce the final CSV; PASS alone is insufficient."""
    root=Path(result_root)
    receipt_path=root/'receipt.json';base_path=root/'base'/'receipt.json';csv_path=root/'visible-predictions.csv'
    require(all(x.is_file() for x in (receipt_path,base_path,csv_path)),'FULL_REHEARSAL_FILES')
    import numpy as np
    r=json.loads(receipt_path.read_bytes());base=json.loads(base_path.read_bytes())
    WS2.require_rehearsal(r,binding,base_path,csv_path)
    ids=json.loads((root/'base'/'assignment.json').read_bytes())['ids']
    require(len(ids)==39 and len(set(ids))==39 and ids==base['binding']['s6_probe_ids']+base['binding']['old_historical_ids']+base['visible_ids'],
            'EXACT_FROZEN39_ORDER')
    arms={}
    for name,base_name,folder,reports in (
            ('old320','old',root/'base',base['workers']['old']),
            ('r7','s6',root/'base',base['workers']['s6']),
            ('crop224','crop224',root,r['crop224_workers'])):
        arms[name]=WS2.merged_arm(ids,reports,folder,base_name)
    computed=REC.recover(*(arms[name][0] for name in REC.ARMS),
                         *(arms[name][1] for name in REC.ARMS))
    frozen=R7.recover(arms['old320'][0],arms['r7'][0],arms['old320'][1],arms['r7'][1])
    require(np.array_equal(frozen['control_rank'],computed['ranks']['old320'])
            and np.array_equal(frozen['s6_rank'],computed['ranks']['r7']),
            'SAVED_UNCHANGED_BASELINE_RANKS')
    hashes={'old_rank_sha256':hashlib.sha256(frozen['control_rank'].tobytes()).hexdigest(),
        'r7_rank_sha256':hashlib.sha256(frozen['s6_rank'].tobytes()).hexdigest(),
        'blend_sha256':hashlib.sha256(frozen['blend'].tobytes()).hexdigest()}
    require(r.get('baseline_full_population')==hashes,'SAVED_BASELINE_OUTPUT_HASHES')
    require(computed['union_recovered']==0,'FRESH_REHEARSAL_NO_FAILURE')
    index={sid:i for i,sid in enumerate(ids)}
    with csv_path.open(newline='') as handle:rows=list(csv.reader(handle))
    actual=np.array([[float(x) for x in row[1:]] for row in rows[1:]])
    expected=computed['blend'][[index[s] for s in base['visible_ids']]]
    require(actual.shape==(3,12) and np.max(np.abs(actual-expected))<=5e-10,'RECOMPUTED_CSV_MATCH')
    return {'receipt_sha256':file_sha(receipt_path),'base_receipt_sha256':file_sha(base_path),
            'csv_sha256':file_sha(csv_path),'qualified_r224_state':r['crop224_workers'][0]['state_before'],
            'projection':r['projection'],'all_six_ledgers_recomputed':True}


def bootstrap(pack):
    encoded=base64.b64encode(lzma.compress(canonical(pack),preset=9)).decode()
    prefix="""import time
started=time.monotonic()
import os,sys,json,base64,lzma,hashlib,signal,subprocess
from pathlib import Path
def deadline(signum,frame):raise TimeoutError('Actual nine-hour runtime exhausted')
signal.signal(signal.SIGALRM,deadline);signal.alarm(9*3600)
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',BUILD_MONAI='0',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
"""
    tail="""
payload=json.loads(lzma.decompress(base64.b64decode(encoded,validate=True)))
if hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=expected_payload:raise ValueError('A-prime payload changed')
project=Path('/kaggle/working/ws2-a-prime-project');code=project/'source'
for name,content in payload['sources'].items():
    p=project/name if name.startswith('knee/') else code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode())
sys.path.insert(0,str(project))
from knee.public_ws2_entry import run
from knee.public_s6_qualification import file_sha,save,require
cfg=payload['cfg'];codecs=payload['codecs'];out=Path('/kaggle/working/ws2-a-prime')
def mount(ref,required):
    owner,slug=ref.split('/');roots=[Path('/kaggle/input/datasets')/owner/slug,Path('/kaggle/input')/slug]
    matches=[p for p in roots if all((p/n).is_file() for n in required)]
    require(len(matches)==1,'UNIQUE_IMMUTABLE_ATTACHMENT');return matches[0]
try:
    require(sys.version_info[:2]==(3,12),'PYTHON_ABI')
    codec=mount('goodpjw2008/rsna-knee-2-5d-convnext-reader',list(codecs))
    for name,pin in codecs.items():require(file_sha(codec/name)==pin,'EXACT_OFFLINE_CODEC')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*[str(codec/n) for n in codecs]],check=True)
    compact=mount('heliosli/rsna-knee-compact-r7cnxfsss',['manifest.json',*cfg['manifest']['sha256']])
    require(file_sha(compact/'manifest.json')==cfg['asset_binding']['manifest_sha256'],'EXACT_R7_V1_MANIFEST')
    anatomy=mount('heliosli/rsna-orthofoundation-fold0-epoch7-20261004',['anatomy.pt'])/'anatomy.pt'
    weights=mount('nartaa/rsna-knee-publication-swa-weights-20261007',[cfg['public_manifest']['checkpoint_file'],cfg['ws2_binding']['crop224_checkpoint']['file']])
    roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
    matches=[p for p in roots if all((p/n).is_file() for n in ('test.csv','test_series.csv','sample_submission.csv'))]
    require(len(matches)==1,'UNIQUE_ACTUAL_TEST_ROOT')
    cfg.update(code=str(code),compact=str(compact),anatomy=str(anatomy),
        control_checkpoint=str(weights/cfg['public_manifest']['checkpoint_file']),
        crop224_checkpoint=str(weights/cfg['ws2_binding']['crop224_checkpoint']['file']))
    save(project/'runtime-config.json',cfg)
    report=run(matches[0],cfg,out,started=started)
    print(json.dumps({'status':report['status'],'completed':report['completed'],'mode':report['mode'],
        'elapsed_seconds':report['elapsed_seconds'],'conservative_seconds':(report.get('projection') or {}).get('conservative_seconds')}),flush=True)
except BaseException as e:
    save(out/'setup-failure-private.json',{'type':type(e).__name__,'detail':str(e)})
    raise
"""
    script=prefix+f'encoded={encoded!r}\nexpected_payload={digest(pack)!r}\n'+tail
    compile(script,'<ws2-a-prime>','exec')
    return script


def build(*,archive_rehearsal,archive_rehearsal_receipt,archive_passing_receipt,
          archive_entry,archive_entry_receipt,destination,slug,mode,
          rehearsal_result=None):
    project=Path(__file__).resolve().parents[1];dest=Path(destination)
    require(not dest.exists() and dest.resolve().is_relative_to(project/'outputs'),'FRESH_IGNORED_BUILD')
    replay,_,frozen_build=archived_payload(archive_rehearsal,archive_rehearsal_receipt,'rehearsal')
    for name in ('knee/public_r7_recovery_entry.py','knee/public_r7_recovery.py',
                 'knee/public_s6_recovery_entry.py'):
        require(file_sha(project/name)==hashlib.sha256(replay['sources'][name].encode()).hexdigest(),
                'LOCAL_CHECKER_MATCHES_ARCHIVED_'+name)
    require(file_sha(archive_passing_receipt)==BASE_PASS_SHA,'EXACT_PASSING_V7_RECEIPT')
    qualified=json.loads(Path(archive_passing_receipt).read_bytes())
    require(qualified.get('status')=='PASS' and qualified.get('completed')==39 and qualified.get('binding')==replay['cfg']['binding'],
            'FROZEN_BASELINE_PASS')
    if mode=='entry':pack,_,_=archived_payload(archive_entry,archive_entry_receipt,'entry')
    else:pack=json.loads(canonical(replay))
    shared=set(pack['sources'])&set(replay['sources'])
    require(all(pack['sources'][name]==replay['sources'][name] for name in shared if name!='knee/public_r7_recovery_rehearsal.py'),
            'ENTRY_REHEARSAL_BASE_SOURCE_IDENTITY')
    for name in NEW_FILES:pack['sources'][name]=(project/name).read_text()
    source_hashes={name:file_sha(project/name) for name in NEW_FILES}
    binding=ws2_binding(replay,source_hashes)
    cfg=pack['cfg'];cfg['ws2_binding']=binding
    verification=None
    if mode=='entry':
        require(rehearsal_result is not None,'FULL_WS2_REHEARSAL_REQUIRED')
        verification=verify_fresh_rehearsal(rehearsal_result,binding)
        cfg['r224_state_pin']=verification['qualified_r224_state']
        require(not any(k in cfg for k in PRIVATE_CFG) and not any(k in cfg['binding'] for k in PRIVATE_BINDING),
                'PURE_ENTRY_PRIVATE_FIXTURES_STRIPPED')
    else:
        require(len(cfg.get('probes',{}))==36 and len(cfg.get('probe_order',[]))==36,'FROZEN39_REGISTRY')
    script=bootstrap(pack)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},
        'cells':[{'id':'a-prime-image-inference','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                  'source':script.splitlines(True)}]}
    archived_meta=Path(archive_rehearsal).with_name('kernel-metadata.json')
    meta=json.loads(archived_meta.read_bytes())
    require(meta['dataset_sources'][-1]=='nartaa/rsna-knee-publication-swa-weights-20261007'
            and meta['machine_shape']=='NvidiaTeslaT4' and meta['enable_internet'] is False,
            'FROZEN_OFFLINE_BASE_METADATA')
    meta.update(id='ozgung/'+slug,title='A-prime R224 '+mode)
    require(len(slug)<=50 and mode in ('rehearsal','entry'),'WS2_BUILD_MODE_SLUG')
    raw=json.dumps(nb,indent=2)+'\n';require(len(raw.encode())<900000,'NOTEBOOK900K_CAP')
    dest.mkdir(parents=True)
    (dest/'notebook.ipynb').write_text(raw)
    (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    receipt={'status':'PREPARATION_ONLY' if mode=='rehearsal' else 'BOUND_ENTRY_BUILD',
        'mode':mode,'binding':binding,'archived_base_notebook_sha256':BASE_NOTEBOOK_SHA[mode],
        'archived_base_payload_sha256':frozen_build['payload_sha256'],
        'builder_sha256':file_sha(__file__),
        'payload_sha256':digest(pack),'source_hashes':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in pack['sources'].items()},
        'files':{name:file_sha(dest/name) for name in ('notebook.ipynb','kernel-metadata.json')},
        'notebook_bytes':len(raw.encode()),'source_files':len(pack['sources']),
        'runtime_checks_exercised':False,'rehearsal_verification':verification}
    save(dest/'build-receipt.json',receipt)
    if mode=='rehearsal':
        from shutil import copyfile
        (dest/'colab-payload.xz').write_bytes(lzma.compress(canonical(pack),preset=9))
        copyfile(project/'scripts/run_public_ws2_colab_diag.py',dest/'run_r224_diag.py')
        config=pack['cfg'];old_name=config['public_manifest']['checkpoint_file']
        requirements={'external_policy':'Externally staged by Claude; no image or weight bytes are included in this package.',
            'datasets':{'r7':'heliosli/rsna-knee-compact-r7cnxfsss/1',
                'weights':C224.DATASET+'/'+str(C224.DATASET_VERSION),
                'anatomy':'heliosli/rsna-orthofoundation-fold0-epoch7-20261004',
                'codecs':'goodpjw2008/rsna-knee-2-5d-convnext-reader',
                'competition':'rsna-knee-abnormality-detection, exact archived frozen39 registry'},
            'files':{'competition_root':{n:None for n in ('test.csv','test_series.csv','sample_submission.csv','train_series.csv')},
                'r7_root':{'manifest.json':config['asset_binding']['manifest_sha256'],**config['manifest']['sha256']},
                'anatomy_root':{'anatomy.pt':config['manifest']['anchor_weights_sha256']},
                'weights_root':{old_name:config['public_manifest']['checkpoint_sha256'],C224.CHECKPOINT_NAME:C224.CHECKPOINT_SHA256},
                'codec_root':pack['codecs']},
            'directories':{'competition_root':['test_series/<visible3 UID>/<series>/images',
                'train_series/<frozen36 UID>/<series>/images']},
            'private_frozen39_registry':[[sid,'train_series'] for sid in config['probe_order']]
                +[[sid,'test_series'] for sid in config['binding']['visible_ids']],
            'environment':{'python':'3.12','torch':config['environment']['torch'],
                'packages':config['environment']['packages'],'gpu':'one dedicated Tesla T4'}}
        (dest/'input-requirements.json').write_text(json.dumps(requirements,indent=2)+'\n')
        example={name:'/content/r224_inputs/'+name for name in requirements['files']}
        (dest/'inputs.example.json').write_text(json.dumps(example,indent=2)+'\n')
        files=('notebook.ipynb','kernel-metadata.json','build-receipt.json','colab-payload.xz',
               'run_r224_diag.py','input-requirements.json','inputs.example.json')
        manifest={'status':'PREPARATION_ONLY','payload_sha256':digest(pack),
            'files':{name:{'sha256':file_sha(dest/name),'bytes':(dest/name).stat().st_size} for name in files}}
        (dest/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    if '--verify-result' in sys.argv:
        check=argparse.ArgumentParser(description='Verify actual saved full WS2 rehearsal evidence')
        check.add_argument('--verify-result',required=True)
        check.add_argument('--rehearsal-build',required=True)
        check.add_argument('--archive-rehearsal',required=True)
        check.add_argument('--archive-rehearsal-receipt',required=True)
        args=check.parse_args()
        prepared=verify_preparation_build(args.rehearsal_build,
            args.archive_rehearsal,args.archive_rehearsal_receipt)
        outcome=verify_fresh_rehearsal(args.verify_result,prepared['binding'])
        print(json.dumps({'status':'QUALIFIED_TWO_T4_REHEARSAL','evidence':outcome}))
        raise SystemExit(0)
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive-rehearsal','archive-rehearsal-receipt','archive-passing-receipt',
                 'archive-entry','archive-entry-receipt','destination','slug'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--mode',choices=('rehearsal','entry'),required=True)
    p.add_argument('--rehearsal-result')
    r=build(**vars(p.parse_args()))
    print(json.dumps({'status':r['status'],'notebook_sha256':r['files']['notebook.ipynb'],
        'payload_sha256':r['payload_sha256'],'runtime_checks_exercised':False}))
