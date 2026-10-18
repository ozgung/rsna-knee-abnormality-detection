"""Build a private, offline A‴ rehearsal packet; this command never launches it."""
import argparse, ast, base64, hashlib, json, lzma, subprocess, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee import public_nafull_aprime as P
from knee import public_nafull_aprime_runtime as RT
from knee.public_s6_qualification import canonical,digest,file_sha,require,save

PROJECT=Path(__file__).resolve().parents[1]
MAIN=Path('/Users/ozgun/Dev/rsna-knee-abnormality-detection')
ARCH=MAIN/'outputs/kaggle/ws2-r224-v2-qual-20261015'
AR=MAIN/'outputs/kaggle/ws2-r224-v2-qual-20261015-result/ws2-a-prime'
B0=MAIN/'outputs/kaggle/knee-s6-dread-b0-rehearsal-20261013-result-v1/s6-recovery-entry'
B0_ARCH=MAIN/'outputs/kaggle/knee-s6-dread-b0-rehearsal-20261013'
FROZEN=MAIN/'outputs/analysis/na-full-aprime-rehearsal-20261016/frozen-aprime-sources.json'
NB_SHA='412eef8fc280e4596808cc4d4ae0aab4027bb632872125d910782a0105e29620'
PAYLOAD_SHA='ecb7db28c1989957b83c8c95ec0796bb241427feacc1297511a9c335ecd7008e'
NEW=('knee/public_nafull_aprime.py','knee/public_nafull_aprime_runtime.py',
'knee/public_nafull_aprime_recovery_adapter.py','knee/public_nafull_aprime_entry.py',
'knee/public_nafull_aprime_rehearsal.py','knee/public_s6_recovery_b0.py',
'knee/public_s6_recovery_b0prime.py','knee/public_s6_recovery_b0_rehearsal.py',
'knee/public_s6_recovery_b0prime_rehearsal.py','knee/public_nafull_raptor_input.py','knee/public_nafull_reference.py')
DATASETS=['heliosli/rsna-knee-compact-r7cnxfsss/1','heliosli/rsna-orthofoundation-fold0-epoch7-20261004',
'goodpjw2008/rsna-knee-2-5d-convnext-reader','nartaa/rsna-knee-publication-swa-weights-20261007',
P.CHECKPOINT['dataset']+'/1']

def read(p): return json.loads(Path(p).read_bytes())
def unpack(path):
    nb=read(path); code=''.join(nb['cells'][0]['source']); tree=ast.parse(code)
    vals={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('encoded','expected_payload')}
    require(set(vals)=={'encoded','expected_payload'},'ARCHIVED_PAYLOAD_LITERALS')
    pack=json.loads(lzma.decompress(base64.b64decode(vals['encoded'],validate=True)))
    frozen=read(FROZEN)
    require(file_sha(path)==NB_SHA and digest(pack)==vals['expected_payload']==PAYLOAD_SHA
        and frozen['archive_notebook_sha256']==NB_SHA and frozen['payload_sha256']==PAYLOAD_SHA,'FROZEN_APRIME_ARCHIVE')
    for name,pin in frozen['protected_source_hashes'].items():
        require(hashlib.sha256(pack['sources'][name].encode()).hexdigest()==pin,'FROZEN_SOURCE_'+name)
    return pack,code

def probabilities(root,ids,arm,folders):
    import numpy as np
    out={}
    for folder in folders:
        with np.load(root/folder/'predictions.npz',allow_pickle=False) as z:
            for sid,p,status in zip(z['study_ids'].tolist(),z['probabilities'],z['statuses'].tolist()):
                require(status=='genuine' and sid not in out,'ARCHIVED_GENUINE_UNIQUE_'+arm)
                out[sid]=np.asarray(p,dtype=float).tolist()
    require(set(out)==set(ids),'ARCHIVED_ALL39_'+arm)
    return out

def packet(cpu_receipt):
    from knee import public_nafull_aprime as PROFILE
    from knee import public_r7_recovery_entry as R7
    from knee import public_s6_recovery_b0prime as B0P
    from knee.public_s6_recovery_b0 import RAPTOR62_COUNT_KEYS
    from scripts import build_public_b0prime as B0BUILD
    a,_=unpack(ARCH/'notebook.ipynb')
    require(file_sha(B0_ARCH/'notebook.ipynb')==B0BUILD.NOTEBOOK_SHA,'EXACT_B0_SOURCE_ARCHIVE')
    b,_=B0BUILD.unpack_notebook(B0_ARCH/'notebook.ipynb')
    require(digest(b)==B0BUILD.PAYLOAD_SHA,'EXACT_B0_PAYLOAD')
    require(file_sha(B0/'receipt.json')==B0BUILD.RESULT_SHA,'EXACT_B0_READER_RESULT')
    br=read(B0/'receipt.json')
    from knee import public_s6_recovery_b0 as B0_RUNTIME
    B0_RUNTIME.require_rehearsal(br,b['cfg']['binding'],B0/'visible-predictions.csv')
    for name,pin in B0BUILD.FROZEN.items():
        require(file_sha(PROJECT/name)==pin,'UNCHANGED_B0_SUPPORT_'+name)
    require(file_sha(PROJECT/'knee/public_nafull_raptor_input.py')==hashlib.sha256(b['sources']['knee/raptor_input.py'].encode()).hexdigest(),'EXACT_ISOLATED_B0_READER')
    require(file_sha(PROJECT/'knee/public_nafull_reference.py')==hashlib.sha256(b['sources']['knee/reference.py'].encode()).hexdigest(),'EXACT_ISOLATED_B0_REFERENCE')
    cpu=PROFILE.checkpoint_manifest(cpu_receipt)
    bc=b['cfg']; ids=bc['probe_order']+bc['binding']['visible_ids']
    require(len(ids)==39 and len(set(ids))==39,'K62_39_ORDER')
    reader={x['id']:{'input':x['input'],'decode_counters':x['decode_counters']}
        for w in br['workers']['old'] for x in w['records']}
    require(set(reader)==set(ids),'K62_READER_ALL39')
    for sid,row in reader.items():require(set(row['decode_counters'])>=set(RAPTOR62_COUNT_KEYS),'K62_COUNTER_SCHEMA')
    base=read(AR/'base/receipt.json'); ar=read(AR/'receipt.json')
    require(base['status']=='PASS' and base['completed']==39 and base['zero_failed'] is True
        and ar['completed']==39 and ar['zero_failed'] is True,'ARCHIVED_ARM_RECEIPTS_GENUINE')
    rids=read(AR/'base/assignment.json')['ids']; require(rids==ids,'FROZEN39_ORDER')
    r7rows={x['id']:x for w in base['workers']['s6'] for x in w['records']}
    r224rows={x['id']:x for w in ar['crop224_workers'] for x in w['records']}
    require(set(r7rows)==set(r224rows)==set(ids),'ARCHIVED_INPUT_COVERAGE')
    from knee import public_ws2_entry as WS2
    r7values,r7missing=WS2.merged_arm(ids,base['workers']['s6'],AR/'base','s6')
    r224values,r224missing=WS2.merged_arm(ids,ar['crop224_workers'],AR,'crop224')
    require(not r7missing.any() and not r224missing.any(),'ARCHIVED_GENUINE_SAVED_LEDGERS')
    r7p={sid:row.tolist() for sid,row in zip(ids,r7values)}
    r224p={sid:row.tolist() for sid,row in zip(ids,r224values)}
    for folder,arm,reports in [(AR/'base','s6',base['workers']['s6']), (AR,'crop224',ar['crop224_workers'])]:
        for i,report in enumerate(reports):
            require(read(folder/f'board{i}-{arm}'/arm/'receipt.json')==report,'ARCHIVED_SAVED_RECEIPT_IDENTITY')
    final=read(MAIN/'outputs/analysis/na-full-export-acceptance-20261016/per-file-final-verification.json')
    require(final['state']=='verified' and final['dataset']==PROFILE.CHECKPOINT['dataset'] and final['version']==1
        and final['all_11_files_verified'] is True and final['is_private'] is True
        and final['files'][PROFILE.CHECKPOINT['file']]=={'sha256':PROFILE.CHECKPOINT['sha256'],'bytes':PROFILE.CHECKPOINT['bytes']},
        'INDEPENDENT_FINAL_EXPORT_VERIFICATION')
    oldinp={sid:x['input'] for sid,x in r7rows.items()}
    r224inp={sid:x['input'] for sid,x in r224rows.items()}
    paths=[AR/'receipt.json',AR/'base/receipt.json']
    paths += [AR/f'base/board{i}-s6/s6/{n}' for i in (0,1) for n in ('receipt.json','predictions.npz')]
    paths += [AR/f'board{i}-crop224/crop224/{n}' for i in (0,1) for n in ('receipt.json','predictions.npz')]
    source_files={str(x.relative_to(MAIN)):file_sha(x) for x in paths}
    binding={**base['binding'],'version':PROFILE.VERSION,
        'method':{**base['binding']['method'],'old':PROFILE.METHOD_OLD},
        'old_profile':dict(RT.NA_WORKER_GLOBALS['OLD_PROFILE']),
        'public_manifest':None,
        'sentinel_inputs':{**base['binding']['sentinel_inputs'],'old':{sid:reader[sid]['input'] for sid in bc['old_sentinels']}},
        'qualified_state':{**base['binding']['qualified_state'],'old_state':PROFILE.CHECKPOINT['state_sha256'],'old_grid':PROFILE.CONSTRUCTOR_GRID_SHA256},
        'sentinel_probabilities':{**base['binding']['sentinel_probabilities'],'old':{sid:None for sid in bc['old_sentinels']}},
        'nafull_cpu_binding':cpu,'final_export_verification_sha256':file_sha(MAIN/'outputs/analysis/na-full-export-acceptance-20261016/per-file-final-verification.json'),
        'nafull_reader_reference':reader,
        'r7_input_reference':oldinp,'r7_probability_reference':r7p,'r224_input_reference':r224inp,'r224_probability_reference':r224p,
        'archived_aprime_reference':{'notebook_sha256':NB_SHA,'payload_sha256':PAYLOAD_SHA,
         'receipt_sha256':file_sha(AR/'receipt.json'),'base_receipt_sha256':file_sha(AR/'base/receipt.json'),
         'combined_status':'FAIL','combined_failure':'ordinary projection exceeded 8.3h',
         'files':source_files,'probability_tolerances':{'r7':R7.CROSS_BOARD_TOLERANCE['s6'],'r224':1e-6}},
        'arithmetic':{'formula':'.5*(.65 rank(na-full)+.35 rank(R224))+.5 rank(R7)',
         'population':'all39; per-arm genuine global-mean fill before percentile ranking','union_cap':'floor(.03*N)'}}
    out=json.loads(canonical(a)); cfg=out['cfg']
    worker={k:bc[k] for k in ('public_source','decoder_source','public_manifest','control_checkpoint',
        'control_state_sha256','control_grid_sha256','probes','old_sentinels','raptor62_reference','raptor62_visible_atol') if k in bc}
    worker['public_manifest']={**bc['public_manifest'],'profile':'raptor','checkpoint_sha256':PROFILE.CHECKPOINT['sha256'],
        'checkpoint_file':PROFILE.CHECKPOINT['file'],'weight_dataset':PROFILE.CHECKPOINT['dataset'],
        'attribution':'na-full Raptor64 SWA epochs 9-11; immutable version-1 export'}
    worker['probes']={sid:{**probe,'old':{'input':reader[sid]['input'],'probabilities':None}} for sid,probe in worker['probes'].items()}
    worker['control_checkpoint']=PROFILE.CHECKPOINT['file']; worker['control_state_sha256']=PROFILE.CHECKPOINT['state_sha256']
    worker['control_grid_sha256']=PROFILE.CONSTRUCTOR_GRID_SHA256
    worker['raptor62_reference']={sid:{'counts':reader[sid]['decode_counters']} for sid in bc['binding']['visible_ids']}
    worker.pop('raptor62_visible_atol',None)
    RT.nafull_profile_gate(None,worker['public_manifest'])
    binding['public_manifest']=worker['public_manifest']
    cfg.update(mode='rehearsal',nafull_worker_cfg=worker,nafull_base_binding=binding)
    cfg['ws2_binding']={**cfg['ws2_binding'],'version':PROFILE.VERSION,'variant':'A-triple-prime-nafull-R224-R7',
        'weights':{'old320':.325,'crop224':.175,'r7':.5},
        'legacy_slot_mapping':{'old320':'na-full Raptor64 K62 one-view','crop224':'unchanged R224','r7':'unchanged R7'},
        'ancestor_r7_binding_sha256':cfg['ws2_binding']['frozen_r7_binding_sha256'],
        'frozen_r7_binding_sha256':digest(binding),
        'nafull_reader_reference':reader,'r7_input_reference':oldinp,'r7_probability_reference':r7p,
        'r224_input_reference':r224inp,'r224_probability_reference':r224p,'nafull_cpu_binding':cpu,
        'nafull_base_binding_sha256':digest(binding),'archived_aprime_reference':binding['archived_aprime_reference'],
        'development_windows_sha256':digest(cfg['development_windows']),
        'projection_policy':'fresh ordinary projection <= 8.3h and actual runtime < 9h'}
    out['sources'].update({n:(PROJECT/n).read_text() for n in NEW})
    for name,content in a['sources'].items():
        require(out['sources'][name]==content,'UNCHANGED_ARCHIVED_SOURCE_'+name)
    extension_names={name for name in out['sources'] if name.endswith('_rehearsal.py')}
    binding['source_bundle_sha256']=digest({n:text for n,text in out['sources'].items() if n not in extension_names})
    binding['rehearsal_extension_sha256']=digest({n:out['sources'][n] for n in extension_names})
    cfg['ws2_binding']['frozen_r7_binding_sha256']=digest(binding)
    cfg['ws2_binding']['nafull_base_binding_sha256']=digest(binding)
    code=bootstrap(out)
    for n,s in out['sources'].items(): compile(s,n,'exec')
    return out,code

def bootstrap(pack):
    # Keep the established WS2 setup/source extraction/offline dependencies.
    from scripts.build_public_ws2_entry import bootstrap as archived_bootstrap
    code=archived_bootstrap(pack)
    edits={
        'from knee.public_ws2_entry import run':'from knee.public_nafull_aprime_entry import run',
        "project=Path('/kaggle/working/ws2-a-prime-project')":"project=Path('/kaggle/working/nafull-aprime-project')",
        "out=Path('/kaggle/working/ws2-a-prime')":"out=Path('/kaggle/working/nafull-aprime')",
    }
    for before,after in edits.items():
        require(code.count(before)==1,'EXACT_BOOTSTRAP_SITE_'+before)
        code=code.replace(before,after)
    before="    save(project/'runtime-config.json',cfg)"
    after=f"""    naweights=mount({P.CHECKPOINT['dataset']!r},[{P.CHECKPOINT['file']!r}])
    nafile=naweights/{P.CHECKPOINT['file']!r}
    require(nafile.stat().st_size=={P.CHECKPOINT['bytes']} and file_sha(nafile)=={P.CHECKPOINT['sha256']!r},'NAFULL_VERSION1_CHECKPOINT_BYTES')
    cfg['nafull_worker_cfg']['control_checkpoint']=str(nafile)
    save(project/'runtime-config.json',cfg)"""
    require(code.count(before)==1,'EXACT_NAFULL_CHECKPOINT_MOUNT_SITE')
    code=code.replace(before,after)
    compile(code,'<nafull-aprime-bootstrap>','exec')
    return code

def metadata(slug):
 m=read(ARCH/'kernel-metadata.json')
 require(isinstance(slug,str) and 3<=len(slug)<=50 and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in slug),'SAFE_KERNEL_SLUG')
 m.update(id='ozgung/'+slug,title=slug,dataset_sources=m['dataset_sources']+[P.CHECKPOINT['dataset']+'/1'],machine_shape='NvidiaTeslaT4',
    is_private=True,enable_internet=False,enable_gpu=True);m.pop('timeout',None);return m

def build(cpu_receipt,destination,slug):
 dest=Path(destination);require(not dest.exists() and dest.resolve().is_relative_to(PROJECT/'outputs'),'FRESH_IGNORED_OUTPUT')
 pack,code=packet(cpu_receipt)
 nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},
 'cells':[{'id':'nafull-aprime-private-rehearsal','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
 raw=json.dumps(nb,indent=2)+'\n'; require(len(raw.encode())<900000,'NOTEBOOK900K_CAP')
 dest.mkdir(parents=True);(dest/'notebook.ipynb').write_text(raw);save(dest/'kernel-metadata.json',metadata(slug))
 receipt={'status':'PREPARATION_ONLY','mode':'rehearsal','payload_sha256':digest(pack),
 'binding':pack['cfg']['ws2_binding'],'source_hashes':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in pack['sources'].items()},
 'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
 'source_files':{n:file_sha(PROJECT/n) for n in NEW},'builder_sha256':file_sha(__file__),
 'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},
 'notebook_bytes':len(raw.encode()),'runtime_checks_exercised':False}
 save(dest/'build-receipt.json',receipt)
 save(dest/'MANIFEST.json',{'mode':'rehearsal','payload_sha256':digest(pack),'files':{
  n:{'sha256':file_sha(dest/n),'bytes':(dest/n).stat().st_size} for n in ('notebook.ipynb','kernel-metadata.json','build-receipt.json')}})
 return receipt

def verify_build(root,cpu_receipt):
 root=Path(root);r=read(root/'build-receipt.json');m=read(root/'MANIFEST.json');expected,code=packet(cpu_receipt)
 nb=read(root/'notebook.ipynb');meta=read(root/'kernel-metadata.json')
 require(len(nb['cells'])==1 and ''.join(nb['cells'][0]['source'])==code,'DETERMINISTIC_BOOTSTRAP_REPLAY')
 require(meta==metadata(meta['title']) and meta['is_private'] is True and meta['enable_internet'] is False
  and meta['machine_shape']=='NvidiaTeslaT4' and meta['title']==meta['id'].split('/')[-1],'PRIVATE_OFFLINE_T4')
 require(r['status']=='PREPARATION_ONLY' and r['payload_sha256']==digest(expected)
  and r['source_hashes']=={n:hashlib.sha256(s.encode()).hexdigest() for n,s in expected['sources'].items()},'EXACT_PAYLOAD_CLOSURE')
 require(set(m['files'])=={'notebook.ipynb','kernel-metadata.json','build-receipt.json'},'MANIFEST_COVERAGE')
 require(r['binding']==expected['cfg']['ws2_binding'] and r['builder_sha256']==file_sha(__file__),'BOUND_BUILDER_AND_BINDING')
 require(r['files']=={n:file_sha(root/n) for n in ('notebook.ipynb','kernel-metadata.json')},'BUILD_FILE_HASHES')
 for n,pin in m['files'].items():require(pin=={'sha256':file_sha(root/n),'bytes':(root/n).stat().st_size},'MANIFEST_'+n)
 return r

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-receipt',required=True)
 p.add_argument('--destination');p.add_argument('--slug',default='nafull-aprime-private-rehearsal')
 p.add_argument('--verify-build');a=p.parse_args()
 out=verify_build(a.verify_build,a.cpu_receipt) if a.verify_build else build(a.cpu_receipt,a.destination,a.slug)
 print(json.dumps({'status':out['status'],'payload_sha256':out['payload_sha256'],'files':out.get('files')}))
