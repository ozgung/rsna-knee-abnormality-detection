"""Build the separately sealed A-prime-w projection-waived ENTRY packet.

A-prime-w is the sealed A-prime entry (notebook 33654dbd..., payload bdf9e4c8...) with only
the final per-finding Raptor-half share changed by one added module,
knee/public_ws2w_entry.py. Every A-prime source byte, checkpoint, member, recovery,
rank, worker, runtime alarm and guard is carried over unchanged. The A-prime v2
qualification stays an ordinary FAIL; its exact failed receipt is replayed by the
frozen A-prime probe validator, and only the ordinary 8.3 h conservative projection
is waived under the separately named user waiver below. Validator code here is never
shipped in the entry closure.
"""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from scripts import build_public_ws2_entry as BASE
from scripts import build_public_ws2_projection_probe as PROBE
from knee import public_ws2_entry as WS2
from knee import public_ws2_recovery as REC
from knee import public_ws2w_entry as W
from knee.public_s6_qualification import LABELS,canonical,digest,file_sha,require,save

PROJECT=Path(__file__).resolve().parents[1]
MAIN=Path('~/Dev/rsna-knee-abnormality-detection')
KAGGLE=MAIN/'outputs/kaggle'
BUILDER_VERSION='ws2w-projection-probe-v1'
WAIVER='aprime-w-projection-20261017'
SLUG='aprime-w-probe-waiver-20261017'
NEW_SOURCE='knee/public_ws2w_entry.py'
OLD_IMPORT='from knee.public_ws2_entry import run\n'
NEW_IMPORT='from knee.public_ws2w_entry import run\n'
# Independent declaration of the intended map (Raptor-half share; R7 gets 1 - w).
EXPECTED_SHARES={'Lateral OA':.2,'PF OA':.6,'Synovitis':.8,"Baker's":.35}
CHANGED=tuple(c for c in LABELS if c in EXPECTED_SHARES)
APRIME={'kernel':'ozgung/ws2-r224-probe-waiver-20261015','submission':57229227,
    'root':KAGGLE/'ws2-r224-probe-waiver-20261015',
    'notebook':'33654dbd793569bd685e17a6cf3906401be5a3088c769861ed487623d6541c14',
    'payload':'bdf9e4c8b9bf6186d42bd02e4f44fe906940205bc312bcab4a81a0bff0fceadd',
    'kernel_metadata':'9f6a5805afa572da4214e64f1613fbf63ef9624547286e59640be53468a33fa1',
    'waiver':PROBE.WAIVER,'failed_receipt_sha256':PROBE.FAILED_RECEIPT_SHA}
ARCHIVE_V7=KAGGLE/'public-r7-recovery-rehearsal-20261013-v7'
QUAL_BUILD=KAGGLE/'ws2-r224-v2-qual-20261015'
QUAL_RESULT=KAGGLE/'ws2-r224-v2-qual-20261015-result/ws2-a-prime'
# Visible A-prime run: collected receipts/CSV plus ledgers pulled read-only from the same kernel output.
VISIBLE=KAGGLE/'ws2-r224-probe-waiver-20261015-visible'
VISIBLE_LEDGERS=KAGGLE/'ws2-r224-probe-waiver-20261015-visible-ledgers'
AUTHORIZATION={'reported_by':'Claude',
    'source':'docs/archive/handoff/claude-astra-handoff.md, 2026-10-17 16:52 UTC (USER WAIVER DECISIONS, given directly and prospectively)',
    'reported_user_choice':'Yes, submit A-prime-w',
    'independently_verified_direct_user_message':False,
    'scope':'one A-prime-w probe bound to the A-prime v2 failed qualification receipt; waive only the ordinary 8.3h '
            'conservative projection; actual nine-hour limit and every other check unchanged; no other receipt, model or gate inherits it'}


def read(path):return json.loads(Path(path).read_bytes())


def unpack_notebook(path):
    import base64,lzma
    nb=read(path);require(len(nb['cells'])==1 and nb['cells'][0]['cell_type']=='code','SINGLE_CODE_CELL')
    code=''.join(nb['cells'][0]['source']);tree=ast.parse(code)
    values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('encoded','expected_payload')}
    require(set(values)=={'encoded','expected_payload'},'PAYLOAD_LITERALS')
    pack=json.loads(lzma.decompress(base64.b64decode(values['encoded'],validate=True)))
    require(digest(pack)==values['expected_payload'],'EMBEDDED_PAYLOAD_HASH')
    return pack,code


def sealed_aprime():
    """Exact submitted A-prime entry bytes; local frozen modules must equal its sources."""
    root=APRIME['root']
    require(file_sha(root/'notebook.ipynb')==APRIME['notebook'] and file_sha(root/'kernel-metadata.json')==APRIME['kernel_metadata'],
            'EXACT_SEALED_APRIME_FILES')
    receipt=read(root/'build-receipt.json')
    require(receipt['payload_sha256']==APRIME['payload'] and receipt['files']['notebook.ipynb']==APRIME['notebook']
            and receipt['waiver']['id']==APRIME['waiver'] and receipt['waiver']['failed_receipt_sha256']==APRIME['failed_receipt_sha256'],
            'SEALED_APRIME_RECEIPT')
    pack,code=unpack_notebook(root/'notebook.ipynb')
    require(digest(pack)==APRIME['payload'] and code==BASE.bootstrap(pack),'SEALED_APRIME_PAYLOAD_BOOTSTRAP')
    require(NEW_SOURCE not in pack['sources'],'NEW_MODULE_ABSENT_FROM_APRIME')
    # Local modules used by the host-side replay (merge, recovery, ranks) equal the sealed sources.
    for name in (*BASE.NEW_FILES,'knee/public_r7_recovery_entry.py','knee/public_s6_recovery_entry.py','knee/public_r7_recovery.py'):
        require((PROJECT/name).read_text()==pack['sources'][name],'LOCAL_FROZEN_SOURCE_EQUALS_APRIME_'+name)
    require(pack['cfg']['ws2_binding']['labels']==LABELS,'APRIME_LABEL_ORDER')
    return pack,code,read(root/'kernel-metadata.json')


def admit(rehearsal_build,rehearsal_result,user_waiver,failed_receipt_sha256):
    """Same failed v2 receipt and full replay that sealed A-prime; bind A-prime pack to it."""
    require(user_waiver==WAIVER,'EXPLICIT_NAMED_APRIME_W_WAIVER_REQUIRED')
    require(failed_receipt_sha256==APRIME['failed_receipt_sha256']
            and file_sha(Path(rehearsal_result)/'receipt.json')==failed_receipt_sha256,'EXACT_APRIME_FAILED_V2_RECEIPT')
    pack,code,meta=sealed_aprime()
    prepared=BASE.verify_preparation_build(rehearsal_build,ARCHIVE_V7/'notebook.ipynb',ARCHIVE_V7/'build-receipt.json')
    # Frozen A-prime validator; its own token names the evidence replay, not this waiver.
    checked=PROBE.verify_probe_evidence(rehearsal_result,prepared['binding'],PROBE.WAIVER)
    require(pack['cfg']['ws2_binding']==prepared['binding'] and pack['cfg']['r224_state_pin']==checked['qualified_r224_state']
            and pack['cfg']['projection_probe_waiver']['failed_receipt_sha256']==failed_receipt_sha256,
            'APRIME_PACK_BOUND_TO_SAME_EVIDENCE')
    return pack,code,meta,checked


def entry_pack(pack,checked):
    entry=copy.deepcopy(pack)
    entry['sources'][NEW_SOURCE]=(PROJECT/NEW_SOURCE).read_text()
    cfg=entry['cfg']
    cfg['ws2w_blend']=W.profile()
    cfg['ws2w_base_entry']={'kernel':APRIME['kernel'],'submission':APRIME['submission'],
        'notebook_sha256':APRIME['notebook'],'payload_sha256':APRIME['payload'],'waiver':APRIME['waiver']}
    cfg['projection_probe_waiver']={'id':WAIVER,'authorization':AUTHORIZATION,
        'failed_receipt_sha256':APRIME['failed_receipt_sha256'],
        'status':'USER_REPORTED_PROJECTION_WAIVER_NOT_QUALIFICATION_PASS',
        'projection':checked['projection'],'actual_limit_seconds':WS2.MAX_SECONDS,
        'all_six_ledgers_recomputed':checked['all_six_ledgers_recomputed']}
    require_no_other_change(pack,entry)
    return entry


def require_no_other_change(aprime,entry):
    """Sources: A-prime plus exactly one module; cfg: three added/replaced metadata keys."""
    require(set(entry)==set(aprime) and entry['codecs']==aprime['codecs'],'PAYLOAD_KEYS_CODECS')
    require(set(entry['sources'])-set(aprime['sources'])=={NEW_SOURCE}
            and all(entry['sources'][n]==s for n,s in aprime['sources'].items()),'APRIME_SOURCES_BYTE_IDENTICAL_PLUS_ONE')
    keys=set(entry['cfg'])|set(aprime['cfg'])
    changed={k for k in keys if entry['cfg'].get(k)!=aprime['cfg'].get(k)}
    require(changed=={'ws2w_blend','ws2w_base_entry','projection_probe_waiver'},'ONLY_BLEND_AND_WAIVER_CFG_CHANGE')
    require(not any(k in entry['cfg'] for k in BASE.PRIVATE_CFG)
            and not any(k in entry['cfg']['binding'] for k in BASE.PRIVATE_BINDING),'PURE_ENTRY_PRIVATE_FIXTURES_STRIPPED')
    return True


def bootstrap(entry):
    code=BASE.bootstrap(entry)
    require(code.count(OLD_IMPORT)==1,'ONE_APRIME_ENTRY_IMPORT')
    code=code.replace(OLD_IMPORT,NEW_IMPORT);compile(code,'<aprime-w>','exec')
    return code


def require_bootstrap_preservation(code,aprime_code):
    """Bootstrap equals A-prime except the payload literals and the one entry import."""
    def body(text):
        tree=ast.parse(text)
        return [ast.dump(n) for n in tree.body if not (isinstance(n,ast.Assign) and len(n.targets)==1
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('encoded','expected_payload'))]
    a,b=body(aprime_code),body(code)
    diff=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    require(len(a)==len(b) and len(diff)==1 and b[diff[0]]==ast.dump(ast.parse(NEW_IMPORT).body[0])
            and a[diff[0]]==ast.dump(ast.parse(OLD_IMPORT).body[0])
            and code.count('signal.alarm(9*3600)')==1,'EXACT_APRIME_BOOTSTRAP_EXCEPT_ENTRY_IMPORT')
    return True


def metadata(aprime_meta):
    meta={**aprime_meta,'id':'ozgung/'+SLUG,'title':SLUG}
    require(meta['is_private'] is True and meta['enable_internet'] is False and meta['enable_gpu'] is True
            and meta['machine_shape']=='NvidiaTeslaT4' and 'timeout' not in meta
            and {k:v for k,v in meta.items() if k not in ('id','title')}=={k:v for k,v in aprime_meta.items() if k not in ('id','title')},
            'PRIVATE_OFFLINE_T4_NO_TIMEOUT_SAME_ATTACHMENTS')
    return meta


def evidence_arms(root,ledger_root=None):
    """Merged per-arm predictions/masks through the frozen ledger-binding merge."""
    root=Path(root);ledgers=Path(ledger_root or root)
    r=read(root/'receipt.json');base=read(root/'base'/'receipt.json')
    ids=read(ledgers/'base'/'assignment.json')['ids']
    arms={}
    for name,base_name,folder,reports in (('old320','old',ledgers/'base',base['workers']['old']),
            ('r7','s6',ledgers/'base',base['workers']['s6']),('crop224','crop224',ledgers,r['crop224_workers'])):
        arms[name]=WS2.merged_arm(ids,reports,folder,base_name)
    return ids,base['visible_ids'],arms,r


def csv_rows(path):
    with Path(path).open(newline='') as handle:rows=list(csv.reader(handle))
    require(rows[0][1:]==LABELS,'CSV_LABELS')
    return [x[0] for x in rows[1:]],[[float(v) for v in x[1:]] for x in rows[1:]]


REPLAY_CHILD=r'''
import importlib,json,sys
from pathlib import Path
import numpy as np
project=Path(sys.argv[1]).resolve();cfg=json.loads(Path(sys.argv[2]).read_text());sys.path.insert(0,str(project))
modules=['knee.public_ws2w_entry','knee.public_ws2_entry','knee.public_ws2_recovery','knee.public_r7_recovery_entry']
for name in modules:
    assert Path(importlib.import_module(name).__file__).resolve().is_relative_to(project),name
from knee import public_ws2w_entry as W,public_ws2_entry as WS2
assert cfg['mode']=='entry' and cfg['ws2w_blend']==W.profile() and cfg['r224_state_pin']
assert WS2.REC is importlib.import_module('knee.public_ws2_recovery')
out={}
for key in sys.argv[3:]:
    with np.load(Path(sys.argv[2]).parent/(key+'.npz'),allow_pickle=False) as a:
        r=W.recover(a['old320'],a['r7'],a['crop224'],a['old320_missing'],a['r7_missing'],a['crop224_missing'])
    np.save(Path(sys.argv[2]).parent/(key+'-blend.npy'),r['blend'])
    out[key]={'recipe':r['recipe'],'base_recipe':r['base_recipe'],'population':r['population']}
assert all(not name.startswith('knee.') or Path(m.__file__).resolve().is_relative_to(project)
           for name,m in list(sys.modules.items()) if getattr(m,'__file__',None))
print(json.dumps({'imports':modules,'all_from_unpacked_project':True,'replayed':out}))
'''


def isolated_replay(notebook,datasets):
    """Fresh `python -I` child imports the unpacked entry and runs its embedded blend."""
    import numpy as np
    pack,_=unpack_notebook(notebook)
    with tempfile.TemporaryDirectory(prefix='aprime-w-entry-import-') as folder:
        root=Path(folder);project=root/'project'
        for name,content in pack['sources'].items():
            path=Path(name);require(not path.is_absolute() and '..' not in path.parts,'SAFE_SOURCE_PATH')
            path=project/path if name.startswith('knee/') else project/'source'/path
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content.encode())
        (root/'cfg.json').write_text(json.dumps(pack['cfg']));(root/'child.py').write_text(REPLAY_CHILD)
        for key,arms in datasets.items():
            np.savez(root/(key+'.npz'),**{n:arms[n][0] for n in REC.ARMS},**{n+'_missing':arms[n][1] for n in REC.ARMS})
        proc=subprocess.run([sys.executable,'-I',str(root/'child.py'),str(project),str(root/'cfg.json'),*datasets],
                            cwd=root,capture_output=True,text=True)
        require(proc.returncode==0,'FRESH_ISOLATED_ENTRY_IMPORT: '+proc.stderr[-1500:])
        blends={key:np.load(root/(key+'-blend.npy')) for key in datasets}
    report={'status':'PASS','python':sys.version.split()[0],'isolated':True,'payload_sha256':digest(pack),
            **json.loads(proc.stdout.splitlines()[-1]),'gpu_exercised':False,'uses_real_images':False}
    return report,blends


def compare(name,ids,visible,arms,embedded,aprime_csv=None):
    """Embedded A-prime-w blend vs independent arithmetic and vs frozen A-prime blend."""
    import numpy as np
    rec=REC.recover(*(arms[n][0] for n in REC.ARMS),*(arms[n][1] for n in REC.ARMS))
    require(rec['union_recovered']==0,'NO_RECOVERY_IN_REPLAY_'+name)
    w=np.asarray([EXPECTED_SHARES.get(c,.5) for c in LABELS])
    half=(1.0-.35)*rec['ranks']['old320']+.35*rec['ranks']['crop224']
    independent=w*half+(1.0-w)*rec['ranks']['r7']
    require(np.array_equal(half,rec['raptor_slot']),'RAPTOR_HALF_ARITHMETIC_'+name)
    require(embedded.shape==independent.shape==rec['blend'].shape and np.array_equal(embedded,independent),
            'EMBEDDED_EQUALS_INDEPENDENT_'+name)
    differs=[c for j,c in enumerate(LABELS) if not np.array_equal(embedded[:,j],rec['blend'][:,j])]
    require(set(differs)<=set(CHANGED),'ONLY_FOUR_FINDINGS_DIFFER_'+name)
    require(all(np.array_equal(embedded[:,j],rec['blend'][:,j]) for j,c in enumerate(LABELS) if c not in CHANGED),
            'UNCHANGED_FINDINGS_BIT_IDENTICAL_'+name)
    out={'studies':len(ids),'differing_findings':differs,
         'max_abs_delta':{c:float(np.max(np.abs(embedded[:,j]-rec['blend'][:,j]))) for j,c in enumerate(LABELS) if c in CHANGED},
         'unchanged_findings_bit_identical':True,'embedded_equals_independent_exactly':True,
         'aprime_blend_sha256':hashlib.sha256(rec['blend'].tobytes()).hexdigest(),
         'aprime_w_blend_sha256':hashlib.sha256(embedded.tobytes()).hexdigest()}
    if aprime_csv is not None:
        got_ids,values=csv_rows(aprime_csv);index={s:i for i,s in enumerate(ids)}
        require(got_ids==visible,'APRIME_CSV_ORDER_'+name)
        require(np.max(np.abs(np.asarray(values)-rec['blend'][[index[s] for s in visible]]))<=5e-10,'APRIME_CSV_RECOMPUTED_'+name)
        out['aprime_csv_sha256']=file_sha(aprime_csv);out['aprime_csv_recomputed']=True
    return out


def replay(notebook,*,qual_result=QUAL_RESULT,visible=VISIBLE,visible_ledgers=VISIBLE_LEDGERS):
    qual=evidence_arms(qual_result)
    vis=evidence_arms(visible/'ws2-a-prime',visible_ledgers/'ws2-a-prime')
    aprime_binding=sealed_aprime()[0]['cfg']['ws2_binding']
    require(vis[3]['status']=='PASS' and vis[3]['mode']=='entry' and vis[3]['binding']==aprime_binding
            and vis[3]['csv_sha256']==file_sha(visible/'submission.csv'),'APRIME_VISIBLE_ENTRY_RECEIPT')
    require(all(file_sha(p)==file_sha(visible_ledgers/p.relative_to(visible)) for p in (visible/'ws2-a-prime').rglob('receipt.json')),
            'PULLED_LEDGER_RECEIPTS_EQUAL_COLLECTED')
    imports,blends=isolated_replay(notebook,{'qual39':qual[2],'visible3':vis[2]})
    return imports,{'qual39':compare('qual39',qual[0],qual[1],qual[2],blends['qual39'],Path(qual_result)/'visible-predictions.csv'),
        'visible3':compare('visible3',vis[0],vis[1],vis[2],blends['visible3'],visible/'submission.csv')}


def build_probe(*,rehearsal_build,rehearsal_result,destination,slug,user_waiver,failed_receipt_sha256):
    require(user_waiver==WAIVER,'EXPLICIT_NAMED_APRIME_W_WAIVER_REQUIRED')
    require(slug==SLUG,'SEPARATELY_NAMED_ENTRY_SLUG')
    require(W.RAPTOR_PER_FINDING==EXPECTED_SHARES and W.RAPTOR_DEFAULT==.5 and W.VERSION=='ws2-a-prime-r224-v2w','WS2W_PROFILE')
    dest=Path(destination)
    require(not dest.exists() and any(dest.resolve().is_relative_to(x/'outputs') for x in (PROJECT,MAIN)),'FRESH_IGNORED_BUILD')
    pack,aprime_code,aprime_meta,checked=admit(rehearsal_build,rehearsal_result,user_waiver,failed_receipt_sha256)
    entry=entry_pack(pack,checked);code=bootstrap(entry)
    require_bootstrap_preservation(code,aprime_code)
    meta=metadata(aprime_meta)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},
        'cells':[{'id':'aprime-w-probe-waiver-entry','cell_type':'code','metadata':{},'execution_count':None,
                  'outputs':[],'source':code.splitlines(True)}]}
    raw=json.dumps(nb,indent=2)+'\n';require(len(raw.encode())<900000,'NOTEBOOK900K_CAP')
    dest.mkdir(parents=True);(dest/'notebook.ipynb').write_text(raw);save(dest/'kernel-metadata.json',meta)
    imports,replayed=replay(dest/'notebook.ipynb')
    receipt={'status':'BOUND_PROJECTION_WAIVED_PROBE_BUILD','mode':'entry','builder_version':BUILDER_VERSION,
        'version':W.VERSION,'qualification_pass':False,'competitive_result':False,'runtime_checks_exercised':False,
        'waiver':entry['cfg']['projection_probe_waiver'],'blend_profile':entry['cfg']['ws2w_blend'],
        'base_entry':entry['cfg']['ws2w_base_entry'],
        'builder_sha256':file_sha(__file__),'frozen_aprime_builder_sha256':file_sha(PROBE.__file__),
        'frozen_builder_sha256':file_sha(BASE.__file__),
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        'binding':entry['cfg']['ws2_binding'],'r224_state_pin':entry['cfg']['r224_state_pin'],
        'payload_sha256':digest(entry),
        'source_closure':'A-prime payload sources byte-identical plus knee/public_ws2w_entry.py',
        'new_source_sha256':{NEW_SOURCE:file_sha(PROJECT/NEW_SOURCE)},
        'source_hashes':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in entry['sources'].items()},
        'rehearsal_verification':checked,'fresh_isolated_entry_import':imports,'blend_replay':replayed,
        'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},
        'source_files':len(entry['sources']),'notebook_bytes':len(raw.encode())}
    save(dest/'build-receipt.json',receipt)
    save(dest/'MANIFEST.json',{'status':receipt['status'],'payload_sha256':receipt['payload_sha256'],
        'files':{n:{'sha256':file_sha(dest/n),'bytes':(dest/n).stat().st_size}
                 for n in ('notebook.ipynb','kernel-metadata.json','build-receipt.json')}})
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rehearsal-build','rehearsal-result','destination','slug','user-waiver','failed-receipt-sha256'):
        p.add_argument('--'+name,required=True)
    r=build_probe(**vars(p.parse_args()))
    print(json.dumps({k:r[k] for k in ('status','version','payload_sha256','files','qualification_pass','source_files',
                                        'notebook_bytes','blend_replay')}))
