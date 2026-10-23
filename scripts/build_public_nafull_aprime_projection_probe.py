"""Build one separately named, receipt-bound A‴ projection-waived ENTRY packet.

The r3 rehearsal stays an ordinary FAIL. Its exact failed receipt, sealed packet,
na-full checkpoint/CPU binding and Codex failure acceptance are replayed. Every
original non-projection gate runs from the archived checker; only the ordinary
8.3 h conservative projection bound is waived. Actual elapsed < 9 h, all-39
coverage, recovery/ranks/CSV, frozen R7/R224 sources and image-only inference
remain enforced. Validator code here is never shipped in the entry closure.
"""
import argparse
import ast
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_public_nafull_aprime as B
from scripts.build_public_ws2_entry import PRIVATE_BINDING,PRIVATE_CFG
from knee import public_nafull_aprime as PROFILE
from knee import public_nafull_aprime_runtime as RT
from knee import public_ws2_checkpoint as C224
from knee import public_ws2_entry as WS2
from knee.public_s6_qualification import LABELS,canonical,digest,file_sha,require,save

PROJECT=B.PROJECT;MAIN=B.MAIN
WAIVER='a3prime-projection-20261016'
SLUG='nafull-aprime-probe-waiver-20261017'
FAILED_RECEIPT_SHA='97c8a8adbc1c62bc31003348c2dc8433325c1dddd3a088ed3a8fb1289acaccbc'
FAILURE={'type':'ValueError','detail':'COMBINED_PROJECTION_AND_ACTUAL_LIMIT'}
R3={'notebook':'1d7089e0df4283a45108d27c1ca5cc42604bdcaf49d1a1592b38aec519a519ce',
    'payload':'68cae8745bfeddb449b541347fc371a083ce92966fee56076d61722ad4d15b98',
    'build_receipt':'d716b74ec1f7fee3f1ddd428417597b446b2bc1f48b869b805d3bab4283d7281',
    'manifest':'48bb700a99d873fc8929d27245b428ae262fde6d05d4fae7bc1337c1309b29b0',
    'kernel':'ozgung/nafull-aprime-rehearsal-r3-20261016','source_revision':'996d41050ef9cc85fd803fe6656f9dc89e83107e'}
ACCEPTANCE=MAIN/'outputs/analysis/na-full-aprime-rehearsal-20261016/r3-real/failure-acceptance.json'
ACCEPTANCE_SHA='9a4e70ff48dab145e7cc55be2e3842b3ddf8a423ad1efe033577814c75605096'
AUDIT_SHA='3ae25102ebd3ec317d0aa513599ce684d5daaab754f5d5b424dab1d7c34604c3'
# Frozen arm states equal the archived A′ entry pins (R7 pilot, R224) and the na-full export.
FROZEN_R7_STATE='f90ddf4bb4f58217cd164cff128126286f3a65816582ebf5333f8744b2bde790'
FROZEN_R224_STATE='b24081b46e3540e84d8e0f8c6b2490af44a5f112901dadd23be47d530d451da5'
REFERENCES=('nafull_reader_reference','r7_input_reference','r7_probability_reference',
            'r224_input_reference','r224_probability_reference')
WORKER_PRIVATE=('probes','old_sentinels','raptor62_reference')
CHECKER_SOURCES=('knee/public_ws2_entry.py','knee/public_nafull_aprime_runtime.py','knee/public_nafull_aprime.py',
    'knee/public_nafull_aprime_recovery_adapter.py','knee/public_r7_recovery_entry.py','knee/public_s6_recovery_entry.py',
    'knee/public_s6_recovery_b0.py','knee/public_s6_recovery_b0prime.py','knee/public_ws2_recovery.py',
    'knee/public_r7_recovery.py','knee/public_ws2_checkpoint.py',
    'knee/public_nafull_raptor_input.py','knee/public_nafull_reference.py')
AUTHORIZATION={'reported_by':'Claude',
    'source':'docs/archive/handoff/claude-astra-handoff.md and docs/experiments/na-full-aprime-rehearsal-20261016.md, 2026-10-16 22:10 UTC; Codex condition audit 22:45 UTC',
    'reported_user_rule':'Waive if only the estimate fails','reported_choice_utc':'2026-10-16 about 21:55 UTC',
    'independently_verified_direct_user_message':False,
    'scope':'one A-triple-prime probe bound to r3 failed receipt; waive only the ordinary 8.3h conservative projection; '
            'actual nine-hour limit and every other gate unchanged; no other receipt, model or gate inherits it'}


def read(path):return json.loads(Path(path).read_bytes())


def unpack_notebook(path):
    """Decode one single-cell packet notebook; the embedded payload hash must match."""
    import base64,lzma
    nb=read(path);require(len(nb['cells'])==1 and nb['cells'][0]['cell_type']=='code','SINGLE_CODE_CELL')
    code=''.join(nb['cells'][0]['source']);tree=ast.parse(code)
    values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1
            and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('encoded','expected_payload')}
    require(set(values)=={'encoded','expected_payload'},'PAYLOAD_LITERALS')
    pack=json.loads(lzma.decompress(base64.b64decode(values['encoded'],validate=True)))
    require(digest(pack)==values['expected_payload'],'EMBEDDED_PAYLOAD_HASH')
    return pack,code


def gate(tree,name):
    found=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='require'
           and len(n.args)==2 and isinstance(n.args[1],ast.Constant) and n.args[1].value==name]
    require(len(found)==1,'EXACT_GATE_SITE_'+name)
    return found[0]


def r3_packet(build_root,cpu_receipt):
    """Exact sealed r3 packet: deterministic builder replay plus pinned bytes."""
    root=Path(build_root)
    require(file_sha(root/'notebook.ipynb')==R3['notebook'],'EXACT_R3_NOTEBOOK')
    require(file_sha(root/'build-receipt.json')==R3['build_receipt'] and file_sha(root/'MANIFEST.json')==R3['manifest'],
            'EXACT_R3_BUILD_RECEIPTS')
    prepared=B.verify_build(root,cpu_receipt)
    pack,code=unpack_notebook(root/'notebook.ipynb')
    expected,expected_code=B.packet(cpu_receipt)
    require(digest(pack)==digest(expected)==R3['payload']==prepared['payload_sha256'] and code==expected_code,
            'EXACT_R3_PAYLOAD')
    require(read(root/'kernel-metadata.json')==B.metadata(R3['kernel'].split('/')[1])
            and read(root/'build-receipt.json')['source_revision']==R3['source_revision'],'EXACT_R3_METADATA_REVISION')
    return pack,code,prepared


def require_packet_identity(pack):
    """Model, checkpoint, source and checker bindings that the waiver may not change."""
    require(digest(pack)==R3['payload'],'EXACT_R3_PAYLOAD')
    cfg=pack['cfg'];worker=cfg['nafull_worker_cfg'];manifest=worker['public_manifest']
    require(cfg['mode']=='rehearsal' and manifest.get('checkpoint_sha256')==PROFILE.CHECKPOINT['sha256']
            and manifest.get('checkpoint_file')==PROFILE.CHECKPOINT['file']==worker['control_checkpoint']
            and manifest.get('weight_dataset')==PROFILE.CHECKPOINT['dataset']
            and worker['control_state_sha256']==PROFILE.CHECKPOINT['state_sha256']
            and worker['control_grid_sha256']==PROFILE.CONSTRUCTOR_GRID_SHA256
            and cfg['nafull_base_binding']['qualified_state']['old_state']==PROFILE.CHECKPOINT['state_sha256'],
            'EXACT_NAFULL_CHECKPOINT_BINDING')
    ws=cfg['ws2_binding']
    require(ws['crop224_checkpoint']['sha256']==C224.CHECKPOINT_SHA256 and ws['version']==PROFILE.VERSION
            and ws['weights']=={'old320':.325,'crop224':.175,'r7':.5}
            and ws['frozen_r7_binding_sha256']==ws['nafull_base_binding_sha256']==digest(cfg['nafull_base_binding']),
            'EXACT_R224_R7_BLEND_BINDING')
    for name in CHECKER_SOURCES:
        require(file_sha(PROJECT/name)==hashlib.sha256(pack['sources'][name].encode()).hexdigest(),
                'LOCAL_CHECKER_EQUALS_PACKED_'+name)


def waived_contract(ws2):
    """Clone the adapted A‴ WS2 checker with two exact AST edits; nothing global changes.

    The status clause accepts the saved FAIL/ValueError instead of PASS. The
    combined gate requires the failed projection to exceed 8.3 h instead of not
    exceeding it, keeping its unmodified actual 0 < elapsed < 9 h clause.
    """
    wrapper=ws2.require_rehearsal;names=wrapper.__code__.co_freevars
    require('archived_check' in names,'EXACT_NAFULL_WS2_WRAPPER')
    archived=wrapper.__closure__[names.index('archived_check')].cell_contents
    original=inspect.getsource(WS2.require_rehearsal)
    marker="original=old_rows[s['id']]['input']"
    require(original.count(marker)==1,'EXACT_WS2_R224_OLD_INPUT_GATE')
    altered=original.replace(marker,"original=binding['r224_input_reference'][s['id']]")
    def compiled(tree,name,filename):
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef));fn.name=name
        namespace=dict(archived.__globals__)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),filename,'exec'),namespace)
        return namespace[name]
    replay=compiled(ast.parse(altered),'_ws2_nafull_require_rehearsal','<nafull-adapted-ws2-contract>')
    require(replay.__code__==archived.__code__,'WAIVER_STARTS_FROM_EXACT_ADAPTED_CHECKER')
    tree=ast.parse(altered)
    status=gate(tree,'FRESH_FULL_REHEARSAL_BINDING').args[0]
    require(isinstance(status,ast.BoolOp) and ast.unparse(status.values[0])=="r.get('status') == 'PASS'",'EXACT_STATUS_CLAUSE')
    status.values[0:1]=[ast.parse("r.get('status') == 'FAIL'",mode='eval').body,
                        ast.parse("r.get('failure_type') == 'ValueError'",mode='eval').body]
    limit=gate(tree,'COMBINED_PROJECTION_AND_ACTUAL_LIMIT').args[0]
    require(isinstance(limit,ast.BoolOp) and len(limit.values)==2
            and ast.unparse(limit.values[0])=="0 < p['conservative_seconds'] <= PROJECTION_GATE_SECONDS"
            and ast.unparse(limit.values[1])=="0 < r['elapsed_seconds'] < MAX_SECONDS",'EXACT_PROJECTION_VS_ACTUAL_CLAUSES')
    limit.values[0]=ast.parse("PROJECTION_GATE_SECONDS < p['conservative_seconds'] < math.inf",mode='eval').body
    waived=compiled(tree,'_ws2_nafull_projection_waived','<a3prime-exact-projection-waiver>')
    cells=tuple(types.CellType(waived) if name=='archived_check' else cell
                for name,cell in zip(names,wrapper.__closure__))
    return types.FunctionType(wrapper.__code__,wrapper.__globals__,'require_a3prime_projection_waived',
                              wrapper.__defaults__,cells)


def replay(root,pack,r=None):
    """Replay every non-projection check on saved r3 evidence; r may be an in-memory copy."""
    import numpy as np
    root=Path(root);require_packet_identity(pack)
    cfg=pack['cfg'];binding=cfg['ws2_binding']
    base_path=root/'base'/'receipt.json';csv_path=root/'visible-predictions.csv'
    r=read(root/'receipt.json') if r is None else r;base=read(base_path)
    require(r.get('binding')==binding and base.get('binding')==cfg['nafull_base_binding'],'EXACT_R3_BINDINGS')
    p=r.get('projection') or {}
    seconds=p.get('conservative_seconds');elapsed=r.get('elapsed_seconds')
    require(type(seconds) is float and math.isfinite(seconds) and seconds>WS2.PROJECTION_GATE_SECONDS
            and p.get('gate_pass') is False and p.get('gate_seconds')==WS2.PROJECTION_GATE_SECONDS,
            'EXACT_PROJECTION_ONLY_FAILURE')
    require(type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<elapsed<WS2.MAX_SECONDS,'ACTUAL_NINE_HOUR_LIMIT')
    facade=RT.make_base_facade(cfg['nafull_worker_cfg'],rehearsal=True);ws2=RT.make_ws2_facade(facade)
    waived_contract(ws2)(r,binding,base_path,csv_path)
    # The ordinary checker must still reject the saved receipt, and with the
    # transient pre-error PASS status it must fail at exactly the projection gate.
    for status,expected in ((r.get('status'),'FRESH_FULL_REHEARSAL_BINDING'),('PASS',FAILURE['detail'])):
        transient={**r,'status':status}
        try:ws2.require_rehearsal(transient,binding,base_path,csv_path)
        except ValueError as error:require(str(error)==expected,'ORDINARY_CHECK_FAILS_ONLY_AT_PROJECTION')
        else:raise ValueError('ORDINARY_QUALIFICATION_UNEXPECTEDLY_PASSED')
    ids=read(root/'base'/'assignment.json')['ids']
    require(len(ids)==len(set(ids))==39 and ids==base['binding']['s6_probe_ids']+base['binding']['old_historical_ids']+base['visible_ids'],
            'EXACT_FROZEN39_ORDER')
    arms={}
    for name,arm,folder,reports in (('old320','old',root/'base',base['workers']['old']),
            ('r7','s6',root/'base',base['workers']['s6']),('crop224','crop224',root,r['crop224_workers'])):
        require(len(reports)==2,'TWO_BOARD_'+arm)
        for board,worker in enumerate(reports):
            require(read(folder/f'board{board}-{arm}'/arm/'receipt.json')==worker and worker.get('status')=='PASS'
                    and worker.get('state_before')==worker.get('state_after') and worker.get('zero_failed') is True
                    and len(worker['records'])==worker['completed']==len(worker['assigned'])
                    and all(row['status']=='genuine' for row in worker['records']),'SAVED_GENUINE_WORKER_'+arm)
        values,missing=ws2.merged_arm(ids,reports,folder,arm)
        require(values.shape==(39,12) and not missing.any() and np.isfinite(values).all()
                and ((values>=0)&(values<=1)).all(),'ALL39_GENUINE_'+arm)
        arms[name]=(values,missing)
    states={'nafull':[w['state_before']['state'] for w in base['workers']['old']],
            'r7':[w['state_before'] for w in base['workers']['s6']],
            'r224':[w['state_before'] for w in r['crop224_workers']]}
    require(states=={'nafull':[PROFILE.CHECKPOINT['state_sha256']]*2,'r7':[FROZEN_R7_STATE]*2,'r224':[FROZEN_R224_STATE]*2},
            'EXACT_THREE_ARM_STATES')
    rec=ws2.REC.recover(*(arms[a][0] for a in ws2.REC.ARMS),*(arms[a][1] for a in ws2.REC.ARMS))
    frozen=ws2.R7.recover(arms['old320'][0],arms['r7'][0],arms['old320'][1],arms['r7'][1])
    require(rec['union_recovered']==0 and np.array_equal(frozen['control_rank'],rec['ranks']['old320'])
            and np.array_equal(frozen['s6_rank'],rec['ranks']['r7']),'COMPLETE_POPULATION_RANKS')
    hashes={'old_rank_sha256':hashlib.sha256(frozen['control_rank'].tobytes()).hexdigest(),
        'r7_rank_sha256':hashlib.sha256(frozen['s6_rank'].tobytes()).hexdigest(),
        'blend_sha256':hashlib.sha256(frozen['blend'].tobytes()).hexdigest()}
    require(r.get('baseline_full_population')==hashes,'SAVED_BASELINE_OUTPUT_HASHES')
    with csv_path.open(newline='') as handle:rows=list(csv.reader(handle))
    require(rows[0]==[binding['uid']]+LABELS and [x[0] for x in rows[1:]]==base['visible_ids'],'CSV_SCHEMA_ORDER')
    actual=np.array([[float(x) for x in row[1:]] for row in rows[1:]])
    expected=rec['blend'][[ids.index(s) for s in base['visible_ids']]]
    require(actual.shape==(3,12) and np.isfinite(actual).all() and float(np.max(np.abs(actual-expected)))<=5e-10,
            'RECOMPUTED_CSV_MATCH')
    return {'receipt_sha256':file_sha(root/'receipt.json'),'base_receipt_sha256':file_sha(base_path),
        'base_csv_sha256':file_sha(root/'base'/'visible-predictions.csv'),'csv_sha256':file_sha(csv_path),
        'qualified_states':{'nafull':states['nafull'][0],'r7':states['r7'][0],'r224':states['r224'][0]},
        'projection':p,'elapsed_seconds':elapsed,'gate_over_seconds':seconds-WS2.PROJECTION_GATE_SECONDS,
        'ordinary_failure_reproduced':FAILURE['detail'],'all_non_projection_checks_pass':True,
        'all_six_ledgers_recomputed':True,'complete_population_rank_replay':True,
        'visible_csv_max_abs':float(np.max(np.abs(actual-expected)))}


def tree_hashes(root):
    return {str(p.relative_to(root)):file_sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}


def admit(*,rehearsal_build,rehearsal_result,cpu_receipt,user_waiver,failed_receipt_sha256):
    """Named, exact-receipt-bound admission; never writes to the r3 evidence."""
    require(user_waiver==WAIVER,'EXPLICIT_NAMED_A3PRIME_WAIVER_REQUIRED')
    require(failed_receipt_sha256==FAILED_RECEIPT_SHA,'EXACT_NAMED_FAILED_RECEIPT_ARGUMENT')
    root=Path(rehearsal_result)
    require((root/'receipt.json').is_file() and file_sha(root/'receipt.json')==FAILED_RECEIPT_SHA,'EXACT_WAIVED_FAILED_RECEIPT')
    require(all(read(root/name)==FAILURE for name in ('failure-private.json','setup-failure-private.json')),
            'ONLY_PROJECTION_FAILURE_MAY_BE_WAIVED')
    require(file_sha(ACCEPTANCE)==ACCEPTANCE_SHA and file_sha(ACCEPTANCE.with_name('audit_failure.py'))==AUDIT_SHA,
            'EXACT_CODEX_FAILURE_ACCEPTANCE')
    accepted=read(ACCEPTANCE);before=tree_hashes(root)
    require(accepted['status']=='PROJECTION_ONLY_WAIVER_CONDITION_MET' and accepted['ordinary_qualification_pass'] is False
            and accepted['original_failed_receipt_sha256']==FAILED_RECEIPT_SHA and accepted['notebook_sha256']==R3['notebook']
            and accepted['payload_sha256']==R3['payload'] and accepted['only_projection_condition_isolated_in_memory']==[FAILURE['detail']]
            and before==accepted['result_files'],'EXACT_ACCEPTED_R3_RESULT_TREE')
    pack,code,prepared=r3_packet(rehearsal_build,cpu_receipt)
    checked=replay(root,pack)
    require(tree_hashes(root)==before and read(root/'receipt.json')['status']=='FAIL','FAILED_EVIDENCE_UNCHANGED')
    return pack,code,prepared,{**checked,'acceptance_sha256':ACCEPTANCE_SHA,'audit_sha256':AUDIT_SHA,
        'result_files_sha256':digest(before)}


def waiver_record(checked):
    return {'id':WAIVER,'authorization':AUTHORIZATION,'failed_receipt_sha256':FAILED_RECEIPT_SHA,
        'status':'USER_REPORTED_PROJECTION_WAIVER_NOT_QUALIFICATION_PASS','qualification_pass':False,
        'waived_gate':'ordinary 8.3h conservative projection component of COMBINED_PROJECTION_AND_ACTUAL_LIMIT',
        'projection':checked['projection'],'gate_over_seconds':checked['gate_over_seconds'],
        'actual_elapsed_seconds':checked['elapsed_seconds'],'actual_limit_seconds':WS2.MAX_SECONDS,
        'r3':{'kernel':R3['kernel'],'notebook_sha256':R3['notebook'],'payload_sha256':R3['payload']},
        'codex_acceptance_sha256':ACCEPTANCE_SHA}


def entry_pack(pack,checked):
    """Same 251 sources; entry mode with qualified state pins and no private fixtures."""
    out=json.loads(canonical(pack));cfg=out['cfg']
    require(cfg['mode']=='rehearsal','REHEARSAL_TEMPLATE')
    ids=set(cfg['probe_order'])|set(cfg['binding']['visible_ids'])
    require(len(ids)==39,'FROZEN39_REGISTRY')
    cfg['mode']='entry'
    for key in PRIVATE_CFG:cfg.pop(key,None)
    cfg['binding']={k:v for k,v in cfg['binding'].items() if k not in PRIVATE_BINDING}
    states=checked['qualified_states'];waiver=waiver_record(checked)
    base={k:v for k,v in cfg['nafull_base_binding'].items() if k not in (*PRIVATE_BINDING,*REFERENCES)}
    base.update(rehearsal_receipt_sha256=checked['base_receipt_sha256'],rehearsal_csv_sha256=checked['base_csv_sha256'],
                qualified_r7_state=states['r7'],waived_rehearsal_base_binding_sha256=digest(cfg['nafull_base_binding']))
    cfg['nafull_base_binding']=base
    for key in WORKER_PRIVATE:cfg['nafull_worker_cfg'].pop(key)
    ws={k:v for k,v in cfg['ws2_binding'].items() if k not in REFERENCES}
    ws.update(waived_rehearsal_ws2_binding_sha256=digest(cfg['ws2_binding']),projection_probe_waiver=waiver)
    cfg['ws2_binding']=ws
    cfg['pilot_state_sha256']=states['r7'];cfg['r224_state_pin']=states['r224']
    text=canonical(cfg).decode()
    require(not any(sid in text for sid in ids) and not any(k in cfg for k in PRIVATE_CFG)
            and not any(k in cfg['binding'] or k in base for k in PRIVATE_BINDING)
            and not any(k in ws or k in base for k in REFERENCES)
            and not any(k in cfg['nafull_worker_cfg'] for k in WORKER_PRIVATE),'PURE_ENTRY_PRIVATE_FIXTURES_STRIPPED')
    require(out['sources']==pack['sources'] and out['codecs']==pack['codecs'],'EXACT_R3_SOURCE_CLOSURE')
    return out


def require_bootstrap_preservation(code,r3_code):
    """Entry bootstrap equals r3 except for the two payload literals."""
    def body(text):
        tree=ast.parse(text)
        return [ast.dump(n) for n in tree.body if not (isinstance(n,ast.Assign) and len(n.targets)==1
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('encoded','expected_payload'))]
    require(body(code)==body(r3_code) and code.count('signal.alarm(9*3600)')==1,'EXACT_R3_BOOTSTRAP_AND_NINE_HOUR_ALARM')


IMPORT_CHILD=r'''
import importlib,json,sys
from pathlib import Path
project=Path(sys.argv[1]).resolve();cfg=json.loads(Path(sys.argv[2]).read_text());sys.path.insert(0,str(project))
modules=['knee.public_nafull_aprime_entry','knee.public_nafull_aprime_runtime','knee.public_ws2_entry',
 'knee.public_r7_recovery_entry','knee.public_s6_recovery_b0','knee.public_s6_recovery_b0prime',
 'knee.public_s6_recovery_b0_rehearsal','knee.public_nafull_raptor_input','knee.public_nafull_reference']
for name in modules:
    assert Path(importlib.import_module(name).__file__).resolve().is_relative_to(project),name
from knee import public_nafull_aprime_runtime as runtime
assert cfg['mode']=='entry' and cfg['pilot_state_sha256'] and cfg['r224_state_pin']
runtime.nafull_profile_gate(None,cfg['nafull_worker_cfg']['public_manifest'])
base=runtime.make_base_facade(cfg['nafull_worker_cfg'],rehearsal=False)
ws2=runtime.make_ws2_facade(base)
assert base.rehearsal_hook is None and callable(ws2.run) and base.ENTRY_MODULE=='knee.public_nafull_aprime_entry'
assert all(not name.startswith('knee.') or Path(m.__file__).resolve().is_relative_to(project)
           for name,m in list(sys.modules.items()) if getattr(m,'__file__',None))
print(json.dumps({'imports':modules,'entry_facades':True,'all_from_unpacked_project':True}))
'''


def entry_import_check(notebook):
    """Fresh `python -I` child imports the unpacked entry; the checkout is not on its path."""
    pack,_=unpack_notebook(notebook)
    with tempfile.TemporaryDirectory(prefix='a3prime-entry-import-') as folder:
        root=Path(folder);project=root/'project'
        for name,content in pack['sources'].items():
            path=Path(name);require(not path.is_absolute() and '..' not in path.parts,'SAFE_SOURCE_PATH')
            path=project/path if name.startswith('knee/') else project/'source'/path
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content.encode())
        (root/'cfg.json').write_text(json.dumps(pack['cfg']));(root/'child.py').write_text(IMPORT_CHILD)
        proc=subprocess.run([sys.executable,'-I',str(root/'child.py'),str(project),str(root/'cfg.json')],
                            cwd=root,capture_output=True,text=True)
    require(proc.returncode==0,'FRESH_ISOLATED_ENTRY_IMPORT: '+proc.stderr[-1500:])
    return {'status':'PASS','python':sys.version.split()[0],'isolated':True,'payload_sha256':digest(pack),
            **json.loads(proc.stdout.splitlines()[-1]),'gpu_exercised':False,'uses_real_images':False}


def build_probe(*,rehearsal_build,rehearsal_result,cpu_receipt,destination,slug,user_waiver,failed_receipt_sha256):
    require(user_waiver==WAIVER,'EXPLICIT_NAMED_A3PRIME_WAIVER_REQUIRED')
    require(slug==SLUG,'SEPARATELY_NAMED_ENTRY_SLUG')
    dest=Path(destination)
    require(not dest.exists() and any(dest.resolve().is_relative_to(x/'outputs') for x in (PROJECT,MAIN)),'FRESH_IGNORED_BUILD')
    pack,r3_code,prepared,checked=admit(rehearsal_build=rehearsal_build,rehearsal_result=rehearsal_result,
        cpu_receipt=cpu_receipt,user_waiver=user_waiver,failed_receipt_sha256=failed_receipt_sha256)
    entry=entry_pack(pack,checked);code=B.bootstrap(entry)
    require_bootstrap_preservation(code,r3_code)
    meta=B.metadata(slug)
    require(meta['id']=='ozgung/'+SLUG and meta['title']==SLUG and meta['is_private'] is True and meta['enable_internet'] is False
            and meta['enable_gpu'] is True and meta['machine_shape']=='NvidiaTeslaT4' and 'timeout' not in meta
            and meta['dataset_sources']==read(Path(rehearsal_build)/'kernel-metadata.json')['dataset_sources'],
            'PRIVATE_OFFLINE_T4_NO_TIMEOUT')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},
        'cells':[{'id':'nafull-aprime-probe-waiver-entry','cell_type':'code','metadata':{},'execution_count':None,
                  'outputs':[],'source':code.splitlines(True)}]}
    raw=json.dumps(nb,indent=2)+'\n';require(len(raw.encode())<900000,'NOTEBOOK900K_CAP')
    dest.mkdir(parents=True);(dest/'notebook.ipynb').write_text(raw);save(dest/'kernel-metadata.json',meta)
    imports=entry_import_check(dest/'notebook.ipynb')
    receipt={'status':'BOUND_PROJECTION_WAIVED_PROBE_BUILD','mode':'entry','qualification_pass':False,
        'competitive_result':False,'runtime_checks_exercised':False,'waiver':entry['cfg']['ws2_binding']['projection_probe_waiver'],
        'builder_sha256':file_sha(__file__),'frozen_builder_sha256':file_sha(B.__file__),
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        'r3_preparation':{'notebook_sha256':R3['notebook'],'payload_sha256':prepared['payload_sha256'],
            'build_receipt_sha256':R3['build_receipt'],'manifest_sha256':R3['manifest'],'source_revision':R3['source_revision']},
        'nafull_checkpoint':PROFILE.CHECKPOINT,'nafull_cpu_receipt_sha256':file_sha(cpu_receipt),
        'binding':entry['cfg']['ws2_binding'],'base_binding_sha256':digest(entry['cfg']['nafull_base_binding']),
        'state_pins':{'nafull':entry['cfg']['nafull_worker_cfg']['control_state_sha256'],
            'r7':entry['cfg']['pilot_state_sha256'],'r224':entry['cfg']['r224_state_pin']},
        'payload_sha256':digest(entry),'source_closure':'byte-identical to sealed r3 payload sources',
        'source_hashes':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in entry['sources'].items()},
        'rehearsal_verification':checked,'fresh_isolated_entry_import':imports,
        'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},
        'source_files':len(entry['sources']),'notebook_bytes':len(raw.encode())}
    save(dest/'build-receipt.json',receipt)
    save(dest/'MANIFEST.json',{'status':receipt['status'],'payload_sha256':receipt['payload_sha256'],
        'files':{n:{'sha256':file_sha(dest/n),'bytes':(dest/n).stat().st_size}
                 for n in ('notebook.ipynb','kernel-metadata.json','build-receipt.json')}})
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rehearsal-build','rehearsal-result','cpu-receipt','destination','slug','user-waiver','failed-receipt-sha256'):
        p.add_argument('--'+name,required=True)
    r=build_probe(**vars(p.parse_args()))
    print(json.dumps({k:r[k] for k in ('status','payload_sha256','files','qualification_pass','source_files','notebook_bytes')}))
