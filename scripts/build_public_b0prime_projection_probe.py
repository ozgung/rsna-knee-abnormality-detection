"""One exact, explicitly labelled B0′ projection-waived knowledge probe.

Original qualification remains FAIL. Inference and its actual nine-hour limit
are unchanged; this validator/builder is never shipped in the entry closure.
"""
import argparse
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_public_b0prime as BASE
from knee import public_s6_recovery_b0 as B0
from knee import public_s6_recovery_b0prime as P
from knee.public_s6_qualification import digest,file_sha,require,save

WAIVER='b0prime-projection-20261015'
FAILED_RECEIPT_SHA='ef9838a483d02c9e24e7a70553871e4a1d30541949a8f3d4525f0903344abbfe'
AUTHORIZATION={'reported_by':'Claude','source':'docs/claude-astra-handoff.md, 2026-10-15 18:18 UTC',
 'reported_user_choice':'Submit anyway','reported_choice_utc':'2026-10-15 about 17:20 UTC',
 'independently_verified_direct_user_message':False,
 'scope':'one B0prime knowledge probe; waive only projected nine-hour and 8.3h gates; actual nine-hour limit unchanged; not a final candidate'}
FROZEN={'scripts/build_public_b0prime.py':'8a779240c2be032ae4b7e57bb8406120b0b6060b97145e09186188dfa2cbcf21',
 'knee/public_s6_recovery_b0prime.py':'6aede849c9b5afc89e2756b349de3c70a100be1a28f76cdd61c1d60e091e91ba',
 'knee/public_s6_recovery_b0.py':BASE.FROZEN['knee/public_s6_recovery_b0.py']}


def gate(tree,name):
    found=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
           and n.func.id=='require' and len(n.args)==2 and isinstance(n.args[1],ast.Constant) and n.args[1].value==name]
    require(len(found)==1,'EXACT_GATE_SITE_'+name)
    return found[0]


def waived_contract():
    """Narrow AST changes to the frozen validators, in an isolated namespace.

    Accept exact failed status instead of PASS. Remove projected upper bounds
    only; retain actual elapsed<9h, complete binding and all mechanical checks.
    No monkey-patch, status mutation or change to the ordinary validator.
    """
    for name,pin in FROZEN.items():require(file_sha(BASE.PROJECT/name)==pin,'EXACT_FROZEN_VALIDATOR_'+name)
    tree=ast.parse(inspect.getsource(B0.require_rehearsal));body=tree.body[0].body
    require(ast.unparse(body[-2])=="contract = binding['old_qualification']['raptor62']"
            and isinstance(body[-1],ast.For) and ast.unparse(body[-1].iter)=="r['visible_ids']"
            and 'RAPTOR62_VISIBLE_RECEIPT_PARITY' in ast.unparse(body[-1]),'EXACT_B0PRIME_READER_GATE')
    body[-2:]=[ast.Expr(ast.Call(ast.Name('require_visible_decode_counts',ast.Load()),
                     [ast.Name('r',ast.Load()),ast.Name('binding',ast.Load())],[]))]
    status=gate(tree,'PASS_EXACT_REHEARSAL_REQUIRED').args[0]
    require(isinstance(status,ast.BoolOp) and ast.unparse(status.values[0])=="r.get('status') == 'PASS'",'EXACT_STATUS_CLAUSE')
    status.values[0].comparators=[ast.Constant('FAIL')]
    projected=gate(tree,'NINE_HOUR_PROJECTION_GATE').args[0]
    require(isinstance(projected,ast.BoolOp) and len(projected.values)==2
            and ast.unparse(projected.values[0])=="0 < r.get('projection', {}).get('conservative_seconds', math.inf) < MAX_SECONDS"
            and ast.unparse(projected.values[1])=="0 < r.get('elapsed_seconds', math.inf) < MAX_SECONDS",'EXACT_PROJECTION_VS_ACTUAL_CLAUSES')
    projected.values[0]=ast.parse("math.isfinite(r['projection']['conservative_seconds']) and 0 < r['projection']['conservative_seconds']",mode='eval').body
    tree.body[0].name='_require_b0prime'
    namespace=dict(vars(P));exec(compile(ast.fix_missing_locations(tree),'<b0prime-exact-projection-waiver>','exec'),namespace)
    outer=ast.parse(inspect.getsource(P.require_rehearsal))
    condition=gate(outer,'B0PRIME_8_3H_PROJECTION')
    require(ast.unparse(condition.args[0])=='math.isfinite(seconds) and 0 < seconds <= QUALIFICATION_SECONDS','EXACT_8_3H_GATE')
    condition.args[0]=ast.parse('math.isfinite(seconds) and 0 < seconds',mode='eval').body
    exec(compile(ast.fix_missing_locations(outer),'<b0prime-exact-8.3h-waiver>','exec'),namespace)
    return namespace['require_rehearsal']


def verify_probe_evidence(root,binding,user_waiver):
    require(user_waiver==WAIVER,'EXPLICIT_NAMED_B0PRIME_WAIVER_REQUIRED')
    root=Path(root)
    require(file_sha(root/'receipt.json')==FAILED_RECEIPT_SHA,'EXACT_WAIVED_FAILED_RECEIPT')
    r=json.loads((root/'receipt.json').read_bytes())
    require(json.loads((root/'failure-private.json').read_bytes())=={'type':'ValueError','detail':'NINE_HOUR_PROJECTION_GATE'}
        and r['status']=='FAIL' and r['failure_type']=='ValueError' and r['binding']==binding,'ONLY_EXACT_PROJECTION_FAILURE')
    require(math.isfinite(r['projection']['conservative_seconds'])
            and r['projection']['conservative_seconds']>P.MAX_SECONDS>P.QUALIFICATION_SECONDS
            and 0<r['elapsed_seconds']<P.MAX_SECONDS,'EXACT_FAILED_PROJECTION_ACTUAL_LIMIT')
    runtime=types.SimpleNamespace(**{**vars(P),'require_rehearsal':waived_contract()})
    verifier=types.FunctionType(BASE.verify_result.__code__,{**vars(BASE),'P':runtime},BASE.verify_result.__name__)
    checked=verifier(root,binding)
    require(file_sha(root/'receipt.json')==FAILED_RECEIPT_SHA,'FAILED_EVIDENCE_UNCHANGED')
    return checked


def build_probe(*,rehearsal_build,rehearsal_result,cpu_receipt,destination,slug,user_waiver):
    require(user_waiver==WAIVER,'EXPLICIT_NAMED_B0PRIME_WAIVER_REQUIRED')
    dest=Path(destination);require(not dest.exists() and dest.resolve().is_relative_to(BASE.PROJECT/'outputs'),'FRESH_IGNORED_BUILD')
    meta=BASE.metadata(slug)
    prepared=BASE.verify_preparation(rehearsal_build,cpu_receipt)
    checked=verify_probe_evidence(rehearsal_result,prepared['binding'],user_waiver)
    pack=BASE.packet(cpu_receipt);cfg=pack['cfg']
    require(cfg['binding']==prepared['binding'],'EXACT_FROZEN_INFERENCE_BINDING')
    waiver={'id':WAIVER,'authorization':AUTHORIZATION,'failed_receipt_sha256':FAILED_RECEIPT_SHA,
        'status':'USER_REPORTED_PROJECTION_WAIVER_NOT_QUALIFICATION_PASS','qualification_pass':False,
        'projection':checked['projection'],'actual_limit_seconds':P.MAX_SECONDS,'all_four_ledgers_recomputed':True}
    cfg['mode']='entry';cfg['binding']={k:v for k,v in cfg['binding'].items() if k not in (*BASE.PRIVATE,'development_windows')}
    cfg['binding']['projection_probe_waiver']=waiver
    for key in ('probes','probe_order','old_sentinels','s6_sentinels','development_windows','raptor62_reference'):cfg.pop(key,None)
    for name in ('knee/public_s6_recovery_rehearsal.py','knee/public_s6_recovery_b0_rehearsal.py',BASE.EXTENSION):pack['sources'].pop(name)
    original,_=BASE.unpack_notebook(Path(rehearsal_build)/'notebook.ipynb')
    require(all(original['sources'][n]==s for n,s in pack['sources'].items())
        and set(original['sources'])-set(pack['sources'])=={'knee/public_s6_recovery_rehearsal.py','knee/public_s6_recovery_b0_rehearsal.py',BASE.EXTENSION},'UNCHANGED_PRODUCTION_SOURCE_CLOSURE')
    require(not any(k in cfg for k in ('probes','probe_order','old_sentinels','s6_sentinels','development_windows','raptor62_reference'))
        and not any(k in cfg['binding'] for k in (*BASE.PRIVATE,'development_windows')),'PRIVATE_FIXTURES_STRIPPED')
    code=BASE.bootstrap(pack)
    require('signal.alarm(9*3600)' in code,'ACTUAL_NINE_HOUR_ALARM_PRESERVED')
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},
        'cells':[{'id':'b0prime-projection-waived-knowledge-probe','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    raw=json.dumps(nb,indent=2)+'\n';require(len(raw.encode())<900000,'NOTEBOOK900K_CAP')
    dest.mkdir(parents=True);(dest/'notebook.ipynb').write_text(raw);save(dest/'kernel-metadata.json',meta)
    r={'status':'BOUND_PROJECTION_WAIVED_PROBE_BUILD','mode':'entry','qualification_pass':False,'competitive_result':False,
       'builder_sha256':file_sha(__file__),'frozen_builder_sha256':file_sha(BASE.__file__),
       'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE.PROJECT,text=True).strip(),
       'binding':cfg['binding'],'waiver':waiver,'rehearsal_verification':checked,'payload_sha256':digest(pack),
       'source_hashes':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in pack['sources'].items()},
       'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},
       'source_files':len(pack['sources']),'notebook_bytes':len(raw.encode()),'runtime_checks_exercised':False}
    save(dest/'build-receipt.json',r);save(dest/'MANIFEST.json',{'status':r['status'],'payload_sha256':digest(pack),
       'files':{n:{'sha256':file_sha(dest/n),'bytes':(dest/n).stat().st_size} for n in (*r['files'],'build-receipt.json')}})
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('rehearsal-build','rehearsal-result','cpu-receipt','destination','slug','user-waiver'):p.add_argument('--'+name,required=True)
    r=build_probe(**vars(p.parse_args()))
    print(json.dumps({k:r[k] for k in ('status','files','payload_sha256','qualification_pass','notebook_bytes')}))
