"""Exercise worker imports from an unpacked immutable packet in a fresh process.

Optional CPU/reader checks use synthetic inputs only. No cloud call or training.
The checkout's knee package cannot satisfy missing bundled symbols (-I process).
"""
import argparse
import ast
import base64
import hashlib
import json
import lzma
from pathlib import Path
import subprocess
import sys
import tempfile

CHILD = r'''
import importlib,importlib.util,json,sys
from pathlib import Path
project=Path(sys.argv[1]).resolve();cfg=json.loads(Path(sys.argv[2]).read_text())
sys.path.insert(0,str(project))
if sys.argv[6]:sys.path.insert(1,sys.argv[6])
if sys.argv[4]:
    sys.path.insert(1,sys.argv[4])  # Only the selected package and its real metadata.
    import pydicom
modules=['knee.public_nafull_aprime_entry','knee.public_nafull_aprime_rehearsal',
 'knee.public_ws2_entry','knee.public_r7_recovery_entry','knee.public_r7_recovery_rehearsal',
 'knee.public_s6_recovery_b0','knee.public_s6_recovery_b0prime',
 'knee.public_s6_recovery_b0_rehearsal','knee.public_s6_recovery_b0prime_rehearsal']
for name in modules:
    module=importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(project),name
from knee import public_nafull_aprime_runtime as runtime
from knee import public_nafull_aprime as profile
from knee import reference as shared_reference,raptor_input as shared_reader
manifest=cfg['nafull_worker_cfg']['public_manifest']
runtime.nafull_profile_gate(None,manifest)  # Reproduces the real v1 missing-symbol failure.
base=runtime.make_base_facade(cfg['nafull_worker_cfg'],rehearsal=True)
runtime.make_ws2_facade(base)
result={'worker_entry_imports':modules,'all_imported_from_unpacked_project':True,
        'nafull_profile_gate':True,'private_reference_symbols':False,'cpu_strict_load':False,'reader_checks':False}
with runtime.old_source_modules():
    from knee import reference,raptor_input
    from knee.reference import (RAPTOR_WEIGHT_DATASET,RAPTOR_WEIGHT_NAME,infer_checked,
                                 load_definitions,load_model,reference_reader,validate_volume)
    assert reference is not shared_reference and raptor_input is not shared_reader
    assert all(callable(f) for f in (infer_checked,load_definitions,load_model,reference_reader,validate_volume))
    result['private_reference_symbols']=True
    public,decoder=load_definitions(cfg['nafull_worker_cfg']['public_source'],'',manifest)
    assert decoder is None
    runtime.nafull_profile_gate(public,manifest)
    if sys.argv[3]:
        import numpy as np,torch,timm
        torch.set_num_threads(2)
        from knee.public_fullfield import snapshot
        model=load_model(public,sys.argv[3],'cpu',manifest)
        before=snapshot(model)
        assert before['state']==profile.CHECKPOINT['state_sha256']
        p=np.asarray(infer_checked(public,model,[torch.zeros(1,3,384,384)],'cpu',mirror=False))
        assert p.shape==(12,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        assert snapshot(model)==before
        result.update(cpu_strict_load=True,cpu_forward_windows=1,cpu_finite=True,cpu_state_unchanged=True,
                      state_sha256=before['state'],torch=torch.__version__,timm=timm.__version__)
    if sys.argv[5]=='yes':
        from knee import public_s6_recovery_b0 as b0
        from knee.public_s6_recovery_b0_rehearsal import guarded_raptor_partial
        stress=b0.raptor_stress(public)
        b0.install_raptor_old_guards(public)
        guarded=guarded_raptor_partial(public)
        assert stress['case_count']==16 and guarded['case_count']==10
        result.update(reader_checks=True,synthetic_reader_cases=16,guarded_reader_cases=10)
from knee import reference,raptor_input
assert reference is shared_reference and raptor_input is shared_reader
result['shared_modules_restored']=True
# Both original R224 definition imports must also resolve in this unpacked closure.
shared_reference.load_definitions(cfg['public_source'],cfg['decoder_source'],cfg['public_manifest'])
result['original_r224_definition_imports']=True
print(json.dumps(result))
'''


def check(notebook, checkpoint=None, pydicom_source=None, reader_checks=False, dependency_path=None):
    nb=json.loads(Path(notebook).read_bytes());tree=ast.parse(''.join(nb['cells'][0]['source']))
    values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
            and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('encoded','expected_payload')}
    pack=json.loads(lzma.decompress(base64.b64decode(values['encoded'],validate=True)))
    payload_sha=hashlib.sha256(json.dumps(pack,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if payload_sha!=values['expected_payload']:raise ValueError('PAYLOAD_HASH')
    with tempfile.TemporaryDirectory(prefix='nafull-unpacked-check-') as directory:
        root=Path(directory);project=root/'project'
        for name,content in pack['sources'].items():
            path=Path(name)
            if path.is_absolute() or '..' in path.parts:raise ValueError('UNSAFE_SOURCE_PATH')
            path=(project/path) if name.startswith('knee/') else (project/'source'/path)
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content.encode())
        config=root/'config.json';config.write_text(json.dumps(pack['cfg']))
        child=root/'check.py';child.write_text(CHILD)
        borrowed=''
        if pydicom_source:
            origin=Path(pydicom_source).resolve()
            deps=root/'pydicom-only';deps.mkdir()
            (deps/'pydicom').symlink_to(origin.parent,target_is_directory=True)
            metadata=list(origin.parent.parent.glob('pydicom-*.dist-info'))
            if len(metadata)!=1:raise ValueError('PYDICOM_METADATA_COVERAGE')
            (deps/metadata[0].name).symlink_to(metadata[0],target_is_directory=True)
            borrowed=str(deps)
        proc=subprocess.run([sys.executable,'-I',str(child),str(project),str(config),
            str(Path(checkpoint).resolve()) if checkpoint else '',
            borrowed,
            'yes' if reader_checks else 'no',str(Path(dependency_path).resolve()) if dependency_path else ''],cwd=root,capture_output=True,text=True)
    result={'status':'PASS' if proc.returncode==0 else 'FAIL','payload_sha256':payload_sha,
            'notebook_sha256':hashlib.sha256(Path(notebook).read_bytes()).hexdigest(),
            'fresh_isolated_process':True,'python':sys.version.split()[0],'explicit_dependency_path':str(dependency_path) if dependency_path else None,'gpu_exercised':False,'uses_real_images':False}
    if proc.returncode==0:result['checks']=json.loads(proc.stdout.splitlines()[-1])
    else:result['failure']=proc.stderr[-4000:]
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook',required=True);parser.add_argument('--checkpoint')
    parser.add_argument('--pydicom-source');parser.add_argument('--reader-checks',action='store_true')
    parser.add_argument('--dependency-path',help='Explicit installed third-party packages; unpacked project retains priority')
    parser.add_argument('--out',required=True);args=parser.parse_args()
    output=Path(args.out)
    if output.exists():raise ValueError('FRESH_RECEIPT')
    result=check(args.notebook,args.checkpoint,args.pydicom_source,args.reader_checks,args.dependency_path)
    with output.open('x') as handle:handle.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['status']=='PASS' else 1)
