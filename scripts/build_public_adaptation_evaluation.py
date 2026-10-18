"""Build an offline evaluation-only recovery from a pinned immutable fit notebook."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from knee.public_adaptation_fit import fingerprint, fit_config, TRUTH_POLICY
from knee.public_adaptation_pilot import file_sha
from knee.reference import sha256
from scripts.build_public_adaptation_fit import reviewed_json


def build_evaluation(original_notebook, original_metadata, failed_receipt, terminal_receipt,
                     destination, slug, *, expected_notebook_sha256, expected_metadata_sha256,
                     expected_failed_sha256, expected_terminal_sha256, expected_endpoint_sha256,
                     recovery_dataset, recovery_manifest, expected_recovery_manifest_sha256,
                     expected_dataset_version):
    """No downloads, launch, checkpoint selection or training continuation."""
    root=Path(__file__).resolve().parents[1]; dest=Path(destination)
    if dest.exists() or (dest.resolve().is_relative_to(root) and not dest.resolve().is_relative_to(root/'outputs')):
        raise ValueError('Fresh private ignored build required')
    if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug): raise ValueError('Invalid slug')
    if not re.fullmatch('[0-9a-f]{64}',expected_endpoint_sha256 or ''): raise ValueError('Pinned endpoint bytes required')
    if not re.fullmatch('ozgung/[a-z0-9]+(?:-[a-z0-9]+)*',recovery_dataset): raise ValueError('Owned recovery dataset required')
    if type(expected_dataset_version) is not int or expected_dataset_version <= 0: raise ValueError('Positive recovery dataset version required')
    notebook=reviewed_json(original_notebook,expected_notebook_sha256)
    metadata=reviewed_json(original_metadata,expected_metadata_sha256)
    prior=reviewed_json(failed_receipt,expected_failed_sha256)
    terminal=reviewed_json(terminal_receipt,expected_terminal_sha256)
    recovery_manifest=reviewed_json(recovery_manifest,expected_recovery_manifest_sha256)
    if not (prior.get('status')=='FAIL' and prior.get('failure_stage')=='evaluation' and
            prior.get('updates')==432 and prior.get('epoch')==2 and prior.get('consumed_cursor')==0 and
            prior.get('config')==fit_config() and prior.get('binding',{}).get('config')==fit_config() and
            re.fullmatch('[0-9a-f]{64}',prior.get('checkpoint_sha256','')) and prior.get('checkpoint_bytes',0)>0):
        raise ValueError('Finished fixed training receipt required')
    source=metadata['id']
    if not (terminal.get('kernel_source')==source and terminal.get('status')=='ERROR' and
            type(terminal.get('version')) is int and terminal['version']>0 and
            terminal.get('checkpoint_sha256')==prior['checkpoint_sha256']):
        raise ValueError('Exact terminal kernel/version/checkpoint required')
    files=recovery_manifest.get('files',{})
    if not (recovery_manifest.get('schema')==1 and recovery_manifest.get('source_kernel')==source and
            recovery_manifest.get('source_version')==terminal['version'] and
            recovery_manifest.get('original_training_binding')==prior['binding'] and
            set(files)=={'receipt.json','resume.pt','endpoint.pt'} and
            files['resume.pt']=={'sha256':prior['checkpoint_sha256'],'bytes':prior['checkpoint_bytes']} and
            files['receipt.json']=={'sha256':expected_failed_sha256,'bytes':Path(failed_receipt).stat().st_size} and
            files['endpoint.pt'].get('sha256')==expected_endpoint_sha256 and
            type(files['endpoint.pt'].get('bytes')) is int and files['endpoint.pt']['bytes']>0):
        raise ValueError('Exact recovery dataset manifest required')
    cells=[c for c in notebook['cells'] if c['cell_type']=='code']
    if len(cells)!=1: raise ValueError('Expected immutable fit notebook')
    code=''.join(cells[0]['source'])
    assignments={}
    for node in ast.parse(code).body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            try: assignments[node.targets[0].id]=ast.literal_eval(node.value)
            except (ValueError,TypeError): pass
    sources=assignments['sources']; binding=assignments['BINDING']
    if binding!=prior['binding'] or fingerprint(sources)!=binding['source_bundle_sha256'] or assignments.get('RESUME_SOURCE') is not None:
        raise ValueError('Original fit sources/binding differ')
    original_sources=dict(sources)
    # Only the operational entry/evaluation helper changes; original scientific modules remain byte-identical.
    sources['knee/public_adaptation_fit.py']=(root/'knee/public_adaptation_fit.py').read_text()
    digest=fingerprint(sources)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    recovery={'prior':prior,'terminal':terminal,'endpoint_sha256':expected_endpoint_sha256,
              'recovery_dataset':recovery_dataset,'expected_dataset_version':expected_dataset_version,
              'recovery_manifest':recovery_manifest,'recovery_manifest_sha256':expected_recovery_manifest_sha256,
              'dataset_version_binding':'mounted files match externally verified version manifest hashes',
              'truth_policy':TRUTH_POLICY,'endpoint_qualification':'all-tensor fixed average oracle from SHA-pinned resume',
              'operational_bundle_sha256':digest,'operational_code_revision':revision,
              'original_notebook_sha256':expected_notebook_sha256,
              'original_source_bundle_sha256':binding['source_bundle_sha256'],
              'changed_sources':{k:{'original':sha256(original_sources[k].encode()),'operational':sha256(v.encode())}
                                 for k,v in sources.items() if v!=original_sources[k]}}
    replacements=[(f'sources={original_sources!r}',f'sources={sources!r}'),
                  ("RESUME_SOURCE=None",f'RESUME_SOURCE={source!r}'),('PRIOR=None',f'PRIOR={prior!r}\nEVALUATION_RECOVERY={recovery!r}'),
                  (f"os.environ['KNEE_SOURCE_SHA256']={binding['source_bundle_sha256']!r}",f"os.environ['KNEE_SOURCE_SHA256']={digest!r}"),
                  ("['receipt.json','resume.pt']","['receipt.json','resume.pt','endpoint.pt']"),
                  ("shutil.copyfile(previous/'resume.pt',out/'resume.pt'); resume=out/'resume.pt'",
                   "shutil.copyfile(previous/'resume.pt',out/'resume.pt'); resume=out/'resume.pt'\n        if EVALUATION_RECOVERY['endpoint_sha256'] is not None and file_sha(previous/'endpoint.pt')!=EVALUATION_RECOVERY['endpoint_sha256']: raise ValueError('Endpoint artifact bytes differ')\n        shutil.copyfile(previous/'endpoint.pt',out/'endpoint.pt')"),
                  ('BINDING,resume=resume)','BINDING,resume=resume,evaluation_recovery=EVALUATION_RECOVERY)')]
    for old,new in replacements:
        if code.count(old)!=1: raise ValueError('Original immutable entry differs')
        code=code.replace(old,new,1)
    old="previous=notebook_root('/kaggle/input',RESUME_SOURCE,'public-adaptation-fit',['receipt.json','resume.pt','endpoint.pt'])"
    new=("previous=dataset_root('/kaggle/input',EVALUATION_RECOVERY['recovery_dataset'],['receipt.json','resume.pt','endpoint.pt','recovery-manifest.json'])\n"
         "        if file_sha(previous/'recovery-manifest.json')!=EVALUATION_RECOVERY['recovery_manifest_sha256'] or json.loads((previous/'recovery-manifest.json').read_text())!=EVALUATION_RECOVERY['recovery_manifest']: raise ValueError('Recovery manifest differs')\n"
         "        for name,record in EVALUATION_RECOVERY['recovery_manifest']['files'].items():\n"
         "            if file_sha(previous/name)!=record['sha256'] or (previous/name).stat().st_size!=record['bytes']: raise ValueError('Recovery dataset file bytes differ')")
    if code.count(old)!=1: raise ValueError('Original recovery mount differs')
    code=code.replace(old,new,1)
    compile(code,'<evaluation-only-recovery>','exec')
    cells[0]['source']=code.splitlines(True)
    notebook['cells'][0]['source']=['# Fixed endpoint evaluation recovery\n','Finished two-epoch training is preserved. Zero optimizer updates; strict original endpoint average, unchanged fixed population and original truth gate. Observed original y_ is converted to float32 and thresholded at >=0.5 with masks unchanged, matching the historical evaluator. The protocol amendment is recorded before metrics.\n']
    metadata.update(id='ozgung/'+slug,title=slug.replace('-',' ').title(),kernel_sources=[],
                    dataset_sources=list(dict.fromkeys([*metadata.get('dataset_sources',[]),recovery_dataset])))
    if metadata.get('enable_internet') or not metadata.get('is_private'): raise ValueError('Private offline metadata required')
    encoded=json.dumps(notebook,indent=2)+'\n'
    if len(encoded.encode())>900000: raise ValueError('Notebook exceeds source limit')
    dest.mkdir(parents=True)
    (dest/'notebook.ipynb').write_text(encoded)
    (dest/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    receipt={**recovery,'binding':binding,'kernel_source':source,'launched':False,
             'evaluation_only':True,'optimizer_updates_this_run':0,
             'notebook_sha256':file_sha(dest/'notebook.ipynb'),
             'failed_receipt_sha256':expected_failed_sha256,'terminal_receipt_sha256':expected_terminal_sha256,
             'sources':{k:sha256(v.encode()) for k,v in sources.items()}}
    (dest/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('original_notebook','original_metadata','failed_receipt','terminal_receipt','destination'):
        parser.add_argument(name,type=Path)
    parser.add_argument('slug')
    for name in ('notebook','metadata','failed','terminal'):
        parser.add_argument('--expected-'+name+'-sha256',required=True)
    parser.add_argument('--expected-endpoint-sha256',required=True)
    parser.add_argument('--recovery-dataset',required=True)
    parser.add_argument('--recovery-manifest',type=Path,required=True)
    parser.add_argument('--expected-recovery-manifest-sha256',required=True)
    parser.add_argument('--expected-dataset-version',type=int,required=True)
    print(json.dumps(build_evaluation(**vars(parser.parse_args())),indent=2))
