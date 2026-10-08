"""Build only: archived mirror67 -> visible production qualification -> full entry."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee import mirror_input
from knee.mirror_submission import CROSS_RUN_ATOL, MAX_SECONDS, VERSION, compare_csv, require_visible
from knee.reference import audit_source, canonical_bytes, sha256, WEIGHT_DATASET
from knee.reference_blend import CONVNEXT_DATASET, DOCKER_IMAGE

SCIENTIFIC = ('knee/__init__.py', 'knee/reference.py', 'knee/mirror_input.py', 'knee/raptor_input.py',
              'knee/reference_blend.py', 'knee/public_reader_contract.py', 'knee/fresh_reader_setup.py')
NEW_RUNTIME = ('knee/mirror_submission.py', 'scripts/mirror_submission_entry.py')


def archived_build(directory, public_source, root):
    """Parse literals only; verify the complete original executable without executing it."""
    directory, root = Path(directory), Path(root)
    raw = {name:(directory/name).read_bytes() for name in
           ('notebook.ipynb', 'reference-contract.json', 'build-receipt.json', 'kernel-metadata.json')}
    nb, c, b, meta = (json.loads(raw[name]) for name in raw)
    code_cells = [cell for cell in nb['cells'] if cell['cell_type']=='code']
    if len(code_cells) != 1:
        raise ValueError('Archived rehearsal must have one exact code cell')
    code = ''.join(code_cells[0]['source'])
    values, env = {}, {}
    for node in ast.parse(code).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in ('sources','PUBLIC_SOURCE','DECODER_SOURCE','REFERENCE_MANIFEST','REFERENCE_CONTRACT'):
            if target.id in values: raise ValueError('Duplicate archived source assignment')
            values[target.id] = ast.literal_eval(node.value)
        elif (isinstance(target, ast.Subscript) and ast.unparse(target.value)=='os.environ' and
              isinstance(target.slice, ast.Constant) and target.slice.value in ('KNEE_CODE_REVISION','KNEE_SOURCE_SHA256')):
            if target.slice.value in env: raise ValueError('Duplicate archived provenance assignment')
            env[target.slice.value] = ast.literal_eval(node.value)
    files = values['sources']
    if (not isinstance(files,dict) or not all(isinstance(v,str) and re.fullmatch(r'knee/[a-z_]+\.py',k) for k,v in files.items()) or
            not re.fullmatch(r'[0-9a-f]{40}',env.get('KNEE_CODE_REVISION',''))):
        raise ValueError('Archived source map/revision is invalid')
    digest = sha256(json.dumps(files,sort_keys=True).encode())
    module, decoder, manifest = audit_source(Path(public_source).read_text(), profile='mirror')
    publisher = c['source'].get('publisher_manifest',{})
    if publisher.get('filename') != 'checkpoint-manifest.json' or not re.fullmatch(r'[0-9a-f]{64}',publisher.get('sha256','')):
        raise ValueError('Archived publisher manifest pin missing')
    manifest['publisher_manifest'] = publisher
    entry = (root/'scripts/rehearse_reference_entry.py').read_text()
    if (b.get('bundle_sha256') != digest or env.get('KNEE_SOURCE_SHA256') != digest or
            b.get('contract_sha256') != sha256(canonical_bytes(c)) or b.get('source_audit') != manifest or
            b.get('notebook_bytes') != len(raw['notebook.ipynb']) or c['source'] != manifest or
            c.get('input_policy') != mirror_input.policy_contract() or
            c.get('builder_sha256') != sha256((root/'scripts/build_reference_rehearsal.py').read_bytes()) or
            c.get('entry_sha256') != sha256(entry.encode())):
        raise ValueError('Archived rehearsal/source/contract binding differs')
    expected_entry = (f'PUBLIC_SOURCE={module!r}\nDECODER_SOURCE={decoder!r}\n'
                      f'REFERENCE_MANIFEST={manifest!r}\nREFERENCE_CONTRACT={c!r}\n'+entry)
    expected_code = (
        "import os, sys, json\nfrom pathlib import Path\n"
        "project = Path('/kaggle/working/project')\n"
        f'sources = {files!r}\n'
        "for name, content in sources.items():\n"
        "    path = project / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(content)\n"
        "sys.path.insert(0, str(project))\n"
        f"os.environ['KNEE_CODE_REVISION'] = {env['KNEE_CODE_REVISION']!r}\n"
        f"os.environ['KNEE_SOURCE_SHA256'] = {digest!r}\n"+expected_entry)
    if code != expected_code:
        raise ValueError('Archived executable differs from pinned source/entry reconstruction')
    if (meta.get('dataset_sources') != [WEIGHT_DATASET,CONVNEXT_DATASET] or meta.get('kernel_sources') != [] or
            meta.get('model_sources') != [] or meta.get('competition_sources') != ['rsna-knee-abnormality-detection'] or
            meta.get('is_private') is not True or meta.get('enable_internet') is not False or
            meta.get('enable_gpu') is not True or meta.get('docker_image') != DOCKER_IMAGE):
        raise ValueError('Archived offline attachment/environment contract differs')
    for name in SCIENTIFIC:
        if files.get(name) != (root/name).read_text():
            raise ValueError('Scientific dependency changed after rehearsal; new qualification required')
    return files, module, decoder, c, digest, env['KNEE_CODE_REVISION'], {k:sha256(v) for k,v in raw.items()}


def build_mirror(destination, slug, *, rehearsal_build, rehearsal_report, rehearsal_csv, public_source,
                 submission=False, visible_report=None, visible_csv=None):
    root = Path(__file__).resolve().parents[1]
    destination = Path(destination)
    if destination.exists(): raise ValueError('Use a fresh output directory; builds are immutable')
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',slug): raise ValueError('Invalid private notebook slug')
    files, public, decoder, original, digest, revision, archive_hashes = archived_build(rehearsal_build,public_source,root)
    raw = Path(rehearsal_report).read_bytes()
    report = json.loads(raw)
    mirror_input.require_rehearsal(original,report,bundle_sha256=digest)
    if report.get('source_revision') != revision or report.get('visible_csv_sha256') != sha256(Path(rehearsal_csv).read_bytes()):
        raise ValueError('Rehearsal receipt revision/visible output does not match archived build')
    compare_csv(rehearsal_csv,rehearsal_csv)  # Three finite ordered rows, no fabricated-prior rows.
    # Package only the archived dependency closure, plus this new entry. Unrelated
    # current training code cannot silently change the qualified image pipeline.
    sources = {name:files[name] for name in SCIENTIFIC}
    sources.update({name:(root/name).read_text() for name in NEW_RUNTIME})
    q = {'version':VERSION, 'manifest':original['source'], 'input_policy':original['input_policy'],
         'archived_build_sha256':archive_hashes, 'rehearsal_bundle_sha256':digest,
         'rehearsal_source_revision':revision, 'rehearsal_report_sha256':sha256(raw),
         'rehearsal_csv_sha256':report['visible_csv_sha256'],
         'sampling':report['sampling'], 'strict_comparison_counts':report['strict_comparison_counts'],
         'runtime_sources':{name:sha256(value.encode()) for name,value in sources.items()},
         'builder_sha256':sha256(Path(__file__).read_bytes()), 'codec_hashes':original['codec_hashes'],
         'environment':{'python':[3,12],'torch':'2.10.0+cu128','timm':'1.0.26','docker_image':DOCKER_IMAGE}}
    fingerprint = sha256(canonical_bytes(q))
    evidence = None
    if submission:
        if visible_report is None or visible_csv is None:
            raise ValueError('Full entry requires passing standalone visible report and CSV')
        vr = Path(visible_report).read_bytes()
        visible = json.loads(vr)
        require_visible(visible,fingerprint,visible_csv)
        if (visible.get('checkpoint_verified_sha256') != q['manifest']['checkpoint_sha256'] or
                visible.get('checkpoint_bytes') != mirror_input.CHECKPOINT_BYTES or visible.get('codec_hashes') != q['codec_hashes'] or
                visible.get('environment',{}).get('python_major_minor') != [3,12] or
                any(visible.get('environment',{}).get('packages',{}).get(p) != q['environment'][p] for p in ('torch','timm'))):
            raise ValueError('Standalone visible weight/codec/environment evidence differs')
        error = compare_csv(rehearsal_csv,visible_csv)
        evidence = {'report_sha256':sha256(vr),'csv_sha256':sha256(Path(visible_csv).read_bytes()),
                    'rehearsal_probability_max_error':error,'tolerance':CROSS_RUN_ATOL,
                    'visible_rank_and_tie_parity':True}
    elif visible_report is not None or visible_csv is not None:
        raise ValueError('Visible evidence is only accepted when building a full entry')
    contract = {'version':VERSION,'qualification':q,'qualification_sha256':fingerprint,
                'visible':not submission,'max_seconds':MAX_SECONDS,'visible_evidence':evidence,
                'private':True,'internet':False}
    bundle_hash = sha256(canonical_bytes(sources))
    # Timer starts before writing/importing any bundled module. The entry preserves
    # this start and remaining deadline, including mount/stat/install/model setup.
    code = ("import time, signal\nMIRROR_STARTED=time.monotonic()\n"
            "def _mirror_expired(signum, frame):\n    raise TimeoutError('Actual nine-hour mirror runtime exhausted')\n"
            f"signal.signal(signal.SIGALRM,_mirror_expired)\nsignal.setitimer(signal.ITIMER_REAL,{MAX_SECONDS})\n"
            "import os,sys\nfrom pathlib import Path\nproject=Path('/kaggle/working/project')\n"
            f'sources={sources!r}\n'
            "for name,content in sources.items():\n    path=project/name\n    path.parent.mkdir(parents=True,exist_ok=True)\n    path.write_text(content)\n"
            "sys.path.insert(0,str(project))\n"
            f"os.environ['KNEE_SOURCE_SHA256']={bundle_hash!r}\n"
            f'PUBLIC_SOURCE={public!r}\nDECODER_SOURCE={decoder!r}\nMIRROR_CONTRACT={contract!r}\n'
            "exec(compile(sources['scripts/mirror_submission_entry.py'],'<mirror-submission-entry>','exec'))\n")
    compile(code,'<mirror-standalone>','exec')
    nb = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
          'language_info':{'name':'python','version':'3.12'}},'cells':[
        {'cell_type':'markdown','id':'attribution','metadata':{},'source':[
            '# Native320 mirror image-only inference\n',
            f"{q['manifest']['attribution']}. [Public source]({q['manifest']['source_url']}).\n",
            f'[Public accuracy weights](https://www.kaggle.com/datasets/{WEIGHT_DATASET}); publisher competition-use permission, dataset license `other`. No host/prize clearance claimed.\n',
            'Exact native320/K94, equal original/anatomical-mirror probabilities; private/offline, no training labels or reports read.\n',
            f'Qualified67-study archive `{digest}`. Separate entry qualification `{fingerprint}`.\n']},
        {'cell_type':'code','id':'run','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(True)}]}
    notebook = (json.dumps(nb,indent=2)+'\n').encode()
    if len(notebook)>900000 or len(code.encode())>900000: raise ValueError('Generated notebook exceeds source-size limit')
    meta = {'id':f'ozgung/{slug}','title':slug.replace('-',' ').title(),'code_file':'notebook.ipynb',
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,
        'enable_internet':False,'competition_sources':['rsna-knee-abnormality-detection'],
        'dataset_sources':[WEIGHT_DATASET,CONVNEXT_DATASET],'kernel_sources':[],'model_sources':[], 'docker_image':DOCKER_IMAGE}
    receipt = {'bundle_sha256':bundle_hash,'qualification_sha256':fingerprint,
        'contract_sha256':sha256(canonical_bytes(contract)), 'notebook_sha256':sha256(notebook),
        'notebook_bytes':len(notebook),'rehearsal_bundle_sha256':digest,
        'build_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'notebook.ipynb').write_bytes(notebook)
    for name,value in [('kernel-metadata.json',meta),('mirror-contract.json',contract),('build-receipt.json',receipt)]:
        (destination/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('slug')
    for option in ('rehearsal-build','rehearsal-report','rehearsal-csv','public-source'):
        parser.add_argument('--'+option,required=True,type=Path)
    parser.add_argument('--submission',action='store_true')
    parser.add_argument('--visible-report',type=Path)
    parser.add_argument('--visible-csv',type=Path)
    print(json.dumps(build_mirror(**vars(parser.parse_args())),indent=2))
