"""Offline entry; whole-process deadline starts in the generated bootstrap."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from knee.fresh_reader_setup import dataset_root, exact_root
from knee.reference import canonical_bytes, sha256
from knee.reference_blend import CONVNEXT_DATASET, select_codec_wheels, read_test_contract
from knee.versia_fusion import MAX_SECONDS, VERSION, verify_sources, run_process, combine

c=FUSION_CONTRACT;q=c['qualification'];deadline=FUSION_STARTED+MAX_SECONDS
if (c['version']!=VERSION or c['max_seconds']!=MAX_SECONDS or c['private'] is not True or c['internet'] is not False or
        sha256(canonical_bytes(q))!=c['qualification_sha256']):raise ValueError('Frozen fusion contract changed')
verify_sources(project,q)
if sys.version_info[:2]!=(3,12) or any(importlib.metadata.version(p)!=v for p,v in q['environment']['packages'].items()):
    raise ValueError('Qualified fusion environment changed')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
                  PYTHONPATH=str(project),PYTHONUNBUFFERED='1')
out=Path('/kaggle/working/versia-fusion');out.mkdir(parents=True,exist_ok=False)
# Never accept a stale entry or stale component output from a previous execution.
if Path('/kaggle/working/submission.csv').exists():raise ValueError('Stale submission output exists')
try:
    manifest=q['mirror']['manifest']
    weights=dataset_root('/kaggle/input',manifest['weight_dataset'],[manifest['checkpoint_file']])
    publisher=manifest['publisher_manifest']
    if sha256((weights/publisher['filename']).read_bytes())!=publisher['sha256']:raise ValueError('Mirror publisher manifest changed')
    assets=dataset_root('/kaggle/input',CONVNEXT_DATASET,list(q['convnext']['contract']['weights']))
    wheels=select_codec_wheels(assets);hashes={}
    for path in wheels:
        with path.open('rb') as h:hashes[path.name]=hashlib.file_digest(h,'sha256').hexdigest()
    if hashes!=q['codecs']:raise ValueError('Offline codec hashes changed')
    run_process([sys.executable,'-m','pip','install','--no-index','--no-deps',*map(str,wheels)],deadline=deadline)
    root=exact_root([Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')],
                    ['test.csv','test_series.csv','sample_submission.csv'])
    columns,ids,_=read_test_contract(root,rehearsal=c['visible'])
    payload={'contract':c,'project':str(project),'root':str(root),'assets':str(assets),
             'checkpoint':str(weights/manifest['checkpoint_file']),'output':str(out),
             'mirror_public':MIRROR_PUBLIC,'mirror_decoder':MIRROR_DECODER,'versia_sources':VERSIA_SOURCES}
    payload_path=out/'runtime-input.json';payload_path.write_text(json.dumps(payload,allow_nan=False))
    reports={};paths={};name='visible-predictions.csv' if c['visible'] else 'submission.csv'
    for arm in ('mirror','convnext'):
        run_process([sys.executable,'-m','knee.versia_fusion','--component',arm,'--payload',str(payload_path),'--deadline',str(deadline)],deadline=deadline)
        report_file='mirror-audit.json' if arm=='mirror' else 'convnext-audit.json'
        reports[arm]=json.loads((out/arm/report_file).read_text());paths[arm]=out/arm/name
        if (reports[arm]['ordered_ids_sha256']!=sha256(canonical_bytes(ids)) or
                reports[arm]['target_order_sha256']!=sha256(canonical_bytes(columns[1:]))):
            raise ValueError('Component output differs from actual official test order')
    report=combine(out,q,reports,paths,visible=c['visible'],started=FUSION_STARTED)
    if not c['visible']:
        with (out/'submission.csv').open('rb') as src,Path('/kaggle/working/submission.csv').open('xb') as dst:shutil.copyfileobj(src,dst)
    report['elapsed_seconds']=time.monotonic()-FUSION_STARTED
    (out/'fusion-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'event':'fusion_done','status':'PASS','test_studies':len(ids),'elapsed_seconds':report['elapsed_seconds']}),flush=True)
except BaseException as error:
    Path('/kaggle/working/submission.csv').unlink(missing_ok=True)
    if (out/'fusion-audit.json').exists():
        failed=json.loads((out/'fusion-audit.json').read_text());failed['status']='FAIL'
        (out/'fusion-audit.json').write_text(json.dumps(failed,indent=2,allow_nan=False)+'\n')
    (out/'fusion-failure.json').write_text(json.dumps({'status':'FAIL','failure_type':type(error).__name__,
                                                    'elapsed_seconds':time.monotonic()-FUSION_STARTED})+'\n')
    raise RuntimeError('Fusion failed; no other-arm fallback or submission') from None
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
