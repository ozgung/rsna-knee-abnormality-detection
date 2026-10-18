"""Separately seal R7 prefetch P3/native224 mirror qualification or a genuinely bound entry.

Legacy v7/WS2 files and build identities are never rewritten. The prospective v3
projection excludes only the separately measured serial-control diagnostic; its
wall remains in actual rehearsal elapsed time under the unchanged nine-hour limit.
"""
import argparse
import ast
import base64
import csv
import hashlib
import json
import lzma
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import build_public_ws2_entry as BASE
from knee import public_ws3_entry as WS3
from knee import public_ws3_checkpoint as C224
from knee.public_ws3_prefetch import load_v7_input_reference, reference_from_receipt
from knee.public_s6_qualification import canonical,digest,file_sha,require,save

CANDIDATE_SHA='09ad6adb10bb1346a8751b9dfe33c3d732ce1616692a1e42179abd5fa43419de'
EXTRA=('knee/public_ws2_checkpoint.py','knee/public_ws2_recovery.py','knee/public_ws2_entry.py',
       'knee/public_ws3_checkpoint.py','knee/public_ws3_entry.py','knee/public_ws3_prefetch.py',
       'scripts/compare_r7_prefetch_p3.py')
PRIVATE=('evaluation_input_reference','p3_v7_reference',*BASE.PRIVATE_CFG)
POLICY={'version':'ws3-prefetch-projection-v1','gate_seconds':8.3*3600,
    'serial_control':'diagnostic excluded from hidden projection; included in actual elapsed under 9h',
    'other_overhead':'all original measured per-arm qualification overhead retained once',
    'mirror_speed_limit':.85,'worker_rss_limit_bytes':9.5e9,'phase_rss_limit_bytes':22e9,
    'prospective_decision':'docs/claude-astra-handoff.md 2026-10-15 12:31 UTC'}


def templates():
    project=Path(__file__).resolve().parents[1]
    archive=project/'outputs/kaggle/public-r7-recovery-rehearsal-20261013-v7'
    BASE.verify_preparation_build(project/'outputs/kaggle/ws2-r224-v2-qual-20261015',archive/'notebook.ipynb',archive/'build-receipt.json')
    replay,_,_=BASE.archived_payload(archive/'notebook.ipynb',archive/'build-receipt.json','rehearsal')
    entry_dir=project/'outputs/kaggle/public-r7-recovery-entry-20261013-v2'
    entry,_,_=BASE.archived_payload(entry_dir/'notebook.ipynb',entry_dir/'build-receipt.json','entry')
    require(file_sha(project/'knee/public_r7_prefetch_candidate.py')==CANDIDATE_SHA,'EXACT_PREFETCH_CANDIDATE_BYTES')
    def overlay(pack):
        pack=json.loads(canonical(pack))
        pack['sources']['knee/public_r7_recovery_entry.py']=(project/'knee/public_r7_prefetch_candidate.py').read_text()
        for name in EXTRA:pack['sources'][name]=(project/name).read_text()
        return pack
    replay,entry=overlay(replay),overlay(entry)
    # The worker inference closure is newly bound; old public scientific sources stay exact.
    shared=set(replay['sources'])&set(entry['sources'])
    require(all(replay['sources'][n]==entry['sources'][n] for n in shared if n!='knee/public_r7_recovery_rehearsal.py'),
            'V3_REHEARSAL_ENTRY_SOURCE_IDENTITY')
    source_binding=digest(entry['sources'])
    replay['cfg']['binding']['source_bundle_sha256']=source_binding
    # Keep the original v7 archive provenance, separately from this new source closure.
    replay['cfg']['binding']['prefetch_candidate_sha256']=CANDIDATE_SHA
    v7_path=project/'outputs/kaggle/public-r7-recovery-rehearsal-20261013-result-v7/s6-recovery-entry/receipt.json'
    load_v7_input_reference(v7_path)  # exact pinned receipt and 39 x 8 validation
    v7=json.loads(v7_path.read_bytes())
    reference={'workers':{'s6':[{k:w[k] for k in ('board','assigned','state_before')}|
        {'records':[{'id':x['id'],'input':x['input']} for x in w['records']]} for w in v7['workers']['s6']]}}
    return replay,entry,reference,v7_path


def make_binding(replay,mirror,reference):
    binding=BASE.ws2_binding(replay,{n:hashlib.sha256(replay['sources'][n].encode()).hexdigest() for n in BASE.NEW_FILES})
    binding.update(version=WS3.VERSION,variant='A-double-prime-prefetch-native224-'+('mirror' if mirror else 'plain'),
        r224_mirror=mirror,baseline_visible_reference_sha256=digest(replay['cfg']['heliosli_reference']['submission']),crop224_recipe=C224.RECIPE if mirror else BASE.C224.RECIPE,
        prefetch_candidate_sha256=CANDIDATE_SHA,p3_v7_reference_sha256=digest(reference),
        projection_policy=POLICY,
        runtime_source_hashes={n:hashlib.sha256(v.encode()).hexdigest() for n,v in replay['sources'].items()})
    # Entry excludes only rehearsal extension sources; the binding still pins their qualification origin.
    return binding


def bootstrap(pack):
    script=BASE.bootstrap(pack)
    script=script.replace("name.startswith('knee/')","name.startswith(('knee/','scripts/'))")
    script=script.replace('from knee.public_ws2_entry import run','from knee.public_ws3_entry import run')
    compile(script,'<ws3-prefetch>','exec')
    return script


def verify_result(root,binding):
    import numpy as np
    root=Path(root);r=json.loads((root/'receipt.json').read_bytes())
    base_path=root/'base/receipt.json';base=json.loads(base_path.read_bytes())
    WS3.require_rehearsal(r,binding,base_path,root/'visible-predictions.csv')
    ids=json.loads((root/'base/assignment.json').read_bytes())['ids']
    require(len(ids)==len(set(ids))==39 and ids==base['binding']['s6_probe_ids']+base['binding']['old_historical_ids']+base['visible_ids'],
            'EXACT_FROZEN39_ORDER')
    arms={}
    for name,base_name,folder,reports in [('old320','old',root/'base',base['workers']['old']),
            ('r7','s6',root/'base',base['workers']['s6']),('crop224','crop224',root,r['crop224_workers'])]:
        arms[name]=WS3.merged_arm(ids,reports,folder,base_name)
    old={row['id']:row for w in base['workers']['old'] for row in w['records']}
    for i,w in enumerate(r['crop224_workers']):
        require(w['model']['state_sha256']==w['state_before']==w['state_after'],'STRICT_NATIVE_STATE_BINDING')
        with np.load(root/f'board{i}-crop224/crop224/predictions.npz',allow_pickle=False) as ledger:
            for index,row in enumerate(w['records']):
                require(row['status']=='genuine' and row['state_healthy'] and all(row['input'][k]==old[row['id']]['input'][k]
                        for k in ('volume_sha256','mask_sha256')),'ALL39_NATIVE_SOURCE_STATE')
                C224.validate_view_probabilities(row['view_probabilities'],ledger['probabilities'][index],mirror_enabled=binding['r224_mirror'])
    rec=BASE.REC.recover(*(arms[n][0] for n in BASE.REC.ARMS),*(arms[n][1] for n in BASE.REC.ARMS))
    frozen=BASE.R7.recover(arms['old320'][0],arms['r7'][0],arms['old320'][1],arms['r7'][1])
    require(np.array_equal(frozen['control_rank'],rec['ranks']['old320']) and np.array_equal(frozen['s6_rank'],rec['ranks']['r7']),
            'UNCHANGED_BASELINE_RANKS')
    hashes={k:hashlib.sha256(frozen[v].tobytes()).hexdigest() for k,v in
            [('old_rank_sha256','control_rank'),('r7_rank_sha256','s6_rank'),('blend_sha256','blend')]}
    require(r['baseline_full_population']==hashes and rec['union_recovered']==0,'BASELINE_HASHES_AND_NO_RECOVERY')
    rows=list(csv.reader((root/'visible-predictions.csv').open()))
    actual=np.asarray([[float(x) for x in row[1:]] for row in rows[1:]])
    expected=rec['blend'][[ids.index(s) for s in base['visible_ids']]]
    require(actual.shape==(3,12) and np.max(np.abs(actual-expected))<=5e-10,'RECOMPUTED_V3_CSV')
    return {'receipt_sha256':file_sha(root/'receipt.json'),'base_receipt_sha256':file_sha(base_path),
        'csv_sha256':file_sha(root/'visible-predictions.csv'),'qualified_r224_state':r['crop224_workers'][0]['state_before'],
        'qualified_r7_state':base['workers']['s6'][0]['state_before'],'all_six_ledgers_recomputed':True,
        'p3_replayed':True,'projection':r['projection']}


def verify_mirror_admission(path):
    from decimal import Decimal, InvalidOperation
    require(path is not None,'OFFICIAL_APRIME_SCORE_REQUIRED_FOR_MIRROR_ENTRY')
    rows=list(csv.DictReader(Path(path).open()))
    match=[row for row in rows if row.get('ref')=='57229227']
    require(len(match)==1 and match[0].get('status') in ('SubmissionStatus.COMPLETE','complete','COMPLETE'),
            'APRIME_OFFICIAL_COMPLETE_REQUIRED')
    try:score=Decimal(match[0]['publicScore'])
    except (KeyError,InvalidOperation):raise ValueError('APRIME_OFFICIAL_SCORE_REQUIRED')
    require(score.is_finite() and score>=Decimal('0.956'),'APRIME_DISPLAYED_SCORE_ADMITS_MIRROR')
    return {'submission_id':'57229227','status':match[0]['status'],'publicScore':str(score),
            'official_snapshot_sha256':file_sha(path),'source':'kaggle competitions submissions CSV supplied by owner'}


def build(*,destination,slug,mode,mirror=False,rehearsal_result=None,rehearsal_build=None,mirror_admission=None):
    project=Path(__file__).resolve().parents[1];dest=Path(destination)
    require(not dest.exists() and dest.resolve().is_relative_to(project/'outputs'),'FRESH_IGNORED_BUILD')
    require(re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',slug) is not None and len(slug)<=50,'TITLE_ID_SLUG_CONSISTENCY')
    require(mode in ('rehearsal','entry') and type(mirror) is bool,'V3_MODE_RECIPE')
    if mode=='entry':require(rehearsal_result is not None and rehearsal_build is not None,'REAL_V3_QUALIFICATION_REQUIRED')
    admission=verify_mirror_admission(mirror_admission) if mirror else None
    replay,entry,reference,v7_path=templates();binding=make_binding(replay,mirror,reference)
    verification=None
    if mode=='entry':
        require(rehearsal_result is not None and rehearsal_build is not None,'REAL_V3_QUALIFICATION_REQUIRED')
        verify_preparation(rehearsal_build,mirror)
        verification=verify_result(rehearsal_result,binding)
        pack=entry;cfg=pack['cfg']
        cfg['binding']={k:v for k,v in replay['cfg']['binding'].items() if k not in BASE.PRIVATE_BINDING}
        cfg['binding'].update(rehearsal_receipt_sha256=verification['base_receipt_sha256'],
            qualified_r7_state=verification['qualified_r7_state'])
        cfg['pilot_state_sha256']=verification['qualified_r7_state'];cfg['r224_state_pin']=verification['qualified_r224_state']
        require(not any(k in cfg for k in PRIVATE),'PURE_ENTRY_NO_CONTROL_OR_PRIVATE_REFERENCE')
    else:
        pack=replay;cfg=pack['cfg']
        cfg['evaluation_input_reference']=reference_from_receipt(reference)
        cfg['p3_v7_reference']=reference
    cfg.update(ws2_binding=binding,r224_mirror=mirror,r7_prefetch=True)
    require(cfg['mode']==mode,'EXACT_MODE')
    script=bootstrap(pack)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},
        'cells':[{'id':'ws3-image-only-prefetch','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],
                  'source':script.splitlines(True)}]}
    meta=json.loads(Path('outputs/kaggle/ws2-r224-v2-qual-20261015/kernel-metadata.json').read_bytes())
    meta.update(id='ozgung/'+slug,title=slug)
    raw=json.dumps(nb,indent=2)+'\n';require(len(raw.encode())<900000,'NOTEBOOK900K_CAP')
    dest.mkdir(parents=True);(dest/'notebook.ipynb').write_text(raw)
    (dest/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    receipt={'status':'PREPARATION_ONLY' if mode=='rehearsal' else 'QUALIFIED_V3_ENTRY_BUILD','mode':mode,
        'binding':binding,'payload_sha256':digest(pack),'builder_sha256':file_sha(__file__),
        'source_hashes':{n:hashlib.sha256(v.encode()).hexdigest() for n,v in pack['sources'].items()},
        'files':{n:file_sha(dest/n) for n in ('notebook.ipynb','kernel-metadata.json')},
        'rehearsal_verification':verification,'notebook_bytes':len(raw.encode()),'runtime_checks_exercised':False,
        'mirror_admission':admission,'p2_status':'failed CPU speed screen; mechanics only; fresh P3 required',
        'projection_policy':POLICY,'mirror_score_admission':'requires official displayed A-prime>=0.956 before mirror GPU/entry admission'}
    save(dest/'build-receipt.json',receipt)
    save(dest/'MANIFEST.json',{'status':receipt['status'],'payload_sha256':digest(pack),
        'files':{n:{'sha256':file_sha(dest/n),'bytes':(dest/n).stat().st_size}
                 for n in ('notebook.ipynb','kernel-metadata.json','build-receipt.json')}})
    return receipt


def verify_preparation(root,mirror):
    root=Path(root);receipt=json.loads((root/'build-receipt.json').read_bytes())
    require(receipt['status']=='PREPARATION_ONLY' and receipt['builder_sha256']==file_sha(__file__),'EXACT_V3_PREPARATION')
    replay,_,reference,_=templates();binding=make_binding(replay,mirror,reference)
    require(receipt['binding']==binding,'EXACT_V3_BINDING')
    if mirror:
        from decimal import Decimal
        admission=receipt.get('mirror_admission') or {}
        require(admission.get('submission_id')=='57229227' and admission.get('status') in ('SubmissionStatus.COMPLETE','complete','COMPLETE')
                and Decimal(admission.get('publicScore','NaN')).is_finite()
                and Decimal(admission['publicScore'])>=Decimal('0.956'),'BOUND_MIRROR_SCORE_ADMISSION')
    replay['cfg'].update(evaluation_input_reference=reference_from_receipt(reference),p3_v7_reference=reference,
                         ws2_binding=binding,r224_mirror=mirror,r7_prefetch=True)
    require(receipt['payload_sha256']==digest(replay),'EXACT_V3_PREPARATION_PAYLOAD')
    nb=json.loads((root/'notebook.ipynb').read_bytes())
    require(''.join(nb['cells'][0]['source'])==bootstrap(replay),'EXACT_V3_PREPARATION_NOTEBOOK')
    manifest=json.loads((root/'MANIFEST.json').read_bytes())
    require(manifest['status']=='PREPARATION_ONLY' and manifest['payload_sha256']==digest(replay),'EXACT_V3_MANIFEST')
    BASE.verify_manifest_files(root,manifest,('notebook.ipynb','kernel-metadata.json','build-receipt.json'))
    require(receipt['files']=={n:file_sha(root/n) for n in ('notebook.ipynb','kernel-metadata.json')},'EXACT_V3_FILES')
    meta=json.loads((root/'kernel-metadata.json').read_bytes())
    expected_meta=json.loads(Path('outputs/kaggle/ws2-r224-v2-qual-20261015/kernel-metadata.json').read_bytes())
    expected_meta.update(id='ozgung/'+meta['title'],title=meta['title'])
    require(meta==expected_meta,'EXACT_V3_OFFLINE_METADATA')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination',required=True);p.add_argument('--slug',required=True)
    p.add_argument('--mode',choices=('rehearsal','entry'),required=True)
    p.add_argument('--mirror',action='store_true');p.add_argument('--rehearsal-result');p.add_argument('--rehearsal-build');p.add_argument('--mirror-admission')
    r=build(**vars(p.parse_args()))
    print(json.dumps({k:r[k] for k in ('status','payload_sha256','files','notebook_bytes')}))
