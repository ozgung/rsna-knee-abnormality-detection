"""B0′: dr-full checkpoint at B0 K62, unchanged S6 and 70:30 capped recovery.

The existing B0 modules are shipped unchanged. Workers, launch, merge and run
reuse their exact bytecode against this checkpoint's bindings. This is an
exposed-development leaderboard probe, not a clean held-out fine-tuning result.
"""
import ast
import inspect
import knee.public_s6_recovery_b0 as B0

globals().update({k:v for k,v in vars(B0).items() if not k.startswith('__')})

VERSION='s6-drfull-k62-b0prime-v1'
ENTRY_MODULE='knee.public_s6_recovery_b0prime'
REHEARSAL_MODULE='knee.public_s6_recovery_b0prime_rehearsal'
DATASET='ozgung/knee-raptor64-dr-full-fold0-final-20261013'
DATASET_VERSION=1
CHECKPOINT_FILE='raptor_ft_fold0_swa.pt'
DREAD_CHECKPOINT_SHA256='35b7fc2cc95155b13271d3bc43ebd4931a915eaa490c572d2fd02d3cac2a061c'
DREAD_CHECKPOINT_BYTES=292826627
DREAD_STATE_SHA256='a3dbd31bdd65f306e987808708ef7afe691f9fb61e67fb4ddd86843cd7ce7812'
DREAD_GRID_SHA256=B0.DREAD_GRID_SHA256
OLD_PROFILE={**B0.OLD_PROFILE,'checkpoint_sha256':DREAD_CHECKPOINT_SHA256,'checkpoint_bytes':DREAD_CHECKPOINT_BYTES}
METHOD={**B0.METHOD,'old':B0.METHOD['old'].replace('Dread CC0 raptor v4','own dr-full SWA9-11, Dread v4 init')}
QUALIFICATION_SECONDS=8.3*3600
rehearsal_hook=None


def raptor_profile_gate(public,manifest):
    """Only checkpoint identity changes; require the complete B0 K62 recipe."""
    require(manifest.get('checkpoint_sha256')==DREAD_CHECKPOINT_SHA256
            and manifest.get('checkpoint_file')==CHECKPOINT_FILE
            and manifest.get('weight_dataset')==DATASET,'B0PRIME_CHECKPOINT_MANIFEST')
    from knee.reference import RAPTOR_WEIGHT_DATASET,RAPTOR_WEIGHT_NAME
    original={**manifest,'checkpoint_sha256':B0.DREAD_CHECKPOINT_SHA256,
              'checkpoint_file':RAPTOR_WEIGHT_NAME,'weight_dataset':RAPTOR_WEIGHT_DATASET}
    B0.raptor_profile_gate(public,original)
    return dict(OLD_PROFILE)


def raptor62_gate(sid,p,counters,cfg):
    """Input-only comparison: Dread probabilities are invalid for new weights."""
    reference=cfg.get('raptor62_reference',{}).get(sid)
    if reference is None:return None
    require(set(reference)=={'counts'} and set(reference['counts'])==set(RAPTOR62_COUNT_KEYS)
            and all(counters.get(k)==v for k,v in reference['counts'].items()),'B0PRIME_VISIBLE_DECODE_COUNTS')
    return {'decode_counts_equal':True,'checkpoint_probability_reference':False}


def require_visible_decode_counts(r,binding):
    rows={row['id']:row for w in r['workers']['old'] for row in w['records']}
    for sid in r['visible_ids']:
        g=rows[sid].get('raptor62',{})
        require(g=={'decode_counts_equal':True,'checkpoint_probability_reference':False},'B0PRIME_VISIBLE_READER_PARITY')


def _rehearsal_contract():
    """Keep every B0 gate except its unavailable old-weight visible reference.

    Replace exactly the final visible-probability loop with input-only checks.
    The original B0 source is pinned by the builder and retained in the packet.
    """
    tree=ast.parse(inspect.getsource(B0.require_rehearsal));body=tree.body[0].body
    tail=body[-1]
    require(isinstance(tail,ast.For) and ast.unparse(tail.iter)=="r['visible_ids']"
            and 'RAPTOR62_VISIBLE_RECEIPT_PARITY' in ast.unparse(tail),'EXACT_B0_VISIBLE_GATE_REPLACEMENT')
    require(ast.unparse(body[-2])=="contract = binding['old_qualification']['raptor62']",'EXACT_B0_REFERENCE_LOOKUP')
    body[-2:]=[ast.Expr(ast.Call(ast.Name('require_visible_decode_counts',ast.Load()),
                              [ast.Name('r',ast.Load()),ast.Name('binding',ast.Load())],[]))]
    namespace=dict(globals());exec(compile(ast.fix_missing_locations(tree),'<b0prime-rehearsal-contract>','exec'),namespace)
    return namespace['require_rehearsal']


_require_b0prime=_rehearsal_contract()


def require_rehearsal(r,binding,csv_path):
    _require_b0prime(r,binding,csv_path)
    seconds=r.get('projection',{}).get('conservative_seconds',math.inf)
    require(math.isfinite(seconds) and 0<seconds<=QUALIFICATION_SECONDS,'B0PRIME_8_3H_PROJECTION')
    for arm in ('old','s6'):
        rows={row['id']:row for w in r['workers'][arm] for row in w['records']}
        reference=binding['reader_reference'][arm]
        require(set(rows)==set(reference),'B0PRIME_ALL39_REFERENCE_COVERAGE')
        for sid,expected in reference.items():
            require(rows[sid]['input']==expected['input'],'B0PRIME_UNCHANGED_IMAGE_INPUTS')
            if arm=='old':require(rows[sid]['decode_counters']==expected['decode_counters'],'B0PRIME_UNCHANGED_READER_COUNTS')
        require(all(0<w['peak_rss_bytes']<=9.5e9 for w in r['workers'][arm])
                and 0<r['phases'][arm]['peaks']['host_rss_bytes']<=22e9,'B0PRIME_HOST_RSS_LIMIT')


def projection(cfg,phases,reports,setup,merge_seconds):
    result=B0.projection(cfg,phases,reports,setup,merge_seconds)
    return {**result,'measured_inputs':{'setup_seconds':setup,'merge_seconds':merge_seconds}}


def _bind(function):
    clone=types.FunctionType(function.__code__,globals(),function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=function.__kwdefaults__
    return clone


old_worker=_bind(B0.old_worker)
s6_worker=_bind(B0.s6_worker)
launch=_bind(B0.launch)
merge=_bind(B0.merge)
run=_bind(B0.run)


if __name__=='__main__':
    import argparse
    require(Path(__file__).resolve()==(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_s6_recovery_b0prime.py').resolve(),'EXACT_FRESH_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.worker).read_bytes())
    (old_worker if cfg['arm']=='old' else s6_worker)(a.worker)
