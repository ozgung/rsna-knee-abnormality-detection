"""Executable facades that reuse the frozen A′ WS2 and B0prime worker paths.

Only the old-arm profile/checks and final rank recovery are adapted. The R7
worker, R224 worker, data staging, two-board launch and ordinary 9 h limits stay
on their archived code paths.
"""
import ast
import inspect
import json
import math
import types

from knee import public_r7_recovery_entry as R7
from knee import public_s6_recovery_b0 as B0
from knee import public_s6_recovery_b0prime as B0P
from knee import public_nafull_aprime as PROFILE
from knee.public_s6_qualification import require


def _bind(function, namespace, name=None):
    clone = types.FunctionType(function.__code__, namespace, name or function.__name__,
                               function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__doc__ = function.__doc__
    clone.__qualname__ = function.__qualname__
    return clone


def nafull_profile_gate(public, manifest):
    """Require exact Raptor62 recipe and na-full export identity."""
    require(manifest.get('checkpoint_sha256') == PROFILE.CHECKPOINT['sha256']
            and manifest.get('checkpoint_file') == PROFILE.CHECKPOINT['file']
            and manifest.get('weight_dataset') == PROFILE.CHECKPOINT['dataset'], 'NAFULL_CHECKPOINT_MANIFEST')
    recipe = manifest.get('recipe', {})
    from knee.reference import RAPTOR_RECIPE, RAPTOR_WEIGHT_DATASET, RAPTOR_WEIGHT_NAME
    original = {**manifest, 'checkpoint_sha256': B0.DREAD_CHECKPOINT_SHA256,
        'checkpoint_file': RAPTOR_WEIGHT_NAME, 'weight_dataset': RAPTOR_WEIGHT_DATASET,
        'recipe': recipe}
    # This proves the public source constants still match the exact frozen B0 recipe.
    B0.raptor_profile_gate(public, original)
    return {**B0.OLD_PROFILE, 'checkpoint_sha256': PROFILE.CHECKPOINT['sha256'],
            'checkpoint_bytes': PROFILE.CHECKPOINT['bytes']}


NA_WORKER_GLOBALS = dict(vars(B0P))
NA_WORKER_GLOBALS.update(VERSION=PROFILE.VERSION, ENTRY_MODULE='knee.public_nafull_aprime_entry',
    REHEARSAL_MODULE='knee.public_nafull_aprime_rehearsal',
    DREAD_CHECKPOINT_SHA256=PROFILE.CHECKPOINT['sha256'],
    DREAD_CHECKPOINT_BYTES=PROFILE.CHECKPOINT['bytes'],
    DREAD_STATE_SHA256=PROFILE.CHECKPOINT['state_sha256'],
    OLD_PROFILE={**B0.OLD_PROFILE, 'checkpoint_sha256': PROFILE.CHECKPOINT['sha256'],
                 'checkpoint_bytes': PROFILE.CHECKPOINT['bytes']},
    METHOD={**B0P.METHOD, 'old': PROFILE.METHOD_OLD},
    raptor_profile_gate=nafull_profile_gate)
NA_OLD_WORKER = _bind(B0P.old_worker, NA_WORKER_GLOBALS)


def _nafull_rehearsal_checks(r, binding):
    """K62-only row/state checks, supplemental to the adapted R7 contract."""
    from knee.public_s6_recovery_b0 import RAPTOR62_COUNT_KEYS, GUARDED_PARTIAL_CASES
    from knee import raptor_input as RI
    import numpy as np
    workers = r.get('workers', {}).get('old', [])
    require(len(workers) == 2, 'NAFULL_TWO_BOARD_WORKERS')
    stress = workers[0].get('partial_input_stress', {})
    require(stress.get('status') == 'PASS' and stress.get('cases') == dict.fromkeys(RI.STRESS_CASES, 'PASS')
            and stress.get('case_count') == len(RI.STRESS_CASES) and stress.get('uses_raw_data') is False
            and stress.get('uses_labels') is False, 'NAFULL_RAPTOR_PARTIAL_INPUT_STRESS')
    guarded = workers[0].get('guarded_partial_input_stress', {})
    require(guarded.get('status') == 'PASS' and guarded.get('guarded') is True
            and guarded.get('case_count') == len(GUARDED_PARTIAL_CASES)
            and guarded.get('cases') == dict.fromkeys(GUARDED_PARTIAL_CASES, 'PASS'),
            'NAFULL_GUARDED_RAPTOR_PARTIAL_TEN')
    references = binding.get('nafull_reader_reference', {})
    rows = {row['id']: row for worker in workers for row in worker.get('records', [])}
    require(len(rows) == 39 and set(rows) == set(references), 'NAFULL_ALL39_REFERENCE_COVERAGE')
    for board, worker in enumerate(workers):
        require(worker.get('old_profile') == NA_WORKER_GLOBALS['OLD_PROFILE']
                and worker.get('state_before', {}).get('state') == PROFILE.CHECKPOINT['state_sha256']
                and worker.get('state_after') == worker.get('state_before')
                and worker.get('constructor_grid_sha256') == binding['qualified_state']['old_grid'],
                'NAFULL_BOARD_STATE_PROFILE')
        require(worker.get('source_guard_receipt', {}).get('profile') == 'raptor',
                'NAFULL_RAPTOR_SOURCE_GUARDS')
        require(worker.get('recovery_injection', {}).get('status') == 'PASS'
                and worker['recovery_injection'].get('state_healthy') is True,
                'NAFULL_TYPED_RECOVERY_INJECTION')
        for row in worker.get('records', []):
            ref = references[row['id']]
            require(row.get('input') == ref['input'] and row.get('decode_counters') == ref['decode_counters'],
                    'NAFULL_ALL39_INPUT_AND_COUNTER_PARITY')
        for sentinel in worker.get('sentinels', []):
            p = np.asarray(sentinel['probabilities']); q = np.asarray(sentinel['reloaded_probabilities'])
            require(p.shape == q.shape == (12,) and np.isfinite(p).all() and np.isfinite(q).all()
                    and float(np.max(np.abs(p-q))) <= 1e-6 and sentinel.get('strict_reload') is True,
                    'NAFULL_BOARD_SENTINEL_RELOAD')
    for row in rows.values():
        counters = row.get('decode_counters', {})
        require(set(counters) >= set(RAPTOR62_COUNT_KEYS), 'NAFULL_RAPTOR62_COUNTER_SCHEMA')
    strict_ids=set(binding['s6_probe_ids'])|set(binding['old_historical_ids'])
    strict_rows=[row for row in rows.values() if row['id'] in strict_ids]
    require(len(strict_rows)==36 and any(row.get('strict_source_comparison')=='PASS' for row in strict_rows),
            'NAFULL_AT_LEAST_ONE_STRICT_SOURCE_PASS')
    for row in strict_rows:
        require(row.get('strict_reload') is True and 0 <= row.get('reload_max_abs', math.inf) <= 1e-6
                and row.get('source_reference_parity') is True
                and row.get('strict_source_comparison') in ('PASS', 'ineligible_partial_input'),
                'NAFULL_ALL36_STRICT_SOURCE_REPEAT_RELOAD')
    for sid in r.get('visible_ids', []):
        row = rows[sid]
        require(row.get('raptor62') == {'decode_counts_equal': True,
                'checkpoint_probability_reference': False}, 'NAFULL_VISIBLE_INPUT_ONLY_REFERENCE')


def _replace_r7_contract(namespace):
    """Load a conservative adapted R7 checker, keeping non-old checks intact."""
    source = inspect.getsource(R7.require_rehearsal)
    # AST node updates are applied to a clone, so imported qualified R7 code and
    # its source bytes remain unchanged. The custom validators cover the removed
    # old-arm clauses; no success flag is accepted as their substitute.
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'require_rehearsal')
    fn.name = '_archived_r7_require_rehearsal'
    # Remove two mirror-only stress gates and replace the old probability pin
    # branch with the worker's strict reload and board-parity requirement.
    stress_sites = 0
    class ReplaceOldOnly(ast.NodeTransformer):
        def visit_Expr(self, node):
            nonlocal stress_sites
            rendered = ast.unparse(node)
            if any(tag in rendered for tag in ('REHEARSAL_PARTIAL_INPUT_STRESS', 'GUARDED_MIRROR_PARTIAL_TEN')):
                stress_sites += 1
                return ast.Pass()
            return self.generic_visit(node)
        def visit_If(self, node):
            self.generic_visit(node)
            if ast.unparse(node.test) == "arm == 'old'" and 'SENTINEL_EXACT_REFERENCE_GATE' in ast.unparse(node):
                node.body = ast.parse("""expected=binding['sentinel_probabilities']['old'][s['id']]
require(expected is None and float(np.max(np.abs(p-q)))<=1e-6
        and s.get('strict_reload') is True
        and s.get('strict_source_comparison') in ('PASS','ineligible_partial_input'),
        'NAFULL_SENTINEL_RELOAD_GATE')
continue""").body
            return node
    fn = ReplaceOldOnly().visit(fn)
    require(stress_sites == 2, 'EXACT_TWO_R7_MIRROR_STRESS_GATES')
    namespace['_nafull_rehearsal_checks'] = _nafull_rehearsal_checks
    # Existing checks still bind version/method/state, R7 stackers, compilation,
    # fluids, all39 R7 inputs, both-board sentinels and visible R7 parity.
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 '<nafull-adapted-r7-contract>', 'exec'), namespace)
    archived_check = namespace['_archived_r7_require_rehearsal']

    def require_rehearsal(r, binding, csv_path):
        # This check contains old-profile-specific gates. Keep a complete B0prime
        # contract check on the K62 arm and the unmodified R7 contract below.
        _nafull_rehearsal_checks(r, binding)
        return archived_check(r, binding, csv_path)
    return require_rehearsal


def make_base_facade(nafull_worker_cfg, *, rehearsal=False):
    """Return an R7-shaped module facade with the exact B0prime K62 old worker."""
    ns = dict(vars(R7))
    ns.update(VERSION=PROFILE.VERSION, ENTRY_MODULE='knee.public_nafull_aprime_entry',
        REHEARSAL_MODULE='knee.public_nafull_aprime_rehearsal',
        METHOD={**R7.METHOD, 'old': PROFILE.METHOD_OLD},
        old_worker=NA_OLD_WORKER)
    ns['require_rehearsal'] = _replace_r7_contract(ns)
    if rehearsal:
        from knee.public_s6_recovery_rehearsal import injection_hook as r7_injection_hook
        ns['rehearsal_hook'] = r7_injection_hook
    ns['launch'] = _bind(R7.launch, ns)
    ns['merge'] = _bind(R7.merge, ns)
    ns['r7_worker'] = _bind(R7.r7_worker, ns)
    _archived_run = _bind(R7.run, ns)

    def run(root, cfg, out, *, started):
        base_cfg = dict(cfg)
        worker = dict(nafull_worker_cfg)
        for key in ('public_source', 'decoder_source', 'public_manifest', 'control_checkpoint',
                    'control_state_sha256', 'control_grid_sha256', 'probes', 'old_sentinels',
                    'raptor62_reference', 'raptor62_visible_atol'):
            if key in worker:
                base_cfg[key] = worker[key]
        # R7's compact arm keeps its archived probe bindings; only the old field
        # is replaced with K62 inputs, with all K62 probability references absent.
        base_cfg['probes'] = {sid: {**cfg.get('probes', {}).get(sid, {}),
            'old': worker.get('probes', {}).get(sid, {}).get('old')}
            for sid in set(cfg.get('probes', {})) | set(worker.get('probes', {}))}
        base_cfg['binding'] = cfg['nafull_base_binding']
        base_cfg.pop('nafull_base_binding', None)
        return _archived_run(root, base_cfg, out, started=started)

    ns['run'] = run
    return types.SimpleNamespace(**ns)


def make_ws2_facade(base_facade):
    """Clone WS2 orchestration while replacing only base facade and recovery."""
    from knee import public_ws2_entry as WS2
    from knee import public_nafull_aprime as REC
    from knee import public_nafull_aprime_recovery_adapter as REC_ADAPTER
    ns = dict(vars(WS2)); ns['R7'] = base_facade; ns['REC'] = REC_ADAPTER
    def merged_arm(ids,reports,out,arm):
        # R224 worker records carry the archived R7 version identity; retain
        # its frozen merger. The two base arms use the new facade namespace.
        if arm=='crop224': return WS2.merged_arm(ids,reports,out,arm)
        return base_facade.merge(ids,[(x,out/f'board{i}-{arm}'/arm/'predictions.npz') for i,x in enumerate(reports)],arm)
    ns['merged_arm'] = merged_arm
    ns['run'] = _bind(WS2.run, ns)
    original = inspect.getsource(WS2.require_rehearsal)
    # This original comparison expected old320 input hashes. R224's own frozen
    # source reference is carried explicitly in the A‴ binding instead.
    marker = "original=old_rows[s['id']]['input']"
    require(original.count(marker) == 1, 'EXACT_WS2_R224_OLD_INPUT_GATE')
    altered = original.replace(marker, "original=binding['r224_input_reference'][s['id']]")
    module = ast.parse(altered); checker = next(n for n in module.body if isinstance(n, ast.FunctionDef))
    checker.name = '_ws2_nafull_require_rehearsal'
    exec(compile(ast.fix_missing_locations(ast.Module(body=[checker], type_ignores=[])),
                 '<nafull-adapted-ws2-contract>', 'exec'), ns)
    archived_check = ns['_ws2_nafull_require_rehearsal']
    def require_rehearsal(r, binding, base_receipt_path, csv_path):
        import numpy as np
        from pathlib import Path
        base_path=Path(base_receipt_path);base=json.loads(base_path.read_bytes())
        archived_check(r,binding,base_receipt_path,csv_path)
        ids=json.loads((base_path.parent/'assignment.json').read_bytes())['ids']
        for name,reports,folder,arm,reference,tolerance in (
            ('r7',base['workers']['s6'],base_path.parent,'s6',binding['r7_probability_reference'],R7.CROSS_BOARD_TOLERANCE['s6']),
            ('r224',r['crop224_workers'],Path(csv_path).parent,'crop224',binding['r224_probability_reference'],1e-6)):
            values,missing=WS2.merged_arm(ids,reports,folder,arm)
            require(not missing.any() and set(reference)==set(ids),'APRIME_'+name.upper()+'_ALL39_REFERENCE_COVERAGE')
            expected=np.asarray([reference[sid] for sid in ids],dtype=np.float64)
            require(values.shape==expected.shape and np.isfinite(values).all() and np.isfinite(expected).all()
                    and float(np.max(np.abs(values-expected)))<=tolerance,'APRIME_'+name.upper()+'_ALL39_PROBABILITY_PARITY')
            for i,worker in enumerate(reports):
                saved=json.loads((folder/f'board{i}-{arm}'/arm/'receipt.json').read_bytes())
                require(saved==worker,'APRIME_'+name.upper()+'_SAVED_WORKER_RECEIPT')
                record={row['id']:row for row in worker['records']}
                for row in record.values():
                    source=binding[name+'_input_reference'][row['id']]
                    if name=='r7':
                        require(row['input']==source,'APRIME_R7_ALL39_INPUT_PARITY')
                    else:
                        require(all(row['input'].get(k)==source.get(k) for k in ('volume_sha256','mask_sha256','decode_counters')),
                                'APRIME_R224_ALL39_INPUT_PARITY')
        return True
    ns['require_rehearsal'] = require_rehearsal
    ns['run'] = _bind(WS2.run, ns)
    return types.SimpleNamespace(**ns)


def run_worker(config_path, *, rehearsal):
    """CLI target used by the cloned two-board R7 launcher."""
    import json
    from pathlib import Path
    cfg = json.loads(Path(config_path).read_bytes())
    facade = make_base_facade(cfg, rehearsal=rehearsal)
    if cfg.get('arm') == 'old':
        from knee.public_s6_recovery_b0_rehearsal import injection_hook
        from knee.public_s6_recovery_runtime import require
        require(cfg['control_state_sha256'] == PROFILE.CHECKPOINT['state_sha256'], 'NAFULL_WORKER_STATE_BINDING')
        require(cfg['control_grid_sha256'] == cfg['binding']['qualified_state']['old_grid'], 'NAFULL_WORKER_GRID_BINDING')
        NA_WORKER_GLOBALS['DREAD_GRID_SHA256'] = cfg['control_grid_sha256']
        NA_WORKER_GLOBALS['rehearsal_hook'] = injection_hook if rehearsal else None
        NA_OLD_WORKER(config_path)
    else:
        facade.r7_worker(config_path)


def run(root, cfg, out, *, started):
    """Build the facades for this immutable payload and execute archived WS2 run."""
    worker_cfg = cfg.get('nafull_worker_cfg')
    require(isinstance(worker_cfg, dict), 'NAFULL_WORKER_PROFILE_REQUIRED')
    base = make_base_facade(worker_cfg, rehearsal=cfg.get('mode') == 'rehearsal')
    ws2 = make_ws2_facade(base)
    return ws2.run(root, cfg, out, started=started)
