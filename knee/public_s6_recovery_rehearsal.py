"""Rehearsal-only faults. Never packaged in a pure recovery entry."""
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def guarded_mirror_partial(public,decoder):
    """Ten declared partial-input cases using the actual installed guarded reader.

    This deliberately does not relabel the old seventeen-case baseline as a
    guarded suite. Typed failure and fatal cases have separate regression gates.
    """
    import ast,inspect,textwrap
    import knee.mirror_input as MI
    source=textwrap.dedent(inspect.getsource(MI.stress_inputs));tree=ast.parse(source)
    function=tree.body[0]
    block=next(n for n in function.body if isinstance(n,ast.With))
    first_failure=next(i for i,n in enumerate(block.body) if isinstance(n,ast.Try))
    block.body=block.body[:first_failure]
    namespace=dict(MI.__dict__)
    namespace['STRESS_CASES']=frozenset({'healthy','missing_iop','missing_ipp','missing_spacing','empty_series','missing_series','bad_header','bad_pixel','black_frame','short_series'})
    exec(compile(ast.fix_missing_locations(tree),'<guarded-mirror-six>','exec'),namespace)
    result=namespace['stress_inputs'](public,decoder)
    if result['cases']!=dict.fromkeys(namespace['STRESS_CASES'],'PASS'):raise ValueError('GUARDED_MIRROR_TEN')
    return dict(result,policy='classified-data-local-v1',guarded=True,case_count=10)


def injection_hook(arm, cfg, context):
    import numpy as np
    from knee.public_s6_recovery_entry import quantize
    from knee.public_s6_recovery_runtime import DataLocalFailure, cleanup_study
    if arm == 'old':
        import knee.mirror_input as MI
        infer = context['infer'];sid = cfg['old_sentinels'][0]
        genuine = context['sentinels'][0]['probabilities']
        original = MI.validate_production_volume
        def all_empty(volume, mask, plan, **kw):
            return original(np.zeros_like(volume), np.zeros_like(mask), plan, **kw)
        caught = False
        with patch.object(MI, 'validate_production_volume', side_effect=all_empty):
            try:infer(sid)
            except DataLocalFailure as e:
                caught = str(e) == 'old_no_usable_image'
        if not caught:raise ValueError('ACTUAL_OLD_ALL_EMPTY_INJECTION')
        context['state_check']()
        p, fp, row = infer(sid)
        if np.max(np.abs(p-np.asarray(genuine))) > 1e-6:raise ValueError('OLD_NEXT_HEALTHY_PARITY')
        context['state_check']()
        return {'status':'PASS','cases':['actual_old_all_empty','next_healthy_same_study'],
                'failure_status':'data_local_failure','injected_rows_genuine':False,
                'next_healthy_max_abs':float(np.max(np.abs(p-np.asarray(genuine)))), 'cleanup':{'scratch_removed':True,'side_override_removed':True},'state_healthy':True}
    CI, ds = context['CI'], context['ds']
    index = 0;sid=ds.uids[index]
    original_rows = ds.window_rows
    caught=False
    def no_rows(*a, **kw):
        CI.FORCED_SIDE[sid]='R'
        return []
    try:
        ds.window_rows=no_rows
        try:ds[index]
        except DataLocalFailure as e:
            caught=str(e)=='s6_no_usable_window'
    finally:
        ds.window_rows=original_rows
        clean=cleanup_study(ds.tmp,sid,CI.FORCED_SIDE)
    if not caught:raise ValueError('ACTUAL_S6_NO_WINDOW_INJECTION')
    context['state_check']()
    # Force permitted source diagnostics through the actual source record/build
    # paths, then require exact next item pixels/meta and subsequent main logits.
    healthy=ds[index]
    if healthy['error'] is not None or not len(healthy['images']):raise ValueError('S6_NEXT_HEALTHY_INPUT')
    expected=context['probes'][sid]['s6']['input']
    from knee.public_s6_evaluation import array_sha
    a=healthy['images'].numpy();meta=healthy['meta'].numpy()
    if array_sha(a)!=expected['render_windows'] or array_sha(meta)!=expected['metadata']:
        raise ValueError('S6_NEXT_HEALTHY_HASH_PARITY')
    cleanup_study(ds.tmp,sid,CI.FORCED_SIDE);context['state_check']()
    original_record=ds.bank.record;original_build=ds.builder.build
    def diagnostics_record(*a, **kw):
        value=original_record(*a, **kw);value['why']=list(value['why'])+['error:declared_data_local_diagnostic'];return value
    def diagnostics_build(*a, **kw):
        value=original_build(*a, **kw);value['skipped_series']=list(value['skipped_series'])+['declared_missing_companion'];return value
    try:
        ds.bank.record=diagnostics_record;ds.builder.build=diagnostics_build
        diagnostic=ds[index]
        if diagnostic['error'] is not None or not np.array_equal(diagnostic['images'].numpy(),a) or not np.array_equal(diagnostic['meta'].numpy(),meta):
            raise ValueError('SOURCE_VALID_DIAGNOSTICS_PARITY')
    finally:
        ds.bank.record=original_record;ds.builder.build=original_build;cleanup_study(ds.tmp,sid,CI.FORCED_SIDE)
    context['state_check']()
    return {'status':'PASS','cases':['actual_s6_no_window','forced_side_cleanup','next_healthy_input_parity','source_valid_diagnostics_parity'],
            'failure_status':'data_local_failure','injected_rows_genuine':False,'cleanup':clean,'state_healthy':True,
            'next_healthy_input':{'render_windows':array_sha(a),'metadata':array_sha(meta)},'main_rehearsal_following_logit_parity_required':True}


if __name__ == '__main__':
    import argparse
    from knee import public_s6_recovery_entry as entry
    if Path(__file__).resolve() != (Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_s6_recovery_rehearsal.py').resolve():
        raise ValueError('EXACT_REHEARSAL_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.worker).read_text())
    if cfg['mode']!='rehearsal':raise ValueError('INJECTIONS_REHEARSAL_ONLY')
    entry.rehearsal_hook=injection_hook
    (entry.old_worker if cfg['arm']=='old' else entry.s6_worker)(a.worker)
