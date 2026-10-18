"""B0 no-OAI entry: Dread CC0 Raptor v4 old arm, unchanged S6 arm and recovery.

The qualified default (nartaa original320 mirror) modules stay byte-identical and
are shipped unchanged. This module replaces only the old-arm worker, its source
guards, the launch module names and the rehearsal contract. The S6 worker, merge
and complete run core reuse the exact default bytecode against B0 globals, so the
fixed 70:30 rank blend, capped recovery and S6 pins cannot drift.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types
import knee.public_s6_recovery_entry as base
import knee.public_s6_recovery_runtime as runtime
from knee.public_s6_qualification import (LABELS,digest,file_sha,require,save,synthetic_source_checks,verify_assets,verify_sources)
from knee.public_s6_evaluation import array_sha,model_digest
from knee.public_adaptation_fit import PlatformInterruption,atomic_save
from knee.public_adaptation_pilot import state_fingerprint
from knee.public_fullfield_entry import read_series,test_contract,write_csv
from knee.public_s6_recovery import recover
from knee.public_s6_recovery_runtime import (DataLocalFailure,FatalSourceError,cleanup_study,install_source_guards,save_ledger)
from knee.public_s6_recovery_entry import (INPUT_POLICY,MAX_SECONDS,SENTINELS,assignments,board_environment,native_availability_stress,
    old_decode_gate,probabilities,probe_gate,projection,quantize,reset_native_renderer,resource_snapshot,stage,verify_staged,worker_environment)

VERSION='s6-dual-capped-recovery-dread-b0-v1'
ENTRY_MODULE='knee.public_s6_recovery_b0'
REHEARSAL_MODULE='knee.public_s6_recovery_b0_rehearsal'
DREAD_CHECKPOINT_SHA256='89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded'
DREAD_CHECKPOINT_BYTES=292829402
# CPU-derived from the verified local copy (scripts/qualify_b0_dread_cpu.py).
# The same CPU method reproduces the default nartaa constructor-grid pin exactly.
DREAD_STATE_SHA256='e0f0e7e1b10b03a4611283be8da4d95fb69d37f8e6d91929e5fd33d8291b62f2'
DREAD_GRID_SHA256='67ae62b6e4ba164abf4757e0948b4fa9cc410b587538dd19ce7be58186d465bd'
RAPTOR_WINDOWS=62
RAPTOR_INPUT=384
RAPTOR_IMAGE=336
RAPTOR_SLOTS=64
OLD_PROFILE={'profile':'raptor','checkpoint_sha256':DREAD_CHECKPOINT_SHA256,'checkpoint_bytes':DREAD_CHECKPOINT_BYTES,'windows':RAPTOR_WINDOWS,
             'input_res':RAPTOR_INPUT,'image_size':RAPTOR_IMAGE,'slots':RAPTOR_SLOTS,'slot_sizes':[18,14,12,8,12],'slice_span':[.06,.94],
             'tta':'none','precision':'CUDA fp16; no fp32 fallback','decoder':'memoized pydicom (raptor_input.prepare_study)',
             'header_data_local':['ValueError','TypeError']}
METHOD={**base.METHOD,'old':'Dread CC0 raptor v4 widedense: 64 slots 18/14/12/8/12, span 6-94%, 336 px bilinear to 384, K62, no TTA, CUDA fp16','old_profile':'raptor'}
# Exact protected Dread catch functions mapped onto the default classified scopes:
# header reads accept only InvalidDicomError; pixel reads also the audited pydicom
# buffer error. Every other source exception remains fatal, as in the default arm.
PUBLIC_CATCH_SCOPES={'order_and_meta':'old_public:_hdr','build_study':'old_public:_px'}
HEADER_DATA_LOCAL={'scope':'old_public:_hdr','exact_types':['ValueError','TypeError'],'code':'old_header_geometry_malformed',
                   'site':'dread:_make_reader.order_and_meta','outcome':'whole-study data_local_failure; unchanged capped global arm-mean recovery'}
ALL_EMPTY_SITE='raptor_input.py:validate_production_volume:164'
GUARDED_PARTIAL_CASES=frozenset({'healthy','missing_iop','missing_ipp','missing_spacing','empty_series','missing_series','bad_header','bad_pixel','black_frame','short_series'})
RAPTOR62_COUNT_KEYS=('headers','header_parse_failures','geometry_defaults','spacing_defaults','strict_ineligible_headers','pixel_failures','decoded_unique_slices',
    'empty_series','resource_errors','invalid_pixels','valid_slices','missing_slots','selected_slots','masked_selected_frames','partial_slots',
    'empty_selected_slots','zero_studies','sequence_preference_substitutions')
rehearsal_hook=None


def raptor_profile_gate(public,manifest):
    """Explicit B0 profile selection; any nartaa/mirror manifest is rejected."""
    from knee.reference import RAPTOR_WEIGHT_DATASET,RAPTOR_WEIGHT_NAME
    recipe=manifest.get('recipe',{})
    require(manifest.get('profile')=='raptor' and manifest.get('checkpoint_sha256')==DREAD_CHECKPOINT_SHA256 and manifest.get('checkpoint_file')==RAPTOR_WEIGHT_NAME
            and manifest.get('weight_dataset')==RAPTOR_WEIGHT_DATASET and manifest.get('decoded_fastread_sha256') is None,'B0_DREAD_RAPTOR_MANIFEST')
    require(recipe.get('windows')==RAPTOR_WINDOWS and recipe.get('input_res')==RAPTOR_INPUT and recipe.get('image_size')==RAPTOR_IMAGE and recipe.get('input_mode')=='resize'
            and recipe.get('slice_span')==OLD_PROFILE['slice_span'] and [s[2] for s in recipe.get('slots',[])]==OLD_PROFILE['slot_sizes'],'B0_RAPTOR62_RECIPE')
    if public is not None:
        require(public.K_EVAL==RAPTOR_WINDOWS and public.IMG==RAPTOR_IMAGE and public.MAXS==RAPTOR_SLOTS and [list(s) for s in public.SLOTS]==recipe['slots']
                and list(public.LAB)==LABELS and public.NORM=='imagenet','B0_RAPTOR_SOURCE_CONSTANTS')
    return dict(OLD_PROFILE)


def raptor_windows(public,volume,mask):
    """Exact public CPU eval_windows at K62/384; no mirror, no crop."""
    import torch
    w=public.eval_windows(volume,mask,RAPTOR_WINDOWS,RAPTOR_INPUT)
    require(tuple(w.shape)==(RAPTOR_WINDOWS,3,RAPTOR_INPUT,RAPTOR_INPUT) and w.dtype==torch.float32 and bool(torch.isfinite(w).all()),'RAPTOR62_WINDOW_SCHEMA')
    return w


def raptor_forward(public,model,windows,device):
    """One plain pass through infer_checked(mirror=False): fp16 autocast, nonfinite fatal."""
    import numpy as np
    from knee.reference import infer_checked
    p=np.asarray(infer_checked(public,model,[windows.to(device)],device,mirror=False),dtype=np.float32)
    probabilities([p],1);return p


def install_raptor_old_guards(public):
    """Raptor-profile counterpart of runtime.install_old_guards.

    Patches raptor_input.validate_production_volume so only the all-empty
    invariant becomes a typed data-local failure, and routes both protected Dread
    catches through the unchanged classified runtime policy. There is no pinned
    fast decoder, so no NeedMore parser class is registered.
    """
    import ast
    import inspect
    import textwrap
    import knee.raptor_input as RI
    from pydicom.errors import InvalidDicomError
    require(callable(getattr(public,'_raptor_propagate_fatal',None)) and isinstance(getattr(public,'_raptor_protected_source',None),str),'RAPTOR_PROTECTED_SOURCE_REQUIRED')
    class Sites(ast.NodeVisitor):
        def __init__(self):self.stack=[];self.found=[]
        def visit_FunctionDef(self,node):self.stack.append(node.name);self.generic_visit(node);self.stack.pop()
        def visit_ExceptHandler(self,node):self.found.append(self.stack[-1]);self.generic_visit(node)
    sites=Sites();sites.visit(ast.parse(public._raptor_protected_source))
    require(sorted(sites.found)==sorted(PUBLIC_CATCH_SCOPES),'EXACT_RAPTOR_CATCH_SITES')
    runtime.REGISTERED_DICOM_CLASS=InvalidDicomError
    pixel_guard_sha=runtime.install_pixel_buffer_guard()
    original=getattr(RI,'_b0_source_validate_production_volume',RI.validate_production_volume)
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    raises=[x for x in ast.walk(tree) if isinstance(x,ast.Raise) and isinstance(x.exc,ast.Call) and isinstance(x.exc.func,ast.Name) and x.exc.func.id=='ValueError' and 'Raptor study has no usable image content:' in ast.unparse(x.exc)]
    require(len(raises)==1,'EXACT_OLD_ALL_EMPTY_SITE')
    detail=raises[0].exc.args[0]
    raises[0].exc=ast.Call(ast.Name('_s6_data_local_failure',ast.Load()),[ast.Constant('old_no_usable_image')],[ast.keyword('site',ast.Constant(ALL_EMPTY_SITE)),ast.keyword('detail',detail)])
    transformed=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    namespace=dict(original.__globals__);namespace['_s6_data_local_failure']=DataLocalFailure
    exec(compile(transformed,'<b0-raptor-all-empty>','exec'),namespace)
    RI._b0_source_validate_production_volume=original;RI.validate_production_volume=namespace['validate_production_volume']
    public._raptor_propagate_fatal=public_guard
    return {'profile':'raptor','all_empty_guard_sha256':hashlib.sha256(transformed.encode()).hexdigest(),'pixel_buffer_guard_sha256':pixel_guard_sha,
            'catch_scopes':dict(PUBLIC_CATCH_SCOPES),'header_data_local':HEADER_DATA_LOCAL,'unknown_source_exceptions':'fatal'}


def public_guard(error):
    """Classified policy for the two protected Dread catches.

    Exactly the default runtime policy, plus one B0 addition: at the header scope
    only, an exact ValueError/TypeError (malformed but readable geometry tags in
    Dread's order_and_meta try body) with no fatal cause becomes a typed
    whole-study data-local failure, filled by the unchanged capped recovery.
    """
    scope=PUBLIC_CATCH_SCOPES.get(sys._getframe(1).f_code.co_name)
    if scope is None:raise FatalSourceError(error) from error
    try:runtime.guard_source_error(error,scope)
    except FatalSourceError as fatal:
        # fatal.error is error only when the default rejected the class itself,
        # not a fatal cause/context (resource, KeyError, ...) in its chain.
        if scope==HEADER_DATA_LOCAL['scope'] and fatal.error is error and type(error).__name__ in HEADER_DATA_LOCAL['exact_types'] and type(error) in (ValueError,TypeError):
            raise DataLocalFailure(HEADER_DATA_LOCAL['code'],site=HEADER_DATA_LOCAL['site'],detail=type(error).__name__) from error
        raise


def raptor_stress(public):
    """Unchanged sixteen-case raptor_input stress, before recovery guards."""
    from knee import raptor_input as RI
    result=RI.stress_inputs(public)
    require(result.get('cases')==dict.fromkeys(RI.STRESS_CASES,'PASS') and result.get('uses_raw_data') is False and result.get('uses_labels') is False,'ACTUAL_RAPTOR_STRESS_REQUIRED')
    return {'status':'PASS',**result,'case_count':len(result['cases'])}


def raptor62_gate(sid,p,counters,cfg):
    """Rehearsal-only receipt parity for the three visible studies.

    Decoding is deterministic CPU work: counts must equal the raptor62 receipt.
    Probabilities ran with different cuDNN settings, so only a declared bound applies.
    """
    import numpy as np
    reference=cfg.get('raptor62_reference',{}).get(sid)
    if reference is None:return None
    require(all(counters.get(k)==v for k,v in reference['counts'].items()) and set(reference['counts'])==set(RAPTOR62_COUNT_KEYS),'RAPTOR62_VISIBLE_DECODE_COUNTS')
    err=float(np.max(np.abs(np.asarray(p,dtype=np.float64)-np.asarray(reference['probabilities'],dtype=np.float64))));atol=cfg['raptor62_visible_atol']
    return {'decode_counts_equal':True,'max_abs':err,'atol':atol,'within_atol':err<=atol}


def old_worker(config_path):
    import numpy as np
    import torch
    import resource
    from knee.reference import load_definitions,load_model
    from knee import raptor_input as RI
    from knee.public_fullfield import GRID_NAMES,snapshot,require_immutable
    cfg=json.loads(Path(config_path).read_bytes());out=Path(cfg['out'])/'old';out.mkdir();start=time.monotonic()
    r={'status':'RUNNING','version':VERSION,'board':cfg['board'],'arm':'old','profile':'raptor','assigned':cfg['dev'],'completed':0,'records':[],'sentinels':[]};save(out/'receipt.json',r)
    try:
        r['environment']=board_environment(torch,cfg);verify_staged(cfg['staged'],cfg['staged_ids'],cfg['stage_binding'])
        require(cfg['decoder_source']=='','RAPTOR_HAS_NO_PINNED_DECODER')
        public,decoder=load_definitions(cfg['public_source'],cfg['decoder_source'],cfg['public_manifest']);require(decoder is None,'RAPTOR_HAS_NO_PINNED_DECODER')
        r['old_profile']=raptor_profile_gate(public,cfg['public_manifest'])
        if cfg['mode']=='rehearsal' and cfg['board']==0:r['partial_input_stress']=dict(raptor_stress(public),guarded=False,policy='baseline_source_before_recovery_guards');save(out/'receipt.json',r)
        r['source_guard_receipt']=install_raptor_old_guards(public)
        if cfg['mode']=='rehearsal' and cfg['board']==0:
            from knee.public_s6_recovery_b0_rehearsal import guarded_raptor_partial
            r['guarded_partial_input_stress']=guarded_raptor_partial(public);save(out/'receipt.json',r)
        require(Path(cfg['control_checkpoint']).stat().st_size==DREAD_CHECKPOINT_BYTES,'DREAD_CHECKPOINT_BYTES')
        model=load_model(public,cfg['control_checkpoint'],'cuda',cfg['public_manifest']);before=snapshot(model);r['state_before']=before
        require(before['state']==cfg['control_state_sha256']==DREAD_STATE_SHA256,'QUALIFIED_OLD_STATE')
        grid={k:v.detach().cpu().clone() for k,v in model.named_buffers() if k in GRID_NAMES};r['constructor_grid_sha256']=state_fingerprint(grid)
        require(len(grid)==len(GRID_NAMES) and r['constructor_grid_sha256']==cfg['control_grid_sha256']==DREAD_GRID_SHA256,'OLD_CONSTRUCTOR_GRID')
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
        records=read_series(Path(cfg['staged'])/'train_series.csv');frozen=out/'frozen-state.pt';atomic_save(frozen,{k:v.detach().cpu() for k,v in model.state_dict().items()})
        def infer(sid):
            t=time.monotonic();root=Path(cfg['staged'])/'train_series';probe=sid in cfg.get('probes',{})
            v,m,counters=RI.prepare_study(public,sid,records,root)
            old_decode_gate(counters)
            reference_seconds=0.;strict=None;rc=None
            if probe:
                reference_start=time.monotonic()
                rv,rm,rc=RI.prepare_study(public,sid,records,root,cache_pixels=False,reference=True)
                require(np.array_equal(v,rv) and np.array_equal(m,rm),'OLD_ORIGINAL_SOURCE_REFERENCE_INPUT');old_decode_gate(rc)
                strict='ineligible_partial_input'
                if RI.strict_eligible(counters):RI.strict_compare(public,sid,records,root,v,m);strict='PASS'
                refw=raptor_windows(public,rv,rm);refp=raptor_forward(public,model,refw,'cuda');torch.cuda.synchronize();reference_seconds=time.monotonic()-reference_start
            w=raptor_windows(public,v,m);p=raptor_forward(public,model,w,'cuda')
            if probe:require(torch.equal(w,refw) and float(np.max(np.abs(p-refp)))<=1e-6,'OLD_ORIGINAL_SOURCE_REFERENCE_PROBABILITY')
            fp={'volume_sha256':hashlib.sha256(v.tobytes()).hexdigest(),'mask_sha256':hashlib.sha256(m.tobytes()).hexdigest(),'raptor62_windows':state_fingerprint(w)}
            torch.cuda.synchronize();return p,fp,{'seconds':time.monotonic()-t,'production_seconds':time.monotonic()-t-reference_seconds,'reference_seconds':reference_seconds,
                'decode_counters':counters,'reference_decode_counters':rc,'strict_source_comparison':strict}
        ps=[];statuses=[]
        with PlatformInterruption() as stop:
            for sid in cfg.get('old_sentinels',[]):
                p,fp,d=infer(sid);probe_gate(sid,'old',p,fp,cfg)
                model.load_state_dict(torch.load(frozen,map_location='cpu',weights_only=True),strict=True);model.eval();require_immutable(model,before)
                q,g,_=infer(sid);probe_gate(sid,'old',q,g,cfg)
                require(g==fp and float(np.max(np.abs(p-q)))<=1e-6,'OLD_FRESH_DECODE_RELOAD')
                r['sentinels'].append({'id':sid,'input':fp,'probabilities':p.tolist(),'reloaded_probabilities':q.tolist(),'strict_reload':True,'strict_source_comparison':d['strict_source_comparison']});save(out/'receipt.json',r)
                if stop.requested:raise InterruptedError('Interrupted sentinel')
            if cfg['mode']=='rehearsal':
                require(rehearsal_hook is not None,'FRESH_INJECTION_EXTENSION_REQUIRED')
                r['recovery_injection']=rehearsal_hook('old',cfg,{'infer':infer,'sentinels':r['sentinels'],'state_check':lambda:require_immutable(model,before)});save(out/'receipt.json',r)
            for sid in cfg['dev']:
                study_started=time.monotonic()
                try:
                    r['current_id']=sid;save(out/'receipt.json',r);p,fp,d=infer(sid);err=probe_gate(sid,'old',p,fp,cfg)
                    if sid in cfg.get('probes',{}):
                        model.load_state_dict(torch.load(frozen,map_location='cpu',weights_only=True),strict=True);model.eval();require_immutable(model,before)
                        q,g,_=infer(sid);probe_gate(sid,'old',q,g,cfg);require(fp==g and float(np.max(np.abs(p-q)))<=1e-6,'OLD32_REPEAT_RELOAD')
                        d.update(source_reference_parity=True,strict_reload=True,reload_max_abs=float(np.max(np.abs(p-q))))
                    receipt62=raptor62_gate(sid,p,d['decode_counters'],cfg)
                    if receipt62 is not None:d['raptor62']=receipt62
                    ps.append(p);statuses.append('genuine');row={'id':sid,'status':'genuine','input':fp,'qualified_error':err,**d}
                except DataLocalFailure as e:
                    require(cfg['mode']!='rehearsal','HEALTHY39_MUST_BE_GENUINE')
                    ps.append(np.full(12,np.nan));statuses.append('data_local_failure')
                    row={'id':sid,'status':'data_local_failure','phase':'old_input','failure_code':str(e),'production_seconds':time.monotonic()-study_started}
                    save(out/('failure-'+str(len(ps)-1)+'-private.json'),{'id':sid,'type':type(e).__name__,'detail':e.detail,'site':e.site,'phase':'old_input'})
                finally:
                    state_started=time.monotonic()
                    if sys.exc_info()[0] is None:require_immutable(model,before)
                    state_seconds=time.monotonic()-state_started
                require(statuses.count('data_local_failure')<=math.floor(3*len(cfg['staged_ids'])/100),'GLOBAL_UNION_CAP_ALREADY_EXCEEDED')
                row['production_seconds']+=state_seconds
                row.update(cleanup={'scratch_removed':True,'side_override_removed':True},state_healthy=True)
                r['records'].append(row);r['completed']=len(ps)
                save_ledger(out/'predictions.npz',cfg['dev'][:len(ps)],ps,statuses);save(out/'receipt.json',r)
                if stop.requested:raise InterruptedError('Interrupted study boundary')
        require_immutable(model,before);r.update(status='PASS',state_after=snapshot(model),zero_failed=all(s=='genuine' for s in statuses),genuine=statuses.count('genuine'),failed=statuses.count('data_local_failure'),zero_imputed=True,fp32_recoveries=0,
            ledger_sha256=file_sha(out/'predictions.npz'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        frozen.unlink()
    except BaseException as e:
        actual=e.error if isinstance(e,FatalSourceError) else e
        save(out/'failure-private.json',{'type':type(actual).__name__,'detail':str(actual),'boundary_type':type(e).__name__});r.update(status='INTERRUPTED' if isinstance(actual,(InterruptedError,KeyboardInterrupt,TimeoutError)) else 'FAIL',failure_type=type(actual).__name__)
        if actual is not e:raise actual from e
        raise
    finally:r['elapsed_seconds']=time.monotonic()-start;save(out/'receipt.json',r)


def launch(cfg,out,arm):
    """Default launch semantics with the B0 worker modules."""
    import signal
    children=[];start=time.monotonic();high={'host_rss_bytes':0,'output_disk_bytes':0};min_free=shutil.disk_usage(out).free
    try:
        with PlatformInterruption() as stop:
            for board in range(2):
                folder=out/f'board{board}-{arm}';folder.mkdir()
                c={**cfg,'board':board,'arm':arm,'dev':cfg['assignment'][board],'physical':cfg['boards'][board],'out':str(folder)}
                path=out/f'board{board}-{arm}-config.json';save(path,c)
                env=worker_environment(board)
                log=(out/f'board{board}-{arm}.log').open('xb')
                child=subprocess.Popen([sys.executable,'-m',(REHEARSAL_MODULE if cfg['mode']=='rehearsal' else ENTRY_MODULE),'--worker',str(path)],env=env,stdout=log,stderr=subprocess.STDOUT);log.close();children.append(child)
            while any(c.poll() is None for c in children):
                s=resource_snapshot(out)
                for k in high:high[k]=max(high[k],s[k])
                min_free=min(min_free,s['disk_free_bytes'])
                if stop.requested or any(c.poll() not in (None,0) for c in children):
                    for c in children:
                        if c.poll() is None:c.send_signal(stop.signum or signal.SIGTERM)
                    if stop.requested:raise InterruptedError('Platform interruption')
                    raise RuntimeError('Worker failure; no partial merge')
                time.sleep(.5)
            require(all(c.returncode==0 for c in children),'WORKER_EXIT_FAILURE')
        reports=[json.loads((out/f'board{b}-{arm}'/arm/'receipt.json').read_bytes()) for b in range(2)]
        return reports,{'wall_seconds':time.monotonic()-start,'sample_interval_seconds':.5,'peaks':high,'minimum_disk_free_bytes':min_free}
    finally:
        for c in children:
            if c.poll() is None:c.terminate()
        for c in children:
            try:c.wait(timeout=45)
            except subprocess.TimeoutExpired:c.kill();c.wait()


def require_rehearsal(r,binding,csv_path):
    """B0 rehearsal contract: default gates, with Dread qualification in place of nartaa history."""
    import csv
    import numpy as np
    from knee import raptor_input as RI
    from knee.public_s6_qualification import UID
    require(binding.get('version')==VERSION and binding.get('method')==METHOD and binding.get('input_policy')==INPUT_POLICY and binding.get('old_profile')==OLD_PROFILE,'EXACT_FROZEN_METHOD')
    require(r.get('status')=='PASS' and r.get('version')==VERSION and r.get('mode')=='rehearsal' and r.get('binding')==binding,'PASS_EXACT_REHEARSAL_REQUIRED')
    require(r.get('completed')==39 and r.get('completed_visible')==3 and r.get('completed_probes')==36 and r.get('completed_s6_probes')==32 and r.get('completed_old_historical')==4 and r.get('all_input_state_probability_gates') is True and r.get('zero_failed') is True and r.get('zero_imputed') is True,'REHEARSAL_COMPLETE_GATES')
    require(len(r.get('boards',[]))==2 and len({x['uuid'] for x in r['boards']})==2 and all(x['name']=='Tesla T4' for x in r['boards']),'TWO_PHYSICAL_T4_GATES')
    require(0<r.get('projection',{}).get('conservative_seconds',math.inf)<MAX_SECONDS and 0<r.get('elapsed_seconds',math.inf)<MAX_SECONDS,'NINE_HOUR_PROJECTION_GATE')
    require(r.get('csv_sha256')==file_sha(csv_path),'REHEARSAL_CSV_BINDING')
    with Path(csv_path).open(newline='') as h:rows=list(csv.reader(h))
    require(rows[0]==[UID]+LABELS and len(rows)==4 and [x[0] for x in rows[1:]]==r['visible_ids'],'VISIBLE_CSV_COVERAGE')
    probabilities([[float(x) for x in row[1:]] for row in rows[1:]],3)
    old0=r.get('workers',{}).get('old',[{}])[0]
    stress=old0.get('partial_input_stress',{})
    require(stress.get('status')=='PASS' and stress.get('cases')==dict.fromkeys(RI.STRESS_CASES,'PASS') and stress.get('case_count')==16 and stress.get('uses_raw_data') is False and stress.get('uses_labels') is False,'REHEARSAL_RAPTOR_PARTIAL_INPUT_STRESS')
    guarded=old0.get('guarded_partial_input_stress',{})
    require(guarded.get('status')=='PASS' and guarded.get('guarded') is True and guarded.get('case_count')==10 and guarded.get('cases')==dict.fromkeys(GUARDED_PARTIAL_CASES,'PASS'),'GUARDED_RAPTOR_PARTIAL_TEN')
    s60=r.get('workers',{}).get('s6',[{}])[0]
    guarded_native=s60.get('guarded_native_availability_stress',{})
    require(guarded_native.get('status')=='PASS' and guarded_native.get('guarded') is True and guarded_native.get('case_count')==2 and set(guarded_native.get('cases',{}))=={'absent','empty'},'GUARDED_NATIVE_PARTIAL_TWO')
    native=s60.get('native_availability_stress',{})
    require(native.get('status')=='PASS' and native.get('case_count')==2 and native.get('uses_raw_data') is False and native.get('uses_labels') is False and set(native.get('cases',{}))=={'absent','empty'} and all(x.get('status')=='PASS' and x.get('usable_series')==1 and x.get('slices',0)>0 for x in native['cases'].values()),'REHEARSAL_NATIVE_AVAILABILITY_STRESS')
    for arm in ('old','s6'):
        require(arm in r.get('phases',{}) and r['phases'][arm]['wall_seconds']>0 and r['phases'][arm]['peaks']['host_rss_bytes']>0 and r['phases'][arm]['peaks']['output_disk_bytes']>0 and r['phases'][arm]['minimum_disk_free_bytes']>0,'CONCURRENT_RESOURCE_MEASUREMENT')
        rr=r.get('workers',{}).get(arm,[]);require(len(rr)==2,'REHEARSAL_WORKERS')
        require(all(x.get('recovery_injection',{}).get('status')=='PASS' and x['recovery_injection'].get('failure_status')=='data_local_failure' and x['recovery_injection'].get('injected_rows_genuine') is False and x['recovery_injection'].get('state_healthy') is True for x in rr),'ACTUAL_CLASSIFIED_RECOVERY_INJECTIONS')
        for board,x in enumerate(rr):
            require(x['status']=='PASS' and x['board']==board and x['state_before']==x['state_after'] and x['completed']==len(x['assigned']) and x['peak_cuda_bytes']>0 and x['peak_rss_bytes']>0 and x['zero_failed'] is True and x['zero_imputed'] is True and x['fp32_recoveries']==0,'REHEARSAL_WORKER_STATE_RESOURCE')
            require((x['state_before']['state']==binding['qualified_state']['old_state'] and x.get('constructor_grid_sha256')==binding['qualified_state']['old_grid'] and x.get('old_profile')==OLD_PROFILE and x.get('source_guard_receipt',{}).get('profile')=='raptor') if arm=='old' else x['state_before']==binding['qualified_state']['s6_state'],'QUALIFIED_COMPLETE_STATE_BINDING')
            require(x['environment']['torch']==binding['environment']['torch'] and x['environment']['packages']==binding['environment']['packages'],'QUALIFIED_ENVIRONMENT_BINDING')
            require(x['environment']['physical_uuid']==r['boards'][board]['uuid'] and x['environment']['cuda_device_count']==1,'ISOLATED_BOARD_RUNTIME')
            require(len(x['sentinels'])==4 and [s['id'] for s in x['sentinels']]==r['sentinel_ids'][arm] and all(s['strict_reload'] for s in x['sentinels']),'BOTH_BOARD_SENTINELS')
            for s in x['sentinels']:
                require(all(s['input'].get(k)==v for k,v in binding['sentinel_inputs'][arm][s['id']].items()),'BOUND_SENTINEL_INPUTS')
                p=np.array(s['probabilities']);q=np.array(s['reloaded_probabilities'])
                if arm=='old':
                    # No Dread probabilities exist for these train IDs: fresh decode,
                    # strict reload and cross-board parity are the declared gates.
                    require(binding['sentinel_probabilities']['old'][s['id']] is None and float(np.max(np.abs(p-q)))<=1e-6 and s.get('strict_source_comparison') in ('PASS','ineligible_partial_input'),'OLD_SENTINEL_RELOAD_GATE')
                    continue
                expected=np.array(binding['sentinel_probabilities'][arm][s['id']])
                require(max(float(np.max(np.abs(p-expected))),float(np.max(np.abs(q-expected))),float(np.max(np.abs(p-q))))<=1e-6,'SENTINEL_EXACT_REFERENCE_GATE')
                require(np.array_equal(quantize([p]),quantize([expected])) and np.array_equal(quantize([q]),quantize([expected])),'SENTINEL_EXACT_ROUNDING_GATE')
                for field,target in [('logits',p),('reloaded_logits',q)]:
                    z=np.asarray(s[field],dtype=np.float32);require(z.shape==(4,12) and np.isfinite(z).all(),'SENTINEL_RAW_FOUR_LOGITS')
                    mean=z.mean(0).astype(np.float64);sigmoid=np.exp(-np.logaddexp(0.,-mean))
                    require(float(np.max(np.abs(sigmoid-target)))<=1e-6,'SENTINEL_MEAN_LOGIT_SIGMOID')
        for a,b in zip(rr[0]['sentinels'],rr[1]['sentinels']):
            require(a['input']==b['input'] and max(abs(x-y) for x,y in zip(a['probabilities'],b['probabilities']))<=1e-6,'CROSS_BOARD_SENTINEL_PARITY')
    oldrows={row['id']:row for worker in r['workers']['old'] for row in worker['records']}
    s6rows={row['id']:row for worker in r['workers']['s6'] for row in worker['records']}
    require(len(oldrows)==len(s6rows)==39 and set(oldrows)==set(s6rows),'COMPLETE39_MECHANICS')
    require(set(binding['s6_probe_ids']).isdisjoint(binding['old_historical_ids']) and not set(binding['s6_probe_ids']+binding['old_historical_ids'])&set(r['visible_ids']),'DISJOINT39_MECHANICS')
    strict=[]
    for sid in binding['s6_probe_ids']+binding['old_historical_ids']:
        row=oldrows[sid];strict.append(row.get('strict_source_comparison'))
        require(row.get('source_reference_parity') is True and row.get('strict_reload') is True and 0<=row.get('reload_max_abs',math.inf)<=1e-6 and row.get('strict_source_comparison') in ('PASS','ineligible_partial_input'),'ALL36_OLD_SOURCE_STRICT_REPEAT_RELOAD')
    require('PASS' in strict,'STRICT_PYDICOM_SOURCE_COMPARISON_EXERCISED')
    for sid in binding['s6_probe_ids']:
        require(s6rows[sid].get('qualified_error') is not None and 0<=s6rows[sid]['qualified_error']<=1e-6,'ALL32_S6_PRIOR_PREDICTION')
    contract=binding['old_qualification']['raptor62']
    for sid in r['visible_ids']:
        g=oldrows[sid].get('raptor62',{})
        require(g.get('decode_counts_equal') is True and g.get('atol')==contract['visible_atol'] and g.get('within_atol') is True and 0<=g.get('max_abs',math.inf)<=contract['visible_atol'],'RAPTOR62_VISIBLE_RECEIPT_PARITY')


def _bind(function):
    """Exact default bytecode, resolved against B0 globals (version, launch, contract)."""
    clone=types.FunctionType(function.__code__,globals(),function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=function.__kwdefaults__;clone.__doc__=function.__doc__;clone.__qualname__=function.__qualname__
    return clone


s6_worker=_bind(base.s6_worker)
merge=_bind(base.merge)
run=_bind(base.run)


if __name__=='__main__':
    import argparse
    require(Path(__file__).resolve()==(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_s6_recovery_b0.py').resolve(),'EXACT_FRESH_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args();cfg=json.loads(Path(a.worker).read_bytes())
    (old_worker if cfg['arm']=='old' else s6_worker)(a.worker)
