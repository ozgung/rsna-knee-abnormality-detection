"""R7 entry: unchanged original320 old arm and recovery, heliosli R7 in the compact slot.

The qualified default (0.954) modules stay byte-identical and are shipped unchanged.
The old-arm worker, merge and complete run core reuse the exact default bytecode
against this module's globals (the B0 pattern), so the nartaa original320 K94
mirror arm, the 2xT4 sharding, the classified source guards, the 3% capped
arm-mean recovery and the submission schema checks cannot drift. Only the compact
slot (still named 's6' in ledgers and receipts) changes:

* heliosli R7 four-fold ConvNeXt-small bundle, dataset version 1, bundle 8c7b666e27dc0e8f;
* R7 compact_infer.py / compact_train.py (embedded in heliosli's 0.957 notebook);
* the bundle's fluid stacker (Effusion) and state stacker (Baker's), applied as in
  compact_infer v4c (port in knee/public_r7_stackers.py, cross-checked at runtime);
* blend 0.5/0.5 per finding (OA findings also 0.5) with R7 values at '%.9g'.

Declared deviations from heliosli's notebook (see docs/experiments/r7-recovery-entry-20261013.md):
torch.compile is applied in place (nn.Module.compile, same inductor mode, shapes and gate) so
parameter names and state digests are unchanged; the check runs on the first item before any
inference, and a failed check or failing compiled forward is fatal (heliosli continues eager, which
projects past the nine-hour limit). fluid_d2's catch-all is replaced by an allowlist boundary:
typed data-local errors become capped study failures, every other exception is fatal (heliosli
scores any fluid exception on the missing-volume path). Non-finite fp16 output and stacker exceptions are fatal instead of an
fp32 rescue / model-probability fallback, and our 9 h hard limit plus the
rehearsal projection gate replace the 8.3 h deadline guard, whose unreached-study
fallback (other model's rank for up to 50% of studies) is incompatible with the
frozen 3% capped recovery.
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
from knee import public_r7_stackers as stackers
from knee.public_s6_qualification import (LABELS,RECIPE as S6_RECIPE,digest,file_sha,require,save,synthetic_source_checks,verify_assets,verify_sources)
from knee.public_s6_evaluation import array_sha,model_digest
from knee.public_adaptation_fit import PlatformInterruption,atomic_save
from knee.public_adaptation_pilot import state_fingerprint
from knee.public_fullfield_entry import read_series,test_contract,write_csv
from knee.public_r7_recovery import recover,RECIPE as RECOVERY_RECIPE
from knee.public_s6_recovery_runtime import (DataLocalFailure,FatalSourceError,cleanup_study,install_old_guards,install_source_guards,save_ledger)
from knee.public_s6_recovery_entry import (INPUT_POLICY,MAX_SECONDS,SENTINELS,assignments,board_environment,mirror_stress,native_availability_stress,
    old_decode_gate,probabilities,probe_gate,projection,quantize,reset_native_renderer,resource_snapshot,stage,verify_staged,window_audit,worker_environment)

VERSION='r7-dual-capped-recovery-v1'
ENTRY_MODULE='knee.public_r7_recovery_entry'
REHEARSAL_MODULE='knee.public_r7_recovery_rehearsal'
R7_DATASET='heliosli/rsna-knee-compact-r7cnxfsss'
R7_DATASET_VERSION=1
R7_MANIFEST_SHA='c4bb0f7399ada9a020e0c15cae6252033cf0c4335e999961793a176063074ec6'
R7_BUNDLE_ID='8c7b666e27dc0e8f'
R7_TRAINER_MD5='8622009e3db0414124d5614ed252d7a2'
# heliosli "0.957 latest dataset score in one click" SOURCE_SHA entries that differ from S6.
R7_SOURCE_SHA={'compact_infer.py':'8401812cd63737ec99839d317c4037c4efb445d1baec892fa7481bb30399cc2d',
               'compact_train.py':'04df7e2e154b2c2521696908f272940c63ce17e9b6337b85847fee64504eeaba',
               'probe_dual.py':'a6155eb08701f9bb746f838bc8ef8e2129c85f76a5c58e379692ffe2e76bbef0'}
R7_ASSET_SHA={'fluid_stacker.json':'be5a1664077300db1fdd0af8daaf44a3ebc41f91152e4f0ec124f805a7a51efb',
              'state_stacker.json':'7b54a3f774dad3f842d68aaccbb03771806c26c0a3ca3a07fea8575a21f3ead3',
              'state_spec.json':'2c976e70773b745cb357b992ba04946c157aa478e65d82f2883039c038b6ab3c'}
# compact_infer main(): --chunk default 192 when no CoAtNet group is loaded.
R7_CHUNK=192
R7_RECIPE={**S6_RECIPE,'chunk':R7_CHUNK,'bundle_id':R7_BUNDLE_ID,'dataset':R7_DATASET+'/'+str(R7_DATASET_VERSION),'syn_decomp':True,'flip_tta':False,
           'compile':{'mode':'default','backend':'inductor','dynamic':False,'tol_prob':1e-3,'synthetic_dp':.05,'shapes':'32..192 step 32','fallback':'none: a failed compile check or failing compiled forward is fatal (no eager continuation in production or rehearsal)','method':'nn.Module.compile in place'},'fluid_stacker':list(stackers.FLUID_ADOPTED),'state_stacker':list(stackers.STATE_ADOPTED),
           'state_readout':'arithmetic mean over fold models of state_probs, plain pass','output_precision':'%.9g'}
METHOD={**base.METHOD,'s6':R7_RECIPE,'blend':[.5,.5],'raptor_per_finding':{'Medial OA':.5,'Lateral OA':.5,'PF OA':.5},
        's6_quantization':"source Python '%.9g' float64 (R7 predictions.csv)",'slot':"compact slot 's6' carries heliosli R7",'recovery_recipe':RECOVERY_RECIPE}
HELIOSLI_VISIBLE_ATOL=5e-3
# heliosli's deadline horizon (8.3 h from notebook start) as the R7 conservative-projection gate.
R7_PROJECTION_GATE_SECONDS=8.3*3600
# Each board compiles its own R7 encoders (inductor), so the two shards run slightly different kernels
# (rehearsal v3: up to 2.6e-4 in probability). Within a board strict reload stays exact (<=1e-6, observed 0);
# the old arm is unchanged at 1e-6. 1e-3 is heliosli's own compiled-vs-eager acceptance tolerance.
CROSS_BOARD_TOLERANCE={'old':1e-6,'s6':1e-3}
rehearsal_hook=None


def r7_manifest_gate(manifest):
    """Exact R7 v1 bundle; any S6/compact-v1 or later R7 manifest is rejected."""
    require(manifest.get('bundle_id')==R7_BUNDLE_ID and manifest.get('tag')=='cnx_r7' and manifest.get('arm')=='E_lrprior'
            and manifest.get('models')==['E_lrprior_s0_f%d.pt'%i for i in range(4)] and manifest.get('models2') is None,'R7_BUNDLE_IDENTITY')
    require(manifest.get('compile') is True and manifest.get('flip_tta') is False and manifest.get('fluid_stacker')=='fluid_stacker.json'
            and manifest.get('state_stacker')=='state_stacker.json' and manifest.get('state_spec')=='state_spec.json','R7_V4C_FLAGS')
    require(manifest.get('side_model')=='side_cls.pt' and manifest.get('side_threshold')==2.0 and manifest.get('side_position_mm')==20.0
            and manifest.get('side_max_x')==0.0 and manifest.get('freeze_blocks')==16 and manifest.get('trainer_md5')==R7_TRAINER_MD5,'R7_SIDE_TRAINER')
    require(all(manifest.get('src_sha256',{}).get(k)==v for k,v in R7_SOURCE_SHA.items()),'R7_SOURCE_PINS')
    require(all(manifest.get('sha256',{}).get(k)==v for k,v in R7_ASSET_SHA.items()),'R7_STACKER_SPEC_PINS')
    for key in ('backbone','size','crop','views','slices','phys_mm','eval_stride'):require(manifest.get(key)==S6_RECIPE[key],'R7_RENDER_RECIPE')
    return {'bundle_id':R7_BUNDLE_ID,'manifest_compile':True,'deployed_compile':True,'flip_tta':False}


FLUID_MISSING_STATUSES=('no fluid-sensitive sagittal series',)
FLUID_NO_REGION_PREFIX='no region view for '


def fluid_gate(record):
    """Accept only the source's legitimate fluid outcomes: 'ok', no fluid-sensitive sagittal
    series, or no region view (the last two keep the source's missing-volume path).
    Exceptions never reach this gate as a status: install_fluid_guard replaces fluid_d2's
    catch-all, so typed data-local errors are recovered as counted, capped study failures
    (DataLocalFailure) and every unknown, resource or programming error is fatal. An 'error:'
    status (unguarded source) or any other status is therefore fatal here."""
    require(isinstance(record,dict) and isinstance(record.get('status'),str),'FLUID_RECORD_SCHEMA')
    status=record['status']
    if status=='ok' or status in FLUID_MISSING_STATUSES:return status
    if status.startswith(FLUID_NO_REGION_PREFIX):
        require(set(status[len(FLUID_NO_REGION_PREFIX):].split(','))<=set(stackers.FLUID_REGION),'FLUID_NO_REGION_SCHEMA');return status
    # With the boundary installed the source's catch-all can no longer produce an error status:
    # typed data errors raise DataLocalFailure and everything else escapes fatally.
    if status.startswith('error:'):raise ValueError('UNGUARDED_FLUID_ERROR_STATUS')
    raise ValueError('UNKNOWN_FLUID_STATUS')


FLUID_SITE='compact_infer.py:fluid_d2'
# Same causal fatal set as runtime.guard_source_error: these never enter recovery, even when wrapped.
FLUID_FATAL_CAUSES=(MemoryError,TimeoutError,OSError,RuntimeError,KeyboardInterrupt,SystemExit,KeyError,IndexError,AssertionError,ImportError)


def fluid_boundary(error):
    """Replacement body of fluid_d2's `except Exception as exc`: allowlist of the recovery's typed
    data-local classes only; always raises (never the source's 'error:' missing-path status)."""
    chain=[];current=error
    while current is not None and id(current) not in {id(x) for x in chain}:
        chain.append(current);current=current.__cause__ or current.__context__
    for cause in chain:
        if type(cause) not in runtime.REGISTERED_DATA_CLASSES and isinstance(cause,FLUID_FATAL_CAUSES):raise FatalSourceError(cause) from cause
    cls=type(error)
    allowed=cls is DataLocalFailure or cls in runtime.REGISTERED_DATA_CLASSES or cls is runtime.AuditedPixelBufferError or (runtime.REGISTERED_DICOM_CLASS is not None and cls is runtime.REGISTERED_DICOM_CLASS)
    if not allowed:raise FatalSourceError(error) from error
    raise DataLocalFailure('r7_fluid_measurement_error',site=FLUID_SITE,detail=cls.__name__) from error


def guard_fluid_source(source,namespace):
    """AST-rewrite the single `except Exception as exc` handler of fluid_d2 to call fluid_boundary;
    numerical statements are unchanged. Returns (function, transformed source sha256)."""
    import ast
    tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='fluid_d2')
    handlers=[x for x in ast.walk(node) if isinstance(x,ast.ExceptHandler)]
    require(len(handlers)==1 and isinstance(handlers[0].type,ast.Name) and handlers[0].type.id=='Exception' and handlers[0].name=='exc','EXACT_FLUID_CATCH_SITE')
    handlers[0].body=[ast.Expr(ast.Call(ast.Name('_r7_fluid_boundary',ast.Load()),[ast.Name('exc',ast.Load())],[])),ast.Raise(exc=None,cause=None)]
    transformed=ast.unparse(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])))+'\n'
    namespace['_r7_fluid_boundary']=fluid_boundary
    exec(compile(transformed,'<r7-fluid-boundary>','exec'),namespace)
    return namespace['fluid_d2'],hashlib.sha256(transformed.encode()).hexdigest()


def install_fluid_guard(CI,cfg):
    """Guard the pinned R7 compact_infer.fluid_d2 in place (CStudies.__getitem__ resolves it as a module global)."""
    path=(Path(cfg['code'])/'compact_infer.py').resolve()
    require(Path(CI.__file__).resolve()==path and file_sha(path)==R7_SOURCE_SHA['compact_infer.py'],'EXACT_FLUID_SOURCE')
    function,sha=guard_fluid_source(path.read_text(),CI.__dict__);CI.fluid_d2=function
    return {'site':FLUID_SITE,'guarded_sha256':sha,'allowlist':'DataLocalFailure, registered data classes, InvalidDicomError, AuditedPixelBufferError','unknown':'fatal'}


def require_compiled(receipt):
    """Production and rehearsal: the qualified runtime is compiled; no eager continuation."""
    require(receipt.get('requested') is True and receipt.get('active') is True and not receipt.get('fallback') and receipt.get('runtime_fallback_study') is None
            and str(receipt.get('decision','')).startswith('compiled'),'R7_COMPILED_NO_FALLBACK')
    return True


def fluid_selfcheck(CI):
    """Actual pinned fluid_d2 on synthetic rows: the cv2 path must run (it never raises)."""
    import numpy as np
    n=4;px=np.zeros((3*n,384,384),np.uint8);px[:n]=60
    px[n:2*n,100:200,100:200]=255;px[2*n:,150:180,150:180]=240
    meta=np.zeros((3*n,12),np.float32);meta[:,5]=np.log1p(3.0);meta[:,6]=384/130;meta[:,7]=1
    blocks=[{'plane':'Sagittal','n_slices':n,'views':[[0,0],[4,n],[5,2*n]],'series':'synthetic'}]
    rec=CI.fluid_d2(px,meta,blocks);bound={'eff_sag_p99_open':100*100*3./1000*n,'bak_sag_p99_open':30*30*3./1000*n}
    require(rec['status']=='ok' and all(rec[k] is not None and .95*v<=rec[k]<=v*1.0001 for k,v in bound.items()),'FLUID_D2_SYNTHETIC')
    meta[:,7]=0;missing=CI.fluid_d2(px,meta,blocks)
    require(missing['status']=='no fluid-sensitive sagittal series' and missing['eff_sag_p99_open'] is None,'FLUID_D2_MISSING_PATH')
    return {'status':'PASS','volumes':{k:rec[k] for k in bound},'upper_bounds':bound,'missing_status':missing['status']}


def load_stackers(CI,cfg,models,spec):
    """compact_infer main() checks with the pinned source, then the tested port must agree."""
    compact=Path(cfg['compact']);spec_raw=(compact/'state_spec.json').read_bytes()
    stk=CI.fluid_stacker_load(compact/'fluid_stacker.json',1);layout=CI.state_layout(spec,LABELS)
    for m in models:
        valid=m.state_valid.detach().cpu()
        for c,b_ in enumerate([c_ for c_ in layout['columns'] if 'categories' in c_]):
            kc=b_['stop']-b_['start']
            require(b_['class']==LABELS[c] and m.state_k[c]==kc and bool(valid[c,:kc].all()) and not bool(valid[c,kc:].any()),'R7_STATE_SLOTS')
        require([(c_['binary'],c_['class']) for c_ in layout['columns'] if 'binary' in c_]==[tuple(x) for x in m.bin_names],'R7_STATE_BINARIES')
    sst=CI.state_stacker_load(compact/'state_stacker.json',LABELS,layout,hashlib.md5(spec_raw).hexdigest())
    fluid=stackers.load_fluid((compact/'fluid_stacker.json').read_bytes(),LABELS)
    ours=stackers.state_layout(spec,LABELS);require(ours==layout,'PORTED_STATE_LAYOUT')
    state=stackers.load_state((compact/'state_stacker.json').read_bytes(),LABELS,ours,hashlib.md5(spec_raw).hexdigest())
    require({k:{c:v[c] for c in ('b0','b_z','b_lv','b_miss','volume_column','weights')} for k,v in stk['classes'].items()}=={k:{c:v[c] for c in ('b0','b_z','b_lv','b_miss','volume_column','weights')} for k,v in fluid['classes'].items()},'PORTED_FLUID_COEFFICIENTS')
    require({k:{c:v[c] for c in ('features','columns','coef','b0','volume_column')} for k,v in sst['classes'].items()}=={k:{c:v[c] for c in ('features','columns','coef','b0','volume_column')} for k,v in state['classes'].items()},'PORTED_STATE_COEFFICIENTS')
    pair=stackers.check_pair(fluid,state)
    require(tuple(pair['fluid'])==stackers.FLUID_ADOPTED and tuple(pair['state'])==stackers.STATE_ADOPTED and not state['needs_volume'],'PINNED_ADOPTED_CLASSES')
    return stk,fluid,state,{'fluid_sha256':fluid['sha256'],'state_sha256':state['sha256'],'adopted':pair,'state_spec_md5':hashlib.md5(spec_raw).hexdigest()}


COMPILE_TOL=1e-3
COMPILE_SANITY_DP=.05


def compile_receipt(receipt,sizes,why,d_final,err_c,err_e,gw):
    """compact_infer COMPILE record: the check's per-model maxima stay in per_group (they share key
    names such as study_dprob with the study-level decision fields, which caused rehearsal v2's
    TypeError and eager fallback)."""
    group={'group':'group1','crop':352,**{k:round(float(v),7) for k,v in gw.items()}}
    receipt.update(shapes=list(sizes),decision=why,study_dprob=round(float(d_final),7),fp32_err_compiled=None if err_c is None else round(float(err_c),7),
                   fp32_err_eager=None if err_e is None else round(float(err_e),7),per_group=[group],
                   max_abs_dlogit=float(gw.get('study_dlogit',0.)),max_abs_dprob=float(gw.get('study_dprob',0.)))
    return receipt


def compile_encoders(models,item,torch):
    """compact_infer v3 --compile and compile_check, ported: every fold encoder compiled in place
    (nn.Module.compile keeps parameter names, so state digests and strict reloads are unchanged);
    every encoder batch shape (32..chunk) checked on synthetic windows, then the first real study
    compiled vs eager (study probabilities within 1e-3, else an fp32 eager tie-break). Any error or
    failed gate records the fallback in the receipt; the worker then stops at require_compiled
    before inference, so production never continues on eager encoders."""
    import contextlib
    receipt={'requested':True,'active':False,'mode':'default','backend':'inductor','tol_prob':COMPILE_TOL,'fallback':None,'decision':None}
    def set_eager(reason):
        for m in models:m.enc._compiled_call_impl=None
        try:
            import torch._dynamo as _dynamo;_dynamo.reset()
        except Exception:pass
        receipt.update(active=False,fallback=str(reason)[:400])
    @contextlib.contextmanager
    def eager(m):
        impl=m.enc._compiled_call_impl;m.enc._compiled_call_impl=None
        try:yield
        finally:m.enc._compiled_call_impl=impl
    def forward(m,img,meta,mask,mode):
        if mode=='fp32':return m(img,meta,mask,chunk=R7_CHUNK).float()
        with torch.autocast('cuda',dtype=torch.float16):return m(img,meta,mask,chunk=R7_CHUNK).float()
    def compare(zc,ze,gw,key):
        if not torch.equal(torch.isfinite(zc),torch.isfinite(ze)):raise ValueError('compiled and eager outputs differ in finiteness')
        fin=torch.isfinite(zc)&torch.isfinite(ze)
        if bool(fin.any()):
            gw[key+'_dlogit']=max(gw.get(key+'_dlogit',0.),float((zc-ze).abs()[fin].max()))
            gw[key+'_dprob']=max(gw.get(key+'_dprob',0.),float((torch.sigmoid(zc.double())-torch.sigmoid(ze.double())).abs()[fin].max()))
    start=time.perf_counter()
    try:
        import torch._dynamo as _dynamo
        for name,limit in (('cache_size_limit',64),('recompile_limit',64),('accumulated_cache_size_limit',2048),('accumulated_recompile_limit',2048)):
            if hasattr(_dynamo.config,name):setattr(_dynamo.config,name,max(limit,int(getattr(_dynamo.config,name))))
        for m in models:m.enc.compile(dynamic=False,mode='default')
        receipt['active']=True
        sizes=list(range(32,R7_CHUNK+1,32));gen=torch.Generator().manual_seed(20261010);gw={}
        full=item['images'].to('cuda');meta_r=item['meta'].to('cuda')[None];mask_r=torch.ones((1,full.shape[0]),dtype=torch.bool,device='cuda')
        with torch.inference_mode():
            for s in sizes:
                img_s=torch.randint(0,256,(1,s,3,352,352),generator=gen,dtype=torch.uint8).to('cuda')
                meta_s=torch.zeros((1,s,int(item['meta'].shape[1])),dtype=torch.float32,device='cuda');mask_s=torch.ones((1,s),dtype=torch.bool,device='cuda')
                for m in models:
                    zc=forward(m,img_s,meta_s,mask_s,'fp16')
                    with eager(m):ze=forward(m,img_s,meta_s,mask_s,'fp16')
                    compare(zc,ze,gw,'synthetic')
            if not gw.get('synthetic_dprob',0.)<=COMPILE_SANITY_DP:raise ValueError('synthetic compiled vs eager |dp| %.3g'%gw['synthetic_dprob'])
            img_r=full[None,:,:,16:368,16:368];zc_l,ze_l=[],[]
            for m in models:
                zc=forward(m,img_r,meta_r,mask_r,'fp16')[0]
                with eager(m):ze=forward(m,img_r,meta_r,mask_r,'fp16')[0]
                compare(zc,ze,gw,'study');zc_l.append(zc);ze_l.append(ze)
            pc=torch.sigmoid(torch.stack(zc_l).mean(0).double());pe=torch.sigmoid(torch.stack(ze_l).mean(0).double());d=float((pc-pe).abs().max())
            ok=d<=COMPILE_TOL;why='compiled == eager within %.3g (max |dp| %.3g)'%(COMPILE_TOL,d);err_c=err_e=None
            if not ok:
                z32=[]
                for m in models:
                    with eager(m):z32.append(forward(m,img_r,meta_r,mask_r,'fp32')[0])
                p32=torch.sigmoid(torch.stack(z32).mean(0).double());err_c,err_e=float((pc-p32).abs().max()),float((pe-p32).abs().max())
                ok=err_c<=err_e+COMPILE_TOL;why='compiled vs eager %.3g > tol; fp32 error compiled %.3g eager %.3g'%(d,err_c,err_e)
        torch.cuda.synchronize()
        compile_receipt(receipt,sizes,why,d,err_c,err_e,gw)
        if not ok:set_eager('compile check: '+why)
    except Exception as exc:
        set_eager('compile check failed: '+repr(exc))
    receipt['check_seconds']=time.perf_counter()-start
    return receipt,set_eager


def r7_worker(config_path):
    """Default S6 worker with the R7 bundle, aux state read-out and both stackers."""
    import numpy as np
    import torch
    import resource
    import setuptools
    import distutils.util
    cfg=json.loads(Path(config_path).read_bytes());out=Path(cfg['out'])/'s6';out.mkdir();start=time.perf_counter()
    r={'status':'RUNNING','version':VERSION,'phase':'source_state','board':cfg['board'],'arm':'s6','slot':'r7','assigned':cfg['dev'],'completed':0,'optimizer_updates':0,'records':[]};save(out/'receipt.json',r)
    try:
        r['environment']=board_environment(torch,cfg);save(out/'receipt.json',r)
        verify_sources(cfg['code'],cfg['source_pins']);verify_assets(cfg['compact'],cfg['anatomy'],cfg['manifest'],cfg['asset_binding']);r['r7_manifest']=r7_manifest_gate(cfg['manifest'])
        require(hashlib.md5((Path(cfg['code'])/'compact_train.py').read_bytes()).hexdigest()==R7_TRAINER_MD5,'R7_TRAINER_MD5')
        verify_staged(cfg['staged'],cfg['staged_ids'],cfg['stage_binding'])
        require(sys.version_info[:2]==(3,12) and r['environment']['packages']['timm']=='1.0.26' and torch.cuda.is_available() and torch.cuda.get_device_name(0)=='Tesla T4' and torch.__version__=='2.10.0+cu128','S6_CUDA_ENVIRONMENT')
        require('monai' not in sys.modules,'FRESH_MONAI_NAMESPACE');sys.path.insert(0,cfg['code']);os.environ['RSNA_CODE_ROOT']=cfg['code']
        import compact_infer as CI
        import compact_train as CT
        import cv2
        r['environment']['cv2']=cv2.__version__;r['fluid_guard']=install_fluid_guard(CI,cfg);r['fluid_selfcheck']=fluid_selfcheck(CI);save(out/'receipt.json',r)
        if cfg['mode']=='rehearsal' and cfg['board']==0:r['native_availability_stress']=native_availability_stress(cfg);r['guarded_native_availability_stress']=native_availability_stress(cfg,guarded=True);save(out/'receipt.json',r)
        require(CT.LABELS==LABELS,'S6_TARGET_ORDER');r['synthetic_source_checks']=synthetic_source_checks()
        torch.manual_seed(42);np.random.seed(42);torch.backends.cudnn.benchmark=False
        spec=json.loads((Path(cfg['compact'])/'state_spec.json').read_bytes());models=[]
        for name in cfg['manifest']['models']:
            sd=torch.load(Path(cfg['compact'])/name,map_location='cpu',weights_only=True)
            kw={'spatial':any(k=='sp_q' or k.endswith('.sp_q') for k in sd),'extent':any(k.split('.')[-1] in ('ext_u','ext_view','ext_gamma') for k in sd),
                'syn_decomp':any(k=='syn_beta' or k.endswith('.syn_beta') for k in sd)}
            require(all(kw.values()),'R7_SPATIAL_EXTENT_SYN_FOLDS')
            model=CT.Net('convnext_small',None,freeze_blocks=16,crop=352,state_spec=spec,in_chans=3,**kw);model.load_state_dict(sd,strict=True)
            models.append(model.eval().requires_grad_(False).to('cuda').to(memory_format=torch.channels_last));del sd
        stk,fluid,state,r['stackers']=load_stackers(CI,cfg,models,spec);save(out/'receipt.json',r)
        pilot_ids=cfg.get('s6_sentinels',[])*2;all_ids=pilot_ids+cfg['dev']
        def dataset():
            d=CI.CStudies(cfg['staged'],'train',all_ids,out/'scratch',cfg['code'],384,1,'v4',3,cfg['anatomy'],3.5,str(Path(cfg['compact'])/'side_cls.pt'),2.,0.,20.,extent_meta=True,fluid=True);d._setup();return d
        ds=dataset()
        import monai
        r['environment']['vendored_monai']={'version':monai.__version__,'path':str(Path(monai.__file__).resolve())}
        require(monai.__version__=='0.6.0' and Path(monai.__file__).resolve().is_relative_to(Path(cfg['code'])/'training/vendor_runtime/monai060'),'VENDORED_MONAI')
        states=models+[ds.bank.model,ds.side_net];initial=model_digest(states);r['state_before']=initial
        require(cfg['mode']=='rehearsal' or cfg.get('pilot_state_sha256') is not None,'R7_STATE_PIN_REQUIRED')
        if cfg.get('pilot_state_sha256') is not None:require(initial==cfg['pilot_state_sha256'],'QUALIFIED_R7_STATE')
        def finite_aux(module,inputs,output):require(torch.isfinite(output).all().item(),'NONFINITE_AUXILIARY')
        capture={}
        def observe_record(*a,**kw):
            record=original_record(*a,**kw);capture['anchors']=record;return record
        def observe_build(uid):
            item=original_build(uid);capture['item']=item
            study=json.loads((Path(ds.builder.cache)/'studies'/(hashlib.sha256(uid.encode()).hexdigest()[:24]+'.json')).read_bytes())
            capture['source_diagnostics']={'study_errors':study.get('errors',[]),'series_errors':[s.get('errors',[]) for s in study['series']],'skipped_series':item['skipped_series']}
            capture['scratch_bytes']=sum(p.stat().st_size for p in Path(ds.builder.cache).rglob('*') if p.is_file());return item
        def observe_rows(*a,**kw):
            rows=original_rows(*a,**kw);capture['rows']=rows;return rows
        def instrument():
            nonlocal original_record,original_build,original_rows
            ds.bank.model.register_forward_hook(finite_aux);ds.side_net.register_forward_hook(finite_aux)
            original_rows=ds.window_rows;r['source_guard_receipt']=install_source_guards(CI,ds,cfg)
            original_record=ds.bank.__class__.record.__get__(ds.bank);original_build=ds.builder.__class__.build.__get__(ds.builder)
            ds.bank.record=observe_record;ds.builder.build=observe_build;ds.window_rows=observe_rows
        original_record=original_build=original_rows=None;instrument()
        if cfg['mode']=='rehearsal':
            require(rehearsal_hook is not None,'FRESH_INJECTION_EXTENSION_REQUIRED')
            r['recovery_injection']=rehearsal_hook('s6',cfg,{'CI':CI,'ds':ds,'probes':cfg['probes'],'state_check':lambda:require(model_digest(states)==initial,'INJECTION_STATE_PARITY')});save(out/'receipt.json',r)
        reference=cfg.get('heliosli_reference',{}).get('r7',{});logits=[];probs=[];statuses=[];compiled=set_eager=None
        with PlatformInterruption() as stop:
            for i,sid in enumerate(all_ids):
                if i<len(pilot_ids):
                    require(model_digest(states)==initial,'S6_STATE_BEFORE_RELOAD')
                    for m,name in zip(models,cfg['manifest']['models']):
                        m.load_state_dict(torch.load(Path(cfg['compact'])/name,map_location='cpu',weights_only=True),strict=True);m.eval()
                    CI.FORCED_SIDE.clear();reset_native_renderer(cfg)
                    ds=dataset();states=models+[ds.bank.model,ds.side_net];require(model_digest(states)==initial,'S6_STRICT_RELOAD_STATE');instrument()
                try:
                    study_start=time.perf_counter();capture.clear();r['phase']='pilot_sentinel' if i<len(pilot_ids) else 'evaluation';r['current_ordinal']=i-len(pilot_ids);save(out/'receipt.json',r)
                    item=ds[i]
                    if item['error'] is not None:
                        save(out/'current-study-error-private.json',{'ordinal':r['current_ordinal'],'study':sid,'source_error':item['error']})
                    require(item['error'] is None and len(item['images'])>0,'UNCLASSIFIED_FAILED_STUDY')
                    a=item['images'].numpy();meta=item['meta'].numpy();require(a.dtype==np.uint8 and a.shape[1:]==(3,384,384) and meta.shape[1:]==(9,) and np.isfinite(meta).all(),'S6_INPUT_SCHEMA')
                    fp={'render_windows':array_sha(a),'metadata':array_sha(meta),'retained_crop':array_sha(a[:,:,16:368,16:368]),'mask':digest([True]*len(a)),
                        'pixel_rows':array_sha(capture['item']['px'].numpy()),'row_metadata':array_sha(capture['item']['meta'].numpy()),'window_blocks':digest(capture['item']['blocks']),'window_rows':digest(capture['rows'])}
                    rawmeta=capture['item']['meta'].numpy()
                    require(all(len(set(rawmeta[list(t),0]))==1 and len(set(rawmeta[list(t),3]))==1 for t in [row[2:] for row in capture['rows']]),'TRIPLET_SERIES_VIEW')
                    fluid_gate(item.get('fluid'))
                    if compiled is None:
                        compiled,_=compile_encoders(models,item,torch);r['compile']=compiled;save(out/'receipt.json',r)
                        require_compiled(compiled);require(model_digest(states)==initial,'COMPILE_STATE_PARITY')
                    t=time.perf_counter();img=item['images'].to('cuda')[None,:,:,16:368,16:368];mt=item['meta'].to('cuda')[None];mask=torch.ones((1,len(a)),dtype=torch.bool,device='cuda')
                    require(all(not module.training for model in states for module in model.modules()),'S6_EVAL_STATE')
                    with torch.inference_mode():
                        passes=[]
                        for m in models:
                            # Qualified contract: compiled encoders only. A failing compiled forward is fatal
                            # (heliosli would continue eager, which projects past the nine-hour limit).
                            with torch.autocast('cuda',dtype=torch.float16):z_,sl_,bl_=m(img,mt,mask,chunk=R7_CHUNK,return_aux=True)
                            z_,sl_,bl_=z_.float(),sl_.float(),bl_.float();passes.append((z_,CI.state_readout(m,sl_,bl_)))
                        zs=torch.stack([z_[0] for z_,_ in passes]);require(torch.isfinite(zs).all().item(),'NONFINITE_FP16_NO_RESCUE')
                        stv=torch.stack([s_[0] for _,s_ in passes]).double().mean(0)
                        zmean=zs.mean(0);p_model=torch.sigmoid(zmean).double().cpu().numpy();zf=zs.mean(0).double().cpu().numpy();stv=stv.cpu().numpy()
                    z=zs.cpu().numpy();oracle=torch.sigmoid(torch.from_numpy(z).mean(0)).double().numpy();err=float(np.max(np.abs(p_model-oracle)));require(err<=1e-6,'MEAN_SIGMOID_RECONSTRUCTION')
                    p,stack=stackers.apply(p_model,zf,stv,item.get('fluid'),fluid,state,LABELS)
                    source_fluid=CI.fluid_stack(stk,LABELS,zf,item.get('fluid'))[0]
                    require(all(stack[k]['z_out']==source_fluid[k] for k in stackers.FLUID_ADOPTED),'PORTED_FLUID_EQUALS_SOURCE')
                    if cfg['mode']=='rehearsal':require(all(stack[k]['fallback'] is None for k in stack),'FINITE_STACKER_LOGITS')
                    probabilities([p],1);torch.cuda.synchronize();elapsed=time.perf_counter()-t
                    require(not any((out/'scratch').glob('s_*')),'PER_STUDY_SCRATCH_NOT_REMOVED')
                    fl=item.get('fluid') or {}
                    row={'ordinal':i-len(pilot_ids),'windows':len(a),'input':fp,'preprocess_seconds':item['seconds'],'four_model_seconds':elapsed,'mean_sigmoid_error':err,
                         'anchor_method':capture['anchors']['method'],'anchor_reasons':capture['anchors']['why'],'side':item.get('side'),'native_scratch_bytes':capture['scratch_bytes'],'field_center_fallbacks':sum(x=='field_center_fallback' for x in capture['anchors']['method']),
                         'fluid':{k:fl.get(k) for k in ('status','eff_sag_p99_open','bak_sag_p99_open','n_sag_fluid_series')},'fluid_seconds':item.get('fluid_s'),'stackers':stack,'model_probabilities':p_model.tolist(),
                         'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
                    expected_mechanics=cfg.get('probes',{}).get(sid,{}).get('s6',{}).get('mechanics')
                    if expected_mechanics is not None:require(all(row.get(k)==v for k,v in expected_mechanics.items()),'QUALIFIED_S6_ANCHOR_FALLBACK_SEMANTICS')
                    if i<len(pilot_ids):
                        expected=cfg['probes'][sid]['s6'];require(fp==expected['input'],'PILOT_INPUT_WINDOW_PARITY');probe_gate(sid,'s6',p,fp,cfg)
                        row.update(id=sid,probabilities=p.tolist(),strict_reload=True,logits=z.tolist(),pilot_input_parity=True)
                        r.setdefault('sentinel_repeats',[]).append(row)
                        if i>=len(pilot_ids)//2:
                            prior=r['sentinel_repeats'][i-len(pilot_ids)//2]
                            require(prior['input']==fp and float(np.max(np.abs(p-np.array(prior['probabilities']))))<=1e-6,'S6_BOTH_FRESH_RELOAD_PARITY')
                            row['reloaded_probabilities']=p.tolist();row['reloaded_logits']=z.tolist();row['reloaded_model_probabilities']=row['model_probabilities'];row['reloaded_stackers']=stack
                            row['logits']=prior['logits'];row['probabilities']=prior['probabilities'];row['model_probabilities']=prior['model_probabilities'];row['stackers']=prior['stackers'];r.setdefault('sentinels',[]).append(row)
                    else:
                        probe='s6' in cfg.get('probes',{}).get(sid,{});probe_gate(sid,'s6',p,fp,cfg)
                        row.update(id=sid,status='genuine',source_diagnostics=capture.get('source_diagnostics',{}),qualified_error=None,pilot_input_parity=True if probe else None,probabilities=p.tolist())
                        if sid in reference:
                            d=np.abs(p-np.asarray(reference[sid],dtype=np.float64));atol=cfg['heliosli_visible_atol']
                            row['heliosli_reference']={'max_abs':float(d.max()),'per_target_abs':d.tolist(),'atol':atol,'within_atol':bool(d.max()<=atol)}
                        logits.append(z);probs.append(p);statuses.append('genuine');r['completed']=len(probs)
                    row['production_seconds']=time.perf_counter()-study_start
                    del img,mt,mask,zs,item,a,meta,passes
                except DataLocalFailure as e:
                    require(i>=len(pilot_ids) and cfg['mode']!='rehearsal','HEALTHY39_MUST_BE_GENUINE')
                    probs.append(np.full(12,np.nan));logits.append(np.full((4,12),np.nan,dtype=np.float32));statuses.append('data_local_failure')
                    row={'id':sid,'status':'data_local_failure','phase':'s6_input','failure_code':str(e),'production_seconds':time.perf_counter()-study_start}
                    r['completed']=len(probs)
                    save(out/('failure-'+str(len(probs)-1)+'-private.json'),{'id':sid,'type':type(e).__name__,'detail':e.detail,'site':e.site,'phase':'s6_input'})
                finally:
                    clean=cleanup_study(out/'scratch',sid,CI.FORCED_SIDE);capture.clear()
                    if sys.exc_info()[0] is None:require(model_digest(states)==initial,'S6_STATE_AFTER_EACH_STUDY')
                require(statuses.count('data_local_failure')<=math.floor(3*len(cfg['staged_ids'])/100),'GLOBAL_UNION_CAP_ALREADY_EXCEEDED')
                row.update(cleanup=clean,state_healthy=True)
                row['production_seconds']=time.perf_counter()-study_start
                if i>=len(pilot_ids):
                    r['records'].append(row);save_ledger(out/'predictions.npz',cfg['dev'][:len(probs)],probs,statuses,logits=logits)
                r['elapsed_seconds']=time.perf_counter()-start;save(out/'receipt.json',r)
                if stop.requested:raise InterruptedError('Platform interruption at study boundary')
        require(len(probs)==len(cfg['dev']),'COMPLETE_ASSIGNED_REQUIRED');r['state_after']=model_digest(states);require(r['state_after']==initial,'S6_PARAMETER_BUFFER_MUTATION')
        r.update(status='PASS',genuine=statuses.count('genuine'),failed=statuses.count('data_local_failure'),zero_imputed=True,zero_failed=all(s=='genuine' for s in statuses),fp32_recoveries=0,ledger_sha256=file_sha(out/'predictions.npz'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved());save(out/'receipt.json',r)
    except BaseException as e:
        actual=e.error if isinstance(e,FatalSourceError) else e
        save(out/'failure-private.json',{'type':type(actual).__name__,'detail':str(actual),'boundary_type':type(e).__name__});r.update(status='INTERRUPTED' if isinstance(actual,(InterruptedError,KeyboardInterrupt,TimeoutError)) else 'FAIL',failure_type=type(actual).__name__,elapsed_seconds=time.perf_counter()-start);save(out/'receipt.json',r)
        if actual is not e:raise actual from e
        raise


def launch(cfg,out,arm):
    """Default launch semantics with the R7 worker modules."""
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


def recompute_stackers(row,fluid_doc,state_doc):
    """Rehearsal contract: recorded stacker outputs must follow from recorded features."""
    for name,s in row['stackers'].items():
        j=LABELS.index(name);require(s['p_model']==row['model_probabilities'][j],'STACKER_MODEL_PROBABILITY')
        if s['stacker']=='fluid':
            co=fluid_doc['classes'][name];vol=row['fluid'].get(co['volume_column']);miss=s['volume_missing']
            require(miss==(vol is None or not math.isfinite(float(vol)) or float(vol)<0),'FLUID_MISSING_FLAG')
            lv=0. if miss else math.log1p(float(vol))
            z=co['b0']+co['b_z']*s['z_feature']+co['b_lv']*lv+co['b_miss']*(1. if miss else 0.)
        else:
            co=state_doc['classes'][name];x=s['features'];require(len(x)==len(co['coef']) and x[0]==s['z_feature'],'STATE_FEATURE_VECTOR')
            z=co['b0']+sum(a*b for a,b in zip(x,co['coef']))
        require(s['z_out'] is not None and abs(z-s['z_out'])<=1e-9*max(1.,abs(z)) and abs(stackers.sigmoid(s['z_out'])-s['p_out'])<=1e-15 and s['p_out']==row['probabilities'][j],'STACKER_RECOMPUTATION')


def require_rehearsal(r,binding,csv_path):
    """R7 rehearsal contract: default gates; R7 slot qualified by pilot-input parity, reload, stackers and heliosli visible parity."""
    import csv
    import numpy as np
    from knee.public_s6_qualification import UID
    from knee.mirror_input import STRESS_CASES
    require(binding.get('version')==VERSION and binding.get('method')==METHOD and binding.get('input_policy')==INPUT_POLICY and binding.get('r7',{}).get('recipe')==R7_RECIPE,'EXACT_FROZEN_METHOD')
    require(r.get('status')=='PASS' and r.get('version')==VERSION and r.get('mode')=='rehearsal' and r.get('binding')==binding,'PASS_EXACT_REHEARSAL_REQUIRED')
    require(r.get('completed')==39 and r.get('completed_visible')==3 and r.get('completed_probes')==36 and r.get('completed_s6_probes')==32 and r.get('completed_old_historical')==4 and r.get('all_input_state_probability_gates') is True and r.get('zero_failed') is True and r.get('zero_imputed') is True,'REHEARSAL_COMPLETE_GATES')
    require(len(r.get('boards',[]))==2 and len({x['uuid'] for x in r['boards']})==2 and all(x['name']=='Tesla T4' for x in r['boards']),'TWO_PHYSICAL_T4_GATES')
    require(0<r.get('projection',{}).get('conservative_seconds',math.inf)<MAX_SECONDS and 0<r.get('elapsed_seconds',math.inf)<MAX_SECONDS,'NINE_HOUR_PROJECTION_GATE')
    require(r['projection']['conservative_seconds']<=R7_PROJECTION_GATE_SECONDS,'R7_HELIOSLI_HORIZON_PROJECTION_GATE')
    require(r.get('csv_sha256')==file_sha(csv_path),'REHEARSAL_CSV_BINDING')
    with Path(csv_path).open(newline='') as h:rows=list(csv.reader(h))
    require(rows[0]==[UID]+LABELS and len(rows)==4 and [x[0] for x in rows[1:]]==r['visible_ids'],'VISIBLE_CSV_COVERAGE')
    probabilities([[float(x) for x in row[1:]] for row in rows[1:]],3)
    old0=r.get('workers',{}).get('old',[{}])[0]
    stress=old0.get('partial_input_stress',{})
    require(stress.get('status')=='PASS' and stress.get('cases')==dict.fromkeys(STRESS_CASES,'PASS') and stress.get('case_count')==17 and stress.get('uses_raw_data') is False and stress.get('uses_labels') is False,'REHEARSAL_PARTIAL_INPUT_STRESS')
    guarded=old0.get('guarded_partial_input_stress',{})
    require(guarded.get('status')=='PASS' and guarded.get('guarded') is True and guarded.get('case_count')==10 and set(guarded.get('cases',{}))=={'healthy','missing_iop','missing_ipp','missing_spacing','empty_series','missing_series','bad_header','bad_pixel','black_frame','short_series'},'GUARDED_MIRROR_PARTIAL_TEN')
    s60=r.get('workers',{}).get('s6',[{}])[0]
    guarded_native=s60.get('guarded_native_availability_stress',{})
    require(guarded_native.get('status')=='PASS' and guarded_native.get('guarded') is True and guarded_native.get('case_count')==2 and set(guarded_native.get('cases',{}))=={'absent','empty'},'GUARDED_NATIVE_PARTIAL_TWO')
    native=s60.get('native_availability_stress',{})
    require(native.get('status')=='PASS' and native.get('case_count')==2 and native.get('uses_raw_data') is False and native.get('uses_labels') is False and set(native.get('cases',{}))=={'absent','empty'} and all(x.get('status')=='PASS' and x.get('usable_series')==1 and x.get('slices',0)>0 for x in native['cases'].values()),'REHEARSAL_NATIVE_AVAILABILITY_STRESS')
    r7=binding['r7'];fluid_doc,state_doc=r7['fluid'],r7['state']
    for arm in ('old','s6'):
        require(arm in r.get('phases',{}) and r['phases'][arm]['wall_seconds']>0 and r['phases'][arm]['peaks']['host_rss_bytes']>0 and r['phases'][arm]['peaks']['output_disk_bytes']>0 and r['phases'][arm]['minimum_disk_free_bytes']>0,'CONCURRENT_RESOURCE_MEASUREMENT')
        rr=r.get('workers',{}).get(arm,[]);require(len(rr)==2,'REHEARSAL_WORKERS')
        require(all(x.get('recovery_injection',{}).get('status')=='PASS' and x['recovery_injection'].get('failure_status')=='data_local_failure' and x['recovery_injection'].get('injected_rows_genuine') is False and x['recovery_injection'].get('state_healthy') is True for x in rr),'ACTUAL_CLASSIFIED_RECOVERY_INJECTIONS')
        if arm=='s6':require(len({x['state_before'] for x in rr})==1 and rr[0]['state_before'] is not None,'CROSS_BOARD_R7_STATE')
        for board,x in enumerate(rr):
            require(x['status']=='PASS' and x['board']==board and x['state_before']==x['state_after'] and x['completed']==len(x['assigned']) and x['peak_cuda_bytes']>0 and x['peak_rss_bytes']>0 and x['zero_failed'] is True and x['zero_imputed'] is True and x['fp32_recoveries']==0,'REHEARSAL_WORKER_STATE_RESOURCE')
            if arm=='old':require(x['state_before']['state']==binding['qualified_state']['old_state'] and x.get('constructor_grid_sha256')==binding['qualified_state']['old_grid'],'QUALIFIED_COMPLETE_STATE_BINDING')
            else:
                require(x.get('slot')=='r7' and x.get('r7_manifest',{}).get('bundle_id')==R7_BUNDLE_ID and x.get('fluid_selfcheck',{}).get('status')=='PASS'
                        and x.get('stackers',{}).get('adopted')=={'fluid':list(stackers.FLUID_ADOPTED),'state':list(stackers.STATE_ADOPTED)}
                        and x['stackers'].get('fluid_sha256')==R7_ASSET_SHA['fluid_stacker.json'] and x['stackers'].get('state_sha256')==R7_ASSET_SHA['state_stacker.json'],'QUALIFIED_R7_SLOT')
                require_compiled(x.get('compile',{}));require(x.get('fluid_guard',{}).get('site')==FLUID_SITE and x['fluid_guard'].get('unknown')=='fatal','R7_FLUID_BOUNDARY_INSTALLED')
                require(all((row.get('fluid') or {}).get('status')=='ok' or (row.get('fluid') or {}).get('status') in FLUID_MISSING_STATUSES or str((row.get('fluid') or {}).get('status','')).startswith(FLUID_NO_REGION_PREFIX) for row in x['records']+x['sentinels']),'R7_FLUID_STATUSES')
            require(x['environment']['torch']==binding['environment']['torch'] and x['environment']['packages']==binding['environment']['packages'],'QUALIFIED_ENVIRONMENT_BINDING')
            require(x['environment']['physical_uuid']==r['boards'][board]['uuid'] and x['environment']['cuda_device_count']==1,'ISOLATED_BOARD_RUNTIME')
            require(len(x['sentinels'])==4 and [s['id'] for s in x['sentinels']]==r['sentinel_ids'][arm] and all(s['strict_reload'] for s in x['sentinels']),'BOTH_BOARD_SENTINELS')
            for s in x['sentinels']:
                require(all(s['input'].get(k)==v for k,v in binding['sentinel_inputs'][arm][s['id']].items()),'BOUND_SENTINEL_INPUTS')
                p=np.array(s['probabilities']);q=np.array(s['reloaded_probabilities'])
                if arm=='old':
                    expected=np.array(binding['sentinel_probabilities'][arm][s['id']])
                    require(max(float(np.max(np.abs(p-expected))),float(np.max(np.abs(q-expected))),float(np.max(np.abs(p-q))))<=1e-6,'SENTINEL_EXACT_REFERENCE_GATE')
                    continue
                # No R7 probabilities exist for these train IDs: pilot input parity,
                # fresh strict reload, mean-logit reconstruction and stacker recomputation.
                require(binding['sentinel_probabilities']['s6'][s['id']] is None and float(np.max(np.abs(p-q)))<=1e-6 and s.get('pilot_input_parity') is True,'R7_SENTINEL_RELOAD_GATE')
                for field,target in [('logits',s['model_probabilities']),('reloaded_logits',s['reloaded_model_probabilities'])]:
                    z=np.asarray(s[field],dtype=np.float32);require(z.shape==(4,12) and np.isfinite(z).all(),'SENTINEL_RAW_FOUR_LOGITS')
                    mean=z.mean(0).astype(np.float64);sigmoid=np.exp(-np.logaddexp(0.,-mean))
                    require(float(np.max(np.abs(sigmoid-np.asarray(target))))<=1e-6,'SENTINEL_MEAN_LOGIT_SIGMOID')
                recompute_stackers(s,fluid_doc,state_doc)
        for a,b in zip(rr[0]['sentinels'],rr[1]['sentinels']):
            require(a['input']==b['input'] and max(abs(x-y) for x,y in zip(a['probabilities'],b['probabilities']))<=CROSS_BOARD_TOLERANCE[arm],'CROSS_BOARD_SENTINEL_PARITY')
    oldrows={row['id']:row for worker in r['workers']['old'] for row in worker['records']}
    s6rows={row['id']:row for worker in r['workers']['s6'] for row in worker['records']}
    require(len(oldrows)==len(s6rows)==39 and set(oldrows)==set(s6rows),'COMPLETE39_MECHANICS')
    require(set(binding['s6_probe_ids']).isdisjoint(binding['old_historical_ids']) and not set(binding['s6_probe_ids']+binding['old_historical_ids'])&set(r['visible_ids']),'DISJOINT39_MECHANICS')
    for sid in binding['s6_probe_ids']:
        require(oldrows[sid].get('source_reference_parity') is True and oldrows[sid].get('strict_reload') is True and 0<=oldrows[sid].get('reload_max_abs',math.inf)<=1e-6,'ALL32_OLD_SOURCE_REPEAT_RELOAD')
        require(s6rows[sid].get('pilot_input_parity') is True,'ALL32_R7_PILOT_INPUT_PARITY')
    for row in s6rows.values():recompute_stackers(row,fluid_doc,state_doc)
    for sid in r['visible_ids']:
        g=s6rows[sid].get('heliosli_reference',{})
        require(g.get('atol')==r7['visible_atol'] and g.get('within_atol') is True and 0<=g.get('max_abs',math.inf)<=r7['visible_atol'],'R7_HELIOSLI_VISIBLE_PARITY')


def heliosli_parity(out,cfg):
    """Rehearsal diagnostic: visible-3 per-arm and blend parity against heliosli's published output."""
    import numpy as np
    out=Path(out);ref=cfg['heliosli_reference'];ids=cfg['binding']['visible_ids'];arms={}
    for arm in ('old','s6'):
        rows={}
        for b in range(2):
            with np.load(out/f'board{b}-{arm}'/arm/'predictions.npz',allow_pickle=False) as a:
                rows.update(zip(a['study_ids'].tolist(),a['probabilities']))
        arms[arm]=np.array([rows[s] for s in ids],dtype=np.float64)
    blend=recover(arms['old'],arms['s6'],np.zeros(3,bool),np.zeros(3,bool))['blend']
    result={}
    for name,ours,theirs in (('raptor_raw_vs_old',arms['old'],ref['raptor_raw']),('r7_vs_ours_sub',arms['s6'],ref['r7']),('blend_vs_submission',blend,ref['submission'])):
        d=np.abs(ours-np.array([theirs[s] for s in ids],dtype=np.float64))
        result[name]={'max_abs':float(d.max()),'max_abs_per_target':dict(zip(LABELS,d.max(0).tolist()))}
    save(out/'heliosli-parity.json',result);return result


def _bind(function):
    """Exact default bytecode, resolved against R7 globals (version, launch, recovery, contract)."""
    clone=types.FunctionType(function.__code__,globals(),function.__name__,function.__defaults__,function.__closure__)
    clone.__kwdefaults__=function.__kwdefaults__;clone.__doc__=function.__doc__;clone.__qualname__=function.__qualname__
    return clone


old_worker=_bind(base.old_worker)
merge=_bind(base.merge)
run=_bind(base.run)


if __name__=='__main__':
    import argparse
    require(Path(__file__).resolve()==(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_r7_recovery_entry.py').resolve(),'EXACT_FRESH_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args();cfg=json.loads(Path(a.worker).read_bytes())
    (old_worker if cfg['arm']=='old' else r7_worker)(a.worker)
