"""Separate, frozen Versia ConvNeXt16 component qualification; no fusion/training."""
import ast
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
import types
import warnings

from knee.reference import bounded_ids, canonical_bytes, sha256, validate_predictions
from knee.raptor_input import fatal_error
from knee.reference_blend import read_test_contract, write_probabilities

VERSION = 'versia-convnext16-slot-copy-v1'
SPEC_SHA256 = '61931a5687982af68f50df34a6a67fbbc76ff875869758f08e0d8b3dac41a4cd'
SOURCE_HASHES = {
 'published-preprocess.py':'4491963e4d2156625c9cf79051f4a25c32d2d761f1f07df3117a439ed8029ae2',
 'published-knee.py':'47986ada814961199221cd1563c36bc8e71b25f87eaa1a1a96376a9c0ad16630',
 'published-infer.py':'ec5a570c03061496199b473fa91c865f324ecc47aded44a9a119e3fe8483e6f4'}
WEIGHTS = {'cnxt_v0_fold0.pt':'0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80',
 'cnxt_v0_fold1.pt':'61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb',
 'cnxt_v0_fold2.pt':'f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6'}
RECIPE = {'version':VERSION,'eval_windows':16,'slots':6,'model_res':256,'fov':0.92,
 'batch_size':4,'workers':4,'prefetch_factor':2,'pin_memory':True,'multiprocessing':'fork',
 'split_batches':'three visible together, then sixteen training batches of4; no cross-split batch',
 'candidate_order':'stable descending min(directory entries,64), metadata order breaks ties',
 'missing_slot':'copy opposite fat/nonfat candidate list before candidate0 selection; no retry',
 'folds':3,'aggregation':'mean of three sigmoid(float32 logits) probabilities','tta':'none',
 'autocast':'cuda.float16','cudnn_benchmark':False,'cuda_matmul_allow_tf32':False,
 'failure_policy':'public usable partial/missing slots; resource, unsafe pixels and whole-empty fatal',
 'max_seconds':32400,'runtime_estimate_policy':'informational_only'}
COLUMNS = ['StudyInstanceUID','SeriesInstanceUID','Fluid_Sensitive','Fat_Suppression','Anatomical_Plane']
COUNTERS = ('selected_slots','missing_slots','slot_copies','bad_items','partial_series','series_exceptions',
 'short_series','black_series','geometry_default_frames','resource_errors','invalid_pixels','zero_studies')
STRESS_CASES = {'healthy_public_parity','copy_both_directions','no_copy_when_both_present','both_missing',
 'partial_public_parity','failed_best_no_retry','short_best_no_retry','all_empty_fails',
 'memory_read_fatal','timeout_read_fatal','opencv_read_fatal','memory_append_fatal','timeout_append_fatal'}


def policy_contract():
    return {'version':VERSION,'sha256':sha256(Path(__file__).read_bytes())}


def audit_sources(directory):
    directory=Path(directory);raw=(directory/'spec.json').read_bytes()
    if sha256(raw)!=SPEC_SHA256: raise ValueError('Frozen pre-score Versia specification changed')
    spec=json.loads(raw)
    if spec['convnext_checkpoints_sha256']!=WEIGHTS: raise ValueError('Frozen checkpoint identities differ')
    for name,digest in spec['source_cell_sha256'].items():
        if sha256((directory/name).read_bytes())!=digest: raise ValueError('Frozen Versia source cell changed')
    sources={name:(directory/name).read_text() for name in SOURCE_HASHES}
    if any(sha256(v.encode())!=SOURCE_HASHES[k] for k,v in sources.items()):
        raise ValueError('Unreviewed Versia scientific source')
    return sources,spec


def extract(sources):
    """Keep only pinned inference definitions; never execute public orchestration."""
    inventories = {
      'published-preprocess.py':({'MM_PER_PX','SIZE','MAX_SLICES','CANON','PLANES'}, {'_read','_canon_slice','_resample','load_series'}),
      'published-knee.py':({'LABELS','PLANES','N_SLOTS','MEAN','STD'}, {'slot_of','build_study_table','window_centres','make_views','KneeNet'}),
      'published-infer.py':(set(),{'DicomStudyDataset'})}
    prefixes = {'published-preprocess.py':'import os\nimport cv2\nimport numpy as np\nimport pydicom\n',
      'published-knee.py':'import math\nimport numpy as np\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\n',
      'published-infer.py':'import numpy as np\nimport torch\n'}
    result={}
    for name,(constants,defs) in inventories.items():
        if sha256(sources[name].encode())!=SOURCE_HASHES[name]: raise ValueError('Unreviewed Versia source')
        nodes=[];found=set()
        for node in ast.parse(sources[name]).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in defs:
                nodes.append(node);found.add(node.name)
            elif isinstance(node,ast.Assign):
                names={t.id for t in node.targets if isinstance(t,ast.Name)}
                if names and names<=constants:nodes.append(node);found.update(names)
        if found!=constants|defs: raise ValueError('Versia inference inventory changed')
        result[name]=prefixes[name]+ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n'
    return result


def protect(source):
    """Exactly one catch in each extracted preprocess/Dataset source; math unchanged."""
    tree=ast.parse(source);count=0
    for node in ast.walk(tree):
        if isinstance(node,ast.ExceptHandler):
            if ast.unparse(node.type)!='Exception' or node.name is not None:
                raise ValueError('Versia public exception region changed')
            node.name='_versia_caught'
            node.body.insert(0,ast.Expr(value=ast.Call(func=ast.Name(id='_versia_propagate',ctx=ast.Load()),
                args=[ast.Name(id=node.name,ctx=ast.Load())],keywords=[])))
            count+=1
    if count!=1: raise ValueError('Exactly one Versia protected catch expected')
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def definitions(sources, *, protected=True):
    modules={}
    extracted=extract(sources)
    for key in ('published-preprocess.py','published-knee.py','published-infer.py'):
        source=extracted[key]
        if protected and key!='published-knee.py':source=protect(source)
        module=types.ModuleType('versia_'+key[10:-3].replace('-','_'))
        if key=='published-infer.py':
            module.__dict__.update({n:getattr(modules['published-knee.py'],n) for n in ('LABELS','N_SLOTS','KneeNet','build_study_table','window_centres')})
            module.__dict__.update({n:getattr(modules['published-preprocess.py'],n) for n in ('MAX_SLICES','SIZE','load_series')})
        # Overridden with the current study's counters before any data read.
        def propagate(error):
            if fatal_error(error):raise error
        module._versia_propagate=propagate
        exec(compile(source,'<audited-versia-'+key+'>','exec'),module.__dict__)
        modules[key]=module
    return modules['published-preprocess.py'],modules['published-knee.py'],modules['published-infer.py']


def study_arrays(objects,study,table,directory,*,strict_public=False):
    """Call the exact public Dataset with counters/guards, leaving all image math intact."""
    import numpy as np
    prep,model,infer=objects
    counts=Counter({k:0 for k in COUNTERS});slots=table.get(study,[[] for _ in range(6)])
    counts['selected_slots']=sum(bool(x) for x in slots);counts['missing_slots']=6-counts['selected_slots']
    counts['slot_copies']=sum(bool(slots[p]) and slots[p]==slots[p+3] for p in range(3))
    original_read,original_load=prep._read,infer.load_series
    original_guards=(prep._versia_propagate,infer._versia_propagate)
    class UnsafePixels(BaseException):pass
    def propagate(error):
        if fatal_error(error):
            if not getattr(error,'_versia_counted',False):counts['resource_errors']+=1;error._versia_counted=True
            raise error
    def read(path):
        result=original_read(path)
        for arr,iop,ipp,ps,_ in result:
            if arr.ndim!=2 or not np.isfinite(arr).all():
                counts['invalid_pixels']+=1;raise UnsafePixels
            if iop is None or ipp is None or ps is None:counts['geometry_default_frames']+=1
        return result
    def load(path):
        try:vol,info=prep.load_series(path)  # plane argument intentionally omitted, as published
        except Exception as error:
            propagate(error);counts['series_exceptions']+=1;raise
        if vol.ndim!=3 or vol.shape[1:]!=(prep.SIZE,prep.SIZE) or vol.dtype!=np.uint8:
            raise UnsafePixels
        counts['bad_items']+=int(info['n_bad']);counts['partial_series']+=int(info['n_bad']>0)
        counts['short_series']+=int(len(vol)<3);counts['black_series']+=int(not np.any(vol))
        return vol,info
    try:
        if not strict_public:
            prep._read=read;infer.load_series=load
            prep._versia_propagate=infer._versia_propagate=propagate
        arrays=tuple(x.numpy() for x in infer.DicomStudyDataset([study],table,str(directory),16)[0])
        x,mask,pos=arrays
        if (x.shape!=(6,16,3,prep.SIZE,prep.SIZE) or x.dtype!=np.uint8 or mask.shape!=(6,) or
                mask.dtype!=bool or pos.shape!=(6,16) or not np.isfinite(pos).all() or np.any((pos<0)|(pos>1))):
            raise ValueError('Versia study tensor contract differs')
        if not mask.any() or not np.any(x):counts['zero_studies']+=1;raise ValueError('Whole-empty Versia study; no prediction fallback')
        return arrays,dict(counts)
    except UnsafePixels:
        error=ValueError('Unsafe Versia pixels; no selected-series fallback')
        error.versia_counts=dict(counts)
        raise error from None
    except Exception as error:
        if fatal_error(error) and not getattr(error,'_versia_counted',False):counts['resource_errors']+=1
        error.versia_counts=dict(counts)
        raise
    finally:
        prep._read=original_read;infer.load_series=original_load
        prep._versia_propagate,infer._versia_propagate=original_guards


def strict_eligible(counts):
    return all(counts[k]==0 for k in ('bad_items','series_exceptions','short_series','black_series','geometry_default_frames','invalid_pixels','resource_errors'))


def series_table(model, prep, root, split, ids):
    import pandas as pd
    with (Path(root)/f'{split}_series.csv').open(newline='') as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=COLUMNS:raise ValueError('Official Versia series schema differs')
        rows=list(reader)
    seen=set();chosen=[]
    for r in rows:
        if set(r)!=set(COLUMNS) or any(r[k] is None for k in COLUMNS):raise ValueError('Malformed series row')
        bounded_ids([r['StudyInstanceUID']],count=1);bounded_ids([r['SeriesInstanceUID']],count=1)
        if (r['SeriesInstanceUID'] in seen or r['Fat_Suppression'] not in ('0','1') or
                r['Fluid_Sensitive'] not in ('0','1') or r['Anatomical_Plane'] not in ('Sagittal','Coronal','Axial')):
            raise ValueError('Invalid/duplicate official series metadata')
        seen.add(r['SeriesInstanceUID'])
        if r['StudyInstanceUID'] in set(ids):chosen.append(r)
    if {r['StudyInstanceUID'] for r in chosen}!=set(ids):raise ValueError('Exact selected metadata coverage missing')
    sizes={}
    for r in chosen:
        d=Path(root)/f'{split}_series'/r['StudyInstanceUID']/r['SeriesInstanceUID']
        sizes[r['SeriesInstanceUID']]=min(len(list(d.iterdir())),prep.MAX_SLICES) if d.is_dir() else 0
    return model.build_study_table(pd.DataFrame(chosen),sizes)


def load_models(model_code,assets):
    import timm
    import torch
    paths=[Path(assets)/name for name in WEIGHTS]
    # All three pins pass before any checkpoint is deserialized; no glob expansion.
    for path in paths:
        with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        if digest!=WEIGHTS[path.name]:raise ValueError('Versia intended fold checkpoint changed')
    model_code.timm=timm;models=[]
    for path in paths:
        ck=torch.load(path,map_location='cpu',weights_only=True);args=ck['args']
        if (args.get('backbone')!='convnext_tiny.fb_in22k_ft_in1k' or args.get('res')!=256 or args.get('k_eval')!=12 or
                not ck.get('model') or not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in ck['model'].values())):
            raise ValueError('Versia checkpoint model/training metadata differs')
        model=model_code.KneeNet(args['backbone'].split('.')[0],pretrained=False,img_size=args['res'])
        model.load_state_dict(ck['model'],strict=True)
        models.append((model.to('cuda').to(memory_format=torch.channels_last).eval(),args['res']))
    return models


def predict(models, tensors):
    import numpy as np
    import torch
    x,mask,pos=tensors
    x,mask,pos=x.cuda(non_blocking=True),mask.cuda(),pos.cuda()
    if any(m.training or any(layer.training for layer in m.modules()) for m,_ in models):
        raise ValueError('Versia inference model must be eval')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        logits=[m(x,mask,pos,res=res).float() for m,res in models]
        if not all(torch.isfinite(a).all() for a in logits):raise ValueError('Versia logits nonfinite')
        values=torch.stack([a.sigmoid() for a in logits]).mean(0).cpu().numpy()
    validate_predictions(values[0] if len(values)==1 else values,rows=len(values))
    return values


def stress_inputs(sources):
    """Small synthetic production-path checks; no raw data mutation or model forward."""
    import numpy as np
    import pandas as pd
    import cv2
    prod=definitions(sources);plain=definitions(sources,protected=True)
    for prep,_,infer in (prod,plain):prep.SIZE=infer.SIZE=8  # synthetic windows only; no image preprocessing change
    prep,model,infer=prod
    rows=pd.DataFrame([{'StudyInstanceUID':'synthetic','SeriesInstanceUID':s,'Fat_Suppression':f,'Anatomical_Plane':p}
        for s,f,p in [('first',1,'Sagittal'),('never-retry',1,'Sagittal'),('second',0,'Coronal')]])
    table=model.build_study_table(rows,{'first':8,'never-retry':4,'second':8})
    checks={};volume=np.arange(8*8*8,dtype=np.uint16).reshape(8,8,8).astype(np.uint8)
    def healthy(path):
        if str(path).endswith('never-retry'):raise AssertionError('Candidate retry is not published behavior')
        return volume,{'n_bad':0}
    def install(loader):
        prep.load_series=loader;plain[2].load_series=loader
    def pair():
        result=study_arrays(prod,'synthetic',table,'synthetic')[0]
        expected=study_arrays(plain,'synthetic',table,'synthetic',strict_public=True)[0]
        if not all(np.array_equal(a,b) for a,b in zip(result,expected)):
            raise ValueError('Versia protected/public synthetic tensors differ')
        return result
    install(healthy);baseline=pair();checks['healthy_public_parity']='PASS'
    if table['synthetic']!=[['first','never-retry'],['second'],[],['first','never-retry'],['second'],[]]:
        raise ValueError('Published paired candidate copy changed')
    checks['copy_both_directions']='PASS'
    both=pd.concat([rows,pd.DataFrame([{'StudyInstanceUID':'synthetic','SeriesInstanceUID':'opposite','Fat_Suppression':0,'Anatomical_Plane':'Sagittal'}])])
    two=model.build_study_table(both,{'first':8,'never-retry':4,'second':8,'opposite':7})['synthetic']
    if two[0]!=['first','never-retry'] or two[3]!=['opposite']:raise ValueError('Existing slots substituted')
    checks['no_copy_when_both_present']='PASS'
    if baseline[1].tolist()!=[True,True,False,True,True,False]:raise ValueError('Missing-plane mask differs')
    checks['both_missing']='PASS'
    install(lambda path:(volume,{'n_bad':1}));partial=pair()
    if not all(np.array_equal(a,b) for a,b in zip(baseline,partial)):raise ValueError('Usable partial volume changed')
    checks['partial_public_parity']='PASS'
    for mode in ('failed','short'):
        def changed(path):
            if str(path).endswith('first'):
                if mode=='failed':raise ValueError('synthetic unreadable')
                return volume[:2],{'n_bad':0}
            return healthy(path)
        install(changed);arrays=pair()
        if arrays[1].tolist()!=[False,True,False,False,True,False]:raise ValueError('Unusable best series not masked')
        checks[mode+'_best_no_retry']='PASS'
    try:study_arrays(prod,'synthetic',{},'synthetic')
    except ValueError:checks['all_empty_fails']='PASS'
    else:raise AssertionError('Whole-empty source result accepted')
    # Execute the actual protected load_series broad catch, including append-region
    # errors after _read returns. It must propagate instead of incrementing bad.
    protected=protect(extract(sources)['published-preprocess.py'])
    handler=next(n for n in ast.walk(ast.parse(protected)) if isinstance(n,ast.ExceptHandler))
    for error_type,name in ((MemoryError,'memory_append_fatal'),(TimeoutError,'timeout_append_fatal')):
        attempted=ast.Try(body=[ast.Raise(exc=ast.Name(id='_error',ctx=ast.Load()),cause=None)],handlers=[handler],orelse=[],finalbody=[])
        code=compile(ast.fix_missing_locations(ast.Module(body=[attempted],type_ignores=[])),'<versia-whole-catch-stress>','exec')
        try:exec(code,{'_error':error_type('synthetic'), '_versia_propagate':lambda e:(_ for _ in ()).throw(e)}, {'bad':0})
        except error_type:checks[name]='PASS'
        else:raise AssertionError('Whole read/append catch swallowed resources')
    original=definitions(sources)[0].load_series
    # Same source function/global namespace, injected _read failure inside public catch.
    for error,name in ((MemoryError('synthetic'),'memory_read_fatal'),(TimeoutError('synthetic'),'timeout_read_fatal'),(cv2.error('synthetic'),'opencv_read_fatal')):
        if isinstance(error,cv2.error):error.code=-4
        from unittest.mock import patch
        prep.load_series=types.FunctionType(original.__code__,prep.__dict__,argdefs=original.__defaults__)
        with patch.object(prep.os,'listdir',return_value=['file']),patch.object(prep,'_read',side_effect=error):
            try:study_arrays(prod,'synthetic',table,'synthetic')
            except type(error) as caught:
                if caught.versia_counts['resource_errors']!=1:raise ValueError('Fatal resource accounting differs')
                checks[name]='PASS'
            else:raise AssertionError('Public catches masked resource failure')
    if set(checks)!=STRESS_CASES:raise ValueError('Versia stress coverage incomplete')
    return {'status':'PASS','cases':checks,'uses_raw_data':False,'uses_labels':False}


def require_rehearsal(contract,report,*,bundle_sha256):
    def digest(x):return isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x)
    if (contract.get('recipe')!=RECIPE or contract.get('policy')!=policy_contract() or
        contract.get('spec_sha256')!=SPEC_SHA256 or contract.get('sources')!=SOURCE_HASHES or
        contract.get('weights')!=WEIGHTS or contract.get('private') is not True or contract.get('internet') is not False or
        report.get('status')!='PASS' or report.get('contract_sha256')!=sha256(canonical_bytes(contract)) or
        not digest(bundle_sha256) or report.get('source_bundle_sha256')!=bundle_sha256 or
        report.get('checkpoint_sha256')!=WEIGHTS or report.get('studies')!=67 or report.get('train_studies')!=64 or report.get('test_studies')!=3 or
        report.get('batch_sizes')!=[3]+[4]*16 or
        report.get('prediction_fallbacks')!=0 or report.get('input_counts',{}).get('resource_errors')!=0 or
        report.get('input_counts',{}).get('zero_studies')!=0 or report.get('public_tensor_max_error')!=0 or
        not 0<=report.get('repeat_max_abs_error',float('inf'))<=1e-6 or
        not math.isfinite(report.get('batch4_vs_single_max_abs_error',float('inf'))) or
        report.get('batch4_vs_single_max_abs_error',-1)<0 or
        not isinstance(report.get('batch4_vs_single_rank_order_equal'),bool) or
        report.get('execution')!={'batch_size':4,'workers':4,'prefetch_factor':2,'pin_memory':True,'multiprocessing':'fork'} or
        report.get('stress',{}).get('cases')!=dict.fromkeys(STRESS_CASES,'PASS') or report.get('stress',{}).get('status')!='PASS' or
        report.get('stress',{}).get('uses_labels') is not False or report.get('stress',{}).get('uses_raw_data') is not False or
        not math.isfinite(report.get('elapsed_seconds',float('inf'))) or not 0<report['elapsed_seconds']<=32400):
        raise ValueError('Fresh exact Versia67 component qualification required')
    if any(not isinstance(report.get(k),(int,float)) or not math.isfinite(report[k]) or report[k]<=0
           for k in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_parent_rss_bytes','disk_free_bytes','output_disk_bytes')):
        raise ValueError('Versia actual resource evidence missing')
    sampling=report.get('sampling',{});coverage=report.get('strict_coverage',{})
    if (sampling.get('seed')!=20261008 or sampling.get('uses_labels') is not False or
        sampling.get('selection')!='SHA256(seed:StudyInstanceUID) order' or not digest(sampling.get('ordered_ids_sha256')) or
        coverage.get('PASS',0)+coverage.get('ineligible_partial_input',0)!=67 or coverage.get('PASS',0)<1 or
        not digest(report.get('visible_csv_sha256'))):raise ValueError('Versia cohort/healthy coverage missing')
    env=report.get('environment',{})
    if (env.get('python_major_minor')!=[3,12] or env.get('packages',{}).get('torch')!='2.10.0+cu128' or
        env.get('packages',{}).get('timm')!='1.0.26' or env.get('cudnn_benchmark') is not False or
        env.get('cuda_matmul_allow_tf32') is not False or report.get('codec_hashes')!=contract.get('codecs')):
        raise ValueError('Versia common environment/codec evidence differs')


def rehearsal_batches():
    return [[0,1,2]]+[list(range(i,i+4)) for i in range(3,67,4)]


def rehearse(root, assets, sources, contract, output, *, started, bundle_sha256):
    import numpy as np
    import torch
    if sys.version_info[:2]!=(3,12) or not torch.cuda.is_available():raise RuntimeError('Versia qualification requires Python3.12/CUDA')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    report={'status':'running','contract_sha256':sha256(canonical_bytes(contract)),'contract':contract,
            'source_revision':__import__('os').environ.get('KNEE_CODE_REVISION'),
            'source_bundle_sha256':bundle_sha256,'prediction_fallbacks':0,'input_counts':{k:0 for k in COUNTERS},
            'studies':0,'train_studies':64,'test_studies':3,
            'execution':{k:RECIPE[k] for k in ('batch_size','workers','prefetch_factor','pin_memory','multiprocessing')}}
    def save():
        report['elapsed_seconds']=time.monotonic()-started
        (out/'versia-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        warnings.filterwarnings('ignore',module=r'pydicom(?:\..*)?')
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
        # Independent reference decoding keeps public math but must never turn a
        # resource failure into a matching partial volume and false parity PASS.
        protected=definitions(sources);plain=definitions(sources,protected=True)
        prep,model_code,_=protected
        columns,ids,_=read_test_contract(root,rehearsal=True)
        if columns!=['StudyInstanceUID',*model_code.LABELS]:raise ValueError('Official Versia target order differs')
        with (Path(root)/'train_series.csv').open(newline='') as handle:
            train_ids=sorted({r['StudyInstanceUID'] for r in csv.DictReader(handle)})
        train_ids=bounded_ids(train_ids,count=64,seed=20261008)
        tables={split:series_table(model_code,prep,root,split,rows) for split,rows in [('test',ids),('train',train_ids)]}
        jobs=[('test',sid) for sid in ids]+[('train',sid) for sid in train_ids]
        report['sampling']={'seed':20261008,'uses_labels':False,'selection':'SHA256(seed:StudyInstanceUID) order',
                            'ordered_ids_sha256':sha256(canonical_bytes(jobs))}
        report['stress']=stress_inputs(sources)
        class QualificationDataset(torch.utils.data.Dataset):
            def __len__(self):return len(jobs)
            def __getitem__(self,index):
                split,sid=jobs[index];before=time.monotonic()
                arrays,counts=study_arrays(protected,sid,tables[split],Path(root)/f'{split}_series')
                production_seconds=time.monotonic()-before
                expected,_=study_arrays(plain,sid,tables[split],Path(root)/f'{split}_series',strict_public=True)
                if not all(np.array_equal(a,b) for a,b in zip(arrays,expected)):
                    raise ValueError('Protected/public Versia study tensors differ')
                return (*[torch.from_numpy(a) for a in arrays],counts,production_seconds,strict_eligible(counts))
        dl=torch.utils.data.DataLoader(QualificationDataset(),batch_sampler=rehearsal_batches(),num_workers=4,pin_memory=True,
                                      prefetch_factor=2,multiprocessing_context='fork')
        # Start CPU-only workers before loading CUDA models. This is an explicit
        # orchestration safety choice; original batch/worker counts and math remain.
        iterator=iter(dl)
        models=load_models(model_code,assets);torch.cuda.reset_peak_memory_stats()
        report['checkpoint_sha256']=WEIGHTS
        report['environment']={'python':sys.version,'python_major_minor':list(sys.version_info[:2]),
            'packages':{p:importlib.metadata.version(p) for p in ('torch','timm','numpy','pandas','pydicom','opencv-python')},
            'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'cudnn_benchmark':torch.backends.cudnn.benchmark,'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
            'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
        report['setup_seconds']=time.monotonic()-started
        values=[];single_values=[];repeat_error=0.;batch_error=0.;inference_times=[];decode_times=[];coverage=Counter();counts=Counter()
        report['batch_sizes']=[]
        before_batches=time.monotonic()
        for x,mask,pos,events,durations,eligible in iterator:
            if time.monotonic()-started>=32400:raise TimeoutError('Versia nine-hour platform limit exhausted')
            before=time.monotonic();p=predict(models,(x,mask,pos));torch.cuda.synchronize()
            inference_times.append(time.monotonic()-before)
            q=predict(models,(x,mask,pos));repeat_error=max(repeat_error,float(np.max(np.abs(p-q))))
            if repeat_error>1e-6:raise ValueError('Versia batch4 within-process repeat differs')
            singles=np.concatenate([predict(models,(x[i:i+1],mask[i:i+1],pos[i:i+1])) for i in range(len(p))])
            batch_error=max(batch_error,float(np.max(np.abs(p-singles))))
            report['batch_sizes'].append(len(p))
            values.extend(p);single_values.extend(singles);decode_times.extend(float(t) for t in durations)
            counts.update({k:int(v.sum()) for k,v in events.items()})
            coverage.update('PASS' if bool(x) else 'ineligible_partial_input' for x in eligible)
            report.update(studies=len(values),input_counts=dict(counts),strict_coverage=dict(coverage),
                          repeat_max_abs_error=repeat_error,batch4_vs_single_max_abs_error=batch_error)
            save();print(json.dumps({'event':'versia_batch','completed_studies':len(values),'total':67,'elapsed_seconds':report['elapsed_seconds']}),flush=True)
        del iterator,dl
        values=np.stack(values);single_values=np.stack(single_values)
        validate_predictions(values,rows=67)
        report.update(status='PASS',public_tensor_max_error=0,visible_csv_sha256=write_probabilities(out/'visible-predictions.csv',columns,ids,values[:3]),
            batch4_vs_single_rank_order_equal=bool(np.array_equal(np.argsort(values,axis=0,kind='stable'),np.argsort(single_values,axis=0,kind='stable'))),
            batch4_vs_single_note='Diagnostic only; production retains published batch4. No automatic batch1 fallback.',
            decode_seconds_sum=sum(decode_times),batch_inference_seconds_sum=sum(inference_times),
            qualification_batches_seconds=time.monotonic()-before_batches,
            estimated_1300_seconds_serial_decode=report['setup_seconds']+(sum(decode_times)+sum(inference_times))*1300/67,
            runtime_estimate_policy='informational_only; includes qualification-only reference work in wall clock; production decode can overlap across4workers',
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_parent_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            child_rusage_maxrss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
            disk_free_bytes=shutil.disk_usage(out).free,output_disk_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),
            codec_hashes=contract['codecs'])
        save();require_rehearsal(contract,report,bundle_sha256=bundle_sha256)
        return report
    except Exception as exc:
        if hasattr(exc,'versia_counts'):report['failure_input_counts']=exc.versia_counts
        report.update(status='FAIL',failure_type=type(exc).__name__);save()
        raise RuntimeError('Versia component qualification failed; inspect aggregate report') from None
