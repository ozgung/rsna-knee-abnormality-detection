"""Own224-d96 execution safeguards; model/sampler/optimization remain historical."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

from knee import source96_loader as loader,public_training_smoke as smoke,source96_cache as cache

ARCH='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k'
LABEL_SHA='d2cdb8a37e0c4c491758d17c96e3bd4081f06b257c7849608df88590368e94e2'
TRUTH_SHA='a134cfce2f751f90f4219d791a563ee677b6223ac28f32e456bb3759c1d3fba9'
FOLDS_SHA='12bebc897a549d660cf5e6ab597c9a8d8f84dc887b4abd4c99b67fb4a25cb3ae'
EXCLUSIONS_SHA='44834d7638eb60142a668a7597bd3093fe569e13b6f77e60a9c185a4ac3206f1'
FROZEN={'arch':ARCH,'res':224,'k':8,'k_eval':16,'epochs':10,'bs':4,'accum':2,
        'win_chunk':8,'topk':3,'seed':42,'norm':'imagenet','grad_ckpt':True,
        'fp16':True,'xla':False,'smoke':False,'roi':False,'freeze_blocks':0,'limit':0,
        'folds':5,'fold':0,'time_budget_h':0.,'ckpt':'timm','bb_lr':3e-5,'head_lr':1e-3,'wd':.02}
_CORPUS=None
_CACHE_IDENTITY=None


def checked_json(path,sha):
    if not loader._hash(sha) or smoke.file_sha(path)!=sha:raise ValueError('Pinned JSON bytes differ')
    return json.loads(Path(path).read_bytes())


def staged_corpus(*,allow_fixture=False):
    global _CORPUS,_CACHE_IDENTITY
    if _CORPUS is not None:
        if _CACHE_IDENTITY.get('fixture') and not allow_fixture:raise ValueError('Fixture corpus cannot train production')
        return _CORPUS
    root=Path(os.environ['SOURCE96_STAGE_ROOT'])
    expected=os.environ['SOURCE96_MANIFEST_SHA256'];content=os.environ['SOURCE96_CONTENT_SHA256']
    receipt=checked_json(root/'staging-receipt.json',os.environ['SOURCE96_STAGE_RECEIPT_SHA256'])
    if receipt.get('status')!='PASS' or receipt.get('manifest_sha256')!=expected or receipt.get('content_sha256')!=content:
        raise ValueError('Trusted staging PASS required')
    manifest=checked_json(root/'cache96-manifest-private.json',receipt['manifest_file_sha256'])
    if manifest.get('content_sha256')!=content:raise ValueError('Verified image/mask identity differs')
    vols,masks,ids=loader.open_corpus(manifest,expected_manifest_sha256=expected,allow_fixture=allow_fixture)
    _CORPUS=(vols,masks,ids)
    _CACHE_IDENTITY={'content_sha256':content,'membership':manifest['membership'],'fixture':manifest['fixture'],
                     'sources':[{k:v for k,v in s.items() if k!='root'} for s in manifest['sources']],
                     'manifest_sha256_observed':expected}
    return _CORPUS


def recipe(script):
    root=Path(script).parent
    manifest=json.loads((root/'manifest.json').read_bytes())
    for name, sha in manifest['sources'].items():
        if smoke.file_sha(root/name)!=sha:raise ValueError('Packaged runtime module differs')
    return checked_json(root/'source96-recipe.json',manifest['recipe_sha256'])



SPLITS={0:{'fold':0,'eligible':4345,'training':3475,'development':870,
    'ordered_eligible_sha256':'15c3d837146dbf280272451d082d2e5e996805844b8e727b49426d6a92c956db',
    'ordered_training_sha256':'5c669e06b582172921cf64b0cab925b74ee9e119ac572e044bbbe38f3c8923e6',
    'ordered_development_sha256':'40460d3250445caccd9158f988e3955e2a52392951b00a5104b0d34e87c6f246'},
 1:{'fold':1,'eligible':4345,'training':3479,'development':866,
    'ordered_eligible_sha256':'15c3d837146dbf280272451d082d2e5e996805844b8e727b49426d6a92c956db',
    'ordered_training_sha256':'af3f37b61ab1a64e03563e0798f6771a9b2c9f101a1f415bdfde8e3b1d321f09',
    'ordered_development_sha256':'a4291641b93ac8b5a9d836b9293fb778fa9474ff7dfa782e8c4fd5a1391987ff'}}


def policy_fold(policy):
    split=policy.get('experiment_split')
    if split is None:
        if policy.get('version')!=1 or policy.get('frozen')!=FROZEN:raise ValueError('Legacy policy is fold0 only')
        return 0
    fold=split.get('fold')
    if type(fold) is not int or fold not in SPLITS or split!=SPLITS[fold] or policy.get('version')!=2 or policy.get('experiment_split_sha256')!=smoke.digest(smoke.canonical(split)) or policy.get('frozen')!={**FROZEN,'fold':fold}:raise ValueError('Pinned experiment split policy differs')
    return fold


def experiment_split(ids,folds,excluded,fold):
    if type(fold) is not int or fold not in SPLITS:raise ValueError('Only reviewed folds0/1 supported')
    ids=list(map(str,ids));excluded=set(excluded)
    if len(ids)!=4345 or len(set(ids))!=4345 or len(excluded)!=4 or excluded.intersection(ids) or any(u not in folds or type(folds[u]) is not int or not 0<=folds[u]<=4 for u in ids):raise ValueError('Eligible population/fold/exclusion coverage differs')
    train=[u for u in ids if folds[u]!=fold];dev=[u for u in ids if folds[u]==fold]
    result={'fold':fold,'eligible':len(ids),'training':len(train),'development':len(dev),
            'ordered_eligible_sha256':smoke.digest(smoke.canonical(ids)),
            'ordered_training_sha256':smoke.digest(smoke.canonical(train)),
            'ordered_development_sha256':smoke.digest(smoke.canonical(dev))}
    if result!=SPLITS[fold]:raise ValueError('Ordered experiment split differs')
    return result,train,dev


def runtime_split(fold,folds_path,excluded_path,script):
    if policy_fold(recipe(script))!=fold:raise ValueError('Cross-fold experiment forbidden')
    fmap=checked_json(folds_path,FOLDS_SHA)['folds'];excluded=checked_json(excluded_path,EXCLUSIONS_SHA)['quarantined_study_ids']
    staged_corpus()
    result,train,dev=experiment_split(_CORPUS[2],fmap,excluded,fold)
    return result,train,dev


def require_recipe(args,script):
    policy=recipe(script)
    fold=policy_fold(policy)
    for key,value in {**FROZEN,'fold':fold}.items():
        if getattr(args,key,None)!=value:raise ValueError('Frozen training recipe differs: '+key)
    if args.workers!=policy['workers'] or (args.workers!=2 and not policy.get('worker_choice_name')):
        raise ValueError('Unnamed worker override forbidden')
    return policy


def strict_cuda(torch_mod=None):
    torch=torch_mod or __import__('torch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU fallback forbidden')
    torch.set_num_threads(1)
    return 'cuda'


def tensor_hash(state):
    h=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        array=tensor.detach().cpu().contiguous().numpy()
        h.update(smoke.canonical([name,str(array.dtype),list(array.shape)]));h.update(array.tobytes())
    return h.hexdigest()


def generic_backbone(backbone):
    """Strict normalized generic ImageNet backbone-only artifact, never prefixes/subsets."""
    import torch
    descriptor_path=os.environ['SOURCE96_INITIALIZER_DESCRIPTOR']
    descriptor_sha=os.environ['SOURCE96_INITIALIZER_DESCRIPTOR_SHA256']
    d=checked_json(descriptor_path,descriptor_sha)
    if (d.get('kind')!='imagenet_backbone' or d.get('arch')!=ARCH or d.get('pretrained_tag')!='sw_in12k_ft_in1k' or
            d.get('hf_repo')!='timm/'+ARCH or not re.fullmatch('[a-f0-9]{40}',d.get('hf_revision','')) or
            not loader._hash(d.get('origin_file_sha256')) or not loader._hash(d.get('weights_sha256'))):
        raise ValueError('Verified generic ImageNet provenance required')
    path=Path(d['weights_file'])
    if path.is_symlink() or smoke.file_sha(path)!=d['weights_sha256']:raise ValueError('Generic initializer file differs')
    state=torch.load(path,map_location='cpu',weights_only=True)
    expected=backbone.state_dict()
    if not isinstance(state,dict) or not state or set(state)!=set(expected):raise ValueError('Initializer must exactly cover backbone keys')
    if any(not torch.is_tensor(v) or v.shape!=expected[k].shape or v.dtype!=expected[k].dtype or not bool(torch.isfinite(v).all()) for k,v in state.items()):
        raise ValueError('Initializer tensor schema/finiteness differs')
    backbone.load_state_dict(state,strict=True)
    return {'descriptor_sha256':descriptor_sha,'file_sha256':d['weights_sha256'],'origin_file_sha256':d['origin_file_sha256'],
            'hf_repo':d['hf_repo'],'hf_revision':d['hf_revision'],'backbone_state_sha256':tensor_hash(state),
            'kind':'generic ImageNet only','historical_initializer_identity':'unresolved; recipe-level comparator'}


def initial_record(model,initializer,script):
    import torch,timm
    record={**initializer,'full_initial_model_sha256':tensor_hash(model.state_dict()),
            'torch':torch.__version__,'timm':timm.__version__,'device':torch.cuda.get_device_name(0),
            'worker_rng_resume':'Main RNG restored; persistent worker RNG/prefetch not restored; no bit-identical continuation claim'}
    cache.atomic_json(Path(script).parent/'initial-state-record.json',record)
    return record


def run_identity(args,script,labels,initial):
    import torch,timm,numpy
    staged_corpus()
    if smoke.file_sha(labels)!=LABEL_SHA or smoke.file_sha(args.oof_mask)!=TRUTH_SHA or smoke.file_sha(args.fold_file)!=FOLDS_SHA or smoke.file_sha(args.excluded_ids)!=EXCLUSIONS_SHA:
        raise ValueError('Historical supervision/split pins differ')
    body={k:v for k,v in vars(args).items() if k not in ('labels','oof_mask','fold_file','excluded_ids','resume','roi_boxes')}
    return {'version':1,'recipe':recipe(script)['recipe'],'experiment_split':runtime_split(args.fold,args.fold_file,args.excluded_ids,script)[0],'cache_reference_partition_fold':0,'fold':args.fold,'n_folds':args.folds,'tag':args.tag,
            'corpus_ids_sha256':{'source96_ordered_eligible':_CACHE_IDENTITY['membership']['ordered_eligible_sha256']},
            'epochs_requested':10,'script_sha256':smoke.file_sha(script),
            'helper_sha256':smoke.file_sha(__file__),'loader_sha256':smoke.file_sha(loader.__file__),
            'config':body,'labels_sha256':LABEL_SHA,'oof_mask_sha256':TRUTH_SHA,'fold_file_sha256':FOLDS_SHA,'excluded_ids_sha256':EXCLUSIONS_SHA,
            'cache':{k:v for k,v in _CACHE_IDENTITY.items() if k!='manifest_sha256_observed'},'initializer':initial,
            'policy':recipe(script),'environment':{'torch':torch.__version__,'timm':timm.__version__,'numpy':numpy.__version__,
                                                   'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0)}}


def validate_resume(state,identity):
    required=('model','opt','sched','scaler','best','best_state','best_P','hist','topk','epoch','rng','identity')
    if not isinstance(state,dict) or any(k not in state for k in required) or state['identity']!=identity:
        raise ValueError('Resume state/image/initializer/config identity differs')
    epoch=state['epoch']
    if type(epoch) is not int or not 0<=epoch<10 or [h['ep'] for h in state['hist']]!=list(range(epoch+1)):
        raise ValueError('Resume epoch history incomplete')
    if not state['topk'] or len(state['topk'])>3 or len({t['epoch'] for t in state['topk']})!=len(state['topk']) or any(not 0<=t['epoch']<=epoch for t in state['topk']):
        raise ValueError('Resume top3 state invalid')


def restore_rng(rng):
    import numpy as np,torch
    random.setstate(rng['py']);np.random.set_state(rng['np']);torch.set_rng_state(rng['torch'].cpu())
    if torch.cuda.is_available():
        if not rng.get('cuda') or len(rng['cuda'])!=torch.cuda.device_count():raise ValueError('Resume CUDA RNG topology differs')
        torch.cuda.set_rng_state_all([s.cpu() for s in rng['cuda']])


def save_resume(path,state,identity):
    import numpy as np,torch
    state={**state,'identity':identity,'rng':{'py':random.getstate(),'np':np.random.get_state(),'torch':torch.get_rng_state(),
          'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    validate_resume(state,identity)
    path=Path(path);tmp=path.with_name(path.name+'.tmp');torch.save(state,tmp);tmp.replace(path)
    # A caller may supply only an independently verified real platform deadline.
    deadline=os.environ.get('SOURCE96_ACTUAL_PLATFORM_DEADLINE_UTC_SECONDS')
    if deadline and time.time()+120>=float(deadline) and state['epoch']<9:
        print('[resume] completed epoch checkpoint saved before supplied actual platform deadline',flush=True)
        raise SystemExit(0)


def verify_membership(train_ids,dev_ids,gold_ids,fold=0,folds_path=None,excluded_path=None,script=None):
    staged_corpus();m=_CACHE_IDENTITY['membership']
    if script is not None:m=runtime_split(fold,folds_path,excluded_path,script)[0]
    elif fold!=0:raise ValueError('Legacy membership is fold0 only')
    if gold_ids or (len(train_ids),len(dev_ids))!=(m['training'],m['development']) or smoke.digest(smoke.canonical(train_ids))!=m['ordered_training_sha256'] or smoke.digest(smoke.canonical(dev_ids))!=m['ordered_development_sha256']:
        raise ValueError('Fixed eligible training/development membership differs')



def verify_export_inputs(labels,truth,folds,excluded,development_ids,fold=0,script=None):
    staged_corpus()
    pins=((labels,LABEL_SHA),(truth,TRUTH_SHA),(folds,FOLDS_SHA),(excluded,EXCLUSIONS_SHA))
    if any(smoke.file_sha(path)!=sha for path,sha in pins):raise ValueError('Export supervision/split pins differ')
    m=_CACHE_IDENTITY['membership']
    if script is not None:m=runtime_split(fold,folds,excluded,script)[0]
    elif fold!=0:raise ValueError('Legacy export is fold0 only')
    if len(development_ids)!=m['development'] or smoke.digest(smoke.canonical(list(development_ids)))!=m['ordered_development_sha256']:
        raise ValueError('Export development membership differs')


def entry_guard(c,script):
    policy=recipe(script)
    expected={'ARCH':ARCH,'RES':'224','EPOCHS':'10','CKPT':'timm','BS':'4','K':'8','K_EVAL':'16','SEED':'42','TOPK':'3',
              'GRAD_CKPT':'1','WIN_CHUNK':'8','ACCUM':'2','LABELS_FILE':'labels_v1_blend.parquet','TIME_BUDGET_H':'0',
              'FOLD_A':str(policy_fold(policy)),'FOLD_B':'','BB_LR':'3e-5','HEAD_LR':'1e-3','WD':'0.02','WORKERS':str(policy['workers'])}
    if c['SMOKE'] or any(str(c.get(k))!=v for k,v in expected.items()):raise ValueError('Frozen entry parameters differ')
    return policy


def export_guard(work,fold,checkpoint_path,kind,epochs,partial,secondary):
    if (Path(work)/'manifest.json').exists() and policy_fold(recipe(Path(work)/'train_knee_patched.py'))!=fold:raise ValueError('Cross-fold checkpoint export forbidden')
    if partial or kind!='swa_topk' or len(epochs)!=3:raise ValueError('Only completed frozen report-selected top3 SWA may export')
    meta=json.loads((Path(work)/f'raptor_ft_fold{fold}.json').read_bytes())
    if meta.get('partial') or len(meta.get('hist',[]))!=10:raise ValueError('Ten completed epochs required')
    if secondary:
        primary=json.loads((Path(work)/f'run_fold{fold}.json').read_bytes())
        if primary.get('export',{}).get('k_eval')!=16 or primary['checkpoint_sha256']!=smoke.file_sha(checkpoint_path) or primary['selected_epochs']!=epochs:
            raise ValueError('K94 requires unchanged qualified primaryK16 checkpoint')
