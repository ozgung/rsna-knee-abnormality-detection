"""Frozen separate CoAtNet224 rehearsal; CUDA qualification remains artifact-specific.

Copied from own_model_rehearsal at56516d0 to preserve qualified ConvNeXt archives.
Reader and runtime checks are unchanged; only the native224 family and stricter
training provenance are admitted. Subsets prove wiring, never qualify allfive.
"""
import ast
import csv
import importlib.metadata
import json
import math
import re
from pathlib import Path
import resource
import sys
import time
import warnings
from knee import public_training_smoke as smoke
from knee import corpus44_parity as parity
from knee import raptor_input
from knee.fresh_reader_setup import dataset_root,notebook_root,exact_root,time_limit

ARCH='coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k'
KERNELS=['ozgung/knee-tracka-coatnet224-fold01-20261009',
         'ozgung/knee-tracka-coatnet224-fold23-20261009',
         'ozgung/knee-tracka-coatnet224-fold4-20261009']
PATCHED_SHA='eb3d34ffc6aa6f7c500f8ef2039dba0f313b42e371e075e58d4b7a47dde0bd3c'
PATCH_SHA='a254b568eb6a69999899fd0a34402202cbfdc24d17765a1b49f817ab8109ec55'
ENTRY_SHA='c1abaf134622f4e4ea9ffd9330991fd34c0d1185d1ccabae28e3ec55095aa519'
READER_SHA='e331776e19800ef584c653cb9bb90698ff708027b0f9f483ce7afb18338a2f35'
MODEL_DEFINITIONS_SHA='5d02ae67ada6e70cba8f4c4eebd5659d67d328eac5a275a5a465e2731c8b5297'
SELECTION='masked macro AUC of the FULL held-out fold vs report labels (train_labels_v1 y_/m_), not gold'
POLICY={'family':'coatnet224-original-labels-v1','labels_file':'labels_v1_blend.parquet',
        'labels_sha256':smoke.LABEL_SHA,'truth_sha256':'a134cfce2f751f90f4219d791a563ee677b6223ac28f32e456bb3759c1d3fba9',
        'folds_sha256':'12bebc897a549d660cf5e6ab597c9a8d8f84dc887b4abd4c99b67fb4a25cb3ae',
        'excluded_ids_sha256':'44834d7638eb60142a668a7597bd3093fe569e13b6f77e60a9c185a4ac3206f1',
        'selection':SELECTION,'train_knee_patched_sha256':PATCHED_SHA,
        'train_knee_patch_sha256':PATCH_SHA,'tracka_entry_sha256':ENTRY_SHA}
TRAIN_PARAMS={'ARCH':ARCH,'RES':'224','K':'8','K_EVAL':'16','EPOCHS':'10','BS':'4',
              'CKPT':'timm','SEED':'42','TOPK':'3','LABELS_FILE':'labels_v1_blend.parquet',
              'GRAD_CKPT':'1','WIN_CHUNK':'8','ACCUM':'2','BB_LR':'3e-5','HEAD_LR':'1e-3','WD':'0.02'}
SETTINGS={'res':224,'k_eval':16,'norm':'imagenet','precision':'cuda_float16','batch_studies':1,
          'ensemble':'equal_probability_mean','tta':False,'threads':4,'platform_seconds':43200}


def model_definitions(source):
    if smoke.digest(source.encode())!=smoke.SOURCE_SHA:raise ValueError('Public model source differs')
    names=('StudyWindows','build_backbone','RaptorClassifier')
    nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    if tuple(n.name for n in nodes)!=names:raise ValueError('Model definition inventory differs')
    return ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n'


def namespace(code):
    import numpy as np
    import torch,timm,random
    from torch import nn
    from torch.nn import functional as F
    ns={'np':np,'torch':torch,'timm':timm,'random':random,'nn':nn,'F':F,'Dataset':torch.utils.data.Dataset,
        '_roi_square_box':None,'_open_corpus':lambda:None}
    exec(compile(code,'<own-public-model-definitions>','exec'),ns)
    return ns


def windows(ns,volume,mask):
    import numpy as np
    raptor_input.validate_production_volume(volume,mask,[({},44)],image_size=336)
    ds=ns['StudyWindows']('', ['synthetic'], {'synthetic':0}, {'synthetic':np.zeros(12,np.float32)},
                          224,16,train=False,aug=False,norm='imagenet',roi=False)
    ds.vols=volume[None];ds.masks=mask[None]
    x,_=ds[0]
    if tuple(x.shape)!=(16,3,224,224) or not bool(x.isfinite().all()):raise ValueError('Window schema differs')
    return x


def probabilities(values,rows):
    import numpy as np
    a=np.asarray(values)
    if a.shape!=(rows,12) or not np.isfinite(a).all() or not ((a>=0)&(a<=1)).all():
        raise ValueError('Prediction schema/range differs')
    return a


def mean_probabilities(members,rows):
    import numpy as np
    if not members:raise ValueError('No qualified members')
    return probabilities(np.mean(np.stack([probabilities(v,rows) for v in members]),axis=0),rows)


def raw_input(public,uid,records,image_root):
    if not (Path(image_root)/uid).is_dir() or not records.get(uid):raise ValueError('Raw study input absent')
    volume,mask,counts=raptor_input.prepare_study(public,uid,records,image_root,cache_pixels=True)
    if counts.get('headers',0)<=0 or counts.get('decoded_unique_slices',0)<=0:raise ValueError('No raw decoding evidence')
    return volume,mask,counts


def verify_run(record,member):
    """Build-time provenance gate. Inference only hashes receipts, never reads metrics."""
    verify_member(member)
    if (record.get('fold')!=member['fold'] or record.get('smoke') is not False or
        record.get('checkpoint_kind')!='swa_topk' or record.get('checkpoint_file')!=member['checkpoint_file'] or
        record.get('checkpoint_sha256')!=member['checkpoint_sha256'] or record.get('train_returncode')!=0 or
        record.get('epochs_completed')!=10 or record.get('selected_epochs')!=member['selected_epochs'] or
        record.get('train_knee_original_sha256')!=smoke.SOURCE_SHA or
        any(record.get('params',{}).get(k)!=v for k,v in TRAIN_PARAMS.items()) or
        record.get('export',{}).get('k_eval')!=16 or record.get('export',{}).get('res')!=224 or
        record.get('epochs_requested')!=10 or record.get('stopped_by_time_budget') is not False or
        any(record.get(k)!=v for k,v in POLICY.items() if k!='family') or
        record.get('torch')!='2.10.0+cu128' or record.get('timm')!='1.0.26' or record.get('device')!='cuda'):
        raise ValueError('Fitted run identity/settings differ')
    history=record.get('history',[])
    if ([h.get('ep') for h in history]!=list(range(10)) or
        any(type(h.get('oof_auc')) not in (int,float) or not math.isfinite(h['oof_auc']) for h in history) or
        [h['ep'] for h in sorted(history,key=lambda h:-h['oof_auc'])[:3]]!=member['selected_epochs']):
        raise ValueError('Ten complete report-selected epochs required')
    command=record.get('train_cmd',[])
    if '--resume' in command or '--ckpt' not in command or command[command.index('--ckpt')+1:] == [] or command[command.index('--ckpt')+1]!='timm':
        raise ValueError('Generic timm initialization required')


def verify_member(member):
    fold=member.get('fold')
    if (type(fold) is not int or fold not in range(5) or
        member.get('kernel')!=KERNELS[0 if fold<2 else (1 if fold<4 else 2)] or
        member.get('run_file')!=f'run_fold{fold}.json' or
        member.get('checkpoint_file')!=f'raptor_ft_fold{fold}_swa.pt' or
        any(not isinstance(member.get(k),str) or not re.fullmatch('[0-9a-f]{64}',member[k]) for k in ('run_sha256','checkpoint_sha256')) or
        not isinstance(member.get('selected_epochs'),list) or len(member['selected_epochs'])!=3 or
        any(type(e) is not int or e not in range(10) for e in member['selected_epochs']) or len(set(member['selected_epochs']))!=3):
        raise ValueError('Native CoAtNet224 member provenance differs')


def require_contract(contract,*,complete=False):
    members=contract.get('members',[])
    if (contract.get('version')!=1 or contract.get('settings')!=SETTINGS or
        any(contract.get(k)!=v for k,v in POLICY.items()) or
        contract.get('model_source_sha256')!=smoke.SOURCE_SHA or contract.get('reader_sha256')!=READER_SHA or
        contract.get('model_definitions_sha256')!=MODEL_DEFINITIONS_SHA or
        not members):raise ValueError('Frozen CoAtNet224 contract required')
    for member in members:verify_member(member)
    folds=[m['fold'] for m in members]
    if folds!=sorted(set(folds)) or contract.get('complete_family')!=(folds==list(range(5))):
        raise ValueError('Explicit unique ordered folds required')
    if complete and folds!=list(range(5)):raise ValueError('Subset proves wiring only; fullfive qualification required')


def verify_checkpoint(state,member):
    """Metadata and finite tensor checks supplement the byte pin and strict load."""
    import torch
    if (state.get('arch')!=ARCH or type(state.get('res')) is not int or state['res']!=224 or
        state.get('lab')!=smoke.TARGETS or state.get('swa_over')!=member['selected_epochs'] or
        not isinstance(state.get('model'),dict) or not state['model'] or
        any(not torch.is_tensor(value) or not bool(torch.isfinite(value).all()) for value in state['model'].values())):
        raise ValueError('CoAtNet224 checkpoint metadata/tensors differ')


def require_result(report,contract):
    require_contract(contract)
    if (report.get('status')!='PASS' or report.get('contract_sha256')!=smoke.digest(smoke.canonical(contract)) or
        report.get('qualified_folds')!=[m['fold'] for m in contract['members']] or
        report.get('train_studies')!=5 or report.get('test_studies')!=3 or
        report.get('settings')!=SETTINGS or report.get('labels_read') is not False or
        len(report.get('training_parity',[]))!=5 or len(report.get('models',[]))!=len(contract['members']) or
        report.get('stress_checks',{}).get('cases')!=dict.fromkeys(raptor_input.STRESS_CASES,'PASS') or
        report.get('environment',{}).get('packages',{}).get('torch')!='2.10.0+cu128' or
        report.get('environment',{}).get('packages',{}).get('timm')!='1.0.26' or
        not 0<report.get('elapsed_seconds',0)<=43200 or not report.get('visible_csv_sha256')):
        raise ValueError('Complete CoAtNet224 rehearsal evidence required')
    for row in report['training_parity']:
        if len(row['slots'])!=5 or not all(s['exact_hash_match'] and s['mask_match'] for s in row['slots']) or row['windows_exact'] is not True:
            raise ValueError('Raw/corpus tensor parity differs')
    for row,member in zip(report['models'],contract['members']):
        if (row.get('fold')!=member['fold'] or row.get('checkpoint_sha256')!=member['checkpoint_sha256'] or row.get('strict_load') is not True or row.get('repeat_max_error')!=0 or row.get('train_probability_max_error')!=0 or
            row.get('missing_plane_finite') is not True or not row.get('checkpoint_sha256') or row.get('cuda_peak_bytes',0)<=0):
            raise ValueError('CoAtNet224 inference qualification differs')


def run(inputs,out,contract,reader_code,model_code,*,started):
    import numpy as np
    import torch
    require_contract(contract)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','phase':'setup','contract_sha256':smoke.digest(smoke.canonical(contract)),
            'settings':SETTINGS,'labels_read':False,'train_studies':5,'test_studies':3,
            'qualified_folds':[],'training_parity':[],'models':[]}
    def save():
        report['elapsed_seconds']=time.monotonic()-started
        report['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        p=out/'rehearsal-report.json.tmp';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(out/'rehearsal-report.json')
    save()
    try:
        with time_limit(43200-(time.monotonic()-started),'actual12h platform limit'):
            if sys.version_info[:2]!=(3,12) or not torch.cuda.is_available():raise RuntimeError('Python3.12 CUDA required')
            warnings.filterwarnings('ignore',module=r'pydicom(?:\..*)?')
            import subprocess
            codecroot=dataset_root(inputs,'goodpjw2008/rsna-knee-2-5d-convnext-reader',list(contract['codecs']))
            wheels=[]
            for name,sha in contract['codecs'].items():smoke.require_file(codecroot/name,sha);wheels.append(str(codecroot/name))
            subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*wheels],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            versions={p:importlib.metadata.version(p) for p in ('torch','timm','numpy','pydicom','pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg')}
            if versions['torch']!='2.10.0+cu128' or versions['timm']!='1.0.26' or versions['numpy']!='2.0.2':raise ValueError('Pinned inference dependencies differ')
            torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            report['environment']={'python':sys.version.split()[0],'packages':versions,'gpu':torch.cuda.get_device_name(0),
                'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'device':'cuda:0'}
            report['backend']={'cudnn_benchmark':False,'cudnn_deterministic':True,'matmul_tf32':False,'cudnn_tf32':False}
            if smoke.digest(reader_code.encode())!=contract['reader_sha256'] or smoke.digest(model_code.encode())!=contract['model_definitions_sha256']:
                raise ValueError('Frozen scientific definitions differ')
            public=parity.reader_module(reader_code)
            protected,metadata=raptor_input.transform_exception_handlers(reader_code)
            if metadata['sha256']!=contract['protected_reader_sha256']:raise ValueError('Fatal catch policy differs')
            exec(compile(protected,'<fatal-protected-reader>','exec'),public.__dict__)
            public._raptor_protected_source=protected;public._raptor_propagate_fatal=raptor_input.propagate_fatal
            ns=namespace(model_code)
            identity=notebook_root(inputs,smoke.IDENTITY_KERNEL,'sealed_identity',['sealed-manifest-private.json'])
            selected,cohort=smoke.membership(identity/'sealed-manifest-private.json');selected=selected[:5]
            if cohort!=contract['smoke_cohort'] or smoke.digest(smoke.canonical(selected))!=contract['selected5_sha256']:raise ValueError('Bounded cohort differs')
            inputs=Path(inputs);competition=exact_root([inputs/'competitions/rsna-knee-abnormality-detection',inputs/'rsna-knee-abnormality-detection'],['train_series.csv','test_series.csv','test.csv','sample_submission.csv'])
            with (competition/'test.csv').open(newline='') as f:
                table=csv.DictReader(f)
                if table.fieldnames!=['StudyInstanceUID']:raise ValueError('Visible test schema differs')
                visible=[r['StudyInstanceUID'] for r in table]
            if len(visible)!=3 or len(set(visible))!=3:raise ValueError('Exactly3 distinct visible studies required')
            with (competition/'sample_submission.csv').open(newline='') as f:
                columns=next(csv.reader(f))
            if columns!=['StudyInstanceUID',*smoke.TARGETS]:raise ValueError('Official target order differs')
            records={}
            for split,uids in [('train',selected),('test',visible)]:
                records[split]={u:[] for u in uids}
                with (competition/f'{split}_series.csv').open(newline='') as f:
                    for row in csv.DictReader(f):
                        if row['StudyInstanceUID'] in records[split]:records[split][row['StudyInstanceUID']].append(row)
            parity.require_image_tree(competition,selected,records['train'])
            roots=[dataset_root(inputs,ref,[f'{prefix}_{suffix}.npy' for suffix in ('ids','masks','vols')]) for ref,prefix in zip(smoke.CORPORA,['all','extra'])]
            ids=[];vols=[];masks=[]
            for root,prefix,size in zip(roots,['all','extra'],[3200,1207]):
                for suffix in ('ids','masks'):
                    name=f'{prefix}_{suffix}.npy';smoke.require_file(root/name,smoke.PINS[name])
                ids.extend(np.load(root/f'{prefix}_ids.npy',allow_pickle=False).astype(str))
                v=np.load(root/f'{prefix}_vols.npy',mmap_mode='r',allow_pickle=False)
                if v.shape!=(size,44,336,336) or v.dtype!=np.uint8:raise ValueError('Corpus volume schema differs')
                vols.append(v);masks.append(np.load(root/f'{prefix}_masks.npy',allow_pickle=False))
            if len(set(ids))!=4407 or not set(selected)<=set(ids):raise ValueError('Corpus coverage differs')
            report['phase']='reader_stress';report['stress_checks']=raptor_input.stress_inputs(public);save()
            raw_windows=[];corpus_windows=[];missing=None;report['phase']='input_parity'
            for split,uids in [('train',selected),('test',visible)]:
                for ordinal,uid in enumerate(uids):
                    begin=time.monotonic();volume,mask,counts=raw_input(public,uid,records[split],competition/f'{split}_series')
                    x=windows(ns,volume,mask);raw_windows.append(x)
                    if split=='train':
                        row=ids.index(uid);part=0 if row<3200 else 1;offset=row if part==0 else row-3200
                        other=vols[part][offset];othermask=masks[part][offset];slots=parity.compare(volume,mask,other,othermask)
                        ox=windows(ns,other,othermask);same=bool(torch.equal(x,ox))
                        report['training_parity'].append({'ordinal':ordinal,'slots':slots,'windows_exact':same,'reader_counts':counts,'elapsed_seconds':time.monotonic()-begin})
                        if not same or not all(s['exact_hash_match'] and s['mask_match'] for s in slots):raise ValueError('Raw/corpus tensors differ')
                        corpus_windows.append(ox)
                        if missing is None:
                            mv=volume.copy();mm=mask.copy();mv[:22]=0;mm[:22]=0
                            if mm.sum()>0:missing=windows(ns,mv,mm)
                    else:
                        report.setdefault('visible_input',[]).append({'ordinal':ordinal,'reader_counts':counts,'elapsed_seconds':time.monotonic()-begin})
                    save()
            if missing is None:raise ValueError('No eligible missing-sagittal stress fixture')
            members=[];report['phase']='models'
            for member in contract['members']:
                assets=notebook_root(inputs,member['kernel'],'trackA',[member['checkpoint_file'],member['run_file']])
                # Hash provenance bytes only; never parse fitted training/gold metrics during inference.
                smoke.require_file(assets/member['run_file'],member['run_sha256'])
                smoke.require_file(assets/member['checkpoint_file'],member['checkpoint_sha256'])
                state=torch.load(assets/member['checkpoint_file'],map_location='cpu',weights_only=True)
                verify_checkpoint(state,member)
                begin=time.monotonic();bb=ns['build_backbone'](ARCH,pretrained=False);model=ns['RaptorClassifier'](bb,F_dim=bb.num_features)
                model.load_state_dict(state['model'],strict=True);model=model.to('cuda').eval();del state
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();load_seconds=time.monotonic()-begin
                def predict(xs):
                    output=[]
                    with torch.inference_mode():
                        for x in xs:
                            with torch.autocast('cuda',dtype=torch.float16):p=torch.sigmoid(model(x[None].to('cuda')).float())
                            output.append(p.cpu().numpy()[0])
                    torch.cuda.synchronize();return probabilities(np.stack(output),len(xs))
                begin=time.monotonic();pred=predict(raw_windows);inference_seconds=time.monotonic()-begin
                corpus_pred=predict(corpus_windows);repeat=predict(raw_windows);stress=predict([missing])
                err=float(np.max(np.abs(pred[:5]-corpus_pred)));repeat_err=float(np.max(np.abs(pred-repeat)))
                if err!=0 or repeat_err!=0:raise ValueError('Probability parity/repeat differs')
                report['models'].append({'fold':member['fold'],'strict_load':True,'checkpoint_sha256':member['checkpoint_sha256'],
                    'train_probability_max_error':err,'repeat_max_error':repeat_err,'missing_plane_finite':bool(np.isfinite(stress).all()),
                    'load_seconds':load_seconds,'inference8_seconds':inference_seconds,'cuda_peak_bytes':torch.cuda.max_memory_allocated()})
                members.append(pred[5:]);report['qualified_folds'].append(member['fold']);del model,bb;torch.cuda.empty_cache();save()
            final=mean_probabilities(members,3)
            path=out/'visible.csv'
            with path.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(columns)
                for uid,row in zip(visible,final):w.writerow([uid,*[format(float(v),'.9g') for v in row]])
            with path.open(newline='') as f:
                reader=csv.DictReader(f)
                if reader.fieldnames!=columns:raise ValueError('Written CSV columns differ')
                rows=list(reader)
            if [r['StudyInstanceUID'] for r in rows]!=visible:raise ValueError('Written CSV coverage differs')
            probabilities([[float(r[t]) for t in smoke.TARGETS] for r in rows],3)
            report['visible_csv_sha256']=smoke.file_sha(path);report['output_disk_bytes']=path.stat().st_size
            report['status']='PASS';report['phase']='complete';save();require_result(report,contract)
    except BaseException as error:
        report['status']='ERROR';report['failure']={'phase':report['phase'],'type':type(error).__name__};save()
        raise RuntimeError('CoAtNet224 rehearsal failed; sanitized report saved') from None
    return report
