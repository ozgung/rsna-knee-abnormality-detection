"""Own strict-loaded fold models: offline image-only bounded rehearsal."""
import ast
import csv
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
import warnings
from knee import public_training_smoke as smoke
from knee import corpus44_parity as parity
from knee import raptor_input
from knee.fresh_reader_setup import dataset_root,notebook_root,exact_root,time_limit

ARCH='convnext_tiny.fb_in22k_ft_in1k'
SETTINGS={'res':256,'k_eval':16,'norm':'imagenet','precision':'cuda_float16','batch_studies':1,
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
                          256,16,train=False,aug=False,norm='imagenet',roi=False)
    ds.vols=volume[None];ds.masks=mask[None]
    x,_=ds[0]
    if tuple(x.shape)!=(16,3,256,256) or not bool(x.isfinite().all()):raise ValueError('Window schema differs')
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
    if (record.get('fold')!=member['fold'] or record.get('smoke') is not False or
        record.get('checkpoint_kind')!='swa_topk' or record.get('checkpoint_file')!=member['checkpoint_file'] or
        record.get('checkpoint_sha256')!=member['checkpoint_sha256'] or record.get('train_returncode')!=0 or
        record.get('epochs_completed')!=10 or record.get('selected_epochs')!=member['selected_epochs'] or
        record.get('train_knee_original_sha256')!=smoke.SOURCE_SHA or
        record.get('params',{}).get('ARCH')!=ARCH or record.get('params',{}).get('RES')!='256' or
        record.get('export',{}).get('k_eval')!=16):raise ValueError('Fitted run identity/settings differ')


def require_result(report,contract):
    if (report.get('status')!='PASS' or report.get('contract_sha256')!=smoke.digest(smoke.canonical(contract)) or
        report.get('qualified_folds')!=[m['fold'] for m in contract['members']] or
        report.get('train_studies')!=5 or report.get('test_studies')!=3 or
        report.get('settings')!=SETTINGS or report.get('labels_read') is not False or
        len(report.get('training_parity',[]))!=5 or len(report.get('models',[]))!=len(contract['members']) or
        report.get('stress_checks',{}).get('cases')!=dict.fromkeys(raptor_input.STRESS_CASES,'PASS') or
        report.get('environment',{}).get('packages',{}).get('torch')!='2.10.0+cu128' or
        report.get('environment',{}).get('packages',{}).get('timm')!='1.0.26' or
        not 0<report.get('elapsed_seconds',0)<=43200 or not report.get('visible_csv_sha256')):
        raise ValueError('Complete own-model rehearsal evidence required')
    for row in report['training_parity']:
        if not all(s['exact_hash_match'] and s['mask_match'] for s in row['slots']) or row['windows_exact'] is not True:
            raise ValueError('Raw/corpus tensor parity differs')
    for row,member in zip(report['models'],contract['members']):
        if (row.get('fold')!=member['fold'] or row.get('checkpoint_sha256')!=member['checkpoint_sha256'] or row.get('strict_load') is not True or row.get('repeat_max_error')!=0 or row.get('train_probability_max_error')!=0 or
            row.get('missing_plane_finite') is not True or not row.get('checkpoint_sha256') or row.get('cuda_peak_bytes',0)<=0):
            raise ValueError('Own model inference qualification differs')


def run(inputs,out,contract,reader_code,model_code,*,started):
    import numpy as np
    import torch
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
                if state.get('arch')!=ARCH or state.get('res')!=256 or state.get('lab')!=smoke.TARGETS or state.get('swa_over')!=member['selected_epochs']:
                    raise ValueError('Checkpoint metadata differs')
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
        raise RuntimeError('Own model rehearsal failed; sanitized report saved') from None
    return report
