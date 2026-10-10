"""Separate frozen CoAtNet224 five-fold streaming inference.

Derived from own_model_submission at56516d0; ConvNeXt archives stay unchanged.
Only exact fullfive CoAtNet224 rehearsal evidence is admitted.
"""
import csv,importlib.metadata,json,re,resource,sys,time,warnings
from pathlib import Path
from knee import coatnet224_rehearsal as own,public_training_smoke as smoke,corpus44_parity as parity,raptor_input
from knee.fresh_reader_setup import dataset_root,notebook_root,exact_root,time_limit
from knee.reference_blend import read_test_contract
MAX_SECONDS=9*3600
SERIES_COLUMNS=['StudyInstanceUID','SeriesInstanceUID','Fluid_Sensitive','Fat_Suppression','Anatomical_Plane']
KERNELS=own.KERNELS


def training_policy(contract):
    own.require_contract(contract,complete=True)
    return KERNELS


def require_qualification(contract,report):
    """Fullfive measured rehearsal required; build-only/subset receipts are insufficient."""
    own.require_contract(contract,complete=True)
    own.require_result(report,contract)


def require_cuda(torch):
    if sys.version_info[:2]!=(3,12) or not torch.cuda.is_available():
        raise RuntimeError('Python3.12 CUDA required; no CPU fallback')


def read_inputs(root):
    root=Path(root);columns=['StudyInstanceUID',*smoke.TARGETS]
    def ids_from(name,expected):
        with (root/name).open(newline='') as f:
            reader=csv.reader(f)
            if next(reader,None)!=expected:raise ValueError('Official test schema differs')
            rows=list(reader)
        if not rows or any(len(r)!=len(expected) for r in rows):raise ValueError('Malformed test rows')
        ids=[r[0] for r in rows]
        if len(set(ids))!=len(ids) or any(not re.fullmatch('[A-Za-z0-9_.-]+',u) for u in ids):raise ValueError('Invalid test coverage')
        return ids
    with (root/'test_series.csv').open(newline='') as f:
        if next(csv.reader(f),None)!=SERIES_COLUMNS:raise ValueError('Official series schema differs')
    with (root/'sample_submission.csv').open(newline='') as f:
        if next(csv.reader(f),None)!=columns:raise ValueError('Official target order differs')
    actual_columns,ids,_=read_test_contract(root,rehearsal=False)
    if actual_columns!=columns or ids!=ids_from('test.csv',['StudyInstanceUID']):raise ValueError('Official test schema/order differs')
    records={u:[] for u in ids};seen=set()
    with (root/'test_series.csv').open(newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=SERIES_COLUMNS:raise ValueError('Official series schema differs')
        for row in reader:
            uid,sid=row['StudyInstanceUID'],row['SeriesInstanceUID']
            if (set(row)!=set(SERIES_COLUMNS) or any(v is None for v in row.values()) or
                uid not in records or not re.fullmatch('[A-Za-z0-9_.-]+',sid or '') or sid in seen or
                row['Fluid_Sensitive'] not in ('0','1') or row['Fat_Suppression'] not in ('0','1') or
                row['Anatomical_Plane'] not in ('Sagittal','Coronal','Axial')):raise ValueError('Malformed series metadata')
            seen.add(sid);records[uid].append(row)
    if any(not rows for rows in records.values()):raise ValueError('Study has no series metadata')
    return columns,ids,records


def validate_csv(path,ids):
    if not ids or len(set(ids))!=len(ids) or any(not isinstance(uid,str) or not re.fullmatch('[A-Za-z0-9_.-]+',uid) for uid in ids):
        raise ValueError('Distinct authoritative test IDs required')
    with Path(path).open(newline='') as f:
        reader=csv.reader(f)
        if next(reader,None)!=['StudyInstanceUID',*smoke.TARGETS]:raise ValueError('Written target order differs')
        rows=list(reader)
    if any(len(r)!=13 for r in rows) or [r[0] for r in rows]!=ids:raise ValueError('Written ordered coverage differs')
    own.probabilities([[float(v) for v in r[1:]] for r in rows],len(ids))


def require_bundle(contract,reader_code,model_code):
    """Validate fullfive evidence and its canonical binding before loading models."""
    if contract.get('version')!=1 or contract.get('max_seconds')!=MAX_SECONDS:
        raise ValueError('CoAtNet224 nine-hour bundle required')
    original=contract['rehearsal_contract']
    require_qualification(original,contract['rehearsal_report'])
    if (smoke.digest(smoke.canonical(contract['rehearsal_report']))!=contract.get('rehearsal_report_canonical_sha256') or
        smoke.digest(reader_code.encode())!=original['reader_sha256'] or
        smoke.digest(model_code.encode())!=original['model_definitions_sha256']):
        raise ValueError('Qualified evidence/definitions differ')
    return original


def run(inputs,out,contract,reader_code,model_code,*,started):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','phase':'setup','study_count':0,'completed_studies':0,'labels_read':False,
            'settings':own.SETTINGS,'max_seconds':MAX_SECONDS,'contract_sha256':smoke.digest(smoke.canonical(contract)),
            'models':[],'reader_counts':{}}
    def save():
        report['elapsed_seconds']=time.monotonic()-started
        report['peak_host_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        p=out/'operational-receipt.json.tmp';p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');p.replace(out/'operational-receipt.json')
    save()
    try:
        with time_limit(MAX_SECONDS-(time.monotonic()-started),'Actual nine-hour inference'):
            import numpy as np,torch
            require_cuda(torch);original=require_bundle(contract,reader_code,model_code)
            import subprocess
            codecroot=dataset_root(inputs,'goodpjw2008/rsna-knee-2-5d-convnext-reader',list(original['codecs']));wheels=[]
            for name,sha in original['codecs'].items():smoke.require_file(codecroot/name,sha);wheels.append(str(codecroot/name))
            subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*wheels],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            versions={p:importlib.metadata.version(p) for p in ('torch','timm','numpy','pydicom','pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg')}
            if any(versions[p]!=v for p,v in {'torch':'2.10.0+cu128','timm':'1.0.26','numpy':'2.0.2'}.items()):raise ValueError('Pinned dependencies differ')
            warnings.filterwarnings('ignore',module=r'pydicom(?:\..*)?')
            torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
            report['environment']={'python':sys.version.split()[0],'packages':versions,'gpu':torch.cuda.get_device_name(0),
                                   'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}
            protected,meta=raptor_input.transform_exception_handlers(reader_code)
            if meta['sha256']!=original['protected_reader_sha256']:raise ValueError('Fatal resource policy differs')
            public=parity.reader_module(reader_code);public._raptor_protected_source=protected;public._raptor_propagate_fatal=raptor_input.propagate_fatal
            exec(compile(protected,'<fatal-protected-reader>','exec'),public.__dict__);ns=own.namespace(model_code)
            inputs=Path(inputs);competition=exact_root([inputs/'competitions/rsna-knee-abnormality-detection',inputs/'rsna-knee-abnormality-detection'],['test_series.csv','test.csv','sample_submission.csv'])
            columns,ids,records=read_inputs(competition);report['study_count']=len(ids);models=[];report['phase']='strict_load'
            for m in original['members']:
                assets=notebook_root(inputs,m['kernel'],'trackA',[m['run_file'],m['checkpoint_file']])
                # Hash provenance bytes only; never parse training/gold fields.
                smoke.require_file(assets/m['run_file'],m['run_sha256']);smoke.require_file(assets/m['checkpoint_file'],m['checkpoint_sha256'])
                state=torch.load(assets/m['checkpoint_file'],map_location='cpu',weights_only=True)
                own.verify_checkpoint(state,m)
                bb=ns['build_backbone'](own.ARCH,pretrained=False);model=ns['RaptorClassifier'](bb,F_dim=bb.num_features)
                model.load_state_dict(state['model'],strict=True);models.append(model.to('cuda').eval());del state,bb,model
                report['models'].append({'fold':m['fold'],'strict_load':True,'checkpoint_sha256':m['checkpoint_sha256']});save()
            report['phase']='inference';torch.cuda.reset_peak_memory_stats();temp=out/'submission.csv.tmp'
            with temp.open('w',newline='') as f,torch.inference_mode():
                writer=csv.writer(f);writer.writerow(columns)
                for uid in ids:
                    volume,mask,counts=own.raw_input(public,uid,records,competition/'test_series');x=own.windows(ns,volume,mask);members=[]
                    for model in models:
                        with torch.autocast('cuda',dtype=torch.float16):p=torch.sigmoid(model(x[None].to('cuda')).float())
                        members.append(p.cpu().numpy())
                    final=own.mean_probabilities(members,1)[0];writer.writerow([uid,*[format(float(v),'.9g') for v in final]])
                    for key,value in counts.items():
                        if isinstance(value,(int,float)):report['reader_counts'][key]=report['reader_counts'].get(key,0)+value
                    report['completed_studies']+=1;del volume,mask,x,p,members,final;save()
            torch.cuda.synchronize();validate_csv(temp,ids);report['submission_sha256']=smoke.file_sha(temp)
            if smoke.digest(smoke.canonical(ids))==contract['visible_ids_sha256']:
                if report['submission_sha256']!=contract['rehearsal_report']['visible_csv_sha256']:raise ValueError('Exact five-fold visible CSV parity differs')
                report['visible_csv_exact']=True
            else:report['visible_csv_exact']=None
            temp.replace(out/'submission.csv');report['cuda_peak_bytes']=torch.cuda.max_memory_allocated()
            report['output_disk_bytes']=(out/'submission.csv').stat().st_size;report['status']='PASS';report['phase']='complete';save()
    except BaseException as error:
        report['status']='ERROR';report['failure']={'phase':report['phase'],'type':type(error).__name__};save()
        raise RuntimeError('CoAtNet224 five-fold inference failed; sanitized receipt saved') from None
    return report
