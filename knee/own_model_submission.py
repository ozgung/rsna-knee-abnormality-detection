"""Image-only streaming inference for a qualified own five-fold family."""
import csv,importlib.metadata,json,re,resource,sys,time,warnings
from pathlib import Path
from knee import own_model_rehearsal as own,public_training_smoke as smoke,corpus44_parity as parity,raptor_input
from knee.fresh_reader_setup import dataset_root,notebook_root,exact_root,time_limit
from knee.reference_blend import read_test_contract
MAX_SECONDS=9*3600
SERIES_COLUMNS=['StudyInstanceUID','SeriesInstanceUID','Fluid_Sensitive','Fat_Suppression','Anatomical_Plane']
KERNELS=['ozgung/knee-tracka-convnext-fold01-20261009','ozgung/knee-tracka-convnext-fold23-20261009','ozgung/knee-tracka-convnext-fold4-20261009']


ROUND1_LABEL_SHA='b260412732573fa9601907501b753f73d7d4725c3d0942cb463428510c1d48f4'
LABEL_PINS={0:('labels_v1_blend.parquet',smoke.LABEL_SHA),1:('labels_bootstrap_v1.parquet',ROUND1_LABEL_SHA)}
ROUND_KERNELS={0:KERNELS,1:['ozgung/knee-tracka-convnext-r1-fold01-20261009',
                          'ozgung/knee-tracka-convnext-r1-fold23-20261009','ozgung/knee-tracka-convnext-r1-fold4-20261009']}
SELECTION='masked macro AUC of the FULL held-out fold vs report labels (train_labels_v1 y_/m_), not gold'
FOLDS_SHA='12bebc897a549d660cf5e6ab597c9a8d8f84dc887b4abd4c99b67fb4a25cb3ae'
EXCLUDED_SHA='44834d7638eb60142a668a7597bd3093fe569e13b6f77e60a9c185a4ac3206f1'


def training_policy(contract):
    """Absent metadata is legacy round0; round1 always requires its exact label pin."""
    number=contract.get('training_round',0)
    if type(number) is not int or number not in ROUND_KERNELS:raise ValueError('Unsupported training round')
    filename,sha=LABEL_PINS[number]
    explicit=any(k in contract for k in ('training_round','labels_file','labels_sha256'))
    if explicit and (contract.get('labels_file')!=filename or contract.get('labels_sha256')!=sha):
        raise ValueError('Training round label pin differs')
    return ROUND_KERNELS[number]


def verify_training_run(record,member,training_round):
    """Build-time only: preserve gold-free selection and the exact fitted inputs."""
    policy={'training_round':training_round,'labels_file':LABEL_PINS.get(training_round,(None,None))[0],
            'labels_sha256':LABEL_PINS.get(training_round,(None,None))[1]}
    kernels=training_policy(policy);filename,sha=LABEL_PINS[training_round]
    own.verify_run(record,member)
    if (member['kernel']!=kernels[0 if member['fold']<2 else (1 if member['fold']<4 else 2)] or
        record.get('labels_file')!=filename or record.get('labels_sha256')!=sha or record.get('selection')!=SELECTION or
        record.get('epochs_requested')!=10 or record.get('stopped_by_time_budget') is not False or
        record.get('folds_sha256')!=FOLDS_SHA or record.get('excluded_ids_sha256')!=EXCLUDED_SHA):
        raise ValueError('Fitted training round/labels/selection/folds differ')


def require_qualification(contract,report):
    if (contract.get('complete_family') is not True or contract.get('settings')!=own.SETTINGS or
        [m['fold'] for m in contract.get('members',[])]!=list(range(5)) or
        contract.get('model_source_sha256')!=smoke.SOURCE_SHA or
        contract.get('reader_sha256')!='e331776e19800ef584c653cb9bb90698ff708027b0f9f483ce7afb18338a2f35'):
        raise ValueError('Complete frozen five-fold family required')
    kernels=training_policy(contract)
    own.require_result(report,contract)
    if any(len(r['slots'])!=5 for r in report['training_parity']):raise ValueError('All25 slot parity checks required')
    for m in contract['members']:
        if (m['kernel']!=kernels[0 if m['fold']<2 else (1 if m['fold']<4 else 2)] or
            m['run_file']!=f"run_fold{m['fold']}.json" or Path(m['checkpoint_file']).name!=m['checkpoint_file'] or
            len(m['selected_epochs'])!=3 or len(set(m['selected_epochs']))!=3 or
            any(type(e) is not int or e not in range(10) for e in m['selected_epochs']) or
            any(not re.fullmatch('[0-9a-f]{64}',m[k]) for k in ('run_sha256','checkpoint_sha256'))):
            raise ValueError('Frozen member provenance differs')


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
    with Path(path).open(newline='') as f:
        reader=csv.reader(f)
        if next(reader,None)!=['StudyInstanceUID',*smoke.TARGETS]:raise ValueError('Written target order differs')
        rows=list(reader)
    if any(len(r)!=13 for r in rows) or [r[0] for r in rows]!=ids:raise ValueError('Written ordered coverage differs')
    own.probabilities([[float(v) for v in r[1:]] for r in rows],len(ids))


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
            require_cuda(torch);original=contract['rehearsal_contract'];require_qualification(original,contract['rehearsal_report'])
            if (smoke.digest(smoke.canonical(contract['rehearsal_report']))!=contract['rehearsal_report_canonical_sha256'] or
                smoke.digest(reader_code.encode())!=original['reader_sha256'] or
                smoke.digest(model_code.encode())!=original['model_definitions_sha256']):raise ValueError('Qualified evidence/definitions differ')
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
                if state.get('arch')!=own.ARCH or state.get('res')!=256 or state.get('lab')!=smoke.TARGETS or state.get('swa_over')!=m['selected_epochs']:raise ValueError('Checkpoint metadata differs')
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
        raise RuntimeError('Own five-fold inference failed; sanitized receipt saved') from None
    return report
