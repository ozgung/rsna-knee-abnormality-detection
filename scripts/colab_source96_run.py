"""Single VM source96 controller. CLI: config.json --sha256 REVIEWED_CONFIG_SHA.

Config v1: name, vm_session_id, package_root/package_manifest_sha256,
workdir, input_root, stage_root, manifest_sha256/content_sha256/stage_receipt_sha256,
initializer_descriptor/initializer_descriptor_sha256, persister/persister_sha256,
persistence {dataset, private:true, staging, status_file}, controller_lock,
workers (default2), worker_choice_name (defaultnull), resume (defaultfalse),
resume_sha256 (required for resume), actual_platform_deadline (optional UTCseconds).
No installation/staging/initialization/benchmark happens here. Existing private
Kaggle dataset and credential configuration must already be qualified by caller.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

PERSISTER_SHA='8923bd872500b87f85be78aefd9fc799a6f88942e94503045b5e01ea66c118df'
VERSION='colab-source96-run-v1'


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def checked(path,digest):
    p=Path(path)
    if p.is_symlink() or not re.fullmatch('[0-9a-f]{64}',str(digest)) or sha(p)!=digest:raise ValueError('Artifact pin differs')
    return json.loads(p.read_bytes())


def atomic_json(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.tmp')
    if p.is_symlink() or tmp.is_symlink():raise ValueError('Unsafe receipt path')
    with open(tmp,'w') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def private_diagnostic(path,text):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW,0o600)
    os.fchmod(fd,0o600)
    with os.fdopen(fd,'a') as f:f.write(text);f.flush()


def validate(c):
    allowed={'version','name','vm_session_id','package_root','package_manifest_sha256','workdir','input_root','stage_root','manifest_sha256','content_sha256','stage_receipt_sha256','initializer_descriptor','initializer_descriptor_sha256','persister','persister_sha256','persistence','controller_lock','workers','worker_choice_name','resume','resume_sha256','actual_platform_deadline','fold'}
    if set(c)-allowed:raise ValueError('Unknown runner config parameter')
    if c.get('version')!=VERSION or not re.fullmatch('[a-z0-9][a-z0-9-]{2,63}',c.get('name','')) or not re.fullmatch('[a-zA-Z0-9_-]{3,80}',c.get('vm_session_id','')):raise ValueError('Named immutable VM config required')
    paths=('package_root','workdir','input_root','stage_root','initializer_descriptor','persister','controller_lock')
    for name in paths:
        p=Path(c[name])
        if not p.is_absolute() or p.is_symlink():raise ValueError('Explicit safe absolute VM paths required')
    work=Path(c['workdir']);package=Path(c['package_root']);persist=c['persistence']
    if persist.get('private') is not True or not re.fullmatch('[a-z0-9_-]+/[a-z0-9-]{3,50}',persist.get('dataset','')):raise ValueError('Private persistence dataset required')
    for name in ('staging','status_file'):
        p=Path(persist[name])
        if not p.is_absolute() or p.is_symlink():raise ValueError('Explicit private persistence paths required')
    if work==package or work in package.parents or package in work.parents or Path(persist['staging'])==work or work in Path(persist['staging']).parents:raise ValueError('Package/work/persistence staging must be separate')
    m=checked(package/'manifest.json',c['package_manifest_sha256'])
    for file,key in [('train_knee_original.py','original_sha256'),('train_knee_patched.py','patched_sha256'),('tracka_entry.py','entry_sha256'),('train_knee.patch','patch_sha256'),('folds_v0.json','folds_sha256'),('excluded_ids.json','excluded_ids_sha256'),('source96-recipe.json','recipe_sha256')]:
        if (package/file).is_symlink() or sha(package/file)!=m[key]:raise ValueError('Package artifact differs')
    for file,digest in m['sources'].items():
        p=Path(file)
        if p.is_absolute() or '..' in p.parts or (package/p).is_symlink() or sha(package/p)!=digest:raise ValueError('Runtime module pin differs')
    policy=json.loads((package/'source96-recipe.json').read_bytes())
    if c.get('workers',2)!=policy['workers'] or c.get('worker_choice_name')!=policy.get('worker_choice_name') or (policy['workers']!=2 and not policy.get('worker_choice_name')):raise ValueError('Frozen worker choice differs')
    if policy['frozen']['epochs']!=10 or policy['frozen']['k']!=8 or policy['frozen']['k_eval']!=16 or policy['frozen']['time_budget_h']!=0:raise ValueError('Frozen training recipe differs')
    if c['persister_sha256']!=PERSISTER_SHA or sha(c['persister'])!=PERSISTER_SHA:raise ValueError('Authoritative persister source differs')
    stage=Path(c['stage_root']);r=checked(stage/'staging-receipt.json',c['stage_receipt_sha256'])
    if r.get('status')!='PASS' or r.get('manifest_sha256')!=c['manifest_sha256'] or r.get('content_sha256')!=c['content_sha256']:raise ValueError('Verified staged cache required')
    d=checked(c['initializer_descriptor'],c['initializer_descriptor_sha256'])
    if Path(d['weights_file']).is_symlink() or sha(d['weights_file'])!=d['weights_sha256']:raise ValueError('Initializer weights differ')
    fold=c.get('fold',0)
    if type(fold) is not int or fold not in (0,1):raise ValueError('Only reviewed folds0/1 supported')
    split=policy.get('experiment_split')
    selected=0 if split is None else split.get('fold')
    if fold!=selected or policy['frozen']['fold']!=fold or (split is None and policy.get('version')!=1):raise ValueError('Package/config fold differs; legacy policy is fold0 only')
    if split is not None:
        expected=hashlib.sha256(json.dumps(split,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if policy.get('version')!=2 or policy.get('experiment_split_sha256')!=expected or (split.get('eligible'),split.get('training'),split.get('development'))!=(4345,3479,866) or fold!=1:raise ValueError('Fold1 experiment split differs')
    if fold==1 and any('fold1' not in value for value in (c['name'],work.name,Path(persist['staging']).name,Path(persist['status_file']).name,Path(c['controller_lock']).name,persist['dataset'].split('/')[-1])):raise ValueError('Fold1 requires isolated named work/backup/status/lock paths')
    if work.exists() and any(re.search(r'fold([01])',p.name) and int(re.search(r'fold([01])',p.name).group(1))!=fold for p in work.glob('raptor_*')):raise ValueError('Cross-fold work artifacts forbidden')
    resume=work/f'raptor_resume_fold{fold}.pt'
    if c.get('resume',False):
        if type(c['resume']) is not bool or not resume.is_file() or resume.is_symlink() or sha(resume)!=c.get('resume_sha256'):raise ValueError('Explicit pinned resume required')
    elif work.exists() and any(work.glob('raptor_*')):raise ValueError('Existing training artifacts require explicit resume')
    return m,policy,r,d


def environment(c):
    tree=ast.parse((Path(c['package_root'])/'tracka_entry.py').read_text())
    defaults=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PARAM_DEFAULTS' for t in n.targets)))
    env=dict(os.environ);env.update({k:str(v) for k,v in defaults.items()})
    env.update(FOLD_A=str(c.get('fold',0)),WORKDIR=c['workdir'],ASSET_DIR=c['package_root'],RAPTOR_INPUT=c['input_root'],RESUME='1' if c.get('resume') else '0',
        RAPTOR_PRELOAD='0',PYTHONPATH=c['package_root'],PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        SOURCE96_STAGE_ROOT=c['stage_root'],SOURCE96_MANIFEST_SHA256=c['manifest_sha256'],SOURCE96_CONTENT_SHA256=c['content_sha256'],
        SOURCE96_STAGE_RECEIPT_SHA256=c['stage_receipt_sha256'],SOURCE96_INITIALIZER_DESCRIPTOR=c['initializer_descriptor'],
        SOURCE96_INITIALIZER_DESCRIPTOR_SHA256=c['initializer_descriptor_sha256'])
    env.pop('SOURCE96_RESUME_SHA256',None);env.pop('SOURCE96_ACTUAL_PLATFORM_DEADLINE_UTC_SECONDS',None);env.pop('ORIGINAL_SCRIPT',None)
    if c.get('resume'):env['SOURCE96_RESUME_SHA256']=c['resume_sha256']
    if c.get('actual_platform_deadline') is not None:env['SOURCE96_ACTUAL_PLATFORM_DEADLINE_UTC_SECONDS']=str(float(c['actual_platform_deadline']))
    return env


def runtime_probe(c,env):
    code='''import json,platform,torch,timm,numpy
from knee import source96_training as s
s.strict_cuda(torch)
v,m,ids=s.staged_corpus()
b=timm.create_model(s.ARCH,pretrained=False,num_classes=0,in_chans=3,global_pool="avg")
i=s.generic_backbone(b)
print(json.dumps({"python":platform.python_version(),"torch":torch.__version__,"timm":timm.__version__,"numpy":numpy.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"threads":torch.get_num_threads(),"studies":len(ids),"initializer":i,"device":"cuda"}))
'''
    p=subprocess.run([sys.executable,'-c',code],cwd=c['package_root'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    private_diagnostic(Path(c['workdir'])/'source96-runtime-stdout.log',p.stdout)
    private_diagnostic(Path(c['workdir'])/'source96-runtime-stderr.log',p.stderr)
    if p.returncode:raise RuntimeError('CUDA/cache/initializer runtime preflight failed; private diagnostics retained in workdir')
    result=json.loads(p.stdout.strip().splitlines()[-1])
    if result.get('device')!='cuda' or result.get('threads')!=1 or result.get('studies')!=4345 or result.get('timm')!='1.0.26':raise ValueError('Runtime environment differs')
    return result


def private_dataset(dataset):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    with tempfile.TemporaryDirectory() as t:
        m=json.loads(Path(api.dataset_metadata(dataset,t)).read_bytes())
    info=m.get('info',m)
    identity=(str(info.get('ownerUser'))+'/'+str(info.get('datasetSlug'))) if 'info' in m else info.get('id')
    if identity!=dataset or info.get('isPrivate') is not True:raise ValueError('Existing backup dataset identity/privacy is unverified/public')
    return {'dataset':dataset,'private':True}


def sanitized(line):
    line=re.sub(r'^\[(?:train:fold[01]|persist [0-9:]{8})\]\s*','',line.strip())
    for pattern in (r'^ep\d+ loss [0-9.eE+-]+ \| OOF masked macro-AUC [0-9.eE+-]+',r'^\[resume\] (?:recoverable completed epoch|completed epoch checkpoint saved)\s*\d*',r'^FINAL PERSIST (?:VERIFIED epoch \d+ version \d+|UNVERIFIED)'):
        match=re.search(pattern,line.strip())
        if match:return match.group(0)
    return None


def stream(process,label,diagnostic_path):
    for line in process.stdout:
        private_diagnostic(diagnostic_path,line)
        safe=sanitized(line)
        if safe:print('['+label+'] '+safe,flush=True)


def primary_complete(work,fold=0,development=870,experiment_split=None,cache_content=None):
    record=json.loads((Path(work)/f'run_fold{fold}.json').read_bytes())
    meta=json.loads((Path(work)/f'raptor_ft_fold{fold}.json').read_bytes())
    if experiment_split is not None and (record.get('source96_experiment_split')!=experiment_split or record.get('cache_reference_partition_fold')!=0 or record.get('source96_cache_content_sha256')!=cache_content):raise ValueError('Export experiment/cache identity differs')
    if record.get('fold')!=fold or record.get('epochs_requested')!=10 or record.get('epochs_completed')!=10 or record.get('stopped_by_time_budget') or record.get('checkpoint_kind')!='swa_topk' or record.get('comparison_endpoint')!='matched K16 primary' or record.get('export',{}).get('k_eval')!=16 or record['export'].get('n_val')!=development or record['export'].get('n_gold')!=0 or len(record.get('selected_epochs',[]))!=3 or record.get('history')!=meta.get('hist') or meta.get('partial') or [h['ep'] for h in meta.get('hist',[])]!=list(range(10)):
        raise ValueError('Complete ten-epoch primary K16 export required')
    file=record['checkpoint_file']
    if file!=f'raptor_ft_fold{fold}_swa.pt' or (Path(work)/file).is_symlink() or sha(Path(work)/file)!=record['checkpoint_sha256']:raise ValueError('Primary checkpoint identity differs')
    if (Path(work)/f'oof_fold{fold}.csv').is_symlink() or not (Path(work)/f'oof_fold{fold}.csv').is_file():raise ValueError('Primary prediction artifact missing')
    return record


def final_verified(c,status):
    fold=c.get('fold',0);p=c['persistence'];resume=Path(c['workdir'])/f'raptor_resume_fold{fold}.pt';st=resume.stat()
    final=status.get('final') or {};version=status.get('verified_version')
    if status.get('dataset')!=p['dataset'] or status.get('fold')!=fold or final.get('state')!='verified' or status.get('verified_epoch')!=9 or final.get('epoch')!=9 or type(version) is not int or version<=0 or final.get('version')!=version or status.get('verified_signature')!=[st.st_size,st.st_mtime_ns]:raise ValueError('Latest final epoch9 backup is not verified')
    payload=json.loads((Path(p['staging'])/'payload_manifest.json').read_bytes())
    if payload.get('dataset')!=p['dataset'] or payload.get('fold')!=fold or payload.get('epoch')!=9 or payload.get('resume_sha256')!=sha(resume):raise ValueError('Final backup payload differs from latest resume')
    return {'epoch':9,'version':version,'resume_sha256':payload['resume_sha256']}


def run(c,*,popen=subprocess.Popen,probe=runtime_probe,privacy=private_dataset,sleep=time.sleep,read_status=None,install_signals=True,monotonic=time.monotonic):
    m,policy,stage,init=validate(c);env=environment(c);fold=c.get('fold',0);development=866 if fold==1 else 870
    lock=Path(c['controller_lock']);lock.parent.mkdir(parents=True,exist_ok=True)
    if lock.is_symlink():raise ValueError('Unsafe controller lock')
    with open(lock,'a+') as locked:
        try:fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another controller owns this VM lock') from None
        runtime=probe(c,env);privacy(c['persistence']['dataset'])
        work=Path(c['workdir']);work.mkdir(parents=True,exist_ok=True)
        fingerprint={'controller_sha256':sha(__file__),'config_sha256_canonical':hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'version':VERSION,'name':c['name'],'vm_session_id':c['vm_session_id'],'package_manifest_sha256':c['package_manifest_sha256'],
            'cache_manifest_sha256':c['manifest_sha256'],'cache_content_sha256':c['content_sha256'],'initializer_descriptor_sha256':c['initializer_descriptor_sha256'],
            'initializer_weights_sha256':init['weights_sha256'],'persister_sha256':PERSISTER_SHA,'policy':policy,'runtime':runtime,'workers':c.get('workers',2),'RAPTOR_PRELOAD':'0',
            'fold':fold,'experiment_split':policy.get('experiment_split'),'cache_reference_partition_fold':0,'resume':bool(c.get('resume')),'worker_rng_resume':'Main RNG only; persistent worker RNG/prefetch not restored', 'benchmark':'No optimizer throughput benchmark performed'}
        atomic_json(work/'source96-run-fingerprint.json',fingerprint)
        p=c['persistence'];pcmd=[sys.executable,'-u',c['persister'],'--workdir',c['workdir'],'--fold',str(fold),'--dataset',p['dataset'],'--staging',p['staging'],'--status-file',p['status_file'],'--keep-old-versions']
        persist=train=None;interrupted={'signal':None,'deadline':None};old={};threads=[];receipt={**fingerprint,'status':'FAIL'}
        def stop(signum,_frame):
            if interrupted['signal'] is not None:return
            interrupted['signal']=signum;interrupted['deadline']=monotonic()+60
            if train is not None and train.poll() is None:
                try:os.killpg(train.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        if install_signals:
            for signum in (signal.SIGTERM,signal.SIGINT):old[signum]=signal.signal(signum,stop)
        try:
            persist=popen(pcmd,env=env,cwd=c['package_root'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            threads.append(threading.Thread(target=stream,args=(persist,'backup',work/'source96-backup.log'),daemon=True));threads[-1].start()
            if interrupted['signal']:raise RuntimeError('Interrupted before training launch')
            train=popen([sys.executable,'-u',str(Path(c['package_root'])/'tracka_entry.py')],env=env,cwd=c['package_root'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            threads.append(threading.Thread(target=stream,args=(train,'training',work/'source96-training.log'),daemon=True));threads[-1].start()
            while train.poll() is None:
                if persist.poll() is not None:stop(signal.SIGTERM,None)
                if interrupted['deadline'] is not None and monotonic()>=interrupted['deadline']:break
                sleep(.5)
            receipt['training_returncode']=train.returncode
        except Exception as error:receipt['failure']=type(error).__name__
        finally:
            if train is not None and train.poll() is None:
                stop(signal.SIGTERM,None)
                try:train.wait(timeout=max(0,interrupted['deadline']-monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(train.pid,signal.SIGKILL);train.wait(timeout=60)
            if train is not None:receipt['training_returncode']=train.returncode
            if persist is not None:
                if persist.poll() is None:persist.send_signal(signal.SIGUSR1)
                try:persist.wait(timeout=900) # Existing persister final-timeout840 plus shutdown margin, no training cap.
                except subprocess.TimeoutExpired:
                    persist.send_signal(signal.SIGTERM)
                    try:persist.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(persist.pid,signal.SIGKILL);persist.wait(timeout=60)
                receipt['persister_returncode']=persist.returncode
            for t in threads:t.join(timeout=1)
            for signum,handler in old.items():signal.signal(signum,handler)
        try:
            if train is None or train.returncode!=0 or interrupted['signal'] or persist is None or persist.returncode!=0:raise ValueError('Training/controller/backup subprocess failed')
            primary=primary_complete(work,fold,development,policy.get('experiment_split'),c['content_sha256'])
            status=read_status(c) if read_status else json.loads(Path(p['status_file']).read_bytes())
            backup=final_verified(c,status);privacy(p['dataset'])
            receipt.update(status='PASS',checkpoint_sha256=primary['checkpoint_sha256'],backup=backup)
        except Exception as error:receipt.setdefault('failure',type(error).__name__)
        receipt['interrupted_signal']=interrupted['signal'];receipt['resume_preserved']=(work/f'raptor_resume_fold{fold}.pt').is_file()
        atomic_json(work/'source96-run-receipt.json',receipt)
        return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('config',type=Path);parser.add_argument('--sha256',required=True)
    a=parser.parse_args();config=checked(a.config,a.sha256);result=run(config)
    print(json.dumps({'status':result['status'],'failure':result.get('failure'),'resume_preserved':result['resume_preserved']}))
    raise SystemExit(0 if result['status']=='PASS' else 1)
