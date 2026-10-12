"""Private sharded96 cache: fixed source policy, atomic files, fail-closed resume."""
from collections import Counter
import csv
import importlib.metadata
import json
from pathlib import Path
import re
import resource
import shutil
import sys
import time
import zipfile

from knee import public_training_smoke as smoke, source_span_pilot as pilot

VERSION = 'own96-wide336-sharded-v1'
RAW_BYTES = 96 * 336 * 336 + 96
MAX_FILE_BYTES = RAW_BYTES + 65536  # NPY/ZIP/deflate overhead, including incompressible input.
DISK_RESERVE = 512 * 1024**2
CONFIG = {'reader_arm':'wide96','depth':96,'storage':336,'crop_mm':140.,
          'slots':[list(s) for s in pilot.SLOTS96],'span':[.02,.98],
          'cpu_threads':1,'windows':[8,16],'seed':pilot.SEED,'default_shards':4,
          'raw_bytes_per_study':RAW_BYTES,'max_npz_bytes_per_study':MAX_FILE_BYTES,
          'disk_reserve_bytes':DISK_RESERVE,'training':False,'models_loaded':0}


def safe_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',value) or value in ('.','..'):
        raise ValueError('Unsafe private identity')
    return value


def make_shards(ids, count=4):
    if type(count) is not int or not 1<=count<=64 or len(ids)!=len(set(ids)):
        raise ValueError('Invalid partition specification')
    for uid in ids: safe_id(uid)
    shards=[list(range(i,len(ids),count)) for i in range(count)]
    flattened=[n for shard in shards for n in shard]
    if len(flattened)!=len(ids) or len(set(flattened))!=len(ids) or sorted(flattened)!=list(range(len(ids))):
        raise ValueError('Partitions not exhaustive/disjoint')
    return shards


def membership(ids, official, folds, identity):
    """Read target presence solely to exclude gold; never report text or magnitudes."""
    ids=list(map(str,ids))
    if len(ids)!=4407 or len(set(ids))!=4407:raise ValueError('Historical corpus identity coverage differs')
    for uid in ids:safe_id(uid)
    records=identity['records'];known={r['uid']:r for r in records}
    if len(known)!=4407 or len(records)!=4407 or set(known)!=set(ids) or set(folds)!=set(ids):
        raise ValueError('Identity/fold coverage differs')
    gold, observed=set(),set()
    with Path(official).open(newline='') as stream:
        reader=csv.DictReader(stream)
        if not {'StudyInstanceUID',*smoke.TARGETS}<=set(reader.fieldnames or []):raise ValueError('Official schema differs')
        for row in reader:
            uid=safe_id(row['StudyInstanceUID'])
            if uid in observed:raise ValueError('Duplicate official identity')
            observed.add(uid)
            present=[row[t] is not None and bool(row[t].strip()) for t in smoke.TARGETS]
            if any(present) and not all(present):raise ValueError('Ambiguous gold membership')
            if all(present):gold.add(uid)
    excluded={r['uid'] for r in records if r['future_role']=='quarantined'}
    if observed!=set(ids) or len(gold)!=58 or len(excluded)!=4 or gold&excluded:
        raise ValueError('Official exclusions differ')
    if gold!={r['uid'] for r in records if r['future_role']=='gold'}:raise ValueError('Gold identity differs')
    eligible=[uid for uid in ids if uid not in gold|excluded]
    train=[uid for uid in eligible if folds[uid]!=0];dev=[uid for uid in eligible if folds[uid]==0]
    if any(type(folds[u]) is not int or not -1<=folds[u]<=4 for u in ids) or {u for u in ids if folds[u]==-1}!=gold:raise ValueError('Invalid folds')
    if (len(eligible),len(train),len(dev))!=(4345,3475,870):raise ValueError('Pinned own224 fold0 population differs')
    return eligible, {'historical':4407,'eligible':4345,'training':3475,'development':870,
                      'gold_excluded':58,'quarantine_excluded':4,
                      'ordered_eligible_sha256':smoke.digest(smoke.canonical(eligible)),
                      'ordered_training_sha256':smoke.digest(smoke.canonical(train)),
                      'ordered_development_sha256':smoke.digest(smoke.canonical(dev)),
                      'grouping':'Historical fixed fold0 membership; patient identities unavailable; cache makes no new split'}


def filename(ordinal):
    if type(ordinal) is not int or ordinal<0:raise ValueError('Invalid global ordinal')
    return f'{ordinal:06d}.npz'


def safe_file(root,name):
    if not isinstance(name,str) or not re.fullmatch(r'[0-9]{6}\.npz',name):raise ValueError('Unsafe cache filename')
    path=Path(root)/name
    if path.is_symlink():raise ValueError('Cache symlinks forbidden')
    return path


def atomic_json(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    if path.is_symlink() or tmp.is_symlink():raise ValueError('Manifest symlinks forbidden')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def array_hashes(volume,mask):
    pilot.validate_arrays(volume,mask,depth=96)
    if str(mask.dtype)!='uint8':raise ValueError('Mask dtype must be uint8')
    return {'volume_sha256':smoke.digest(volume.tobytes()),'mask_sha256':smoke.digest(mask.tobytes())}


def verify_file(path, expected=None):
    import numpy as np
    path=Path(path)
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=MAX_FILE_BYTES:
        raise ValueError('Invalid cache file')
    with zipfile.ZipFile(path) as zipped:
        infos=zipped.infolist()
        limits={'volume.npy':96*336*336+4096,'mask.npy':96+4096}
        if len(infos)!=2 or {i.filename for i in infos}!=set(limits) or any(i.file_size>limits[i.filename] for i in infos):
            raise ValueError('Cache archive members exceed bounded schema')
        for info in infos:
            with zipped.open(info) as stream:
                if np.lib.format.read_magic(stream)!=(1,0):raise ValueError('Unexpected NPY format')
                shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                expected_shape=(96,336,336) if info.filename=='volume.npy' else (96,)
                if shape!=expected_shape or fortran or dtype!=np.dtype('uint8') or info.file_size!=stream.tell()+int(np.prod(shape)):
                    raise ValueError('Cache NPY header differs from bounded schema')
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!={'volume','mask'}:raise ValueError('Cache archive schema differs')
        volume,mask=archive['volume'],archive['mask'];hashes=array_hashes(volume,mask)
    result={**hashes,'file_sha256':smoke.file_sha(path),'bytes':path.stat().st_size}
    if expected is not None and any(result[k]!=expected.get(k) for k in result):
        raise ValueError('Cache hashes/bytes differ')
    return result


def atomic_npz(path,volume,mask):
    import numpy as np
    path=Path(path);tmp=path.with_name(path.name+'.tmp');hashes=array_hashes(volume,mask)
    if path.is_symlink() or tmp.is_symlink():raise ValueError('Cache symlinks forbidden')
    # Never overwrite an unindexed completed file: re-decode and require raw equality.
    if path.exists():
        result=verify_file(path)
        if any(result[k]!=v for k,v in hashes.items()):raise ValueError('Unindexed final differs from current decode')
        return result
    try:
        with tmp.open('wb') as stream:np.savez_compressed(stream,volume=volume,mask=mask)
        result=verify_file(tmp)
        if any(result[k]!=v for k,v in hashes.items()):raise ValueError('Save/reload bytes differ')
        tmp.replace(path)
        return result
    except BaseException:
        # Own partial temporary artifact only; prior completed files remain untouched.
        if tmp.exists() and not tmp.is_symlink():tmp.unlink()
        raise


def index_rows(root,document,expected_contract,members,ids):
    if document.get('version')!=VERSION or document.get('contract_sha256')!=expected_contract:
        raise ValueError('Resume contract differs')
    entries=document.get('entries')
    if not isinstance(entries,list):raise ValueError('Private index missing')
    seen={}
    for entry in entries:
        n=entry.get('ordinal')
        if type(n) is not int or n not in members or n in seen or entry.get('StudyInstanceUID')!=ids[n] or entry.get('file')!=filename(n):
            raise ValueError('Resume membership/order differs')
        verify_file(safe_file(root,entry['file']),entry);seen[n]=entry
    return seen


def disk_required(remaining, import_bytes=0):
    if type(remaining) is not int or remaining<0 or type(import_bytes) is not int or import_bytes<0:
        raise ValueError('Invalid disk projection')
    return remaining*MAX_FILE_BYTES+import_bytes+2*MAX_FILE_BYTES+DISK_RESERVE


def require_disk(root,remaining,import_bytes=0):
    required=disk_required(remaining,import_bytes);free=shutil.disk_usage(root).free
    if free<required:raise OSError('Insufficient disk for worst-case outputs and temporary space')
    return {'free_bytes':free,'required_bytes':required,'remaining_studies':remaining,'import_bytes':import_bytes}


def stream_cache(out,contract,ids,records,raw,codes,sampler_code,*,resume_root=None,started=None):
    """Stream one study; private aligned index only, public aggregate receipt."""
    import torch,cv2
    from knee.fresh_reader_setup import time_limit
    started=time.monotonic() if started is None else started
    torch.set_num_threads(1);cv2.setNumThreads(1)
    out=Path(out)
    if out.is_symlink():raise ValueError('Output symlink forbidden')
    out.mkdir(parents=True,exist_ok=True)
    fingerprint=smoke.digest(smoke.canonical(contract))
    runfile=out/'run-contract.json';indexfile=out/'index-private.json'
    if runfile.exists():
        if json.loads(runfile.read_bytes())!=contract:raise ValueError('Output contract differs')
    elif any(out.iterdir()):raise ValueError('Unbound existing output directory')
    else:atomic_json(runfile,contract)
    # Preserve the preceding attempt's exact aggregate evidence before updating current state.
    previous=out/'receipt.json'
    if previous.exists():
        if previous.is_symlink():raise ValueError('Receipt symlink forbidden')
        archive=out/'receipt-history';archive.mkdir(exist_ok=True)
        if archive.is_symlink():raise ValueError('Receipt history symlink forbidden')
        atomic_json(archive/(smoke.file_sha(previous)+'.json'),json.loads(previous.read_bytes()))
    members=make_shards(ids,contract['shard_count'])[contract['shard_index']]
    report={'status':'RUNNING','phase':'resume','version':VERSION,'contract_sha256':fingerprint,
            'config':CONFIG,'shard_index':contract['shard_index'],'shard_count':contract['shard_count'],
            'membership':contract['membership'],'partition':contract['partitions'][contract['shard_index']],
            'code_revision':contract['code_revision'],'source_hashes':contract['sources'],
            'input_pins':contract['input_pins'],'private_outputs':True,'aggregate_contains_ids':False,
            'environment':{'python':sys.version.split()[0],'device':'cpu','torch_threads':torch.get_num_threads(),'opencv_threads':cv2.getNumThreads(),
                           'packages':{p:importlib.metadata.version(p) for p in ('torch','numpy','pydicom','opencv-python','pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg')}},
            'completed_studies':0,'newly_decoded_studies':0,'resumed_studies':0,'bytes':0,'decode_seconds':0.,'write_seconds':0.,'counters':{}}
    entries={};counters=Counter()
    def write():
        report.update(completed_studies=len(entries),bytes=sum(e['bytes'] for e in entries.values()),
                      elapsed_seconds=time.monotonic()-started,counters=dict(counters),
                      peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        report['minimum_free_disk_bytes']=min(report.get('minimum_free_disk_bytes',shutil.disk_usage(out).free),shutil.disk_usage(out).free)
        atomic_json(out/'receipt.json',report)
    def save_index():
        atomic_json(indexfile,{'version':VERSION,'contract_sha256':fingerprint,'entries':[entries[n] for n in members if n in entries]})
        report['private_index_sha256']=smoke.file_sha(indexfile)
    write()
    try:
        with time_limit(max(1,43200-(time.monotonic()-started)),'actual CPU platform12h limit'):
            if report['environment']!=contract['qualified_environment']:raise ValueError('Qualified CPU environment differs')
            if indexfile.exists():entries=index_rows(out,json.loads(indexfile.read_bytes()),fingerprint,members,ids)
            if resume_root is not None:
                if entries:raise ValueError('Cannot combine local and external resume')
                resume_root=Path(resume_root)
                prior=json.loads((resume_root/'run-contract.json').read_bytes())
                if prior!=contract:raise ValueError('External resume contract differs')
                imported=index_rows(resume_root,json.loads((resume_root/'index-private.json').read_bytes()),fingerprint,members,ids)
                report['disk_preflight']=require_disk(out,len(members)-len(imported),sum(e['bytes'] for e in imported.values()))
                for n,entry in imported.items():
                    src=safe_file(resume_root,entry['file']);dest=safe_file(out,entry['file']);tmp=dest.with_name(dest.name+'.tmp')
                    if tmp.is_symlink():raise ValueError('Resume temporary symlink forbidden')
                    shutil.copyfile(src,tmp);verify_file(tmp,entry);tmp.replace(dest);entries[n]=entry;save_index()
            if not indexfile.exists():save_index()
            report['resumed_studies']=len(entries)
            report['disk_preflight_after_resume']=require_disk(out,len(members)-len(entries))
            # Unknown final files cannot be silently omitted; legitimate crash-gap orphans are re-decoded below.
            allowed={filename(n) for n in members}
            if any(p.name.endswith('.npz') and p.name not in allowed for p in out.iterdir()):raise ValueError('Unexpected final cache file')
            sampler=pilot.sampler_class(sampler_code);public=pilot.arm_module(codes,'wide96')
            report['phase']='decode';write()
            for n in members:
                if n in entries:continue
                report['active_ordinal']=n;write();require_disk(out,1)
                begin=time.monotonic();volume,mask,audit=pilot.decode(public,ids[n],records,raw,pilot.SPANS['wide96'])
                report['decode_seconds']+=time.monotonic()-begin;counters.update(audit['reader_counts'])
                windows=pilot.window_checks(volume,mask,sampler,slots=public.SLOTS)
                begin=time.monotonic();name=filename(n);saved=atomic_npz(safe_file(out,name),volume,mask)
                report['write_seconds']+=time.monotonic()-begin
                entries[n]={'ordinal':n,'StudyInstanceUID':ids[n],'file':name,**saved,'windows':windows}
                save_index();report['newly_decoded_studies']+=1;del volume,mask;write()
            if len(entries)!=len(members):raise ValueError('Incomplete partition')
            report.pop('active_ordinal',None);report.update(status='PASS',phase='complete',private_index_sha256=smoke.file_sha(indexfile),
                final_free_disk_bytes=shutil.disk_usage(out).free);write()
    except BaseException as error:
        report.update(status='ERROR',failure={'phase':report['phase'],'type':type(error).__name__});write()
        raise RuntimeError('Source96 cache failed; completed private files and sanitized aggregate receipt preserved') from None
    return report


def _run(inputs,out,contract,codes,sampler_code,folds_sha256,*,started,resume_kernel=None,resume_dataset=None):
    import numpy as np
    from knee.fresh_reader_setup import dataset_root,notebook_root,exact_root
    if contract['version']!=VERSION or contract['config']!=CONFIG or folds_sha256!=contract['input_pins']['folds_sha256']:raise ValueError('Cache configuration differs')
    if type(contract['shard_index']) is not int or not 0<=contract['shard_index']<contract['shard_count']:raise ValueError('Invalid shard index')
    pilot.validate_recipes(codes['narrow'],codes['wide'])
    if {k:smoke.digest(v.encode()) for k,v in codes.items()}!=contract['reader_hashes'] or smoke.digest(sampler_code.encode())!=contract['sampler_sha256']:
        raise ValueError('Qualified source differs')
    identity_root=notebook_root(inputs,smoke.IDENTITY_KERNEL,'sealed_identity',['sealed-manifest-private.json'])
    identity_path=identity_root/'sealed-manifest-private.json';smoke.require_file(identity_path,smoke.MANIFEST_SHA)
    identity=json.loads(identity_path.read_bytes())
    competition=exact_root([Path(inputs)/'competitions/rsna-knee-abnormality-detection',Path(inputs)/'rsna-knee-abnormality-detection'],['train.csv','train_series.csv'])
    for name,sha in contract['input_pins']['competition'].items():smoke.require_file(competition/name,sha)
    historical=[]
    for ref,prefix in zip(smoke.CORPORA,['all','extra']):
        root=dataset_root(inputs,ref,[prefix+'_ids.npy']);name=prefix+'_ids.npy'
        smoke.require_file(root/name,smoke.PINS[name]);historical.extend(np.load(root/name,allow_pickle=False).astype(str).tolist())
    hpo=dataset_root(inputs,smoke.ASSETS,['folds_v0.json']);smoke.require_file(hpo/'folds_v0.json',folds_sha256)
    ids,summary=membership(historical,competition/'train.csv',json.loads((hpo/'folds_v0.json').read_bytes())['folds'],identity)
    if summary!=contract['membership']:raise ValueError('Eligible population differs')
    partitions=make_shards(ids,contract['shard_count'])
    for i,members in enumerate(partitions):
        expected={'studies':len(members),'ordinals_sha256':smoke.digest(smoke.canonical(members)),
                  'ordered_ids_sha256':smoke.digest(smoke.canonical([ids[n] for n in members]))}
        if expected!=contract['partitions'][i]:raise ValueError('Partition differs')
    selected=[ids[n] for n in partitions[contract['shard_index']]];records={u:[] for u in selected}
    with (competition/'train_series.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['StudyInstanceUID'] in records:records[row['StudyInstanceUID']].append(row)
    raw=pilot.parity.require_image_tree(competition,selected,records)
    resume=None
    if resume_kernel:resume=notebook_root(inputs,resume_kernel,'source96_cache',['run-contract.json','index-private.json'])
    if resume_dataset:resume=dataset_root(inputs,resume_dataset,['run-contract.json','index-private.json'])
    return stream_cache(out,contract,ids,records,raw,codes,sampler_code,resume_root=resume,started=started)


def run(inputs,out,contract,codes,sampler_code,folds_sha256,*,started,resume_kernel=None,resume_dataset=None):
    try:
        return _run(inputs,out,contract,codes,sampler_code,folds_sha256,started=started,
                    resume_kernel=resume_kernel,resume_dataset=resume_dataset)
    except BaseException as error:
        path=Path(out)
        if not path.is_symlink():
            path.mkdir(parents=True,exist_ok=True)
            if not (path/'receipt.json').exists():
                atomic_json(path/'receipt.json',{'version':VERSION,'status':'ERROR','phase':'setup',
                    'contract_sha256':smoke.digest(smoke.canonical(contract)),
                    'failure':{'type':type(error).__name__},'elapsed_seconds':time.monotonic()-started,
                    'aggregate_contains_ids':False,'private_outputs':True})
        raise RuntimeError('Source96 cache failed; completed private files and sanitized aggregate receipt preserved') from None
