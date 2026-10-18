"""VM-only minimal pinned supervision staging; no images/model/dependency install.

CLI: --input-root DIR --package DIR --manifest-sha256 REVIEWED_SHA.
Uses existing Kaggle credentials; never accepts terms/licenses or downloads a
whole dataset/competition. Failed payloads stay in private attempt directories.
"""
import argparse
import contextlib
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
import uuid
import zipfile

CHUNK=1024*1024
REMOTE_LIMIT=32*1024*1024
LOCAL_LIMIT=4*1024*1024
RESERVE=32*1024*1024
TOTAL_LIMIT=3*REMOTE_LIMIT+2*LOCAL_LIMIT
DISK_REQUIRED=TOTAL_LIMIT+3*REMOTE_LIMIT+REMOTE_LIMIT+RESERVE #264MiB worst-case new attempt.
DATASET='nartaa/rsna-knee-hpo-assets'
COMPETITION='rsna-knee-abnormality-detection'
REMOTE_PINS={
 'labels_v1_blend.parquet':('dataset',DATASET,'d2cdb8a37e0c4c491758d17c96e3bd4081f06b257c7849608df88590368e94e2'),
 'train_labels_v1.parquet':('dataset',DATASET,'a134cfce2f751f90f4219d791a563ee677b6223ac28f32e456bb3759c1d3fba9'),
 'train.csv':('competition',COMPETITION,'8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33')}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(CHUNK),b''):h.update(block)
    return h.hexdigest()


def safe_file(path,limit):
    if Path(path).is_symlink() or not Path(path).is_file() or Path(path).stat().st_size>limit:raise ValueError('Unsafe or oversized input artifact')


def bounded_copy(source,target,limit):
    count=0
    with open(target,'xb') as out:
        os.chmod(target,0o600)
        while True:
            block=source.read(CHUNK)
            if not block:break
            count+=len(block)
            if count>limit:raise ValueError('Decoded input exceeds size bound')
            out.write(block)
        out.flush();os.fsync(out.fileno())
    return count


def decode(payload,target,name):
    safe_file(payload,REMOTE_LIMIT)
    with open(payload,'rb') as f:magic=f.read(4)
    if magic[:2]==b'PK':
        with zipfile.ZipFile(payload) as archive:
            entries=archive.infolist()
            if len(entries)!=1:raise ValueError('ZIP must contain exactly the requested file')
            item=entries[0]
            if item.filename!=name or item.is_dir() or item.flag_bits&1 or stat.S_ISLNK(item.external_attr>>16) or item.file_size>REMOTE_LIMIT or item.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED):raise ValueError('Unsafe ZIP member')
            with archive.open(item) as source:return bounded_copy(source,target,REMOTE_LIMIT)
    if magic[:2]==b'\x1f\x8b':
        with gzip.open(payload,'rb') as source:return bounded_copy(source,target,REMOTE_LIMIT)
    with open(payload,'rb') as source:return bounded_copy(source,target,REMOTE_LIMIT)


class KaggleTransport:
    """Existing configured auth, exact single-file SDK requests, bounded streaming."""
    def __init__(self):self.api=None
    def fetch(self,kind,source,name,target):
        from kaggle.api.kaggle_api_extended import KaggleApi,ApiDownloadDatasetRequest,ApiDownloadDataFileRequest
        if self.api is None:self.api=KaggleApi();self.api.authenticate()
        with self.api.build_kaggle_client() as client:
            if kind=='dataset':
                owner,slug=source.split('/');request=ApiDownloadDatasetRequest()
                request.owner_slug=owner;request.dataset_slug=slug;request.file_name=name
                response=client.datasets.dataset_api_client.download_dataset(request)
            elif kind=='competition':
                request=ApiDownloadDataFileRequest();request.competition_name=source;request.file_name=name
                response=client.competitions.competition_api_client.download_data_file(request)
            else:raise ValueError('Unknown input source')
            try:
                length=response.headers.get('Content-Length')
                if length and int(length)>REMOTE_LIMIT:raise ValueError('Remote input exceeds size bound')
                count=0
                with open(target,'xb') as out:
                    os.chmod(target,0o600)
                    blocks=response.iter_content(CHUNK) if hasattr(response,'iter_content') else iter(lambda:response.read(CHUNK),b'')
                    for block in blocks:
                        count+=len(block)
                        if count>REMOTE_LIMIT:raise ValueError('Remote input exceeds size bound')
                        out.write(block)
                    out.flush();os.fsync(out.fileno())
            finally:response.close()


def publish(candidate,target):
    # Same filesystem: hard-link publication is atomic and never replaces a racing destination.
    os.link(candidate,target,follow_symlinks=False);Path(candidate).unlink()


def receipt_file(path,value):
    data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    if path.exists() or path.is_symlink():
        safe_file(path,LOCAL_LIMIT)
        if path.read_bytes()!=data:raise ValueError('Different existing receipt preserved')
        return
    temp=path.with_name(path.name+'.tmp')
    with open(temp,'xb') as f:os.chmod(temp,0o600);f.write(data);f.flush();os.fsync(f.fileno())
    publish(temp,path)


def _prepare(input_root,package,manifest_sha256,*,transport=None,remote_pins=REMOTE_PINS,disk_usage=shutil.disk_usage):
    root=Path(input_root);package=Path(package)
    if not root.is_absolute() or not package.is_absolute() or root.is_symlink() or package.is_symlink() or root==package or root in package.parents or package in root.parents:raise ValueError('Separate safe absolute input/package paths required')
    mp=package/'manifest.json';safe_file(mp,LOCAL_LIMIT)
    if not re.fullmatch('[a-f0-9]{64}',manifest_sha256) or sha(mp)!=manifest_sha256:raise ValueError('Reviewed package manifest differs')
    m=json.loads(mp.read_bytes())
    local={'train_knee.py':('train_knee_original.py',m['original_sha256']),'folds_v0.json':('folds_v0.json',m['folds_sha256'])}
    pins={name:value[2] for name,value in remote_pins.items()};pins.update({name:v[1] for name,v in local.items()})
    if any(Path(name).name!=name or name in ('.','..') for name in pins):raise ValueError('Unsafe requested filename')
    for name,(src,digest) in local.items():
        safe_file(package/src,LOCAL_LIMIT)
        if sha(package/src)!=digest:raise ValueError('Reviewed package input differs')
    root.mkdir(parents=True,exist_ok=True)
    if disk_usage(root).free<DISK_REQUIRED:raise OSError('Requires 264MiB free for bounded input staging')
    attempts=root/'.source96-input-attempts'
    if attempts.is_symlink():raise ValueError('Unsafe evidence directory')
    attempts.mkdir(exist_ok=True);os.chmod(attempts,0o700)
    attempt=attempts/uuid.uuid4().hex;attempt.mkdir(mode=0o700);started=time.monotonic()
    state={'status':'FAIL','version':'source96-inputs-v1','package_manifest_sha256':manifest_sha256}
    try:
        for name,digest in pins.items():
            target=root/name
            if target.exists() or target.is_symlink():
                safe_file(target,REMOTE_LIMIT if name in remote_pins else LOCAL_LIMIT)
                if sha(target)!=digest:raise ValueError('Different existing input preserved')
        transport=transport or KaggleTransport();total=0
        for i,(name,(kind,source,digest)) in enumerate(remote_pins.items()):
            target=root/name
            if target.exists():total+=target.stat().st_size;continue
            private=attempt/f'remote-{i}';private.mkdir(mode=0o700);payload=private/'payload.bin';candidate=private/'candidate.bin'
            # SDK diagnostics may contain signed URLs: retain privately, never console/receipt.
            with open(private/'sdk-diagnostics.log','x') as logs:
                os.chmod(private/'sdk-diagnostics.log',0o600)
                with contextlib.redirect_stdout(logs),contextlib.redirect_stderr(logs):transport.fetch(kind,source,name,payload)
            decoded=decode(payload,candidate,name)
            if sha(candidate)!=digest:raise ValueError('Downloaded input hash differs')
            total+=decoded
            if total>TOTAL_LIMIT:raise ValueError('Total input bound exceeded')
            if target.exists() or target.is_symlink():raise ValueError('Input appeared during staging; preserved')
            publish(candidate,target)
        for i,(name,(src,digest)) in enumerate(local.items()):
            target=root/name
            if target.exists():total+=target.stat().st_size;continue
            candidate=attempt/f'local-{i}.bin'
            with open(package/src,'rb') as source:count=bounded_copy(source,candidate,LOCAL_LIMIT)
            if sha(candidate)!=digest:raise ValueError('Package input changed while copying')
            total+=count
            if total>TOTAL_LIMIT or target.exists() or target.is_symlink():raise ValueError('Total bound or destination identity differs')
            publish(candidate,target)
        for name,digest in pins.items():
            safe_file(root/name,REMOTE_LIMIT if name in remote_pins else LOCAL_LIMIT)
            if sha(root/name)!=digest:raise ValueError('Final input identity differs')
        state={'status':'PASS','version':'source96-inputs-v1','package_manifest_sha256':manifest_sha256,
            'files':{name:{'sha256':digest,'bytes':(root/name).stat().st_size} for name,digest in pins.items()},
            'total_bytes':total,'total_limit_bytes':TOTAL_LIMIT,'training':False,'images':False}
        receipt_file(root/'source96-inputs-receipt.json',state)
        return state
    except Exception as error:
        state.update(status='FAIL',failure=type(error).__name__);raise
    finally:
        receipt_file(attempt/'attempt-receipt.json',{**state,'runtime_seconds':time.monotonic()-started})


def prepare(input_root,package,manifest_sha256):return _prepare(input_root,package,manifest_sha256)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--package',type=Path,required=True);p.add_argument('--manifest-sha256',required=True)
    try:
        r=prepare(**vars(p.parse_args()));print(json.dumps({'status':r['status'],'files':len(r['files']),'total_bytes':r['total_bytes']}))
    except Exception as error:
        print(json.dumps({'status':'FAIL','failure':type(error).__name__,'evidence':'Private attempt directory; no different existing input overwritten'}));raise SystemExit(1)
