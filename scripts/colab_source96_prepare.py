"""VM-only private96 staging; no model, training or local image relay.

Config: version, cache_root, validation (loader's private membership/folds), and
four sources: ref, version=1, notebook_path, notebook_sha256,
contract_file_sha256, index_sha256, receipt_sha256. Accepted notebook files are
small reviewed uploads. URLs exist only in transport memory, never receipts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urlparse, urljoin

CODE_ROOT=Path(os.environ.get('SOURCE96_CODE_ROOT','/content/trackA_pkg'))
if CODE_ROOT.is_dir():sys.path.insert(0,str(CODE_ROOT))
from knee import source96_cache as cache,source96_loader as loader,public_training_smoke as smoke

VERSION='colab-source96-stage-v1'
CHUNK_BYTES=1024*1024
METADATA_MAX_BYTES=16*1024*1024
DISK_RESERVE=5*1024**3
METADATA={'run-contract.json':'contract_file_sha256','index-private.json':'index_sha256','receipt.json':'receipt_sha256'}


def source_cells_sha256(source):
    notebook=json.loads(source) if isinstance(source,str) else source
    return loader.digest_document([{'cell_type':c['cell_type'],
        'source':''.join(c['source']) if isinstance(c['source'],list) else c['source']} for c in notebook['cells']])


def checked_url(url):
    p=urlparse(url)
    host=(p.hostname or '').lower()
    allowed=(host in ('storage.googleapis.com','kaggleusercontent.com','www.kaggle.com','api.kaggle.com') or
             host.endswith('.storage.googleapis.com') or host.endswith('.kaggleusercontent.com'))
    if p.scheme!='https' or not allowed or p.username or p.password or p.port not in (None,443):raise ValueError('Unsafe artifact URL')
    return url


def checked_name(name):
    if not isinstance(name,str) or not re.fullmatch(r'source96_cache/(?:run-contract\.json|index-private\.json|receipt\.json|[0-9]{6}\.npz)',name):
        raise ValueError('Unsafe or unrequested artifact path')
    return name.split('/')[-1]


class KaggleTransport:
    """Existing VM Kaggle credential configuration; nothing printed or persisted."""
    def __init__(self):
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.api=KaggleApi();self.api.authenticate()

    def snapshot(self,ref):
        from kaggle.api.kaggle_api_extended import ApiGetKernelRequest
        owner,slug=ref.split('/')
        with self.api.build_kaggle_client() as client:
            req=ApiGetKernelRequest();req.user_name=owner;req.kernel_slug=slug
            kernel=client.kernels.kernels_api_client.get_kernel(req)
        status=str(self.api.kernels_status(ref).status).split('.')[-1].upper()
        return {'version':kernel.metadata.current_version_number,'is_private':kernel.metadata.is_private,
                'cell_sources_sha256':source_cells_sha256(kernel.blob.source),'status':status}

    def list_files(self,ref):
        # version_label='1' failed on actual parent probe; no guessed version API.
        from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
        owner,slug=ref.split('/');token=None;seen=set();files=[]
        with self.api.build_kaggle_client() as client:
            while True:
                req=ApiListKernelSessionOutputRequest();req.user_name=owner;req.kernel_slug=slug
                self.api._set_paging(req,200,token)
                response=client.kernels.kernels_api_client.list_kernel_session_output(req)
                files.extend({'name':f.file_name,'url':f.url} for f in response.files or [])
                token=response.next_page_token
                if not token:break
                if token in seen:raise ValueError('Repeated output page token')
                seen.add(token)
        return files

    def iter_chunks(self,url):
        import requests
        url=checked_url(url)
        for _ in range(6):
            response=requests.get(url,stream=True,allow_redirects=False,timeout=(15,120))
            if response.status_code in (301,302,303,307,308):
                try:url=checked_url(urljoin(url,response.headers['Location']))
                finally:response.close()
                continue
            try:
                response.raise_for_status()
                yield from response.iter_content(chunk_size=CHUNK_BYTES)
                return
            finally:response.close()
        raise ValueError('Too many artifact redirects')


def _binding(descriptor):
    ref=descriptor['ref']
    if not re.fullmatch('ozgung/[a-z0-9_-]+',ref) or type(descriptor['version']) is not int or descriptor['version']!=1:
        raise ValueError('Only reviewed owned immutablev1 sources supported')
    for field in ('notebook_sha256','contract_file_sha256','index_sha256','receipt_sha256'):
        if not loader._hash(descriptor[field]):raise ValueError('Source pin missing')
    path=Path(descriptor['notebook_path'])
    if path.is_symlink() or smoke.file_sha(path)!=descriptor['notebook_sha256']:raise ValueError('Accepted notebook upload differs')
    return {k:descriptor[k] for k in ('ref','version','notebook_sha256','contract_file_sha256','index_sha256','receipt_sha256')}|{'cell_sources_sha256':source_cells_sha256(path.read_text())}


def _snapshot(transport,binding):
    actual=transport.snapshot(binding['ref'])
    if (type(actual.get('version')) is not int or actual.get('version')!=1 or actual.get('is_private') is not True or actual.get('status')!='COMPLETE' or
            actual.get('cell_sources_sha256')!=binding['cell_sources_sha256']):
        raise ValueError('Accepted source/version/private completion changed')


def _inventory(transport,binding):
    _snapshot(transport,binding);items=transport.list_files(binding['ref']);_snapshot(transport,binding)
    selected={}
    for item in items:
        name=item['name']
        # Harmless unrelated code/log outputs are ignored; traversal is never used.
        if not name.startswith('source96_cache/'):continue
        if name.endswith('.npz') or name.split('/')[-1] in METADATA:
            checked_name(name)
            if name in selected:raise ValueError('Duplicate artifact listing')
            selected[name]=checked_url(item['url'])
    return selected


def _download(transport,url,path,sha,*,expected_bytes=None,max_bytes):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    if path.is_symlink() or tmp.is_symlink():raise ValueError('Artifact symlink forbidden')
    if path.exists():
        if (expected_bytes is not None and path.stat().st_size!=expected_bytes) or smoke.file_sha(path)!=sha:
            raise ValueError('Completed staged artifact differs; never overwrite')
        return 0
    written=0;digest=hashlib.sha256()
    try:
        with tmp.open('wb') as stream:
            for chunk in transport.iter_chunks(checked_url(url)):
                if not isinstance(chunk,bytes) or len(chunk)>CHUNK_BYTES:raise ValueError('Transport chunk exceeds1MiB contract')
                if not chunk:continue
                written+=len(chunk)
                if written>max_bytes or (expected_bytes is not None and written>expected_bytes):raise ValueError('Artifact size exceeds expected bound')
                digest.update(chunk);stream.write(chunk)
            stream.flush();os.fsync(stream.fileno())
        if digest.hexdigest()!=sha or (expected_bytes is not None and written!=expected_bytes):raise ValueError('Downloaded artifact hash/bytes differ')
        tmp.replace(path)
        return written
    except BaseException:
        if tmp.exists() and not tmp.is_symlink():tmp.unlink()
        raise


def _metadata_plan(plans,validation):
    _,counts=loader._validate_profile(validation);common=None;seen=set();rows={};files=[]
    for plan in plans:
        root=plan['root'];contract=json.loads((root/'run-contract.json').read_bytes())
        index=json.loads((root/'index-private.json').read_bytes());receipt=json.loads((root/'receipt.json').read_bytes())
        n=contract['shard_index'];contract_sha=loader.digest_document(contract)
        if type(n) is not int or not 0<=n<4 or n in seen or contract['shard_count']!=4:raise ValueError('Shard identity differs')
        seen.add(n)
        if contract['version']!=cache.VERSION or contract['config']!=cache.CONFIG or contract['membership']!=validation['membership']:
            raise ValueError('Cache policy/population differs')
        current={k:contract[k] for k in loader.COMMON}
        if common is None:common=current
        elif current!=common:raise ValueError('Cache source/pins differ across shards')
        members=list(range(n,counts[0],4))
        if len(contract['partitions'])!=4 or contract['partitions'][n]['ordinals_sha256']!=loader.digest_document(members) or contract['partitions'][n]['studies']!=len(members):
            raise ValueError('Partition differs')
        if (receipt.get('status')!='PASS' or receipt.get('version')!=cache.VERSION or receipt.get('contract_sha256')!=contract_sha or
                receipt.get('private_index_sha256')!=plan['binding']['index_sha256'] or receipt.get('completed_studies')!=len(members) or
                receipt.get('shard_index')!=n or receipt.get('shard_count')!=4 or index.get('version')!=cache.VERSION or index.get('contract_sha256')!=contract_sha):
            raise ValueError('Cache metadata not fully qualified')
        entries=index['entries']
        if [e['ordinal'] for e in entries]!=members:raise ValueError('Index partition incomplete/reordered')
        ids=[]
        for row in entries:
            ordinal=row['ordinal'];uid=cache.safe_id(row['StudyInstanceUID']);name=cache.filename(ordinal)
            if row['file']!=name or type(row['bytes']) is not int or not 0<row['bytes']<=cache.MAX_FILE_BYTES or not all(loader._hash(row[k]) for k in ('file_sha256','volume_sha256','mask_sha256')):
                raise ValueError('Private file index invalid')
            if ordinal in rows:raise ValueError('Overlapping ordinals')
            rows[ordinal]=uid;ids.append(uid);remote='source96_cache/'+name
            if remote not in plan['inventory']:raise ValueError('Expected cache artifact absent')
            files.append({'path':cache.safe_file(root,name),'url':plan['inventory'][remote],'row':row,'binding':plan['binding']})
        if contract['partitions'][n]['ordered_ids_sha256']!=loader.digest_document(ids):raise ValueError('Shard ID order differs')
        expected={'source96_cache/'+cache.filename(i) for i in members}
        if {name for name in plan['inventory'] if name.endswith('.npz')}!=expected:raise ValueError('Extra or missing output NPZ')
    ids=[rows[i] for i in range(counts[0])]
    if len(set(ids))!=len(ids) or loader.digest_document(ids)!=validation['membership']['ordered_eligible_sha256']:
        raise ValueError('Global eligible order differs')
    folds=validation['folds']
    if any(uid not in folds or type(folds[uid]) is not int or not 0<=folds[uid]<=4 for uid in ids):raise ValueError('Eligible fold metadata invalid')
    train=[u for u in ids if folds[u]!=0];dev=[u for u in ids if folds[u]==0]
    if (len(train),len(dev))!=counts[1:] or loader.digest_document(train)!=validation['membership']['ordered_training_sha256'] or loader.digest_document(dev)!=validation['membership']['ordered_development_sha256']:
        raise ValueError('Historical train/development order differs')
    return files


def prepare(config,transport,*,allow_fixture=False):
    root=Path(config['cache_root'])
    if config.get('version')!=VERSION or root.is_symlink() or len(config['sources'])!=4:raise ValueError('Staging configuration differs')
    if config['validation'].get('fixture') and not allow_fixture:raise ValueError('Fixture staging not allowed in production')
    bindings=[_binding(d) for d in config['sources']]
    if len({b['ref'] for b in bindings})!=4:raise ValueError('Four unique immutable source slugs required')
    root.mkdir(parents=True,exist_ok=True)
    authority={'version':VERSION,'bindings':bindings,'validation':config['validation']}
    authority_sha=loader.digest_document(authority);control=root/'staging-contract-private.json'
    if control.exists():
        if control.is_symlink() or json.loads(control.read_bytes())!=authority:raise ValueError('Prior staged authority differs')
    elif any(root.iterdir()):raise ValueError('Unbound existing staging root')
    else:cache.atomic_json(control,authority)
    # Prior ready metadata remains recoverable, but is not a readiness signal for this attempt.
    for name in ('staging-receipt.json','cache96-manifest-private.json'):
        previous=root/name
        if previous.exists():
            if previous.is_symlink():raise ValueError('Prior metadata symlink forbidden')
            history=root/'staging-history-private';history.mkdir(exist_ok=True)
            if history.is_symlink():raise ValueError('Staging history symlink forbidden')
            previous.replace(history/(name+'.'+smoke.file_sha(previous)))
    report={'version':VERSION,'status':'RUNNING','phase':'metadata','authority_sha256':authority_sha,
            'sources':[{'ref':b['ref'],'version':1,'cell_sources_sha256':b['cell_sources_sha256']} for b in bindings],
            'new_downloaded_bytes':0,'completed_files':0,'aggregate_contains_ids':False}
    started=time.monotonic()
    def write():
        report['elapsed_seconds']=time.monotonic()-started;cache.atomic_json(root/'staging-receipt.json',report)
    write()
    try:
        plans=[]
        if shutil.disk_usage(root).free<4*3*METADATA_MAX_BYTES+DISK_RESERVE:raise OSError('Insufficient metadata staging disk')
        for i,binding in enumerate(bindings):
            directory=root/f'source{i}'
            if directory.is_symlink():raise ValueError('Shard destination symlink forbidden')
            directory.mkdir(exist_ok=True);inventory=_inventory(transport,binding)
            for name,field in METADATA.items():
                remote='source96_cache/'+name
                if remote not in inventory:raise ValueError('Required metadata artifact absent')
                report['new_downloaded_bytes']+=_download(transport,inventory[remote],directory/name,binding[field],max_bytes=METADATA_MAX_BYTES)
            _snapshot(transport,binding);plans.append({'root':directory,'binding':binding,'inventory':inventory})
        files=_metadata_plan(plans,config['validation'])
        for plan in plans:
            allowed={item['path'].name for item in files if item['path'].parent==plan['root']}
            if not {p.name for p in plan['root'].glob('*.npz')}<=allowed:raise ValueError('Unexpected existing cache file; preserve and stop')
        missing=[]
        for item in files:
            path=item['path'];row=item['row']
            if path.exists() or path.is_symlink():
                if path.is_symlink() or path.stat().st_size!=row['bytes'] or smoke.file_sha(path)!=row['file_sha256']:
                    raise ValueError('Completed private artifact differs; preserve and stop')
            else:missing.append(item)
        required=sum(i['row']['bytes'] for i in missing)+2*max((i['row']['bytes'] for i in files),default=0)+DISK_RESERVE
        free=shutil.disk_usage(root).free;report['disk_preflight']={'free_bytes':free,'required_bytes':required,'remaining_files':len(missing)}
        if free<required:raise OSError('Insufficient complete cache staging disk')
        report['phase']='images';report['completed_files']=len(files)-len(missing);write()
        for item in missing:
            row=item['row']
            if shutil.disk_usage(root).free<2*row['bytes']+DISK_RESERVE:raise OSError('Disk changed during staging')
            report['new_downloaded_bytes']+=_download(transport,item['url'],item['path'],row['file_sha256'],expected_bytes=row['bytes'],max_bytes=cache.MAX_FILE_BYTES)
            report['completed_files']+=1;write()
        for binding in bindings:_snapshot(transport,binding)
        report['phase']='assemble';write()
        descriptors=[]
        for plan in plans:
            for path in plan['root'].glob('*.npz'):
                if path.stat().st_mode&0o222:path.chmod(0o444)
            descriptors.append({'root':str(plan['root']),**{k:plan['binding'][k] for k in ('contract_file_sha256','index_sha256','receipt_sha256')}})
        manifest=loader.assemble(descriptors,config['validation'])
        for binding in bindings:_snapshot(transport,binding)
        cache.atomic_json(root/'cache96-manifest-private.json',manifest)
        report.update(status='PASS',phase='complete',studies=len(manifest['entries']),
                      manifest_sha256=manifest['manifest_sha256'],content_sha256=manifest['content_sha256'],
                      manifest_file_sha256=smoke.file_sha(root/'cache96-manifest-private.json'))
        write();return report
    except BaseException as error:
        report.update(status='ERROR',failure={'phase':report['phase'],'type':type(error).__name__});write()
        raise RuntimeError('Private96 staging failed; completed files and sanitized receipt preserved') from None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args(argv)
    try:
        config=json.loads(Path(args.config).read_bytes());report=prepare(config,KaggleTransport())
    except BaseException:
        raise RuntimeError('Private96 staging failed; inspect sanitized staging receipt') from None
    print(json.dumps({'event':'SOURCE96_PREPARE_OK','studies':report['studies'],'manifest_sha256':report['manifest_sha256'],'content_sha256':report['content_sha256']}))


if __name__=='__main__':main()
