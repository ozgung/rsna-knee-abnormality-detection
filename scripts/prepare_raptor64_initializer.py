"""VM-only pinned Raptor64 initializer preparation. CLI: ROOT --init {imagenet,dread_v4,nartaa_oai}.

imagenet: fixed-revision timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k safetensors from the
public Hugging Face endpoint, normalized to backbone-only tensors (reviewed classifier keys
omitted), exactly like the qualified source96 224 initializer.
dread_v4: the CC0 Dread checkpoint raptor_ft_coatnet_v4_full.pt as one exact file from the
public Kaggle dataset dreaddevelopment/raptor-knee-widedense (existing VM Kaggle credential;
no terms accepted). Bytes and size are pinned; the trainer later loads backbone tensors only
and enforces missing=0/unexpected=0.
nartaa_oai: nartaa's OAI-pretrained SWA Raptor raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt from the public
Kaggle dataset nartaa/rsna-knee-publication-swa-weights-20261007, same exact-file path and pins as dread_v4.
Writes ROOT/initializer.json; never overwrites a completed descriptor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import importlib.util
_spec = importlib.util.spec_from_file_location('prepare_source96_initializer', Path(__file__).resolve().parent / 'prepare_source96_initializer.py')
s96 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s96)

ARCH = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
IMAGENET_PIN = {'repo': 'timm/' + ARCH, 'revision': '2c31d82376cc244c907c064332941e6c65be81f1', 'file': 'model.safetensors',
                'sha256': '91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e', 'size': 295695280}
DREAD_PIN = {'dataset': 'dreaddevelopment/raptor-knee-widedense', 'file': 'raptor_ft_coatnet_v4_full.pt',
             'sha256': '89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded', 'size': 292829402}
NARTAA_PIN = {'dataset': 'nartaa/rsna-knee-publication-swa-weights-20261007', 'file': 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt',
              'sha256': '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', 'size': 292840787}
TIMM_VERSION = s96.TIMM_VERSION
CHUNK = 1024 * 1024


def model_factory(num_classes):
    import timm
    if timm.__version__ != TIMM_VERSION:
        raise RuntimeError('Requires historical timm ' + TIMM_VERSION)
    return timm.create_model(ARCH, pretrained=False, num_classes=num_classes, in_chans=3, global_pool='avg')


def _fresh(root):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('Symlink output root rejected')
    root.mkdir(parents=True, exist_ok=True)
    descriptor = root / 'initializer.json'
    if descriptor.exists() or descriptor.is_symlink():
        raise ValueError('Completed descriptor preserved; use a fresh directory')
    return root, descriptor


def prepare_imagenet(root, *, fetch=s96.download, factory=model_factory, load=s96.safe_load, disk_usage=shutil.disk_usage):
    import torch
    root, descriptor = _fresh(root)
    origin = root / 'model.safetensors'; weights = root / 'backbone.pt'
    if weights.exists():
        raise ValueError('Normalized output preserved; use a fresh directory')
    needed = (0 if origin.exists() else IMAGENET_PIN['size']) + IMAGENET_PIN['size'] + 512 * 1024 * 1024
    if disk_usage(root).free < needed:
        raise OSError('Insufficient disk for origin plus normalized output and 512MiB reserve')
    if not origin.exists():
        url = 'https://huggingface.co/' + IMAGENET_PIN['repo'] + '/resolve/' + IMAGENET_PIN['revision'] + '/' + IMAGENET_PIN['file']
        temp = origin.with_name(origin.name + '.part'); h = hashlib.sha256(); count = 0
        try:
            with fetch(url) as response, open(temp, 'wb') as out:
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    count += len(block)
                    if count > IMAGENET_PIN['size']:
                        raise ValueError('Origin exceeds pinned size')
                    h.update(block); out.write(block)
                out.flush(); os.fsync(out.fileno())
            if count != IMAGENET_PIN['size'] or h.hexdigest() != IMAGENET_PIN['sha256']:
                raise ValueError('Origin size/checksum differs')
            os.replace(temp, origin)
        finally:
            if temp.exists():
                temp.unlink()
    if origin.stat().st_size != IMAGENET_PIN['size'] or s96.file_sha(origin) != IMAGENET_PIN['sha256']:
        raise ValueError('Existing origin size/checksum differs')
    with s96.preserved_rng():
        full = factory(1000); backbone = factory(0)
        normalized, audit = s96.normalize(load(origin), full, backbone)
        temp = weights.with_name(weights.name + '.part')
        try:
            with open(temp, 'wb') as f:
                torch.save(normalized, f); f.flush(); os.fsync(f.fileno())
            reloaded = torch.load(temp, map_location='cpu', weights_only=True)
            if s96.state_sha(reloaded) != audit['backbone_tensor_sha256']:
                raise ValueError('Normalized reload tensor identity differs')
            backbone.load_state_dict(reloaded, strict=True); os.replace(temp, weights)
        finally:
            if temp.exists():
                temp.unlink()
    d = {'init': 'imagenet', 'kind': 'imagenet_backbone', 'arch': ARCH, 'pretrained_tag': 'sw_in12k_ft_in1k', 'hf_repo': IMAGENET_PIN['repo'],
         'hf_revision': IMAGENET_PIN['revision'], 'origin_file_sha256': IMAGENET_PIN['sha256'], 'origin_file_bytes': IMAGENET_PIN['size'],
         'weights_file': str(weights.resolve()), 'weights_sha256': s96.file_sha(weights), 'weights_bytes': weights.stat().st_size,
         'timm': TIMM_VERSION, 'torch': torch.__version__, 'audit': audit,
         'source': 'Official public timm ImageNet12k-to-1k artifact at 384; no knee-trained weights'}
    s96.atomic_json(descriptor, d)
    return {'descriptor': str(descriptor.resolve()), 'descriptor_sha256': s96.file_sha(descriptor), 'init': 'imagenet',
            'weights_sha256': d['weights_sha256'], 'backbone_tensor_sha256': audit['backbone_tensor_sha256'], 'training': False}


class KaggleFile:
    """One exact public dataset file through the existing configured credential; bounded streaming."""
    def fetch(self, dataset, name, target, limit):
        from kaggle.api.kaggle_api_extended import KaggleApi, ApiDownloadDatasetRequest
        api = KaggleApi(); api.authenticate()
        owner, slug = dataset.split('/')
        with api.build_kaggle_client() as client:
            request = ApiDownloadDatasetRequest(); request.owner_slug = owner; request.dataset_slug = slug; request.file_name = name
            response = client.datasets.dataset_api_client.download_dataset(request)
            try:
                count = 0
                with open(target, 'xb') as out:
                    blocks = response.iter_content(CHUNK) if hasattr(response, 'iter_content') else iter(lambda: response.read(CHUNK), b'')
                    for block in blocks:
                        count += len(block)
                        if count > limit:
                            raise ValueError('Remote file exceeds size bound')
                        out.write(block)
                    out.flush(); os.fsync(out.fileno())
            finally:
                response.close()


def prepare_dread(root, *, transport=None, disk_usage=shutil.disk_usage):
    return _prepare_published(root, DREAD_PIN, {
        'init': 'dread_v4', 'kind': 'dread_v4_full', 'label': 'Dread v4_full', 'licence': 'CC0-1.0 (dataset card fetched 2026-10-12)',
        'exposure': 'Trained by Dread on all non-gold competition studies; fold-0 OOF is exposed'}, transport, disk_usage)


def prepare_nartaa(root, *, transport=None, disk_usage=shutil.disk_usage):
    return _prepare_published(root, NARTAA_PIN, {
        'init': 'nartaa_oai', 'kind': 'nartaa_oai_swa', 'label': 'nartaa OAI SWA', 'licence': 'per dataset card (other; OAI acknowledgement)',
        'exposure': 'nartaa trained on all competition studies plus OAI; fold-0 OOF is exposed'}, transport, disk_usage)


def _prepare_published(root, pin, meta, transport, disk_usage):
    root, descriptor = _fresh(root)
    weights = root / pin['file']; payload = root / 'download.part'
    if payload.exists() or payload.is_symlink():
        raise ValueError('Partial download preserved; use a fresh directory')
    if not weights.exists():
        if disk_usage(root).free < 3 * pin['size'] + 512 * 1024 * 1024:
            raise OSError('Insufficient disk for download, extraction and reserve')
        (transport or KaggleFile()).fetch(pin['dataset'], pin['file'], payload, pin['size'] + 64 * 1024 * 1024)
        try:
            with open(payload, 'rb') as f:
                magic = f.read(2)
            if magic == b'PK':
                with zipfile.ZipFile(payload) as archive:
                    items = archive.infolist()
                    # Kaggle may rename the sole member. Its name is never used as an
                    # output path; accept only a regular file and verify pinned bytes below.
                    mode = items[0].external_attr >> 16 if len(items) == 1 else 0
                    if (len(items) != 1 or items[0].is_dir() or
                            stat.S_IFMT(mode) not in (0, stat.S_IFREG) or
                            items[0].file_size != pin['size'] or items[0].flag_bits & 1):
                        raise ValueError('Unexpected archive member')
                    with archive.open(items[0]) as source, open(weights.with_name(weights.name + '.part'), 'xb') as out:
                        shutil.copyfileobj(source, out, CHUNK); out.flush(); os.fsync(out.fileno())
                os.replace(weights.with_name(weights.name + '.part'), weights)
            else:
                os.replace(payload, weights)
        finally:
            for p in (payload, weights.with_name(weights.name + '.part')):
                if p.exists():
                    p.unlink()
    if weights.is_symlink() or weights.stat().st_size != pin['size'] or s96.file_sha(weights) != pin['sha256']:
        raise ValueError(meta['label'] + ' size/checksum differs from the pin')
    os.chmod(weights, 0o444)
    d = {'init': meta['init'], 'kind': meta['kind'], 'dataset': pin['dataset'], 'file': pin['file'],
         'weights_file': str(weights.resolve()), 'weights_sha256': pin['sha256'], 'weights_bytes': pin['size'],
         'licence': meta['licence'], 'use': 'backbone.* tensors only; head fresh; missing=0/unexpected=0 enforced at load',
         'exposure': meta['exposure']}
    s96.atomic_json(descriptor, d)
    return {'descriptor': str(descriptor.resolve()), 'descriptor_sha256': s96.file_sha(descriptor), 'init': meta['init'],
            'weights_sha256': pin['sha256'], 'training': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', type=Path)
    parser.add_argument('--init', choices=('imagenet', 'dread_v4', 'nartaa_oai'), required=True)
    a = parser.parse_args()
    print(json.dumps({'imagenet': prepare_imagenet, 'dread_v4': prepare_dread, 'nartaa_oai': prepare_nartaa}[a.init](a.root), indent=2))
