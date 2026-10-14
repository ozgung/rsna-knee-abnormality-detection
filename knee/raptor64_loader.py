"""Verified private Raptor64 staging and lazy StudyWindows-compatible array proxies.

Parallel to the qualified source96 loader (unchanged), for the 64-slot Dread v4 cache.
All file/raw hashes are checked once by assemble(); every later access checks the staged
file's device/inode/size/mtime/ctime signature. Per-slot plane/Fluid_Sensitive/series-length
metadata from the private index is carried for the plane-aware head.
"""
import copy
import os
from pathlib import Path

from knee import raptor64_cache as cache, public_training_smoke as smoke
from knee.source96_loader import digest_document, _hash, _signature, _pinned_json, _validate_profile

VERSION = 'raptor64-lazy-loader-v1'
PRODUCTION_COUNTS = (4345, 3475, 870)
COMMON = ('version', 'config', 'membership', 'partitions', 'reader', 'input_pins', 'qualified_environment', 'sources')
SLOT_KEYS = {'plane', 'selected', 'fluid_sensitive', 'frames', 'count'}


def validate_slots(slots):
    if not isinstance(slots, list) or len(slots) != len(cache.SLOTS):
        raise ValueError('Slot metadata must cover the five v4 slots')
    for s, (plane, _, count) in zip(slots, cache.SLOTS):
        if (not isinstance(s, dict) or set(s) != SLOT_KEYS or s['plane'] != plane or s['count'] != count or
                type(s['selected']) is not bool or type(s['frames']) is not int or s['frames'] < 0 or
                (s['selected'] and s['fluid_sensitive'] not in (0, 1)) or (not s['selected'] and (s['fluid_sensitive'] is not None or s['frames']))):
            raise ValueError('Slot metadata schema differs')
    cache.slice_metadata(slots)
    return slots


def assemble(shards, validation, *, manifest_path=None):
    """Verify four completed shards; return a PRIVATE aligned manifest (fixture=True only for synthetic cohorts)."""
    fixture, counts = _validate_profile(validation)
    if not fixture and counts != PRODUCTION_COUNTS:
        raise ValueError('Production population must be 4345/3475/870')
    if len(shards) != 4:
        raise ValueError('Exactly four qualified shards required')
    records = {}; sources = {}; common = None
    for descriptor in shards:
        root = Path(descriptor['root'])
        if root.is_symlink() or not root.is_dir():
            raise ValueError('Invalid shard root')
        root = root.resolve()
        contract = _pinned_json(root / 'run-contract.json', descriptor['contract_file_sha256'])
        contract_sha = digest_document(contract)
        index = _pinned_json(root / 'index-private.json', descriptor['index_sha256'])
        receipt = _pinned_json(root / 'receipt.json', descriptor['receipt_sha256'])
        shard = contract['shard_index']
        if type(shard) is not int or not 0 <= shard < 4 or shard in sources or contract['shard_count'] != 4:
            raise ValueError('Duplicate or invalid shard identity')
        if contract['version'] != cache.VERSION or contract['config'] != cache.CONFIG or contract['membership'] != validation['membership']:
            raise ValueError('Qualified cache population/policy differs')
        this_common = {k: contract[k] for k in COMMON}
        if common is None:
            common = this_common
        elif this_common != common:
            raise ValueError('Shards differ in scientific source/pins')
        members = list(range(shard, counts[0], 4)); partition = contract['partitions'][shard]
        if len(contract['partitions']) != 4 or partition['studies'] != len(members) or partition['ordinals_sha256'] != digest_document(members):
            raise ValueError('Shard partition differs')
        if (receipt.get('status') != 'PASS' or receipt.get('version') != cache.VERSION or receipt.get('contract_sha256') != contract_sha or
                receipt.get('private_index_sha256') != descriptor['index_sha256'] or receipt.get('completed_studies') != len(members) or
                receipt.get('shard_index') != shard or receipt.get('shard_count') != 4):
            raise ValueError('Only fully completed reviewed shards may train')
        if index.get('version') != cache.VERSION or index.get('contract_sha256') != contract_sha:
            raise ValueError('Private index contract differs')
        entries = index.get('entries')
        if not isinstance(entries, list) or [e.get('ordinal') for e in entries] != members:
            raise ValueError('Private shard index must have exact ordered coverage')
        if {p.name for p in root.glob('*.npz')} != {cache.filename(n) for n in members}:
            raise ValueError('Unexpected or missing final shard files')
        shard_ids = []
        for entry in entries:
            n = entry['ordinal']; uid = cache.safe_id(entry['StudyInstanceUID'])
            if entry.get('file') != cache.filename(n) or any(not _hash(entry.get(k)) for k in ('file_sha256', 'volume_sha256', 'mask_sha256')):
                raise ValueError('Private index schema differs')
            if n in records:
                raise ValueError('Overlapping shard ordinal')
            slots = validate_slots(entry.get('slots'))
            path = cache.safe_file(root, entry['file']); before = _signature(path)
            cache.verify_file(path, entry)
            if _signature(path) != before:
                raise ValueError('File mutated during staging verification')
            records[n] = {'ordinal': n, 'StudyInstanceUID': uid, 'shard': shard, 'file': entry['file'], 'bytes': entry['bytes'],
                          'file_sha256': entry['file_sha256'], 'volume_sha256': entry['volume_sha256'], 'mask_sha256': entry['mask_sha256'],
                          'slots': slots, 'stat': before}
            shard_ids.append(uid)
        if partition['ordered_ids_sha256'] != digest_document(shard_ids):
            raise ValueError('Shard identity hash differs')
        sources[shard] = {'root': str(root), 'contract_sha256': contract_sha, 'contract_file_sha256': descriptor['contract_file_sha256'],
                          'index_sha256': descriptor['index_sha256'], 'receipt_sha256': descriptor['receipt_sha256']}
    if sorted(records) != list(range(counts[0])):
        raise ValueError('Four-shard coverage not exhaustive')
    entries = [records[n] for n in range(counts[0])]; ids = [e['StudyInstanceUID'] for e in entries]
    if len(set(ids)) != len(ids) or digest_document(ids) != validation['membership']['ordered_eligible_sha256']:
        raise ValueError('Global historical eligible order differs')
    folds = validation['folds']
    if any(u not in folds or type(folds[u]) is not int or not 0 <= folds[u] <= 4 for u in ids):
        raise ValueError('Eligible fold metadata differs')
    train = [u for u in ids if folds[u] != 0]; dev = [u for u in ids if folds[u] == 0]
    if ((len(train), len(dev)) != counts[1:] or digest_document(train) != validation['membership']['ordered_training_sha256'] or
            digest_document(dev) != validation['membership']['ordered_development_sha256']):
        raise ValueError('Training/development membership differs')
    identity = {'membership': validation['membership'], 'common': common,
                'sources': [{k: v for k, v in sources[i].items() if k != 'root'} for i in range(4)],
                'entries': [{k: v for k, v in e.items() if k != 'stat'} for e in entries]}
    manifest = {'version': VERSION, 'fixture': fixture, 'membership': validation['membership'],
                'sources': [sources[i] for i in range(4)], 'entries': entries,
                'content_sha256': digest_document(identity), 'verification': 'all file/raw hashes checked once at staging',
                'trust_boundary': 'Read-only staged files; signatures checked at every access; re-stage after mutation'}
    manifest['manifest_sha256'] = digest_document(manifest)
    if manifest_path is not None:
        cache.atomic_json(manifest_path, manifest)
    return manifest


def _load_pair(path):
    """Bounded schema/CRC loading; no repeated SHA over staged image bytes."""
    import numpy as np
    try:
        volume, mask = cache.read_bounded(path)
        cache.validate_arrays(volume, mask)
        volume.flags.writeable = False; mask.flags.writeable = False
        return volume, mask
    except Exception:
        raise ValueError('Staged cache loading failed; re-stage verified inputs') from None


class _PairStore:
    def __init__(self, manifest):
        self.entries = copy.deepcopy(manifest['entries']); self.sources = copy.deepcopy(manifest['sources'])
        self.pid = os.getpid(); self.row = None; self.pair = None

    def __getstate__(self):
        state = self.__dict__.copy(); state.update(pid=None, row=None, pair=None); return state

    def get(self, row):
        if type(row) is not int or not 0 <= row < len(self.entries):
            raise IndexError('Integer cache row required')
        pid = os.getpid()
        if self.pid != pid:
            self.pid = pid; self.row = None; self.pair = None
        entry = self.entries[row]; path = cache.safe_file(self.sources[entry['shard']]['root'], entry['file'])
        if _signature(path) != entry['stat']:
            self.row = None; self.pair = None; raise ValueError('Staged file mutated; re-stage verified inputs')
        if self.row != row:
            self.row = None; self.pair = None
            pair = _load_pair(path)
            if _signature(path) != entry['stat']:
                raise ValueError('Staged file mutated while loading')
            self.row = row; self.pair = pair
        return self.pair


class _Proxy:
    def __init__(self, store, kind):
        self.store, self.kind = store, kind
        self.shape = (len(store.entries), cache.DEPTH, cache.STORAGE, cache.STORAGE) if kind == 0 else (len(store.entries), cache.DEPTH)

    @property
    def dtype(self):
        import numpy as np
        return np.dtype('uint8')

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.store.get(row)[self.kind]

    def __array__(self, dtype=None, copy=None):
        raise ValueError('Dense conversion/preload forbidden for lazy Raptor64 corpus')


def slice_metadata_table(manifest):
    """(N,64) int64 codes and float32 depths in manifest row order."""
    import numpy as np
    codes, depth = zip(*(cache.slice_metadata(e['slots']) for e in manifest['entries']))
    return np.stack(codes), np.stack(depth)


def open_corpus(manifest, *, expected_manifest_sha256, allow_fixture=False, preload=False):
    """Caller supplies SHA from independently verified staging, not file self-claims."""
    import numpy as np
    if preload or os.environ.get('RAPTOR_PRELOAD') == '1':
        raise ValueError('RAPTOR_PRELOAD incompatible with lazy Raptor64 corpus')
    body = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if (manifest.get('version') != VERSION or not _hash(expected_manifest_sha256) or
            digest_document(body) != expected_manifest_sha256 or manifest.get('manifest_sha256') != expected_manifest_sha256):
        raise ValueError('Merged manifest differs from trusted staging digest')
    if manifest.get('fixture') and not allow_fixture:
        raise ValueError('Synthetic fixture corpus cannot be used as production')
    counts = tuple(manifest['membership'][k] for k in ('eligible', 'training', 'development'))
    if not manifest.get('fixture') and counts != PRODUCTION_COUNTS:
        raise ValueError('Production population differs')
    entries = manifest['entries']
    if [e['ordinal'] for e in entries] != list(range(counts[0])):
        raise ValueError('Merged row order differs')
    ids = np.asarray([e['StudyInstanceUID'] for e in entries], dtype=str)
    if digest_document(ids.tolist()) != manifest['membership']['ordered_eligible_sha256']:
        raise ValueError('Merged IDs differ')
    for e in entries:
        validate_slots(e['slots'])
    store = _PairStore(manifest)
    return _Proxy(store, 0), _Proxy(store, 1), ids
