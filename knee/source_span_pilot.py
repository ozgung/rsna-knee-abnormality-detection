"""Label-blind paired span mechanics; no models or persistent image outputs."""
import ast
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import random
import resource
import sys
import tempfile
import time
from knee import corpus44_parity as parity, public_training_smoke as smoke
from knee.public_rich_cache import select_inputs, SEED, COUNT
from knee.raptor_input import guarded_reader

VERSION = 'own224-span-depth-three32-v2'
SPANS = {'narrow': [.15, .85], 'wide': [.02, .98], 'wide96': [.02, .98]}
SLOTS96 = [('Sagittal',1,26), ('Sagittal',0,22), ('Coronal',1,18), ('Coronal',0,12), ('Axial',-1,18)]
ARMS = {
    'narrow': {'reader':'narrow', 'slots':[list(s) for s in parity.SLOTS], 'depth':44},
    'wide': {'reader':'wide', 'slots':[list(s) for s in parity.SLOTS], 'depth':44},
    'wide96': {'reader':'wide', 'slots':[list(s) for s in SLOTS96], 'depth':96},
}
CONFIG = {'storage': 336,
          'crop_mm': 140., 'tensor_resolution': 224, 'windows': [8, 16],
          'cpu_threads': 1, 'seed': SEED, 'studies': COUNT, 'spans': SPANS, 'arms': ARMS,
          'norm': 'imagenet', 'training': False, 'models_loaded': 0,
          'sampler': 'pinned StudyWindows evaluation, legacy boundary/mask behavior'}


def recipes(source):
    narrow = parity.recipe(source)
    tree = ast.parse(narrow)
    changed = Counter()
    class Span(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is float and node.value in (.15, .85):
                changed[node.value] += 1
                return ast.copy_location(ast.Constant({.15:.02, .85:.98}[node.value]), node)
            return node
    Span().visit(tree.body[-1])
    if changed != {.15: 1, .85: 1}: raise ValueError('Exactly two source span literals required')
    wide = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    validate_recipes(narrow, wide)
    return {'narrow': narrow, 'wide': wide}


def validate_recipes(narrow, wide):
    a, b = ast.parse(narrow), ast.parse(wide)
    changed = Counter()
    class Undo(ast.NodeTransformer):
        def visit_Constant(self, node):
            if type(node.value) is float and node.value in (.02, .98):
                changed[node.value] += 1
                return ast.copy_location(ast.Constant({.02:.15, .98:.85}[node.value]), node)
            return node
    Undo().visit(b.body[-1])
    if changed != {.02: 1, .98: 1} or ast.dump(a) != ast.dump(b):
        raise ValueError('Paired reader AST differs beyond two span constants')


def arm_module(codes, arm):
    """Only declared slots/MAXS differ; shared scientific functions remain identical."""
    spec = ARMS[arm]
    module = parity.reader_module(codes[spec['reader']])
    module.SLOTS = [tuple(s) for s in spec['slots']]
    module.MAXS = spec['depth']
    if sum(s[2] for s in module.SLOTS) != module.MAXS:
        raise ValueError('Declared source allocation differs')
    return module


def sampler_source(source):
    if smoke.digest(source.encode()) != smoke.SOURCE_SHA: raise ValueError('Training source changed')
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'StudyWindows']
    if len(nodes) != 1: raise ValueError('Sampler inventory changed')
    return ast.unparse(ast.Module(body=nodes, type_ignores=[])) + '\n'


def sampler_class(code):
    import numpy as np
    import torch
    from torch.nn import functional as F
    ns = {'np': np, 'torch': torch, 'F': F, 'random': random,
          'Dataset': torch.utils.data.Dataset, '_open_corpus': lambda: None}
    exec(compile(code, '<pinned-StudyWindows-only>', 'exec'), ns)
    return ns['StudyWindows']


def validate_arrays(volume, mask, *, depth=44):
    import numpy as np
    if depth not in (44,96) or volume.shape != (depth, 336, 336) or volume.dtype != np.uint8: raise ValueError('Volume geometry changed')
    if mask.shape != (depth,) or not np.isin(mask, [0, 1]).all(): raise ValueError('Mask geometry changed')
    if not np.array_equal(mask.astype(bool), volume.reshape(depth, -1).any(axis=1)):
        raise ValueError('Pixel/mask consistency failed')
    if not mask.any(): raise ValueError('Whole-empty study cannot qualify')


def window_checks(volume, mask, sampler, *, slots=parity.SLOTS):
    import numpy as np
    import torch
    depth=sum(s[2] for s in slots)
    validate_arrays(volume, mask, depth=depth)
    slots = np.repeat(np.arange(5), [s[2] for s in slots]); result = {}
    for k in CONFIG['windows']:
        ds = sampler('', ['synthetic'], {'synthetic': 0}, {'synthetic': np.zeros(12, np.float32)},
                     224, k, train=False, aug=False, norm='imagenet')
        ds.vols, ds.masks, ds._D = volume[None], mask[None], depth
        valid = np.where(mask > 0)[0]
        if len(valid) < 3: valid = np.arange(3)
        centers = ds._centers(valid, k); x, _ = ds[0]
        if tuple(x.shape) != (k, 3, 224, 224) or not torch.isfinite(x).all():
            raise ValueError('Own224 window shape/finiteness failed')
        triplets = np.stack([np.asarray(centers)-1, centers, np.asarray(centers)+1], 1)
        result[str(k)] = {'shape': list(x.shape), 'finite': True,
                         'tensor_sha256': smoke.digest(x.contiguous().numpy().tobytes()),
                         'unique_centers': len(set(centers)),
                         'cross_slot_windows': int(np.count_nonzero(np.ptp(slots[triplets], axis=1))),
                         'masked_slice_windows': int(np.count_nonzero(~mask[triplets].astype(bool).all(axis=1)))}
    return result


def roundtrip(volume, mask):
    """Ephemeral MRI files are removed before returning."""
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='source-span-') as directory:
        paths = [Path(directory)/'volume.npy', Path(directory)/'mask.npy']
        for path, array in zip(paths, [volume, mask]): np.save(path, array, allow_pickle=False)
        for path, array in zip(paths, [volume, mask]):
            loaded = np.load(path, allow_pickle=False)
            if loaded.dtype != array.dtype or loaded.shape != array.shape or loaded.tobytes() != array.tobytes():
                raise ValueError('Save/reload bytes differ')
        return {'volume_sha256': smoke.digest(volume.tobytes()), 'mask_sha256': smoke.digest(mask.tobytes()),
                'ephemeral_npy_bytes': sum(p.stat().st_size for p in paths), 'save_reload_equal': True}


def decode(public, sid, records, root, span):
    """Observe normalized source ordinal extent; emit no paths or DICOM fields."""
    import numpy as np
    counters = Counter(); selected, used = [], set()
    for plane, fluid, count in public.SLOTS:
        row = public._pick_series_for_slot(records[sid], plane, fluid, used)
        if row is not None: used.add(row['SeriesInstanceUID'])
        selected.append((row, count))
    observations = []; original = public._make_reader
    def tracked_factory():
        order, pixel, resize = original()
        def tracked_order(path):
            files, spacing = order(path); n = len(files)
            slot = next(i for i, (row, _) in enumerate(selected) if row is not None and path.endswith('/'+row['SeriesInstanceUID']))
            count = selected[slot][1]
            lo, hi = int(n*span[0]), max(int(n*span[1])-1, int(n*span[0]))
            picks = np.linspace(lo, hi, count).round().astype(int) if n > 1 else np.zeros(count, int)
            picks = np.minimum(picks, max(0, n-1))
            unique = len(set(picks.tolist())) if n else 0
            observations.append({'slot': slot, 'raw_frames': n, 'selected_frames': count if n else 0,
                                 'unique_picks': unique, 'repeated_picks': count-unique if n else 0,
                                 'first_fraction': float(picks.min()/max(n-1, 1)) if n else None,
                                 'last_fraction': float(picks.max()/max(n-1, 1)) if n else None})
            return files, spacing
        def tracked_pixel(path):
            try:
                a = pixel(path); counters['decoded_frames'] += 1; return a
            except Exception:
                counters['decode_failures'] += 1; raise
        return tracked_order, tracked_pixel, resize
    public._make_reader = tracked_factory
    try:
        with guarded_reader(public, counters):
            volume, mask = public.build_study(sid, records, str(root), public._make_reader())
    finally: public._make_reader = original
    parity.require_decode_evidence(counters); validate_arrays(volume, mask, depth=public.MAXS)
    offset = 0
    for row, count in selected:
        valid = int(mask[offset:offset+count].sum()); offset += count
        counters['missing_slots'] += int(row is None)
        counters['partial_slots'] += int(row is not None and 0 < valid < count)
        counters['empty_selected_slots'] += int(row is not None and valid == 0)
        counters['masked_frames'] += count-valid
    return volume, mask, {'reader_counts': dict(counters), 'ordinal_depth': observations,
                          'coverage_limitation': 'Normalized ordinal extent; physical positions unavailable after pinned reader projection'}


def run(inputs, out, contract, codes, sampler_code, *, started):
    import numpy as np
    import torch
    import cv2
    from knee.fresh_reader_setup import dataset_root, notebook_root, exact_root, time_limit
    torch.set_num_threads(1); cv2.setNumThreads(1)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    report = {'version': VERSION, 'status': 'RUNNING', 'phase': 'setup', 'config': CONFIG,
              'contract_sha256': smoke.digest(smoke.canonical(contract)), 'studies': [],
              'images_exported': False, 'identifiers_exported': False, 'models_loaded': 0,
              'input_pins': {k:contract[k] for k in ('competition_pins','identity_sha256','cache_pins','codecs','readers','sampler_sha256')},
              'code_revision':contract['code_revision'], 'source_hashes':contract['sources']}
    def save():
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        p = out/'receipt.json.tmp'; p.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); p.replace(out/'receipt.json')
    save()
    try:
        with time_limit(max(1, 43200-(time.monotonic()-started)), 'actual platform12h limit'):
            if contract['config'] != CONFIG or contract['version'] != VERSION: raise ValueError('Pilot configuration changed')
            validate_recipes(codes['narrow'], codes['wide'])
            if {k:smoke.digest(v.encode()) for k,v in codes.items()} != contract['readers'] or smoke.digest(sampler_code.encode()) != contract['sampler_sha256']:
                raise ValueError('Source extraction hash changed')
            identity = notebook_root(inputs, smoke.IDENTITY_KERNEL, 'sealed_identity', ['sealed-manifest-private.json'])
            smoke.require_file(identity/'sealed-manifest-private.json', smoke.MANIFEST_SHA)
            items = json.loads((identity/'sealed-manifest-private.json').read_bytes())['records']
            excluded = {'quarantined_study_ids': [r['uid'] for r in items if r['future_role']=='quarantined']}
            competition = exact_root([Path(inputs)/'competitions/rsna-knee-abnormality-detection', Path(inputs)/'rsna-knee-abnormality-detection'], ['train.csv','train_series.csv'])
            for name, pin in contract['competition_pins'].items(): smoke.require_file(competition/name, pin)
            ids, records, membership = select_inputs(competition, smoke.TARGETS, excluded)
            if smoke.digest(smoke.canonical(ids)) != contract['selected_sha256'] or membership != contract['membership']: raise ValueError('Frozen cohort changed')
            records = {u:records[u] for u in ids}; raw = parity.require_image_tree(competition, ids, records)
            report.update(membership=membership, selected_sha256=contract['selected_sha256'],
                          environment={'python':sys.version.split()[0], 'device':'cpu', 'torch_threads':torch.get_num_threads(), 'opencv_threads':cv2.getNumThreads(),
                          'packages':{p:importlib.metadata.version(p) for p in ('torch','numpy','pydicom','opencv-python','pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg')}})
            roots = [dataset_root(inputs, ref, [f'{prefix}_{s}.npy' for s in ('ids','masks','vols')]) for ref,prefix in zip(smoke.CORPORA, ['all','extra'])]
            arrays, masks, lookup = [], [], {}; report['phase'] = 'corpus_hashing'; save()
            for part,(root,prefix,size) in enumerate(zip(roots, ['all','extra'], [3200,1207])):
                for suffix in ('ids','masks','vols'):
                    name=f'{prefix}_{suffix}.npy'; pin=contract['cache_pins'][name]
                    if (root/name).stat().st_size != pin['bytes']: raise ValueError('Corpus size changed')
                    smoke.require_file(root/name,pin['sha256'])
                a=np.load(root/f'{prefix}_vols.npy',mmap_mode='r',allow_pickle=False)
                m=np.load(root/f'{prefix}_masks.npy',allow_pickle=False)
                local_ids=np.load(root/f'{prefix}_ids.npy',allow_pickle=False).astype(str)
                if a.shape!=(size,44,336,336) or a.dtype!=np.uint8 or m.shape!=(size,44) or not np.isin(m,[0,1]).all(): raise ValueError('Mounted corpus schema differs')
                for row,uid in enumerate(local_ids):
                    if uid in lookup: raise ValueError('Duplicate corpus identity')
                    lookup[uid]=(part,row)
                arrays.append(a); masks.append(m)
            if len(lookup)!=4407 or not set(ids)<=set(lookup): raise ValueError('Corpus identity coverage differs')
            report['verified_corpus_pins']=True
            sampler=sampler_class(sampler_code); modules={arm:arm_module(codes,arm) for arm in ARMS}
            report['phase']='paired_decode'; save(); matched=True
            for ordinal,sid in enumerate(ids):
                row={'ordinal':ordinal, 'arms':{}}
                report['studies'].append(row)
                for arm in ARMS:
                    report['active']={'ordinal':ordinal,'arm':arm};save()
                    begin=time.monotonic(); volume,mask,audit=decode(modules[arm],sid,records,raw,SPANS[arm])
                    armrow={**audit, **roundtrip(volume,mask), 'windows':window_checks(volume,mask,sampler,slots=modules[arm].SLOTS), 'valid_frames':int(mask.sum())}
                    if arm=='narrow':
                        part,index=lookup[sid]; slots=parity.compare(volume,mask,arrays[part][index],masks[part][index])
                        armrow['control_slots']=slots; equal=all(s['exact_hash_match'] and s['mask_match'] for s in slots)
                        matched &= equal; armrow['control_equal']=equal
                    if ordinal<2:
                        v2,m2,_=decode(modules[arm],sid,records,raw,SPANS[arm])
                        if not np.array_equal(volume,v2) or not np.array_equal(mask,m2): raise ValueError('Repeated decode differs')
                        armrow['repeated_decode_equal']=True; del v2,m2
                    armrow['elapsed_seconds']=time.monotonic()-begin; row['arms'][arm]=armrow; del volume,mask
                    save()
                report['completed_studies']=ordinal+1;save()
            report.pop('active',None)
            report.update(status='PASS' if matched else 'MISMATCH', phase='complete', completed_studies=len(report['studies']),
                          exact_control_equal=matched, repeat_decodes=6,
                          bytes_generated_ephemerally=sum(a['ephemeral_npy_bytes'] for r in report['studies'] for a in r['arms'].values()))
            save()
    except BaseException as error:
        report.update(status='ERROR',failure={'phase':report['phase'],'type':type(error).__name__}); save()
        raise RuntimeError('Source-span pilot failed; sanitized aggregate receipt saved') from None
    return report
