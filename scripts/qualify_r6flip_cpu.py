"""Actual R6 (or, with --profile r7, R7) checkpoint CPU pins for the R6-flip packets: constructor, strict state, buffers, side net, flip maps.

Runs the hash-verified pinned compact_train/compact_infer/side_common sources (the seven-file set already sealed in
A-prime) against the staged R6 v1 assets. Synthetic windows only: no DICOM reader, GPU, compile or runtime claim.
Needs torch 2.10.0 / timm 1.0.26 / opencv (the Kaggle pins), e.g.
  uv run --offline --no-project --python 3.12 --with torch==2.10.0 --with timm==1.0.26 --with numpy \
     --with opencv-python-headless python -I scripts/qualify_r6flip_cpu.py --assets ... --sources ... --out ...
"""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from knee import public_r6flip as R6F
from knee.public_r6_profile import SOURCE_SHA256, validate_r6_assets


def validate(profile, assets, sources):
    if profile == 'r7':
        from knee.public_r7_profile import validate_r7_assets      # P6: R7 v1 bundle, plain pass only (no flip maps pinned)
        return validate_r7_assets(assets, source_dir=sources)
    return validate_r6_assets(assets, source_dir=sources)


def import_pinned(name, root):
    path = Path(root) / (name + '.py')
    R6F.require(hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA256[name + '.py'], 'PINNED_SOURCE_' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--assets', required=True)
    p.add_argument('--sources', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--profile', choices=('r6', 'r7'), default='r6')
    a = p.parse_args(argv)
    t0 = time.time()
    import numpy as np
    import torch
    import timm
    torch.manual_seed(20261017)
    assets, sources = Path(a.assets).resolve(), Path(a.sources).resolve()
    profile = validate(a.profile, assets, sources)
    flip_pins = a.profile == 'r6'
    sys.path.insert(0, str(sources))
    CT = import_pinned('compact_train', sources)
    CI = import_pinned('compact_infer', sources)
    SC = import_pinned('side_common', sources)
    R6F.require(CT.LABELS == R6F.LABELS, 'TARGET_ORDER')
    manifest = json.loads((assets / 'manifest.json').read_bytes())
    spec = json.loads((assets / manifest['state_spec']).read_bytes())
    folds = []
    models = []
    for name in manifest['models']:
        path = assets / name
        sd = torch.load(path, map_location='cpu', weights_only=True)
        kw = {}
        if any(k == 'sp_q' or k.endswith('.sp_q') for k in sd):
            kw['spatial'] = True
        if any(k.split('.')[-1] in ('ext_u', 'ext_view', 'ext_gamma') for k in sd):
            kw['extent'] = True
        if any(k == 'syn_beta' or k.endswith('.syn_beta') for k in sd):
            kw['syn_decomp'] = True
        R6F.require(kw == {'spatial': True, 'extent': True, 'syn_decomp': True}, 'R6_FOLD_HEAD_KEYS')
        # compact_infer load_group's exact call
        m = CT.Net(manifest['backbone'], None, freeze_blocks=manifest['freeze_blocks'], crop=manifest['crop'], state_spec=spec,
                   in_chans=manifest['slices'], **kw)
        res = m.load_state_dict(sd, strict=True)
        bound = inspect.signature(CT.Net.__init__).bind(m, manifest['backbone'], None, freeze_blocks=manifest['freeze_blocks'],
                                                         crop=manifest['crop'], state_spec=spec, in_chans=manifest['slices'], **kw)
        bound.apply_defaults()
        args = {k: v for k, v in bound.arguments.items() if k != 'self'}
        args['state_spec'] = R6F.digest(args['state_spec'])
        folds.append({'file': name, 'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'strict': True,
                      'missing_keys': list(res.missing_keys), 'unexpected_keys': list(res.unexpected_keys),
                      'constructor_args': args, 'state': R6F.module_digest(m), 'buffers': R6F.buffers_digest(m),
                      'constructor': R6F.constructor_digest(m), 'bin_names': [list(x) for x in m.bin_names],
                      'state_k': [int(x) for x in m.state_k], 'parameters': sum(t.numel() for t in m.parameters())})
        R6F.require(not res.missing_keys and not res.unexpected_keys, 'R6_STRICT_LOAD')
        models.append(m.eval().requires_grad_(False))
        del sd
    R6F.require(all(f['bin_names'] == folds[0]['bin_names'] and f['state_k'] == folds[0]['state_k'] for f in folds), 'R6_FOLD_HEADS')
    side = SC.build_side_net()
    side_res = side.load_state_dict(torch.load(assets / manifest['side_model'], map_location='cpu', weights_only=True), strict=True)
    side = side.eval()
    side_pin = {'state': R6F.module_digest(side), 'buffers': R6F.buffers_digest(side), 'constructor': R6F.constructor_digest(side),
                'strict': not side_res.missing_keys and not side_res.unexpected_keys}
    if flip_pins:
        perm, bin_perm = R6F.mirror_perm(R6F.LABELS, spec, folds[0]['bin_names'])
        ci_perm, ci_bin = CI.mirror_perm(R6F.LABELS, spec, [tuple(x) for x in folds[0]['bin_names']])
        R6F.require(perm == ci_perm and bin_perm == ci_bin, 'FLIP_PERMUTATION_EQUALS_PINNED_SOURCE')
    else:
        perm = bin_perm = ci_perm = ci_bin = None
    # synthetic windows through fold 0: transform, aux read-out and deployed-logit arithmetic against the pinned source
    w = 32
    img = torch.randint(0, 256, (1, w, 3, manifest['crop'], manifest['crop']), dtype=torch.uint8)
    meta = torch.zeros((1, w, 9), dtype=torch.float32)
    meta[0, :, 7] = torch.tensor([i % 3 for i in range(w)], dtype=torch.float32)
    meta[0, :, 1] = torch.tensor([i % 6 for i in range(w)], dtype=torch.float32)
    meta[0, :, 2] = torch.linspace(-1, 1, w)
    mask = torch.ones((1, w), dtype=torch.bool)
    if not flip_pins:
        return plain_receipt(a, t0, profile, folds, side_pin, models, CI, img, meta, mask, manifest, np, torch, timm)
    img_m, meta_m = CI.mirror_windows(img, meta)
    ours_img, ours_meta = R6F.mirror_inputs(img, meta)
    R6F.require(torch.equal(img_m, ours_img) and torch.equal(meta_m, ours_meta) and not torch.equal(img_m, img), 'FLIP_INPUT_TRANSFORM')
    with torch.inference_mode():
        z, sl, bl = models[0](img, meta, mask, chunk=32, return_aux=True)
        zm, slm, blm = models[0](img_m, meta_m, mask, chunk=32, return_aux=True)
    mcl = R6F.LABELS.index('MCL')
    source_zt = 0.5 * (z[0] + zm[0][ci_perm])
    source_zt[mcl] = z[0][mcl]
    ours_zt = R6F.deployed_logits([z[0]], [zm[0]], perm, mcl)[0]
    R6F.require(torch.equal(source_zt, ours_zt), 'FLIP_DEPLOYED_LOGITS')
    R6F.require(torch.equal(CI.state_readout(models[0], sl, bl), R6F.state_readout(folds[0]['state_k'], sl, bl)), 'STATE_READOUT')
    layout = CI.state_layout(spec, R6F.LABELS)
    from knee import public_r7_stackers as stackers
    R6F.require(stackers.state_layout(spec, R6F.LABELS) == layout, 'PORTED_STATE_LAYOUT')
    mcl_cols = [i for c_ in layout['columns'] if c_['class'] == 'MCL'
                for i in (range(c_['start'], c_['stop']) if 'categories' in c_ else [c_['column']])]
    R6F.require(mcl_cols == R6F.mcl_state_columns(layout), 'MCL_STATE_COLUMNS')
    st = CI.state_readout(models[0], sl, bl)
    stm = CI.state_readout(models[0], slm[:, ci_perm], blm[:, ci_bin])
    source_st = 0.5 * (st[0] + stm[0])
    source_st[mcl_cols] = st[0][mcl_cols]
    ours_st = R6F.deployed_state(folds[0]['state_k'], [(sl, bl)], [(slm, blm)], perm, bin_perm, mcl_cols)
    R6F.require(torch.equal(source_st.double(), ours_st), 'FLIP_DEPLOYED_STATE')
    receipt = {'status': 'PASS_ACTUAL_CHECKPOINT_CPU_SYNTHETIC_INPUTS', 'profile': profile, 'folds': folds, 'side': side_pin,
               'flip': {'perm': perm, 'bin_perm': bin_perm, 'equals_pinned_compact_infer': True},
               'synthetic_forward': {'windows': w, 'fold': manifest['models'][0], 'mirror_changes_logits': bool((z - zm).abs().max() > 0),
                                     'max_abs_plain_mirror_dlogit': float((z - zm).abs().max()), 'deployed_logits_equal_source': True,
                                     'deployed_state_equal_source': True},
               'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'timm': timm.__version__,
                               'numpy': np.__version__, 'device': 'cpu'},
               'limits': ['CPU fp32 eager only: no CUDA, autocast, torch.compile, DataLoader workers or real DICOM input',
                          'anatomy V-Net not available locally: bound by file hash and runtime cross-board digest only'],
               'seconds': round(time.time() - t0, 1), 'gpu_exercised': False, 'real_images_used': False}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'status': receipt['status'], 'seconds': receipt['seconds'], 'out': str(out)}))
    return 0


def plain_receipt(a, t0, profile, folds, side_pin, models, CI, img, meta, mask, manifest, np, torch, timm):
    """R7 plain pass: the deployed state read-out of every fold model on synthetic windows equals the pinned source's."""
    with torch.inference_mode():
        z, sl, bl = models[0](img, meta, mask, chunk=32, return_aux=True)
    R6F.require(torch.isfinite(z).all().item(), 'FINITE_PLAIN_LOGITS')
    R6F.require(torch.equal(CI.state_readout(models[0], sl, bl), R6F.state_readout(folds[0]['state_k'], sl, bl)), 'STATE_READOUT')
    receipt = {'status': 'PASS_ACTUAL_CHECKPOINT_CPU_SYNTHETIC_INPUTS', 'profile': profile, 'folds': folds, 'side': side_pin,
               'flip': None, 'synthetic_forward': {'windows': int(img.shape[1]), 'fold': manifest['models'][0], 'pass': 'plain only',
                                                   'state_readout_equal_source': True},
               'environment': {'python': sys.version.split()[0], 'torch': torch.__version__, 'timm': timm.__version__,
                               'numpy': np.__version__, 'device': 'cpu'},
               'limits': ['CPU fp32 eager only: no CUDA, autocast, torch.compile, DataLoader workers or real DICOM input',
                          'anatomy V-Net not available locally: bound by file hash and runtime cross-board digest only'],
               'seconds': round(time.time() - t0, 1), 'gpu_exercised': False, 'real_images_used': False}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'status': receipt['status'], 'seconds': receipt['seconds'], 'out': str(out)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
