"""Isolated WS2 native224 primitives; no reader, entry, fallback or source execution.

The qualification host supplies already audited Raptor definitions. Callers must
pass the guarded384 volume before any old320 crop and propagate every error.
CPU windows support synthetic checks; production inference requires CUDA fp16.
"""
import hashlib
from pathlib import Path

from knee.public_s6_qualification import LABELS

DATASET = 'nartaa/rsna-knee-publication-swa-weights-20261007'
DATASET_VERSION = 1
CHECKPOINT_NAME = 'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt'
CHECKPOINT_SHA256 = '3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6'
CHECKPOINT_BYTES = 292842755
ARCH = 'coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k'
RECIPE = {'image_size': 384, 'depth': 96, 'windows': 94, 'input_res': 224,
          'crop_frac': .8, 'crop_size': 307, 'crop_offset': 38,
          'arch': ARCH, 'normalization': 'imagenet', 'tta': 'none',
          'amp': 'CUDA fp16; no fp32 rescue', 'resize': 'bilinear; align_corners=False'}


def verify_checkpoint_file(path):
    path = Path(path)
    if path.name != CHECKPOINT_NAME or not path.is_file() or path.stat().st_size != CHECKPOINT_BYTES:
        raise ValueError('WS2_CHECKPOINT_FILE')
    with path.open('rb') as handle:
        observed = hashlib.file_digest(handle, 'sha256').hexdigest()
    if observed != CHECKPOINT_SHA256:
        raise ValueError('WS2_CHECKPOINT_HASH')
    return observed


def state_digest(model):
    """Exact model tensor digest, including buffers and integer tensor bytes."""
    from knee.public_adaptation_pilot import state_fingerprint
    return state_fingerprint(model.state_dict())


def require_eval(model):
    if any(module.training for module in model.modules()):
        raise ValueError('WS2_MODEL_EVAL_REQUIRED')


def validate_checkpoint(ck, torch):
    if not isinstance(ck, dict) or ck.get('arch') != ARCH or type(ck.get('res')) is not int or ck['res'] != 224:
        raise ValueError('WS2_CHECKPOINT_ARCH_RES')
    if ck.get('lab') != LABELS:
        raise ValueError('WS2_CHECKPOINT_TARGET_ORDER')
    state = ck.get('model')
    if not isinstance(state, dict) or not state or not all(
            isinstance(k, str) and torch.is_tensor(v) and torch.isfinite(v).all().item()
            for k, v in state.items()):
        raise ValueError('WS2_CHECKPOINT_FINITE_STATE')
    return state


def load_model(public, path, device):
    """Hash first; restricted deserialization and strict native224 state loading.

    No unreviewed public top-level code, pretrained download or384-grid adaptation
    is called. Missing checkpoint metadata fails qualification rather than inheriting
    public fallback defaults. The caller owns audited-definition/source bindings.
    """
    verify_checkpoint_file(path)
    import torch
    if list(public.LAB) != LABELS:
        raise ValueError('WS2_PUBLIC_TARGET_ORDER')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    state = validate_checkpoint(ck, torch)
    import timm
    # The frozen 384 source constructor supplies its own img_size. Native 224 must
    # be constructed explicitly; interpolating a 384 attention grid is forbidden.
    backbone = timm.create_model(ARCH, pretrained=False, in_chans=3,
                                 num_classes=0, global_pool='avg')
    model = public.RaptorClassifier(backbone, F_dim=backbone.num_features)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    require_eval(model)
    return model, {'checkpoint_sha256': CHECKPOINT_SHA256, 'arch': ARCH, 'res': 224,
                   'labels': list(LABELS), 'state_sha256': state_digest(model),
                   'device': str(device), 'weights_only': True, 'strict': True}


def validate_volume(volume, mask):
    import numpy as np
    v, m = np.asarray(volume), np.asarray(mask)
    if v.shape != (96, 384, 384) or v.dtype != np.uint8:
        raise ValueError('WS2_GUARDED384_VOLUME_REQUIRED')
    if m.shape != (96,) or m.dtype not in (np.dtype('uint8'), np.dtype('bool')) or not np.isin(m, [0, 1]).all():
        raise ValueError('WS2_MASK_SCHEMA')
    nonzero = v.reshape(96, -1).any(axis=1)
    if not np.array_equal(nonzero, m.astype(bool)) or not nonzero.any():
        raise ValueError('WS2_MASK_VOLUME_MISMATCH_OR_EMPTY')
    return v, m


def eval_centers(mask):
    """Public K94 selection, including repeated positions and cross-slot windows."""
    import numpy as np
    m = np.asarray(mask)
    if m.shape != (96,) or m.dtype not in (np.dtype('uint8'), np.dtype('bool')) or not np.isin(m, [0, 1]).all():
        raise ValueError('WS2_MASK_SCHEMA')
    valid = np.where(m > 0)[0]
    if len(valid) < 3:
        valid = np.arange(3)
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, 94))]
    idx = np.linspace(0, len(cs) - 1, 94).round().astype(int)
    return [max(1, min(cs[i], 94)) for i in idx]


def center_crop(tensor):
    """Exact public round(.8*H/W), floor-centred crop; canonical384 becomes307."""
    h, w = tensor.shape[-2:]
    if (h, w) != (384, 384):
        raise ValueError('WS2_CROP_REQUIRES384')
    ch, cw = int(round(h * .8)), int(round(w * .8))
    y, x = (h - ch) // 2, (w - cw) // 2
    return tensor[..., y:y + ch, x:x + cw]


def windows(volume, mask, device='cpu'):
    """K94 plain triplets: uint8 LUT -> crop -> resize -> ImageNet normalization.

    CPU-computed float32 LUT mirrors the public GPU path exactly. Resize executes
    on the requested device; CPU/GPU interpolation parity needs host qualification.
    Inputs are read-only and are not globally cached.
    """
    import numpy as np
    import torch
    from torch.nn import functional as F
    v, m = validate_volume(volume, mask)
    cs = np.asarray(eval_centers(m))
    tri = torch.from_numpy(np.ascontiguousarray(v[np.stack([cs - 1, cs, cs + 1], 1)])).to(device)
    lut = torch.from_numpy(np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0).to(device)
    x = F.interpolate(center_crop(lut[tri.long()]), size=(224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([.485, .456, .406], dtype=torch.float32, device=device).view(3, 1, 1)
    std = torch.tensor([.229, .224, .225], dtype=torch.float32, device=device).view(3, 1, 1)
    return (x - mean) / std


def infer_probs(model, xwins, device, *, expected_state):
    """One CUDAfp16 view; no rescue. Enforce immutable eval state before/after."""
    import torch
    dev = torch.device(device)
    if dev.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('WS2_CUDA_REQUIRED')
    if (tuple(xwins.shape) != (94, 3, 224, 224) or xwins.dtype != torch.float32
            or xwins.device.type != 'cuda' or xwins.device != torch.empty(0, device=dev).device
            or not torch.isfinite(xwins).all().item()):
        raise ValueError('WS2_WINDOWS_SCHEMA')
    require_eval(model)
    if state_digest(model) != expected_state:
        raise ValueError('WS2_MODEL_STATE_BEFORE')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        p = torch.sigmoid(model(xwins.unsqueeze(0)).float())
    if p.shape != (1, 12) or not torch.isfinite(p).all().item() or not ((p >= 0) & (p <= 1)).all().item():
        raise ValueError('WS2_FINITE_PROBABILITIES')
    require_eval(model)
    if state_digest(model) != expected_state:
        raise ValueError('WS2_MODEL_STATE_AFTER')
    return p[0].cpu().numpy()
