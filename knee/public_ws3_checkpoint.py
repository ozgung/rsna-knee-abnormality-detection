"""Prospective native224 anatomical mirror; WS2 primitives stay byte-identical.

Mirror raw uint8 triplets through the pinned public _mirror_tri before per-channel
normalization. Reversing already-normalized sagittal channels is incorrect.
"""
from knee import public_ws2_checkpoint as BASE
from knee.public_ws2_checkpoint import (DATASET, DATASET_VERSION, CHECKPOINT_NAME,
    CHECKPOINT_SHA256, CHECKPOINT_BYTES, ARCH, verify_checkpoint_file, state_digest,
    require_eval, validate_checkpoint, load_model, validate_volume, eval_centers,
    center_crop, infer_probs)

RECIPE={**BASE.RECIPE,'tta':'anatomical mirror; probability mean',
    'mirror':'pinned public _mirror_tri on uint8 before LUT and ImageNet normalization'}


def windows(volume,mask,device='cpu',*,public=None,mirrored=False):
    if not mirrored:return BASE.windows(volume,mask,device)
    import numpy as np
    import torch
    from torch.nn import functional as F
    v,m=validate_volume(volume,mask)
    if public is None or not callable(getattr(public,'_mirror_tri',None)):
        raise ValueError('WS3_PINNED_PUBLIC_MIRROR_REQUIRED')
    if list(public.SLOT_SIZES)!=[26,22,18,12,18] or tuple(public.SAGITTAL_SLOTS)!=(0,1):
        raise ValueError('WS3_PUBLIC_ANATOMICAL_SLOT_SCHEMA')
    cs=np.asarray(eval_centers(m))
    tri=torch.from_numpy(np.ascontiguousarray(v[np.stack([cs-1,cs,cs+1],1)])).to(device)
    tri=public._mirror_tri(tri,cs,96)
    if tri.shape!=(94,3,384,384) or tri.dtype!=torch.uint8:
        raise ValueError('WS3_PUBLIC_MIRROR_TRIPLET_SCHEMA')
    lut=torch.from_numpy(np.arange(256,dtype=np.uint8).astype(np.float32)/255.).to(device)
    x=F.interpolate(center_crop(lut[tri.long()]),size=(224,224),mode='bilinear',align_corners=False)
    mean=torch.tensor([.485,.456,.406],dtype=torch.float32,device=device).view(3,1,1)
    std=torch.tensor([.229,.224,.225],dtype=torch.float32,device=device).view(3,1,1)
    return (x-mean)/std


def average_probabilities(normal,mirrored):
    import numpy as np
    a,b=np.asarray(normal),np.asarray(mirrored)
    if (a.shape!=(12,) or b.shape!=(12,) or a.dtype!=np.float32 or b.dtype!=np.float32
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not ((a>=0)&(a<=1)).all() or not ((b>=0)&(b<=1)).all()):
        raise ValueError('WS3_FINITE_FLOAT32_VIEW_PROBABILITIES')
    return (a+b)*np.float32(.5)


def validate_view_probabilities(views,blended,*,mirror_enabled):
    import numpy as np
    if not isinstance(views,dict) or set(views)!={'normal','mirrored'}:
        raise ValueError('WS3_VIEW_EVIDENCE_SCHEMA')
    a=np.asarray(views['normal'],dtype=np.float32)
    if a.shape!=(12,) or not np.isfinite(a).all() or not ((a>=0)&(a<=1)).all():
        raise ValueError('WS3_VIEW_EVIDENCE_PROBABILITIES')
    if mirror_enabled:
        b=np.asarray(views['mirrored'],dtype=np.float32)
        expected=average_probabilities(a,b)
    else:
        if views['mirrored'] is not None:raise ValueError('WS3_UNREQUESTED_MIRROR')
        expected=a
    p=np.asarray(blended,dtype=np.float32)
    if p.shape!=(12,) or not np.isfinite(p).all() or not np.allclose(p,expected,rtol=0,atol=1e-6):
        raise ValueError('WS3_VIEW_PROBABILITY_MEAN')
    return True
