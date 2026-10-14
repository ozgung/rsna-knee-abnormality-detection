"""Raptor64 (Dread widedense v4 contract) fold-0 execution safeguards.

The historical trainer's sampler, model, optimizer, OneCycle schedule, loss, report-masked
top-3 selection and SWA are retained (AST-checked by the package builder). This module adds:
  * the verified lazy 64-slot cache (`knee.raptor64_loader`),
  * a pinned initializer descriptor per arm: ImageNet (`imagenet`) or Dread CC0 v4_full
    backbone (`dread_v4`, backbone tensors only, missing=0 and unexpected=0 enforced),
  * the head variant per arm: `original` (public RaptorClassifier) or `plane` (plane-aware
    head: per-window plane x Fluid_Sensitive and normalized depth embeddings concatenated to
    the window feature before the per-finding attention logits and the classifier),
  * resume identity/RNG and fold-0 isolation, as in the qualified source96 package.
The pinned public trainer source is never modified in place; variants are subclasses.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import time

from knee import raptor64_loader as loader, raptor64_cache as cache, public_training_smoke as smoke, source96_cache
from knee.source96_training import (LABEL_SHA, TRUTH_SHA, FOLDS_SHA, EXCLUSIONS_SHA, SPLITS, checked_json, strict_cuda,
                                    tensor_hash, restore_rng, experiment_split)

ARCH = 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
IMAGENET_PIN = {'repo': 'timm/' + ARCH, 'revision': '2c31d82376cc244c907c064332941e6c65be81f1', 'file': 'model.safetensors',
                'sha256': '91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e', 'size': 295695280}
DREAD_PIN = {'dataset': 'dreaddevelopment/raptor-knee-widedense', 'file': 'raptor_ft_coatnet_v4_full.pt',
             'sha256': cache.V4_CHECKPOINT_SHA256, 'size': 292829402, 'licence': 'CC0-1.0',
             'exposure': 'Dread trained on all non-gold studies; fold-0 OOF is exposed'}
# Optional, not launched without coordinator confirmation: nartaa's OAI-pretrained SWA Raptor (OAI rules unresolved).
NARTAA_PIN = {'dataset': 'nartaa/rsna-knee-publication-swa-weights-20261007', 'file': 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt',
              'sha256': '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', 'size': 292840787,
              'licence': 'per dataset card (verify before use)',
              'exposure': 'nartaa trained on all competition studies plus OAI; fold-0 OOF is exposed; OAI host ruling unresolved'}
HEAD_KEYS = frozenset({'norm.weight', 'norm.bias', 'att.0.weight', 'att.0.bias', 'att.3.weight', 'att.3.bias', 'clsW', 'clsb'})
LAB = list(smoke.TARGETS)
INITS = ('imagenet', 'dread_v4', 'nartaa_oai')
HEADS = ('original', 'plane')
ARMS = {'in-orig': ('imagenet', 'original'), 'in-plane': ('imagenet', 'plane'),
        'dr-orig': ('dread_v4', 'original'), 'dr-plane': ('dread_v4', 'plane'), 'na-orig': ('nartaa_oai', 'original'),
        'dr-full': ('dread_v4', 'original'), 'in-mirror': ('imagenet', 'original'), 'in-distil': ('imagenet', 'original'),
        'in-distil-w1': ('imagenet', 'original'), 'in-distil-flip': ('imagenet', 'original')}
# Original-methods iteration 2: single-factor recipe variants against in-orig (everything else frozen).
VARIANTS = {'in-mirror': 'mirror', 'in-distil': 'distil', 'in-distil-w1': 'distil', 'in-distil-flip': 'distil_flip'}
# Iteration 3 (single factors against in-distil): teacher weight per arm; label-preserving flip spec.
DISTIL_WEIGHTS = {'in-distil': 0.5, 'in-distil-w1': 1.0, 'in-distil-flip': 0.5}
FLIP_SPEC = {'p': 0.5, 'perm': list(range(12)), 'masked': [], 'tta_export': False}
MIRROR_P = 0.5
DISTIL_WEIGHT = 0.5
SAG_SLOTS = ((0, 18), (18, 32))          # sagittal FS / non-FS slot ranges in the 64-slice volume
WIDTH_FLIP = (32, 64)                    # coronal + axial slots: flip image width (patient L/R)
SIDE_PERM = [0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10, 11]   # swap Medial<->Lateral Meniscus and Medial<->Lateral OA (an involution)
MCL_INDEX = 1                            # no lateral counterpart labelled: masked on mirrored samples
# Full-data mode: all 4,345 eligible studies train; no held-out set, so top-3 SWA uses pre-registered
# fixed epochs. File tags stay "fold0" for persister/runner compatibility only.
FULL_ARMS = frozenset({'dr-full'})
ARM_ORDER = ('in-orig', 'in-plane', 'dr-orig', 'dr-plane')
EPOCHS = 12
EXPORT_K = 42
FROZEN = {'arch': ARCH, 'res': 384, 'k': 8, 'k_eval': 16, 'epochs': EPOCHS, 'bs': 4, 'accum': 2,
          'win_chunk': 8, 'topk': 3, 'seed': 42, 'norm': 'imagenet', 'grad_ckpt': True,
          'fp16': True, 'xla': False, 'smoke': False, 'roi': False, 'freeze_blocks': 0, 'limit': 0,
          'folds': 5, 'fold': 0, 'time_budget_h': 0., 'ckpt': 'pinned-descriptor', 'bb_lr': 3e-5, 'head_lr': 1e-3, 'wd': .02}
FROZEN_FULL = {**FROZEN, 'folds': 0, 'fold': -1}
FULL_SPLIT = {'fold': None, 'eligible': 4345, 'training': 4345, 'development': 0,
              'ordered_eligible_sha256': SPLITS[0]['ordered_eligible_sha256'],
              'ordered_training_sha256': SPLITS[0]['ordered_eligible_sha256'],
              'ordered_development_sha256': smoke.digest(smoke.canonical([]))}
PLANE_CODES = 6
CODE_DIM = 16
DEPTH_DIM = 8
ENV = 'RAPTOR64_'
_CORPUS = None
_META = None
_CACHE_IDENTITY = None


def arm_slug(arm):
    if arm not in ARMS:
        raise ValueError('Unknown Raptor64 arm')
    return arm


# ----------------------------------------------------------------------------- staged corpus
def staged_corpus(*, allow_fixture=False):
    global _CORPUS, _META, _CACHE_IDENTITY
    if _CORPUS is not None:
        if _CACHE_IDENTITY.get('fixture') and not allow_fixture:
            raise ValueError('Fixture corpus cannot train production')
        return _CORPUS
    root = Path(os.environ[ENV + 'STAGE_ROOT'])
    expected = os.environ[ENV + 'MANIFEST_SHA256']; content = os.environ[ENV + 'CONTENT_SHA256']
    receipt = checked_json(root / 'staging-receipt.json', os.environ[ENV + 'STAGE_RECEIPT_SHA256'])
    if receipt.get('status') != 'PASS' or receipt.get('manifest_sha256') != expected or receipt.get('content_sha256') != content:
        raise ValueError('Trusted staging PASS required')
    manifest = checked_json(root / 'cache64-manifest-private.json', receipt['manifest_file_sha256'])
    if manifest.get('content_sha256') != content:
        raise ValueError('Verified image/mask identity differs')
    vols, masks, ids = loader.open_corpus(manifest, expected_manifest_sha256=expected, allow_fixture=allow_fixture)
    _CORPUS = (vols, masks, ids)
    _META = loader.slice_metadata_table(manifest)
    _CACHE_IDENTITY = {'content_sha256': content, 'membership': manifest['membership'], 'fixture': manifest['fixture'],
                       'sources': [{k: v for k, v in s.items() if k != 'root'} for s in manifest['sources']],
                       'manifest_sha256_observed': expected}
    return _CORPUS


def slice_meta(row):
    if _META is None:
        staged_corpus()
    return _META[0][row], _META[1][row]


# ----------------------------------------------------------------------------- recipe/policy
def recipe(script):
    root = Path(script).parent
    manifest = json.loads((root / 'manifest.json').read_bytes())
    for name, sha in manifest['sources'].items():
        if smoke.file_sha(root / name) != sha:
            raise ValueError('Packaged runtime module differs')
    policy = checked_json(root / 'raptor64-recipe.json', manifest['recipe_sha256'])
    validate_policy(policy)
    return policy


def full_mode(policy):
    return policy.get('mode') == 'full'


def validate_policy(policy):
    arm = policy.get('arm'); full = arm in FULL_ARMS
    split = FULL_SPLIT if full else SPLITS[0]
    if (policy.get('version') != 1 or arm not in ARMS or (policy.get('init'), policy.get('head')) != ARMS[arm] or
            policy.get('mode', 'fold0') != ('full' if full else 'fold0') or
            policy.get('frozen') != (FROZEN_FULL if full else FROZEN) or policy.get('export_k') != EXPORT_K or policy.get('experiment_split') != split or
            policy.get('experiment_split_sha256') != smoke.digest(smoke.canonical(split))):
        raise ValueError('Pinned Raptor64 policy differs')
    variant = VARIANTS.get(arm)
    if policy.get('variant') != variant:
        raise ValueError('Recipe variant differs from the arm')
    if variant == 'mirror' and policy.get('mirror') != {'p': MIRROR_P, 'perm': SIDE_PERM, 'masked': [MCL_INDEX], 'tta_export': True}:
        raise ValueError('Mirror recipe differs from the pre-registered one')
    if variant == 'distil_flip' and policy.get('flip') != FLIP_SPEC:
        raise ValueError('Label-preserving flip recipe differs from the pre-registered one')
    if variant in ('distil', 'distil_flip'):
        teacher = policy.get('teacher') or {}
        if teacher.get('weight') != DISTIL_WEIGHTS[arm] or not loader._hash(teacher.get('sha256')) or teacher.get('source') != 'dread_v4_full_k42':
            raise ValueError('Distillation teacher pin differs from the pre-registered one')
    if full:
        epochs = policy.get('swa_epochs')
        if (not isinstance(epochs, list) or len(epochs) != 3 or len(set(epochs)) != 3 or
                any(type(e) is not int or not 0 <= e < EPOCHS for e in epochs)):
            raise ValueError('Full-data mode requires three distinct pre-registered SWA epochs')
    return policy


def full_selection_score(ep, script):
    """Indicator of the pre-registered SWA epochs; historical top-3 rule then keeps exactly those epochs."""
    policy = recipe(script)
    if not full_mode(policy):
        raise ValueError('Fixed-epoch selection is full-data mode only')
    return 1.0 if ep in policy['swa_epochs'] else 0.0


def require_recipe(args, script):
    policy = recipe(script)
    for key, value in policy['frozen'].items():
        if getattr(args, key, None) != value:
            raise ValueError('Frozen training recipe differs: ' + key)
    if args.workers != policy['workers'] or (args.workers != 2 and not policy.get('worker_choice_name')):
        raise ValueError('Unnamed worker override forbidden')
    if args.tag != 'fold0':
        raise ValueError('Fold-0 tag required')
    return policy


# ----------------------------------------------------------------------------- initializers
def _descriptor():
    return checked_json(os.environ[ENV + 'INITIALIZER_DESCRIPTOR'], os.environ[ENV + 'INITIALIZER_DESCRIPTOR_SHA256'])


def _finite(torch, state):
    return all(torch.is_tensor(v) and (not v.is_floating_point() or bool(torch.isfinite(v).all())) for v in state.values())


def load_imagenet(backbone, d, torch):
    if (d.get('kind') != 'imagenet_backbone' or d.get('arch') != ARCH or d.get('hf_repo') != IMAGENET_PIN['repo'] or
            d.get('hf_revision') != IMAGENET_PIN['revision'] or d.get('origin_file_sha256') != IMAGENET_PIN['sha256'] or
            not loader._hash(d.get('weights_sha256'))):
        raise ValueError('Verified pinned ImageNet-384 provenance required')
    path = Path(d['weights_file'])
    if path.is_symlink() or smoke.file_sha(path) != d['weights_sha256']:
        raise ValueError('ImageNet initializer file differs')
    state = torch.load(path, map_location='cpu', weights_only=True)
    expected = backbone.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected) or not _finite(torch, state):
        raise ValueError('Initializer must exactly cover backbone keys with finite tensors')
    if any(v.shape != expected[k].shape or v.dtype != expected[k].dtype for k, v in state.items()):
        raise ValueError('Initializer tensor schema differs')
    backbone.load_state_dict(state, strict=True)
    return {'kind': 'imagenet_backbone', 'file_sha256': d['weights_sha256'], 'origin_file_sha256': d['origin_file_sha256'],
            'hf_repo': d['hf_repo'], 'hf_revision': d['hf_revision'], 'backbone_state_sha256': tensor_hash(state),
            'loaded_tensors': len(state), 'missing': 0, 'unexpected': 0}


def load_published(backbone, d, torch, pin, kind, *, require_lab=True):
    """train_knee.py --ckpt semantics (backbone.* prefix, strict=False) with exact coverage enforced."""
    if d.get('kind') != kind or d.get('weights_sha256') != pin['sha256'] or d.get('dataset') != pin['dataset']:
        raise ValueError(f'Verified pinned {kind} provenance required')
    path = Path(d['weights_file'])
    if path.is_symlink() or path.stat().st_size != pin['size'] or smoke.file_sha(path) != pin['sha256']:
        raise ValueError(f'{kind} checkpoint bytes differ')
    ck = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(ck, dict) or ck.get('arch') != ARCH or int(ck.get('res', 0)) != 384:
        raise ValueError('Checkpoint arch/resolution differs')
    if (require_lab or 'lab' in ck) and list(ck.get('lab', [])) != LAB:
        raise ValueError('Checkpoint target order differs')
    state = ck['model']
    if not isinstance(state, dict) or not _finite(torch, state):
        raise ValueError('Checkpoint tensors invalid or nonfinite')
    head = {k for k in state if not k.startswith('backbone.')}
    if head != HEAD_KEYS:
        raise ValueError('Checkpoint non-backbone inventory differs')
    bbst = {k[len('backbone.'):]: v for k, v in state.items() if k.startswith('backbone.')}
    missing, unexpected = backbone.load_state_dict(bbst, strict=False)
    if len(missing) != 0 or len(unexpected) != 0:
        raise ValueError(f'Backbone load incomplete: missing={len(missing)} unexpected={len(unexpected)}')
    loaded = backbone.state_dict()
    if tensor_hash({k: loaded[k] for k in bbst}) != tensor_hash(bbst):
        raise ValueError('Backbone tensors differ after load')
    return {'kind': kind, 'file_sha256': pin['sha256'], 'dataset': pin['dataset'], 'licence': pin['licence'],
            'backbone_state_sha256': tensor_hash(bbst), 'loaded_tensors': len(bbst), 'missing': 0, 'unexpected': 0,
            'head': 'fresh (train_knee.py --ckpt loads backbone only)', 'exposure': pin['exposure']}


def load_dread(backbone, d, torch):
    return load_published(backbone, d, torch, DREAD_PIN, 'dread_v4_full')


def load_nartaa(backbone, d, torch):
    return load_published(backbone, d, torch, NARTAA_PIN, 'nartaa_oai_swa', require_lab=False)


def initialize_backbone(backbone, script=None):
    import torch
    d = _descriptor()
    init = d.get('init')
    if script is not None and recipe(script)['init'] != init:
        raise ValueError('Initializer descriptor does not match the arm policy')
    if init == 'imagenet':
        result = load_imagenet(backbone, d, torch)
    elif init == 'dread_v4':
        result = load_dread(backbone, d, torch)
    elif init == 'nartaa_oai':
        result = load_nartaa(backbone, d, torch)
    else:
        raise ValueError('Unknown initializer')
    return {**result, 'init': init, 'descriptor_sha256': os.environ[ENV + 'INITIALIZER_DESCRIPTOR_SHA256']}


def initial_record(model, initializer, script):
    import torch, timm
    record = {**initializer, 'full_initial_model_sha256': tensor_hash(model.state_dict()),
              'torch': torch.__version__, 'timm': timm.__version__, 'device': torch.cuda.get_device_name(0),
              'worker_rng_resume': 'Main RNG restored; persistent worker RNG/prefetch not restored; no bit-identical continuation claim'}
    source96_cache.atomic_json(Path(script).parent / 'initial-state-record.json', record)
    return record


# ----------------------------------------------------------------------------- plane-aware head
class WindowBatch:
    """Windows plus per-window metadata; moves/pins together so the historical loop is unchanged."""
    def __init__(self, x, meta):
        self.x, self.meta = x, meta

    def to(self, *args, **kwargs):
        return WindowBatch(self.x.to(*args, **kwargs), self.meta.to(*args, **kwargs))

    def pin_memory(self):
        return WindowBatch(self.x.pin_memory(), self.meta.pin_memory())

    @property
    def shape(self):
        return self.x.shape


def eval_centers(mask, D, k):
    """Deterministic evaluation centres (identical to StudyWindows eval and public _eval_centers)."""
    import numpy as np
    valid = np.where(np.asarray(mask) > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def inference_window_metadata(slots, mask, k):
    """Image-only inference path: slot records from the reader, mask from the volume."""
    codes, depth = cache.slice_metadata(slots)
    return cache.window_metadata(codes, depth, eval_centers(mask, len(codes), k))


def head_variant(StudyWindows, RaptorClassifier, collate, script):
    """Return (dataset, model, collate) for the arm's head; `original` returns the inputs unchanged."""
    head = recipe(script)['head']
    if head == 'original':
        return StudyWindows, RaptorClassifier, collate
    if head != 'plane':
        raise ValueError('Unknown head variant')
    return plane_variant(StudyWindows, RaptorClassifier)


def plane_variant(StudyWindows, RaptorClassifier):
    import torch
    from torch import nn

    class PlaneStudyWindows(StudyWindows):
        def _centers(self, valid, count):
            centers = super()._centers(valid, count)
            self._raptor64_centers = list(centers)  # same RNG consumption as the original sampler
            return centers

        def __getitem__(self, i):
            if self.roi:
                raise ValueError('ROI windows are not supported by the plane-aware head')
            x, y = super().__getitem__(i)
            codes, depth = slice_meta(self.id2row[self.ids[i]])
            meta = torch.from_numpy(cache.window_metadata(codes, depth, self._raptor64_centers))
            if meta.shape[0] != x.shape[0]:
                raise ValueError('Window metadata count differs')
            return WindowBatch(x, meta), y

    class PlaneAwareClassifier(RaptorClassifier):
        def __init__(self, backbone, F_dim=384, n=12, drop=0.2):
            super().__init__(backbone, F_dim=F_dim, n=n, drop=drop)
            width = F_dim + CODE_DIM + DEPTH_DIM
            self.code_embed = nn.Embedding(PLANE_CODES, CODE_DIM)
            self.depth_embed = nn.Linear(1, DEPTH_DIM)
            self.att = nn.Sequential(nn.Linear(width, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
            self.clsW = nn.Parameter(torch.zeros(n, width))
            nn.init.trunc_normal_(self.clsW, std=0.02)

        def head_meta(self, feats, meta):
            h = self.norm(feats)
            code = meta[..., 0].round().long()
            if bool((code < 0).any()) or bool((code >= PLANE_CODES).any()):
                raise ValueError('Plane code out of range')
            e = torch.cat([self.code_embed(code), self.depth_embed(meta[..., 1:2].float())], -1).to(h.dtype)
            z = torch.cat([h, e], -1)
            a = torch.softmax(self.att(z), dim=1)
            pooled = torch.einsum('bkn,bkf->bnf', a, z)
            return (pooled * self.clsW).sum(-1) + self.clsb

        def forward(self, x):
            if not isinstance(x, WindowBatch):
                raise TypeError('Plane-aware head requires window metadata')
            return self.head_meta(self.encode(x.x), x.meta)

    def plane_collate(batch):
        xs = torch.stack([b[0].x for b in batch]); metas = torch.stack([b[0].meta for b in batch])
        ys = torch.stack([b[1] for b in batch])
        return WindowBatch(xs, metas), ys

    # Fork-started DataLoader workers (Linux default on Python 3.12) inherit these classes;
    # a spawn start method would fail loudly rather than silently change the head.
    return PlaneStudyWindows, PlaneAwareClassifier, plane_collate


# ----------------------------------------------------------------------------- identity/resume
def runtime_split(folds_path, excluded_path, script):
    policy = recipe(script)
    fmap = checked_json(folds_path, FOLDS_SHA)['folds']
    excluded = checked_json(excluded_path, EXCLUSIONS_SHA)['quarantined_study_ids']
    staged_corpus()
    if not full_mode(policy):
        return experiment_split(_CORPUS[2], fmap, excluded, 0)
    ids = list(map(str, _CORPUS[2]))
    if len(ids) != len(set(ids)) or set(excluded) & set(ids):
        raise ValueError('Eligible population/exclusion coverage differs')
    result = {'fold': None, 'eligible': len(ids), 'training': len(ids), 'development': 0,
              'ordered_eligible_sha256': smoke.digest(smoke.canonical(ids)), 'ordered_training_sha256': smoke.digest(smoke.canonical(ids)),
              'ordered_development_sha256': smoke.digest(smoke.canonical([]))}
    if result != policy['experiment_split']:
        raise ValueError('Full-data eligible population differs')
    return result, ids, []


def run_identity(args, script, labels, initial):
    import torch, timm, numpy
    staged_corpus()
    if (smoke.file_sha(labels) != LABEL_SHA or smoke.file_sha(args.oof_mask) != TRUTH_SHA or
            smoke.file_sha(args.fold_file) != FOLDS_SHA or smoke.file_sha(args.excluded_ids) != EXCLUSIONS_SHA):
        raise ValueError('Historical supervision/split pins differ')
    policy = recipe(script)
    body = {k: v for k, v in vars(args).items() if k not in ('labels', 'oof_mask', 'fold_file', 'excluded_ids', 'resume', 'roi_boxes')}
    return {'version': 1, 'recipe': policy['recipe'], 'arm': policy['arm'], 'init': policy['init'], 'head': policy['head'],
            'experiment_split': runtime_split(args.fold_file, args.excluded_ids, script)[0], 'fold': 0, 'n_folds': args.folds, 'tag': args.tag,
            'corpus_ids_sha256': {'raptor64_ordered_eligible': _CACHE_IDENTITY['membership']['ordered_eligible_sha256']},
            'epochs_requested': EPOCHS, 'script_sha256': smoke.file_sha(script), 'helper_sha256': smoke.file_sha(__file__),
            'loader_sha256': smoke.file_sha(loader.__file__), 'config': body, 'labels_sha256': LABEL_SHA, 'oof_mask_sha256': TRUTH_SHA,
            'fold_file_sha256': FOLDS_SHA, 'excluded_ids_sha256': EXCLUSIONS_SHA,
            'cache': {k: v for k, v in _CACHE_IDENTITY.items() if k != 'manifest_sha256_observed'}, 'initializer': initial,
            'policy': policy, 'environment': {'torch': torch.__version__, 'timm': timm.__version__, 'numpy': numpy.__version__,
                                              'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0)}}


def validate_resume(state, identity):
    required = ('model', 'opt', 'sched', 'scaler', 'best', 'best_state', 'best_P', 'hist', 'topk', 'epoch', 'rng', 'identity')
    if not isinstance(state, dict) or any(k not in state for k in required) or state['identity'] != identity:
        raise ValueError('Resume state/image/initializer/config identity differs')
    epoch = state['epoch']
    if type(epoch) is not int or not 0 <= epoch < EPOCHS or [h['ep'] for h in state['hist']] != list(range(epoch + 1)):
        raise ValueError('Resume epoch history incomplete')
    if (not state['topk'] or len(state['topk']) > 3 or len({t['epoch'] for t in state['topk']}) != len(state['topk']) or
            any(not 0 <= t['epoch'] <= epoch for t in state['topk'])):
        raise ValueError('Resume top3 state invalid')


def save_resume(path, state, identity):
    import random
    import numpy as np, torch
    state = {**state, 'identity': identity, 'rng': {'py': random.getstate(), 'np': np.random.get_state(), 'torch': torch.get_rng_state(),
             'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    validate_resume(state, identity)
    path = Path(path); tmp = path.with_name(path.name + '.tmp'); torch.save(state, tmp); tmp.replace(path)
    deadline = os.environ.get(ENV + 'ACTUAL_PLATFORM_DEADLINE_UTC_SECONDS')
    if deadline and time.time() + 120 >= float(deadline) and state['epoch'] < EPOCHS - 1:
        print('[resume] completed epoch checkpoint saved before supplied actual platform deadline', flush=True)
        raise SystemExit(0)


def verify_membership(train_ids, dev_ids, gold_ids, fold, folds_path, excluded_path, script):
    if fold != (-1 if full_mode(recipe(script)) else 0):
        raise ValueError('Raptor64 fold-0 or explicit full-data mode only')
    m = runtime_split(folds_path, excluded_path, script)[0]
    if (gold_ids or (len(train_ids), len(dev_ids)) != (m['training'], m['development']) or
            smoke.digest(smoke.canonical(train_ids)) != m['ordered_training_sha256'] or
            smoke.digest(smoke.canonical(dev_ids)) != m['ordered_development_sha256']):
        raise ValueError('Fixed eligible training/development membership differs')


def verify_export_inputs(labels, truth, folds, excluded, development_ids, fold, script):
    if fold != 0:
        raise ValueError('Raptor64 export is fold-0 only')
    staged_corpus()
    if any(smoke.file_sha(path) != sha for path, sha in ((labels, LABEL_SHA), (truth, TRUTH_SHA), (folds, FOLDS_SHA), (excluded, EXCLUSIONS_SHA))):
        raise ValueError('Export supervision/split pins differ')
    m = runtime_split(folds, excluded, script)[0]
    if len(development_ids) != m['development'] or smoke.digest(smoke.canonical(list(development_ids))) != m['ordered_development_sha256']:
        raise ValueError('Export development membership differs')


def entry_guard(c, script):
    policy = recipe(script)
    expected = {'ARCH': ARCH, 'RES': '384', 'EPOCHS': str(EPOCHS), 'CKPT': 'pinned-descriptor', 'BS': '4', 'K': '8', 'K_EVAL': '16',
                'SEED': '42', 'TOPK': '3', 'GRAD_CKPT': '1', 'WIN_CHUNK': '8', 'ACCUM': '2', 'LABELS_FILE': 'labels_v1_blend.parquet',
                'TIME_BUDGET_H': '0', 'FOLD_A': '0', 'FOLD_B': '', 'BB_LR': '3e-5', 'HEAD_LR': '1e-3', 'WD': '0.02', 'WORKERS': str(policy['workers'])}
    if c['SMOKE'] or any(str(c.get(k)) != v for k, v in expected.items()):
        raise ValueError('Frozen entry parameters differ')
    return policy


def export_guard(work, fold, checkpoint_path, kind, epochs, partial):
    if fold != 0:
        raise ValueError('Raptor64 export is fold-0 only')
    recipe(Path(work) / 'train_knee_patched.py')
    if partial or kind != 'swa_topk' or len(epochs) != 3 or len(set(epochs)) != 3:
        raise ValueError('Only completed frozen report-selected top3 SWA may export')
    meta = json.loads((Path(work) / 'raptor_ft_fold0.json').read_bytes())
    if meta.get('partial') or len(meta.get('hist', [])) != EPOCHS:
        raise ValueError('Twelve completed epochs required')


def full_record(work, k, entry_file):
    """Full-data export step: no predictions (no held-out); bind the fixed-epoch SWA checkpoint and history."""
    import torch
    work = Path(work); script = work / 'train_knee_patched.py'; policy = recipe(script)
    if not full_mode(policy) or k != 0:
        raise ValueError('Full-data record only for the full-data package')
    meta = json.loads((work / 'raptor_ft_fold0.json').read_bytes())
    if meta.get('partial') or [h['ep'] for h in meta.get('hist', [])] != list(range(EPOCHS)):
        raise ValueError('Twelve completed epochs required')
    swa = work / 'raptor_ft_fold0_swa.pt'
    state = torch.load(swa, map_location='cpu', weights_only=False)
    if sorted(state.get('swa_over', [])) != sorted(policy['swa_epochs']):
        raise ValueError('Full-data SWA epochs differ from the pre-registered epochs')
    tmeta = json.loads((work / 'train_meta_fold0.json').read_bytes())
    record = {'fold': 0, 'mode': 'full', 'tag': 'fold0', 'train_cmd': tmeta['cmd'], 'train_returncode': tmeta['rc'],
              'epochs_requested': EPOCHS, 'epochs_completed': len(meta['hist']), 'stopped_by_time_budget': False, 'history': meta['hist'],
              'selection': 'pre-registered fixed SWA epochs (no held-out set)', 'checkpoint_kind': 'swa_fixed_epochs',
              'selected_epochs': [int(e) for e in state['swa_over']], 'checkpoint_file': swa.name, 'checkpoint_sha256': smoke.file_sha(swa),
              'export': None, 'comparison_endpoint': 'none (full-data fit for an LB probe)', 'n_train': meta.get('n_train'),
              'raptor64_identity': json.loads((work / 'initial-state-record.json').read_bytes()), 'raptor64_policy': policy,
              'raptor64_experiment_split': runtime_split(work / 'folds_v0.json', work / 'excluded_ids.json', script)[0],
              'raptor64_cache_content_sha256': os.environ['RAPTOR64_CONTENT_SHA256'], 'entry_sha256': smoke.file_sha(entry_file),
              'caveat': 'No development estimate exists for a full-data fit; judge only by a separately rehearsed LB probe.'}
    source96_cache.atomic_json(work / 'run_fold0.json', record)
    print(f"[export fold0] FULL-DATA swa_fixed_epochs {record['selected_epochs']} | wrote run_fold0.json (no predictions)", flush=True)
    return record


# ----------------------------------------------------------------------------- iteration-2 variants
def mirror_volume(volume):
    """Anatomical L/R mirror of a 64-slice raptor64 volume (or its mask): reverse slice order inside each
    sagittal slot; flip image width in coronal and axial slots. An involution."""
    import numpy as np
    out = np.array(volume, copy=True)
    for a, b in SAG_SLOTS:
        out[a:b] = volume[a:b][::-1]
    if out.ndim == 3:
        out[WIDTH_FLIP[0]:WIDTH_FLIP[1]] = volume[WIDTH_FLIP[0]:WIDTH_FLIP[1]][..., ::-1]
    return out


def mirror_targets(y):
    return y[..., SIDE_PERM]


def recipe_variant(StudyWindows, script):
    """Dataset variant for the arm (None -> unchanged). Training targets become (B, 24): [targets, mask] for mirror,
    [report targets, teacher probabilities] for distil; evaluation items are unchanged."""
    import random
    import torch
    policy = recipe(script); variant = policy.get('variant')
    if variant is None:
        return StudyWindows
    class _View:
        def __init__(self, base, owner):
            self.base, self.owner = base, owner; self.shape = base.shape

        def __getitem__(self, row):
            value = self.base[row]
            return mirror_volume(value) if self.owner._mirror_now else value

    if variant in ('distil', 'distil_flip'):
        teacher = load_teacher(policy)
        Base = StudyWindows
        if variant == 'distil_flip':
            class FlipStudyWindows(StudyWindows):
                """Label-preserving anatomical mirror (p from the policy): image only; labels, masks, teacher unchanged."""
                force_mirror = False

                def __getitem__(self, i):
                    self._ensure()
                    if not isinstance(self.vols, _View):
                        self.vols, self.masks = _View(self.vols, self), _View(self.masks, self)
                    self._mirror_now = (random.random() < policy['flip']['p']) if self.train else bool(self.force_mirror)
                    return super().__getitem__(i)
            Base = FlipStudyWindows

        class DistilStudyWindows(Base):
            def __getitem__(self, i):
                x, y = super().__getitem__(i)
                if not self.train:
                    return x, y
                return x, torch.cat([y, torch.from_numpy(teacher[self.ids[i]])])
        return DistilStudyWindows

    class MirrorStudyWindows(StudyWindows):
        force_mirror = False   # evaluation-time TTA dataset only

        def __getitem__(self, i):
            self._ensure()
            if not isinstance(self.vols, _View):
                self.vols, self.masks = _View(self.vols, self), _View(self.masks, self)
            self._mirror_now = (random.random() < MIRROR_P) if self.train else bool(self.force_mirror)
            x, y = super().__getitem__(i)
            if not self.train:
                return x, y
            mask = torch.ones(12, dtype=y.dtype)
            if self._mirror_now:
                y = mirror_targets(y); mask[MCL_INDEX] = 0
            return x, torch.cat([y, mask])
    return MirrorStudyWindows


def load_teacher(policy):
    import numpy as np
    path = Path(os.environ[ENV + 'TEACHER'])
    if path.is_symlink() or smoke.file_sha(path) != policy['teacher']['sha256']:
        raise ValueError('Teacher file differs from the pinned SHA')
    with np.load(path, allow_pickle=False) as z:
        ids, probs = z['ids'].astype(str).tolist(), z['probs'].astype(np.float32)
    staged_corpus()
    if ids != [str(u) for u in _CORPUS[2]] or probs.shape != (len(ids), 12) or not np.isfinite(probs).all() or probs.min() < 0 or probs.max() > 1:
        raise ValueError('Teacher coverage/order/range differs from the staged corpus')
    return {u: probs[j] for j, u in enumerate(ids)}


class MaskedPosWeightBCE:
    """Mirror arm: per-element masked BCE with the historical pos_weight; identical to the historical mean when the mask is all ones."""
    def __init__(self, pos_weight):
        self.pos_weight = pos_weight

    def __call__(self, logits, y):
        from torch.nn import functional as F
        if y.shape[-1] == 12:
            return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        target, mask = y[..., :12], y[..., 12:]
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight, reduction='none')
        return (loss * mask).sum() / mask.sum().clamp_min(1)


class DistilBCE:
    """Distil arm: historical report-label loss + fixed 0.5 x plain BCE toward the teacher probabilities (all targets)."""
    def __init__(self, pos_weight, weight):
        self.pos_weight, self.weight = pos_weight, weight

    def __call__(self, logits, y):
        from torch.nn import functional as F
        if y.shape[-1] == 12:
            return F.binary_cross_entropy_with_logits(logits, y, pos_weight=self.pos_weight)
        return (F.binary_cross_entropy_with_logits(logits, y[..., :12], pos_weight=self.pos_weight) +
                self.weight * F.binary_cross_entropy_with_logits(logits, y[..., 12:]))


def loss_variant(lossf, script):
    policy = recipe(script); variant = policy.get('variant')
    if variant is None:
        return lossf
    if variant == 'mirror':
        return MaskedPosWeightBCE(lossf.pos_weight)
    return DistilBCE(lossf.pos_weight, policy['teacher']['weight'])


def mirror_tta_csv(tk, model, val_ids, id2row, res, k_eval, dev, workers, plain, path):
    """Mirror TTA at export (secondary file): mean of plain and mirrored-then-side-swapped probabilities; MCL plain only."""
    import csv
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    ds = tk.StudyWindows('', list(val_ids), id2row, {u: np.zeros(12, np.float32) for u in val_ids}, res, k_eval, train=False, norm='imagenet')
    ds.force_mirror = True
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=4, shuffle=False, num_workers=workers, collate_fn=tk.collate):
            with torch.autocast(dev, dtype=torch.float16, enabled=dev == 'cuda'):
                out.append(torch.sigmoid(model(x.to(dev)).float()).cpu().numpy())
    mirrored = mirror_targets(np.concatenate(out))
    tta = (np.asarray(plain) + mirrored) / 2; tta[:, MCL_INDEX] = np.asarray(plain)[:, MCL_INDEX]
    if not np.isfinite(tta).all() or tta.min() < 0 or tta.max() > 1:
        raise SystemExit('non-finite or out-of-range mirror TTA predictions')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['StudyInstanceUID', *LAB, 'set'])
        for u, row in zip(val_ids, tta):
            w.writerow([u, *row.tolist(), 'val'])
    return tta
