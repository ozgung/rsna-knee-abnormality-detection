"""Raptor64 loader/training-package tests on synthetic fixtures (no real images, labels or GPU)."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from knee import raptor64_cache as cache, public_training_smoke as smoke

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT', '/nonexistent'))
HISTORICAL = EVIDENCE / 'outputs/kaggle/trackA-coatnet224-fold01-20261009'
V4 = EVIDENCE / 'outputs/research/2026-10-08/raptor-audit/raptor_ft_coatnet_v4_full.pt'
HAVE_NUMPY = importlib.util.find_spec('numpy') is not None
HAVE_TORCH = importlib.util.find_spec('torch') is not None
HAVE_TIMM = HAVE_TORCH and importlib.util.find_spec('timm') is not None

if HAVE_NUMPY:
    from knee import raptor64_loader as loader
if HAVE_NUMPY:
    from knee import raptor64_training as guard
    _spec = importlib.util.spec_from_file_location('build_raptor64_training', ROOT / 'scripts/build_raptor64_training.py')
    builder = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(builder)

SLOTS_A = [{'plane': 'Sagittal', 'selected': True, 'fluid_sensitive': 1, 'frames': 30, 'count': 18},
           {'plane': 'Sagittal', 'selected': True, 'fluid_sensitive': 0, 'frames': 22, 'count': 14},
           {'plane': 'Coronal', 'selected': True, 'fluid_sensitive': 1, 'frames': 9, 'count': 12},
           {'plane': 'Coronal', 'selected': False, 'fluid_sensitive': None, 'frames': 0, 'count': 8},
           {'plane': 'Axial', 'selected': True, 'fluid_sensitive': 0, 'frames': 40, 'count': 12}]


def fixture_arrays(i):
    import numpy as np
    base = (np.arange(336, dtype=np.uint16) % 250 + 1).astype(np.uint8)
    v = np.broadcast_to(base, (64, 336, 336)).copy(); v = (v + i).astype(np.uint8); m = np.ones(64, np.uint8)
    v[44:52] = 0; m[44:52] = 0  # missing coronal non-FS slot, as in SLOTS_A
    return v, m


@unittest.skipUnless(HAVE_NUMPY, 'NumPy required')
class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.ids = [f'fixture-study-{i}' for i in range(4)]
        self.folds = {u: (0 if i == 0 else 1) for i, u in enumerate(self.ids)}
        train, dev = self.ids[1:], self.ids[:1]
        self.membership = {'eligible': 4, 'training': 3, 'development': 1,
                           'ordered_eligible_sha256': loader.digest_document(self.ids),
                           'ordered_training_sha256': loader.digest_document(train), 'ordered_development_sha256': loader.digest_document(dev)}
        self.validation = {'fixture': True, 'membership': self.membership, 'folds': self.folds}
        parts = [{'studies': 1, 'ordinals_sha256': loader.digest_document([i]), 'ordered_ids_sha256': loader.digest_document([self.ids[i]])} for i in range(4)]
        self.shards = []
        for i in range(4):
            root = self.root / f'shard{i}'; root.mkdir()
            v, m = fixture_arrays(i); saved = cache.atomic_npz(root / cache.filename(i), v, m)
            contract = {'version': cache.VERSION, 'config': cache.CONFIG, 'membership': self.membership, 'shard_index': i, 'shard_count': 4,
                        'partitions': parts, 'reader': {'module_sha256': '1' * 64}, 'input_pins': {'identity': '3' * 64},
                        'qualified_environment': {'device': 'cpu'}, 'sources': {'reader': '4' * 64}}
            cache.atomic_json(root / 'run-contract.json', contract)
            index = {'version': cache.VERSION, 'contract_sha256': loader.digest_document(contract),
                     'entries': [{'ordinal': i, 'StudyInstanceUID': self.ids[i], 'file': cache.filename(i), **saved, 'windows': {}, 'slots': SLOTS_A}]}
            cache.atomic_json(root / 'index-private.json', index)
            receipt = {'status': 'PASS', 'version': cache.VERSION, 'contract_sha256': loader.digest_document(contract),
                       'private_index_sha256': smoke.file_sha(root / 'index-private.json'), 'completed_studies': 1, 'shard_index': i, 'shard_count': 4}
            cache.atomic_json(root / 'receipt.json', receipt)
            self.shards.append({'root': str(root), 'contract_file_sha256': smoke.file_sha(root / 'run-contract.json'),
                                'index_sha256': smoke.file_sha(root / 'index-private.json'), 'receipt_sha256': smoke.file_sha(root / 'receipt.json')})

    def tearDown(self):
        self.tmp.cleanup()

    def test_assemble_order_open_and_slot_metadata(self):
        import numpy as np
        manifest = loader.assemble([self.shards[i] for i in (2, 0, 3, 1)], self.validation)
        self.assertEqual([e['StudyInstanceUID'] for e in manifest['entries']], self.ids)
        with self.assertRaises(ValueError):
            loader.open_corpus(manifest, expected_manifest_sha256=manifest['manifest_sha256'])
        vols, masks, ids = loader.open_corpus(manifest, expected_manifest_sha256=manifest['manifest_sha256'], allow_fixture=True)
        self.assertEqual(vols.shape, (4, 64, 336, 336)); self.assertEqual(masks.shape, (4, 64))
        v, m = fixture_arrays(2)
        self.assertTrue(np.array_equal(vols[2], v)); self.assertTrue(np.array_equal(masks[2], m))
        with self.assertRaises(ValueError):
            np.asarray(vols)
        codes, depth = loader.slice_metadata_table(manifest)
        self.assertEqual(codes.shape, (4, 64)); self.assertEqual(codes[0, :18].tolist(), [1] * 18); self.assertEqual(codes[0, 52:].tolist(), [4] * 12)

    def test_slot_schema_and_production_counts_enforced(self):
        with self.assertRaises(ValueError):
            loader.assemble(self.shards, self.validation | {'fixture': False})
        for broken in ([*SLOTS_A[:4]], [dict(SLOTS_A[0], plane='Axial'), *SLOTS_A[1:]],
                       [dict(SLOTS_A[0], fluid_sensitive=None), *SLOTS_A[1:]], [*SLOTS_A[:3], dict(SLOTS_A[3], frames=3), SLOTS_A[4]]):
            with self.assertRaises(ValueError):
                loader.validate_slots(broken)
        index = json.loads(Path(self.shards[1]['root'], 'index-private.json').read_text())
        index['entries'][0]['slots'] = SLOTS_A[:4]
        cache.atomic_json(Path(self.shards[1]['root'], 'index-private.json'), index)
        shards = copy.deepcopy(self.shards); shards[1]['index_sha256'] = smoke.file_sha(Path(self.shards[1]['root'], 'index-private.json'))
        with self.assertRaises(ValueError):
            loader.assemble(shards, self.validation)

    def test_source96_shards_rejected(self):
        contract = json.loads(Path(self.shards[0]['root'], 'run-contract.json').read_text()); contract['version'] = 'own96-wide336-sharded-v1'
        cache.atomic_json(Path(self.shards[0]['root'], 'run-contract.json'), contract)
        shards = copy.deepcopy(self.shards); shards[0]['contract_file_sha256'] = smoke.file_sha(Path(self.shards[0]['root'], 'run-contract.json'))
        with self.assertRaises(ValueError):
            loader.assemble(shards, self.validation)


@unittest.skipUnless(HAVE_NUMPY, 'NumPy required')
class PolicyTests(unittest.TestCase):
    def test_arms_and_frozen_recipe(self):
        self.assertEqual(set(guard.ARMS), {'in-orig', 'in-plane', 'dr-orig', 'dr-plane', 'na-orig', 'dr-full', 'in-mirror', 'in-distil', 'in-distil-w1', 'in-distil-flip',
                                           'ax-crop', 'ax-full'})
        self.assertEqual(guard.ARM_ORDER, ('in-orig', 'in-plane', 'dr-orig', 'dr-plane'))
        self.assertEqual((guard.FROZEN['res'], guard.FROZEN['k'], guard.FROZEN['epochs'], guard.FROZEN['seed'], guard.FROZEN['topk']), (384, 8, 12, 42, 3))
        self.assertEqual(guard.EXPORT_K, 42); self.assertEqual(guard.FROZEN['arch'], 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k')
        self.assertEqual(guard.SPLITS[0]['development'], 870)
        for arm in guard.ARMS:
            self.assertLessEqual(len(f'knee-raptor64-{arm}-fold0-final-20261013'), 50)
        with self.assertRaises(ValueError):
            guard.arm_slug('imagenet-original')

    def test_eval_centres_match_public_definition_and_inference_metadata(self):
        import numpy as np
        mask = np.ones(64, np.uint8); mask[44:52] = 0; mask[:2] = 0
        centres = guard.eval_centers(mask, 64, 42)
        valid = np.where(mask > 0)[0]; lo, hi = valid.min(), valid.max()
        cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
        self.assertEqual(centres, [cs[i] for i in np.linspace(0, len(cs) - 1, 42).round().astype(int)])
        meta = guard.inference_window_metadata(SLOTS_A, mask, 42)
        codes, depth = cache.slice_metadata(SLOTS_A)
        self.assertTrue(np.array_equal(meta, cache.window_metadata(codes, depth, centres)))

    def test_resume_epoch_bounds(self):
        state = {k: None for k in ('model', 'opt', 'sched', 'scaler', 'best', 'best_state', 'best_P', 'rng')}
        ident = {'arm': 'in-orig'}
        good = {**state, 'identity': ident, 'epoch': 11, 'hist': [{'ep': e} for e in range(12)], 'topk': [{'epoch': 9}, {'epoch': 10}, {'epoch': 11}]}
        guard.validate_resume(good, ident)
        for bad in (good | {'epoch': 12, 'hist': [{'ep': e} for e in range(13)]}, good | {'identity': {'arm': 'dr-orig'}},
                    good | {'topk': [{'epoch': 1}, {'epoch': 1}]}, good | {'hist': []}):
            with self.assertRaises(ValueError):
                guard.validate_resume(bad, ident)

    def test_staging_env_and_fixture_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'cache64-manifest-private.json'
            path.write_text(json.dumps({'content_sha256': 'a' * 64, 'membership': {}, 'fixture': True, 'sources': [], 'entries': []}))
            rp = root / 'staging-receipt.json'
            rp.write_text(json.dumps({'status': 'PASS', 'manifest_sha256': 'b' * 64, 'content_sha256': 'a' * 64, 'manifest_file_sha256': smoke.file_sha(path)}))
            env = {'RAPTOR64_STAGE_ROOT': str(root), 'RAPTOR64_MANIFEST_SHA256': 'b' * 64, 'RAPTOR64_CONTENT_SHA256': 'a' * 64,
                   'RAPTOR64_STAGE_RECEIPT_SHA256': smoke.file_sha(rp)}
            with patch.dict(os.environ, env), patch.object(guard, '_CORPUS', None), patch.object(guard, '_META', None), \
                    patch.object(guard.loader, 'open_corpus', return_value=('v', 'm', ['fixture'])) as open_, \
                    patch.object(guard.loader, 'slice_metadata_table', return_value=('codes', 'depth')):
                self.assertEqual(guard.staged_corpus(allow_fixture=True), ('v', 'm', ['fixture']))
                self.assertTrue(open_.call_args.kwargs['allow_fixture'])
                with self.assertRaisesRegex(ValueError, 'Fixture corpus'):
                    guard.staged_corpus()
            with patch.dict(os.environ, {**env, 'RAPTOR64_CONTENT_SHA256': 'c' * 64}), patch.object(guard, '_CORPUS', None):
                with self.assertRaisesRegex(ValueError, 'staging PASS'):
                    guard.staged_corpus()


@unittest.skipUnless(HAVE_TORCH and HAVE_NUMPY, 'PyTorch required')
class InitializerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def descriptor(self, **fields):
        dp = self.root / 'initializer.json'; dp.write_text(json.dumps(fields))
        return {'RAPTOR64_INITIALIZER_DESCRIPTOR': str(dp), 'RAPTOR64_INITIALIZER_DESCRIPTOR_SHA256': smoke.file_sha(dp)}

    def dread_file(self, state, **ck):
        import torch
        path = self.root / 'v4.pt'
        torch.save({'model': state, 'arch': guard.ARCH, 'res': 384, 'lab': guard.LAB, **ck}, path)
        return path

    def test_dread_backbone_only_exact_coverage(self):
        import torch
        bb = torch.nn.Linear(3, 2)
        head = {k: torch.zeros(1) for k in guard.HEAD_KEYS}
        good = {**{'backbone.' + k: torch.full_like(v, 2.0) for k, v in bb.state_dict().items()}, **head}

        def load(state, **ck):
            path = self.dread_file(state, **ck)
            env = self.descriptor(init='dread_v4', kind='dread_v4_full', dataset=guard.DREAD_PIN['dataset'],
                                  weights_file=str(path), weights_sha256=smoke.file_sha(path))
            pin = dict(guard.DREAD_PIN, sha256=smoke.file_sha(path), size=path.stat().st_size)
            with patch.dict(os.environ, env), patch.object(guard, 'DREAD_PIN', pin):
                return guard.initialize_backbone(bb)
        record = load(good)
        self.assertEqual((record['missing'], record['unexpected'], record['loaded_tensors']), (0, 0, 2))
        self.assertTrue(bool((bb.weight == 2).all()))
        broken = [{k: v for k, v in good.items() if k != 'backbone.bias'},                  # missing backbone tensor
                  {**good, 'backbone.extra': torch.zeros(1)},                              # unexpected backbone tensor
                  {**good, 'extra_head': torch.zeros(1)},                                  # head inventory differs
                  {**good, 'backbone.weight': torch.full((2, 3), float('nan'))}]           # nonfinite
        for state in broken:
            with self.assertRaises((ValueError, RuntimeError)):
                load(state)
        with self.assertRaises(ValueError):
            load(good, res=320)
        with self.assertRaises(ValueError):
            load(good, lab=list(reversed(guard.LAB)))

    def test_nartaa_optional_init_exact_coverage_without_lab(self):
        import torch
        bb = torch.nn.Linear(3, 2)
        state = {**{'backbone.' + k: torch.full_like(v, 3.0) for k, v in bb.state_dict().items()}, **{k: torch.zeros(1) for k in guard.HEAD_KEYS}}
        path = self.root / 'nartaa.pt'; torch.save({'model': state, 'arch': guard.ARCH, 'res': 384}, path)
        env = self.descriptor(init='nartaa_oai', kind='nartaa_oai_swa', dataset=guard.NARTAA_PIN['dataset'], weights_file=str(path), weights_sha256=smoke.file_sha(path))
        pin = dict(guard.NARTAA_PIN, sha256=smoke.file_sha(path), size=path.stat().st_size)
        with patch.dict(os.environ, env), patch.object(guard, 'NARTAA_PIN', pin):
            record = guard.initialize_backbone(bb)
        self.assertEqual((record['init'], record['missing'], record['unexpected'], record['loaded_tensors']), ('nartaa_oai', 0, 0, 2))
        self.assertIn('OAI', record['exposure'])
        self.assertEqual((guard.NARTAA_PIN['sha256'][:12], guard.NARTAA_PIN['size']), ('7e5315dad125', 292840787))
        with patch.dict(os.environ, env), self.assertRaises(ValueError):  # real pin, synthetic bytes
            guard.initialize_backbone(bb)
        torch.save({'model': {**state, 'backbone.extra': torch.zeros(1)}, 'arch': guard.ARCH, 'res': 384}, path)
        env = self.descriptor(init='nartaa_oai', kind='nartaa_oai_swa', dataset=guard.NARTAA_PIN['dataset'], weights_file=str(path), weights_sha256=smoke.file_sha(path))
        with patch.dict(os.environ, env), patch.object(guard, 'NARTAA_PIN', dict(pin, sha256=smoke.file_sha(path), size=path.stat().st_size)), self.assertRaisesRegex(ValueError, 'unexpected=1'):
            guard.initialize_backbone(bb)

    def test_dread_bytes_must_match_pinned_sha(self):
        import torch
        bb = torch.nn.Linear(3, 2)
        path = self.dread_file({'backbone.' + k: v for k, v in bb.state_dict().items()})
        env = self.descriptor(init='dread_v4', kind='dread_v4_full', dataset=guard.DREAD_PIN['dataset'],
                              weights_file=str(path), weights_sha256=guard.DREAD_PIN['sha256'])
        with patch.dict(os.environ, env), self.assertRaises(ValueError):
            guard.initialize_backbone(bb)

    def test_imagenet_strict_cover(self):
        import torch
        bb = torch.nn.Linear(3, 2)

        def load(state, **override):
            wp = self.root / 'imagenet.pt'; torch.save(state, wp)
            fields = dict(init='imagenet', kind='imagenet_backbone', arch=guard.ARCH, hf_repo=guard.IMAGENET_PIN['repo'],
                          hf_revision=guard.IMAGENET_PIN['revision'], origin_file_sha256=guard.IMAGENET_PIN['sha256'],
                          weights_file=str(wp), weights_sha256=smoke.file_sha(wp)); fields.update(override)
            with patch.dict(os.environ, self.descriptor(**fields)):
                return guard.initialize_backbone(bb)
        good = {k: torch.ones_like(v) for k, v in bb.state_dict().items()}
        self.assertEqual(load(good)['init'], 'imagenet')
        for state in ({}, {**good, 'extra': torch.zeros(1)}, {'module.' + k: v for k, v in good.items()}):
            with self.assertRaises(ValueError):
                load(state)
        with self.assertRaises(ValueError):
            load(good, hf_revision='0' * 40)
        with self.assertRaises(ValueError):
            load(good, init='random')

    @unittest.skipUnless(HAVE_TIMM and V4.is_file(), 'timm and the local CC0 v4_full checkpoint required')
    def test_real_v4_full_backbone_missing0_unexpected0(self):
        import timm
        bb = timm.create_model(guard.ARCH, pretrained=False, num_classes=0, in_chans=3, global_pool='avg')
        env = self.descriptor(init='dread_v4', kind='dread_v4_full', dataset=guard.DREAD_PIN['dataset'],
                              weights_file=str(V4), weights_sha256=guard.DREAD_PIN['sha256'])
        with patch.dict(os.environ, env):
            record = guard.initialize_backbone(bb)
        self.assertEqual((record['missing'], record['unexpected']), (0, 0)); self.assertEqual(record['loaded_tensors'], len(bb.state_dict()))


def historical_classes():
    import numpy as np, torch
    from torch import nn
    from torch.nn import functional as F
    text = (HISTORICAL / 'train_knee_patched.py').read_text()
    ns = {'np': np, 'torch': torch, 'nn': nn, 'F': F, 'random': random, 'Dataset': torch.utils.data.Dataset, '_open_corpus': lambda: None,
          '_roi_square_box': None, '_roi_compartments': None}
    for name in ('StudyWindows', 'RaptorClassifier', 'collate'):
        exec(builder.function_source(text, name), ns)
    return ns['StudyWindows'], ns['RaptorClassifier'], ns['collate']


@unittest.skipUnless(HAVE_TORCH and HAVE_NUMPY and (HISTORICAL / 'train_knee_patched.py').is_file(), 'Pinned historical trainer and PyTorch required')
class PlaneHeadTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.SW, self.RC, self.collate = historical_classes()
        self.PSW, self.PRC, self.pcollate = guard.plane_variant(self.SW, self.RC)
        self.vols = np.stack([fixture_arrays(i)[0] for i in range(3)]); self.masks = np.stack([fixture_arrays(i)[1] for i in range(3)])
        codes, depth = cache.slice_metadata(SLOTS_A)
        self.meta = (np.stack([codes] * 3), np.stack([depth] * 3))
        self.ids = ['a', 'b', 'c']; self.id2row = {u: i for i, u in enumerate(self.ids)}
        self.labels = {u: np.zeros(12, np.float32) for u in self.ids}

    def dataset(self, cls, k, train, res=32):
        ds = cls('', self.ids, self.id2row, self.labels, res, k, train=train, aug=train, norm='imagenet')
        ds.vols, ds.masks = self.vols, self.masks
        return ds

    def test_train_windows_and_rng_identical_to_original_sampler(self):
        import numpy as np, torch
        with patch.object(guard, '_META', self.meta):
            random.seed(7); plain = [self.dataset(self.SW, 8, True)[i][0] for i in range(3)]
            random.seed(7); ds = self.dataset(self.PSW, 8, True); items = [ds[i] for i in range(3)]
        for (wb, _), x in zip(items, plain):
            self.assertTrue(torch.equal(wb.x, x)); self.assertEqual(tuple(wb.meta.shape), (8, 2))
        codes, depth = cache.slice_metadata(SLOTS_A)
        random.seed(7); centres = self.dataset(self.SW, 8, True)._centers(np.where(self.masks[0] > 0)[0], 8)
        self.assertTrue(np.array_equal(items[0][0].meta.numpy(), cache.window_metadata(codes, depth, centres)))

    def test_eval_metadata_parity_with_inference_path(self):
        import numpy as np
        with patch.object(guard, '_META', self.meta):
            ds = self.dataset(self.PSW, 42, False)
            for i in range(3):
                wb, _ = ds[i]
                expected = guard.inference_window_metadata(SLOTS_A, self.masks[i], 42)
                self.assertTrue(np.array_equal(wb.meta.numpy(), expected))

    def test_plane_model_forward_backward_swa_and_original_unchanged(self):
        import torch
        from torch import nn
        torch.manual_seed(0)
        backbone = nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(12, 16))
        backbone.num_features = 16
        with patch.object(guard, '_META', self.meta):
            ds = self.dataset(self.PSW, 8, True); batch, y = self.pcollate([ds[0], ds[1]])
        model = self.PRC(backbone, F_dim=16)
        self.assertEqual(tuple(model.clsW.shape), (12, 16 + guard.CODE_DIM + guard.DEPTH_DIM))
        out = model(batch.to('cpu')); self.assertEqual(tuple(out.shape), (2, 12))
        nn.BCEWithLogitsLoss()(out, y).backward()
        self.assertIsNotNone(model.code_embed.weight.grad); self.assertIsNotNone(model.depth_embed.weight.grad)
        head = [n for n, _ in model.named_parameters() if not n.startswith('backbone.')]
        self.assertIn('code_embed.weight', head); self.assertIn('att.0.weight', head)
        with self.assertRaises(TypeError):
            model(batch.x)
        avg = {k: (v.double() * 2 / 2).to(v.dtype) if v.is_floating_point() else v for k, v in model.state_dict().items()}
        model.load_state_dict(avg)
        plain = self.RC(nn.Sequential(nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(12, 16)), F_dim=16)
        self.assertFalse(hasattr(plain, 'code_embed')); self.assertEqual(tuple(plain.clsW.shape), (12, 16))


@unittest.skipUnless(HAVE_NUMPY and (HISTORICAL / 'train_knee_patched.py').is_file(), 'Pinned historical assets required')
class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_four_arm_packages_ast_hooks_and_policy(self):
        policies = {}
        for arm in guard.ARM_ORDER:
            out = self.root / arm; r = builder.build(out, evidence_root=EVIDENCE, arm=arm)
            self.assertTrue(r['scientific_ast_equal'])
            trainer = (out / 'train_knee_patched.py').read_text(); entry = (out / 'tracka_entry.py').read_text()
            self.assertEqual(builder.scientific_ast(trainer), builder.scientific_ast((HISTORICAL / 'train_knee_patched.py').read_text()))
            self.assertIn(builder.HEAD_HOOK.strip(), trainer); self.assertIn('raptor64.initialize_backbone(bb, __file__)', trainer)
            self.assertNotIn('src = load_raptor(bb, a.ckpt)', trainer); self.assertIn('RAPTOR64_RESUME_SHA256', trainer)
            self.assertIn('k_eval = raptor64.EXPORT_K', entry); self.assertIn("'EPOCHS': '12'", entry); self.assertIn("'RES': '384'", entry)
            self.assertIn(f"/content/raptor64-{arm}-fold0", entry)
            policy = guard.recipe(out / 'train_knee_patched.py'); policies[arm] = policy
            self.assertEqual((policy['init'], policy['head']), guard.ARMS[arm])
            args = SimpleNamespace(**guard.FROZEN, workers=2, tag='fold0')
            guard.require_recipe(args, out / 'train_knee_patched.py')
            for key, value in (('k_eval', 42), ('epochs', 10), ('res', 224), ('fold', 1)):
                with self.assertRaises(ValueError):
                    guard.require_recipe(SimpleNamespace(**{**vars(args), key: value}), out / 'train_knee_patched.py')
            sentinel = (type('S', (), {}), type('R', (), {}), object())
            variant = guard.head_variant(*sentinel, out / 'train_knee_patched.py')
            self.assertEqual(variant == sentinel, policy['head'] == 'original')
        strip = lambda p: {k: v for k, v in p.items() if k not in ('recipe', 'arm', 'init', 'head', 'initializer', 'head_definition')}
        self.assertEqual(len({json.dumps(strip(p), sort_keys=True) for p in policies.values()}), 1)
        out = self.root / 'in-orig'
        (out / 'knee/raptor64_loader.py').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'runtime module'):
            guard.recipe(out / 'train_knee_patched.py')
        with self.assertRaises(ValueError):
            builder.build(out, evidence_root=EVIDENCE, arm='in-orig')
        with self.assertRaises(ValueError):
            builder.build(self.root / 'x', evidence_root=EVIDENCE, arm='in-orig', workers=10)

    def test_export_guard_requires_twelve_epoch_swa(self):
        out = self.root / 'pkg'; builder.build(out, evidence_root=EVIDENCE, arm='dr-plane')
        ckpt = out / 'state.pt'; ckpt.write_bytes(b'fixed')
        (out / 'raptor_ft_fold0.json').write_text(json.dumps({'partial': False, 'hist': [{}] * 12}))
        guard.export_guard(out, 0, ckpt, 'swa_topk', [9, 10, 11], False)
        for args in ((out, 1, ckpt, 'swa_topk', [9, 10, 11], False), (out, 0, ckpt, 'best_epoch', [9], False),
                     (out, 0, ckpt, 'swa_topk', [9, 10, 11], True), (out, 0, ckpt, 'swa_topk', [9, 9, 11], False)):
            with self.assertRaises(ValueError):
                guard.export_guard(*args)
        (out / 'raptor_ft_fold0.json').write_text(json.dumps({'partial': False, 'hist': [{}] * 10}))
        with self.assertRaises(ValueError):
            guard.export_guard(out, 0, ckpt, 'swa_topk', [7, 8, 9], False)


if __name__ == '__main__':
    unittest.main()
