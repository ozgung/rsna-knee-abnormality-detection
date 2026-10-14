"""CPU wiring test: generated Raptor64 trainer + export run end to end on a synthetic cache.

A tiny stand-in backbone replaces timm's CoAtNet and CUDA is replaced by CPU *only inside this
test*; all other code paths (historical sampler/model/optimizer/schedule/top-3/SWA, Raptor64
hooks, resume saving, plane-aware head, K42 export) are the generated package code. Study IDs
are a small subset of the pinned fold map so the real folds/exclusion pins apply; images,
labels and truth are synthetic. No real images, reports or outcomes are read.
"""
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT', '/nonexistent'))
HISTORICAL = EVIDENCE / 'outputs/kaggle/trackA-coatnet224-fold01-20261009'
READY = ((HISTORICAL / 'train_knee_patched.py').is_file() and
         all(importlib.util.find_spec(p) for p in ('numpy', 'torch', 'timm', 'pandas', 'pyarrow', 'sklearn')))


def tiny_factory():
    import torch
    from torch import nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(4); self.fc = nn.Linear(48, 16); self.num_features = 16

        def forward(self, x):
            return self.fc(self.pool(x).flatten(1))

        def set_grad_checkpointing(self, enable=True):
            return None
    return Tiny


@unittest.skipUnless(READY, 'Pinned historical trainer plus numpy/torch/timm/pandas/pyarrow/sklearn required')
class EndToEndTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from knee import raptor64_training as guard, raptor64_cache as cache, raptor64_loader as loader, public_training_smoke as smoke
        self.guard, self.cache, self.loader, self.smoke = guard, cache, loader, smoke
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        folds = json.loads((HISTORICAL / 'folds_v0.json').read_text())['folds']
        excluded = set(json.loads((HISTORICAL / 'excluded_ids.json').read_text())['quarantined_study_ids'])
        pool = sorted(u for u in folds if u not in excluded and folds[u] >= 0)
        dev = [u for u in pool if folds[u] == 0][:4]; train = [u for u in pool if folds[u] != 0][:8]
        self.ids = sorted(dev + train); self.dev = [u for u in self.ids if folds[u] == 0]; self.train = [u for u in self.ids if folds[u] != 0]
        rng = np.random.default_rng(0)
        # Synthetic four-shard cache with slot metadata.
        slots = [{'plane': p, 'selected': True, 'fluid_sensitive': int(f == 1), 'frames': 20 + 3 * i, 'count': c}
                 for i, (p, f, c) in enumerate(cache.SLOTS)]
        parts = [{'studies': len(range(s, len(self.ids), 4)), 'ordinals_sha256': loader.digest_document(list(range(s, len(self.ids), 4))),
                  'ordered_ids_sha256': loader.digest_document([self.ids[n] for n in range(s, len(self.ids), 4)])} for s in range(4)]
        membership = {'eligible': len(self.ids), 'training': len(self.train), 'development': len(self.dev),
                      'ordered_eligible_sha256': loader.digest_document(self.ids), 'ordered_training_sha256': loader.digest_document(self.train),
                      'ordered_development_sha256': loader.digest_document(self.dev)}
        shards = []
        for s in range(4):
            root = self.root / f'shard{s}'; root.mkdir(); entries = []
            for n in range(s, len(self.ids), 4):
                v = rng.integers(1, 255, (64, 336, 336), dtype=np.uint8); m = np.ones(64, np.uint8); v[50:52] = 0; m[50:52] = 0
                saved = cache.atomic_npz(root / cache.filename(n), v, m)
                entries.append({'ordinal': n, 'StudyInstanceUID': self.ids[n], 'file': cache.filename(n), **saved, 'windows': {}, 'slots': slots})
            contract = {'version': cache.VERSION, 'config': cache.CONFIG, 'membership': membership, 'shard_index': s, 'shard_count': 4,
                        'partitions': parts, 'reader': {}, 'input_pins': {}, 'qualified_environment': {}, 'sources': {}}
            cache.atomic_json(root / 'run-contract.json', contract)
            cache.atomic_json(root / 'index-private.json', {'version': cache.VERSION, 'contract_sha256': loader.digest_document(contract), 'entries': entries})
            cache.atomic_json(root / 'receipt.json', {'status': 'PASS', 'version': cache.VERSION, 'contract_sha256': loader.digest_document(contract),
                              'private_index_sha256': smoke.file_sha(root / 'index-private.json'), 'completed_studies': len(entries), 'shard_index': s, 'shard_count': 4})
            shards.append({'root': str(root), 'contract_file_sha256': smoke.file_sha(root / 'run-contract.json'),
                           'index_sha256': smoke.file_sha(root / 'index-private.json'), 'receipt_sha256': smoke.file_sha(root / 'receipt.json')})
        self.manifest = loader.assemble(shards, {'fixture': True, 'membership': membership, 'folds': {u: folds[u] for u in self.ids}})
        self.split = {'fold': 0, 'eligible': len(self.ids), 'training': len(self.train), 'development': len(self.dev),
                      'ordered_eligible_sha256': smoke.digest(smoke.canonical(self.ids)),
                      'ordered_training_sha256': smoke.digest(smoke.canonical(self.train)),
                      'ordered_development_sha256': smoke.digest(smoke.canonical(self.dev))}
        # Synthetic supervision and truth (both classes for every target on the four development studies).
        import pandas as pd
        self.input = self.root / 'input'; self.input.mkdir()
        labels = pd.DataFrame(rng.uniform(0, 1, (len(self.ids), 12)).astype(np.float32), columns=guard.LAB); labels.insert(0, 'StudyInstanceUID', self.ids)
        labels.to_parquet(self.input / 'labels_v1_blend.parquet')
        truth = pd.DataFrame({'StudyInstanceUID': self.ids})
        for j, name in enumerate(guard.LAB):
            truth['y_' + name] = [float((i + j) % 2) for i in range(len(self.ids))]; truth['m_' + name] = 1.0
        truth.to_parquet(self.input / 'train_labels_v1.parquet')
        pd.DataFrame({'StudyInstanceUID': self.ids, **{n: [None] * len(self.ids) for n in guard.LAB}}).to_csv(self.input / 'train.csv', index=False)
        shutil.copyfile(HISTORICAL / 'train_knee_original.py', self.input / 'train_knee.py')
        shutil.copyfile(HISTORICAL / 'folds_v0.json', self.input / 'folds_v0.json')

    def tearDown(self):
        for name in ('train_knee_patched', 'tracka_entry_r64'):
            sys.modules.pop(name, None)
        self.tmp.cleanup()

    def fixture_split(self, ids, fmap, excluded, fold):
        if list(ids) != self.ids or fold != 0:
            raise ValueError('fixture split differs')
        return self.split, self.train, self.dev

    def run_arm(self, arm, swa_epochs=None, teacher=None):
        import numpy as np
        import torch, timm
        guard, smoke = self.guard, self.smoke
        spec = importlib.util.spec_from_file_location('build_raptor64_training', ROOT / 'scripts/build_raptor64_training.py')
        builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        Tiny = tiny_factory()
        init, head = guard.ARMS[arm]
        weights = self.root / f'{arm}-init.pt'
        if init == 'imagenet':
            torch.save(Tiny().state_dict(), weights)
            descriptor = dict(init='imagenet', kind='imagenet_backbone', arch=guard.ARCH, hf_repo=guard.IMAGENET_PIN['repo'],
                              hf_revision=guard.IMAGENET_PIN['revision'], origin_file_sha256=guard.IMAGENET_PIN['sha256'],
                              weights_file=str(weights), weights_sha256='')
        else:
            state = {**{'backbone.' + k: v for k, v in Tiny().state_dict().items()}, **{k: torch.zeros(1) for k in guard.HEAD_KEYS}}
            torch.save({'model': state, 'arch': guard.ARCH, 'res': 384, 'lab': guard.LAB}, weights)
            descriptor = dict(init='dread_v4', kind='dread_v4_full', dataset=guard.DREAD_PIN['dataset'], weights_file=str(weights), weights_sha256='')
        descriptor['weights_sha256'] = smoke.file_sha(weights)
        dread_pin = dict(guard.DREAD_PIN, sha256=descriptor['weights_sha256'], size=weights.stat().st_size)
        dp = self.root / f'{arm}-initializer.json'; dp.write_text(json.dumps(descriptor))
        work = self.root / f'work-{arm}'; package = self.root / f'pkg-{arm}'
        env = {'RAPTOR_INPUT': str(self.input), 'WORKDIR': str(work), 'ASSET_DIR': str(package), 'RAPTOR_PRELOAD': '0',
               'RAPTOR64_INITIALIZER_DESCRIPTOR': str(dp), 'RAPTOR64_INITIALIZER_DESCRIPTOR_SHA256': smoke.file_sha(dp),
               'RAPTOR64_CONTENT_SHA256': self.manifest['content_sha256'], **({'RAPTOR64_TEACHER': str(teacher)} if teacher else {})}
        corpus = self.loader.open_corpus(self.manifest, expected_manifest_sha256=self.manifest['manifest_sha256'], allow_fixture=True)
        identity = {'content_sha256': self.manifest['content_sha256'], 'membership': self.manifest['membership'], 'fixture': False, 'sources': []}
        full_split = {'fold': None, 'eligible': len(self.ids), 'training': len(self.ids), 'development': 0,
                      'ordered_eligible_sha256': smoke.digest(smoke.canonical(self.ids)), 'ordered_training_sha256': smoke.digest(smoke.canonical(self.ids)),
                      'ordered_development_sha256': smoke.digest(smoke.canonical([]))}
        patches = [patch.dict(os.environ, env), patch.dict(guard.SPLITS, {0: self.split}), patch.object(guard, 'FULL_SPLIT', full_split), patch.object(guard, 'experiment_split', self.fixture_split),
                   patch.object(guard, 'LABEL_SHA', smoke.file_sha(self.input / 'labels_v1_blend.parquet')),
                   patch.object(guard, 'TRUTH_SHA', smoke.file_sha(self.input / 'train_labels_v1.parquet')),
                   patch.object(guard, 'DREAD_PIN', dread_pin), patch.object(guard, 'strict_cuda', lambda torch_mod=None: 'cpu'),
                   patch.object(guard, '_CORPUS', corpus), patch.object(guard, '_META', self.loader.slice_metadata_table(self.manifest)),
                   patch.object(guard, '_CACHE_IDENTITY', identity), patch('torch.cuda.get_device_name', return_value='cpu-fixture'),
                   patch.object(guard, '_AXIAL', guard.axial_presence(self.manifest)),
                   patch.object(timm, 'create_model', lambda *a, **k: Tiny())]
        for p in patches:
            p.start()
        cwd = os.getcwd()
        try:
            builder.build(package, evidence_root=EVIDENCE, arm=arm, swa_epochs=swa_epochs, teacher_sha256=smoke.file_sha(teacher) if teacher else None)
            entry_spec = importlib.util.spec_from_file_location('tracka_entry_r64', package / 'tracka_entry.py')
            entry = importlib.util.module_from_spec(entry_spec); sys.modules['tracka_entry_r64'] = entry; entry_spec.loader.exec_module(entry)
            c = entry.cfg(); p = entry.prepare(c); cmd = entry.train_cmd(c, p, 0)
            frozen = guard.frozen_for(arm)
            self.assertEqual(cmd[cmd.index('--k_eval') + 1], str(frozen['k_eval'])); self.assertEqual(cmd[cmd.index('--k') + 1], str(frozen['k']))
            self.assertEqual(cmd[cmd.index('--epochs') + 1], '12')
            self.assertEqual(cmd[cmd.index('--folds') + 1], '0' if swa_epochs else '5')
            os.chdir(work); sys.modules.pop('train_knee_patched', None); sys.path.insert(0, str(work))
            import train_knee_patched as tk
            random.seed(0); np.random.seed(0); torch.manual_seed(0)
            with patch.object(sys, 'argv', cmd[2:]), patch('torch.multiprocessing.get_start_method', return_value='fork'):
                torch.multiprocessing.set_start_method('fork', force=True)
                tk.main()
            json.dump({'cmd': cmd, 'rc': 0, 'seconds': 1.0}, open(work / 'train_meta_fold0.json', 'w'))
            entry.export_main(0)
        finally:
            os.chdir(cwd)
            if str(work) in sys.path:
                sys.path.remove(str(work))
            for p in reversed(patches):
                p.stop()
        return work

    def check(self, work, arm):
        import pandas as pd
        record = json.loads((work / 'run_fold0.json').read_text()); meta = json.loads((work / 'raptor_ft_fold0.json').read_text())
        axial = arm in self.guard.AXIAL_SPECS
        self.assertEqual(record['export']['k_eval'], 10 if axial else 42); self.assertEqual(record['export']['n_val'], len(self.dev)); self.assertEqual(record['export']['n_gold'], 0)
        self.assertEqual(record['checkpoint_kind'], 'swa_topk'); self.assertEqual(len(record['selected_epochs']), 3)
        self.assertEqual(record['comparison_endpoint'], 'K10 primary (selection K10; axial windows)' if axial else 'K42 primary (selection K16)')
        self.assertEqual(record['raptor64_policy']['arm'], arm)
        self.assertEqual([h['ep'] for h in meta['hist']], list(range(12))); self.assertFalse(meta['partial'])
        oof = pd.read_csv(work / 'oof_fold0.csv'); self.assertEqual(oof['StudyInstanceUID'].tolist(), self.dev)
        self.assertTrue(((oof[self.guard.LAB] >= 0) & (oof[self.guard.LAB] <= 1)).all().all())
        init = json.loads((work / 'initial-state-record.json').read_text())
        self.assertEqual((init['missing'], init['unexpected']), (0, 0)); self.assertEqual(init['init'], self.guard.ARMS[arm][0])
        self.assertTrue((work / 'raptor_resume_fold0.pt').is_file()); self.assertTrue((work / 'raptor_ft_fold0_swa.pt').is_file())
        return record

    def test_mirror_and_distil_variants_end_to_end(self):
        import numpy as np
        import pandas as pd
        work = self.run_arm('in-mirror')
        record = self.check(work, 'in-mirror')
        tta = pd.read_csv(work / 'oof_fold0_mirror_tta.csv'); plain = pd.read_csv(work / 'oof_fold0.csv')
        self.assertEqual(tta['StudyInstanceUID'].tolist(), self.dev)
        self.assertTrue(np.allclose(tta['MCL'], plain['MCL']))
        self.assertEqual(record['raptor64_policy']['variant'], 'mirror')
        teacher = self.root / 'teacher.npz'
        np.savez(teacher, ids=np.asarray(self.ids), probs=np.random.default_rng(1).uniform(0, 1, (len(self.ids), 12)).astype(np.float32))
        work = self.run_arm('in-distil', teacher=teacher)
        record = self.check(work, 'in-distil')
        self.assertEqual(record['raptor64_policy']['teacher']['sha256'], self.smoke.file_sha(teacher))
        self.assertFalse((work / 'oof_fold0_mirror_tta.csv').exists())

    def test_it3_w1_and_flip_end_to_end(self):
        import numpy as np
        teacher = self.root / 'teacher3.npz'
        np.savez(teacher, ids=np.asarray(self.ids), probs=np.random.default_rng(2).uniform(0, 1, (len(self.ids), 12)).astype(np.float32))
        for arm, weight in (('in-distil-w1', 1.0), ('in-distil-flip', 0.5)):
            record = self.check(self.run_arm(arm, teacher=teacher), arm)
            self.assertEqual(record['raptor64_policy']['teacher']['weight'], weight)
            self.assertEqual('flip' in record['raptor64_policy'], arm == 'in-distil-flip')

    def test_it4_axial_arms_end_to_end(self):
        import numpy as np
        spec = importlib.util.spec_from_file_location('r64run_ax_e2e', ROOT / 'scripts/colab_raptor64_run.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        teacher = self.root / 'teacher4.npz'
        np.savez(teacher, ids=np.asarray(self.ids), probs=np.random.default_rng(4).uniform(0, 1, (len(self.ids), 12)).astype(np.float32))
        for arm in ('ax-crop', 'ax-full'):
            work = self.run_arm(arm, teacher=teacher)
            record = self.check(work, arm)
            self.assertEqual(record['raptor64_policy']['axial'], self.guard.AXIAL_SPECS[arm])
            self.assertEqual(runner.primary_complete(work, arm=arm)['checkpoint_sha256'], record['checkpoint_sha256'])
            with self.assertRaises(ValueError):
                runner.primary_complete(work, arm='in-distil')

    def test_full_data_fixed_epoch_swa_end_to_end(self):
        import torch
        spec = importlib.util.spec_from_file_location('r64run', ROOT / 'scripts/colab_raptor64_run.py')
        runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
        work = self.run_arm('dr-full', swa_epochs=[8, 10, 11])
        record = json.loads((work / 'run_fold0.json').read_text()); meta = json.loads((work / 'raptor_ft_fold0.json').read_text())
        self.assertEqual((record['mode'], record['checkpoint_kind'], record['export']), ('full', 'swa_fixed_epochs', None))
        self.assertEqual(sorted(record['selected_epochs']), [8, 10, 11]); self.assertEqual(meta['n_train'], len(self.ids))
        self.assertFalse((work / 'oof_fold0.csv').exists())
        self.assertEqual(torch.load(work / 'raptor_ft_fold0_swa.pt', weights_only=False)['swa_over'], record['selected_epochs'])
        self.assertEqual(runner.primary_complete(work, arm='dr-full')['checkpoint_sha256'], record['checkpoint_sha256'])
        with self.assertRaises(ValueError):
            runner.primary_complete(work, arm='dr-orig')

    def test_plane_head_imagenet_and_original_head_dread_end_to_end(self):
        import torch
        plane = self.check(self.run_arm('in-plane'), 'in-plane')
        state = torch.load(self.root / 'work-in-plane/raptor_ft_fold0_swa.pt', weights_only=False)['model']
        self.assertIn('code_embed.weight', state); self.assertEqual(tuple(state['clsW'].shape), (12, 16 + 16 + 8))
        resume = torch.load(self.root / 'work-in-plane/raptor_resume_fold0.pt', weights_only=False)
        self.assertEqual(resume['epoch'], 11); self.assertEqual(resume['identity']['head'], 'plane')
        original = self.check(self.run_arm('dr-orig'), 'dr-orig')
        state = torch.load(self.root / 'work-dr-orig/raptor_ft_fold0_swa.pt', weights_only=False)['model']
        self.assertNotIn('code_embed.weight', state); self.assertEqual(tuple(state['clsW'].shape), (12, 16))
        self.assertNotEqual(plane['checkpoint_sha256'], original['checkpoint_sha256'])


if __name__ == '__main__':
    unittest.main()
