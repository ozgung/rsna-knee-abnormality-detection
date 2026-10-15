import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import stat
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HAVE_TORCH = importlib.util.find_spec('torch') is not None
HAVE_TIMM = HAVE_TORCH and importlib.util.find_spec('timm') is not None and importlib.util.find_spec('safetensors') is not None
spec = importlib.util.spec_from_file_location('prep_r64_init', ROOT / 'scripts/prepare_raptor64_initializer.py')
prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)


class FakeKaggle:
    def __init__(self, payload):
        self.payload = payload; self.calls = []

    def fetch(self, dataset, name, target, limit):
        self.calls.append((dataset, name))
        if len(self.payload) > limit:
            raise ValueError('limit')
        Path(target).write_bytes(self.payload)


class DreadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.data = b'synthetic-checkpoint-bytes' * 100
        self.pin = dict(prep.DREAD_PIN, sha256=hashlib.sha256(self.data).hexdigest(), size=len(self.data))

    def tearDown(self):
        self.tmp.cleanup()

    def test_raw_and_zip_payloads_pinned(self):
        with patch.object(prep, 'DREAD_PIN', self.pin):
            t = FakeKaggle(self.data); r = prep.prepare_dread(self.root / 'raw', transport=t)
            self.assertEqual(t.calls, [('dreaddevelopment/raptor-knee-widedense', 'raptor_ft_coatnet_v4_full.pt')])
            d = json.loads(Path(r['descriptor']).read_text())
            self.assertEqual((d['init'], d['kind'], d['weights_sha256']), ('dread_v4', 'dread_v4_full', self.pin['sha256']))
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as z:
                z.writestr(self.pin['file'], self.data)
            r = prep.prepare_dread(self.root / 'zip', transport=FakeKaggle(buffer.getvalue()))
            self.assertEqual(Path(json.loads(Path(r['descriptor']).read_text())['weights_file']).read_bytes(), self.data)
            with self.assertRaises(ValueError):
                prep.prepare_dread(self.root / 'raw', transport=FakeKaggle(self.data))  # completed descriptor preserved

    def test_wrong_bytes_or_archive_rejected_without_descriptor(self):
        with patch.object(prep, 'DREAD_PIN', self.pin):
            with self.assertRaises(ValueError):
                prep.prepare_dread(self.root / 'bad', transport=FakeKaggle(b'x' * len(self.data)))
            self.assertFalse((self.root / 'bad/initializer.json').exists())
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as z:
                z.writestr('other.pt', b'x' * len(self.data))
            with self.assertRaises(ValueError):
                prep.prepare_dread(self.root / 'badzip', transport=FakeKaggle(buffer.getvalue()))
            self.assertFalse((self.root / 'badzip/initializer.json').exists())
            with self.assertRaises(OSError):
                prep.prepare_dread(self.root / 'disk', transport=FakeKaggle(self.data), disk_usage=lambda _: SimpleNamespace(free=1))

    def test_renamed_member_uses_pinned_destination_and_bytes(self):
        with patch.object(prep, 'DREAD_PIN', self.pin):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as z:
                z.writestr('../../renamed-checkpoint.pt', self.data)
            result = prep.prepare_dread(self.root / 'renamed', transport=FakeKaggle(buffer.getvalue()))
            descriptor = json.loads(Path(result['descriptor']).read_text())
            self.assertEqual(Path(descriptor['weights_file']), (self.root / 'renamed' / self.pin['file']).resolve())
            self.assertEqual(Path(descriptor['weights_file']).read_bytes(), self.data)
            self.assertFalse((self.root.parent / 'renamed-checkpoint.pt').exists())

    def test_multiple_directory_and_symlink_members_rejected(self):
        with patch.object(prep, 'DREAD_PIN', self.pin):
            for kind in ('multiple', 'directory', 'symlink'):
                with self.subTest(kind=kind):
                    buffer = io.BytesIO()
                    with zipfile.ZipFile(buffer, 'w') as z:
                        if kind == 'multiple':
                            z.writestr('a.pt', self.data); z.writestr('b.pt', self.data)
                        else:
                            item = zipfile.ZipInfo('weights/' if kind == 'directory' else 'weights.pt')
                            item.create_system = 3
                            item.external_attr = ((stat.S_IFDIR if kind == 'directory' else stat.S_IFLNK) | 0o777) << 16
                            z.writestr(item, self.data)
                    destination = self.root / kind
                    with self.assertRaises(ValueError):
                        prep.prepare_dread(destination, transport=FakeKaggle(buffer.getvalue()))
                    self.assertFalse((destination / 'initializer.json').exists())

    def test_pins_match_guard_module(self):
        if not importlib.util.find_spec('numpy'):
            self.skipTest('NumPy required')
        from knee import raptor64_training as guard
        self.assertEqual(prep.IMAGENET_PIN, guard.IMAGENET_PIN)
        self.assertEqual({k: prep.DREAD_PIN[k] for k in ('dataset', 'file', 'sha256', 'size')}, {k: guard.DREAD_PIN[k] for k in ('dataset', 'file', 'sha256', 'size')})
        self.assertEqual({k: prep.NARTAA_PIN[k] for k in ('dataset', 'file', 'sha256', 'size')}, {k: guard.NARTAA_PIN[k] for k in ('dataset', 'file', 'sha256', 'size')})

    def test_raw_payload_that_is_itself_a_zip_accepted_when_pinned(self):
        """Regression (na-orig boot 2026-10-15): torch .pt files are zip archives; Kaggle served the raw file."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as z:
            z.writestr('archive/data.pkl', b'p' * 50); z.writestr('archive/data/0', b'q' * 70)
        data = buffer.getvalue(); pin = dict(prep.NARTAA_PIN, sha256=hashlib.sha256(data).hexdigest(), size=len(data))
        with patch.object(prep, 'NARTAA_PIN', pin):
            r = prep.prepare_nartaa(self.root / 'rawzip', transport=FakeKaggle(data))
            self.assertEqual(Path(json.loads(Path(r['descriptor']).read_text())['weights_file']).read_bytes(), data)
        with patch.object(prep, 'DREAD_PIN', dict(prep.DREAD_PIN, sha256=hashlib.sha256(data).hexdigest(), size=len(data))):
            prep.prepare_dread(self.root / 'rawzip-dread', transport=FakeKaggle(data))
            with self.assertRaises(ValueError):   # a multi-member zip that is NOT the pinned bytes is still rejected
                other = io.BytesIO()
                with zipfile.ZipFile(other, 'w') as z:
                    z.writestr('archive/data.pkl', b'P' * 50); z.writestr('archive/data/0', b'q' * 70)
                prep.prepare_dread(self.root / 'otherzip', transport=FakeKaggle(other.getvalue()))
            self.assertFalse((self.root / 'otherzip/initializer.json').exists())

    def test_nartaa_descriptor_matches_trainer_loader_contract(self):
        pin = dict(prep.NARTAA_PIN, sha256=hashlib.sha256(self.data).hexdigest(), size=len(self.data))
        with patch.object(prep, 'NARTAA_PIN', pin):
            t = FakeKaggle(self.data); r = prep.prepare_nartaa(self.root / 'na', transport=t)
            self.assertEqual(t.calls, [('nartaa/rsna-knee-publication-swa-weights-20261007', 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt')])
            d = json.loads(Path(r['descriptor']).read_text())
            # knee.raptor64_training.load_published requires exactly this kind/dataset/sha for init nartaa_oai
            self.assertEqual((d['init'], d['kind'], d['dataset'], d['weights_sha256']),
                             ('nartaa_oai', 'nartaa_oai_swa', 'nartaa/rsna-knee-publication-swa-weights-20261007', pin['sha256']))
            with self.assertRaises(ValueError):
                prep.prepare_nartaa(self.root / 'nabad', transport=FakeKaggle(b'x' * len(self.data)))
            self.assertFalse((self.root / 'nabad/initializer.json').exists())


@unittest.skipUnless(HAVE_TIMM, 'timm/safetensors required')
class ImagenetTests(unittest.TestCase):
    def test_normalized_backbone_of_real_384_schema(self):
        import torch
        full = prep.model_factory(1000)
        state = {k: torch.randn_like(v) if v.is_floating_point() else v for k, v in full.state_dict().items()}
        data = b'synthetic-origin'
        pin = dict(prep.IMAGENET_PIN, sha256=hashlib.sha256(data).hexdigest(), size=len(data))
        with tempfile.TemporaryDirectory() as tmp, patch.object(prep, 'IMAGENET_PIN', pin):
            r = prep.prepare_imagenet(Path(tmp) / 'init', fetch=lambda url: io.BytesIO(data), load=lambda path: state)
            d = json.loads(Path(r['descriptor']).read_text())
            self.assertEqual((d['init'], d['arch'], d['audit']['omitted_keys']), ('imagenet', prep.ARCH, ['head.fc.bias', 'head.fc.weight']))
            loaded = torch.load(d['weights_file'], weights_only=True)
            self.assertEqual(set(loaded), set(prep.model_factory(0).state_dict()))
            with self.assertRaises(ValueError):
                prep.prepare_imagenet(Path(tmp) / 'bad', fetch=lambda url: io.BytesIO(b'tampered-origin!'), load=lambda path: state)


if __name__ == '__main__':
    unittest.main()
