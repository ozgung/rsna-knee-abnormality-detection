import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from knee import public_training_smoke as smoke

ROOT = Path(__file__).resolve().parents[1]
HAVE_NUMPY = importlib.util.find_spec('numpy') is not None
if HAVE_NUMPY:
    import tests.test_raptor64_training as training_tests
    spec = importlib.util.spec_from_file_location('colab_raptor64_prepare', ROOT / 'scripts/colab_raptor64_prepare.py')
    stage = importlib.util.module_from_spec(spec); spec.loader.exec_module(stage)


class FakeTransport:
    def __init__(self, fixture, prefix='raptor64_cache/'):
        self.states = {}; self.inventory = {}; self.data = {}; self.calls = []; self.snapshots = {}; self.race = None; self.descriptors = []
        for i, old in enumerate(fixture.shards):
            ref = f'ozgung/knee-raptor64-cache-shard{i}-fixture'
            notebook = fixture.root / f'accepted{i}.ipynb'
            notebook.write_text(json.dumps({'cells': [{'cell_type': 'code', 'source': [f'# accepted fixture{i}\n']}]}) + '\n')
            self.states[ref] = {'version': 1, 'is_private': True, 'status': 'COMPLETE', 'cell_sources_sha256': stage.s96.source_cells_sha256(notebook.read_text())}
            self.descriptors.append({k: v for k, v in old.items() if k != 'root'} | {'ref': ref, 'version': 1, 'notebook_path': str(notebook),
                                                                                    'notebook_sha256': smoke.file_sha(notebook)})
            items = []
            for p in Path(old['root']).iterdir():
                if p.name not in stage.METADATA and not p.name.endswith('.npz'):
                    continue
                url = f'https://storage.googleapis.com/fixture/{i}/{p.name}?signed=DO_NOT_LOG'
                items.append({'name': prefix + p.name, 'url': url}); self.data[url] = p.read_bytes()
            self.inventory[ref] = items

    def snapshot(self, ref):
        self.snapshots[ref] = self.snapshots.get(ref, 0) + 1
        result = copy.deepcopy(self.states[ref])
        return self.race(ref, self.snapshots[ref], result) if self.race else result

    def list_files(self, ref):
        return copy.deepcopy(self.inventory[ref])

    def iter_chunks(self, url):
        self.calls.append(url); data = self.data[url]
        for i in range(0, len(data), stage.s96.CHUNK_BYTES):
            yield data[i:i + stage.s96.CHUNK_BYTES]


@unittest.skipUnless(HAVE_NUMPY, 'NumPy required')
class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.fixture = training_tests.LoaderTests(); self.fixture.setUp(); self.transport = FakeTransport(self.fixture)
        self.output = self.fixture.root / 'staged'
        self.config = {'version': stage.VERSION, 'cache_root': str(self.output), 'validation': self.fixture.validation,
                       'sources': [self.transport.descriptors[i] for i in (3, 1, 0, 2)]}

    def tearDown(self):
        self.fixture.tearDown()

    def prepare(self, **kw):
        return stage.prepare(self.config, self.transport, allow_fixture=True, **kw)

    def npz_calls(self):
        return [u for u in self.transport.calls if '.npz?' in u]

    def test_stage_manifest_slots_resume_and_no_urls_or_ids(self):
        report = self.prepare(download_workers=2); self.assertEqual(report['status'], 'PASS'); self.assertEqual(report['studies'], 4)
        manifest = json.loads((self.output / stage.MANIFEST).read_text())
        self.assertEqual([e['StudyInstanceUID'] for e in manifest['entries']], self.fixture.ids)
        self.assertEqual(manifest['entries'][0]['slots'], training_tests.SLOTS_A)
        public = (self.output / 'staging-receipt.json').read_text()
        self.assertNotIn('DO_NOT_LOG', public); self.assertFalse(any(uid in public for uid in self.fixture.ids))
        for p in self.output.glob('source*/*.npz'):
            self.assertEqual(p.stat().st_mode & 0o222, 0)
        before = len(self.transport.calls); again = self.prepare()
        self.assertEqual(again['content_sha256'], report['content_sha256']); self.assertEqual(len(self.transport.calls), before)
        opened = stage.loader.open_corpus(manifest, expected_manifest_sha256=report['manifest_sha256'], allow_fixture=True)
        self.assertEqual(opened[0].shape, (4, 64, 336, 336))

    def test_source96_prefixed_outputs_and_slugs_rejected(self):
        other = FakeTransport(self.fixture, prefix='source96_cache/')
        config = dict(self.config, sources=[other.descriptors[i] for i in range(4)])
        with self.assertRaises(RuntimeError):
            stage.prepare(config, other, allow_fixture=True)
        self.assertFalse([u for u in other.calls if '.npz?' in u])
        bad = copy.deepcopy(self.config); bad['sources'][0]['ref'] = 'ozgung/knee-source96-cache-shard0-20261012'
        with self.assertRaises(ValueError):
            stage.prepare(bad, self.transport, allow_fixture=True)

    def test_corruption_fatal_and_version_race_never_publishes_manifest(self):
        ref = self.config['sources'][0]['ref']
        url = next(i['url'] for i in self.transport.inventory[ref] if i['name'].endswith('.npz'))
        original = self.transport.data[url]; self.transport.data[url] = original + b'bad'
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.assertFalse((self.output / stage.MANIFEST).exists()); self.assertFalse(list(self.output.glob('source*/*.tmp')))
        self.transport.data[url] = original
        self.transport.race = lambda r, count, result: result | {'version': 2} if r == ref and count >= 5 else result
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.assertFalse((self.output / stage.MANIFEST).exists())
        self.transport.race = None; self.transport.snapshots = {}
        self.assertEqual(self.prepare()['status'], 'PASS')

    def test_local_metadata_collector_binds_build_contracts(self):
        import shutil
        spec = importlib.util.spec_from_file_location('collect_r64', ROOT / 'scripts/collect_raptor64_cache_metadata.py')
        collect = importlib.util.module_from_spec(spec); spec.loader.exec_module(collect)
        evidence = self.fixture.root / 'evidence'; slug = 'knee-raptor64-cache-shard{}-fixture'
        for i, old in enumerate(self.fixture.shards):
            build = evidence / 'outputs/kaggle' / slug.format(i); build.mkdir(parents=True)
            shutil.copyfile(self.fixture.root / f'accepted{i}.ipynb', build / 'notebook.ipynb')
            shutil.copyfile(Path(old['root']) / 'run-contract.json', build / 'build-contract.json')
        shards, membership = collect.qualify(evidence, self.transport, slug=slug, build_dir=slug)
        self.assertEqual([s['index_sha256'] for s in shards], [s['index_sha256'] for s in self.fixture.shards])
        self.assertEqual(membership, self.fixture.membership); self.assertFalse(self.npz_calls())
        (evidence / 'outputs/kaggle' / slug.format(2) / 'build-contract.json').write_text('{}')
        with self.assertRaises(ValueError):
            collect.qualify(evidence, self.transport, slug=slug, build_dir=slug)

    def test_fixture_and_worker_bounds(self):
        with self.assertRaises(ValueError):
            stage.prepare(self.config, self.transport)
        for count in (0, 9, True):
            with self.assertRaises(ValueError):
                self.prepare(download_workers=count)
        self.assertFalse(self.transport.calls)


if __name__ == '__main__':
    unittest.main()
