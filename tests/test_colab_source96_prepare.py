import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tests.test_source96_loader as loader_tests
from knee import public_training_smoke as smoke

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('colab_source96_prepare',ROOT/'scripts/colab_source96_prepare.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)


class FakeTransport:
    def __init__(self,fixture):
        self.states={};self.inventory={};self.data={};self.calls=[];self.snapshots={};self.race=None;self.interrupt=None
        self.descriptors=[]
        for i,old in enumerate(fixture.shards):
            ref=f'ozgung/fixture-source96-shard{i}'
            notebook=fixture.root/f'accepted{i}.ipynb';notebook.write_text(json.dumps({'cells':[{'cell_type':'code','source':[f'# accepted fixture{i}\n']}]})+'\n')
            self.states[ref]={'version':1,'is_private':True,'status':'COMPLETE','cell_sources_sha256':stage.source_cells_sha256(notebook.read_text())}
            descriptor={k:v for k,v in old.items() if k!='root'}|{'ref':ref,'version':1,'notebook_path':str(notebook),'notebook_sha256':smoke.file_sha(notebook)}
            self.descriptors.append(descriptor);items=[]
            for p in Path(old['root']).iterdir():
                if p.name not in stage.METADATA and not p.name.endswith('.npz'):continue
                name='source96_cache/'+p.name;url=f'https://storage.googleapis.com/fixture/{i}/{p.name}?signed=DO_NOT_LOG'
                items.append({'name':name,'url':url});self.data[url]=p.read_bytes()
            self.inventory[ref]=items
    def snapshot(self,ref):
        self.snapshots[ref]=self.snapshots.get(ref,0)+1
        result=copy.deepcopy(self.states[ref])
        if self.race:result=self.race(ref,self.snapshots[ref],result)
        return result
    def list_files(self,ref):return copy.deepcopy(self.inventory[ref])
    def iter_chunks(self,url):
        self.calls.append(url)
        data=self.data[url]
        for i in range(0,len(data),stage.CHUNK_BYTES):
            if self.interrupt and self.interrupt in url:
                yield data[i:i+1];raise OSError('private signedURL '+url)
            yield data[i:i+stage.CHUNK_BYTES]


@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy required')
class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.fixture=loader_tests.LoaderTests();self.fixture.setUp();self.transport=FakeTransport(self.fixture)
        self.output=self.fixture.root/'staged'
        self.config={'version':stage.VERSION,'cache_root':str(self.output),'validation':self.fixture.validation,
                     'sources':[self.transport.descriptors[i] for i in (3,1,0,2)]}
    def tearDown(self):self.fixture.tearDown()
    def prepare(self):return stage.prepare(self.config,self.transport,allow_fixture=True)
    def receipt(self):return json.loads((self.output/'staging-receipt.json').read_text())
    def npz_calls(self):return [u for u in self.transport.calls if '.npz?' in u]
    def test_concurrent_downloads_are_bounded_and_receipts_serialized(self):
        import threading,time
        active=0;peak=0;lock=threading.Lock();caller=threading.get_ident()
        original=self.transport.iter_chunks
        def chunks(url):
            nonlocal active,peak
            if '.npz?' not in url:
                yield from original(url);return
            with lock:active+=1;peak=max(peak,active)
            try:
                time.sleep(.03)
                yield from original(url)
            finally:
                with lock:active-=1
        self.transport.iter_chunks=chunks
        atomic=stage.cache.atomic_json
        def write(path,value):
            self.assertEqual(threading.get_ident(),caller);return atomic(path,value)
        with patch.object(stage.cache,'atomic_json',side_effect=write):
            report=stage.prepare(self.config,self.transport,allow_fixture=True,download_workers=2)
        self.assertEqual(report['status'],'PASS');self.assertEqual(peak,2);self.assertEqual(active,0)
        self.assertEqual(report['completed_files'],4)
        again=self.prepare();self.assertEqual(again['content_sha256'],report['content_sha256'])
        self.assertEqual(len(self.npz_calls()),4)

    def test_concurrent_corruption_preserves_good_files_and_resumes_missing(self):
        ref=self.config['sources'][0]['ref']
        url=next(i['url'] for i in self.transport.inventory[ref] if i['name'].endswith('.npz'))
        original=self.transport.data[url];self.transport.data[url]=original+b'bad'
        with self.assertRaises(RuntimeError):stage.prepare(self.config,self.transport,allow_fixture=True,download_workers=2)
        self.assertEqual(self.receipt()['status'],'ERROR');self.assertFalse(list(self.output.glob('source*/*.tmp')))
        good={p:smoke.file_sha(p) for p in self.output.glob('source*/*.npz')}
        self.transport.data[url]=original
        report=stage.prepare(self.config,self.transport,allow_fixture=True,download_workers=2)
        self.assertEqual(report['status'],'PASS')
        for p,h in good.items():self.assertEqual(smoke.file_sha(p),h)

    def test_parallel_source_race_is_fatal_after_completed_bytes(self):
        first=self.config['sources'][0]['ref']
        self.transport.race=lambda ref,count,result:result|{'version':2} if ref==first and count>=4 else result
        with self.assertRaises(RuntimeError):stage.prepare(self.config,self.transport,allow_fixture=True,download_workers=2)
        self.assertEqual(self.receipt()['status'],'ERROR');self.assertEqual(len(list(self.output.glob('source*/*.npz'))),4)
        self.assertFalse((self.output/'cache96-manifest-private.json').exists())

    def test_worker_count_rejected_before_output(self):
        for count in [0,9,True,1.5]:
            with self.assertRaises(ValueError):stage.prepare(self.config,self.transport,allow_fixture=True,download_workers=count)
        self.assertFalse(self.output.exists())

    def test_direct_stage_shuffled_sources_manifest_and_resume_no_urls_or_ids(self):
        report=self.prepare();self.assertEqual(report['status'],'PASS');self.assertEqual(report['studies'],4)
        manifest=json.loads((self.output/'cache96-manifest-private.json').read_text())
        self.assertEqual([e['StudyInstanceUID'] for e in manifest['entries']],self.fixture.ids)
        self.assertEqual(len(self.npz_calls()),4)
        for p in self.output.glob('source*/*.npz'):self.assertEqual(p.stat().st_mode&0o222,0)
        public=(self.output/'staging-receipt.json').read_text()
        self.assertNotIn('DO_NOT_LOG',public);self.assertNotIn('https://',public)
        self.assertFalse(any(uid in public for uid in self.fixture.ids))
        before=len(self.transport.calls);again=self.prepare()
        self.assertEqual(again['status'],'PASS');self.assertEqual(len(self.transport.calls),before)
        self.assertEqual(report['content_sha256'],again['content_sha256'])

    def test_version_and_code_races_stop_before_metadata_or_after_images(self):
        first=self.config['sources'][0]['ref']
        self.transport.states[first]['version']=2
        with self.assertRaises(RuntimeError):self.prepare()
        self.assertFalse(self.transport.calls)
        self.transport.states[first]['version']=1
        self.transport.race=lambda ref,count,result:result|{'cell_sources_sha256':'0'*64} if ref==first and count>=4 else result
        self.transport.snapshots={}
        with self.assertRaises(RuntimeError):self.prepare()
        self.assertEqual(len(self.npz_calls()),4)
        self.assertEqual(self.receipt()['status'],'ERROR')
        self.assertFalse((self.output/'cache96-manifest-private.json').exists())
        self.assertEqual(len(list(self.output.glob('source*/*.npz'))),4)

    def test_post_assembly_race_preserves_cache_but_never_publishes_ready_manifest(self):
        first=self.config['sources'][0]['ref']
        self.transport.race=lambda ref,count,result:result|{'version':2} if ref==first and count>=5 else result
        with self.assertRaises(RuntimeError):self.prepare()
        self.assertEqual(len(list(self.output.glob('source*/*.npz'))),4)
        self.assertFalse((self.output/'cache96-manifest-private.json').exists())
        self.assertEqual(self.receipt()['status'],'ERROR')

    def test_index_url_path_and_extra_npz_allowlist_fail_before_bulk(self):
        ref=self.config['sources'][0]['ref'];items=self.transport.inventory[ref]
        url=items[0]['url'];items[0]['url']='http://127.0.0.1/secret'
        with self.assertRaises(RuntimeError):self.prepare()
        self.assertFalse(self.npz_calls())
        items[0]['url']=url;items.append({'name':'source96_cache/../../escape.npz','url':url})
        with self.assertRaises(RuntimeError):self.prepare()
        self.assertFalse(self.npz_calls())
        items.pop();items.append({'name':'source96_cache/999999.npz','url':url})
        with self.assertRaises(RuntimeError):self.prepare()
        self.assertFalse(self.npz_calls())

    def test_interruption_preserves_complete_files_cleans_temp_and_retries_missing(self):
        # First source in descriptor order is shard3, so fail on next shard1 file.
        self.transport.interrupt='/1/000001.npz?'
        with self.assertRaises(RuntimeError):self.prepare()
        self.assertEqual(len(list(self.output.glob('source*/*.npz'))),1)
        self.assertFalse(list(self.output.glob('source*/*.npz.tmp')))
        text=(self.output/'staging-receipt.json').read_text();self.assertNotIn('DO_NOT_LOG',text);self.assertNotIn('https://',text)
        self.transport.interrupt=None;report=self.prepare();self.assertEqual(report['status'],'PASS')
        first_url=[u for u in self.npz_calls() if '/3/' in u];self.assertEqual(len(first_url),1)

    def test_metadata_corruption_and_artifact_hash_corruption_are_fatal(self):
        source=self.config['sources'][0]['ref'];items=self.transport.inventory[source]
        receipt_url=next(i['url'] for i in items if i['name'].endswith('receipt.json'))
        original=self.transport.data[receipt_url];self.transport.data[receipt_url]=b'bad'
        with self.assertRaises(RuntimeError):self.prepare()
        self.assertFalse(self.npz_calls());self.assertFalse(list(self.output.glob('source*/*.tmp')))
        self.transport.data[receipt_url]=original
        npz_url=next(i['url'] for i in items if i['name'].endswith('.npz'))
        self.transport.data[npz_url]=self.transport.data[npz_url]+b'corruption'
        with self.assertRaises(RuntimeError):self.prepare()
        self.assertFalse(list(self.output.glob('source*/*.npz.tmp')))
        self.assertEqual(self.receipt()['status'],'ERROR')

    def test_metadata_then_actual_image_disk_projection_refuses_bulk(self):
        # Metadata gate passes, exact remaining-bytes+temporary buffers gate fails.
        with patch('shutil.disk_usage',side_effect=lambda p:SimpleNamespace(free=stage.DISK_RESERVE+4*3*stage.METADATA_MAX_BYTES if len(self.transport.calls)<12 else stage.DISK_RESERVE)):
            with self.assertRaises(RuntimeError):self.prepare()
        self.assertEqual(len(self.transport.calls),12);self.assertFalse(self.npz_calls())
        self.assertEqual(self.receipt()['failure']['type'],'OSError')

    def test_production_and_accepted_notebook_pins_are_explicit(self):
        with self.assertRaises(ValueError):stage.prepare(self.config,self.transport)
        self.config['sources'][0]['notebook_sha256']='0'*64
        with self.assertRaises(ValueError):self.prepare()
        self.assertFalse(self.transport.calls)
        for url in ('file:///tmp/secret','https://user:pass@storage.googleapis.com/x','https://localhost/x','https://storage.googleapis.com:444/x'):
            with self.assertRaises(ValueError):stage.checked_url(url)
