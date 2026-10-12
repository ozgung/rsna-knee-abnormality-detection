import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('inputs96',ROOT/'scripts/colab_source96_inputs.py')
inputs=importlib.util.module_from_spec(spec);spec.loader.exec_module(inputs)
EVIDENCE=Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT','/nonexistent'))

class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.package=self.root/'pkg';self.package.mkdir();self.output=self.root/'input'
        for name in ('train_knee_original.py','folds_v0.json'):(self.package/name).write_bytes(b'fixture '+name.encode())
        self.manifest={'original_sha256':inputs.sha(self.package/'train_knee_original.py'),'folds_sha256':inputs.sha(self.package/'folds_v0.json')}
        (self.package/'manifest.json').write_text(json.dumps(self.manifest));self.msha=inputs.sha(self.package/'manifest.json')
        self.data={'labels_v1_blend.parquet':b'fixture labels','train_labels_v1.parquet':b'fixture truth','train.csv':b'fixture official csv'}
        self.pins={name:('dataset' if name.endswith('.parquet') else 'competition','fixture/source',hashlib.sha256(data).hexdigest()) for name,data in self.data.items()}
        self.calls=[];outer=self
        class Transport:
            def fetch(self,kind,source,name,target):
                outer.calls.append((kind,source,name));print('https://secret-signed-url?token=hidden');target.write_bytes(outer.data[name])
        self.transport=Transport()
    def tearDown(self):self.tmp.cleanup()
    def prepare(self,**kw):
        args={'transport':self.transport,'remote_pins':self.pins,'disk_usage':lambda _:SimpleNamespace(free=10**10),**kw}
        return inputs._prepare(self.output,self.package,self.msha,**args)
    def test_success_exact_files_receipt_idempotent_and_no_console(self):
        import contextlib
        console=io.StringIO()
        with contextlib.redirect_stdout(console):r=self.prepare()
        self.assertEqual(console.getvalue(),'')
        self.assertEqual(r['status'],'PASS');self.assertEqual(len(r['files']),5)
        for name,data in self.data.items():self.assertEqual((self.output/name).read_bytes(),data)
        self.assertEqual((self.output/'train_knee.py').read_bytes(),(self.package/'train_knee_original.py').read_bytes())
        self.assertEqual((self.output/'folds_v0.json').read_bytes(),(self.package/'folds_v0.json').read_bytes())
        before=(self.output/'source96-inputs-receipt.json').read_bytes();self.calls.clear()
        self.prepare();self.assertEqual(self.calls,[]);self.assertEqual((self.output/'source96-inputs-receipt.json').read_bytes(),before)
        self.assertEqual(len(list((self.output/'.source96-input-attempts').glob('*/attempt-receipt.json'))),2)
    def test_archive_gzip_zip_and_sdk_logs_private(self):
        import contextlib
        outer=self
        class Archives:
            def fetch(self,kind,source,name,target):
                print('private SDK signed-url diagnostic')
                if name=='train.csv':
                    b=io.BytesIO()
                    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:z.writestr(name,outer.data[name])
                    target.write_bytes(b.getvalue())
                else:target.write_bytes(gzip.compress(outer.data[name]))
        console=io.StringIO()
        with contextlib.redirect_stdout(console):self.prepare(transport=Archives())
        self.assertEqual(console.getvalue(),'')
        logs=list((self.output/'.source96-input-attempts').glob('*/remote-*/sdk-diagnostics.log'))
        self.assertTrue(logs);self.assertIn('signed-url',logs[0].read_text());self.assertEqual(logs[0].stat().st_mode&0o777,0o600)
    def test_corrupt_payload_preserved_no_success_receipt(self):
        self.data['labels_v1_blend.parquet']=b'corrupt bytes'
        with self.assertRaises(ValueError):self.prepare()
        self.assertFalse((self.output/'source96-inputs-receipt.json').exists());self.assertFalse((self.output/'labels_v1_blend.parquet').exists())
        candidates=list((self.output/'.source96-input-attempts').glob('*/remote-*/candidate.bin'))
        self.assertEqual(candidates[0].read_bytes(),b'corrupt bytes')
        receipts=list((self.output/'.source96-input-attempts').glob('*/attempt-receipt.json'))
        self.assertEqual(json.loads(receipts[0].read_text())['status'],'FAIL')
    def test_interruption_preserves_partial_and_completed_files(self):
        outer=self
        class Interrupted:
            def fetch(self,kind,source,name,target):
                if name=='labels_v1_blend.parquet':target.write_bytes(outer.data[name]);return
                target.write_bytes(b'partial');raise OSError('transport interrupted')
        with self.assertRaises(OSError):self.prepare(transport=Interrupted())
        self.assertEqual((self.output/'labels_v1_blend.parquet').read_bytes(),self.data['labels_v1_blend.parquet'])
        self.assertTrue(any(p.read_bytes()==b'partial' for p in (self.output/'.source96-input-attempts').glob('*/remote-*/payload.bin')))
        self.prepare();self.assertEqual(self.calls[0][2],'train_labels_v1.parquet')
    def test_different_existing_and_symlinks_never_overwritten(self):
        self.output.mkdir();existing=self.output/'train.csv';existing.write_bytes(b'old evidence')
        with self.assertRaises(ValueError):self.prepare()
        self.assertEqual(existing.read_bytes(),b'old evidence');self.assertEqual(self.calls,[])
        existing.unlink();outside=self.root/'outside';outside.write_bytes(b'preserve');existing.symlink_to(outside)
        with self.assertRaises(ValueError):self.prepare()
        self.assertEqual(outside.read_bytes(),b'preserve');self.assertTrue(existing.is_symlink())
    def test_zip_traversal_extra_members_and_symlink_rejected(self):
        for mode in ('traversal','extra','symlink'):
            payload=self.root/(mode+'.zip');target=self.root/(mode+'.bin')
            with zipfile.ZipFile(payload,'w') as z:
                if mode=='traversal':z.writestr('../train.csv',b'x')
                elif mode=='extra':z.writestr('train.csv',b'x');z.writestr('other',b'x')
                else:
                    info=zipfile.ZipInfo('train.csv');info.external_attr=(0o120777<<16);z.writestr(info,b'outside')
            with self.assertRaises(ValueError):inputs.decode(payload,target,'train.csv')
            self.assertFalse(target.exists())
    def test_manifest_disk_package_and_publish_race_fail_closed(self):
        with self.assertRaises(OSError):self.prepare(disk_usage=lambda _:SimpleNamespace(free=0))
        self.assertEqual(self.calls,[])
        (self.package/'train_knee_original.py').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.prepare()
        candidate=self.root/'candidate';candidate.write_bytes(b'new');target=self.root/'target';target.write_bytes(b'old')
        with self.assertRaises(FileExistsError):inputs.publish(candidate,target)
        self.assertEqual(target.read_bytes(),b'old');self.assertEqual(candidate.read_bytes(),b'new')
    def test_decoded_size_bound_and_manifest_pin(self):
        from unittest.mock import patch
        payload=self.root/'small.gz';payload.write_bytes(gzip.compress(b'x'*64));candidate=self.root/'expanded.bin'
        with patch.object(inputs,'REMOTE_LIMIT',32),self.assertRaises(ValueError):inputs.decode(payload,candidate,'train.csv')
        self.assertLessEqual(candidate.stat().st_size,32)
        with self.assertRaises(ValueError):inputs._prepare(self.output,self.package,'0'*64,transport=self.transport)
        self.assertEqual(self.calls,[])

    @unittest.skipUnless((EVIDENCE/'outputs/kaggle/source96-training-20261012-reviewed-v1/manifest.json').exists(),'Reviewed actual package unavailable')
    def test_actual_reviewed_package_copy_and_pins(self):
        package=EVIDENCE/'outputs/kaggle/source96-training-20261012-reviewed-v1'
        r=inputs._prepare(self.output,package,inputs.sha(package/'manifest.json'),transport=self.transport,remote_pins=self.pins,disk_usage=lambda _:SimpleNamespace(free=10**10))
        self.assertEqual(r['files']['train_knee.py']['sha256'],'d39966791ef1c48d9317656e8346c60cae1c064192c87b3f350e298ec4c567e8')
        self.assertEqual(r['files']['folds_v0.json']['sha256'],'12bebc897a549d660cf5e6ab597c9a8d8f84dc887b4abd4c99b67fb4a25cb3ae')
        self.assertEqual(inputs.REMOTE_PINS['train.csv'][2],'8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33')
