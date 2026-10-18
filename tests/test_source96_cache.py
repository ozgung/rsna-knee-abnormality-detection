import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from knee import source96_cache as cache,source_span_pilot as pilot,public_training_smoke as smoke

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT','/nonexistent'))
PUBLIC=EVIDENCE/'outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py'
TRAINING=EVIDENCE/'outputs/kaggle/trackA-coatnet224-fold01-20261009/train_knee_original.py'


class PartitionTests(unittest.TestCase):
    def test_shards_exhaustive_disjoint_historical_order_and_invalid(self):
        ids=[f'fixture{i}' for i in range(17)];shards=cache.make_shards(ids)
        self.assertEqual(shards[0],[0,4,8,12,16])
        self.assertEqual(sorted(n for s in shards for n in s),list(range(17)))
        self.assertEqual(len(set(n for s in shards for n in s)),17)
        for bad in (0,-1,True,65):
            with self.assertRaises(ValueError):cache.make_shards(ids,bad)
        for badids in (['a','a'],['../a'],['.'],['/absolute']):
            with self.assertRaises(ValueError):cache.make_shards(badids)
        for bad in (-1,True,'1'):
            with self.assertRaises(ValueError):cache.filename(bad)

    def test_disk_worst_case_has_full_remaining_and_temporary_space(self):
        self.assertEqual(cache.disk_required(3),5*cache.MAX_FILE_BYTES+cache.DISK_RESERVE)
        self.assertGreater(cache.disk_required(3,1000),cache.disk_required(3))
        with tempfile.TemporaryDirectory() as tmp,patch('shutil.disk_usage',return_value=SimpleNamespace(free=1)):
            with self.assertRaises(OSError):cache.require_disk(tmp,3)
        for bad in (-1,True):
            with self.assertRaises(ValueError):cache.disk_required(bad)

    def test_path_traversal_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('../000000.npz','/000000.npz','id.npz','000000.npz/../a'):
                with self.assertRaises(ValueError):cache.safe_file(root,name)
            (root/'000000.npz').symlink_to(root/'target')
            with self.assertRaises(ValueError):cache.safe_file(root,'000000.npz')


@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy required')
class FileTests(unittest.TestCase):
    def arrays(self):
        import numpy as np
        v=np.zeros((96,336,336),np.uint8);v[0,0,0]=1;m=np.zeros(96,np.uint8);m[0]=1
        return v,m

    def test_atomic_roundtrip_corruption_and_raw_equality(self):
        v,m=self.arrays()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'000000.npz';result=cache.atomic_npz(path,v,m)
            self.assertEqual(cache.verify_file(path,result),result)
            self.assertEqual(cache.atomic_npz(path,v,m),result)
            wrong=v.copy();wrong[0,0,0]=2
            with self.assertRaises(ValueError):cache.atomic_npz(path,wrong,m)
            bad=result|{'file_sha256':'0'*64}
            with self.assertRaises(ValueError):cache.verify_file(path,bad)
            path.write_bytes(b'corrupt compressed archive')
            with self.assertRaises(Exception):cache.verify_file(path,result)

    def test_interrupted_write_cleans_temporary_never_replaces_complete(self):
        v,m=self.arrays()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'000000.npz'
            def broken(stream,**kw):stream.write(b'partial');raise RuntimeError('injected interruption')
            with patch('numpy.savez_compressed',side_effect=broken),self.assertRaises(RuntimeError):cache.atomic_npz(path,v,m)
            self.assertFalse(path.exists());self.assertFalse(path.with_name(path.name+'.tmp').exists())
            result=cache.atomic_npz(path,v,m)
            with patch('numpy.savez_compressed',side_effect=broken):self.assertEqual(cache.atomic_npz(path,v,m),result)

    def test_malformed_header_duplicate_members_rejected_before_array_allocation(self):
        import io,zipfile
        import numpy as np
        volume=io.BytesIO();np.lib.format.write_array_header_1_0(volume,{'descr':'|u1','fortran_order':False,'shape':(10**12,)})
        mask=io.BytesIO();np.save(mask,np.zeros(96,np.uint8),allow_pickle=False)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'000000.npz'
            with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('volume.npy',volume.getvalue());z.writestr('mask.npy',mask.getvalue())
            with patch('numpy.load',side_effect=AssertionError('must reject before allocation')),self.assertRaises(ValueError):cache.verify_file(path)

    def test_private_index_membership_hash_and_uint8_mask(self):
        v,m=self.arrays()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);saved=cache.atomic_npz(root/'000000.npz',v,m)
            row={'ordinal':0,'StudyInstanceUID':'fixture','file':'000000.npz',**saved}
            doc={'version':cache.VERSION,'contract_sha256':'contract','entries':[row]}
            self.assertEqual(cache.index_rows(root,doc,'contract',[0],['fixture']),{0:row})
            for broken in (doc|{'contract_sha256':'wrong'},doc|{'entries':[row,row]},doc|{'entries':[row|{'StudyInstanceUID':'different'}]},doc|{'entries':[row|{'file':'../escape.npz'}]}):
                with self.assertRaises(ValueError):cache.index_rows(root,broken,'contract',[0],['fixture'])
            with self.assertRaises(ValueError):cache.array_hashes(v,m.astype(float))


@unittest.skipUnless(PUBLIC.is_file() and TRAINING.is_file(),'Explicit pinned evidence required')
class ActualMembershipTests(unittest.TestCase):
    def test_actual4345_four_partitions_train3475_dev870_and_build(self):
        spec=importlib.util.spec_from_file_location('source96_builder',ROOT/'scripts/build_source96_cache.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'build'
            kwargs=dict(evidence_root=EVIDENCE,public_source=PUBLIC,training_source=TRAINING,shard_index=0)
            receipt=builder.build(dest,'source96-test',**kwargs)
            c=json.loads((dest/'build-contract.json').read_text())
            self.assertEqual([p['studies'] for p in c['partitions']],[1087,1086,1086,1086])
            self.assertEqual(c['membership']['training'],3475);self.assertEqual(c['membership']['development'],870)
            self.assertEqual(c['membership']['eligible'],4345);self.assertTrue(receipt['build_only'])
            metadata=json.loads((dest/'kernel-metadata.json').read_text())
            self.assertTrue(metadata['is_private']);self.assertFalse(metadata['enable_gpu']);self.assertFalse(metadata['enable_internet'])
            nb=json.loads((dest/'notebook.ipynb').read_text());code=''.join(nb['cells'][1]['source']);compile(code,'draft','exec')
            identity=json.loads((EVIDENCE/'outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json').read_text())
            self.assertFalse(any(r['uid'] in code for r in identity['records']))
            self.assertFalse("mmap_mode" in (ROOT/"knee/source96_cache.py").read_text())
            with self.assertRaises(ValueError):builder.build(dest,'source96-test',**kwargs)
            with self.assertRaises(ValueError):builder.build(Path(tmp)/'bad','source96-test',**(kwargs|{'shard_index':4}))


@unittest.skipUnless(PUBLIC.is_file() and TRAINING.is_file() and all(importlib.util.find_spec(p) for p in ('numpy','torch','cv2')),'Pinned sources and CPU backend required')
class StreamingTests(unittest.TestCase):
    arrays=FileTests.arrays
    def test_stream_failure_preserves_completed_resume_checks_and_no_aggregate_ids(self):
        import time
        import numpy as np
        codes=pilot.recipes(PUBLIC.read_text());sampler=pilot.sampler_source(TRAINING.read_text())
        env={'python':__import__('sys').version.split()[0],'device':'cpu','torch_threads':1,'opencv_threads':1,
             'packages':dict.fromkeys(('torch','numpy','pydicom','opencv-python','pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg'),'test')}
        ids=['fixture-identity-one','fixture-identity-two'];v,m=self.arrays()
        contract={'version':cache.VERSION,'config':cache.CONFIG,'shard_count':1,'shard_index':0,'membership':{'eligible':2},
                  'partitions':[{'studies':2}], 'code_revision':'fixture','sources':{},'input_pins':{},'qualified_environment':env}
        def decoded(public,uid,records,raw,span):
            if uid==ids[1]:raise ValueError('Private error '+uid)
            return v.copy(),m.copy(),{'reader_counts':{'decoded_frames':1}}
        with tempfile.TemporaryDirectory() as tmp,patch('importlib.metadata.version',return_value='test'),patch('cv2.getNumThreads',return_value=1):
            root=Path(tmp)/'cache'
            with patch.object(pilot,'decode',side_effect=decoded),self.assertRaisesRegex(RuntimeError,'sanitized aggregate'):
                cache.stream_cache(root,contract,ids,{},Path(tmp),codes,sampler,started=time.monotonic())
            report=json.loads((root/'receipt.json').read_text());self.assertEqual(report['completed_studies'],1)
            self.assertFalse(any(uid in (root/'receipt.json').read_text() for uid in ids));self.assertEqual(report['status'],'ERROR')
            private=json.loads((root/'index-private.json').read_text());self.assertEqual(private['entries'][0]['StudyInstanceUID'],ids[0])
            self.assertTrue((root/'000000.npz').exists());self.assertFalse((root/'000001.npz').exists())
            def healthy(*args):return v.copy(),m.copy(),{'reader_counts':{'decoded_frames':1}}
            with patch.object(pilot,'decode',side_effect=healthy) as decode:
                finished=cache.stream_cache(root,contract,ids,{},Path(tmp),codes,sampler,started=time.monotonic())
                self.assertEqual(decode.call_count,1);self.assertEqual(finished['resumed_studies'],1);self.assertEqual(finished['status'],'PASS')
                history=[json.loads(p.read_text()) for p in (root/'receipt-history').glob('*.json')]
                self.assertTrue(any(h['status']=='ERROR' and h['completed_studies']==1 for h in history))
            with patch.object(pilot,'decode',side_effect=AssertionError('must not decode')):
                again=cache.stream_cache(root,contract,ids,{},Path(tmp),codes,sampler,started=time.monotonic())
                self.assertEqual(again['resumed_studies'],2)
            with patch.object(pilot,'decode',side_effect=AssertionError('must not decode imported cache')):
                imported=cache.stream_cache(Path(tmp)/'imported',contract,ids,{},Path(tmp),codes,sampler,resume_root=root)
                self.assertEqual(imported['status'],'PASS');self.assertEqual(imported['resumed_studies'],2)
            with self.assertRaises(ValueError):cache.stream_cache(root,contract|{'shard_index':1},ids,{},Path(tmp),codes,sampler)

    def test_low_disk_never_decodes(self):
        import time
        env={'python':__import__('sys').version.split()[0],'device':'cpu','torch_threads':1,'opencv_threads':1,
             'packages':dict.fromkeys(('torch','numpy','pydicom','opencv-python','pylibjpeg','pylibjpeg-libjpeg','pylibjpeg-openjpeg'),'test')}
        c={'version':cache.VERSION,'config':cache.CONFIG,'shard_count':1,'shard_index':0,'membership':{},'partitions':[{'studies':1}],
           'code_revision':'fixture','sources':{},'input_pins':{},'qualified_environment':env}
        with tempfile.TemporaryDirectory() as tmp,patch('importlib.metadata.version',return_value='test'),patch('cv2.getNumThreads',return_value=1),patch('shutil.disk_usage',return_value=SimpleNamespace(free=1)),patch.object(pilot,'decode') as decode:
            with self.assertRaises(RuntimeError):cache.stream_cache(Path(tmp)/'cache',c,['fixture'],{},Path(tmp),pilot.recipes(PUBLIC.read_text()),pilot.sampler_source(TRAINING.read_text()),started=time.monotonic())
            decode.assert_not_called()
            report=json.loads((Path(tmp)/'cache/receipt.json').read_text());self.assertEqual(report['failure']['type'],'OSError')
