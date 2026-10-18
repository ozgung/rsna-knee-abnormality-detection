import copy
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from knee import source96_loader as loader,source96_cache as cache,source_span_pilot as pilot,public_training_smoke as smoke

EVIDENCE=Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT','/nonexistent'))
TRAINING=EVIDENCE/'outputs/kaggle/trackA-coatnet224-fold01-20261009/train_knee_original.py'


def worker_probe(volumes,masks,queue):
    """Top-level importable spawn target: no shared handle or inherited pair reuse."""
    try:
        before_pid=volumes.store.pid
        with patch.object(loader,'_load_pair',wraps=loader._load_pair) as load:
            m=masks[0];v=volumes[0];masks[0]
            queue.put({'pid':os.getpid(),'before_pid':before_pid,'store_pid':volumes.store.pid,
                       'loads':load.call_count,'volume_sha256':smoke.digest(v.tobytes()),
                       'mask_sha256':smoke.digest(m.tobytes()),'readonly':not v.flags.writeable})
    except BaseException as error:queue.put({'error':type(error).__name__})


@unittest.skipUnless(importlib.util.find_spec('numpy'),'NumPy required')
class LoaderTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.ids=[f'fixture-study-{i}' for i in range(4)]
        self.folds={u:(0 if i==0 else 1) for i,u in enumerate(self.ids)}
        train=self.ids[1:];dev=self.ids[:1]
        self.membership={'eligible':4,'training':3,'development':1,
            'ordered_eligible_sha256':loader.digest_document(self.ids),
            'ordered_training_sha256':loader.digest_document(train),'ordered_development_sha256':loader.digest_document(dev)}
        self.validation={'fixture':True,'membership':self.membership,'folds':self.folds}
        self.shards=[];self.dense=[];self.dense_masks=[]
        parts=[{'studies':1,'ordinals_sha256':loader.digest_document([i]),'ordered_ids_sha256':loader.digest_document([self.ids[i]])} for i in range(4)]
        for i in range(4):
            root=self.root/f'shard{i}';root.mkdir()
            # Real shape, deterministic pixels; missing interiors and short masks.
            base=(np.arange(336,dtype=np.uint16)%250+1).astype(np.uint8)
            v=np.broadcast_to(base,(96,336,336)).copy();v=(v+i).astype(np.uint8);m=np.ones(96,np.uint8)
            if i==1:v[27:32]=0;m[27:32]=0
            if i==2:v[2:]=0;m[2:]=0
            if i==3:v[:95]=0;m[:95]=0
            self.dense.append(v);self.dense_masks.append(m)
            saved=cache.atomic_npz(root/cache.filename(i),v,m)
            contract={'version':cache.VERSION,'config':cache.CONFIG,'membership':self.membership,'shard_index':i,'shard_count':4,
                'partitions':parts,'reader_hashes':{'wide':'1'*64},'sampler_sha256':'2'*64,'input_pins':{'identity':'3'*64},
                'qualified_environment':{'device':'cpu'},'sources':{'reader':'4'*64}}
            contract_sha=loader.digest_document(contract)
            index={'version':cache.VERSION,'contract_sha256':contract_sha,'entries':[{'ordinal':i,'StudyInstanceUID':self.ids[i],
                'file':cache.filename(i),**saved,'windows':{}}]}
            index_sha=loader.digest_document(index)
            # Pinned hashes must be actual file-byte hashes, not canonical hashes of pretty JSON.
            cache.atomic_json(root/'run-contract.json',contract);cache.atomic_json(root/'index-private.json',index)
            index_sha=smoke.file_sha(root/'index-private.json');contract_file_sha=smoke.file_sha(root/'run-contract.json')
            receipt={'status':'PASS','version':cache.VERSION,'contract_sha256':contract_sha,'private_index_sha256':index_sha,
                     'completed_studies':1,'shard_index':i,'shard_count':4}
            cache.atomic_json(root/'receipt.json',receipt)
            self.shards.append({'root':str(root),'contract_file_sha256':contract_file_sha,'index_sha256':index_sha,'receipt_sha256':smoke.file_sha(root/'receipt.json')})

    def tearDown(self):self.tmp.cleanup()

    def manifest(self):return loader.assemble([self.shards[i] for i in (3,1,0,2)],self.validation)

    def opened(self,manifest=None):
        manifest=self.manifest() if manifest is None else manifest
        return loader.open_corpus(manifest,expected_manifest_sha256=manifest['manifest_sha256'],allow_fixture=True)

    def test_global_order_profiles_dense_preload_and_pair_cache(self):
        import numpy as np
        manifest=self.manifest();self.assertEqual([e['StudyInstanceUID'] for e in manifest['entries']],self.ids)
        with self.assertRaises(ValueError):loader.open_corpus(manifest,expected_manifest_sha256=manifest['manifest_sha256'])
        with self.assertRaises(ValueError):loader.assemble(self.shards,self.validation|{'fixture':False})
        volumes,masks,ids=self.opened(manifest)
        self.assertEqual(ids.tolist(),self.ids);self.assertEqual(volumes.shape,(4,96,336,336));self.assertEqual(masks.shape,(4,96))
        with patch.object(loader,'_load_pair',wraps=loader._load_pair) as load:
            masks[0];volumes[0];masks[0];self.assertEqual(load.call_count,1)
            volumes[1];masks[1];self.assertEqual(load.call_count,2)
            volumes[0];self.assertEqual(load.call_count,3)
        self.assertFalse(volumes[0].flags.writeable)
        for proxy in (volumes,masks):
            with self.assertRaises(ValueError):np.asarray(proxy)
            with self.assertRaises(IndexError):proxy[True]
        with self.assertRaises(ValueError):loader.open_corpus(manifest,expected_manifest_sha256=manifest['manifest_sha256'],allow_fixture=True,preload=True)
        with patch.dict(os.environ,{'RAPTOR_PRELOAD':'1'}),self.assertRaises(ValueError):self.opened(manifest)

    def test_shard_index_hash_membership_and_raw_corruption_fail_closed(self):
        with self.assertRaises(ValueError):loader.assemble(self.shards[:3],self.validation)
        with self.assertRaises(ValueError):loader.assemble(self.shards[:3]+[self.shards[0]],self.validation)
        bad=copy.deepcopy(self.validation);bad['membership']['ordered_eligible_sha256']='0'*64
        with self.assertRaises(ValueError):loader.assemble(self.shards,bad)
        bad=copy.deepcopy(self.validation);bad['folds'][self.ids[0]]=1
        with self.assertRaises(ValueError):loader.assemble(self.shards,bad)
        descriptor=self.shards[0]|{'index_sha256':'0'*64}
        with self.assertRaises(ValueError):loader.assemble([descriptor]+self.shards[1:],self.validation)
        path=Path(self.shards[0]['root'])/'000000.npz';path.write_bytes(b'corrupt')
        with self.assertRaises(Exception):loader.assemble(self.shards,self.validation)

    def test_raw_hash_and_full_partition_checked_with_rebound_metadata_pins(self):
        # File bytes still valid: a falsely claimed raw hash must fail verification.
        root=Path(self.shards[0]['root']);index=json.loads((root/'index-private.json').read_text())
        index['entries'][0]['volume_sha256']='0'*64;cache.atomic_json(root/'index-private.json',index)
        new_index_sha=smoke.file_sha(root/'index-private.json')
        receipt=json.loads((root/'receipt.json').read_text());receipt['private_index_sha256']=new_index_sha
        cache.atomic_json(root/'receipt.json',receipt)
        changed=self.shards[0]|{'index_sha256':new_index_sha,'receipt_sha256':smoke.file_sha(root/'receipt.json')}
        with self.assertRaisesRegex(ValueError,'hashes/bytes'):loader.assemble([changed]+self.shards[1:],self.validation)
        # Fresh helper fixture for the separate ordered coverage condition.
        index['entries'][0]['ordinal']=1;cache.atomic_json(root/'index-private.json',index)
        receipt['private_index_sha256']=smoke.file_sha(root/'index-private.json');cache.atomic_json(root/'receipt.json',receipt)
        changed=changed|{'index_sha256':receipt['private_index_sha256'],'receipt_sha256':smoke.file_sha(root/'receipt.json')}
        with self.assertRaisesRegex(ValueError,'ordered coverage'):loader.assemble([changed]+self.shards[1:],self.validation)

    def test_missing_symlink_and_mutation_after_validation_on_cache_hits(self):
        manifest=self.manifest();volumes,masks,_=self.opened(manifest);masks[0]
        path=Path(self.shards[0]['root'])/'000000.npz'
        path.write_bytes(path.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError,'mutated'):volumes[0]
        path.unlink()
        with self.assertRaises(ValueError):volumes[0]
        path.symlink_to(Path(self.shards[1]['root'])/'000001.npz')
        with self.assertRaises(ValueError):volumes[0]
        changed=copy.deepcopy(manifest);changed['entries'][1]['volume_sha256']='0'*64
        with self.assertRaises(ValueError):loader.open_corpus(changed,expected_manifest_sha256=manifest['manifest_sha256'],allow_fixture=True)

    def test_staging_hashes_once_then_no_per_access_hashes(self):
        with patch.object(cache,'verify_file',wraps=cache.verify_file) as verify:manifest=self.manifest();self.assertEqual(verify.call_count,4)
        volumes,masks,_=self.opened(manifest)
        with patch.object(cache,'verify_file',side_effect=AssertionError('No SHA re-verification')),patch.object(smoke,'file_sha',side_effect=AssertionError('No file rehash')):
            masks[0];volumes[0];masks[1];volumes[1]

    def test_fork_and_spawn_worker_cache_and_serialization(self):
        manifest=self.manifest();volumes,masks,_=self.opened(manifest);volumes[0]
        expected=manifest['entries'][0]
        methods=[m for m in ('fork','spawn') if m in multiprocessing.get_all_start_methods()]
        if not methods:self.skipTest('fork/spawn unavailable')
        for method in methods:
            context=multiprocessing.get_context(method);queue=context.Queue();process=context.Process(target=worker_probe,args=(volumes,masks,queue))
            process.start()
            try:
                result=queue.get(timeout=30);process.join(timeout=30)
                self.assertFalse(process.is_alive());self.assertEqual(process.exitcode,0)
                self.assertNotIn('error',result);self.assertEqual(result['pid'],result['store_pid'])
                self.assertNotEqual(result['pid'],os.getpid());self.assertEqual(result['loads'],1)
                self.assertEqual(result['volume_sha256'],expected['volume_sha256']);self.assertEqual(result['mask_sha256'],expected['mask_sha256'])
                self.assertTrue(result['readonly'])
                if method=='spawn':self.assertIsNone(result['before_pid'])
                else:self.assertEqual(result['before_pid'],os.getpid())
            finally:
                if process.is_alive():process.terminate();process.join(timeout=5)
                queue.close();queue.join_thread()

    @unittest.skipUnless(TRAINING.is_file() and importlib.util.find_spec('torch'),'Pinned StudyWindows and torch required')
    def test_unchanged_studywindows_augmented_k8_eval_k16_rng_exact(self):
        import numpy as np
        import torch
        torch.set_num_threads(1)
        sampler=pilot.sampler_class(pilot.sampler_source(TRAINING.read_text()))
        volumes,masks,ids=self.opened();dense=np.stack(self.dense);dense_masks=np.stack(self.dense_masks)
        labels={u:np.zeros(12,np.float32) for u in ids.tolist()};id2row={u:i for i,u in enumerate(ids.tolist())}
        for train,k in ((True,8),(False,16)):
            a=sampler('',ids.tolist(),id2row,labels,224,k,train=train,aug=True,norm='imagenet')
            b=sampler('',ids.tolist(),id2row,labels,224,k,train=train,aug=True,norm='imagenet')
            a.vols,a.masks=dense,dense_masks;b.vols,b.masks=volumes,masks
            for i in range(4):
                random.seed(42+i);np.random.seed(42+i);torch.manual_seed(42+i)
                state=(random.getstate(),np.random.get_state(),torch.get_rng_state())
                x,y=a[i];after=(random.getstate(),np.random.get_state(),torch.get_rng_state())
                random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
                xx,yy=b[i]
                self.assertTrue(torch.equal(x,xx));self.assertTrue(torch.equal(y,yy));self.assertTrue(torch.isfinite(xx).all())
                self.assertEqual(random.getstate(),after[0]);self.assertTrue(np.array_equal(np.random.get_state()[1],after[1][1]))
                self.assertTrue(torch.equal(torch.get_rng_state(),after[2]))
