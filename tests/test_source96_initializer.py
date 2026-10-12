import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('initializer96',ROOT/'scripts/prepare_source96_initializer.py')
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
EVIDENCE=Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT','/nonexistent'))

class PinTests(unittest.TestCase):
    @unittest.skipUnless((EVIDENCE/'outputs/analysis/source96-initializer-20261012/hf-model-metadata.json').exists(),'Pinned HF metadata unavailable')
    def test_actual_parent_metadata(self):
        m=json.loads((EVIDENCE/'outputs/analysis/source96-initializer-20261012/hf-model-metadata.json').read_text())
        f=next(f for f in m['siblings'] if f['rfilename']=='model.safetensors')
        self.assertEqual(m['sha'],prep.REVISION);self.assertEqual(m['id'],prep.REPO)
        self.assertEqual(f['lfs']['sha256'],prep.SHA256);self.assertEqual(f['size'],prep.SIZE)
        self.assertEqual(prep.TIMM_VERSION,'1.0.26')
    def test_fixed_pretrained_false_factory_and_timm_version(self):
        calls=[]
        fake=SimpleNamespace(__version__='1.0.26',create_model=lambda *a,**kw:calls.append((a,kw)))
        with patch.dict('sys.modules',{'timm':fake}):prep.model_factory(0)
        self.assertEqual(calls,[((prep.ARCH,),{'pretrained':False,'num_classes':0,'in_chans':3,'global_pool':'avg'})])
        fake.__version__='1.0.27'
        with patch.dict('sys.modules',{'timm':fake}),self.assertRaises(RuntimeError):prep.model_factory(0)
    def test_non_https_redirect_closed(self):
        class Response:
            closed=False
            def geturl(self):return 'http://invalid.example/file'
            def close(self):self.closed=True
        r=Response()
        with patch.object(prep.urllib.request,'urlopen',return_value=r),self.assertRaises(ValueError):prep.download('https://huggingface.co/fixed')
        self.assertTrue(r.closed)

@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU PyTorch required for normalization fixture')
class InitializerTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch;self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        class Model(torch.nn.Module):
            def __init__(self,num_classes):
                super().__init__();self.body=torch.nn.Linear(2,3);self.head=torch.nn.Module()
                self.head.fc=torch.nn.Linear(3,num_classes) if num_classes else torch.nn.Identity()
            def get_classifier(self):return self.head.fc
        self.Model=Model;self.state={k:torch.ones_like(v) for k,v in Model(1000).state_dict().items()}
        self.bytes=b'bounded synthetic safetensors transport fixture'
        self.pin={**prep.PIN,'sha256':hashlib.sha256(self.bytes).hexdigest(),'size':len(self.bytes)}
        self.fetch_calls=[]
        def fetch(url):self.fetch_calls.append(url);return io.BytesIO(self.bytes)
        self.fetch=fetch
    def tearDown(self):self.tmp.cleanup()
    def prepare(self,**kw):
        params={'fetch':self.fetch,'factory':self.Model,'load':lambda _:self.state,'disk_usage':lambda _:SimpleNamespace(free=10**10),**kw}
        return prep._prepare(self.root,self.pin,**params)
    def test_prepare_strict_reload_descriptor_and_rng(self):
        import numpy as np
        from knee import source96_training as guard
        random.seed(9);np.random.seed(9);self.torch.manual_seed(9)
        snapshot=(random.getstate(),np.random.get_state(),self.torch.get_rng_state())
        result=self.prepare()
        actual=(random.random(),float(np.random.rand()),float(self.torch.rand(1)))
        random.setstate(snapshot[0]);np.random.set_state(snapshot[1]);self.torch.set_rng_state(snapshot[2])
        self.assertEqual(actual,(random.random(),float(np.random.rand()),float(self.torch.rand(1))))
        self.assertEqual(self.fetch_calls,['https://huggingface.co/'+prep.REPO+'/resolve/'+prep.REVISION+'/model.safetensors'])
        d=json.loads((self.root/'initializer.json').read_text());self.assertEqual(d['audit']['omitted_keys'],['head.fc.bias','head.fc.weight'])
        self.assertEqual(d['timm'],'1.0.26');self.assertEqual(d['audit']['backbone_keys'],2)
        with patch.dict(os.environ,{'SOURCE96_INITIALIZER_DESCRIPTOR':result['descriptor'],'SOURCE96_INITIALIZER_DESCRIPTOR_SHA256':result['descriptor_sha256']}):
            loaded=guard.generic_backbone(self.Model(0))
        self.assertEqual(loaded['backbone_state_sha256'],result['backbone_tensor_sha256'])
        with self.assertRaises(ValueError):self.prepare()
        self.assertFalse(list(self.root.glob('*.part')))
    def test_missing_extra_resized_nonfinite_head_and_backbone_rejected(self):
        variants=[]
        missing=copy.copy(self.state);del missing['body.weight'];variants.append(missing)
        variants+=[{**self.state,'unexpected':self.torch.zeros(1)},
                   {**self.state,'body.weight':self.torch.zeros(4,2)},
                   {**self.state,'body.weight':self.torch.full((3,2),float('nan'))},
                   {**self.state,'head.fc.bias':self.torch.zeros(999)},
                   {**self.state,'head.fc.weight':self.torch.full((1000,3),float('inf'))}]
        for state in variants:
            with self.subTest(keys=list(state)),self.assertRaises(ValueError):prep.normalize(state,self.Model(1000),self.Model(0))
        full=self.Model(1000);full.get_classifier=lambda:full.body
        with self.assertRaises(ValueError):prep.normalize(self.state,full,self.Model(0))
    def test_disk_and_path_rejected_before_fetch(self):
        with self.assertRaises(OSError):self.prepare(disk_usage=lambda _:SimpleNamespace(free=0))
        self.assertEqual(self.fetch_calls,[])
        target=self.root/'outside';target.write_bytes(b'preserve')
        (self.root/'model.safetensors.part').symlink_to(target)
        with self.assertRaises(ValueError):self.prepare()
        self.assertEqual(target.read_bytes(),b'preserve');self.assertEqual(self.fetch_calls,[])
    def test_bad_hash_oversize_interruption_cleanup_and_origin_resume(self):
        for payload in (b'wrong',self.bytes+b'extra'):
            with self.assertRaises(ValueError):self.prepare(fetch=lambda _:io.BytesIO(payload))
            self.assertFalse((self.root/'model.safetensors').exists());self.assertFalse(list(self.root.glob('*.part')))
        class Interrupted(io.BytesIO):
            def read(self,n):raise OSError('interrupted')
        with self.assertRaises(OSError):self.prepare(fetch=lambda _:Interrupted(self.bytes))
        self.assertFalse(list(self.root.glob('*.part')))
        (self.root/'model.safetensors').write_bytes(self.bytes)
        result=self.prepare(fetch=lambda _:self.fail('Verified source should be reused'))
        self.assertTrue(Path(result['descriptor']).exists())
    def test_conversion_failure_restores_rng_and_retains_origin(self):
        import numpy as np
        snapshot=(random.getstate(),np.random.get_state(),self.torch.get_rng_state())
        def factory(n):
            random.random();np.random.rand();self.torch.rand(1)
            return self.Model(n)
        with self.assertRaises(ValueError):self.prepare(factory=factory,load=lambda _: {})
        self.assertEqual(random.getstate(),snapshot[0]);self.assertEqual(np.random.get_state()[1].tolist(),snapshot[1][1].tolist())
        self.assertTrue(self.torch.equal(self.torch.get_rng_state(),snapshot[2]))
        self.assertEqual((self.root/'model.safetensors').read_bytes(),self.bytes)
        self.assertFalse((self.root/'initializer.json').exists());self.assertFalse((self.root/'backbone.pt').exists())
