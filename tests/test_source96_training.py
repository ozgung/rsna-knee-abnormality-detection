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
from knee import source96_training as guard, public_training_smoke as smoke

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT','/nonexistent'))
spec=importlib.util.spec_from_file_location('build96training',ROOT/'scripts/build_source96_training.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)

class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_strict_cuda_and_threads(self):
        fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        with self.assertRaisesRegex(RuntimeError,'CUDA required'):guard.strict_cuda(fake)
        fake.cuda.is_available=lambda:True
        calls=[];fake.set_num_threads=calls.append
        self.assertEqual(guard.strict_cuda(fake),'cuda');self.assertEqual(calls,[1])
    def test_export_qualification(self):
        ckpt=self.root/'state.pt';ckpt.write_bytes(b'fixed')
        (self.root/'raptor_ft_fold0.json').write_text(json.dumps({'partial':False,'hist':[{}]*10}))
        epochs=[1,4,8]
        guard.export_guard(self.root,0,ckpt,'swa_topk',epochs,False,False)
        with self.assertRaises(FileNotFoundError):guard.export_guard(self.root,0,ckpt,'swa_topk',epochs,False,True)
        record={'export':{'k_eval':16},'checkpoint_sha256':smoke.file_sha(ckpt),'selected_epochs':epochs}
        (self.root/'run_fold0.json').write_text(json.dumps(record))
        guard.export_guard(self.root,0,ckpt,'swa_topk',epochs,False,True)
        ckpt.write_bytes(b'changed')
        with self.assertRaises(ValueError):guard.export_guard(self.root,0,ckpt,'swa_topk',epochs,False,True)
        for kind,partial in [('best_epoch',False),('swa_topk',True)]:
            with self.assertRaises(ValueError):guard.export_guard(self.root,0,ckpt,kind,epochs,partial,False)
    def test_worker_choice_before_read(self):
        with self.assertRaises(ValueError):builder.build(self.root/'bad',evidence_root=EVIDENCE,workers=10)
        self.assertFalse((self.root/'bad').exists())
    @unittest.skipUnless((EVIDENCE/'outputs/kaggle/trackA-coatnet224-fold01-20261009/train_knee_patched.py').exists(),'Pinned historical assets unavailable')
    def test_actual_pinned_build_and_core(self):
        out=self.root/'package';r=builder.build(out,evidence_root=EVIDENCE)
        old=EVIDENCE/'outputs/kaggle/trackA-coatnet224-fold01-20261009'
        trainer=(out/'train_knee_patched.py').read_text();entry=(out/'tracka_entry.py').read_text()
        self.assertEqual(smoke.file_sha(old/'train_knee_patched.py'),builder.TRAIN_SHA)
        self.assertEqual(builder.scientific_ast(trainer),builder.scientific_ast((old/'train_knee_patched.py').read_text()))
        self.assertTrue(r['scientific_ast_equal']);self.assertEqual(r['policy']['workers'],2)
        self.assertEqual(r['policy']['frozen']['time_budget_h'],0)
        self.assertIn('source96.save_resume(rp',trainer);self.assertIn('source96.restore_rng(R_["rng"])',trainer)
        self.assertIn('SOURCE96_RESUME_SHA256',trainer);self.assertIn('pretrained=False',trainer)
        self.assertNotIn('pretrained=use_timm',trainer);self.assertNotIn('dev = "cuda" if torch.cuda.is_available() else "cpu"',trainer+entry)
        self.assertIn('k_eval = 94 if secondary else 16',entry)
        self.assertIn('export_main(int(sys.argv[2]), secondary=True)',entry)
        namespace={'source96':SimpleNamespace(staged_corpus=lambda:('v','m',['synthetic']))}
        for name in ('_open_corpus','_open_ids'):
            exec(builder.function_source(trainer,name),namespace)
        self.assertEqual(namespace['_open_corpus'](),('v','m'));self.assertEqual(namespace['_open_ids'](),['synthetic'])
        args=SimpleNamespace(**guard.FROZEN,workers=2)
        guard.require_recipe(args,out/'train_knee_patched.py')
        args.k_eval=94
        with self.assertRaises(ValueError):guard.require_recipe(args,out/'train_knee_patched.py')
        (out/'knee/source96_loader.py').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'runtime module'):guard.recipe(out/'train_knee_patched.py')
        with self.assertRaises(ValueError):builder.build(out,evidence_root=EVIDENCE)
    def test_staging_pins_and_fixture_boundary(self):
        manifest={'content_sha256':'a'*64,'membership':{},'fixture':True,'sources':[]}
        path=self.root/'cache96-manifest-private.json';path.write_text(json.dumps(manifest))
        receipt={'status':'PASS','manifest_sha256':'b'*64,'content_sha256':'a'*64,'manifest_file_sha256':smoke.file_sha(path)}
        rp=self.root/'staging-receipt.json';rp.write_text(json.dumps(receipt))
        env={'SOURCE96_STAGE_ROOT':str(self.root),'SOURCE96_MANIFEST_SHA256':'b'*64,'SOURCE96_CONTENT_SHA256':'a'*64,'SOURCE96_STAGE_RECEIPT_SHA256':smoke.file_sha(rp)}
        with patch.dict(os.environ,env),patch.object(guard,'_CORPUS',None),patch.object(guard,'_CACHE_IDENTITY',None),patch.object(guard.loader,'open_corpus',return_value=('v','m',['fixture'])) as open_:
            self.assertEqual(guard.staged_corpus(allow_fixture=True),('v','m',['fixture']))
            self.assertTrue(open_.call_args.kwargs['allow_fixture'])
            with self.assertRaisesRegex(ValueError,'Fixture corpus'):guard.staged_corpus()
        with patch.dict(os.environ,{**env,'SOURCE96_CONTENT_SHA256':'c'*64}),patch.object(guard,'_CORPUS',None):
            with self.assertRaisesRegex(ValueError,'staging PASS'):guard.staged_corpus()
        path.write_text('{}')
        with patch.dict(os.environ,env),patch.object(guard,'_CORPUS',None):
            with self.assertRaisesRegex(ValueError,'Pinned JSON'):guard.staged_corpus()

@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch required for CPU state safeguards')
class TensorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_strict_initializer_schema(self):
        import torch
        bb=torch.nn.Linear(2,2)
        def load(state,**override):
            wp=self.root/'generic.pt';torch.save(state,wp)
            desc={'kind':'imagenet_backbone','arch':guard.ARCH,'pretrained_tag':'sw_in12k_ft_in1k','hf_repo':'timm/'+guard.ARCH,'hf_revision':'1'*40,'origin_file_sha256':'2'*64,'weights_file':str(wp),'weights_sha256':smoke.file_sha(wp),**override}
            dp=self.root/'generic.json';dp.write_text(json.dumps(desc))
            with patch.dict(os.environ,{'SOURCE96_INITIALIZER_DESCRIPTOR':str(dp),'SOURCE96_INITIALIZER_DESCRIPTOR_SHA256':smoke.file_sha(dp)}):return guard.generic_backbone(bb)
        good={k:torch.ones_like(v) for k,v in bb.state_dict().items()}
        record=load(good);self.assertEqual(record['backbone_state_sha256'],guard.tensor_hash(bb.state_dict()))
        for state in ({},{'state_dict':good},{'module.'+k:v for k,v in good.items()},{**good,'weight':torch.zeros(3,2)},{**good,'weight':torch.full((2,2),float('nan'))}):
            with self.assertRaises(ValueError):load(state)
        with self.assertRaises(ValueError):load(good,kind='public_knee')
    def test_resume_identity_complete_state_and_rng(self):
        import torch,numpy as np
        identity={'cache':{'content_sha256':'a'*64},'workers':2}
        state={'model':{'w':torch.ones(1)},'opt':{},'sched':{},'scaler':{},'best':.5,'best_state':{},'best_P':np.zeros((1,12)),'hist':[{'ep':0}], 'topk':[{'epoch':0,'model':{},'sel':.5}], 'epoch':0}
        path=self.root/'resume.pt'
        with patch.dict(os.environ,{},clear=True):guard.save_resume(path,state,identity)
        saved=torch.load(path,weights_only=False);guard.validate_resume(saved,identity)
        expected=(random.random(),float(np.random.rand()),float(torch.rand(1)))
        guard.restore_rng(saved['rng']);actual=(random.random(),float(np.random.rand()),float(torch.rand(1)))
        self.assertEqual(actual,expected);self.assertFalse(path.with_name(path.name+'.tmp').exists())
        for changed in ({'cache':{'content_sha256':'b'*64},'workers':2},{**identity,'workers':10}):
            with self.assertRaises(ValueError):guard.validate_resume(saved,changed)
        for field in ('opt','sched','scaler','rng','best_state','topk'):
            bad=copy.copy(saved);del bad[field]
            with self.assertRaises(ValueError):guard.validate_resume(bad,identity)
        bad=copy.deepcopy(saved);bad['hist']=[]
        with self.assertRaises(ValueError):guard.validate_resume(bad,identity)
        before=smoke.file_sha(path)
        with patch('torch.save',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):guard.save_resume(path,state,identity)
        self.assertEqual(smoke.file_sha(path),before)
