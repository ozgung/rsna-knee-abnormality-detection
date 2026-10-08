"""Disposable diagnostic provenance and real optimizer skip boundaries."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from knee.coverage_diagnostic import gradient_summary,movement,parameter_group,update

try:
    import torch
except ImportError:
    torch=None

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('diagnostic_builder',ROOT/'scripts/build_coverage_diagnostic.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
PARENT=Path(os.environ.get('KNEE_REFERENCE_PARENT','/nonexistent'))
ARCHIVE=PARENT/'outputs/kaggle/train04-reference-20261009/notebook.ipynb'


@unittest.skipIf(torch is None,'ML environment unavailable')
class UpdateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=torch.nn.Module();self.model.enc=torch.nn.Linear(3,4);self.model.head=torch.nn.Linear(4,2)
        self.ema=copy.deepcopy(self.model).requires_grad_(False)
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=.01)
        self.scaler=torch.amp.GradScaler('cpu',init_scale=128.)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,12000)
        self.initial=copy.deepcopy(self.model.state_dict())

    def loss(self):
        return self.model.head(self.model.enc(torch.ones(2,3))).square().mean()

    def call(self,loss,accepted=0):
        return update(self.model,self.ema,self.opt,self.scaler,self.scheduler,loss,accepted)

    def assert_state_equal(self,a,b):
        if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assert_state_equal(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_state_equal(x,y)
        else:self.assertEqual(a,b)

    def snapshot(self):
        return copy.deepcopy((self.model.state_dict(),self.ema.state_dict(),self.opt.state_dict(),self.scheduler.state_dict()))

    def test_accepted_step_has_encoder_and_nonbias_head_gradients_and_movement(self):
        result=self.call(self.loss());self.assertEqual(result['kind'],'accepted');self.assertEqual(result['accepted'],1)
        for group in ('encoder','nonbias_head'):
            self.assertTrue(result['gradient'][group]['finite']);self.assertGreater(result['gradient'][group]['norm'],0)
            self.assertGreater(movement(self.model,self.initial)[group]['max_abs'],0)
            self.assertGreater(movement(self.ema,self.initial)[group]['max_abs'],0)
        self.assertEqual(self.scheduler.last_epoch,1)

    def test_post_forward_empty_preserves_optimizer_scheduler_ema_and_scaler(self):
        self.call(self.loss());self.opt.zero_grad(set_to_none=True)
        unused=self.loss();before=self.snapshot();scale=copy.deepcopy(self.scaler.state_dict())
        result=self.call(None,1)
        self.assertEqual(result,{'kind':'empty','accepted':1});self.assert_state_equal(before,self.snapshot())
        self.assert_state_equal(scale,self.scaler.state_dict())

    def test_real_amp_overflow_preserves_accepted_state_but_reduces_scale(self):
        self.call(self.loss());self.opt.zero_grad(set_to_none=True);before=self.snapshot();scale=self.scaler.get_scale()
        handle=self.model.enc.weight.register_hook(lambda grad:torch.full_like(grad,float('inf')))
        result=self.call(self.loss(),1);handle.remove()
        self.assertEqual(result['kind'],'overflow');self.assertEqual(result['accepted'],1)
        self.assert_state_equal(before,self.snapshot());self.assertLess(self.scaler.get_scale(),scale)
        self.assertFalse(result['gradient']['encoder']['finite']);self.assertIsNone(result['preclip_norm'])
        json.dumps(result,allow_nan=False)

    def test_nonfinite_loss_fails_before_mutation(self):
        before=self.snapshot()
        with self.assertRaises(ValueError):self.call(self.loss()*float('nan'))
        self.assert_state_equal(before,self.snapshot())

    def test_nonbias_head_excludes_bias_and_includes_layernorm_weight(self):
        self.assertEqual(parameter_group('cls_b'),'head_bias')
        self.assertEqual(parameter_group('proj.2.bias'),'head_bias')
        self.assertEqual(parameter_group('proj.2.weight'),'nonbias_head')


@unittest.skipUnless(ARCHIVE.exists(),'Original archived notebook fixture unavailable')
class PackagingTests(unittest.TestCase):
    def test_exact_original_archive_and_one_byte_drift(self):
        bundle=builder.archived_sources(ARCHIVE)
        self.assertEqual(bundle['CONTRACT']['revision'],builder.REFERENCE_REVISION)
        nb=json.loads(ARCHIVE.read_text());cell=next(c for c in nb['cells'] if c['cell_type']=='code')
        cell['source']=[ ''.join(cell['source']).replace('TRAIN04 fixed twelve-epoch image training','TRAIN04 changed twelve-epoch image training',1)]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.ipynb';p.write_text(json.dumps(nb))
            with self.assertRaises(ValueError):builder.archived_sources(p)

    @unittest.skipUnless((PARENT/'outputs/kaggle/train04-reference-20261009-result/coverage_training/initial.pt').exists(),
                         'Authentic original initialization fixture unavailable')
    def test_build_failed_reference_without_score_gate_and_no_external_data(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'bundle'
            contract=builder.build(ARCHIVE,PARENT/'outputs/kaggle/train04-cache-full-v2-result/coverage_cache',
                PARENT/'outputs/kaggle/train04-reference-20261009-result/coverage_training',out,'synthetic-diagnostic',
                cache_kernel='ozgung/knee-train04-cache-full-20261008',reference_kernel='ozgung/knee-train04-reference-20261009')
            self.assertEqual(contract['reference_prefix']['reference_status'],'FAIL')
            meta=json.loads((out/'kernel-metadata.json').read_text())
            self.assertFalse(meta['enable_internet']);self.assertEqual(meta['dataset_sources'],[])
            self.assertEqual(meta['competition_sources'],[]);self.assertEqual(len(meta['kernel_sources']),2)
            self.assertEqual(contract['sources']['knee/coverage_training.py'],builder.REFERENCE_RUNNER)
            self.assertFalse(contract['fitted_weights_saved'])
            nb=json.loads((out/'notebook.ipynb').read_text())
            compile(''.join(nb['cells'][1]['source']),'built-diagnostic','exec')


if __name__=='__main__':unittest.main()
