"""Synthetic scope, exact-source and optimizer-accounting regressions."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from knee import public_training_smoke as smoke

try:
    import numpy as np
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
PARENT = Path(os.environ.get('KNEE_REFERENCE_PARENT', '~/Dev/rsna-knee-abnormality-detection'))
SOURCE = PARENT/'outputs/analysis/public-train-recipe-20261009/assets/train_knee.py'


def records():
    rows = []
    for role,count in [('training_candidate',3745),('development',300),('sealed',300),('gold',58),('quarantined',4)]:
        for i in range(count):
            rows.append({'uid':f'{role}-{i}', 'component':f'component-{role}-{i}', 'future_role':role,
                'original_split':'train' if role=='training_candidate' and i<2000 else 'weak_validation' if role=='development' else 'gold' if role=='gold' else None})
    return rows


class CohortTests(unittest.TestCase):
    def test_deterministic_original_training_only_with_reserved_groups(self):
        rows = records(); first, info = smoke.cohort_records(rows)
        second, reversed_info = smoke.cohort_records(list(reversed(rows)))
        self.assertEqual(first,second);self.assertEqual(info,reversed_info)
        self.assertEqual(len(first),40)
        self.assertTrue(all(u.startswith('training_candidate-') and int(u.rsplit('-',1)[1])<2000 for u in first))
        gold=next(r for r in rows if r['future_role']=='gold')
        for r in rows:
            if r['future_role']=='quarantined':r['component']=gold['component']
        self.assertEqual(smoke.cohort_records(rows)[0],first)

    def test_component_overlap_original_role_change_and_duplicate_rejected(self):
        for mutation in ('group','original','duplicate'):
            rows=records()
            if mutation=='group':rows[3745]['component']=rows[0]['component']
            elif mutation=='original':rows[3745]['original_split']='train'
            else:rows[-1]['uid']=rows[0]['uid']
            with self.assertRaises(ValueError):smoke.cohort_records(rows)

    def test_manifest_hash_checked_before_parse(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'private.json';p.write_text('private invalid data')
            with self.assertRaisesRegex(ValueError,'Pinned input bytes'):smoke.membership(p)

    def test_result_gate_rejects_incomplete_or_invented_pass(self):
        contract={'version':smoke.VERSION,'settings':smoke.SETTINGS,'source_sha256':smoke.SOURCE_SHA,
            'manifest_sha256':smoke.MANIFEST_SHA,'label_sha256':smoke.LABEL_SHA,'corpus_pins':smoke.PINS,
            'packages':smoke.PACKAGES,'internet':True,'identity_kernel':smoke.IDENTITY_KERNEL,
            'gold_evaluation':False,'validation_selection':False,'cohort':{},'selected_targets_sha256':'0'*64}
        for report in ({'status':'PASS'}, {'status':'FAIL'}, {'status':'running'}):
            with self.assertRaises(ValueError):smoke.require_result(report,contract)

    def test_exact_settings_contract_rejects_gold_or_recipe_switch(self):
        c={'version':smoke.VERSION,'settings':copy.deepcopy(smoke.SETTINGS),'source_sha256':smoke.SOURCE_SHA,
           'manifest_sha256':smoke.MANIFEST_SHA,'label_sha256':smoke.LABEL_SHA,'corpus_pins':smoke.PINS,
           'packages':smoke.PACKAGES,'internet':True,'identity_kernel':smoke.IDENTITY_KERNEL,'gold_evaluation':False,'validation_selection':False}
        smoke.require_contract(c)
        c['settings']['selection']='gold'
        with self.assertRaises(ValueError):smoke.require_contract(c)
        c['settings']=dict(smoke.SETTINGS,batch_size=2)
        with self.assertRaises(ValueError):smoke.require_contract(c)


@unittest.skipIf(torch is None,'ML environment unavailable')
class UpdateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model=torch.nn.Module();self.model.backbone=torch.nn.Linear(3,4);self.model.head=torch.nn.Linear(4,2)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=.01)
        self.scheduler=torch.optim.lr_scheduler.OneCycleLR(self.optimizer,max_lr=.01,total_steps=10,pct_start=.15)
        self.scaler=torch.amp.GradScaler('cpu',init_scale=128.)
        self.initial=copy.deepcopy(self.model.state_dict())

    def loss(self):
        return self.model.head(self.model.backbone(torch.ones(8,3))).square().mean()

    def assert_state(self,a,b):
        if isinstance(a,torch.Tensor):self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assert_state(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_state(x,y)
        else:self.assertEqual(a,b)

    def test_real_update_moves_encoder_and_nonbias_head_with_finite_gradients(self):
        result=smoke.update(self.model,self.optimizer,self.scaler,self.scheduler,self.loss())
        self.assertTrue(result['accepted']);self.assertEqual(self.scheduler.last_epoch,1)
        for group in ('encoder','nonbias_head'):
            self.assertTrue(result['gradient'][group]['finite'])
            self.assertGreater(result['gradient'][group]['norm'],0)
            self.assertGreater(smoke.movement(self.model,self.initial)[group],0)
        self.assertNotEqual(smoke.tensor_digest(self.initial),smoke.tensor_digest(self.model.state_dict()))

    def test_amp_overflow_preserves_weights_optimizer_scheduler(self):
        smoke.update(self.model,self.optimizer,self.scaler,self.scheduler,self.loss())
        before=copy.deepcopy((self.model.state_dict(),self.optimizer.state_dict(),self.scheduler.state_dict()))
        hook=self.model.backbone.weight.register_hook(lambda g:torch.full_like(g,float('inf')))
        try:result=smoke.update(self.model,self.optimizer,self.scaler,self.scheduler,self.loss())
        finally:hook.remove()
        self.assertFalse(result['accepted']);self.assertLess(result['scale_after'],result['scale_before'])
        self.assert_state(before,(self.model.state_dict(),self.optimizer.state_dict(),self.scheduler.state_dict()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        json.dumps(result,allow_nan=False)

    def test_nonfinite_loss_fails_before_any_mutation(self):
        before=copy.deepcopy((self.model.state_dict(),self.optimizer.state_dict(),self.scheduler.state_dict(),self.scaler.state_dict()))
        with self.assertRaises(FloatingPointError):
            smoke.update(self.model,self.optimizer,self.scaler,self.scheduler,self.loss()*float('nan'))
        self.assert_state(before,(self.model.state_dict(),self.optimizer.state_dict(),self.scheduler.state_dict(),self.scaler.state_dict()))

    def test_train_metrics_prior_and_undefined_targets_are_explicit(self):
        y=np.full((40,12),.2,dtype=np.float32);w=np.full(12,4.);p=np.full((40,12),.5)
        m=smoke.metrics(p,y,w)
        self.assertEqual(m['defined_auc_targets'],0);self.assertIsNone(m['macro_auc_defined_targets'])
        self.assertAlmostEqual(m['macro_bce'],np.log(2),places=6)
        for bad in (float('nan'),-1,2):
            badp=p.copy();badp[0,0]=bad
            with self.assertRaises(FloatingPointError):smoke.metrics(badp,y,w)

    def test_cuda_failure_is_sanitized_before_mount_or_images(self):
        contract={'version':smoke.VERSION,'settings':smoke.SETTINGS,'source_sha256':smoke.SOURCE_SHA,
            'manifest_sha256':smoke.MANIFEST_SHA,'label_sha256':smoke.LABEL_SHA,'corpus_pins':smoke.PINS,
            'packages':smoke.PACKAGES,'internet':True,'identity_kernel':smoke.IDENTITY_KERNEL,'gold_evaluation':False,'validation_selection':False}
        import time
        with tempfile.TemporaryDirectory() as d,patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'sanitized aggregate receipt'):
                smoke.run(Path(d)/'no-inputs',Path(d)/'result',contract,started=time.monotonic())
            report=json.loads((Path(d)/'result/smoke-report.json').read_text())
            self.assertEqual(report['status'],'FAIL');self.assertEqual(report['phase'],'setup')
            self.assertEqual(report['accepted_updates'],0)
            self.assertNotIn('error_message',report)

    @unittest.skipUnless(SOURCE.is_file(),'Pinned public source fixture unavailable')
    def test_exact_public_definitions_no_main_and_sampler_limitation_preserved(self):
        fake_timm=types.SimpleNamespace(create_model=None)
        with patch.dict(sys.modules,{'timm':fake_timm}):
            ns=smoke.public_definitions(SOURCE.read_text())
        self.assertNotIn('main',ns);self.assertNotIn('load_raptor',ns);self.assertNotIn('_find',ns)
        self.assertEqual({n for n in smoke.DEFINITIONS if n in ns},set(smoke.DEFINITIONS))
        dataset=ns['StudyWindows']('', ['synthetic'], {'synthetic':0}, {'synthetic':np.zeros(12,np.float32)}, 8,16,False,norm='imagenet')
        volume=np.ones((1,44,12,12),np.uint8);volume[:,12:22]=0
        mask=np.ones((1,44),np.uint8);mask[:,12:22]=0
        dataset.vols=volume;dataset.masks=mask
        x,y=dataset[0];self.assertEqual(tuple(x.shape),(16,3,8,8));self.assertTrue(bool(torch.isfinite(x).all()))
        centers=dataset._centers(np.flatnonzero(mask[0]),16)
        self.assertTrue(any(not mask[0,c-1:c+2].all() for c in centers))
        bb=torch.nn.Sequential(torch.nn.Conv2d(3,4,1),torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
        model=ns['RaptorClassifier'](bb,F_dim=4).eval()
        with torch.no_grad():self.assertEqual(tuple(model(x[None]).shape),(1,12))
        with self.assertRaisesRegex(ValueError,'source changed'):
            smoke.public_definitions(SOURCE.read_text()+'\n')


class BuilderTests(unittest.TestCase):
    def test_existing_destination_and_noncanonical_slug_fail_early(self):
        spec=importlib.util.spec_from_file_location('public_smoke_builder',ROOT/'scripts/build_public_training_smoke.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'immutable build'):
                builder.build(d,'valid',public_source='',manifest='',label_table='',corpus_metadata='')
            with self.assertRaisesRegex(ValueError,'slug'):
                builder.build(Path(d)/'new','bad/slug',public_source='',manifest='',label_table='',corpus_metadata='')


if __name__=='__main__':unittest.main()
