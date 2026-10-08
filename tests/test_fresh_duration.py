"""Duration-only contract, checkpoint/evidence boundaries and no-update gate."""
from contextlib import nullcontext
import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from knee import fresh_reader as fresh
from knee.paired_validation import evidence_experiment


def fixture():
    plan={'config':copy.deepcopy(fresh.CONFIG),'targets':['a','b'],'records':[
        {'uid':'training','split':'train','labels':[.1,.9]},
        {'uid':'development','split':'weak_validation','labels':[.8,.2]}]}
    plan['fingerprint']=fresh.fingerprint(plan)
    return plan


def builder_module():
    spec=importlib.util.spec_from_file_location('duration_builder',Path(__file__).parents[1]/'scripts/build_fresh_reader.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class DurationContractTests(unittest.TestCase):
    def test_only_epoch_count_changes_and_plan_and_source_are_bound(self):
        config=copy.deepcopy(fresh.CONFIG);before=copy.deepcopy(config)
        contract=fresh.duration_experiment(fresh.DURATION_PLAN,config)
        self.assertEqual(config,before)
        self.assertEqual({k:(v,contract['config'][k]) for k,v in config.items() if v!=contract['config'][k]},
                         {'max_epochs':(4,12)})
        self.assertEqual(contract['config']['training_seconds'],5400)
        self.assertEqual(contract['initial_encoder_sha256'],fresh.INITIAL_ENCODER_SHA256)
        self.assertEqual(contract['runner_sha256'],fresh.file_sha256(fresh.__file__))
        for changed in (config|{'seed':43},config|{'training_seconds':7200},config|{'max_epochs':12}):
            with self.assertRaises(ValueError):fresh.duration_experiment(fresh.DURATION_PLAN,changed)
        with self.assertRaises(ValueError):fresh.duration_experiment('different',config)
        plan={'fingerprint':fresh.DURATION_PLAN,'config':config}
        for key in ('profile','selection','runner_sha256','fingerprint'):
            bad=copy.deepcopy(contract);bad[key]='changed'
            with self.assertRaises(ValueError):fresh.validate_experiment(plan,bad)
        original_hash=fresh.file_sha256
        with patch.object(fresh,'file_sha256',side_effect=lambda p: '0'*64 if str(p).endswith('fresh_reader_setup.py') else original_hash(p)):
            with self.assertRaises(ValueError):fresh.validate_experiment(plan,contract)
        self.assertIn('scripts/build_fresh_reader.py',contract['own_sources'])

    def test_all_twelve_epochs_must_fit_including_startup_and_reserve(self):
        gate=fresh.duration_gate_budget(1.,.1,.1,300.)
        self.assertEqual(gate['planned_epochs'],12)
        self.assertEqual(gate['remaining_after_checkpoint_reserve_seconds'],4920.)
        self.assertTrue(gate['passed'])
        self.assertFalse(fresh.duration_gate_budget(1.,.1,.1,1000.)['passed'])
        self.assertFalse(fresh.duration_gate_budget(2.,.1,.1,0.)['passed'])
        for value in (0.,float('nan'),float('inf'),-1.):
            with self.assertRaises(ValueError):fresh.duration_gate_budget(value,.1,.1,0.)

    def test_resume_rejects_pilot_or_changed_schedule_or_completed_run(self):
        experiment=fresh.duration_experiment(fresh.DURATION_PLAN,fresh.CONFIG)
        state=dict(experiment=experiment,experiment_fingerprint=experiment['fingerprint'],config=experiment['config'],
                   planned_epochs=12,initial_encoder_sha256=fresh.INITIAL_ENCODER_SHA256,epoch=3,
                   best_loss=.4,best_evidence_fingerprint='a'*64,
                   gate={'passed':True,'planned_epochs':12,'experiment_fingerprint':experiment['fingerprint']},
                   scheduler={'T_max':3072})
        fresh.validate_duration_resume(state,experiment)
        for key,value in [('experiment',None),('experiment_fingerprint','changed'),('planned_epochs',4),
                          ('config',fresh.CONFIG),('initial_encoder_sha256','changed'),('epoch',12),
                          ('scheduler',{'T_max':1024}),('best_loss',float('inf')),('best_evidence_fingerprint',None)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                fresh.validate_duration_resume(state|{key:value},experiment)

    def test_v2_evidence_and_analysis_binding_with_explicit_legacy_path(self):
        plan=fixture();p=[[.7,.3]];z=[[math.log(v/(1-v)) for v in p[0]]]
        with patch.object(fresh,'DURATION_PLAN',plan['fingerprint']),tempfile.TemporaryDirectory() as directory:
            root=Path(directory);experiment=fresh.duration_experiment(plan['fingerprint'],plan['config'])
            legacy=fresh.weak_validation_evidence(plan,[1],p,logits=z,epoch=2,weak_bce=.4)
            self.assertEqual(legacy['version'],1);self.assertNotIn('experiment',legacy)
            self.assertIsNone(evidence_experiment(plan,{'evidence':legacy},root))
            evidence=fresh.weak_validation_evidence(plan,[1],p,logits=z,epoch=10,weak_bce=.4,experiment=experiment)
            self.assertEqual(evidence['version'],2)
            payload=dict(plan_fingerprint=plan['fingerprint'],targets=plan['targets'],epoch=10,weak_bce=.4,
                weak_validation_evidence=evidence,experiment=experiment,experiment_fingerprint=experiment['fingerprint'],
                runner_sha256=experiment['runner_sha256'],config=experiment['config'],
                initial_encoder_sha256=fresh.INITIAL_ENCODER_SHA256)
            (root/'best.pt').write_bytes(b'synthetic checkpoint')
            selected=fresh.write_best_evidence(root,plan,payload)
            private=json.loads((root/'best-weak-validation-private.json').read_text())
            run={k:payload[k] for k in ('experiment','experiment_fingerprint','runner_sha256','config',
                                       'initial_encoder_sha256','plan_fingerprint')}
            report=run|{'best_weak_validation_evidence':selected}
            (root/'run-contract.json').write_text(json.dumps(run));(root/'training-report.json').write_text(json.dumps(report))
            self.assertEqual(evidence_experiment(plan,private,root),experiment)
            with self.assertRaisesRegex(ValueError,'cannot accompany'):
                evidence_experiment(plan,{'evidence':legacy},root)
            with self.assertRaisesRegex(ValueError,'horizon'):
                evidence_experiment(plan,{'evidence':legacy|{'selected_epoch':5}},root/'absent')
            # Analysis remains valid after later runner code changes; artifacts stay bound to original bytes.
            with patch.object(fresh,'file_sha256',return_value='0'*64):
                self.assertEqual(evidence_experiment(plan,private,root),experiment)
            for bad in (payload|{'experiment':None},payload|{'config':fresh.CONFIG},payload|{'runner_sha256':'changed'}):
                with self.assertRaises(ValueError):fresh.write_best_evidence(root,plan,bad)
            (root/'training-report.json').write_text(json.dumps(report|{'experiment_fingerprint':'changed'}))
            with self.assertRaises(ValueError):evidence_experiment(plan,private,root)
            with self.assertRaises(ValueError):evidence_experiment(plan,{'evidence':legacy|{'experiment':experiment}},root)

    def test_builder_first_run_is_cache_only_and_rejects_old_resume_or_wrong_profile(self):
        builder=builder_module()
        with tempfile.TemporaryDirectory() as directory,patch.object(builder,'check_sources'):
            root=Path(directory);sources=root/'source';sources.mkdir()
            for name in fresh.SOURCE_HASHES:(sources/name).write_text('pass\n')
            kw=dict(cache_kernel='ozgung/knee-fresh-cache-20261008',plan_fingerprint=fresh.DURATION_PLAN,
                    training_profile=fresh.DURATION_PROFILE)
            contract=builder.build(sources,root/'build','duration-test','train',**kw)
            metadata=json.loads((root/'build/kernel-metadata.json').read_text())
            self.assertEqual(metadata['kernel_sources'],[kw['cache_kernel']])
            self.assertEqual(metadata['competition_sources'],[]);self.assertEqual(metadata['dataset_sources'],[])
            self.assertTrue(metadata['is_private'] and metadata['enable_gpu'])
            self.assertEqual(contract['effective_stage_seconds'],5400)
            self.assertEqual(contract['experiment']['config']['max_epochs'],12)
            code=''.join(json.loads((root/'build/notebook.ipynb').read_text())['cells'][1]['source'])
            compile(code,'duration-notebook','exec');self.assertIn('started_at=_stage_started',code)
            self.assertIn("'cumulative duration training'",code)
            with self.assertRaises(ValueError):builder.build(sources,root/'bad','bad','train',**kw,resume_kernel='ozgung/knee-fresh-train-20261008')
            prior=root/'prior.json';prior.write_text(json.dumps({'plan_fingerprint':fresh.DURATION_PLAN}))
            with self.assertRaises(ValueError):builder.build(sources,root/'bad','bad','train',**kw,resume_kernel='ozgung/old',resume_contract=prior)
            valid={'experiment':contract['experiment'],'experiment_fingerprint':contract['experiment_fingerprint'],
                   'runner_sha256':contract['experiment']['runner_sha256'],
                   'plan_fingerprint':fresh.DURATION_PLAN,'config':contract['experiment']['config']}
            prior.write_text(json.dumps(valid))
            (root/'training-report.json').write_text(json.dumps(valid|{'elapsed_seconds':5000.}))
            resumed=builder.build(sources,root/'resume','resume','train',**kw,resume_kernel='ozgung/new-duration',resume_contract=prior)
            self.assertEqual(resumed['resume_contract_sha256'],fresh.file_sha256(prior))
            self.assertEqual(resumed['effective_stage_seconds'],400.)
            self.assertEqual(resumed['resume_report_sha256'],fresh.file_sha256(root/'training-report.json'))
            resume_code=''.join(json.loads((root/'resume/notebook.ipynb').read_text())['cells'][1]['source'])
            self.assertIn('STAGE_BUDGET = 400.0',resume_code)
            (root/'training-report.json').write_text(json.dumps(valid|{'elapsed_seconds':5300.}))
            with self.assertRaises(TimeoutError):builder.build(sources,root/'exhausted','exhausted','train',**kw,
                resume_kernel='ozgung/new-duration',resume_contract=prior)
            with self.assertRaises(ValueError):builder.build(sources,root/'bad','bad','probe',**kw)


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch unavailable')
class DurationGateTests(unittest.TestCase):
    def test_actual_gate_fixed128_rows_zero_updates_rng_restoration_and_gradient_cleanup(self):
        import numpy as np
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.linear=torch.nn.Linear(2,2);self.dropout=torch.nn.Dropout(.2)
            def forward(self,x,mask,pos,**kwargs):
                random.random();np.random.rand()
                return self.linear(self.dropout(x))
        model=Model();ema=copy.deepcopy(model);optimizer=torch.optim.AdamW(model.parameters())
        state=copy.deepcopy(model.state_dict());before=fresh.rng_state();seen=[];validated=[]
        def batch(indices):
            seen.extend(indices);return torch.ones(2,2),None,None,torch.tensor([[.9,.1],[.1,.9]])
        def validate(current,indices):
            validated.append(indices);random.random();np.random.rand();torch.rand(1)
        scaler=type('ScaleOnly',(),{'scale':lambda self,x:x})()
        with patch('torch.cuda.synchronize'),patch('torch.autocast',side_effect=lambda *a,**k:nullcontext()), \
             patch.object(optimizer,'step',side_effect=AssertionError('Gate must not update weights')):
            gate=fresh.duration_timing_gate(model,ema,optimizer,scaler,batch,validate,list(range(512)),
                                            list(range(512,712)),list(range(64)),fresh.CONFIG,fresh.time.monotonic()-500)
        self.assertEqual(seen,list(range(128)));self.assertEqual(validated,[list(range(512,528)),list(range(16))])
        self.assertEqual((gate['warmup_batches'],gate['timed_batches'],gate['optimizer_updates']),(16,48,0))
        self.assertEqual((gate['warmup_backward_batches'],gate['timed_backward_batches']),(16,48))
        self.assertGreaterEqual(gate['elapsed_before_fit_seconds'],500)
        after=fresh.rng_state();self.assertEqual(before['python'],after['python']);np.testing.assert_equal(before['numpy'],after['numpy'])
        self.assertTrue(torch.equal(before['torch'],after['torch']))
        self.assertTrue(all(torch.equal(state[k],v) for k,v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()));self.assertEqual(optimizer.state,{})
        before=fresh.rng_state()
        def fail(*args):raise RuntimeError('synthetic validation failure')
        with patch('torch.cuda.synchronize'),patch('torch.autocast',side_effect=lambda *a,**k:nullcontext()):
            with self.assertRaises(RuntimeError):fresh.duration_timing_gate(model,ema,optimizer,scaler,batch,fail,
                list(range(512)),list(range(512,712)),list(range(64)),fresh.CONFIG,fresh.time.monotonic())
        after=fresh.rng_state();self.assertEqual(before['python'],after['python']);np.testing.assert_equal(before['numpy'],after['numpy'])
        self.assertTrue(torch.equal(before['torch'],after['torch']));self.assertTrue(all(p.grad is None for p in model.parameters()))
        def missing_supervision(indices):
            x,m,p,y=batch(indices)
            if indices[0] == 32:y.fill_(.5)  # First timed batch cannot be omitted from backward timing.
            return x,m,p,y
        with patch('torch.cuda.synchronize'),patch('torch.autocast',side_effect=lambda *a,**k:nullcontext()):
            with self.assertRaisesRegex(ValueError,'48timed supervised'):
                fresh.duration_timing_gate(model,ema,optimizer,scaler,missing_supervision,validate,list(range(512)),
                    list(range(512,712)),list(range(64)),fresh.CONFIG,fresh.time.monotonic())
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_near_exhausted_resume_fails_before_tensor_loading_cache_hashes_or_model(self):
        plan=fixture()
        with patch.object(fresh,'DURATION_PLAN',plan['fingerprint']),tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cache=root/'cache';previous=root/'previous';cache.mkdir();previous.mkdir()
            (cache/'plan-private.json').write_text(json.dumps(plan))
            (cache/'cache-complete.json').write_text(json.dumps({'status':'PASS','plan_fingerprint':plan['fingerprint']}))
            experiment=fresh.duration_experiment(plan['fingerprint'],plan['config'])
            report=dict(experiment=experiment,experiment_fingerprint=experiment['fingerprint'],
                runner_sha256=experiment['runner_sha256'],plan_fingerprint=plan['fingerprint'],elapsed_seconds=5300.)
            (previous/'training-report.json').write_text(json.dumps(report))
            with patch('torch.cuda.is_available',return_value=True),patch('torch.load') as tensor_load,patch.object(fresh,'load_source') as model_load:
                with self.assertRaisesRegex(TimeoutError,'Cumulative training budget'):
                    fresh.train(cache,root/'sources',root/'new',resume=previous/'last.pt',experiment=experiment)
            tensor_load.assert_not_called();model_load.assert_not_called()
            for elapsed in (-1.,float('nan'),float('inf'),True):
                with self.assertRaises(ValueError):fresh.remaining_training_budget(elapsed,fresh.time.monotonic())
            self.assertAlmostEqual(fresh.remaining_training_budget(5000.,fresh.time.monotonic()),220.,delta=.1)

    def test_best_checkpoint_pair_cannot_cross_experiments_on_resume(self):
        import torch
        plan=fixture()
        with patch.object(fresh,'DURATION_PLAN',plan['fingerprint']),tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prior=root/'prior';out=root/'out';prior.mkdir();out.mkdir()
            exp=fresh.duration_experiment(plan['fingerprint'],plan['config']);p=[[.5,.5]]
            e=fresh.weak_validation_evidence(plan,[1],p,logits=[[0.,0.]],epoch=1,weak_bce=.7,experiment=exp)
            payload=dict(experiment=exp,experiment_fingerprint=exp['fingerprint'],config=exp['config'],
                runner_sha256=exp['runner_sha256'],initial_encoder_sha256=fresh.INITIAL_ENCODER_SHA256,
                weak_bce=.7,epoch=1,targets=plan['targets'],plan_fingerprint=plan['fingerprint'],weak_validation_evidence=e)
            fresh.save_checkpoint(prior/'best.pt',payload)
            fresh.restore_best_evidence(prior,out,plan,best_loss=.7,expected_fingerprint=e['fingerprint'],experiment=exp)
            for wrong in (None,exp|{'fingerprint':'changed'}):
                with self.assertRaises(ValueError):fresh.restore_best_evidence(prior,out,plan,best_loss=.7,expected_fingerprint=e['fingerprint'],experiment=wrong)


if __name__=='__main__':unittest.main()
