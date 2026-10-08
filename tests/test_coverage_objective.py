"""Synthetic loss gradients, real loop skip boundary, and source-bound packaging."""
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch

from knee import coverage_objective as obj, coverage_training as ct
from knee.fresh_reader import fingerprint,rng_state

try:
    import numpy as np
    import torch
except ImportError:
    np=torch=None

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('objective_builder',ROOT/'scripts/build_coverage_training.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def prefix_fixture():
    env={'python':'3.12','versions':{'numpy':'2.0.2'},'device':'cuda','cuda_devices':[{'name':'T4'}]}
    contract={'version':'train04-v1','variant':'v1','label_sources':{},'runner_sha256':obj.REFERENCE_RUNNER,
              'runtime_sources':{n:obj.REFERENCE_SOURCES[n] for n in ('knee/coverage_cache.py','knee/fresh_reader.py',
                                'knee/fresh_reader_setup.py','knee/validation.py')}|
                                {'knee/coverage_training.py':obj.REFERENCE_RUNNER},'runtime_environment':env}
    contract['fingerprint']=fingerprint(contract)
    row={'epoch':1,'weak_bce':.8,'per_target_bce':[.8]*12,'confidence_support':[10.]*12,
         'thresholded_weak_auc':{'macro_all_targets_auc':.5,'undefined_targets':[]}}
    report={'status':'FAIL','completed_epochs':1,'contract':contract,'history':[row],
            'selected_bce':.8,'prevalence_weak_bce':.6,'successful_updates':1000}
    run={'contract':contract,'initial_model_sha256':'a'*64,
         'environment':env|{'code_revision':obj.REFERENCE_REVISION,'source_sha256':obj.REFERENCE_BUILD}}
    return report,run


class ProvenanceTests(unittest.TestCase):
    def test_low_scoring_incomplete_reference_is_allowed_without_mutation(self):
        report,run=prefix_fixture();original=copy.deepcopy(report)
        verified=obj.validate_reference_prefix(report,run)
        self.assertEqual(verified['completed_epochs'],1)
        self.assertEqual(verified['reference_status'],'FAIL')
        self.assertEqual(report,original)

    def test_prefix_rejects_source_environment_missing_or_nonfinite_evidence(self):
        for kind in ('source','fingerprint','environment','history','metric','initial','selection','updates'):
            report,run=prefix_fixture()
            if kind=='source':report['contract']['runner_sha256']='b'*64
            elif kind=='fingerprint':report['contract']['fingerprint']='b'*64
            elif kind=='environment':run['environment']['versions']={'numpy':'different'}
            elif kind=='history':report['history'][0]['epoch']=2
            elif kind=='metric':report['history'][0]['weak_bce']=float('nan')
            elif kind=='initial':run['initial_model_sha256']=None
            elif kind=='selection':report['selected_bce']=.5
            else:report['successful_updates']=0
            with self.subTest(kind=kind),self.assertRaises(ValueError):obj.validate_reference_prefix(report,run)

    def test_exact_initial_bytes_preserved_and_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            src=Path(d)/'initial.pt';dst=Path(d)/'copy.pt';src.write_bytes(b'synthetic initial model and RNG')
            sha=hashlib.sha256(src.read_bytes()).hexdigest()
            obj.copy_initial(src,dst,sha);self.assertEqual(src.read_bytes(),dst.read_bytes())
            with self.assertRaises(ValueError):obj.copy_initial(src,Path(d)/'bad.pt','0'*64)

    def test_invariants_allow_only_named_objective_sources(self):
        equal={k:'same' for k in ('config','plan_fingerprint','source_hashes','index_sha256','complete_sha256',
                                 'labels_fingerprint','weights_fingerprint','runtime_environment')}
        reference=equal|{'variant':'v1','runner_sha256':obj.REFERENCE_RUNNER}
        candidate=equal|{'variant':'v1','objective':obj.GLOBAL_OBJECTIVE,'comparison_sources':ct.comparison_sources()}
        obj.require_objective_invariants(candidate,reference)
        for key in equal:
            with self.subTest(key=key),self.assertRaises(ValueError):
                obj.require_objective_invariants(candidate|{key:'drift'},reference)
        changed=copy.deepcopy(candidate);changed['comparison_sources']['knee/coverage_cache.py']='0'*64
        with self.assertRaises(ValueError):obj.require_objective_invariants(changed,reference)
        with self.assertRaises(ValueError):obj.require_objective_invariants(candidate|{'variant':'consensus'},reference)

    def test_objective_contract_cannot_resume_original_reference_state(self):
        old={'objective':obj.BATCH_OBJECTIVE,'runner_sha256':obj.REFERENCE_RUNNER}
        state={'contract':old,'epoch':11,'elapsed_seconds':5000}
        with self.assertRaises(ValueError):
            ct.validate_resume_state(state,old|{'objective':obj.GLOBAL_OBJECTIVE},5001)


@unittest.skipIf(torch is None,'ML environment unavailable')
class ObjectiveTests(unittest.TestCase):
    def test_exact_initial_restores_full_head_encoder_and_rng_then_copies_bytes(self):
        torch.manual_seed(3);np.random.seed(3);random.seed(3)
        original=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Linear(3,2))
        state=rng_state();params=copy.deepcopy(original.state_dict())
        expected=(torch.rand(4),np.random.random(4),[random.random() for _ in range(4)])
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'initial.pt';copy_path=Path(d)/'candidate-initial.pt'
            torch.save({'model':params,'rng':state,'config':{'seed':3},'plan_fingerprint':'fixed-plan'},source)
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            replacement=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.Linear(3,2))
            obj.load_exact_initial(replacement,source,expected_sha256=sha,plan_fingerprint='fixed-plan',config={'seed':3})
            for k,v in replacement.state_dict().items():self.assertTrue(torch.equal(v,params[k]))
            self.assertTrue(torch.equal(torch.rand(4),expected[0]))
            self.assertTrue(np.array_equal(np.random.random(4),expected[1]))
            self.assertEqual([random.random() for _ in range(4)],expected[2])
            obj.copy_initial(source,copy_path,sha);self.assertEqual(source.read_bytes(),copy_path.read_bytes())
            for kw in ({'expected_sha256':'0'*64},{'plan_fingerprint':'different'},{'config':{'seed':4}}):
                args=dict(expected_sha256=sha,plan_fingerprint='fixed-plan',config={'seed':3});args.update(kw)
                with self.assertRaises(ValueError):obj.load_exact_initial(replacement,source,**args)

    def test_exhaustive_gradients_and_loss_match_whole_cohort(self):
        torch.manual_seed(7)
        y=torch.tensor([[.9,.5,.1],[.1,.8,.5],[.5,.1,.9],[.8,.9,.1]],dtype=torch.float32)
        weights=2*(y-.5).abs();means=torch.tensor(obj.training_mean_weights(weights.numpy(),list(range(4))))
        z=torch.tensor([.2,-.4,.8],requires_grad=True)
        n,d=ct.weighted_terms(z.expand_as(y),y,weights);full=ct.macro_loss(n,d)
        full_grad=torch.autograd.grad(full,z)[0]
        for size in (1,2,4):
            losses=[];gradients=[]
            for indices in itertools.combinations(range(4),size):
                idx=list(indices);n,d=ct.weighted_terms(z.expand(size,-1),y[idx],weights[idx])
                loss=obj.training_loss(n,d,objective=obj.GLOBAL_OBJECTIVE,batch_size=size,mean_weights=means)
                losses.append(loss.detach());gradients.append(torch.autograd.grad(loss,z)[0])
            self.assertTrue(torch.allclose(torch.stack(losses).mean(),full,atol=1e-7))
            self.assertTrue(torch.allclose(torch.stack(gradients).mean(0),full_grad,atol=1e-7))
        # The declared twelve-target / twelve-row case includes very sparse support.
        y=torch.zeros(12,12);weights=torch.ones_like(y);weights[:,11]=0
        y[0,11]=1;weights[0,11]=.2;weights[1,11]=.8
        means=weights.mean(0);z=torch.zeros(12,requires_grad=True)
        n,d=ct.weighted_terms(z.expand_as(y),y,weights);full=ct.macro_loss(n,d)
        full_grad=torch.autograd.grad(full,z)[0]
        for size in (1,2,12):
            gradients=[]
            for indices in itertools.combinations(range(12),size):
                idx=list(indices);n,d=ct.weighted_terms(z.expand(size,-1),y[idx],weights[idx])
                loss=obj.training_loss(n,d,objective=obj.GLOBAL_OBJECTIVE,batch_size=size,mean_weights=means)
                gradients.append(torch.autograd.grad(loss,z)[0])
            self.assertTrue(torch.allclose(torch.stack(gradients).mean(0),full_grad,atol=1e-7))

    def test_training_denominator_excludes_development_and_rejects_invalid_support(self):
        weights=np.array([[.2,1],[.8,0],[100,100.]])
        self.assertEqual(obj.training_mean_weights(weights,[0,1]),[.5,.5])
        weights[2]=np.nan
        self.assertEqual(obj.training_mean_weights(weights,[0,1]),[.5,.5])
        for indices in ([],[0,0],[-1],[9]):
            with self.assertRaises(ValueError):obj.training_mean_weights(weights,indices)
        with self.assertRaises(ValueError):obj.training_mean_weights(np.zeros((2,2)),[0,1])

    def test_unknown_gradients_confidence_and_legacy_objective(self):
        z=torch.zeros(1,2,requires_grad=True);y=torch.tensor([[1.,.5]]);w=torch.tensor([[.2,0.]])
        n,d=ct.weighted_terms(z,y,w);means=torch.ones(2)
        corrected=obj.training_loss(n,d,objective=obj.GLOBAL_OBJECTIVE,batch_size=1,mean_weights=means)
        g=torch.autograd.grad(corrected,z,retain_graph=True)[0]
        self.assertAlmostEqual(float(g[0,0]),-.05);self.assertEqual(float(g[0,1]),0.)
        legacy=obj.training_loss(n,d,objective=obj.BATCH_OBJECTIVE,batch_size=1,mean_weights=means)
        self.assertTrue(torch.equal(legacy,ct.macro_loss(n,d)))

    def test_empty_samples_have_zero_contribution_in_exhaustive_estimator(self):
        z=torch.zeros(2,requires_grad=True)
        y=torch.tensor([[.9,.1],[.5,.5],[.1,.8]]);w=2*(y-.5).abs();means=w.mean(0)
        n,d=ct.weighted_terms(z.expand_as(y),y,w);full=ct.macro_loss(n,d)
        full_grad=torch.autograd.grad(full,z)[0];gradients=[]
        for i in range(3):
            n,d=ct.weighted_terms(z[None],y[i:i+1],w[i:i+1])
            loss=obj.training_loss(n,d,objective=obj.GLOBAL_OBJECTIVE,batch_size=1,mean_weights=means)
            gradients.append(torch.zeros_like(z) if loss is None else torch.autograd.grad(loss,z)[0])
        self.assertTrue(torch.allclose(torch.stack(gradients).mean(0),full_grad,atol=1e-7))

    def test_actual_runner_empty_batch_skips_every_update_but_consumes_forward_rng(self):
        # Execute the actual inner loop AST, not a reimplementation of its skip guard.
        tree=ast.parse((ROOT/'knee/coverage_training.py').read_text())
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.While) and ast.unparse(n.test)=='cursor < len(order)')
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.enc=torch.nn.Linear(2,2);self.cls_w=torch.nn.Parameter(torch.ones(2))
            def forward(self,x,m,p,**kwargs):return torch.nn.functional.dropout(self.enc(x),.5,True)*self.cls_w
        for objective in (obj.BATCH_OBJECTIVE,obj.GLOBAL_OBJECTIVE):
            torch.manual_seed(42);np.random.seed(42);model=Model();ema=copy.deepcopy(model)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.1,weight_decay=.2)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,12000)
            scaler=torch.amp.GradScaler('cpu');telemetry=obj.StepTelemetry('cpu')
            def batch(indices,training):
                np.random.random(16)
                return torch.ones(2,2),None,None,torch.full((2,2),.5),torch.zeros(2,2)
            before={k:copy.deepcopy(v.state_dict()) for k,v in [('model',model),('ema',ema),('optimizer',optimizer),
                                                                  ('scheduler',scheduler),('scaler',scaler)]}
            rng_before=torch.get_rng_state().clone();numpy_before=np.random.get_state()[1].copy()
            namespace=dict(torch=torch,model=model,ema=ema,optimizer=optimizer,scheduler=scheduler,scaler=scaler,
                telemetry=telemetry,objective=objective,mean_weights_tensor=torch.ones(2),order=[0,1],cursor=0,
                epoch=0,step=0,overflow_total=0,consecutive=0,empty_skips=0,batch=batch,deadline=lambda:None,
                progress=lambda *a,**k:None,weighted_terms=ct.weighted_terms,training_loss=obj.training_loss)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                exec(compile(ast.Module(body=[loop],type_ignores=[]),'actual-training-loop','exec'),namespace)
            self.assertEqual(namespace['cursor'],2);self.assertEqual(namespace['empty_skips'],1)
            self.assertEqual(namespace['step'],0);self.assertEqual(namespace['overflow_total'],0)
            for name,value in [('model',model),('ema',ema)]:
                for k,t in value.state_dict().items():self.assertTrue(torch.equal(t,before[name][k]))
            for name,value in [('optimizer',optimizer),('scheduler',scheduler),('scaler',scaler)]:
                self.assertEqual(value.state_dict(),before[name])
            self.assertFalse(torch.equal(rng_before,torch.get_rng_state()))
            self.assertFalse(np.array_equal(numpy_before,np.random.get_state()[1]))
            self.assertEqual(telemetry.report()['supervised_attempts'],0)

    def test_telemetry_is_rng_neutral_finite_and_resumable(self):
        torch.manual_seed(2);before=torch.get_rng_state().clone();t=obj.StepTelemetry('cpu')
        t.add(torch.tensor(.6),torch.tensor(4.),False)
        t.add(torch.tensor(.4),torch.tensor(float('inf')),True)
        report=t.report();self.assertEqual(report['finite_preclip_norms'],1)
        self.assertEqual(report['amp_skips'],1);self.assertEqual(report['clipped_fraction'],1)
        self.assertAlmostEqual(report['objective_mean'],.5)
        self.assertEqual(obj.StepTelemetry('cpu',t.values).report(),report)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_telemetry_does_not_change_stochastic_updates_or_ema(self):
        def run(enabled):
            torch.manual_seed(27);model=torch.nn.Sequential(torch.nn.Linear(3,2),torch.nn.Dropout(.3))
            ema=copy.deepcopy(model).requires_grad_(False);optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,12000)
            scaler=torch.amp.GradScaler('cpu');telemetry=obj.StepTelemetry('cpu')
            for step in range(1,5):
                optimizer.zero_grad(set_to_none=True)
                logits=model(torch.ones(2,3));y=torch.tensor([[.9,.1],[.1,.8]])
                n,d=ct.weighted_terms(logits,y,2*(y-.5).abs());loss=ct.macro_loss(n,d)
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
                scale=scaler.get_scale();scaler.step(optimizer);scaler.update();overflow=scaler.get_scale()<scale
                if enabled:telemetry.add(loss,norm,overflow);telemetry.report()
                if not overflow:
                    scheduler.step();decay=min(.998,(1+step)/(10+step))
                    with torch.no_grad():
                        for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-decay)
            return model.state_dict(),ema.state_dict(),optimizer.state_dict(),scheduler.state_dict(),scaler.state_dict(),torch.get_rng_state()
        a,b=run(False),run(True)
        def equal(x,y):
            if isinstance(x,torch.Tensor):self.assertTrue(torch.equal(x,y))
            elif isinstance(x,dict):
                self.assertEqual(x.keys(),y.keys())
                for key in x:equal(x[key],y[key])
            elif isinstance(x,(tuple,list)):
                self.assertEqual(len(x),len(y))
                for xx,yy in zip(x,y):equal(xx,yy)
            else:self.assertEqual(x,y)
        equal(a,b)

    def test_global_bundle_accepts_low_prefix_and_attaches_no_alternate_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'src';src.mkdir();cache=root/'cache';cache.mkdir();ref=root/'ref';ref.mkdir()
            for n in builder.SOURCE_HASHES:(src/n).write_text('# synthetic public source\n')
            plan={'targets':[str(j) for j in range(12)],'records':[
                {'uid':str(i),'split':'train' if i<2000 else 'weak_validation','labels':[.9,.1]*6} for i in range(2200)]}
            plan['fingerprint']=fingerprint(plan)
            cache_report={'status':'PASS','stage':'complete','cached_studies':2200,'plan_fingerprint':plan['fingerprint'],'gate':{'passed':True}}
            for name,v in [('plan-private.json',plan),('cache-report.json',cache_report),
                           ('index-private.json',{str(i):{} for i in range(2200)})]:
                (cache/name).write_text(json.dumps(v))
            report,run=prefix_fixture();contract=report['contract'];y,w=ct.supervision(plan)
            contract.update(config=ct.CONFIG,plan_fingerprint=plan['fingerprint'],source_hashes=builder.SOURCE_HASHES,
                index_sha256=builder.file_sha256(cache/'index-private.json'),complete_sha256=builder.file_sha256(cache/'cache-report.json'),
                labels_fingerprint=fingerprint(y.tolist()),weights_fingerprint=fingerprint(w.tolist()))
            contract['fingerprint']=fingerprint({k:v for k,v in contract.items() if k!='fingerprint'})
            for name,v in [('training-report.json',report),('run-contract.json',run)]:(ref/name).write_text(json.dumps(v))
            with patch.object(builder,'check_sources'):
                result=builder.build(src,root/'bundle','objective-test',cache,cache_kernel='ozgung/synthetic-cache',
                    reference_result=ref,reference_kernel='ozgung/synthetic-reference',objective=obj.GLOBAL_OBJECTIVE)
            meta=json.loads((root/'bundle/kernel-metadata.json').read_text())
            self.assertFalse(meta['enable_internet']);self.assertEqual(meta['dataset_sources'],[])
            self.assertEqual(result['reference_prefix']['completed_epochs'],1)
            self.assertEqual(result['reference_pins']['initial.pt'],'a'*64)
            nb=json.loads((root/'bundle/notebook.ipynb').read_text())
            source=''.join(nb['cells'][1]['source']);compile(source,'objective-bundle','exec')
            self.assertIn("objective=CONTRACT['objective']",source)
            calls=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='train']
            self.assertEqual(len(calls),1)
            self.assertNotIn('resume',[k.arg for k in calls[0].keywords])


if __name__=='__main__':unittest.main()
