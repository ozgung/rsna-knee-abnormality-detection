import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from knee.public_fullfield import (config,fullfield_windows,qualify_windows,centers,
    prediction_error,decision,make_candidate,snapshot,require_immutable,validate_grid_pair,GRID_NAMES)
try:
    import numpy as np
    import torch
except ImportError:np=None

SOURCE=Path('/Users/ozgun/Dev/rsna-knee-abnormality-detection/outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py')

@unittest.skipIf(np is None or not SOURCE.exists(),'ML packages and audited source needed')
class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from knee.reference import audit_source,load_definitions
        cls.public,_=load_definitions(*audit_source(SOURCE.read_text()));torch.set_num_threads(4)
    def test_exact_original_and_mirrored_center_border_repeat(self):
        v=np.random.default_rng(3).integers(0,256,(96,384,384),dtype=np.uint8);m=np.ones(96,np.uint8)
        result=qualify_windows(self.public,v,m,device='cpu')
        self.assertEqual(set(result),{'normal','mirrored'})
        x=fullfield_windows(self.public,v,m)
        values=((torch.from_numpy(v[[0,1,2]].astype(np.float32)/255)-self.public._MEAN)/self.public._STD)
        self.assertTrue(torch.equal(x[0],values));self.assertEqual(tuple(x.shape),(94,3,384,384))
        before=x[:,:,32:352,32:352].clone();v[:,0,0]^=255
        changed=fullfield_windows(self.public,v,m);self.assertTrue(torch.equal(before,changed[:,:,32:352,32:352]));self.assertFalse(torch.equal(x,changed))
    def test_partial_short_cross_slot_centers_preserved(self):
        v=np.random.default_rng(4).integers(1,256,(96,384,384),dtype=np.uint8)
        for interval in ((48,66),(4,7),(0,96)):
            m=np.zeros(96,np.uint8);m[slice(*interval)]=1
            result=qualify_windows(self.public,v,m,device='cpu');self.assertEqual(len(result),2)
            expected=np.asarray(self.public._eval_centers(m,96,94))
            np.testing.assert_array_equal(centers(self.public,m),expected)
        mirror=fullfield_windows(self.public,v,np.ones(96,np.uint8),mirror=True)
        normal=fullfield_windows(self.public,v,np.ones(96,np.uint8))
        # Center26 crosses sagittal slots but uses inherited center-slot channel reversal.
        raw=v[[24,25,26]][::-1].astype(np.float32)/255
        expected=(torch.from_numpy(raw.copy())-self.public._MEAN)/self.public._STD
        self.assertTrue(torch.equal(mirror[24],expected));self.assertFalse(torch.equal(mirror[24],normal[24]))
    def test_bad_schema_and_untouched320_assertion(self):
        v=np.ones((96,384,384),np.uint8)
        with self.assertRaises(ValueError):fullfield_windows(self.public,v,np.zeros(96,np.uint8))
        with self.assertRaises(ValueError):fullfield_windows(self.public,v.astype(np.float32),np.ones(96,np.uint8))
        with self.assertRaises(AssertionError):self.public.gpu_windows(v,np.ones(96,np.uint8),94,384,'cpu')
    def test_prediction_tolerance_finite_and_shape(self):
        self.assertLessEqual(prediction_error(np.ones(12),np.ones(12)+.5e-6,'TEST'),1e-6)
        for other in (np.ones(12)+2e-6,np.full(12,np.nan),np.ones(11)):
            with self.assertRaises(ValueError):prediction_error(np.ones(12),other,'TEST')
    def test_inference_requires_cuda_and384(self):
        from knee.public_fullfield import infer_candidate
        with self.assertRaises(ValueError):infer_candidate(self.public,None,torch.ones(94,3,320,320),torch.ones(94,3,320,320),'cuda')
        with self.assertRaises(ValueError):infer_candidate(self.public,None,torch.ones(94,3,384,384),torch.ones(94,3,384,384),'cpu')

class DecisionTests(unittest.TestCase):
    def metrics(self,delta,damage=0):
        targets={f't{i}':{'auc':.8,'positive':20,'negative':20} for i in range(4)}
        candidate={t:{**row,'auc':row['auc']-(.02 if i<damage else 0)} for i,(t,row) in enumerate(targets.items())}
        return {'control':{'macro_auc':.8,'targets':targets},'candidate':{'macro_auc':.8+delta,'targets':candidate}}
    def test_fixed_triage_lower_bound_and_damage(self):
        self.assertIn('closed',decision(self.metrics(0),{'ci95':[.001,.01]})['triage'])
        self.assertIn('inconclusive',decision(self.metrics(.001),{'ci95':[-.001,.003]})['triage'])
        self.assertIn('inconclusive',decision(self.metrics(.001),{'ci95':[0,.003]})['triage'])
        self.assertIn('eligible',decision(self.metrics(.001),{'ci95':[.0001,.003]})['triage'])
        self.assertIn('closed',decision(self.metrics(.001,3),{'ci95':[.0001,.003]})['triage'])
    def test_metrics_support_and_training_prevalence_only(self):
        if np is None:self.skipTest('numpy needed')
        from knee.public_adaptation_fit import metric_report,paired_bootstrap
        y=np.array([[0,0],[1,1],[0,1],[1,0]],dtype=float);m=np.array([[1,0],[1,1],[1,1],[1,0]],dtype=float)
        p=np.array([[.1,.5],[.9,.5],[.2,.5],[.8,.5]])
        result=metric_report(y,m,p,['a','b']);self.assertEqual(result['targets']['a']['auc'],1);self.assertIsNone(result['targets']['b']['auc']);self.assertEqual(result['targets']['b']['missing'],2)
        boot=paired_bootstrap(y,m,p,p,['a','a','b','b'],draws=2000,seed=42,targets=['a','b'])
        self.assertEqual(boot['ci95'],[0.,0.]);self.assertEqual(boot['draws'],2000)
    def test_final_analysis_numpy_truth_digest_and_json(self):
        if np is None:self.skipTest('numpy needed')
        from knee.public_fullfield import analyze
        y=np.tile(np.array([[0.,1.],[1.,0.]]),(10,1));m=np.ones_like(y)
        train_y=np.tile(np.array([[0.,0.],[1.,0.],[1.,1.],[1.,0.]]),(5,1));train_m=np.ones_like(train_y)
        predictions=.1+.8*y
        result=analyze(y,m,train_y,train_m,predictions,predictions,np.repeat(np.arange(10),2),['a','b'])
        self.assertEqual(len(result['truth_array_sha256']),64)
        self.assertEqual(result['train_prevalence_fit']['a']['probability'],.75)
        self.assertEqual(result['train_prevalence_fit']['b']['probability'],.25)
        self.assertEqual(result['metrics']['train_prevalence']['macro_auc'],.5)
        self.assertEqual(result['paired_bootstrap']['draws'],2000)
        json.dumps(result,allow_nan=False)
    def test_config_has_no_fit_or_search(self):
        self.assertEqual(config()['optimizer_updates'],0);self.assertEqual(config()['chunk'],16);self.assertEqual(config()['candidate_res'],384)
    def test_non_cuda_runtime_preserves_failure_receipt(self):
        if np is None:self.skipTest('numpy needed')
        from knee.public_fullfield import run
        with tempfile.TemporaryDirectory() as d,patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(RuntimeError):run('','',{},'missing','missing','missing',d,Path(d)/'out',{},[],{},[],{},np.zeros((870,12),np.float32))
            receipt=json.loads(Path(d,'out','receipt.json').read_text())
            self.assertEqual(receipt['status'],'FAIL');self.assertEqual(receipt['failure_stage'],'setup');self.assertEqual(receipt['optimizer_updates'],0)
    def test_build_tamper(self):
        from scripts.build_public_fullfield import verify_build
        from knee.reference import sha256
        with tempfile.TemporaryDirectory() as d:
            p=Path(d,'notebook.ipynb');p.write_text('{}');Path(d,'build-receipt.json').write_text(json.dumps({'files':{'notebook.ipynb':sha256(p.read_bytes())}}))
            verify_build(d);p.write_text('tampered')
            with self.assertRaises(ValueError):verify_build(d)

@unittest.skipIf(np is None or not SOURCE.exists(),'ML packages and audited source needed')
class GridTests(unittest.TestCase):
    def test_actual_timm_state_transfer_grid_oracle_reload_and_mutation(self):
        try:import timm
        except ImportError:self.skipTest('timm1.0.26 unavailable')
        if timm.__version__!='1.0.26':self.skipTest('requires exacttimm1.0.26')
        from knee.reference import audit_source,load_definitions
        public,_=load_definitions(*audit_source(SOURCE.read_text()));public.timm=timm
        backbone=public.build_backbone('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',pretrained=False)
        control=public.RaptorClassifier(backbone,F_dim=backbone.num_features).eval()
        before_namespace=(public.INPUT_RES,public.gpu_windows.__code__,public.build_backbone.__code__)
        candidate,grid=make_candidate(public,control)
        self.assertEqual(grid['allowed_nonpersistent_differences'],32)
        self.assertEqual(before_namespace,(public.INPUT_RES,public.gpu_windows.__code__,public.build_backbone.__code__))
        oracle={320:{k:v.detach().cpu().clone() for k,v in control.named_buffers() if k in GRID_NAMES},
                384:{k:v.detach().cpu().clone() for k,v in candidate.named_buffers() if k in GRID_NAMES}}
        validate_grid_pair(control,candidate,oracle)
        broken={r:dict(v) for r,v in oracle.items()};key=next(iter(GRID_NAMES));broken[384][key]=broken[384][key]+1
        with self.assertRaisesRegex(ValueError,'CONSTRUCTOR_GRID_ORACLE'):validate_grid_pair(control,candidate,broken)
        before=snapshot(candidate);state={k:v.clone() for k,v in control.state_dict().items()}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d,'frozen.pt');torch.save(state,path);candidate.load_state_dict(torch.load(path,weights_only=True),strict=True)
        require_immutable(candidate,before)
        with torch.inference_mode():
            a=candidate(torch.zeros(1,1,3,384,384));b=candidate(torch.zeros(1,1,3,384,384))
        self.assertTrue(torch.equal(a,b));self.assertEqual(tuple(a.shape),(1,12));self.assertTrue(torch.isfinite(a).all())
        require_immutable(candidate,before)
        # Persistent-state and nonpersistent-grid corruption each fail closed.
        with torch.no_grad():candidate.clsb.add_(1)
        with self.assertRaises(ValueError):require_immutable(candidate,before)
        candidate.load_state_dict(state,strict=True)
        buffers=dict(candidate.named_buffers());key=next(iter(GRID_NAMES));buffers[key].add_(1)
        with self.assertRaises(ValueError):require_immutable(candidate,before)
