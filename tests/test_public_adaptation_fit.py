"""Focused fixed-fit mechanics. Synthetic evidence never releases CUDA gates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from knee.public_adaptation_fit import (atomic_save, average_endpoint, fit_config,
    fingerprint, qualification_gate, validate_cursor, PlatformInterruption,
    SCIENTIFIC_REVISION, SCIENTIFIC_BUNDLE, INITIAL_SHA, metric_report,
    paired_bootstrap, report_components, train_epochs, load_binary_truth)
from knee.public_adaptation_pilot import (scientific_config, LABEL_SHA, MASK_SHA,
    masked_loss, study_mean_backward, optimizer_groups, state_fingerprint)

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None


class DependencyFreeTests(unittest.TestCase):
    def test_atomically_preserves_prior_boundary_on_partial_write(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'resume.pt'; p.write_bytes(b'old boundary')
            def fail(value,handle): handle.write(b'incomplete'); raise OSError('actual write failure')
            with self.assertRaises(OSError): atomic_save(p,None,fail)
            self.assertEqual(p.read_bytes(),b'old boundary')
            self.assertFalse(p.with_name(p.name+'.tmp').exists())
            atomic_save(p,b'new boundary',lambda value,h:h.write(value))
            self.assertEqual(p.read_bytes(),b'new boundary')

    def test_cursor_is_an_optimizer_boundary_and_epoch_end_is_normalized(self):
        orders=[list(range(31)),list(reversed(range(31)))]
        for values in ((0,0,0,0),(0,16,1,0),(1,0,2,1),(1,16,3,1),(2,0,4,2)):
            validate_cursor(*values[:3],orders,values[3])
        for values in ((0,15,1,0),(0,31,2,1),(1,0,1,1),(2,1,4,2),(1,16,3,0)):
            with self.assertRaises(ValueError): validate_cursor(*values[:3],orders,values[3])

    def test_platform_signal_requests_boundary_save_without_time_cap(self):
        request=PlatformInterruption(); request.notify(15)
        self.assertTrue(request.requested); self.assertEqual(request.signum,15)
        self.assertNotIn('time_budget',fit_config())

    def test_group_components_are_transitive_and_detect_fixed_split_leakage(self):
        rows=[{'StudyInstanceUID':'a','Report':'different','PatientID':'p0'},
              {'StudyInstanceUID':'b','Report':' SAME  report ','PatientID':'p1'},
              {'StudyInstanceUID':'c','Report':'same report','PatientID':'p2'},
              {'StudyInstanceUID':'d','Report':'other','PatientID':'p2'}]
        groups,summary=report_components(rows,['a'],['b','c','d'])
        self.assertEqual(len(set(groups)),1)
        self.assertEqual(summary['missing_patient_identity'],0)
        with self.assertRaisesRegex(ValueError,'GROUP_SPLIT_LEAKAGE'):
            report_components(rows,['c'],['b','d'])

    def test_builder_requires_exact_parent_reviewed_actual_bytes(self):
        from scripts.build_public_adaptation_fit import reviewed_json
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'receipt.json';p.write_text('{"status":"FAIL"}\n')
            with self.assertRaises(ValueError): reviewed_json(p,'0'*64)
            with self.assertRaises(ValueError): reviewed_json(p,None)

    def test_packaging_compiles_private_inputs_and_preserves_scientific_sources(self):
        from scripts.build_public_adaptation_fit import build_fit
        from knee.public_adaptation_fit import FOLD_SHA, EXCLUDED_SHA
        from knee.public_adaptation_pilot import file_sha as real_sha
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); paths={}
            fixtures={'source':'synthetic audited source','cache':{},'cpu':{},'s1':{'status':'PASS'},
                      's2':{'status':'PASS'},'manifest':{},'folds':{'folds':{'synthetic-private-study':0}},
                      'excluded':{'quarantined_study_ids':[]},'train':'synthetic official bytes'}
            fixtures['pre']={'fingerprints':{'fold_mapping':fingerprint(fixtures['folds']['folds'])}}
            for key,value in fixtures.items():
                path=p/key;path.write_text(value if isinstance(value,str) else json.dumps(value));paths[key]=path
            manifest={'source_sha256':'s','attribution':'Synthetic test','source_url':'https://example.test',
                      'checkpoint_file':'synthetic.pt'}
            contract={'source_sha256':'s','manifest_sha256':fingerprint(manifest)}
            def fake_sha(path):
                if Path(path)==paths['folds']: return FOLD_SHA
                if Path(path)==paths['excluded']: return EXCLUDED_SHA
                return real_sha(path)
            with patch('scripts.build_public_adaptation_fit.cache_contract',return_value=contract), \
                 patch('scripts.build_public_adaptation_fit.stream_contract',return_value={}), \
                 patch('scripts.build_public_adaptation_fit.audit_source',return_value=('synthetic','',manifest)), \
                 patch('scripts.build_public_adaptation_fit.qualification_gate',return_value={'synthetic':True}), \
                 patch('scripts.build_public_adaptation_fit.file_sha',side_effect=fake_sha):
                result=build_fit(paths['source'],paths['cache'],paths['cpu'],paths['s1'],paths['s2'],paths['manifest'],
                    paths['pre'],paths['folds'],paths['excluded'],paths['train'],p/'build','synthetic-fit',
                    expected_stage1_sha256=real_sha(paths['s1']),expected_stage2_sha256=real_sha(paths['s2']),
                    expected_manifest_sha256=real_sha(paths['manifest']),expected_preflight_sha256=real_sha(paths['pre']))
            notebook=json.loads((p/'build/notebook.ipynb').read_text())
            compile(''.join(notebook['cells'][1]['source']),'<synthetic-private-build>','exec')
            self.assertFalse(result['launched'])
            self.assertNotIn('synthetic-private-study',(p/'build/build-receipt.json').read_text())
            metadata=json.loads((p/'build/kernel-metadata.json').read_text())
            self.assertTrue(metadata['is_private']);self.assertFalse(metadata['enable_internet'])
            self.assertEqual(metadata['kernel_sources'],[])

    def test_qualification_rejects_synthetic_incomplete_and_wrong_lineage(self):
        # A structurally complete fixture permits precise negative controls;
        # it is never persisted or used to create an actual build.
        contract={'example':'cache'};stream={'example':'stream'}
        binding={'cache':contract,'stream':stream,'scientific_config':scientific_config(),
                 'source_revision':SCIENTIFIC_REVISION,'bundle_sha256':SCIENTIFIC_BUNDLE}
        manifest={'schema':1,'purpose':'disposable external recovery qualification','status':'PASS',
                  'binding':binding,'checkpoint_file':'recovery.pt','checkpoint_bytes':100,
                  'checkpoint_sha256':'a'*64,'updates_before':2,'updates_after':3,
                  'consumed_before':0,'consumed_after':3,
                  'expected':{k:'b'*64 for k in ('model','optimizer','rng','sampler','gradients','prediction','forward')}}
        common={'status':'PASS','contract':contract,'bundle_sha256':SCIENTIFIC_BUNDLE,
                'source_revision':SCIENTIFIC_REVISION,'checkpoint_sha256':INITIAL_SHA,
                'labels_sha256':LABEL_SHA,'masks_sha256':MASK_SHA,'bn_buffers_unchanged':True,
                'updates':3,'exact_resume_status':'PASS','scientific_config':scientific_config(),
                'qualified_cpu_stream':stream,'input_mode':'protected raw stream'}
        first={**common,'actual_group_denominators':[16,14,3],'independent_inline_gradients_exact':True,
               'stream_rng_unchanged_via_exact_cached_replay':True,'recovery_artifact_exported':True,
               'recovery_manifest_sha256':fingerprint(manifest),'recovery_checkpoint_sha256':'a'*64,
               'recovery_checkpoint_bytes':100,'save_reload_logit_max_error':0,
               'resume_model_max_error':0,'resume_optimizer_max_error':0}
        second={**common,'external_artifact_recovery_status':'PASS','external_artifact_bytes_verified':True,
                'external_expected_fingerprints':manifest['expected'],'exact_preclip_gradients':True,
                'qualified_optimizer_updates':1,'actual_optimizer_step_calls':1,
                'external_next_study_forward_backward_passes':3,'save_reload_status':'PASS'}
        self.assertEqual(qualification_gate(first,second,manifest,contract,stream)['stage1'],fingerprint(first))
        for target,key,value in ((first,'status','RUNNING'),(first,'source_revision','c'*40),
                                 (first,'actual_group_denominators',[16,16,3]),
                                 (second,'external_artifact_bytes_verified',False),
                                 (second,'external_expected_fingerprints',{})):
            bad=copy.deepcopy(target);bad[key]=value
            with self.assertRaises(ValueError): qualification_gate(bad if target is first else first,
                bad if target is second else second,manifest,contract,stream)


@unittest.skipIf(torch is None,'Optional torch/numpy unavailable')
class FitMechanicsTests(unittest.TestCase):
    def model(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone=torch.nn.Sequential(torch.nn.Linear(2,3),torch.nn.BatchNorm1d(3))
                self.att=torch.nn.Sequential(torch.nn.LayerNorm(3),torch.nn.Dropout(.2),torch.nn.Linear(3,2))
            def forward(self,x): return self.att(self.backbone(x))
        torch.manual_seed(7); return Toy().float()
    def optimizer(self,model):
        return torch.optim.AdamW(optimizer_groups(model),betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
    def loss(self,model,sid,epoch,value):
        index=int(sid)
        x=torch.tensor([[index/31,(epoch+1)/3]],dtype=torch.float32)
        y=torch.tensor([[index%2,(index%3)/2]],dtype=torch.float32)
        mask=torch.tensor([[1.,float(index%3!=0)]])
        return masked_loss(model(x),y,mask)

    def test_analytic_unequal_masks_and_15_tail_reject_fixed16_and_pooled_cells(self):
        observed=[1,2,3]*5
        parameter=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
        truths=[torch.linspace(0,1,k,dtype=torch.float64) for k in observed]
        for truth in truths:
            logits=parameter.expand(1,3)
            y=torch.zeros_like(logits);mask=torch.zeros_like(logits)
            y[0,:len(truth)]=truth;mask[0,:len(truth)]=1
            study_mean_backward(masked_loss(logits,y,mask),15)
        expected=sum(float(torch.sigmoid(parameter.detach())-t.mean()) for t in truths)/15
        self.assertAlmostEqual(float(parameter.grad),expected,places=14)
        self.assertGreater(abs(float(parameter.grad)-expected*15/16),1e-4)
        # Unequal masks/truth means provide a concrete pooled-cell counterexample.
        truths=[torch.zeros(1,dtype=torch.float64) if k==1 else torch.ones(k,dtype=torch.float64) for k in observed]
        correct=sum(float(torch.sigmoid(parameter.detach())-t.mean()) for t in truths)/15
        pooled=float(torch.sigmoid(parameter.detach()))-sum(float(t.sum()) for t in truths)/sum(observed)
        self.assertGreater(abs(correct-pooled),.1)

    def test_external_restore_matches_updates_rng_and_fixed_endpoint_with_tail15(self):
        from knee.public_study_stream import StudyStream
        contributors=[str(i) for i in range(31)]; binding={'source':'toy','config':'fixed','fold':'a','bytes':'b'}
        def stream(order,cursor): return StudyStream(order,lambda sid:sid,contract='toy',consumed_cursor=cursor,prefetch=4)
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            model=self.model(); optimizer=self.optimizer(model)
            torch.manual_seed(42);np.random.seed(42)
            full=train_epochs(model,optimizer,contributors,self.loss,binding,p/'full.pt',stream_factory=stream)
            full_state=torch.load(p/'full.pt',weights_only=True)
            model=self.model();optimizer=self.optimizer(model);stop=PlatformInterruption()
            torch.manual_seed(42);np.random.seed(42)
            interrupted=train_epochs(model,optimizer,contributors,self.loss,binding,p/'first.pt',
                stream_factory=stream,interruption=stop,on_boundary=lambda e,c,u:stop.notify(15) if u==1 else None)
            self.assertEqual(interrupted['status'],'INTERRUPTED');self.assertEqual(interrupted['updates'],1)
            (p/'external.pt').write_bytes((p/'first.pt').read_bytes())
            model=self.model(); optimizer=self.optimizer(model)
            restored=train_epochs(model,optimizer,contributors,self.loss,binding,p/'external.pt',stream_factory=stream,resume=True)
            final=torch.load(p/'external.pt',weights_only=True)
            self.assertEqual(restored['updates'],4)
            for key in ('model','optimizer','rng','orders','epoch_states'):
                self.assertEqual(state_fingerprint(full_state[key]),state_fingerprint(final[key]),key)
            self.assertEqual(state_fingerprint(full['endpoint']),state_fingerprint(restored['endpoint']))
            self.assertEqual(state_fingerprint(full_state['original_bn']),state_fingerprint({k:v for k,v in full['endpoint'].items() if k in full_state['original_bn']}))
            for key,value in (('source','foreign'),('config','wrong'),('fold','wrong'),('bytes','wrong')):
                with self.assertRaisesRegex(ValueError,'RESUME_BINDING'):
                    train_epochs(self.model(),self.optimizer(self.model()),contributors,self.loss,
                        {**binding,key:value},p/'external.pt',resume=True)
            wrong=copy.deepcopy(final);wrong['epoch']=1;wrong['cursor']=15
            atomic_save(p/'wrong.pt',wrong)
            bad_model=self.model()
            with self.assertRaisesRegex(ValueError,'RESUME_CURSOR'):
                train_epochs(bad_model,self.optimizer(bad_model),contributors,self.loss,binding,p/'wrong.pt',resume=True)

    def test_initial_boundary_restores_before_first_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'resume.pt';model=self.model(); optimizer=self.optimizer(model)
            with self.assertRaisesRegex(RuntimeError,'decode failed'):
                train_epochs(model,optimizer,['0'],lambda *args:(_ for _ in ()).throw(RuntimeError('decode failed')),{},path)
            model=self.model()
            result=train_epochs(model,self.optimizer(model),['0'],self.loss,{},path,resume=True)
            self.assertEqual(result['updates'],2)

    def test_endpoint_keeps_bn_and_nonfloat_buffers_and_rejects_drift(self):
        a={'weight':torch.tensor([2.,4.]),'running_mean':torch.tensor([3.]),'counter':torch.tensor(2)}
        b={'weight':torch.tensor([4.,8.]),'running_mean':torch.tensor([3.]),'counter':torch.tensor(2)}
        bn={'running_mean':torch.tensor([3.])}
        result=average_endpoint([a,b],['weight'],bn)
        self.assertTrue(torch.equal(result['weight'],torch.tensor([3.,6.])))
        for key,value in (('running_mean',torch.tensor([4.])),('counter',torch.tensor(3))):
            bad={**b,key:value}
            with self.assertRaises(ValueError): average_endpoint([a,bad],['weight'],bn)

    def test_binary_metrics_undefined_missing_and_paired_cluster_coverage(self):
        y=np.array([[0,0],[1,0],[0,0],[1,0]])
        m=np.array([[1,1],[1,1],[1,0],[1,0]])
        c=np.array([[.1,.5],[.2,.5],[.8,.5],[.9,.5]])
        e=np.array([[.1,.5],[.9,.5],[.2,.5],[.8,.5]])
        report=metric_report(y,m,e,['valid','undefined'])
        self.assertEqual(report['defined_targets'],1);self.assertEqual(report['macro_auc'],1.)
        self.assertEqual(report['targets']['undefined']['missing'],2)
        self.assertIsNone(report['targets']['undefined']['auc'])
        result=paired_bootstrap(y,m,c,e,['a','a','b','b'],draws=200)
        self.assertEqual(result['valid_draws'],200);self.assertEqual(result['original_defined_target_indices'],[0])
        self.assertEqual(result['clusters'],2)
        with self.assertRaises(ValueError): paired_bootstrap(y,m,c,e,['a'],draws=200)
        with self.assertRaises(ValueError): metric_report(y.astype(float)+.1,m,e,['a','b'])

    def test_bootstrap_reports_invalid_targets_and_no_ci_without_redrawing(self):
        y=np.array([[0.],[1.]]);m=np.ones_like(y);p=np.array([[.1],[.9]])
        result=paired_bootstrap(y,m,p,p,['a','b'],draws=200)
        self.assertGreater(result['undefined_draws'],0)
        self.assertEqual(result['draws'],200)
        self.assertEqual(result['valid_draws']+result['undefined_draws'],200)
        self.assertEqual(result['invalid_target_causes']['0'],result['undefined_draws'])
        # One deliberately unsupported draw reports null uncertainty honestly.
        no_ci=paired_bootstrap(y,m,p,p,['a','b'],draws=1,seed=3)
        self.assertEqual(no_ci['valid_draws'],0)
        self.assertIsNone(no_ci['ci95'])



@unittest.skipIf(np is None,'Optional numpy unavailable')
class BinaryTruthTests(unittest.TestCase):
    def test_original_truth_alignment_uses_binary_y_and_masks(self):
        try: import pyarrow as pa; import pyarrow.parquet as pq
        except ImportError: self.skipTest('Optional pyarrow unavailable')
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'original.parquet'
            pq.write_table(pa.table({'StudyInstanceUID':['b','a'],'y_acl':[1.,float('nan')],
                                    'm_acl':[1.,0.],'ACL':[.3,.8]}),p)
            with patch('knee.public_adaptation_fit.file_sha',return_value=MASK_SHA):
                y,m=load_binary_truth(p,['a','b'],['ACL'])
                self.assertEqual(y.tolist(),[[0.],[1.]])
                self.assertEqual(m.tolist(),[[0.],[1.]])
                pq.write_table(pa.table({'StudyInstanceUID':['a'],'y_acl':[.3],'m_acl':[1.]}),p)
                with self.assertRaisesRegex(ValueError,'TRUTH_BINARY'): load_binary_truth(p,['a'],['ACL'])
            with self.assertRaisesRegex(ValueError,'TRUTH_BYTES'): load_binary_truth(p,['a'],['ACL'])


if __name__=='__main__': unittest.main()
