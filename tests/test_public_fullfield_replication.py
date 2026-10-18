"""Synthetic fold1 population, leakage, reference and persistence contracts."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from knee.public_adaptation_fit import fingerprint,report_components
from knee.public_fullfield_replication import (config,membership,freeze_sample,construct_preflight,
    save_predictions,reference_control,replication_analysis,validate_fold0,run,SAMPLE_PREFIX)
from knee.reference import sha256
try:
    import numpy as np
except ImportError:np=None

TARGETS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']


def fixture():
    rows=[];folds={};ids=[]
    for i in range(4407):
        uid='synthetic-%04d'%i;ids.append(uid)
        row={'StudyInstanceUID':uid,'Report':'unique-report-%04d'%i,'PatientID':'patient-%04d'%i,**{t:('0' if i>=4349 else '') for t in TARGETS}}
        rows.append(row);folds[uid]=-1 if i>=4349 else (1 if 4<=i<870 else 2)
    return rows,ids,folds,ids[:4]


class MembershipTests(unittest.TestCase):
    def setUp(self):self.rows,self.ids,self.folds,self.excluded=fixture()
    def test_exact866_holdout3479_training4345_eligible(self):
        eligible,train,dev,gold=membership(self.rows,self.ids,self.folds,self.excluded,TARGETS)
        self.assertEqual((len(eligible),len(train),len(dev),len(gold)),(4345,3479,866,58))
        self.assertTrue(all(self.folds[s]==1 for s in dev));self.assertFalse(set(train)&set(dev))
        self.assertFalse((set(train)|set(dev))&set(self.excluded));self.assertFalse(set(eligible)&gold)
    def test_rejects870_or_fold0_membership_substitution(self):
        folds=dict(self.folds)
        for uid in self.ids[870:874]:folds[uid]=1
        with self.assertRaisesRegex(ValueError,'FOLD1_MEMBERSHIP'):membership(self.rows,self.ids,folds,self.excluded,TARGETS)
        folds={s:(0 if f==1 else f) for s,f in self.folds.items()}
        with self.assertRaises(ValueError):membership(self.rows,self.ids,folds,self.excluded,TARGETS)
    def test_duplicate_official_and_missing_eligible_label_rejected(self):
        with self.assertRaises(ValueError):membership(self.rows[:-1]+[self.rows[0]],self.ids,self.folds,self.excluded,TARGETS)
        with self.assertRaises(ValueError):membership(self.rows,self.ids[5:],self.folds,self.excluded,TARGETS)
    def test_sample_precisely_hash_ordered_training_only_without_scores(self):
        sample=freeze_sample(self.rows,self.folds,self.excluded,TARGETS)
        pool=[s for s in self.ids[4:4349] if self.folds[s]!=1]
        expected=sorted(pool,key=lambda s:(sha256((SAMPLE_PREFIX+s).encode()),s))[:32]
        self.assertEqual(sample,expected);self.assertEqual(len(set(sample)),32)
        changed=[{**r,'Report':'completely different'} for r in self.rows]
        self.assertEqual(sample,freeze_sample(changed,self.folds,self.excluded,TARGETS))
    def test_report_and_patient_component_split_leakage_rejected(self):
        _,train,dev,_=membership(self.rows,self.ids,self.folds,self.excluded,TARGETS)
        indexed={r['StudyInstanceUID']:r for r in self.rows}
        indexed[train[0]]['Report']=indexed[dev[0]]['Report']
        with self.assertRaisesRegex(ValueError,'GROUP_SPLIT_LEAKAGE'):report_components(self.rows,train,dev)
        indexed[train[0]]['Report']='separate-report'
        indexed[train[0]]['PatientID']=indexed[dev[0]]['PatientID']
        with self.assertRaisesRegex(ValueError,'GROUP_SPLIT_LEAKAGE'):report_components(self.rows,train,dev)
    def test_schema_targets_exact_count_and_distinct(self):
        for targets in (TARGETS[:-1],TARGETS[:-1]+[TARGETS[0]],TARGETS[:-1]+['unknown']):
            with self.assertRaises(ValueError):membership(self.rows,self.ids,self.folds,self.excluded,targets)
    def test_preflight_masks_and_component_fingerprints(self):
        if np is None:self.skipTest('NumPy unavailable')
        eligible,_,_,_=membership(self.rows,self.ids,self.folds,self.excluded,TARGETS)
        masks={s:np.ones(12,np.float32) for s in eligible};pre,train,dev,groups=construct_preflight(self.rows,self.ids,self.folds,self.excluded,TARGETS,masks)
        self.assertEqual(pre['counts']['development'],866);self.assertEqual(pre['fingerprints']['dev'],fingerprint(dev))
        self.assertEqual(pre['fingerprints']['groups'],fingerprint(groups));self.assertEqual(pre['support']['ACL']['dev_observed'],866)
        masks[eligible[0]][0]=.5
        with self.assertRaisesRegex(ValueError,'MASK_SCHEMA'):construct_preflight(self.rows,self.ids,self.folds,self.excluded,TARGETS,masks)


@unittest.skipIf(np is None,'NumPy unavailable')
class PredictionTests(unittest.TestCase):
    def test_reference_uses_original_eval_window_and_probability_functions(self):
        calls=[]
        def windows(v,m,k,res,mirror=False):calls.append(('windows',k,res,mirror));return mirror
        def infer(model,plain,mirrored,device):calls.append(('infer',plain,mirrored,device));return np.linspace(.2,.8,12)
        public=SimpleNamespace(eval_windows=windows,infer_pair_probs=infer)
        p=reference_control(public,object(),None,None);np.testing.assert_array_equal(p,np.linspace(.2,.8,12))
        self.assertEqual(calls,[('windows',94,320,False),('windows',94,320,True),('infer',False,True,'cuda')])
    def test_atomic_prediction_checkpoint_order_and_failure_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'pred.npz';p=np.full((2,12),.5,np.float32)
            save_predictions(path,['a','b'],p,p);original=path.read_bytes()
            with np.load(path,allow_pickle=False) as ar:self.assertEqual(ar['study_ids'].tolist(),['a','b']);self.assertEqual(ar['control'].dtype,np.float32)
            with self.assertRaises(ValueError):save_predictions(path,['a','a'],p,p)
            with self.assertRaises(ValueError):save_predictions(path,['a','b'],p,np.full((2,12),np.nan))
            self.assertEqual(path.read_bytes(),original)
            with patch('numpy.savez',side_effect=OSError('synthetic interrupted write')):
                with self.assertRaises(OSError):save_predictions(path,['a','b'],p,p)
            self.assertEqual(path.read_bytes(),original);self.assertFalse(path.with_name('pred.npz.tmp').exists())
    def test866_analysis_all12_damage_and_rehearsal_gate(self):
        y=np.tile(np.arange(866)[:,None]%2,(1,12)).astype(float);m=np.ones_like(y)
        pre={'support':{t:{'dev_observed':866} for t in TARGETS}}
        result={'macro_delta':.002,'target_damage':[],'paired_bootstrap':{'ci95':[.001,.003]},'metrics':{'control':{'defined_targets':12},'candidate':{'defined_targets':12}}}
        with patch('knee.public_fullfield_replication.analyze',return_value=result.copy()):
            actual=replication_analysis(y,m,y,m,y,y,np.arange(866),TARGETS,pre);self.assertEqual(actual['triage'],'eligible for image-only offline rehearsal')
        for change in ({'macro_delta':0.},{'target_damage':['a','b','c']},{'paired_bootstrap':{'ci95':[-.001,.003]}}):
            with patch('knee.public_fullfield_replication.analyze',return_value={**result,**change}):
                actual=replication_analysis(y,m,y,m,y,y,np.arange(866),TARGETS,pre);self.assertIn('closed',actual['triage'])
        with self.assertRaises(ValueError):replication_analysis(y[:865],m[:865],y,m,y,y,np.arange(866),TARGETS,pre)
    def test_actual_inherited_2k_component_bootstrap_and_prevalence(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable for inherited truth digest')
        y=np.tile(np.arange(866)[:,None]%2,(1,12)).astype(float);m=np.ones_like(y);p=.1+.8*y
        pre={'support':{t:{'dev_observed':866} for t in TARGETS}}
        result=replication_analysis(y,m,y,m,p,p,np.arange(866)//2,TARGETS,pre)
        self.assertEqual(result['paired_bootstrap']['draws'],2000);self.assertEqual(result['paired_bootstrap']['seed'],42)
        self.assertEqual(result['paired_bootstrap']['ci95'],[0.,0.]);self.assertEqual(result['metrics']['train_prevalence']['macro_auc'],.5)
        self.assertIn('closed',result['triage'])
    def test_no_cuda_saves_sanitized_failure_without_using_labels(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable')
        with tempfile.TemporaryDirectory() as d,patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(RuntimeError):run('','',{},'missing','missing','missing',d,Path(d)/'out',{},[],{},[],{}, {})
            r=json.loads(Path(d,'out/receipt.json').read_bytes());self.assertEqual(r['status'],'FAIL');self.assertEqual(r['failure_stage'],'setup');self.assertEqual(r['optimizer_updates'],0)


class BuilderTests(unittest.TestCase):
    def test_config_fixed_fold1_no_historical_control(self):
        c=config();self.assertEqual(c['holdout_fold'],1);self.assertEqual(c['development_studies'],866)
        self.assertEqual(c['control_res'],320);self.assertEqual(c['candidate_res'],384);self.assertEqual(c['optimizer_updates'],0)
        self.assertIn('no historical',c['control_provenance'])
    def test_fold0_receipt_must_pass_parent_replication_gate(self):
        from knee.public_fullfield import config as oldconfig
        manifest={'synthetic':True};r={'status':'PASS','optimizer_updates':0,'qualification_completed':32,'evaluated_studies':870,
             'mechanical_gate':'PASS','final_state_and_all_buffers_immutable':True,'binding':{'public_manifest_sha256':fingerprint(manifest),'config':oldconfig()},
             'metrics':{'control':{'defined_targets':12},'candidate':{'defined_targets':12}},'macro_delta':.001,'paired_bootstrap':{'ci95':[.0001,.002]},'target_damage':[]}
        validate_fold0(r,manifest)
        for changes in ({'optimizer_updates':1},{'evaluated_studies':866},{'macro_delta':0.},{'target_damage':['a','b','c']}):
            with self.assertRaises(ValueError):validate_fold0({**r,**changes},manifest)
    def test_build_immutable_file_and_no_upstream_module_changes(self):
        from scripts.build_public_fullfield_replication import verify_build
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'notebook.ipynb').write_text('{}');(p/'build-receipt.json').write_text(json.dumps({'files':{'notebook.ipynb':sha256((p/'notebook.ipynb').read_bytes())}}))
            verify_build(p);(p/'notebook.ipynb').write_text('tampered')
            with self.assertRaises(ValueError):verify_build(p)


if __name__=='__main__':unittest.main()
