"""CoAtNet224 provenance, qualification scope and actual pinned window math."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from knee import coatnet224_rehearsal as own


def member(fold=0):
    return {'fold':fold,'kernel':own.KERNELS[0 if fold<2 else (1 if fold<4 else 2)],
            'run_file':f'run_fold{fold}.json','run_sha256':'a'*64,
            'checkpoint_file':f'raptor_ft_fold{fold}_swa.pt','checkpoint_sha256':'b'*64,
            'selected_epochs':[9,8,7]}


def record():
    m=member()
    return {**{k:v for k,v in own.POLICY.items() if k!='family'},'fold':0,'smoke':False,
            'checkpoint_kind':'swa_topk','checkpoint_file':m['checkpoint_file'],
            'checkpoint_sha256':m['checkpoint_sha256'],'selected_epochs':m['selected_epochs'],
            'train_returncode':0,'epochs_requested':10,'epochs_completed':10,'stopped_by_time_budget':False,
            'train_knee_original_sha256':own.smoke.SOURCE_SHA,'params':dict(own.TRAIN_PARAMS),
            'export':{'k_eval':16,'res':224},'torch':'2.10.0+cu128','timm':'1.0.26','device':'cuda',
            'history':[{'ep':ep,'oof_auc':.7+ep/100} for ep in range(10)],
            'train_cmd':['python','train.py','--ckpt','timm']}


def contract(folds=(0,)):
    return {**own.POLICY,'version':1,'settings':dict(own.SETTINGS),'members':[member(f) for f in folds],
            'complete_family':list(folds)==list(range(5)),'model_source_sha256':own.smoke.SOURCE_SHA,
            'model_definitions_sha256':own.MODEL_DEFINITIONS_SHA,'reader_sha256':own.READER_SHA}


class Coatnet224Tests(unittest.TestCase):
    def test_exact_fitted_run_and_reject_foreign_provenance(self):
        own.verify_run(record(),member())
        changes=[{'smoke':True},{'fold':1},{'checkpoint_sha256':'c'*64},{'epochs_completed':9},
                 {'epochs_requested':14},{'stopped_by_time_budget':True},{'selection':'gold'},
                 {'labels_sha256':'c'*64},{'labels_file':'labels_bootstrap_v1.parquet'},
                 {'truth_sha256':'c'*64},{'folds_sha256':'c'*64},{'excluded_ids_sha256':'c'*64},
                 {'train_knee_original_sha256':'c'*64},{'train_knee_patched_sha256':'c'*64},
                 {'train_knee_patch_sha256':'c'*64},{'tracka_entry_sha256':'c'*64},
                 {'selected_epochs':[8,9,7]},{'history':[]},{'device':'cpu'},
                 {'train_cmd':['python','--ckpt','timm','--resume','old.pt']}]
        for change in changes:
            with self.subTest(change=change),self.assertRaises(ValueError):
                own.verify_run(record()|change,member())
        for key,value in [('ARCH','coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'),('RES','384'),
                          ('CKPT','public.pt'),('K_EVAL','24'),('ACCUM','1'),('GRAD_CKPT','0')]:
            bad=record();bad['params'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):own.verify_run(bad,member())

    def test_subset_is_wiring_only_and_contract_rejects_other_family(self):
        own.require_contract(contract())
        with self.assertRaises(ValueError):own.require_contract(contract(),complete=True)
        own.require_contract(contract(range(5)),complete=True)
        for change in [{'settings':own.SETTINGS|{'res':384}},{'family':'convnext'},
                       {'labels_sha256':'c'*64},{'model_source_sha256':'c'*64},
                       {'model_definitions_sha256':'c'*64},{'reader_sha256':'c'*64},
                       {'complete_family':True},{'members':[member(1),member(0)]},
                       {'members':[member(),member()]},{'members':[member()|{'kernel':'other/kernel'}]},
                       {'members':[member()|{'checkpoint_file':'../weights.pt'}]}]:
            with self.subTest(change=change),self.assertRaises(ValueError):own.require_contract(contract()|change)

    def test_checkpoint_wrong_resolution_order_and_nonfinite_rejected(self):
        import torch
        state={'arch':own.ARCH,'res':224,'lab':own.smoke.TARGETS,'swa_over':[9,8,7],
               'model':{'weight':torch.ones(2)}}
        own.verify_checkpoint(state,member())
        for change in [{'res':384},{'arch':'convnext_tiny.fb_in22k_ft_in1k'},
                       {'lab':list(reversed(own.smoke.TARGETS))},{'swa_over':[8,9,7]},
                       {'model':{}},{'model':{'weight':torch.tensor([float('nan')])}}]:
            with self.subTest(change=list(change)),self.assertRaises(ValueError):own.verify_checkpoint(state|change,member())

    def test_probability_mean_and_range(self):
        import numpy as np
        a=np.zeros((3,12),np.float32);b=np.ones_like(a)
        self.assertTrue(np.array_equal(own.mean_probabilities([a,b],3),np.full_like(a,.5)))
        for bad in [a[:2],np.full_like(a,np.nan),np.full_like(a,1.1)]:
            with self.assertRaises(ValueError):own.probabilities(bad,3)
        with self.assertRaises(ValueError):own.mean_probabilities([],3)

    def test_actual224_window_resize_and_missing_plane(self):
        path=os.environ.get('COATNET224_PUBLIC_SOURCE')
        if not path:self.skipTest('Supply reviewed public source for actual window verification')
        import numpy as np,torch
        from torch.nn import functional as F
        code=own.model_definitions(Path(path).read_text())
        self.assertEqual(own.smoke.digest(code.encode()),own.MODEL_DEFINITIONS_SHA)
        node=next(n for n in ast.parse(code).body if isinstance(n,ast.ClassDef) and n.name=='StudyWindows')
        ns={'np':np,'torch':torch,'F':F,'Dataset':torch.utils.data.Dataset,'_roi_square_box':None}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<pinned-windows>','exec'),ns)
        v=np.arange(44*336*336,dtype=np.uint8).reshape(44,336,336);m=np.ones(44,np.uint8)
        a=own.windows(ns,v,m);b=own.windows(ns,v.copy(),m.copy())
        self.assertEqual(tuple(a.shape),(16,3,224,224));self.assertTrue(torch.equal(a,b))
        # Independently check the first evenly-spaced center's exact resize-before-normalization.
        tri=torch.from_numpy(v[:3].astype(np.float32)/255)
        expected=F.interpolate(tri[None],size=(224,224),mode='bilinear',align_corners=False)[0]
        expected=(expected-torch.tensor([.485,.456,.406]).view(3,1,1))/torch.tensor([.229,.224,.225]).view(3,1,1)
        self.assertTrue(torch.equal(a[0],expected))
        v[:22]=0;m[:22]=0;self.assertTrue(bool(own.windows(ns,v,m).isfinite().all()))
        v[:]=0;m[:]=0
        with self.assertRaises(ValueError):own.windows(ns,v,m)

    def test_reader_retains_all16_stresses(self):
        path=os.environ.get('COATNET224_READER_SOURCE')
        if not path:self.skipTest('Supply reviewed reader source for16 stress cases')
        from knee import corpus44_parity as parity,raptor_input
        code=parity.recipe(Path(path).read_text());self.assertEqual(own.smoke.digest(code.encode()),own.READER_SHA)
        public=parity.reader_module(code);protected,_=raptor_input.transform_exception_handlers(code)
        public._raptor_protected_source=protected;public._raptor_propagate_fatal=raptor_input.propagate_fatal
        exec(compile(protected,'<protected-reader>','exec'),public.__dict__)
        self.assertEqual(raptor_input.stress_inputs(public)['cases'],dict.fromkeys(raptor_input.STRESS_CASES,'PASS'))

    def test_complete_report_requires_every_slot_and_exact_probabilities(self):
        c=contract();c['sources']={}
        report={'status':'PASS','contract_sha256':own.smoke.digest(own.smoke.canonical(c)),
                'qualified_folds':[0],'train_studies':5,'test_studies':3,'settings':own.SETTINGS,
                'labels_read':False,'training_parity':[{'slots':[{'exact_hash_match':True,'mask_match':True} for _ in range(5)],'windows_exact':True} for _ in range(5)],
                'models':[{'fold':0,'checkpoint_sha256':'b'*64,'strict_load':True,'repeat_max_error':0,
                           'train_probability_max_error':0,'missing_plane_finite':True,'cuda_peak_bytes':1}],
                'stress_checks':{'cases':dict.fromkeys(own.raptor_input.STRESS_CASES,'PASS')},
                'environment':{'packages':{'torch':'2.10.0+cu128','timm':'1.0.26'}},
                'elapsed_seconds':1,'visible_csv_sha256':'c'*64}
        own.require_result(report,c)
        bad=copy.deepcopy(report);bad['training_parity'][0]['slots'].pop()
        with self.assertRaises(ValueError):own.require_result(bad,c)
        for key,value in [('repeat_max_error',.01),('train_probability_max_error',.01),('strict_load',False),('missing_plane_finite',False)]:
            bad=copy.deepcopy(report);bad['models'][0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):own.require_result(bad,c)

    def test_builder_immutable_subset_and_foreign_run_rejection(self):
        """Build-only fixture uses pinned protocol inputs and synthetic weight bytes.

        This never loads a model or claims CUDA qualification; runtime strict
        loading is still required before the resulting family can be used.
        """
        root=os.environ.get('COATNET224_EVIDENCE_ROOT')
        if not root:self.skipTest('Supply verified protocol artifacts for immutable builder check')
        from scripts import build_coatnet224_rehearsal as builder
        root=Path(root)
        kwargs={'public_model_source':Path(os.environ['COATNET224_PUBLIC_SOURCE']),
                'public_reader_source':Path(os.environ['COATNET224_READER_SOURCE']),
                'manifest':root/'outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json',
                'smoke_contract':root/'outputs/kaggle/public-training-smoke-20261009/build-contract.json',
                'parity_report':root/'outputs/kaggle/corpus44-parity-v2-20261009-result/corpus44_parity/parity-report.json',
                'parity_review':root/'outputs/analysis/corpus44-parity-20261009/parent-v2-review.json','folds':'0'}
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);results=temp/'results';results.mkdir();kwargs['result']=results
            weight=results/member()['checkpoint_file'];weight.write_bytes(b'synthetic build-only checkpoint')
            receipt=record();receipt['checkpoint_sha256']=own.smoke.file_sha(weight)
            path=results/'run_fold0.json';path.write_text(json.dumps(receipt))
            built=builder.build(temp/'build','test-coatnet224-wiring',**kwargs)
            self.assertTrue(built['build_only']);self.assertFalse(built['complete_family'])
            c=json.loads((temp/'build/build-contract.json').read_text());own.require_contract(c)
            metadata=json.loads((temp/'build/kernel-metadata.json').read_text())
            self.assertFalse(metadata['enable_internet']);self.assertTrue(metadata['is_private'])
            self.assertIn(own.KERNELS[0],metadata['kernel_sources'])
            notebook=json.loads((temp/'build/notebook.ipynb').read_text())
            code=''.join(notebook['cells'][1]['source']);compile(code,'<built-coatnet224>','exec')
            self.assertIn('from knee.coatnet224_rehearsal import run',code)
            with self.assertRaises(ValueError):builder.build(temp/'build','test-coatnet224-wiring',**kwargs)
            receipt['selection']='gold';path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):builder.build(temp/'bad','test-coatnet224-wiring',**kwargs)
            self.assertFalse((temp/'bad').exists())


if __name__=='__main__':unittest.main()
