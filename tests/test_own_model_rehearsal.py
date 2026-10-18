import os,unittest
from pathlib import Path
from unittest.mock import patch
from knee import own_model_rehearsal as own
from knee import corpus44_parity as parity
from knee import raptor_input

class OwnModelTests(unittest.TestCase):
    def test_probability_mean_and_schema(self):
        import numpy as np
        a=np.zeros((3,12),np.float32);b=np.ones_like(a)
        self.assertTrue(np.array_equal(own.mean_probabilities([a,b],3),np.full_like(a,.5)))
        for bad in (a[:2],np.full_like(a,np.nan),np.full_like(a,1.1)):
            with self.assertRaises(ValueError):own.probabilities(bad,3)
        with self.assertRaises(ValueError):own.mean_probabilities([],3)
    def test_run_identity_rejects_wrong_checkpoint_or_smoke(self):
        member={'fold':0,'checkpoint_file':'fold0.pt','checkpoint_sha256':'a'*64,'selected_epochs':[8,9,7]}
        record={'fold':0,'smoke':False,'checkpoint_kind':'swa_topk','checkpoint_file':'fold0.pt',
          'checkpoint_sha256':'a'*64,'selected_epochs':[8,9,7],'train_returncode':0,'epochs_completed':10,
          'train_knee_original_sha256':own.smoke.SOURCE_SHA,'params':{'ARCH':own.ARCH,'RES':'256'},'export':{'k_eval':16}}
        own.verify_run(record,member)
        for change in ({'smoke':True},{'checkpoint_sha256':'b'*64},{'epochs_completed':1}):
            with self.assertRaises(ValueError):own.verify_run(record|change,member)
    def test_public_window_tensor_parity_and_missing_plane(self):
        import numpy as np,torch
        path=os.environ.get('OWN_PUBLIC_SOURCE')
        if not path:self.skipTest('Pinned source supplied for focused verification')
        # Window math does not invoke timm or instantiate an architecture.
        import sys,types
        with patch.dict(sys.modules, {'timm':types.ModuleType('timm')}):
            ns=own.namespace(own.model_definitions(Path(path).read_text()))
        volume=np.arange(44*336*336,dtype=np.uint8).reshape(44,336,336);mask=np.ones(44,np.uint8)
        a=own.windows(ns,volume,mask);b=own.windows(ns,volume.copy(),mask.copy())
        self.assertTrue(torch.equal(a,b));self.assertEqual(tuple(a.shape),(16,3,256,256))
        volume[:22]=0;mask[:22]=0
        self.assertTrue(bool(own.windows(ns,volume,mask).isfinite().all()))
        volume[:]=0;mask[:]=0
        with self.assertRaises(ValueError):own.windows(ns,volume,mask)
    def test_frozen_reader_stress_real_policy(self):
        path=os.environ.get('OWN_READER_SOURCE')
        if not path:self.skipTest('Pinned archived reader supplied for focused verification')
        code=parity.recipe(Path(path).read_text());public=parity.reader_module(code)
        protected,_=raptor_input.transform_exception_handlers(code)
        public._raptor_protected_source=protected;public._raptor_propagate_fatal=raptor_input.propagate_fatal
        exec(compile(protected,'<reader>','exec'),public.__dict__)
        checks=raptor_input.stress_inputs(public)
        self.assertEqual(checks['cases'],dict.fromkeys(raptor_input.STRESS_CASES,'PASS'))
    def test_result_missing_evidence_rejected(self):
        with self.assertRaises(ValueError):own.require_result({}, {'members':[]})
