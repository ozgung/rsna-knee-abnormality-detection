"""Synthetic R224 mechanics and qualification boundary regressions."""
import math
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from knee import public_ws2_checkpoint as C224
from knee import public_ws2_recovery as REC
from knee import public_ws2_entry as WS2
from knee.public_s6_qualification import LABELS


class WS2PureTests(unittest.TestCase):
    def test_wrong_weight_file_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/C224.CHECKPOINT_NAME
            path.write_bytes(b'invalid')
            with self.assertRaisesRegex(ValueError,'WS2_CHECKPOINT_FILE'):
                C224.verify_checkpoint_file(path)

    def test_stale_or_caller_only_receipt_is_not_entry_qualification(self):
        from scripts.build_public_ws2_entry import verify_fresh_rehearsal
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'receipt.json').write_text('{"status":"PASS"}')
            with self.assertRaisesRegex(ValueError,'FULL_REHEARSAL_FILES'):
                verify_fresh_rehearsal(root,{'version':WS2.VERSION})

    def test_archived_binding_digest_is_required_before_pass_flags(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'base.json').write_text('{"binding":{"substituted":true}}')
            (root/'visible.csv').write_text('StudyInstanceUID\n')
            with self.assertRaisesRegex(ValueError,'EXACT_ARCHIVED_V7_BINDING_DIGEST'):
                WS2.require_rehearsal({'status':'PASS'},
                    {'frozen_r7_binding_sha256':'0'*64},root/'base.json',root/'visible.csv')

    def test_foreign_preparation_binding_and_tampered_file_rejected(self):
        from scripts.build_public_ws2_entry import require_preparation_identity,verify_manifest_files
        from knee.public_s6_qualification import digest,file_sha
        expected={'cfg':{'ws2_binding':{'version':WS2.VERSION,'frozen_r7_binding_sha256':'a'*64}},
                  'sources':{'knee/example.py':'pass\n'}}
        prepared={'status':'PREPARATION_ONLY','mode':'rehearsal','binding':expected['cfg']['ws2_binding'],
            'builder_sha256':file_sha(__file__),'payload_sha256':digest(expected),
            'source_hashes':{'knee/example.py':hashlib.sha256(b'pass\n').hexdigest()}}
        require_preparation_identity(prepared,expected,__file__)
        with self.assertRaisesRegex(ValueError,'EXACT_PREPARATION_IDENTITY'):
            require_preparation_identity({**prepared,'binding':{'version':'foreign'}},expected,__file__)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'payload.xz').write_bytes(b'original')
            manifest={'files':{'payload.xz':{'sha256':file_sha(root/'payload.xz'),'bytes':8}}}
            verify_manifest_files(root,manifest,['payload.xz'])
            (root/'payload.xz').write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'PREPARATION_MANIFEST_FILE_payload.xz'):
                verify_manifest_files(root,manifest,['payload.xz'])


class WS2ArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            import pandas as pd
            import torch
        except ImportError as error:
            raise unittest.SkipTest('optional ML array libraries unavailable: '+str(error))
        cls.np,cls.pd,cls.torch=np,pd,torch

    def arrays(self,n):
        np=self.np
        a=np.tile(np.linspace(.1,.9,n)[:,None],(1,12))
        return a.copy(),a.copy(),a.copy()

    def test_crop307_offset38_and_normalization_after_bilinear(self):
        np,torch=self.np,self.torch
        from torch.nn import functional as F
        volume=np.broadcast_to(np.arange(384,dtype=np.uint8)[None,None,:],(96,384,384)).copy()
        mask=np.ones(96,dtype=np.uint8)
        x=C224.windows(volume,mask,device='cpu')
        self.assertEqual(tuple(x.shape),(94,3,224,224))
        raw=torch.from_numpy(volume[[0,1,2]].astype(np.float32)/255.)
        cropped=C224.center_crop(raw)
        self.assertEqual(tuple(cropped.shape[-2:]),(307,307))
        self.assertEqual(float(cropped[0,0,0]),float(raw[0,38,38]))
        expected=F.interpolate(cropped[None],size=(224,224),mode='bilinear',align_corners=False)[0]
        expected=(expected-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        self.assertLess(float((x[0]-expected).abs().max()),1e-6)

    def test_checkpoint_arch_targets_finiteness_and_strict_native_constructor(self):
        torch=self.torch
        state={'weight':torch.ones(1)}
        ck={'arch':C224.ARCH,'res':224,'lab':LABELS,'model':state}
        self.assertIs(C224.validate_checkpoint(ck,torch),state)
        for mutation,code in [({'res':384},'ARCH_RES'),({'arch':'coatnet_rmlp_2_rw_384'},'ARCH_RES'),
                              ({'lab':LABELS[::-1]},'TARGET_ORDER'),
                              ({'model':{'weight':torch.tensor([float('nan')])}},'FINITE_STATE')]:
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,code):
                C224.validate_checkpoint({**ck,**mutation},torch)
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
        fake_backbone=type('Backbone',(),{'num_features':1})()
        public=type('Public',(),{'LAB':LABELS,'RaptorClassifier':staticmethod(lambda backbone,F_dim:FakeModel())})()
        import sys,types
        fake_timm=types.SimpleNamespace(create_model=mock.Mock(return_value=fake_backbone))
        with mock.patch.object(C224,'verify_checkpoint_file',return_value=C224.CHECKPOINT_SHA256),\
             mock.patch.object(torch,'load',return_value=ck) as loader,mock.patch.dict(sys.modules,{'timm':fake_timm}):
            model,receipt=C224.load_model(public,C224.CHECKPOINT_NAME,'cpu')
        loader.assert_called_once_with(C224.CHECKPOINT_NAME,map_location='cpu',weights_only=True)
        self.assertEqual(float(model.weight.item()),1.)
        self.assertEqual(receipt['res'],224)
        fake_timm.create_model.assert_called_once_with(C224.ARCH,pretrained=False,in_chans=3,
            num_classes=0,global_pool='avg')
        with mock.patch.object(C224,'verify_checkpoint_file',return_value=C224.CHECKPOINT_SHA256),\
             mock.patch.object(torch,'load',return_value={**ck,'model':{'wrong':torch.ones(1)}}),\
             mock.patch.dict(sys.modules,{'timm':fake_timm}):
            with self.assertRaises(RuntimeError):C224.load_model(public,C224.CHECKPOINT_NAME,'cpu')

    def test_latest_raw224_ties_once_and_fixed_three_weights(self):
        np=self.np
        old,r7,crop=self.arrays(4)
        crop[:,0]=[.2,.2,.8,.8]
        z=np.zeros(4,bool)
        result=REC.recover(old,r7,crop,z,z,z)
        self.assertEqual(result['weights'],{'old320':.325,'r7':.5,'crop224':.175})
        self.assertTrue(np.allclose(result['ranks']['crop224'][:,0],[.375,.375,.875,.875]))
        self.assertEqual(result['source224_csv_values'][0,0],result['source224_csv_values'][1,0])
        expected=.325*result['ranks']['old320']+.175*result['ranks']['crop224']+.5*result['ranks']['r7']
        self.assertTrue(np.allclose(result['blend'],expected))
        self.assertNotIn('source224_subrank',result)

    def test_missing_plane_mask_repeats_center_and_cpu_cannot_infer(self):
        np,torch=self.np,self.torch
        volume=np.zeros((96,384,384),np.uint8);volume[20:23]=100
        mask=np.zeros(96,np.uint8);mask[20:23]=1
        self.assertEqual(C224.eval_centers(mask),[21]*94)
        x=C224.windows(volume,mask,device='cpu')
        self.assertTrue(torch.isfinite(x).all())
        with self.assertRaisesRegex(ValueError,'WS2_CUDA_REQUIRED'):
            C224.infer_probs(None,x,'cpu',expected_state='')

    def test_global_means_union_floor_and_fatal_errors(self):
        np=self.np
        old,r7,crop=self.arrays(100)
        masks=[np.zeros(100,bool) for _ in range(3)]
        masks[0][[0,1,2]]=True;masks[1][[1,2]]=True;masks[2][[2]]=True
        for array,mask in zip((old,r7,crop),masks):array[mask]=np.nan
        rec=REC.recover(old,r7,crop,*masks)
        self.assertEqual(rec['cap'],3);self.assertEqual(rec['union_recovered'],3)
        self.assertTrue(np.allclose(rec['filled']['crop224'][2],np.nanmean(crop,axis=0)))
        masks[2][3]=True;crop[3]=np.nan
        with self.assertRaisesRegex(ValueError,'UNION_RECOVERY_CAP'):
            REC.recover(old,r7,crop,*masks)
        masks[2][3]=False;crop[3]=.5
        with self.assertRaisesRegex(ValueError,'FATAL_ERRORS_REQUIRE_ABORT'):
            REC.recover(old,r7,crop,*masks,fatal_errors=('timeout',))

    def test_partial_shard_and_missing_mask_rejected(self):
        np=self.np
        n=4;old,r7,crop=self.arrays(n);mask=np.zeros(n,bool)
        shard={'ordinals':np.arange(2),'old320':old[:2],'r7':r7[:2],
               'crop224':crop[:2],'old320_missing':mask[:2],
               'r7_missing':mask[:2],'crop224_missing':mask[:2]}
        with self.assertRaisesRegex(ValueError,'SHARD_COVERAGE'):REC.merge_shards([shard],n)
        del shard['r7_missing']
        with self.assertRaisesRegex(ValueError,'MISSING_LEDGER_FIELDS'):REC.merge_shards([shard],n)

    def test_projection_uses_measured_third_arm_and_rejects_missing_rate(self):
        base={'projection':{'conservative_seconds':26553.01589682397,'population':1300}}
        workers=[{'records':[{'production_seconds':1.} for _ in range(20)]},
                 {'records':[{'production_seconds':1.} for _ in range(19)]}]
        phase={'wall_seconds':25.}
        p=WS2.projection(base,phase,workers,2.)
        self.assertAlmostEqual(p['conservative_seconds'],26553.01589682397+20*1300/39*1.15+5+2)
        self.assertTrue(p['gate_pass'])
        workers[0]['records'][0]['production_seconds']=float('nan')
        with self.assertRaisesRegex(ValueError,'R224_MEASURED_TIMES'):
            WS2.projection(base,phase,workers,2.)


if __name__=='__main__':unittest.main()
