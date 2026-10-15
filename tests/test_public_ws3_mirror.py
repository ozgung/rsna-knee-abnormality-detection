"""Synthetic native224 mirror invariants; no training or real CUDA qualification."""
import unittest
import types

from knee import public_ws3_checkpoint as M
from knee import public_ws2_checkpoint as BASE


class MirrorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            import torch
        except ImportError:raise unittest.SkipTest('optional array/torch libraries unavailable')
        cls.np,cls.torch=np,torch

    def public(self):
        np,torch=self.np,self.torch
        def mirror(tri,cs,depth):
            self.assertEqual(tri.dtype,torch.uint8)
            boundaries=np.cumsum([0,26,22,18,12,18])
            slots=np.searchsorted(boundaries,np.asarray(cs),side='right')-1
            sag=torch.from_numpy(np.isin(slots,[0,1])).to(tri.device).view(-1,1,1,1)
            return torch.where(sag,tri.flip(1),tri.flip(-1))
        return types.SimpleNamespace(SLOT_SIZES=[26,22,18,12,18],SAGITTAL_SLOTS=(0,1),_mirror_tri=mirror)

    def fixture(self):
        np=self.np
        # Distinct depth and width gradients expose wrong axis, crop order and channel normalization.
        v=((np.arange(96,dtype=np.uint16)[:,None,None]*2+np.arange(384,dtype=np.uint16)[None,None,:])%251+1).astype(np.uint8)
        return np.broadcast_to(v,(96,384,384)).copy(),np.ones(96,np.uint8)

    def test_plain_view_matches_frozen_v2(self):
        v,m=self.fixture()
        self.assertTrue(self.torch.equal(BASE.windows(v,m),M.windows(v,m)))

    def test_anatomical_axis_and_raw_uint8_before_crop_normalization(self):
        np,torch=self.np,self.torch
        from torch.nn import functional as F
        v,m=self.fixture();out=M.windows(v,m,public=self.public(),mirrored=True)
        centers=BASE.eval_centers(m)
        mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
        for center in (1,48,95-1):
            index=centers.index(center)
            tri=torch.from_numpy(v[[center-1,center,center+1]])
            tri=tri.flip(0) if center<48 else tri.flip(-1)
            x=BASE.center_crop(tri.float()/255.)
            expected=(F.interpolate(x[None],size=(224,224),mode='bilinear',align_corners=False)[0]-mean)/std
            self.assertLess(float((out[index]-expected).abs().max()),1e-6)
        plain=M.windows(v,m)
        self.assertGreater(float((out[0]-plain[0].flip(0)).abs().max()),.01)
        index=centers.index(48)
        self.assertGreater(float((out[index]-plain[index].flip(-1)).abs().max()),1e-4)

    def test_probability_mean_and_view_evidence(self):
        np=self.np;a=np.full(12,.1,np.float32);b=np.full(12,.9,np.float32)
        p=M.average_probabilities(a,b)
        self.assertTrue(np.array_equal(p,np.full(12,.5,np.float32)))
        self.assertTrue(M.validate_view_probabilities({'normal':a.tolist(),'mirrored':b.tolist()},p,mirror_enabled=True))
        self.assertTrue(M.validate_view_probabilities({'normal':a.tolist(),'mirrored':None},a,mirror_enabled=False))
        with self.assertRaisesRegex(ValueError,'VIEW_PROBABILITY_MEAN'):
            M.validate_view_probabilities({'normal':a.tolist(),'mirrored':b.tolist()},a,mirror_enabled=True)
        with self.assertRaisesRegex(ValueError,'FINITE_FLOAT32'):
            M.average_probabilities(a,np.full(12,float('nan'),np.float32))

    def test_missing_public_or_wrong_slot_schema_fails(self):
        v,m=self.fixture()
        with self.assertRaisesRegex(ValueError,'PINNED_PUBLIC_MIRROR'):M.windows(v,m,mirrored=True)
        public=self.public();public.SLOT_SIZES=[96]
        with self.assertRaisesRegex(ValueError,'SLOT_SCHEMA'):M.windows(v,m,public=public,mirrored=True)


if __name__=='__main__':unittest.main()
