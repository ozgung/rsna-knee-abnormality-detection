import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from knee.public_adaptation_pilot import cache_contract,INDEX_SHA,IDS_SHA,CACHE_REVISION,EXCLUSION_SHA,CACHE_BUNDLE_SHA

ROOT=Path(__file__).resolve().parents[1]


def receipt():
    return {'status':'PASS','completed_studies':32,'requested_studies':32,'byte_parity':True,
            'window_parity':True,'repeat_decodes':2,'private_index_sha256':INDEX_SHA,
            'ordered_id_set_sha256':IDS_SHA,'source_revision':CACHE_REVISION,'exclusion_sha256':EXCLUSION_SHA,
            'source_bundle_sha256':CACHE_BUNDLE_SHA,
            'counters':{'headers':0,'fast_headers':4269,'fast_bad_headers':0,'header_parse_failures':0,
                        'geometry_defaults':0,'spacing_defaults':0,'strict_ineligible_headers':0,
                        'pixel_failures':0,'decoded_unique_slices':2897,'resource_errors':0,'invalid_pixels':0,
                        'valid_slices':2990,'missing_slots':6,'selected_slots':154,'masked_selected_frames':0,
                        'partial_slots':0,'empty_selected_slots':0,'zero_studies':0,'sequence_preference_substitutions':0},
            'source_sha256':'a'*64,'manifest_sha256':'b'*64,'series_metadata_sha256':'c'*64}


class PackagingTests(unittest.TestCase):
    def test_cache_binding_rejects_incomplete_and_different_index(self):
        good=receipt(); self.assertEqual(cache_contract(good)['index_sha256'],INDEX_SHA)
        for key,value in [('status','FAIL'),('repeat_decodes',1),('private_index_sha256','d'*64),('completed_studies',31)]:
            bad={**good,key:value}
            with self.assertRaises(ValueError): cache_contract(bad)

    def test_private_cuda_exact_attachments_and_immutable(self):
        spec=importlib.util.spec_from_file_location('adapt_builder',ROOT/'scripts/build_public_adaptation_pilot.py')
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        from knee.reference import canonical_bytes,sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'public.py'; source.write_text('mock audit only')
            manifest={'source_sha256':'a'*64,'attribution':'fixture','source_url':'https://example.com'}
            r=receipt(); r['manifest_sha256']=sha256(canonical_bytes(manifest))
            cache=root/'receipt.json'; cache.write_text(json.dumps(r))
            with patch.object(mod,'audit_source',return_value=('','',manifest)):
                mod.build_pilot(source,cache,root/'build','test-adaptation')
                with self.assertRaises(ValueError): mod.build_pilot(source,cache,root/'build','test-adaptation')
            meta=json.loads((root/'build/kernel-metadata.json').read_text())
            self.assertTrue(meta['is_private']); self.assertTrue(meta['enable_gpu']); self.assertFalse(meta['enable_internet'])
            self.assertEqual(meta['kernel_sources'],['ozgung/knee-public-rich-cache-pilot-20261010'])
            self.assertIn('nartaa/rsna-knee-hpo-assets',meta['dataset_sources'])
            code=''.join(json.loads((root/'build/notebook.ipynb').read_text())['cells'][1]['source'])
            self.assertIn("CUBLAS_WORKSPACE_CONFIG=':4096:8'",code)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),'optional torch/numpy absent')
class MechanicsTests(unittest.TestCase):
    def setUp(self):
        import torch
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4))
                self.norm=torch.nn.LayerNorm(4)
                self.att=torch.nn.Sequential(torch.nn.Dropout(.2),torch.nn.Linear(4,2))
            def head(self,features): return self.att(self.norm(features.mean(1)))
        torch.manual_seed(42); self.model=Toy()

    def test_masks_tail_denominator_and_unknown_invariance(self):
        import torch
        from knee.public_adaptation_pilot import masked_loss
        x=torch.tensor([[.4,-.6]],requires_grad=True); mask=torch.tensor([[1.,0.]])
        a=masked_loss(x,torch.tensor([[.25,0.]]),mask)
        b=masked_loss(x,torch.tensor([[.25,1.]]),mask)
        self.assertEqual(float(a.detach()),float(b.detach()))
        with self.assertRaises(ValueError): masked_loss(x,torch.zeros_like(x),mask*0)
        def gradient(denominator):
            p=torch.tensor([[.4,-.6]],requires_grad=True)
            for value in (.2,.5,.8): (masked_loss(p,torch.tensor([[value,0.]]),mask)/denominator).backward()
            return p.grad
        self.assertTrue(torch.allclose(gradient(3),gradient(1)/3,atol=1e-7))
        self.assertFalse(torch.allclose(gradient(3),gradient(16)))
        self.assertEqual(float(gradient(3)[0,1]),0.)

    def test_bn_affine_groups_and_group_omission(self):
        import torch
        from knee.public_adaptation_pilot import optimizer_groups,training_mode,bn_state,require_bn
        groups=optimizer_groups(self.model)
        self.assertEqual(len(groups),4)
        self.assertEqual(len({id(p) for g in groups for p in g['params']}),len(list(self.model.parameters())))
        for g in groups:
            self.assertEqual(g['base_lr'],1e-6 if g['name'].startswith('encoder') else 1e-5)
        original=bn_state(self.model); training_mode(self.model)
        self.assertFalse(self.model.backbone[1].training); self.assertTrue(self.model.backbone[1].weight.requires_grad)
        self.model.backbone(torch.randn(4,3)); require_bn(self.model,original)
        self.model.backbone[1].running_mean.add_(1)
        with self.assertRaises(ValueError): require_bn(self.model,original)
        self.model.att[1].weight.requires_grad=False
        with self.assertRaises(ValueError): optimizer_groups(self.model)

    def test_adamw_full_saved_resume_rng_sampler_and_counter(self):
        import torch
        from knee.public_adaptation_pilot import optimizer_groups,training_mode,checkpoint_forward,masked_loss,apply_update,save_state,restore_state,assert_state_close
        optimizer=torch.optim.AdamW(optimizer_groups(self.model),betas=(.9,.999),eps=1e-8,foreach=False)
        windows=torch.arange(24,dtype=torch.float32).reshape(1,8,3)/24
        targets=torch.tensor([[.3,.7]]); mask=torch.ones_like(targets)
        def update(number):
            training_mode(self.model); optimizer.zero_grad(set_to_none=True)
            masked_loss(checkpoint_forward(self.model,windows),targets,mask).backward()
            apply_update(self.model,optimizer,number)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'resume.pt'; update(0)
            sampler={'epoch':0,'position':16,'order':list(range(32))}
            save_state(path,self.model,optimizer,sampler,1,{'pin':'fixture'})
            update(1)
            expected={k:v.detach().clone() for k,v in self.model.state_dict().items()}
            expected_rng=torch.get_rng_state().clone()
            restored,count=restore_state(path,self.model,optimizer,{'pin':'fixture'})
            self.assertEqual(restored,sampler); self.assertEqual(count,1)
            update(count)
            self.assertEqual(assert_state_close(expected,self.model.state_dict(),atol=0,rtol=0),0.)
            self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
            self.assertTrue(all(int(s['step'])==2 for s in optimizer.state.values()))
            self.assertTrue(all(g['lr']==g['base_lr']*.1 for g in optimizer.param_groups))
            with self.assertRaises(ValueError): restore_state(path,self.model,optimizer,{'pin':'wrong'})

    def test_no_cuda_fallback(self):
        from knee.public_adaptation_pilot import run
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaises(RuntimeError): run(*([None]*10))


@unittest.skipUnless(importlib.util.find_spec('pyarrow') and importlib.util.find_spec('numpy'),'optional pyarrow/numpy absent')
class LabelSchemaTests(unittest.TestCase):
    def test_explicit_masks_align_selected_ids_and_unknowns(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import numpy as np
        from knee import public_adaptation_pilot as pilot
        targets=['T'+str(i) for i in range(12)]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); labels=root/'labels.parquet'; masks=root/'masks.parquet'
            data={'StudyInstanceUID':['b','a'],'report':['never read','never read']}
            data.update({t:[.7,.2] for t in targets}); data['T0']=[float('nan'),.2]
            pq.write_table(pa.table(data),labels)
            md={'StudyInstanceUID':['a','b']}; md.update({'m_'+t:[1.,1.] for t in targets}); md['m_T0']=[1.,0.]
            pq.write_table(pa.table(md),masks)
            with patch.object(pilot,'LABEL_SHA',pilot.file_sha(labels)),patch.object(pilot,'MASK_SHA',pilot.file_sha(masks)):
                ys,ms=pilot.load_labels(labels,masks,['a','b'],targets)
                self.assertEqual(list(ys),['a','b']); self.assertEqual(list(ms),['a','b'])
                self.assertAlmostEqual(float(ys['a'][1]),.2); self.assertAlmostEqual(float(ys['b'][1]),.7)
                self.assertEqual(float(ys['b'][0]),0.); self.assertEqual(float(ms['b'][0]),0.)
                with self.assertRaises(ValueError): pilot.load_labels(labels,masks,['missing'],targets)
            md['m_T1']=[1.,.5]; pq.write_table(pa.table(md),masks)
            with patch.object(pilot,'LABEL_SHA',pilot.file_sha(labels)),patch.object(pilot,'MASK_SHA',pilot.file_sha(masks)):
                with self.assertRaises(ValueError): pilot.load_labels(labels,masks,['a','b'],targets)
