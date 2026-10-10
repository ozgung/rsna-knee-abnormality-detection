import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from knee import public_adaptation_pilot as pilot
from knee.reference import canonical_bytes,sha256
from tests.test_public_adaptation_pilot import receipt

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stream_gpu_builder',ROOT/'scripts/build_public_stream_gpu_pilot.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def manifest(binding,checkpoint=b'synthetic checkpoint'):
    return {'schema':1,'purpose':'disposable external recovery qualification','status':'PASS',
            'binding':binding,'checkpoint_file':'recovery.pt','checkpoint_bytes':len(checkpoint),
            'checkpoint_sha256':sha256(checkpoint),'updates_before':2,'updates_after':3,
            'consumed_before':0,'consumed_after':3,
            'expected':{k:'a'*64 for k in ('model','optimizer','rng','sampler','gradients','prediction','forward')}}


class RecoveryBindingTests(unittest.TestCase):
    def test_file_bytes_manifest_config_hash_and_consumed_cursor_tamper(self):
        binding={'source':'synthetic','scientific_config':pilot.scientific_config()}
        m=manifest(binding)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'recovery.pt').write_bytes(b'synthetic checkpoint')
            (root/'recovery-manifest.json').write_text(json.dumps(m))
            self.assertEqual(pilot.verify_recovery_artifact(root,m,binding),root/'recovery.pt')
            for key,value in [('consumed_before',1),('updates_before',1),('checkpoint_file','../other'),
                              ('binding',{'source':'wrong'}),('checkpoint_sha256','z'*64)]:
                bad={**m,key:value}
                with self.assertRaises(ValueError):pilot.validate_recovery_manifest(bad,binding)
            changed={**m,'checkpoint_bytes':m['checkpoint_bytes']+1}
            (root/'recovery-manifest.json').write_text(json.dumps(changed))
            with self.assertRaises(ValueError):pilot.verify_recovery_artifact(root,m,binding)
            (root/'recovery-manifest.json').write_text(json.dumps(m))
            (root/'recovery.pt').write_bytes(b'Synthetic checkpoint') # Same size, wrong bytes.
            with self.assertRaises(ValueError):pilot.verify_recovery_artifact(root,m,binding)
        with self.assertRaises(ValueError):pilot.verify_expected_fingerprints({'model':'a'},{'model':'b'})

    def test_completed_decode_records_snapshot_with_new_counter_keys(self):
        import threading
        records=[]
        done=threading.Event()
        def producer():
            for i in range(400):records.append((float(i),float(i)+.25,((f'counter_{i}',1),('decoded',1))))
            done.set()
        thread=threading.Thread(target=producer);thread.start()
        while not done.is_set():
            snapshot=pilot.decode_metrics(records)
            self.assertEqual(snapshot['decode_calls'],snapshot['stream_counters'].get('decoded',0))
            json.dumps(snapshot,allow_nan=False)
        thread.join()
        final=pilot.decode_metrics(records)
        self.assertEqual(final['decode_calls'],400);self.assertEqual(final['decode_seconds'],100.)
        self.assertEqual(final['stream_counters']['decoded'],400)
        self.assertEqual(len(final['stream_counters']),401)

    def test_cpu_stream_receipt_hash_gate(self):
        with self.assertRaises(ValueError):pilot.stream_contract({'status':'PASS'}, {})

    def test_two_stage_same_bundle_private_offline_and_expected_hash(self):
        source_manifest={'source_sha256':'a'*64,'attribution':'synthetic','source_url':'https://example.org'}
        r=receipt();r['manifest_sha256']=sha256(canonical_bytes(source_manifest))
        stream={'receipt_sha256':pilot.STREAM_RECEIPT_SHA,'source_revision':pilot.STREAM_REVISION,'bundle_sha256':pilot.STREAM_BUNDLE_SHA}
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);src=root/'public.py';src.write_text('audit fixture')
            cp=root/'cache.json';cp.write_text(json.dumps(r));sp=root/'stream.json';sp.write_text('{}')
            head=['1'*40]
            def git_output(command,**kwargs):
                if command[1]=='rev-parse':return head[0]
                if command[1]=='show':return (ROOT/command[2].split(':',1)[1]).read_bytes()
                raise AssertionError('Unexpected git command')
            with patch.object(builder,'audit_source',return_value=('','',source_manifest)),patch.object(builder,'stream_contract',return_value=stream),\
                    patch.object(builder.subprocess,'check_output',side_effect=git_output),patch.object(builder.subprocess,'run'):
                first=builder.build_pilot(src,cp,sp,root/'stage1','stream-stage1')
                head[0]='2'*40 # Docs-only packaging HEAD advanced; committed science bytes unchanged.
                m=manifest(first['recovery_binding']);mp=root/'manifest.json';mp.write_text(json.dumps(m))
                reviewed=sha256(canonical_bytes(m))
                second=builder.build_pilot(src,cp,sp,root/'stage2','stream-stage2',recovery_source='ozgung/stream-stage1',
                    recovery_manifest=mp,expected_manifest_sha256=reviewed)
                self.assertEqual(first['bundle_sha256'],second['bundle_sha256'])
                self.assertEqual(first['code_revision'],second['code_revision'])
                self.assertNotEqual(first['packaging_revision'],second['packaging_revision'])
                self.assertEqual(first['recovery_binding'],second['recovery_binding'])
                with self.assertRaises(ValueError):builder.build_pilot(src,cp,sp,root/'bad','stream-bad',
                    recovery_source='ozgung/stream-stage1',recovery_manifest=mp,expected_manifest_sha256='f'*64)
                self.assertFalse((root/'bad').exists())
                # An actual pinned source-blob mismatch rejects despite matching
                # advertised bundle/config. No source mutation on disk.
                def wrong_blob(command,**kwargs):
                    if command[1]=='show':return b'changed scientific bytes'
                    return git_output(command,**kwargs)
                with patch.object(builder.subprocess,'check_output',side_effect=wrong_blob):
                    with self.assertRaises(ValueError):builder.build_pilot(src,cp,sp,root/'bad-blob','stream-bad-blob',
                        recovery_source='ozgung/stream-stage1',recovery_manifest=mp,expected_manifest_sha256=reviewed)
                with patch.object(builder,'SOURCES',builder.SOURCES+('knee/smoke.py',)):
                    with self.assertRaises(ValueError):builder.build_pilot(src,cp,sp,root/'bad-bundle','stream-bad-bundle',
                        recovery_source='ozgung/stream-stage1',recovery_manifest=mp,expected_manifest_sha256=reviewed)
                m['binding']['scientific_config']['head_lr']=.1;mp.write_text(json.dumps(m))
                with self.assertRaises(ValueError):builder.build_pilot(src,cp,sp,root/'bad2','stream-bad2',
                    recovery_source='ozgung/stream-stage1',recovery_manifest=mp,expected_manifest_sha256=sha256(canonical_bytes(m)))
            for name,stage in [('stage1',1),('stage2',2)]:
                meta=json.loads((root/name/'kernel-metadata.json').read_text())
                self.assertTrue(meta['is_private']);self.assertTrue(meta['enable_gpu']);self.assertFalse(meta['enable_internet'])
                self.assertEqual(meta['competition_sources'],['rsna-knee-abnormality-detection'])
                self.assertEqual(meta['kernel_sources'],[pilot.CACHE_SOURCE]+(['ozgung/stream-stage1'] if stage==2 else []))
                self.assertEqual(set(meta['dataset_sources']),{builder.WEIGHT_DATASET,builder.LABEL_DATASET,builder.CONVNEXT_DATASET})
                code=''.join(json.loads((root/name/'notebook.ipynb').read_text())['cells'][1]['source']);compile(code,'<fixture>','exec')
                self.assertIn('streamed=True',code)
                self.assertEqual(json.loads((root/name/'build-receipt.json').read_text())['stage'],stage)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('numpy'),'optional torch/numpy absent')
class ExternalStateProofTests(unittest.TestCase):
    def test_preflight_failure_preserves_stage_before_cleanup_definition(self):
        import torch
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.get_device_name',return_value='synthetic'),\
                    patch('torch.cuda.max_memory_allocated',return_value=0),\
                    patch('importlib.metadata.version',side_effect=lambda p:'2.10.0+cu128' if p=='torch' else '1.0.26'):
                with self.assertRaisesRegex(RuntimeError,'AdamW qualification failed'):
                    pilot.run('','',{'checkpoint_sha256':'a'*64},root,root,root,root,root,root/'out',{},
                              streamed=True,qualified_stream={})
            receipt=json.loads((root/'out/qualification.json').read_text())
            self.assertEqual(receipt['failure_stage'],'stream_binding')
            self.assertEqual(receipt['failure_check'],'ARTIFACT_BINDING_MISMATCH')
            self.assertNotIn('stream_cleanup_failed',receipt)
            self.assertTrue(receipt['disposable_states_deleted'])

    def test_saved_boundary_streamed_replay_all_state_fingerprints(self):
        import torch
        from knee.public_study_stream import StudyStream
        from knee.public_adaptation_pilot import (save_state,restore_state,expected_fingerprints,verify_expected_fingerprints,
            training_mode,optimizer_groups,apply_update,study_mean_backward,masked_loss,checkpoint_forward,state_fingerprint)
        from tests.test_public_adaptation_pilot import MechanicsTests
        fixture=MechanicsTests();fixture.setUp();model=fixture.model
        opt=torch.optim.AdamW(optimizer_groups(model),betas=(.9,.999),eps=1e-8,foreach=False,fused=False)
        windows={i:torch.arange(24,dtype=torch.float32).reshape(1,8,3)/24+i/10 for i in range(6)}
        def update(chosen,number,stream=None):
            training_mode(model);opt.zero_grad(set_to_none=True)
            evidence=[]
            for i in chosen:
                x=next(stream) if stream else windows[i]
                before=state_fingerprint(pilot.rng_state())
                z=checkpoint_forward(model,x)
                loss=masked_loss(z,torch.tensor([[.2,.7]]),torch.ones_like(z))
                evidence.append({'input_sha256':state_fingerprint(x),'rng_sha256':before,'logits':z.detach().clone(),'loss':loss.detach().clone()})
                study_mean_backward(loss,len(chosen))
                if stream:stream.acknowledge()
            grads={n:p.grad.detach().clone() for n,p in model.named_parameters()}
            apply_update(model,opt,number)
            return grads,evidence
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);cp=root/'recovery.pt';post=root/'post.pt'
            update([0,1,2],0);update([3,4,5],1)
            sampler={'epoch':1,'position':0,'order':[2,1,0,5,4,3]};binding={'pin':'synthetic'}
            save_state(cp,model,opt,sampler,2,binding)
            grads,evidence=update([2,1,0],2);sampler['position']=3
            save_state(post,model,opt,sampler,3,binding);state=torch.load(post,weights_only=True)
            model.eval()
            with torch.no_grad():prediction=checkpoint_forward(model,windows[0]).detach()
            expected=expected_fingerprints(state,grads,prediction,evidence)
            restored,count=restore_state(cp,model,opt,binding)
            self.assertEqual(count,2);self.assertEqual(restored['position'],0)
            with StudyStream(restored['order'],windows.__getitem__,contract='synthetic',consumed_cursor=restored['position']) as stream:
                actual_grads,actual_forward=update(restored['order'][:3],count,stream)
                self.assertEqual(stream.consumed_cursor,3);restored['position']=stream.consumed_cursor
            save_state(post,model,opt,restored,3,binding);actual_state=torch.load(post,weights_only=True)
            model.eval()
            with torch.no_grad():actual_prediction=checkpoint_forward(model,windows[0]).detach()
            actual=expected_fingerprints(actual_state,actual_grads,actual_prediction,actual_forward)
            verify_expected_fingerprints(actual,expected)
            actual['gradients']='b'*64
            with self.assertRaises(ValueError):verify_expected_fingerprints(actual,expected)


if __name__=='__main__':unittest.main()
