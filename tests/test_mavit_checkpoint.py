"""Small synthetic checkpoint tests; no external weights, images or model forwards."""
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

from knee import mavit_checkpoint as a

ROOT=Path(__file__).resolve().parents[1]
PARENT=Path(os.environ['KNEE_REFERENCE_PARENT']) if os.environ.get('KNEE_REFERENCE_PARENT') else None
SOURCE=(PARENT/'outputs/analysis/post-fusion-research-20261009/mavit/dataset-code') if PARENT else None


class ArchiveTests(unittest.TestCase):
    def test_zip_metadata_has_no_extraction_and_rejects_unsafe_members(self):
        for extra in ('../private','/absolute','evil\\path','C:private','symlink'):
            data=io.BytesIO()
            with zipfile.ZipFile(data,'w') as z:
                z.writestr('checkpoint/data.pkl',b'not executed')
                if extra=='symlink':
                    info=zipfile.ZipInfo(extra);info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,b'/other')
                else:z.writestr(extra,b'x')
            data.seek(0)
            with self.subTest(extra=extra),self.assertRaises(ValueError):a.archive_audit(data)
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as z:z.writestr('safe/data.pkl',b'never unpickled here');z.writestr('safe/data/0',b'abcd')
        data.seek(0);r=a.archive_audit(data);self.assertFalse(r['extracted']);self.assertEqual(r['members'],2)


@unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch')
class CheckpointTests(unittest.TestCase):
    def setUp(self):
        import torch
        class Small(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.arange(6,dtype=torch.float32).reshape(2,3));self.register_buffer('counter',torch.tensor(7))
            def forward(self,*args):raise AssertionError('No audit forward allowed')
        self.torch=torch;self.factory=Small

    def save(self,d,payload):
        p=Path(d)/'synthetic.pth';self.torch.save(payload,p);return p

    def test_safe_cpu_strict_load_without_any_forward_or_metadata_disclosure(self):
        t=self.torch;report={}
        with tempfile.TemporaryDirectory() as tmp:
            p=self.save(tmp,{'model_state_dict':self.factory().state_dict(),'epoch':8,'private subject key':['secret-study-uid'],
                             'optimizer_state_dict':{'state':{0:{'momentum':t.tensor([1.])}}}})
            original=t.load
            with patch.object(t,'load',wraps=original) as load:
                a.inspect_checkpoint(p,self.factory,report,expected_bytes=p.stat().st_size)
                self.assertEqual(load.call_count,1);self.assertEqual(load.call_args.kwargs,{'map_location':'cpu','weights_only':True})
            self.assertTrue(report['state_compatibility']['strict_load']);self.assertTrue(report['state_compatibility']['loaded_tensors_equal'])
            self.assertEqual(report['all_checkpoint_tensors']['count'],3)
            text=json.dumps(report)
            self.assertNotIn('secret-study-uid',text);self.assertNotIn('private subject key',text)
            self.assertEqual(report['payload_schema']['other_keys'][0]['key_sha256'],a.digest(b'private subject key'))
            self.assertNotIn('epoch": 8',text)

    def test_full_state_key_shape_dtype_and_metadata_tensor_checks(self):
        t=self.torch
        for mode in ('missing','extra','shape','dtype','nonfinite_model','nonfinite_metadata','nonfinite_tensor_key','nonfinite_container_attribute','nonfinite_tensor_attribute'):
            state=dict(self.factory().state_dict());payload={'model_state_dict':state}
            if mode=='missing':state.pop('counter')
            if mode=='extra':state['private/subject']=t.tensor([1.])
            if mode=='shape':state['weight']=t.ones(3,2)
            if mode=='dtype':state['weight']=state['weight'].half()
            if mode=='nonfinite_model':state['weight'][0,0]=float('nan')
            if mode=='nonfinite_metadata':payload['optimizer_state_dict']={'x':t.tensor([float('inf')])}
            if mode=='nonfinite_tensor_key':payload['optimizer_state_dict']={t.tensor(float('inf')):0}
            if mode=='nonfinite_container_attribute':
                from collections import OrderedDict
                payload=OrderedDict(payload);payload.private_metadata=t.tensor(float('inf'))
            if mode=='nonfinite_tensor_attribute':state['weight'].private_metadata=t.tensor(float('inf'))
            report={}
            with tempfile.TemporaryDirectory() as tmp:
                p=self.save(tmp,payload)
                with self.subTest(mode=mode),self.assertRaises(ValueError):a.inspect_checkpoint(p,self.factory,report,expected_bytes=p.stat().st_size)
                self.assertNotIn('private/subject',json.dumps(report))
                if mode.startswith('nonfinite'):self.assertEqual(report['all_checkpoint_tensors']['nonfinite_tensors'],1)
                else:self.assertFalse(report['state_compatibility']['strict_load'])

    def test_unsafe_pickle_rejected_without_allowlist_or_retry(self):
        t=self.torch
        with tempfile.TemporaryDirectory() as tmp:
            marker=Path(tmp)/'MUST_NOT_EXIST'
            class Unsafe:
                def __reduce__(self):return (os.system,(f'touch {marker}',))
            p=self.save(tmp,{'model_state_dict':self.factory().state_dict(),'metadata':Unsafe()});report={}
            with self.assertRaises(pickle.UnpicklingError):a.inspect_checkpoint(p,self.factory,report,expected_bytes=p.stat().st_size)
            self.assertEqual(report['stage'],'safe_deserialization');self.assertFalse(marker.exists())
            self.assertNotIn(str(marker),json.dumps(report))

    def test_size_guard_precedes_safe_load_and_constructor_nested_load_forbidden(self):
        t=self.torch
        with tempfile.TemporaryDirectory() as tmp:
            p=self.save(tmp,{'model_state_dict':self.factory().state_dict()})
            with patch.object(t,'load') as load:
                with self.assertRaises(ValueError):a.inspect_checkpoint(p,self.factory,{},expected_bytes=1)
                load.assert_not_called()
            def bad():t.load(p,weights_only=False);return self.factory()
            report={}
            with self.assertRaisesRegex(RuntimeError,'Nested'):a.inspect_checkpoint(p,bad,report,expected_bytes=p.stat().st_size)
            self.assertEqual(report['stage'],'architecture_construction')


@unittest.skipUnless(SOURCE and SOURCE.is_dir(),'pinned small public source fixtures')
class SourceTests(unittest.TestCase):
    def test_exact_definition_ast_omits_reader_notebook_and_unsafe_pretrain_loader(self):
        code,proof=a.source_contract(SOURCE);nodes={n.name:n for n in ast.parse(code).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for file,names in [('mavit.py',a.BACKBONE_NAMES),('model.py',a.MODEL_NAMES)]:
            original={n.name:n for n in ast.parse((SOURCE/file).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            for name in names:self.assertEqual(ast.dump(nodes[name]),ast.dump(original[name]))
        self.assertEqual(set(nodes),set(a.BACKBONE_NAMES+a.MODEL_NAMES));self.assertNotIn('load_mala_pretrain',nodes)
        self.assertNotIn('torch.load',code);self.assertNotIn('__main__',code);self.assertNotIn('my_fastread',code)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)
            for n in a.SOURCE_HASHES:(d/n).write_bytes((SOURCE/n).read_bytes())
            (d/'model.py').write_text((d/'model.py').read_text()+'\nraise Exception("untrusted")\n')
            with self.assertRaises(ValueError):a.source_contract(d)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch')
    def test_source_constructor_uses_false_pretrain_and_no_download_or_forward(self):
        # Local timm/einops are unavailable. These import-only shims exercise the
        # actual unchanged public constructor/AST, not remote dependency compatibility.
        import torch
        import urllib.request
        code,_=a.source_contract(SOURCE)
        modules={n:types.ModuleType(n) for n in ('timm','timm.layers','timm.models','timm.models.vision_transformer','einops')}
        class DropPath(torch.nn.Module):
            def __init__(self,drop_prob=0.):super().__init__();self.drop_prob=drop_prob
            def forward(self,x):raise AssertionError('Audit must not forward')
        modules['timm.layers'].DropPath=DropPath;modules['timm.layers'].trunc_normal_=torch.nn.init.trunc_normal_
        modules['timm.models.vision_transformer']._cfg=lambda:{}
        modules['einops'].rearrange=lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('No rearrange/forward'))
        with patch.dict(sys.modules,modules),patch.object(torch,'load',side_effect=AssertionError('No pretrain load')),\
             patch.object(torch.hub,'load_state_dict_from_url',side_effect=AssertionError('No download')),\
             patch.object(urllib.request,'urlopen',side_effect=AssertionError('No network')):
            threads=torch.get_num_threads();torch.set_num_threads(2)
            try:model=a.instantiate(code)
            finally:torch.set_num_threads(threads)
        self.assertEqual(tuple(model.cls_w.shape),(12,384));self.assertEqual(tuple(model.slot_emb.weight.shape),(6,384))
        self.assertEqual([len(layer.blocks) for layer in model.backbone.layers],[2,2,6,2])
        self.assertEqual([layer.embed_dim for layer in model.backbone.layers],[64,128,256,512])
        self.assertTrue(all(p.device.type=='cpu' for p in model.parameters()));self.assertFalse(model.training)
        self.assertEqual(len(model.tf.layers),2);self.assertEqual(model.proj[1].in_features,2048)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch')
    def test_aggregate_failure_receipt_never_reads_gold_files_or_leaks_metadata(self):
        import torch
        source,proof=a.source_contract(SOURCE)
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'assets';d.mkdir()
            for n in a.SOURCE_HASHES:(d/n).write_bytes((SOURCE/n).read_bytes())
            (d/'train_golden58.csv').write_text('MUST NOT OPEN')
            model=torch.nn.Linear(2,1);state=model.state_dict();state['weight'][0,0]=float('nan')
            torch.save({'model_state_dict':state,'private subject':['secret private value'],
                        **{f'private key {i}':i for i in range(1500)}},d/a.CHECKPOINT)
            original=a.inspect_checkpoint;opened=[];original_open=Path.open
            def record(path,*args,**kwargs):
                opened.append(path.name)
                if path.suffix=='.csv':raise AssertionError('Gold CSV read')
                return original_open(path,*args,**kwargs)
            def inspect(path,factory,report):return original(path,factory,report,expected_bytes=path.stat().st_size)
            versions={**a.PACKAGES,'einops':'test-only','numpy':'test-only','torchvision':'test-only'}
            out=Path(tmp)/'report'
            with patch.object(Path,'open',record),patch.object(a,'inspect_checkpoint',side_effect=inspect),\
                 patch.object(a.importlib.metadata,'version',side_effect=lambda n:versions[n]),\
                 patch.object(a,'instantiate',return_value=model):
                with self.assertRaisesRegex(RuntimeError,'aggregate'):a.run(d,out,{'version':a.VERSION,'source_contract':proof})
            r=json.loads((out/'checkpoint-audit.json').read_text());text=json.dumps(r)
            self.assertEqual(r['status'],'FAIL');self.assertEqual(r['stage'],'checkpoint_schema')
            self.assertEqual(r['failure_type'],'ValueError');self.assertEqual(r['payload_schema']['other_key_count'],1501)
            self.assertEqual(len(r['payload_schema']['other_keys']),64);self.assertLess(len(text),65536)
            self.assertNotIn('secret private value',text);self.assertNotIn('private subject',text)
            self.assertNotIn('train_golden58.csv',opened);self.assertEqual(r['all_checkpoint_tensors']['nonfinite_tensors'],1)

    def test_build_only_cpu_single_dataset_no_competition_and_isolated_import(self):
        sys.path.insert(0,str(ROOT/'scripts'));import build_mavit_checkpoint_audit as b
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'build';r=b.build(d,'synthetic-mavit-audit',source_dir=SOURCE)
            meta=json.loads((d/'kernel-metadata.json').read_text());c=json.loads((d/'audit-contract.json').read_text())
            self.assertEqual(meta['dataset_sources'],[a.DATASET]);self.assertEqual(meta['competition_sources'],[])
            self.assertFalse(meta['enable_gpu']);self.assertFalse(meta['enable_internet']);self.assertTrue(meta['is_private'])
            self.assertIsNone(c['checkpoint_expected_sha256']);self.assertLess(r['notebook_bytes'],900000)
            code=''.join(json.loads((d/'notebook.ipynb').read_text())['cells'][-1]['source']);compile(code,'<audit-notebook>','exec')
            self.assertNotIn('pip install',code);self.assertNotIn('train_golden58.csv',code)
            own=ast.literal_eval(next(n.value for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets)))
            project=Path(tmp)/'project'
            for name,text in own.items():p=project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
            result=subprocess.run([sys.executable,'-c','import sys;import knee.mavit_checkpoint;assert "torch" not in sys.modules'],
                                  cwd=tmp,env=dict(os.environ,PYTHONPATH=str(project)),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            with self.assertRaises(ValueError):b.build(d,'synthetic-mavit-audit',source_dir=SOURCE)
