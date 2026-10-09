"""Frozen-source component checks; no real weights, labels, GPU or fusion execution."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from knee import versia
from knee.reference import canonical_bytes,sha256


def spec_dir():
    value=os.environ.get('KNEE_VERSIA_SPEC')
    if value:return Path(value)
    parent=os.environ.get('KNEE_REFERENCE_PARENT')
    return Path(parent)/'outputs/analysis/mirror-fusion-spec-20261009' if parent else None


def fixture():
    contract={'recipe':versia.RECIPE,'policy':versia.policy_contract(),'spec_sha256':versia.SPEC_SHA256,
        'sources':versia.SOURCE_HASHES,'weights':versia.WEIGHTS,'private':True,'internet':False,'codecs':{'synthetic':'a'*64}}
    report={'status':'PASS','contract_sha256':sha256(canonical_bytes(contract)),'source_bundle_sha256':'a'*64,
      'checkpoint_sha256':versia.WEIGHTS,'studies':67,'train_studies':64,'test_studies':3,'prediction_fallbacks':0,
      'input_counts':{'resource_errors':0,'zero_studies':0},'public_tensor_max_error':0,'repeat_max_abs_error':0,
      'execution':{'batch_size':4,'workers':4,'prefetch_factor':2,'pin_memory':True,'multiprocessing':'fork'},
      'batch_sizes':[3]+[4]*16,'stress':{'status':'PASS','cases':dict.fromkeys(versia.STRESS_CASES,'PASS'),'uses_labels':False,'uses_raw_data':False},
      'elapsed_seconds':600,'sampling':{'seed':20261008,'uses_labels':False,'selection':'SHA256(seed:StudyInstanceUID) order','ordered_ids_sha256':'b'*64},
      'strict_coverage':{'PASS':67},'visible_csv_sha256':'c'*64,
      'environment':{'python_major_minor':[3,12],'packages':{'torch':'2.10.0+cu128','timm':'1.0.26'},'cudnn_benchmark':False,'cuda_matmul_allow_tf32':False},
      'codec_hashes':contract['codecs'],'batch4_vs_single_max_abs_error':0,'batch4_vs_single_rank_order_equal':True,
      **dict.fromkeys(('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_parent_rss_bytes','disk_free_bytes','output_disk_bytes'),100)}
    return contract,report


class VersiaContractTests(unittest.TestCase):
    def test_old12_or_modified_policy_weights_runtime_receipt_cannot_qualify(self):
        c,r=fixture();versia.require_rehearsal(c,r,bundle_sha256='a'*64)
        for key,value in [('studies',19),('checkpoint_sha256',{}),('source_bundle_sha256','b'*64),('batch_sizes',[4]*16+[3]),
                          ('repeat_max_abs_error',2e-6),('public_tensor_max_error',1),('stress',{}),('elapsed_seconds',float('nan'))]:
            bad=copy.deepcopy(r);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):versia.require_rehearsal(c,bad,bundle_sha256='a'*64)
        c=copy.deepcopy(c);c['recipe']['eval_windows']=12
        with self.assertRaises(ValueError):versia.require_rehearsal(c,r,bundle_sha256='a'*64)
        c,r=fixture();r['strict_coverage']={'PASS':0,'ineligible_partial_input':67}
        with self.assertRaises(ValueError):versia.require_rehearsal(c,r,bundle_sha256='a'*64)
        r['strict_coverage']={'PASS':66,'ineligible_partial_input':1}
        versia.require_rehearsal(c,r,bundle_sha256='a'*64)

    def test_visible_batch_is_not_mixed_with_training(self):
        batches=versia.rehearsal_batches()
        self.assertEqual(batches[0],[0,1,2]);self.assertEqual([len(x) for x in batches],[3]+[4]*16)
        self.assertEqual([i for batch in batches for i in batch],list(range(67)))


@unittest.skipUnless(spec_dir() and importlib.util.find_spec('torch') and importlib.util.find_spec('cv2'),'pinned Versia fixtures and optional torch/OpenCV')
class VersiaSourceTests(unittest.TestCase):
    def setUp(self):self.sources,self.spec=versia.audit_sources(spec_dir())

    def test_protected_ast_only_prefixes_two_exception_handlers(self):
        for name in ('published-preprocess.py','published-infer.py'):
            original=versia.extract(self.sources)[name];tree=ast.parse(versia.protect(original))
            handlers=[n for n in ast.walk(tree) if isinstance(n,ast.ExceptHandler)]
            self.assertEqual(len(handlers),1)
            node=handlers[0];self.assertEqual(node.name,'_versia_caught')
            self.assertEqual(ast.unparse(node.body[0]),'_versia_propagate(_versia_caught)')
            node.name=None;node.body.pop(0)
            self.assertEqual(ast.dump(tree),ast.dump(ast.parse(original)))
        wrong=dict(self.sources);wrong['published-knee.py']+='\n# changed'
        with self.assertRaises(ValueError):versia.extract(wrong)

    def test_actual_public_slot_copy_stress_and_fatal_catches(self):
        report=versia.stress_inputs(self.sources)
        self.assertEqual(report['cases'],dict.fromkeys(versia.STRESS_CASES,'PASS'))

    def test_actual_pydicom_corruption_parity_strict_eligibility_and_fatal_restore(self):
        import numpy as np
        import pydicom
        from pydicom.dataset import Dataset,FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian
        protected=versia.definitions(self.sources);plain=versia.definitions(self.sources,protected=True)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);series=root/'synthetic/series';series.mkdir(parents=True)
            for i in range(4):
                ds=Dataset();ds.file_meta=FileMetaDataset();ds.file_meta.TransferSyntaxUID=ExplicitVRLittleEndian
                ds.SOPClassUID='1.2.840.10008.5.1.4.1.1.4';ds.SOPInstanceUID=f'2.25.{i+1}'
                ds.Rows=16;ds.Columns=16;ds.SamplesPerPixel=1;ds.BitsAllocated=16;ds.BitsStored=12;ds.HighBit=11;ds.PixelRepresentation=0
                ds.PhotometricInterpretation='MONOCHROME2';ds.ImageOrientationPatient=[1,0,0,0,1,0];ds.ImagePositionPatient=[0,0,i]
                ds.PixelSpacing=[.5,.5];ds.InstanceNumber=i
                ds.PixelData=((np.arange(256).reshape(16,16)+i*5)%2000).astype(np.uint16).tobytes()
                pydicom.dcmwrite(series/f'{i}.dcm',ds,enforce_file_format=True)
            table={'synthetic':[['series'],[],[],['series'],[],[]]}
            a,counts=versia.study_arrays(protected,'synthetic',table,root)
            b,_=versia.study_arrays(plain,'synthetic',table,root,strict_public=True)
            self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)));self.assertTrue(versia.strict_eligible(counts))
            self.assertEqual(counts['slot_copies'],1);self.assertTrue(np.array_equal(a[0][0],a[0][3]))
            (series/'unreadable.dcm').write_bytes(b'synthetic bad input')
            a,counts=versia.study_arrays(protected,'synthetic',table,root)
            b,_=versia.study_arrays(plain,'synthetic',table,root,strict_public=True)
            self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)));self.assertEqual(counts['bad_items'],2)
            self.assertFalse(versia.strict_eligible(counts))
            # Build the actual rehearsal's independent reference objects. A resource
            # error only on the same damaged item must not appear as equal pixels.
            import inspect
            tree=ast.parse(inspect.getsource(versia.rehearse))
            setup=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and
                       any(isinstance(t,ast.Name) and t.id=='plain' for t in n.targets))
            env={'definitions':versia.definitions,'sources':self.sources}
            exec(compile(ast.Module(body=[setup],type_ignores=[]),'<actual-reference-setup>','exec'),env)
            reference=env['plain'];original_read=reference[0]._read
            for kind in (MemoryError,TimeoutError):
                def selective(path):
                    if str(path).endswith('unreadable.dcm'):raise kind('synthetic resource')
                    return original_read(path)
                with patch.object(reference[0],'_read',side_effect=selective):
                    with self.assertRaises(kind) as caught:
                        versia.study_arrays(reference,'synthetic',table,root,strict_public=True)
                    self.assertEqual(caught.exception.versia_counts['resource_errors'],1)
            (series/'unreadable.dcm').unlink()
            ds=pydicom.dcmread(series/'0.dcm');del ds.ImageOrientationPatient;del ds.ImagePositionPatient
            pydicom.dcmwrite(series/'0.dcm',ds,enforce_file_format=True)
            a,counts=versia.study_arrays(protected,'synthetic',table,root)
            b,_=versia.study_arrays(plain,'synthetic',table,root,strict_public=True)
            self.assertTrue(all(np.array_equal(x,y) for x,y in zip(a,b)))
            self.assertEqual(counts['geometry_default_frames'],2);self.assertFalse(versia.strict_eligible(counts))
            prep,_,infer=protected;original=(prep._read,infer.load_series,prep._versia_propagate,infer._versia_propagate)
            for error in (MemoryError('private sentinel'),TimeoutError('private sentinel')):
                with patch.object(prep,'_read',side_effect=error):
                    with self.assertRaises(type(error)) as caught:versia.study_arrays(protected,'synthetic',table,root)
                    self.assertEqual(caught.exception.versia_counts['resource_errors'],1)
                self.assertEqual((prep._read,infer.load_series,prep._versia_propagate,infer._versia_propagate),original)

    def test_actual_four_worker_handoff_keeps67_order_masks_counts_and_split_batches(self):
        # Exercise the actual nested production Dataset and DataLoader definitions
        # with tiny synthetic tensors, without a model/CUDA allocation.
        import numpy as np
        import torch
        import time
        import warnings
        import inspect
        protected=versia.definitions(self.sources);plain=versia.definitions(self.sources,protected=False)
        vol=np.arange(8*8*8,dtype=np.uint16).reshape(8,8,8).astype(np.uint8)
        def loader(path):return vol,{'n_bad':0}
        for prep,_,infer in (protected,plain):
            prep.SIZE=infer.SIZE=8;prep.load_series=infer.load_series=loader
        jobs=[('test',f's{i}') for i in range(3)]+[('train',f's{i}') for i in range(3,67)]
        tables={split:{sid:[['series'],[],[],['series'],[],[]] for sp,sid in jobs if sp==split} for split in ('test','train')}
        tree=ast.parse(inspect.getsource(versia.rehearse))
        klass=next(n for n in ast.walk(tree) if isinstance(n,ast.ClassDef) and n.name=='QualificationDataset')
        assign=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dl' for t in n.targets))
        env={'torch':torch,'np':np,'time':time,'Path':Path,'root':Path('/synthetic'),'jobs':jobs,'tables':tables,
             'protected':protected,'plain':plain,'study_arrays':versia.study_arrays,'strict_eligible':versia.strict_eligible,
             'rehearsal_batches':versia.rehearsal_batches}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            exec(compile(ast.Module(body=[klass,assign],type_ignores=[]),'<actual-versia-worker>','exec'),env)
            sizes=[];copied=0
            for x,mask,pos,events,durations,eligible in env['dl']:
                sizes.append(len(x));copied+=int(events['slot_copies'].sum())
                self.assertTrue(eligible.all());self.assertTrue(torch.equal(x[:,0],x[:,3]))
                self.assertEqual(mask[0].tolist(),[True,False,False,True,False,False])
            del env['dl']
        self.assertEqual(sizes,[3]+[4]*16);self.assertEqual(copied,67)

    def test_exact_three_checkpoint_pins_fail_before_deserialization(self):
        import torch
        from types import ModuleType
        _,model,_=versia.definitions(self.sources)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in versia.WEIGHTS:(root/name).write_bytes(b'synthetic-not-checkpoint')
            with patch.dict(sys.modules,{'timm':ModuleType('timm')}),patch.object(torch,'load') as load:
                with self.assertRaisesRegex(ValueError,'checkpoint changed'):versia.load_models(model,root)
                load.assert_not_called()

    def test_frozen_rank_spec_keeps_mirror_tie_break_then_average_percentiles_and_target_weights(self):
        # Execute only selected pure rank statements from pinned source, not its
        # file/GPU/subprocess/fallback orchestration. This is a spec test, not a fused entry.
        import numpy as np
        import pandas as pd
        mirror=(spec_dir()/'published-mirror.py').read_text();tree=ast.parse(mirror)
        rank=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rankpct')
        env={'np':np};exec(compile(ast.Module(body=[rank],type_ignores=[]),'<pinned-rank>','exec'),env)
        raw=np.array([[.2]*12,[.2]*12,[.8]*12,[.1]*12])
        q=env['rankpct'](raw.clip(0,1)).astype(np.float32)
        self.assertFalse(np.array_equal(q[0],q[1]))
        self.assertTrue(np.array_equal(pd.DataFrame(raw).rank(method='average',pct=True).iloc[0],pd.DataFrame(raw).rank(method='average',pct=True).iloc[1]))
        fusion=ast.parse((spec_dir()/'published-fusion.py').read_text())
        selected=[]
        for node in ast.walk(fusion):
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='TARGET_WEIGHTS':
                selected.append(node)
        constants={};exec(compile(ast.Module(body=selected,type_ignores=[]),'<frozen-target-weights>','exec'),constants)
        self.assertEqual(constants['TARGET_WEIGHTS'],self.spec['convnext_weights'])
        # Frozen formula's R uses average ordinal rank / N, including exact ties.
        self.assertEqual(pd.Series([1.,1.,4.,2.]).rank(method='average',pct=True).tolist(),[.375,.375,1.,.75])
        self.assertEqual(list(self.spec['convnext_weights']),self.spec['target_order'])
        self.assertIn('float32(Q)',self.spec['formula'])

    def test_builder_is_offline_component_only_and_bundle_imports_in_isolation(self):
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('versia_builder',root/'scripts/build_versia_rehearsal.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'build';receipt=module.build_versia(out,'synthetic-versia',spec_dir=spec_dir())
            self.assertLess(receipt['notebook_bytes'],900000)
            meta=json.loads((out/'kernel-metadata.json').read_text())
            self.assertFalse(meta['enable_internet']);self.assertTrue(meta['is_private'])
            self.assertEqual(meta['dataset_sources'],[module.CONVNEXT_DATASET]);self.assertEqual(meta['model_sources'],[]);self.assertEqual(meta['kernel_sources'],[])
            nb=json.loads((out/'notebook.ipynb').read_text());code=''.join(nb['cells'][1]['source']);compile(code,'<versia-build>','exec')
            self.assertLess(code.index('signal.setitimer'),code.index('sources='))
            sources=next(ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='sources' for t in n.targets))
            bundle=Path(tmp)/'bundle'
            for name,value in sources.items():
                p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)
            result=subprocess.run([sys.executable,'-I','-c',f"import sys;sys.path.insert(0,{str(bundle)!r});import knee.versia;print('PASS')"],check=True,capture_output=True,text=True)
            self.assertIn('PASS',result.stdout)
            with self.assertRaises(ValueError):module.build_versia(out,'synthetic-versia',spec_dir=spec_dir())
            bad=Path(tmp)/'bad';shutil.copytree(spec_dir(),bad)
            (bad/'published-infer.py').write_text('tampered')
            with self.assertRaises(ValueError):versia.audit_sources(bad)


if __name__=='__main__':unittest.main()
