import copy
import ast
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from knee import raptor_input


class RaptorPolicyTests(unittest.TestCase):
    def test_old_receipt_cannot_qualify_even_with_same_source_and_weights(self):
        with self.assertRaisesRegex(ValueError, 'Fresh expanded'):
            raptor_input.require_rehearsal({}, {'status': 'passed', 'sampling': {'train_count': 16, 'test_count': 3}})

    def test_builder_requires_expanded_python312_offline_codec_only_inputs(self):
        directory = Path(__file__).resolve().parents[1] / 'scripts'
        with patch.object(sys, 'path', [str(directory), *sys.path]):
            spec = importlib.util.spec_from_file_location('raptor_builder', directory/'build_reference_rehearsal.py')
            builder = importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)
        manifest = dict(profile='raptor', attribution='Synthetic author', source_url='https://example.com/source',
                        weight_dataset='synthetic/weights')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root/'source.py'; source.write_text('# synthetic')
            with patch.object(builder, 'audit_source', return_value=('# definitions', '', manifest)):
                for kwargs in (dict(train_count=16), dict(max_seconds=1801)):
                    with self.assertRaises(ValueError):
                        builder.build_reference(source, root/'invalid', 'synthetic', profile='raptor', **kwargs)
                builder.build_reference(source, root/'valid', 'synthetic', profile='raptor')
            metadata = json.loads((root/'valid/kernel-metadata.json').read_text())
            contract = json.loads((root/'valid/reference-contract.json').read_text())
            self.assertEqual(contract['train_count'], 64); self.assertEqual(contract['max_seconds'], 1800)
            self.assertEqual(contract['input_policy'], raptor_input.policy_contract())
            self.assertEqual(metadata['kernel_sources'], [])
            self.assertEqual(metadata['dataset_sources'], ['synthetic/weights', builder.CONVNEXT_DATASET])
            self.assertEqual(metadata['docker_image'], builder.DOCKER_IMAGE)
            self.assertFalse(metadata['enable_internet']); self.assertTrue(metadata['is_private'])
            notebook = json.loads((root/'valid/notebook.ipynb').read_text())
            self.assertEqual(notebook['metadata']['language_info']['version'], '3.12')
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code': compile(''.join(cell['source']), '<raptor-package>', 'exec')


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'optional numpy unavailable')
class RaptorArrayTests(unittest.TestCase):
    def test_partial_slots_are_observable_but_whole_empty_and_mask_mismatch_fail(self):
        import numpy as np
        plan = [({'SeriesInstanceUID':'selected'}, 3), ({'SeriesInstanceUID':'empty'}, 2), (None, 2)]
        volume = np.zeros((7,4,4),np.uint8); volume[1:3] = 7
        mask = np.array([0,1,1,0,0,0,0],np.uint8)
        counts = raptor_input.validate_production_volume(volume, mask, plan, image_size=4)
        self.assertEqual(counts['valid_slices'],2); self.assertEqual(counts['partial_slots'],1)
        self.assertEqual(counts['empty_selected_slots'],1); self.assertEqual(counts['masked_selected_frames'],3)
        self.assertFalse(raptor_input.strict_eligible(counts))
        for wrong_volume,wrong_mask in ((volume*0,mask*0),(volume,mask*0)):
            with self.assertRaises(ValueError):
                raptor_input.validate_production_volume(wrong_volume,wrong_mask,plan,image_size=4)
        volume[6] = 1; mask[6] = 1
        with self.assertRaisesRegex(ValueError, 'Absent'):
            raptor_input.validate_production_volume(volume,mask,plan,image_size=4)


@unittest.skipUnless(os.environ.get('KNEE_RAPTOR_SOURCE') and importlib.util.find_spec('torch')
                     and importlib.util.find_spec('pydicom'), 'requires audited source and optional ML environment')
class RaptorSourceTests(unittest.TestCase):
    def public(self):
        from knee.reference import audit_source, load_definitions
        source=Path(os.environ['KNEE_RAPTOR_SOURCE']).read_text()
        module,decoder,manifest=audit_source(source,profile='raptor',checkpoint_sha256='a'*64)
        return load_definitions(module,decoder,manifest)[0]

    def cv2(self):
        # Local env has no OpenCV. Both source and production use the same resize
        # stub; the remote stress routine uses actual OpenCV on synthetic pixels.
        import numpy as np
        module=ModuleType('cv2'); module.INTER_AREA=3
        class Error(Exception): pass
        module.error=Error
        module.resize=lambda a,size,interpolation: a[np.linspace(0,len(a)-1,size[1]).astype(int)][:,np.linspace(0,a.shape[1]-1,size[0]).astype(int)]
        return module

    def test_pinned_source_all_stress_parities_fatal_reads_and_restoration(self):
        import pydicom
        public=self.public(); original_factory=public._make_reader; original_read=pydicom.dcmread
        with patch.dict(sys.modules, {'cv2': self.cv2()}):
            report=raptor_input.stress_inputs(public)
        self.assertEqual(report['cases'],dict.fromkeys(raptor_input.STRESS_CASES,'PASS'))
        self.assertFalse(report['uses_raw_data']); self.assertFalse(report['uses_labels'])
        self.assertIs(public._make_reader,original_factory); self.assertIs(pydicom.dcmread,original_read)

    def test_transform_changes_only_two_catches_and_geometry_append_failures_are_fatal(self):
        import numpy as np
        import pydicom
        from knee.reference import audit_source
        source=Path(os.environ['KNEE_RAPTOR_SOURCE']).read_text()
        module, _, manifest=audit_source(source,profile='raptor',checkpoint_sha256='a'*64)
        transformed, contract=raptor_input.transform_exception_handlers(module)
        self.assertEqual(contract,manifest['exception_policy']); self.assertEqual(contract['catch_count'],2)
        expected=[node for node in ast.parse(module).body if isinstance(node,ast.FunctionDef) and node.name in {'_make_reader','build_study'}]
        restored=ast.parse(transformed).body
        for function in restored:
            for handler in (node for node in ast.walk(function) if isinstance(node,ast.ExceptHandler)):
                self.assertEqual(handler.name,'_raptor_caught')
                self.assertEqual(handler.body[0].value.func.id,'_raptor_propagate_fatal')
                handler.name=None; handler.body.pop(0)
        self.assertEqual(ast.dump(ast.Module(body=expected,type_ignores=[])),ast.dump(ast.Module(body=restored,type_ignores=[])))
        with self.assertRaisesRegex(ValueError,'exactly one'):
            raptor_input.transform_exception_handlers('def _make_reader(): pass\ndef build_study(): pass')
        public=self.public()
        def read(*args, **kwargs):
            return SimpleNamespace(ImageOrientationPatient=[1,0,0,0,1,0],ImagePositionPatient=[0,0,0],PixelSpacing=[.5,.5])
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules,{'cv2':self.cv2()}), patch.object(pydicom,'dcmread',read):
            path=Path(temporary); (path/'synthetic.dcm').touch()
            for error_type in (MemoryError,TimeoutError):
                counts=Counter()
                with patch.object(public.np,'cross',side_effect=error_type('synthetic arithmetic')):
                    with self.assertRaises(error_type), raptor_input.guarded_reader(public,counts):
                        public._make_reader()[0](str(path))
                self.assertEqual(counts['resource_errors'],1)
                counts=Counter()
                with self.assertRaises(error_type), raptor_input.guarded_reader(public,counts):
                    raptor_input.exercise_pixel_catch(public,error_type('synthetic after read'))
                self.assertEqual(counts['resource_errors'],1)

    def test_strict_decoder_audit_remains_strict_and_healthy_source_windows_unchanged(self):
        import numpy as np
        import pydicom
        import torch
        from knee.reference import reference_reader
        public=self.public()
        class Dataset(pydicom.Dataset):
            @property
            def pixel_array(self): return np.arange(336*336,dtype=np.float32).reshape(336,336)%1000
        missing=False
        def read(path, **kwargs):
            ds=Dataset(); index=int(Path(path).stem)
            ds.ImagePositionPatient=[0,0,index]; ds.PixelSpacing=[.5,.5]; ds.InstanceNumber=index
            if not missing: ds.ImageOrientationPatient=[1,0,0,0,1,0]
            return ds
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {'cv2': self.cv2()}), patch.object(pydicom,'dcmread',read):
            root=Path(temporary); series=root/'study'/'selected'; series.mkdir(parents=True)
            for i in range(3): (series/f'{i}.dcm').touch()
            rows={'study':[dict(SeriesInstanceUID='selected',Anatomical_Plane='Sagittal',Fluid_Sensitive=1)]}
            expected=public.build_study('study',rows,str(root),public._make_reader())
            actual=raptor_input.prepare_study(public,'study',rows,root)
            self.assertTrue(raptor_input.strict_eligible(actual[2]))
            raptor_input.strict_compare(public,'study',rows,root,*actual[:2])
            self.assertTrue(torch.equal(public.eval_windows(*expected,42,384),public.eval_windows(*actual[:2],42,384)))
            missing=True
            self.assertFalse(raptor_input.strict_eligible(raptor_input.prepare_study(public,'study',rows,root)[2]))
            reader,_=reference_reader(public)
            with self.assertRaisesRegex(ValueError,'no geometry fallback'):
                reader[0](str(series))


if __name__ == '__main__': unittest.main()
