import ast
import unittest
from pathlib import Path
from knee import corpus44_parity as parity

class Corpus44Tests(unittest.TestCase):
    def test_canonical_dicom_tree_and_missing_paths(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {'study1': [{'StudyInstanceUID':'study1', 'SeriesInstanceUID':'series1',
                'Fluid_Sensitive':'1', 'Anatomical_Plane':'Sagittal'}]}
            # A legacy-looking train_images directory cannot qualify.
            (root/'train_images/study1/series1').mkdir(parents=True)
            with self.assertRaises(ValueError):parity.require_image_tree(root,['study1'],rows)
            (root/'train_series').mkdir()
            with self.assertRaises(ValueError):parity.require_image_tree(root,['study1'],rows)
            (root/'train_series/study1').mkdir()
            with self.assertRaises(ValueError):parity.require_image_tree(root,['study1'],rows)
            (root/'train_series/study1/series1').mkdir()
            self.assertEqual(parity.require_image_tree(root,['study1'],rows),root/'train_series')
            with self.assertRaises(ValueError):parity.require_image_tree(root,['study1'],{'study1':[]})
            broken={'study1':[rows['study1'][0]|{'Fluid_Sensitive':'unknown'}]}
            with self.assertRaises(ValueError):parity.require_image_tree(root,['study1'],broken)
    def test_zero_decode_cannot_qualify(self):
        for counts in ({},{'headers':5},{'decoded_frames':44},{'headers':5,'decoded_frames':0,'decode_failures':44}):
            with self.subTest(counts=counts),self.assertRaises(ValueError):parity.require_decode_evidence(counts)
        parity.require_decode_evidence({'headers':1,'decoded_frames':1})
    def test_exact_and_mask_only_mismatch(self):
        import numpy as np
        a=np.zeros((44,336,336),np.uint8);m=np.zeros(44,np.uint8)
        a[0,0,0]=255;m[0]=1
        rows=parity.compare(a,m,a.copy(),m.copy())
        self.assertTrue(all(r['exact_hash_match'] and r['mask_match'] for r in rows))
        wrong=m.copy();wrong[22]=1
        rows=parity.compare(a,m,a,wrong)
        self.assertEqual([r['mask_disagreements'] for r in rows],[0,0,1,0,0])
    def test_full_uint8_error_no_overflow(self):
        import numpy as np
        a=np.zeros((44,336,336),np.uint8);b=a.copy();b[43,0,0]=255;m=np.zeros(44,np.uint8)
        rows=parity.compare(a,m,b,m)
        self.assertEqual(rows[4]['max_absolute_error_uint8'],255)
        self.assertEqual(rows[4]['different_pixels'],1)
        self.assertFalse(rows[4]['exact_hash_match'])
    def test_invalid_volume_or_mask_rejected(self):
        import numpy as np
        a=np.zeros((44,336,336),np.uint8);m=np.zeros(44,np.uint8)
        with self.assertRaises(ValueError):parity.compare(a.astype(float),m,a,m)
        m[0]=2
        with self.assertRaises(ValueError):parity.compare(a,m,a,m)
    def test_source_pin_rejects_arbitrary_code(self):
        with self.assertRaises(ValueError):parity.recipe('raise RuntimeError()')
    def test_only_span_ast_changed(self):
        import os
        source=os.environ.get('CORPUS44_PUBLIC_SOURCE')
        if not source:self.skipTest('Archived source supplied only for focused real build verification')
        text=Path(source).read_text();code=parity.recipe(text)
        original=ast.parse(text)
        names={'_make_reader','_pick_series_for_slot','build_study'}
        nodes=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in names]
        transformed=ast.parse(code)
        class Undo(ast.NodeTransformer):
            def visit_Constant(self,n):
                if type(n.value) is float and n.value in (.15,.85):n.value={.15:.06,.85:.94}[n.value]
                return n
        Undo().visit(transformed.body[-1])
        self.assertEqual(ast.dump(ast.Module(body=nodes,type_ignores=[])),ast.dump(transformed))
