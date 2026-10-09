import ast
import unittest
from pathlib import Path
from knee import corpus44_parity as parity

class Corpus44Tests(unittest.TestCase):
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
