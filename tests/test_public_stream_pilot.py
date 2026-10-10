import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from knee.public_stream_pilot import safe_failure
from knee.reference import canonical_bytes,sha256

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('stream_builder',ROOT/'scripts/build_public_stream_pilot.py')
builder=importlib.util.module_from_spec(spec); spec.loader.exec_module(builder)


class PackagingTests(unittest.TestCase):
    def test_sanitization(self):
        result=safe_failure('stream',ValueError('secret-study-ID report text /private/path'))
        self.assertEqual(result,{'failure_stage':'stream','failure_kind':'INPUT_OR_PARITY_CHECK_FAILED'})
        self.assertNotIn('secret',json.dumps(result))
        self.assertEqual(safe_failure('secret-stage',Exception())['failure_stage'],'unknown')

    def test_private_cpu_exact_package(self):
        manifest={'source_sha256':'a','attribution':'synthetic','source_url':'https://example.org'}
        contract={'source_sha256':'a','manifest_sha256':sha256(canonical_bytes(manifest))}
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); src=root/'source.py';src.write_text('synthetic')
            receipt=root/'receipt.json';receipt.write_text('{}')
            with patch.object(builder,'audit_source',return_value=('','',manifest)),patch.object(builder,'cache_contract',return_value=contract):
                result=builder.build_pilot(src,receipt,root/'build','stream-test')
                with self.assertRaises(ValueError): builder.build_pilot(src,receipt,root/'build','stream-test')
            meta=json.loads((root/'build/kernel-metadata.json').read_text())
            self.assertTrue(meta['is_private']); self.assertFalse(meta['enable_gpu']);self.assertFalse(meta['enable_internet'])
            self.assertEqual(meta['kernel_sources'],[builder.CACHE_SOURCE])
            self.assertEqual(meta['dataset_sources'],[builder.CONVNEXT_DATASET])
            self.assertEqual(result['cache_contract'],contract)
            nb=json.loads((root/'build/notebook.ipynb').read_text()); code=''.join(nb['cells'][1]['source'])
            compile(code,'<test>','exec')
            import ast
            calls=[n.func.id for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
            self.assertNotIn('load_model',calls)
            self.assertNotIn('load_labels',calls)
            for name in builder.SOURCES:
                self.assertEqual(result['sources'][name],sha256((ROOT/name).read_bytes()))

    def test_source_contract_mismatch_before_output(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);src=root/'source';src.write_text('x');receipt=root/'receipt';receipt.write_text('{}')
            with patch.object(builder,'audit_source',return_value=('','',{'source_sha256':'a'})),patch.object(builder,'cache_contract',return_value={'source_sha256':'b'}):
                with self.assertRaises(ValueError): builder.build_pilot(src,receipt,root/'build','test')
            self.assertFalse((root/'build').exists())


class WindowTests(unittest.TestCase):
    def test_same_bytes_require_exact_shape_and_dtype(self):
        try: import numpy as np
        except ImportError: self.skipTest('Optional NumPy unavailable')
        from knee.public_stream_pilot import require_byte_parity
        expected=(np.arange(12,dtype=np.uint8).reshape(3,2,2),np.ones(3,dtype=np.uint8))
        require_byte_parity(expected,expected)
        for actual in ((expected[0].reshape(3,4),expected[1]),
                       (expected[0],expected[1].reshape(3,1)),
                       (expected[0].view(np.int8),expected[1])):
            with self.assertRaises(ValueError): require_byte_parity(actual,expected)

    def test_crop_and_rng(self):
        try:
            import numpy as np
            import torch
        except ImportError: self.skipTest('Optional NumPy/Torch unavailable')
        from types import SimpleNamespace
        from knee.public_stream_pilot import window_pair,sample_indices,rng_snapshot,require_rng
        # Nonuniform uint8 pixels include boundaries; sparse mask retains public
        # min/max-span and cross-slot policy rather than silently filtering centers.
        volume=np.broadcast_to(np.arange(384,dtype=np.uint16)[None,None,:],(96,384,384)).astype(np.uint8)
        mask=np.zeros(96,dtype=np.uint8);mask[[5,90]]=1
        public=SimpleNamespace(_MEAN=torch.tensor([.485,.456,.406]).view(3,1,1),_STD=torch.tensor([.229,.224,.225]).view(3,1,1))
        before=rng_snapshot();a,b=window_pair(public,volume,mask,'synthetic')
        self.assertEqual(a.shape,(1,12,3,320,320));self.assertTrue(torch.equal(a,b));require_rng(before)
        # CPU arithmetic oracle for the unchanged CUDA-only sampling helper;
        # intercept only its final device transfer, never model/decoder calls.
        from knee.public_adaptation_pilot import sampled_windows
        original_to=torch.Tensor.to
        def cpu_transfer(tensor,*args,**kwargs):
            if args==('cuda',) and not kwargs:return tensor
            return original_to(tensor,*args,**kwargs)
        with patch.object(torch.Tensor,'to',cpu_transfer):
            oracle=sampled_windows(public,volume,mask,'synthetic',0)
        self.assertTrue(torch.equal(a,oracle));require_rng(before)
        self.assertTrue(np.array_equal(sample_indices(mask,'synthetic'),sample_indices(mask,'synthetic')))
        with self.assertRaises(ValueError):sample_indices(np.zeros(96),'synthetic')


if __name__=='__main__':unittest.main()
