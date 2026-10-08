import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from knee.fresh_reader import fingerprint

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('coverage_packaging',SCRIPTS/'build_coverage_training.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


class PackagingTests(unittest.TestCase):
    def report(self,auc=.84,bce=.5):
        return {'status':'PASS','completed_epochs':12,'contract':{'variant':'v1'},
                'prevalence_weak_bce':.6,'history':[{'epoch':i,'weak_bce':bce,
                'thresholded_weak_auc':{'undefined_targets':[],'macro_all_targets_auc':auc}} for i in range(1,13)]}

    def test_reference_gate_uses_bce_selected_checkpoint(self):
        r=self.report();r['history'][1]['weak_bce']=.49;r['history'][1]['thresholded_weak_auc']['macro_all_targets_auc']=.80
        with self.assertRaises(ValueError):builder.reference_gate(r)
        self.assertEqual(builder.reference_gate(self.report())['epoch'],1)
        for r in (self.report(auc=.81),self.report(bce=.59)):
            with self.assertRaises(ValueError):builder.reference_gate(r)

    def test_complete_cache_binding_and_private_cuda_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'src';src.mkdir();receipts=root/'receipts';receipts.mkdir()
            for n in builder.SOURCE_HASHES:(src/n).write_text('# synthetic source\n')
            p={'records':[{'uid':str(i)} for i in range(2200)]};p['fingerprint']=fingerprint(p)
            r={'status':'PASS','stage':'complete','cached_studies':2200,'plan_fingerprint':p['fingerprint'],'gate':{'passed':True}}
            for name,v in [('plan-private.json',p),('cache-report.json',r),('index-private.json',{str(i):{} for i in range(2200)})]:
                (receipts/name).write_text(json.dumps(v))
            with patch.object(builder,'check_sources'):
                c=builder.build(src,root/'good','test-train',receipts,cache_kernel='ozgung/test-cache')
            m=json.loads((root/'good/kernel-metadata.json').read_text());nb=json.loads((root/'good/notebook.ipynb').read_text())
            self.assertTrue(m['is_private']);self.assertTrue(m['enable_gpu']);self.assertFalse(m['competition_sources'])
            self.assertFalse(m['dataset_sources']);self.assertFalse(m['model_sources'])
            self.assertEqual(c['fit_seconds'],10200);self.assertEqual(c['setup_seconds'],300)
            for cell in nb['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),'bundle','exec')
            r['stage']='pilot';(receipts/'cache-report.json').write_text(json.dumps(r))
            with patch.object(builder,'check_sources'),self.assertRaises(ValueError):
                builder.build(src,root/'bad','test-train',receipts,cache_kernel='ozgung/test-cache')


if __name__=='__main__':unittest.main()
