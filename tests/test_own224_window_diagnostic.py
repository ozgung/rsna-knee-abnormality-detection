"""Synthetic coverage, parity, masks and endpoint tests; no private fixtures."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from knee import own224_window_diagnostic as d
from scripts.build_own224_window_diagnostic import definitions


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy as np
            import torch
        except ImportError:
            raise unittest.SkipTest('Synthetic ML checks require installed numpy/torch')
        cls.np=np;cls.torch=torch
        # Existing production source is a frozen test oracle, never edited.
        # The class in this fixture contains only the actual source method.
        # Public trainer unavailable in tracked tree: use the historical method
        # verbatim documented by the existing public training smoke contract.
        code='''class Legacy:
    train=False
    def _centers(self,valid,count):
        lo,hi=int(valid.min()),int(valid.max())
        cs=[c for c in range(lo+1,hi) if c-1>=lo and c+1<=hi]
        if not cs:cs=[max(1,min((lo+hi)//2,self._D-2))]
        idx=np.linspace(0,len(cs)-1,count).round().astype(int)
        return [cs[i] for i in idx]
'''
        ns={'np':np};exec(code,ns);cls.legacy=ns['Legacy']()

    def test_frozen_k16_and_k42(self):
        n=self.np;m=n.ones(44,n.uint8)
        centers,flags=d.center_plan(m,self.legacy,16)
        self.assertEqual(centers,[1,4,6,9,12,15,17,20,23,26,28,31,34,37,39,42])
        all_centers,flags=d.center_plan(m,self.legacy,42)
        self.assertEqual(all_centers,list(range(1,43)))
        self.assertEqual(flags['duplicate_centers'],0)
        self.assertEqual(flags['masked_triplets'],0)
        self.assertEqual(flags['boundary_triplets'],8)

    def test_short_mask_retains_duplicates(self):
        n=self.np;m=n.zeros(44,n.uint8);m[12:16]=1
        centers,flags=d.center_plan(m,self.legacy,42)
        self.assertEqual(set(centers),{13,14});self.assertEqual(flags['duplicate_centers'],40)
        self.assertEqual(flags['masked_triplets'],0)

    def test_masked_interior_and_fallback_preserved(self):
        n=self.np;m=n.ones(44,n.uint8);m[20]=0
        centers,flags=d.center_plan(m,self.legacy,42)
        self.assertEqual(centers,list(range(1,43)));self.assertEqual(flags['masked_triplets'],3)
        centers,flags=d.center_plan(n.zeros(44,n.uint8),self.legacy,42)
        self.assertEqual(centers,[1]*42);self.assertEqual(flags['fallback_studies'],1)
        self.assertEqual(flags['masked_triplets'],42)
        for bad in (n.ones(43),n.full(44,2)):
            with self.assertRaises(ValueError):d.center_plan(bad,self.legacy,16)

    def test_population_fails_closed(self):
        self.assertEqual(d.population(['a','b','c'],{'a':0,'b':1,'c':0},{'c'},['a'],count=1),['a'])
        for saved in (['b'],['a','a'],[]):
            with self.assertRaises(ValueError):d.population(['a','b'],{'a':0,'b':1},set(),saved,count=1)

    def test_gold_excluded_from_training_membership(self):
        rows=[{'StudyInstanceUID':s,**{t:('0' if s=='gold' else '') for t in d.TARGETS}} for s in ['train','dev','gold','quarantine']]
        train,gold=d.training_population(['train','dev','gold','quarantine'],{'train':1,'dev':0,'gold':1,'quarantine':1},{'quarantine'},rows)
        self.assertEqual(train,['train']);self.assertEqual(gold,{'gold'})
        with self.assertRaises(ValueError):d.training_population(['missing'],{},set(),rows)

    def test_probability_endpoint_and_parity(self):
        n=self.np;p=n.full((4,12),.5);y=n.tile([0,1,0,1],(12,1)).T;m=n.ones((4,12))
        d.endpoint(['a','b','c','d'],y,m,p,p)
        self.assertEqual(d.parity(p,p)['max_absolute_error'],0)
        with self.assertRaises(ValueError):d.parity(p,p+.01)
        for bad in (n.full_like(p,n.nan),p+1,p[:3]):
            with self.assertRaises(ValueError):d.endpoint(['a','b','c','d'],y,m,p,bad)
        with self.assertRaises(ValueError):d.endpoint(['a']*4,y,m,p,p)
        with self.assertRaises(ValueError):d.endpoint(['a','b','c','d'],y,m+1,p,p)

    def test_corpus_schema_without_allocating_images(self):
        n=self.np
        class Volume:
            shape=(2,44,336,336);dtype=n.uint8
        d.validate_corpus(['a','b'],n.ones((2,44)),Volume())
        with self.assertRaises(ValueError):d.validate_corpus(['a','a'],n.ones((2,44)),Volume())
        with self.assertRaises(ValueError):d.validate_corpus(['a','b'],n.ones((2,43)),Volume())

    def test_checkpoint_metadata_and_finite_state(self):
        t=self.torch
        checkpoint={'arch':d.ARCH,'res':224,'lab':d.TARGETS,'swa_over':[8,9,7],'epoch':[8,9,7],'model':{'x':t.zeros(2)}}
        d.verify_checkpoint(checkpoint)
        checkpoint['swa_over']=[7,8,9]
        with self.assertRaises(ValueError):d.verify_checkpoint(checkpoint)
        checkpoint['swa_over']=[8,9,7];checkpoint['model']['x']=t.tensor(float('nan'))
        with self.assertRaises(ValueError):d.verify_checkpoint(checkpoint)

    def test_extracted_actual_metric_functions_and_groups(self):
        n=self.np
        code=definitions(Path('knee/validation.py').read_text(),('_auc','_report_key','_identity_columns'))
        code+=definitions(Path('knee/public_adaptation_fit.py').read_text(),('metric_report','paired_bootstrap','report_components'),remove_imports=True)
        import unicodedata
        ns={'require':d.require,'fingerprint':d.fingerprint,'unicodedata':unicodedata};exec(code,ns)
        y=n.tile([0,1,0,1],(12,1)).T;m=n.ones((4,12));p=n.tile([.1,.9,.2,.8],(12,1)).T
        report=ns['metric_report'](y,m,p,d.TARGETS)
        self.assertEqual(report['macro_auc'],1);self.assertEqual(report['defined_targets'],12)
        self.assertEqual(report['targets']['ACL']['positive'],2)
        b=ns['paired_bootstrap'](y,m,p,p,['a','b','c','d'],draws=40,seed=42,targets=d.TARGETS)
        self.assertEqual(b['ci95'],[0,0]);self.assertEqual(b['original_defined_targets'],d.TARGETS)
        rows=[{'StudyInstanceUID':s,'Report':r} for s,r in [('a','one'),('b','same'),('c','SAME'),('d','two')]]
        groups,receipt=ns['report_components'](rows,['a'],['b','c','d'])
        self.assertEqual(groups[0],groups[1]);self.assertEqual(receipt['grouping'],'exact normalized-report components')
        with self.assertRaises(ValueError):ns['report_components'](rows,['b'],['c'])

    def test_extracted_truth_threshold_mask_and_schema(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow unavailable in torch test environment")
        import re
        n=self.np
        code=definitions(Path('knee/public_adaptation_fit.py').read_text(),('load_binary_truth',),remove_imports=True)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'truth.parquet'
            data={'StudyInstanceUID':['a','b']}
            norm=lambda t:re.sub('[^a-z0-9]','',t.lower())
            for t in d.TARGETS:
                data['y_'+norm(t)]=[.499999999,float('nan')];data['m_'+norm(t)]=[1,0]
            pq.write_table(pa.table(data),p)
            ns={'require':d.require,'file_sha':d.file_sha,'MASK_SHA':d.file_sha(p),'re':re};exec(code,ns)
            y,m=ns['load_binary_truth'](p,['a','b'],d.TARGETS)
            self.assertTrue((y[0]==1).all());self.assertTrue((m[1]==0).all());self.assertTrue((y[1]==0).all())
            with self.assertRaises(ValueError):ns['load_binary_truth'](p,['missing'],d.TARGETS)
            ns['MASK_SHA']='0'*64
            with self.assertRaises(ValueError):ns['load_binary_truth'](p,['a','b'],d.TARGETS)

    def test_pinned_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_bytes(b'synthetic')
            d.checked(p,d.file_sha(p))
            with self.assertRaises(ValueError):d.checked(p,'0'*64)


if __name__=='__main__':unittest.main()
