"""Exact receipt waiver boundaries; ordinary qualification must remain failed."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from scripts import build_public_b0prime_projection_probe as W
from scripts import build_public_b0prime as B
from knee import public_s6_recovery_b0prime as P

ROOT=B.PROJECT/'outputs/kaggle/b0prime-drfull-k62-qual-20261015-r2-result/s6-recovery-entry'
PREP=B.PROJECT/'outputs/kaggle/b0prime-drfull-k62-qual-20261015-r2'
CPU=B.PROJECT/'outputs/analysis/b0prime-20261015/cpu-k62.json'
HAS_NUMPY=importlib.util.find_spec('numpy') is not None
HAS_EVIDENCE=HAS_NUMPY and (ROOT/'receipt.json').exists() and CPU.exists()


class ExactWaiverTests(unittest.TestCase):
    def test_explicit_named_waiver_required_before_evidence(self):
        for value in (None,'projection-8.3h-20261015','wrong'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'EXPLICIT_NAMED_B0PRIME'):
                W.verify_probe_evidence('absent',{},value)

    def test_modified_failed_receipt_is_not_authorized(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'receipt.json').write_text('{"status":"FAIL","changed":true}')
            with self.assertRaisesRegex(ValueError,'EXACT_WAIVED_FAILED_RECEIPT'):
                W.verify_probe_evidence(folder,{},W.WAIVER)


@unittest.skipUnless(HAS_NUMPY,'optional numpy unavailable')
class SyntheticContractTests(unittest.TestCase):
    def setUp(self):
        from test_public_b0prime import B0PrimeSyntheticContract
        fixture=B0PrimeSyntheticContract();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.r,self.binding,self.csv=fixture.r,fixture.binding,fixture.csv
        self.r.update(status='FAIL',failure_type='ValueError')
        self.r['projection']['conservative_seconds']=33645.49299207058
        self.contract=W.waived_contract()

    def test_projection_waiver_does_not_change_ordinary_contract(self):
        before=P.require_rehearsal
        self.contract(self.r,self.binding,self.csv)
        self.assertIs(P.require_rehearsal,before)
        with self.assertRaisesRegex(ValueError,'PASS_EXACT_REHEARSAL_REQUIRED'):
            P.require_rehearsal(self.r,self.binding,self.csv)

    def test_actual_nine_hour_limit_still_fails(self):
        for elapsed in (32400.,float('inf'),float('nan')):
            r=copy.deepcopy(self.r);r['elapsed_seconds']=elapsed
            with self.subTest(elapsed=elapsed),self.assertRaisesRegex(ValueError,'NINE_HOUR_PROJECTION_GATE'):
                self.contract(r,self.binding,self.csv)

    def test_state_input_and_resource_errors_are_never_waived(self):
        def state(r):r['workers']['old'][0]['state_before']={'state':'wrong'}
        def image(r):r['workers']['old'][0]['records'][0]['input']={'changed':True}
        def rss(r):r['phases']['old']['peaks']['host_rss_bytes']=22e9+1
        for mutate in (state,image,rss):
            r=copy.deepcopy(self.r);mutate(r)
            with self.subTest(mutate=mutate.__name__),self.assertRaises(ValueError):self.contract(r,self.binding,self.csv)


@unittest.skipUnless(HAS_EVIDENCE,'optional numpy/exact saved rehearsal evidence unavailable')
class SavedWaiverEvidenceTests(unittest.TestCase):
    def test_real_saved_replay_preserves_failed_receipt_and_normal_rejection(self):
        pr=B.verify_preparation(PREP,CPU);before=B.file_sha(ROOT/'receipt.json')
        checked=W.verify_probe_evidence(ROOT,pr['binding'],W.WAIVER)
        self.assertTrue(checked['all_four_ledgers_recomputed'])
        self.assertEqual(B.file_sha(ROOT/'receipt.json'),before)
        self.assertEqual(json.loads((ROOT/'receipt.json').read_bytes())['status'],'FAIL')
        with self.assertRaisesRegex(ValueError,'PASS_EXACT_REHEARSAL_REQUIRED'):B.verify_result(ROOT,pr['binding'])

    def test_entry_is_image_only_and_retains_runtime_and_source_bytes(self):
        with tempfile.TemporaryDirectory(dir=B.PROJECT/'outputs',prefix='test-b0prime-waiver-') as temp:
            dest=Path(temp)/'entry'
            r=W.build_probe(rehearsal_build=PREP,rehearsal_result=ROOT,cpu_receipt=CPU,
                destination=dest,slug='b0prime-waiver-test',user_waiver=W.WAIVER)
            pack,code=B.unpack_notebook(dest/'notebook.ipynb');cfg=pack['cfg']
            self.assertFalse(r['qualification_pass']);self.assertEqual(r['waiver']['failed_receipt_sha256'],W.FAILED_RECEIPT_SHA)
            self.assertNotIn('qualified_rehearsal',cfg['binding'])
            for key in (*B.PRIVATE,'development_windows'):self.assertNotIn(key,cfg['binding'])
            for key in ('probes','probe_order','old_sentinels','s6_sentinels','development_windows','raptor62_reference'):self.assertNotIn(key,cfg)
            original,_=B.unpack_notebook(PREP/'notebook.ipynb')
            self.assertTrue(all(original['sources'][name]==text for name,text in pack['sources'].items()))
            self.assertFalse(any('rehearsal' in name or 'projection_probe' in name for name in pack['sources']))
            alarms=[n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Call) and ast.unparse(n.func)=='signal.alarm']
            self.assertEqual(len(alarms),1);self.assertEqual(ast.literal_eval(alarms[0].args[0].left)*ast.literal_eval(alarms[0].args[0].right),32400)


if __name__=='__main__':unittest.main()
