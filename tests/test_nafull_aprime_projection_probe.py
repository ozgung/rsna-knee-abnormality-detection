"""A‴ named projection waiver: exact r3 admission, negative boundaries and entry closure."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest import mock

from scripts import build_public_nafull_aprime_projection_probe as W
from knee import public_ws2_entry as WS2

BUILD=W.MAIN/'outputs/kaggle/nafull-aprime-rehearsal-r3-20261016'
ROOT=W.MAIN/'outputs/kaggle/nafull-aprime-rehearsal-r3-20261016-result/nafull-aprime'
CPU=W.MAIN/'outputs/analysis/na-full-export-acceptance-20261016/cpu-k62.json'
HAS_EVIDENCE=(importlib.util.find_spec('numpy') is not None and (ROOT/'receipt.json').is_file()
              and (BUILD/'notebook.ipynb').is_file() and CPU.is_file() and W.ACCEPTANCE.is_file())
ARGS=dict(rehearsal_build=BUILD,rehearsal_result=ROOT,cpu_receipt=CPU)


class NamedWaiverBoundaryTests(unittest.TestCase):
    def test_wrong_or_missing_waiver_name_rejects_before_evidence(self):
        for value in (None,'','projection-8.3h-20261015','b0prime-projection-20261015','a3prime-projection'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'EXPLICIT_NAMED_A3PRIME_WAIVER_REQUIRED'):
                W.admit(rehearsal_build='absent',rehearsal_result='absent',cpu_receipt='absent',
                        user_waiver=value,failed_receipt_sha256=W.FAILED_RECEIPT_SHA)
            with self.subTest(build=value),self.assertRaisesRegex(ValueError,'EXPLICIT_NAMED_A3PRIME_WAIVER_REQUIRED'):
                W.build_probe(rehearsal_build='absent',rehearsal_result='absent',cpu_receipt='absent',destination='absent',
                              slug=W.SLUG,user_waiver=value,failed_receipt_sha256=W.FAILED_RECEIPT_SHA)

    def test_wrong_receipt_sha_argument_or_receipt_bytes_reject(self):
        for value in (None,'0'*64,'8b645d41ded79d946b57e1580586c00ca75849db20dda288d1adad5cc7319dd7'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'EXACT_NAMED_FAILED_RECEIPT_ARGUMENT'):
                W.admit(rehearsal_build='absent',rehearsal_result='absent',cpu_receipt='absent',
                        user_waiver=W.WAIVER,failed_receipt_sha256=value)
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'receipt.json').write_text('{"status":"PASS"}')
            with self.assertRaisesRegex(ValueError,'EXACT_WAIVED_FAILED_RECEIPT'):
                W.admit(rehearsal_build='absent',rehearsal_result=folder,cpu_receipt='absent',
                        user_waiver=W.WAIVER,failed_receipt_sha256=W.FAILED_RECEIPT_SHA)

    def test_only_separately_named_slug(self):
        with self.assertRaisesRegex(ValueError,'SEPARATELY_NAMED_ENTRY_SLUG'):
            W.build_probe(rehearsal_build='absent',rehearsal_result='absent',cpu_receipt='absent',destination='absent',
                          slug='nafull-aprime-rehearsal-r3-20261016',user_waiver=W.WAIVER,failed_receipt_sha256=W.FAILED_RECEIPT_SHA)


@unittest.skipUnless(HAS_EVIDENCE,'optional numpy or exact saved r3 evidence unavailable')
class SavedR3WaiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before=W.tree_hashes(ROOT)
        cls.pack,cls.code,_,cls.checked=W.admit(**ARGS,user_waiver=W.WAIVER,failed_receipt_sha256=W.FAILED_RECEIPT_SHA)
        cls.r=json.loads((ROOT/'receipt.json').read_bytes())

    def contract(self):
        from knee import public_nafull_aprime_runtime as RT
        base=RT.make_base_facade(self.pack['cfg']['nafull_worker_cfg'],rehearsal=True)
        return W.waived_contract(RT.make_ws2_facade(base)),RT.make_ws2_facade(base)

    def rejects(self,message,mutate,root=ROOT):
        r=copy.deepcopy(self.r);mutate(r)
        with self.assertRaisesRegex(ValueError,message):W.replay(root,self.pack,r)

    def test_real_admission_replays_all_non_projection_checks_without_changing_r3(self):
        c=self.checked
        self.assertEqual(c['receipt_sha256'],W.FAILED_RECEIPT_SHA)
        self.assertTrue(c['all_non_projection_checks_pass'] and c['all_six_ledgers_recomputed'])
        self.assertEqual(c['ordinary_failure_reproduced'],'COMBINED_PROJECTION_AND_ACTUAL_LIMIT')
        self.assertGreater(c['projection']['conservative_seconds'],WS2.PROJECTION_GATE_SECONDS)
        self.assertLess(c['elapsed_seconds'],WS2.MAX_SECONDS)
        self.assertEqual(W.tree_hashes(ROOT),self.before)
        self.assertEqual(self.r['status'],'FAIL')

    def test_ordinary_checker_is_unchanged_and_still_rejects(self):
        original=WS2.require_rehearsal;waived,ws2=self.contract()
        self.assertIs(WS2.require_rehearsal,original)
        for status,message in (('FAIL','FRESH_FULL_REHEARSAL_BINDING'),('PASS','COMBINED_PROJECTION_AND_ACTUAL_LIMIT')):
            with self.subTest(status=status),self.assertRaisesRegex(ValueError,message):
                ws2.require_rehearsal({**self.r,'status':status},self.pack['cfg']['ws2_binding'],ROOT/'base/receipt.json',ROOT/'visible-predictions.csv')

    def test_actual_elapsed_at_or_over_nine_hours_rejects(self):
        waived,_=self.contract()
        for elapsed in (32400.,32400.5,float('inf'),float('nan')):
            r=copy.deepcopy(self.r);r['elapsed_seconds']=elapsed
            with self.subTest(elapsed=elapsed):
                with self.assertRaisesRegex(ValueError,'ACTUAL_NINE_HOUR_LIMIT'):W.replay(ROOT,self.pack,r)
                with self.assertRaisesRegex(ValueError,'COMBINED_PROJECTION_AND_ACTUAL_LIMIT'):
                    waived(r,self.pack['cfg']['ws2_binding'],ROOT/'base/receipt.json',ROOT/'visible-predictions.csv')

    def test_missing_measured_projection_inputs_reject(self):
        for key in ('base_setup_seconds','base_merge_seconds','development_windows','final_merge_seconds'):
            with self.subTest(key=key):self.rejects('SAVED_MEASURED_PROJECTION_INPUTS',lambda r:r['projection_inputs'].pop(key))

    def test_waiver_is_projection_only_not_a_relabel(self):
        def within(r):r['projection']['conservative_seconds']=29000.
        def relabel(r):r['status']='PASS'
        def other_type(r):r['failure_type']='RuntimeError'
        self.rejects('EXACT_PROJECTION_ONLY_FAILURE',within)
        self.rejects('FRESH_FULL_REHEARSAL_BINDING',relabel)
        self.rejects('FRESH_FULL_REHEARSAL_BINDING',other_type)

    def test_wrong_model_checkpoint_rejects(self):
        def r224(r):r['crop224_workers'][0]['model']['checkpoint_sha256']='0'*64
        self.rejects('R224_STATE_RELOAD_CROP_RESOURCES',r224)
        pack=copy.deepcopy(self.pack)
        pack['cfg']['nafull_worker_cfg']['public_manifest']['checkpoint_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'EXACT_R3_PAYLOAD'):W.require_packet_identity(pack)
        with mock.patch.dict(W.R3,payload=W.digest(pack)),self.assertRaisesRegex(ValueError,'EXACT_NAFULL_CHECKPOINT_BINDING'):
            W.require_packet_identity(pack)

    def test_wrong_source_or_payload_rejects(self):
        pack=copy.deepcopy(self.pack);pack['sources']['knee/public_ws2_entry.py']+='\n'
        with self.assertRaisesRegex(ValueError,'EXACT_R3_PAYLOAD'):W.replay(ROOT,pack)
        with tempfile.TemporaryDirectory() as folder:
            copy_build=Path(folder)/'build';shutil.copytree(BUILD,copy_build)
            raw=(copy_build/'notebook.ipynb').read_text();(copy_build/'notebook.ipynb').write_text(raw.replace('"metadata": {}','"metadata": {"x": 1}',1))
            with self.assertRaisesRegex(ValueError,'EXACT_R3_NOTEBOOK'):W.r3_packet(copy_build,CPU)

    def test_failed_state_or_genuine_coverage_rejects(self):
        def r224_state(r):r['crop224_workers'][1]['state_after']='0'*64
        def coverage(r):r['completed']=38
        def zero_failed(r):r['zero_failed']=False
        self.rejects('R224_STATE_RELOAD_CROP_RESOURCES',r224_state)
        self.rejects('FRESH39_R224_COVERAGE',coverage)
        self.rejects('FRESH39_R224_COVERAGE',zero_failed)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'r3';shutil.copytree(ROOT,root)
            base=json.loads((root/'base/receipt.json').read_bytes())
            base['workers']['old'][0]['state_before']['state']='0'*64
            (root/'base/receipt.json').write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError,'NAFULL_BOARD_STATE_PROFILE'):W.replay(root,self.pack)
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'r3';shutil.copytree(ROOT,root)
            ledger=root/'board0-crop224/crop224/predictions.npz'
            with np.load(ledger,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            arrays['statuses']=arrays['statuses'].copy();arrays['statuses'][0]='data_local_failure'
            np.savez(ledger,**arrays)
            with self.assertRaises(ValueError):W.replay(root,self.pack)

    def test_other_failure_or_changed_tree_cannot_use_waiver(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'r3';shutil.copytree(ROOT,root)
            (root/'failure-private.json').write_text(json.dumps({'type':'ValueError','detail':'R224_STATE_RELOAD_CROP_RESOURCES'}))
            with self.assertRaisesRegex(ValueError,'ONLY_PROJECTION_FAILURE_MAY_BE_WAIVED'):
                W.admit(**{**ARGS,'rehearsal_result':root},user_waiver=W.WAIVER,failed_receipt_sha256=W.FAILED_RECEIPT_SHA)
            shutil.copyfile(ROOT/'failure-private.json',root/'failure-private.json')
            (root/'base/visible-predictions.csv').write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'EXACT_ACCEPTED_R3_RESULT_TREE'):
                W.admit(**{**ARGS,'rehearsal_result':root},user_waiver=W.WAIVER,failed_receipt_sha256=W.FAILED_RECEIPT_SHA)

    def test_entry_pack_preserves_sources_and_strips_private_fixtures(self):
        entry=W.entry_pack(self.pack,self.checked);cfg=entry['cfg']
        self.assertEqual(entry['sources'],self.pack['sources'])
        self.assertEqual(cfg['mode'],'entry')
        self.assertEqual((cfg['pilot_state_sha256'],cfg['r224_state_pin']),(W.FROZEN_R7_STATE,W.FROZEN_R224_STATE))
        self.assertEqual(cfg['nafull_worker_cfg']['control_state_sha256'],W.PROFILE.CHECKPOINT['state_sha256'])
        waiver=cfg['ws2_binding']['projection_probe_waiver']
        self.assertEqual((waiver['id'],waiver['failed_receipt_sha256'],waiver['qualification_pass']),(W.WAIVER,W.FAILED_RECEIPT_SHA,False))
        self.assertEqual(waiver['authorization']['reported_user_rule'],'Waive if only the estimate fails')
        text=W.canonical(cfg).decode()
        for sid in set(self.pack['cfg']['probe_order'])|set(self.pack['cfg']['binding']['visible_ids']):
            self.assertNotIn(sid,text)
        # Deterministic: an identical second derivation, and only the payload literals change in the bootstrap.
        self.assertEqual(W.digest(W.entry_pack(self.pack,self.checked)),W.digest(entry))
        W.require_bootstrap_preservation(W.B.bootstrap(entry),self.code)
        with self.assertRaisesRegex(ValueError,'EXACT_R3_BOOTSTRAP'):
            W.require_bootstrap_preservation(W.B.bootstrap(entry).replace('signal.alarm(9*3600)','signal.alarm(10*3600)'),self.code)

    def test_entry_mode_base_orchestration_assembles_image_only_worker_config(self):
        """Run the cloned base orchestrator in entry mode with synthetic infrastructure only."""
        import numpy as np
        from knee import public_nafull_aprime_runtime as RT
        cfg=W.entry_pack(self.pack,self.checked)['cfg']
        base=RT.make_base_facade(cfg['nafull_worker_cfg'],rehearsal=False)
        visible=['synthetic-test-'+str(i) for i in range(3)];launched={}
        def launch(worker_cfg,out,arm):
            launched[arm]=copy.deepcopy(worker_cfg)
            return [{'state_before':'s'},{'state_before':'s'}],{'wall_seconds':1.}
        with tempfile.TemporaryDirectory() as folder,mock.patch.dict(base.launch.__globals__,
                test_contract=lambda root,labels,visible_only=None,**k:(['StudyInstanceUID']+W.LABELS,visible,{}),
                stage=lambda *a:{},launch=launch,
                merge=lambda ids,reports,arm:(np.full((len(ids),12),.5),np.zeros(len(ids),dtype=bool)),
                write_csv=lambda *a:'synthetic-csv'),mock.patch.object(base.subprocess,'check_output',
                return_value='0, synthetic-gpu0, Tesla T4\n1, synthetic-gpu1, Tesla T4'):
            receipt=base.run(Path(folder),cfg,Path(folder)/'base',started=time.monotonic())
        self.assertEqual((receipt['status'],receipt['mode'],receipt['projection']),('PASS','entry',None))
        self.assertEqual(set(launched),{'old','s6'})
        for arm,worker in launched.items():
            self.assertEqual(worker['mode'],'entry');self.assertEqual(worker['probes'],{})
            self.assertEqual(worker['staged_ids'],visible)
            self.assertEqual(worker['pilot_state_sha256'],W.FROZEN_R7_STATE)
            self.assertEqual(worker['control_state_sha256'],W.PROFILE.CHECKPOINT['state_sha256'])
            self.assertEqual(worker['public_manifest']['checkpoint_sha256'],W.PROFILE.CHECKPOINT['sha256'])
            self.assertEqual(worker['control_grid_sha256'],worker['binding']['qualified_state']['old_grid'])
            for key in ('old_sentinels','s6_sentinels','raptor62_reference','heliosli_reference','development_windows'):
                self.assertNotIn(key,worker)

    def test_built_entry_packet_is_deterministic_private_offline_and_imports_isolated(self):
        outputs=W.PROJECT/'outputs';outputs.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=outputs,prefix='test-a3prime-waiver-') as inside:
            receipts=[W.build_probe(**ARGS,destination=Path(inside)/name,slug=W.SLUG,user_waiver=W.WAIVER,
                      failed_receipt_sha256=W.FAILED_RECEIPT_SHA) for name in ('one','two')]
            one,two=Path(inside)/'one',Path(inside)/'two'
            for name in ('notebook.ipynb','kernel-metadata.json'):
                self.assertEqual((one/name).read_bytes(),(two/name).read_bytes())
            meta=json.loads((one/'kernel-metadata.json').read_bytes())
            pack,_=W.unpack_notebook(one/'notebook.ipynb')
        self.assertEqual(receipts[0]['payload_sha256'],receipts[1]['payload_sha256'])
        r=receipts[0]
        self.assertEqual((r['status'],r['mode'],r['qualification_pass'],r['competitive_result']),
                         ('BOUND_PROJECTION_WAIVED_PROBE_BUILD','entry',False,False))
        self.assertEqual(r['fresh_isolated_entry_import']['status'],'PASS')
        self.assertEqual((meta['id'],meta['title'],meta['is_private'],meta['enable_internet'],meta['machine_shape']),
                         ('ozgung/'+W.SLUG,W.SLUG,True,False,'NvidiaTeslaT4'))
        self.assertNotIn('timeout',meta)
        self.assertEqual(pack['sources'],self.pack['sources'])
        self.assertFalse(any('projection_probe' in name for name in pack['sources']))
        self.assertEqual(W.tree_hashes(ROOT),self.before)


if __name__=='__main__':unittest.main()
