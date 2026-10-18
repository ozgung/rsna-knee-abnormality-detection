"""Projection-only exception stays separate from normal qualification and inference."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import build_public_ws2_projection_probe as P


class ProjectionProbeBoundaryTests(unittest.TestCase):
    def test_cli_help_is_builder_interface(self):
        result=subprocess.run([sys.executable,P.__file__,'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--user-waiver',result.stdout)
        self.assertNotIn('--worker',result.stdout)

    def test_explicit_waiver_and_exact_failed_receipt_required(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'receipt.json').write_text('{"status":"FAIL"}')
            for value in (None,'projection','other'):
                with self.subTest(value=value),self.assertRaisesRegex(ValueError,'NAMED_PROJECTION_WAIVER'):
                    P.verify_probe_evidence(root,{},value)
            with self.assertRaisesRegex(ValueError,'EXACT_WAIVED_FAILED_RECEIPT'):
                P.verify_probe_evidence(root,{},P.WAIVER)

    def test_another_failure_cannot_use_this_waiver(self):
        from knee.public_s6_qualification import file_sha
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'receipt.json').write_text('{}')
            (root/'failure-private.json').write_text('{"type":"ValueError","detail":"RELOAD_FAILED"}')
            with mock.patch.object(P,'FAILED_RECEIPT_SHA',file_sha(root/'receipt.json')):
                with self.assertRaisesRegex(ValueError,'ONLY_PROJECTION_FAILURE'):
                    P.verify_probe_evidence(root,{},P.WAIVER)


class ProjectionProbeMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: import numpy
        except ImportError: raise unittest.SkipTest('optional numpy unavailable')

    def fixture(self):
        ids=['synthetic-'+str(i) for i in range(39)]
        input_pin={'volume_sha256':'a'*64,'mask_sha256':'b'*64}
        self.binding={'frozen_r7_binding_sha256':P.digest({'synthetic':True}),
            'frozen_base_environment':{'torch':'synthetic','packages':{}},
            'frozen_development_windows_sha256':P.digest([2,4]),'uid':'StudyInstanceUID'}
        phase={'wall_seconds':2.,'peaks':{'host_rss_bytes':10,'output_disk_bytes':5},'minimum_disk_free_bytes':100}
        workers=[]
        for board in range(2):
            assigned=ids[board::2]
            sentinels=[{'id':sid,'input':copy.deepcopy(input_pin)|{'crop224_cpu_cuda_max_abs':2e-5,'crop224_cpu_cuda_mean_abs':5e-8},
                'strict_reload':True,'probabilities':[.5]*12,'reloaded_probabilities':[.5]*12} for sid in ids[:4]]
            workers.append({'status':'PASS','slot':'native224','completed':len(assigned),'assigned':assigned,
                'environment':{'physical_uuid':'synthetic-gpu-'+str(board),'torch':'synthetic','packages':{}},
                'state_before':'state','state_after':'state',
                'model':{'checkpoint_sha256':P.C224.CHECKPOINT_SHA256,'arch':P.C224.ARCH,'res':224},
                'zero_failed':True,'fp32_recoveries':0,'peak_cuda_bytes':10,'peak_rss_bytes':10,
                'ledger_sha256':'synthetic-ledger','sentinels':sentinels})
        self.base={'binding':{'synthetic':True},'boards':[{'uuid':'synthetic-gpu-'+str(i)} for i in range(2)],
            'workers':{'old':[{'records':[{'id':sid,'input':input_pin} for sid in ids],'peak_cuda_bytes':10}],
                's6':[{'peak_cuda_bytes':10}]},'sentinel_ids':{'s6':ids[:4]},
            'projection':{'conservative_seconds':28000.},'phases':{'old':phase,'s6':phase},
            'setup_seconds':1.,'visible_ids':ids[:3]}
        projection={'conservative_seconds':31000.}
        r={'status':'FAIL','failure_type':'ValueError','version':P.VERSION,'mode':'rehearsal',
            'binding':self.binding,'completed':39,'completed_visible':3,'zero_failed':True,
            'recovery':{'union_recovered':0},'crop224_workers':workers,'projection':projection,
            'projection_inputs':{'base_setup_seconds':1.,'base_merge_seconds':0.,'development_windows':[2,4],'final_merge_seconds':0.},
            'crop224_phase':phase,'elapsed_seconds':100.,
            'resources':{'peak_host_rss_bytes':10,'peak_output_disk_bytes':5,'minimum_disk_free_bytes':100,
                'peak_cuda_bytes_by_arm':{'old':[10],'s6':[10],'crop224':[10,10]}}}
        return r

    def exercise(self,r,normal=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=root/'base.json';csv=root/'visible.csv'
            base.write_text(json.dumps(self.base));r['base_receipt_sha256']=P.file_sha(base)
            csv.write_text(','.join(['StudyInstanceUID']+P.LABELS)+'\n'+''.join(
                sid+','+','.join(['0.5']*12)+'\n' for sid in self.base['visible_ids']))
            r['csv_sha256']=P.file_sha(csv)
            with mock.patch.object(P.R7,'require_rehearsal',return_value=True),\
                 mock.patch.object(P.R7,'projection',return_value=self.base['projection']),\
                 mock.patch.object(P,'projection',return_value=r['projection']):
                method=P.WS2.require_rehearsal if normal else P.require_projection_probe
                return method(r,self.binding,base,csv)

    def test_projection_failure_only_has_separate_probe_acceptance(self):
        r=self.fixture()
        with self.assertRaisesRegex(ValueError,'FRESH_FULL_REHEARSAL_BINDING'):self.exercise(copy.deepcopy(r),normal=True)
        self.assertTrue(self.exercise(r))

    def test_actual_nine_hour_limit_is_not_waived(self):
        r=self.fixture();r['elapsed_seconds']=P.MAX_SECONDS
        with self.assertRaisesRegex(ValueError,'COMBINED_PROJECTION_AND_ACTUAL_LIMIT'):self.exercise(r)

    def test_reload_checkpoint_crop_and_resource_failures_remain_fatal(self):
        cases=[('checkpoint',lambda r:r['crop224_workers'][0]['model'].update(checkpoint_sha256='wrong'),'R224_STATE_RELOAD_CROP_RESOURCES'),
            ('crop',lambda r:r['crop224_workers'][0]['sentinels'][0]['input'].update(crop224_cpu_cuda_max_abs=.1),'R224_STATE_RELOAD_CROP_RESOURCES'),
            ('reload',lambda r:r['crop224_workers'][0]['sentinels'][0]['reloaded_probabilities'].__setitem__(0,.6),'R224_SENTINEL_PROBABILITY_RELOAD'),
            ('resource',lambda r:r['resources'].update(peak_host_rss_bytes=999),'REPLAY_COMPLETE_RESOURCES'),
            ('wrong status',lambda r:r.update(status='PASS'),'FRESH_FULL_REHEARSAL_BINDING')]
        for label,change,expected in cases:
            with self.subTest(label=label):
                r=self.fixture();change(r)
                with self.assertRaisesRegex(ValueError,expected):self.exercise(r)


if __name__=='__main__':unittest.main()
