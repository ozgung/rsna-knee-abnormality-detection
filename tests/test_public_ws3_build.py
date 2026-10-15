"""WS3 admission/accounting boundaries, independent of private evidence files."""
import csv
from pathlib import Path
import tempfile
import unittest

from scripts import build_public_ws3_entry as B
from knee import public_ws3_entry as W


class WS3BuildTests(unittest.TestCase):
    def test_real_qualification_is_required_before_entry(self):
        with self.assertRaisesRegex(ValueError,'REAL_V3_QUALIFICATION_REQUIRED'):
            B.build(destination=Path(B.__file__).resolve().parents[1]/'outputs/nonexistent-ws3-test',slug='ws3-test',mode='entry')

    def test_mirror_rehearsal_also_requires_score_admission(self):
        with self.assertRaisesRegex(ValueError,'SCORE_REQUIRED_FOR_MIRROR'):
            B.build(destination=Path(B.__file__).resolve().parents[1]/'outputs/nonexistent-ws3-mirror-test',
                    slug='ws3-mirror-test',mode='rehearsal',mirror=True)

    def admission(self,status,score):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'official.csv'
            with p.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=['ref','status','publicScore']);writer.writeheader()
                writer.writerow({'ref':'57229227','status':status,'publicScore':score})
            return B.verify_mirror_admission(p)

    def test_official_score_required_for_mirror(self):
        with self.assertRaisesRegex(ValueError,'SCORE_REQUIRED_FOR_MIRROR'):B.verify_mirror_admission(None)
        with self.assertRaisesRegex(ValueError,'COMPLETE_REQUIRED'):self.admission('SubmissionStatus.PENDING','0.957')
        with self.assertRaisesRegex(ValueError,'ADMITS_MIRROR'):self.admission('SubmissionStatus.COMPLETE','0.955')
        with self.assertRaisesRegex(ValueError,'ADMITS_MIRROR'):self.admission('SubmissionStatus.COMPLETE','NaN')
        self.assertEqual(self.admission('SubmissionStatus.COMPLETE','0.956')['publicScore'],'0.956')

    def test_diagnostic_control_is_reported_but_not_hidden_projected(self):
        try:import numpy
        except ImportError:raise unittest.SkipTest('optional numpy unavailable')
        base={'projection':{'conservative_seconds':26000.,'population':1300}}
        workers=[{'records':[{'production_seconds':1.} for _ in range(20)]},
                 {'records':[{'production_seconds':1.} for _ in range(19)]}]
        a=W.projection(base,{'wall_seconds':25.},workers,1.,600.)
        b=W.projection(base,{'wall_seconds':25.},workers,1.,1800.)
        self.assertEqual(a['conservative_seconds'],b['conservative_seconds'])
        self.assertEqual(b['serial_control_diagnostic_seconds_excluded'],1800.)
        with self.assertRaisesRegex(ValueError,'SERIAL_CONTROL_OVERHEAD'):
            W.projection(base,{'wall_seconds':25.},workers,1.,float('nan'))

    def test_v3_crop_statistics_must_be_finite_nonnegative(self):
        self.assertTrue(W.crop_parity_ok(2e-5,5e-8))
        for hi,mean in [(-1.,0.),(1e-5,-1e-9),(float('inf'),0.),(1e-5,float('-inf'))]:
            self.assertFalse(W.crop_parity_ok(hi,mean))



class WS3VisibleReferenceTests(unittest.TestCase):
    def test_visible_baseline_reference_cannot_drift(self):
        reference={'synthetic-a':[.5]*12}
        binding={'uid':'StudyInstanceUID','baseline_visible_reference_sha256':W.digest(reference)}
        base={'visible_ids':['synthetic-a']}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'visible.csv'
            with path.open('w') as f:
                writer=csv.writer(f);writer.writerow(['StudyInstanceUID']+W.LABELS)
                writer.writerow(['synthetic-a']+[.5]*12)
            self.assertEqual(W.require_baseline_visible_reference(base,path,reference,binding),0.)
            with path.open('w') as f:
                writer=csv.writer(f);writer.writerow(['StudyInstanceUID']+W.LABELS)
                writer.writerow(['synthetic-a']+[.6]+[.5]*11)
            with self.assertRaisesRegex(ValueError,'BLEND_RANKS_UNCHANGED'):
                W.require_baseline_visible_reference(base,path,reference,binding)


class WS3CoordinatorGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:import numpy
        except ImportError:raise unittest.SkipTest('optional numpy unavailable')

    def test_control_gates_stop_before_mirror_and_plain_can_continue(self):
        import json
        import time
        from unittest import mock
        from scripts import compare_r7_prefetch_p3 as P3
        cases=[(False,True,True,'P3_PREFETCH_MECHANICS_RESOURCES'),
               (True,False,True,'P3_MIRROR_SPEED_ADMISSION'),
               (True,False,False,'SYNTHETIC_STOP_BEFORE_GPU')]
        for mechanics,speed,mirror,error in cases:
            with self.subTest(mechanics=mechanics,speed=speed,mirror=mirror),tempfile.TemporaryDirectory() as folder:
                out=Path(folder)/'run'
                cfg={'mode':'rehearsal','r7_prefetch':True,'ws2_binding':{},'r224_mirror':mirror,
                     'p3_v7_reference':{'synthetic':True},'heliosli_reference':{'submission':{}}}
                def baseline(root,cfg,base_out,**kwargs):
                    base_out.mkdir();(base_out/'assignment.json').write_text(json.dumps({'ids':['synthetic-a','synthetic-b']}))
                    return {'status':'PASS','boards':[{'uuid':'synthetic-0'},{'uuid':'synthetic-1'}],
                            'stage_binding':{},'phases':{'s6':{'wall_seconds':1.}}}
                paired={'mechanics_pass':mechanics,'mirror_speed_pass':speed,'slower_board_paired_wall_ratio':.8 if speed else .95}
                with mock.patch.object(W,'require_baseline_visible_reference',return_value=0.),\
                     mock.patch.object(W.R7,'run',side_effect=baseline),\
                     mock.patch.object(W.R7,'launch',return_value=([],{'wall_seconds':2.})) as control,\
                     mock.patch.object(P3,'compare',return_value=paired),\
                     mock.patch.object(W,'launch_crop224',side_effect=ValueError('SYNTHETIC_STOP_BEFORE_GPU')) as native:
                    with self.assertRaisesRegex(ValueError,error):W.run(Path(folder),cfg,out,started=time.monotonic())
                self.assertEqual(control.call_count,1)
                self.assertFalse(control.call_args.args[0]['r7_prefetch'])
                self.assertEqual(native.call_count,int(mechanics and (speed or not mirror)))


if __name__=='__main__':unittest.main()
