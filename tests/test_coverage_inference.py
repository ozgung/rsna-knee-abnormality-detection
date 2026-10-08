"""Synthetic image-only source/cache parity and qualification regressions."""
import ast
from collections import Counter
import copy
import csv
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from knee import coverage_inference as ci
from knee.fresh_reader import fingerprint

try:
    import numpy as np
    import cv2
    import torch
except ImportError:
    np = None

SOURCE = Path(os.environ.get('KNEE_REFERENCE_PARENT','/nonexistent')) / 'outputs/research/2026-10-08/convnext-audit'


class LightTests(unittest.TestCase):
    def test_dependency_free_import_and_receipt_rejects_old_or_changed_qualification(self):
        subprocess.run([sys.executable,'-S','-c','import sys; import knee.coverage_inference; assert "torch" not in sys.modules'],
                       cwd=Path(__file__).resolve().parents[1],check=True,capture_output=True)
        receipt = {'status':'PASS','rehearsal':True,'qualification_fingerprint':'qualified','policy':ci.POLICY,
            'train_studies':64,'test_studies':3,'strict_cache_comparisons':64,'input_max_abs_error':0,
            'sample_fingerprint':'a'*64,'empty_studies':0,'resource_errors':0,'prediction_fallbacks':0,
            **dict.fromkeys(('selected_series','missing_slots','unreadable_files','partial_series','short_series',
                           'black_frames','empty_series','failed_series','missing_series_directories'),0),'repeat_max_abs_error':0.,'cache_prediction_max_abs_error':0.,
            'elapsed_seconds':200,'stress_checks':{'status':'PASS','uses_images':False,'uses_labels':False,
                'cases':dict.fromkeys(ci.STRESS_CASES,'PASS')}}
        ci.require_rehearsal(receipt,'qualified')
        for key,value in [('policy','old'),('qualification_fingerprint','changed'),('strict_cache_comparisons',63),
                          ('repeat_max_abs_error',.01),('resource_errors',1),('stress_checks',{})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                ci.require_rehearsal(receipt | {key:value},'qualified')

    def test_actual_test_ids_not_sample_coverage_and_strict_target_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); targets=['t'+str(i) for i in range(12)]
            with (root/'sample_submission.csv').open('w',newline='') as f:
                csv.writer(f).writerows([['StudyInstanceUID',*targets],['visible-example',*([.5]*12)]])
            with (root/'test.csv').open('w',newline='') as f: csv.writer(f).writerows([['StudyInstanceUID'],['hidden-b'],['hidden-a']])
            with (root/'test_series.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(ci.SERIES_COLUMNS)
                for uid in ('hidden-b','hidden-a'):w.writerow([uid,uid+'-series',1,1,'Sagittal'])
            cols,ids,_=ci.read_test_contract(root,rehearsal=False)
            self.assertEqual(ids,['hidden-b','hidden-a'])
            ci.write_probabilities(root/'predictions.csv',cols,ids,[[.1]*12,[.2]*12])
            with (root/'predictions.csv').open() as handle:
                self.assertEqual(list(csv.reader(handle))[1][0],'hidden-b')
            with self.assertRaises(ValueError):ci.write_probabilities(root/'bad.csv',cols,ids,[[float('nan')]*12,[.2]*12])
            with self.assertRaises(FileExistsError):ci.write_probabilities(root/'predictions.csv',cols,ids,[[.1]*12,[.2]*12])


@unittest.skipIf(np is None,'Optional ML/image libraries unavailable')
class InputTests(unittest.TestCase):
    def test_production_stress_and_missing_slot_windows(self):
        self.assertEqual(set(ci.stress_inputs()['cases']),ci.STRESS_CASES)
        v=np.stack([np.full((384,384),i+1,np.uint8) for i in range(8)])
        calls=[]
        def reader(path,plane):calls.append((path,plane));return v,{'n_bad':0}
        audit=Counter(); volumes=ci.prepare_volumes('/unused','synthetic',[[],[],['best','unused'],[],[],[]],reader,
            ['Sagittal','Coronal','Axial'],audit)
        x,mask,pos=ci.eval_arrays(volumes)
        self.assertEqual(calls,[('/unused/synthetic/best','Axial')])
        self.assertEqual(mask.tolist(),[False,False,True,False,False,False])
        self.assertFalse(x[~mask].any());self.assertFalse(pos[~mask].any())
        self.assertEqual(x.shape,(6,12,3,256,256));self.assertEqual(audit['missing_slots'],5)

    def test_schema_ties_counts_match_cache_without_candidate_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            rows=[['study',sid,1,1,'Sagittal'] for sid in ('z-last','a-first')]
            with (root/'test_series.csv').open('w',newline='') as f:
                csv.writer(f).writerows([ci.SERIES_COLUMNS,*rows])
            for row in rows:
                folder=root/'test_series'/row[0]/row[1];folder.mkdir(parents=True)
                for i in range(3):(folder/f'{i}.dcm').touch()
            def table(frame,counts):
                self.assertEqual(frame.SeriesInstanceUID.tolist(),['a-first','z-last'])
                self.assertEqual(list(counts.values()),[3,3]);return {'study':[['a-first','z-last'],[],[],[],[],[]]}
            code=types.SimpleNamespace(PLANES=['Sagittal','Coronal','Axial'],build_study_table=table)
            self.assertEqual(ci.series_table(root,'test',['study'],code,Counter())['study'][0][0],'a-first')

    @unittest.skipUnless((SOURCE/'preprocess.py').is_file(),'Pinned public source fixture unavailable')
    def test_actual_public_preprocessing_cache_parity_and_whole_catch_fatal_errors(self):
        prep=ci.load_source(SOURCE,'preprocess')
        original=ast.parse(inspect.getsource(prep.load_series))
        image=np.arange(32*32,dtype=np.float32).reshape(32,32)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(5):(root/f'{i}.dcm').touch()
            def read(path):
                i=int(Path(path).stem)
                return [(image+i*17,None,None,[.4,.4],float(i))]
            with patch.object(prep,'_read',read):
                reference,info=prep.load_series(str(root),'Axial')
                protected=ci.protect_reader(prep,Counter())
                actual,_=protected(str(root),'Axial')
                self.assertTrue(np.array_equal(reference,actual))
                resized=ci.resize_volume(reference)
                persisted=root/'cached.npz'
                np.savez(persisted,slot0=resized,**{f'slot{i}':np.empty((0,256,256),np.uint8) for i in range(1,6)})
                cached=ci.read_volumes(persisted)
                raw=ci.prepare_volumes(root.parent,root.name,[['.'],[],[],[],[],[]],
                    lambda path,plane:protected(str(root),plane),prep.PLANES,Counter())
                self.assertTrue(all(np.array_equal(a,b) for a,b in zip(ci.eval_arrays(cached),ci.eval_arrays(raw))))
            for failure in (MemoryError,TimeoutError):
                with patch.object(prep,'_read',side_effect=failure('synthetic')):
                    protected=ci.protect_reader(prep,Counter())
                    with self.assertRaises(failure):protected(str(root),'Axial')
            # Prove the scientific AST is unchanged except the one handler prefix/name.
            captured=[]
            builtin_compile=compile
            def capture(tree,*args,**kwargs):
                if isinstance(tree,ast.AST):captured.append(copy.deepcopy(tree))
                return builtin_compile(tree,*args,**kwargs)
            with patch('builtins.compile',capture):ci.protect_reader(prep,Counter())
            transformed=captured[-1]
            handler=next(n for n in ast.walk(transformed) if isinstance(n,ast.ExceptHandler))
            handler.name=None;handler.body.pop(0)
            self.assertEqual(ast.dump(original,include_attributes=False),ast.dump(transformed,include_attributes=False))
            # Asynchronous/append-region errors after _read returns must escape too.
            with patch.object(prep,'_read',read):
                protected=ci.protect_reader(prep,Counter())
                line=next(n.lineno for n in ast.walk(original) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                          and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='extend')
                for failure in (MemoryError,TimeoutError):
                    def trace(frame,event,arg):
                        if frame.f_code is protected.__code__ and event=='line' and frame.f_lineno==line:raise failure('synthetic')
                        return trace
                    try:
                        sys.settrace(trace)
                        with self.assertRaises(failure):protected(str(root),'Axial')
                    finally:sys.settrace(None)


if __name__=='__main__':unittest.main()
