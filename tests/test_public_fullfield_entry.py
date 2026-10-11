"""Synthetic schema, evidence and extracted scientific-closure regressions."""
import ast
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from knee.public_fullfield_entry import (VERSION, METHOD, atomic_json, require_rehearsal, safe_id, test_contract, write_csv)
from knee.reference import sha256
from scripts.build_public_fullfield_entry import exact_closure, verify_build


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.columns=['StudyInstanceUID', *['t'+str(i) for i in range(12)]]
        self.ids=['a','b','c']; self.values=[[.1+i*.01 for i in range(12)] for _ in self.ids]
        self.write('sample_submission.csv',[self.columns,*[[sid,*([0]*12)] for sid in self.ids]])
        self.write('test.csv',[['StudyInstanceUID'],*[ [sid] for sid in reversed(self.ids)]])
        self.write('test_series.csv',[['StudyInstanceUID','SeriesInstanceUID','Fluid_Sensitive','Fat_Suppression','Anatomical_Plane'],*[ [sid,'series'+sid,'1','0','Sagittal'] for sid in self.ids]])
    def write(self,name,rows):
        with (self.root/name).open('w',newline='') as h:csv.writer(h).writerows(rows)
    def evidence(self):
        path=self.root/'visible.csv';digest=write_csv(path,self.columns,self.ids,self.values)
        binding={'version':VERSION,'method':METHOD,'primitives_sha256':'p'*64,'source_bundle_sha256':'a'*64,'environment':{'torch':'qualified'},'columns':self.columns,'visible_ids':self.ids,'qualified_grid':{'state_sha256':'s'*64}}
        report={'status':'PASS','version':VERSION,'mode':'rehearsal','binding':binding,'source_bundle_sha256':'a'*64,'environment':binding['environment'],'completed_visible':3,'completed_probes':32,'prediction_fallbacks':0,'final_state_and_all_buffers_immutable':True,'strict_reload_gate':True,'constructor_oracle_gate':True,'same_image_hash_gate':True,'max_prediction_errors':dict(qualified=0,repeat=0,reload=0),'elapsed_seconds':1,'csv_sha256':digest,'state_grid_gate':binding['qualified_grid'],'candidate_state':{'state':'s'*64,'buffers':'b'*64},'decode_counters':dict(resource_errors=0,invalid_pixels=0,pixel_failures=0,fast_bad_headers=0,header_parse_failures=0),'cuda_boards':[{'name':'board'}]}
        for k in ('setup_seconds','peak_host_rss_bytes','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','disk_free_bytes','output_disk_bytes'):report[k]=1
        return report,path,binding
    def test_test_csv_order_authoritative(self):
        _,ids,_=test_contract(self.root,self.columns[1:],visible=True);self.assertEqual(ids,list(reversed(self.ids)))
    def test_hidden_replacement_preserves_header_and_actual_ids(self):
        self.write('test.csv',[['StudyInstanceUID'],['hidden']]);self.write('test_series.csv',[['StudyInstanceUID','SeriesInstanceUID','Fluid_Sensitive','Fat_Suppression','Anatomical_Plane'],['hidden','series','1','0','Axial']])
        _,ids,_=test_contract(self.root,self.columns[1:]);self.assertEqual(ids,['hidden'])
        with self.assertRaises(ValueError):test_contract(self.root,self.columns[1:],visible=True)
    def test_missing_test_metadata_coverage_fails(self):
        self.write('test.csv',[['StudyInstanceUID'],['a']])
        with self.assertRaises(ValueError):test_contract(self.root,self.columns[1:])
    def test_duplicate_series_and_malformed_rows_fail(self):
        for row in (['a','series','2','0','Axial'],['a','../series','1','0','Axial'],['a','series','1','0','Axial','extra']):
            self.write('test_series.csv',[['StudyInstanceUID','SeriesInstanceUID','Fluid_Sensitive','Fat_Suppression','Anatomical_Plane'],row])
            with self.assertRaises(ValueError):test_contract(self.root,self.columns[1:])
    def test_safe_ids(self):
        for value in ('','..','a/b','a\\b',None):self.assertFalse(safe_id(value))
    def test_csv_rejects_invalid_and_preserves_existing(self):
        for values in ([[float('nan')]*12]*3,[[1.1]*12]*3,[[.1]*11]*3):
            with self.assertRaises(ValueError):write_csv(self.root/'bad.csv',self.columns,self.ids,values)
            self.assertFalse((self.root/'bad.csv').exists())
        path=self.root/'good.csv';write_csv(path,self.columns,self.ids,self.values);before=path.read_bytes()
        with self.assertRaises(ValueError):write_csv(path,self.columns,self.ids,self.values)
        self.assertEqual(path.read_bytes(),before)
    def test_passing_gate_and_tamper_rejections(self):
        report,path,binding=self.evidence();require_rehearsal(report,path,binding)
        mutations={'status':'FAIL','completed_probes':31,'strict_reload_gate':False,'same_image_hash_gate':False,'elapsed_seconds':32401,'environment':{},'candidate_state':{},'state_grid_gate':{},'prediction_fallbacks':1,'source_bundle_sha256':'wrong','max_prediction_errors':{'qualified':0,'repeat':1e-5,'reload':0},'cuda_boards':[],'peak_host_rss_bytes':0,'decode_counters':{}}
        for key,value in mutations.items():
            with self.subTest(key=key):
                changed=copy.deepcopy(report);changed[key]=value
                with self.assertRaises(ValueError):require_rehearsal(changed,path,binding)
        path.write_text(path.read_text()+'garbage')
        with self.assertRaises(ValueError):require_rehearsal(report,path,binding)
    def test_atomic_failure_preserves_last_receipt(self):
        path=self.root/'receipt.json';atomic_json(path,{'status':'RUNNING'});before=path.read_bytes()
        with patch('knee.public_fullfield_entry.os.replace',side_effect=OSError('synthetic')):
            with self.assertRaises(OSError):atomic_json(path,{'status':'FAIL'})
        self.assertEqual(path.read_bytes(),before);self.assertFalse(path.with_name('receipt.json.tmp').exists())
    def test_exact_closure_preserves_functions_excludes_metric_label_helpers(self):
        frozen={name:Path(name).read_text() for name in ('knee/reference.py','knee/mirror_input.py','knee/raptor_input.py','knee/public_fullfield.py','knee/public_adaptation_fit.py','knee/public_adaptation_pilot.py')}
        sources,pins=exact_closure(frozen)
        for path,source in sources.items():
            compile(source,path,'exec')
            original=frozen.get(path,'')
            for node in ast.parse(source).body:
                if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
                    old=next(n for n in ast.parse(original).body if isinstance(n,type(node)) and n.name==node.name)
                    self.assertEqual(ast.get_source_segment(source,node),ast.get_source_segment(original,old))
        self.assertIn('knee/public_fullfield.py:make_candidate',pins)
        self.assertNotIn('load_binary_truth',''.join(sources.values()))
        self.assertNotIn('train.csv',''.join(sources.values()))
    def test_same_function_has_no_training_metadata_read(self):
        source=Path('knee/public_fullfield_entry.py').read_text();tree=ast.parse(source)
        inference=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='infer_study')
        code=ast.get_source_segment(source,inference)
        self.assertNotIn('train_series',code);self.assertNotIn('train.csv',source)
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        # Training metadata read is dominated by explicit rehearsal branch.
        branches=[n for n in ast.walk(run) if isinstance(n,ast.If) and ast.unparse(n.test)=="mode == 'rehearsal'"]
        self.assertTrue(any('train_records = read_series' in ast.unparse(n) for n in branches))

    def test_nonpersistent_buffer_mutation_rejected(self):
        try: import torch
        except ImportError: self.skipTest('Torch unavailable')
        from knee.public_fullfield import snapshot, require_immutable
        model=torch.nn.Linear(2,2).eval()
        model.register_buffer('constructor_grid',torch.ones(2),persistent=False)
        before=snapshot(model);require_immutable(model,before)
        model.constructor_grid.add_(1)
        with self.assertRaisesRegex(ValueError,'PARAMETER_OR_BUFFER_MUTATION'):require_immutable(model,before)

    def test_strict_reload_rejects_missing_keys(self):
        try: import torch
        except ImportError: self.skipTest('Torch unavailable')
        model=torch.nn.Linear(2,2).eval()
        with self.assertRaises(RuntimeError):model.load_state_dict({'weight':model.weight},strict=True)

    def test_entry_evidence_required_and_attachment_allowlist(self):
        from knee.reference import WEIGHT_DATASET
        from knee.reference_blend import CONVNEXT_DATASET
        files={'notebook.ipynb':b'{}','kernel-metadata.json':json.dumps({'dataset_sources':[WEIGHT_DATASET,CONVNEXT_DATASET]}).encode()}
        for name,raw in files.items():(self.root/name).write_bytes(raw)
        receipt={'files':{name:sha256(raw) for name,raw in files.items()}}
        (self.root/'build-receipt.json').write_text(json.dumps(receipt));verify_build(self.root)
        (self.root/'notebook.ipynb').write_text('tampered')
        with self.assertRaises(ValueError):verify_build(self.root)
        (self.root/'notebook.ipynb').write_bytes(b'{}')
        raw=json.dumps({'dataset_sources':[WEIGHT_DATASET,CONVNEXT_DATASET,'private/labels']}).encode()
        (self.root/'kernel-metadata.json').write_bytes(raw);receipt['files']['kernel-metadata.json']=sha256(raw)
        (self.root/'build-receipt.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,'checkpoint/codec'):verify_build(self.root)

    def test_full_runtime_exercises_35_and_pure_entry_never_reads_train(self):
        try: import torch
        except ImportError: self.skipTest('Torch unavailable')
        from contextlib import ExitStack
        import importlib.metadata
        import os
        import sys
        import time
        from types import SimpleNamespace
        from knee.public_fullfield_entry import run
        from knee.public_fullfield import snapshot
        import numpy as np
        model=torch.nn.Linear(2,2).eval();model.register_buffer('grid',torch.ones(2),persistent=False)
        before=snapshot(model);grid={'state_sha256':before['state']}
        sample=['probe'+str(i) for i in range(32)]
        self.write('train_series.csv',[['StudyInstanceUID','SeriesInstanceUID','Fluid_Sensitive','Fat_Suppression','Anatomical_Plane'],*[[sid,'s'+sid,'1','0','Axial'] for sid in sample]])
        environment={'python_major_minor':list(sys.version_info[:2]),'torch':torch.__version__,'timm':'qualified','numpy':np.__version__,'cuda':torch.version.cuda}
        binding={'version':VERSION,'method':METHOD,'primitives_sha256':'p'*64,'source_bundle_sha256':'a'*64,'environment':environment,'qualified_grid':grid,'columns':self.columns,'visible_ids':list(reversed(self.ids)),'train_series_sha256':sha256((self.root/'train_series.csv').read_bytes())}
        hashes={'volume_sha256':'v','mask_sha256':'m','windows':{'normal':'n','mirrored':'r'}}
        contract={'mode':'rehearsal','binding':binding,'probes':[{'id':sid,'hashes':hashes,'probabilities':self.values[0]} for sid in sample]}
        events=[]
        def infer(public,decoder,candidate,sid,records,image_root,*,audit):
            events.append((sid,Path(image_root).name,audit))
            counters=dict(resource_errors=0,invalid_pixels=0,pixel_failures=0,fast_bad_headers=0,header_parse_failures=0,masked_selected_frames=3)
            return self.values[0],counters,hashes,dict(decode=.1,windows=.1,forward=.1,total=.3)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ,{'KNEE_SOURCE_SHA256':'a'*64,'KNEE_PRIMITIVES_SHA256':'p'*64}))
            stack.enter_context(patch('knee.reference.load_definitions',return_value=(SimpleNamespace(LAB=self.columns[1:]),object())))
            stack.enter_context(patch('knee.reference.load_model',return_value=object()))
            stack.enter_context(patch('knee.public_fullfield.make_candidate',return_value=(model,grid)))
            stack.enter_context(patch('knee.public_fullfield_entry.infer_study',side_effect=infer))
            stack.enter_context(patch('knee.public_fullfield_entry.importlib.metadata.version',return_value='qualified'))
            for name,value in {'is_available':True,'get_device_name':'synthetic','get_device_properties':SimpleNamespace(total_memory=1),'device_count':1,'max_memory_allocated':1,'max_memory_reserved':1,'reset_peak_memory_stats':None,'empty_cache':None,'manual_seed_all':None}.items():
                stack.enter_context(patch('torch.cuda.'+name,return_value=value))
            stack.enter_context(patch('torch.manual_seed'));stack.enter_context(patch('torch.use_deterministic_algorithms'))
            result=run(self.root,None,'','',{},contract,self.root/'rehearsal',started=time.monotonic())
            self.assertEqual(result['status'],'PASS');self.assertIsNone(result['failure_stage']);self.assertEqual(result['completed_probes'],32)
            self.assertEqual(len(events),105);self.assertEqual(sum(image=='train_series' for _,image,_ in events),96)
            require_rehearsal(result,self.root/'rehearsal/visible-predictions.csv',binding)
            (self.root/'train_series.csv').unlink();events.clear()
            entry={'mode':'entry','binding':{k:v for k,v in binding.items() if k not in ('visible_ids','train_series_sha256')},'expected_state':before}
            result=run(self.root,None,'','',{},entry,self.root/'entry',started=time.monotonic())
            self.assertEqual(result['status'],'PASS');self.assertEqual(result['completed_probes'],0)
            self.assertEqual(len(events),3);self.assertTrue(all(image=='test_series' and not audit for _,image,audit in events))
            events.clear()
            with patch('knee.public_fullfield_entry.infer_study',side_effect=ValueError('synthetic failure')):
                with self.assertRaises(RuntimeError):run(self.root,None,'','',{},entry,self.root/'failed',started=time.monotonic())
            failure=json.loads((self.root/'failed/receipt.json').read_bytes())
            self.assertEqual(failure['status'],'FAIL');self.assertEqual(failure['failure_stage'],'entry')
            self.assertFalse((self.root/'failed/submission.csv').exists())


if __name__=='__main__':unittest.main()
