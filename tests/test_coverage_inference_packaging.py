"""Synthetic own-EMA export integrity and offline package attachment tests."""
import ast
import copy
import hashlib
import os
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from knee.fresh_reader import fingerprint, weak_validation_evidence
from knee.coverage_inference import CODECS, STRESS_CASES, POLICY
import knee.coverage_inference as inference

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
spec=importlib.util.spec_from_file_location('coverage_inference_builder',SCRIPTS/'build_coverage_inference.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
try:
    import torch
    import numpy as np
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Optional ML libraries unavailable')
class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.sources=self.root/'sources';self.sources.mkdir()
        external = {name: '# synthetic source\n' for name in builder.SOURCE_HASHES}
        for name, text in external.items(): (self.sources/name).write_text(text)
        hashes = {n: hashlib.sha256(v.encode()).hexdigest() for n,v in external.items()}
        for module in (builder, inference):
            pin = patch.object(module, 'SOURCE_HASHES', hashes);pin.start();self.addCleanup(pin.stop)
        self.external = external
        self.result=self.root/'result';self.result.mkdir();self.cache=self.root/'cache';self.cache.mkdir()
        self.targets=['target'+str(i) for i in range(12)]
        rows=[{'uid':'private-'+str(i),'group':'group'+str(i),'split':'weak_validation','labels':[.9 if i%2 else .1]*12} for i in range(200)]
        rows += [{'uid':'train-'+str(i),'group':'train-group'+str(i),'split':'train','labels':[.5]*12} for i in range(2000)]
        plan={'version':builder.VERSION,'records':rows,'targets':self.targets}
        plan['fingerprint']=fingerprint(plan)
        self.plan=plan
        self.write(self.cache/'plan-private.json',plan)
        self.write(self.cache/'index-private.json',{})
        self.write(self.cache/'cache-report.json',{'status':'PASS','stage':'complete','cached_studies':2200,'plan_fingerprint':plan['fingerprint']})
        self.training_build=self.root/'training-build';self.training_build.mkdir()
        config=dict(seed=42,backbone='convnext_tiny.fb_in22k_ft_in1k',image_size=256,
                    train_windows=8,eval_windows=12,batch_size=2,epochs=12,lr_encoder=1e-4,
                    lr_head=5e-4,weight_decay=.02,ema=.998,seconds=43200,host_cache_bytes=268435456)
        self.bundle={n:'# synthetic helper\n' for n in builder.TRAINING_BUNDLE}
        self.bundle['knee/coverage_cache.py']=(builder.ROOT/'knee/coverage_cache.py').read_text()
        self.bundle['knee/coverage_training.py']=f'CONFIG = {config!r}\nINITIAL_ENCODER_SHA256 = {builder.INITIAL_ENCODER_SHA256!r}\n'
        self.archive={'revision':'a'*40,'config':config,'sources':{n:hashlib.sha256(v.encode()).hexdigest() for n,v in self.bundle.items()},
                      'external':builder.SOURCE_HASHES,'cache_pins':{n:builder.file_sha256(self.cache/n) for n in
                        ('plan-private.json','cache-report.json','index-private.json')}}
        pin=patch.object(builder,'TRAINING_RUNNER_SHA256',self.archive['sources']['knee/coverage_training.py'])
        pin.start();self.addCleanup(pin.stop)
        self.write_archive()
        self.contract={'version':'train04-v1','variant':'v1','config':config,
            'source_hashes':builder.SOURCE_HASHES,'runtime_sources':{n:self.archive['sources'][n] for n in builder.TRAINING_RUNTIME},
            'runner_sha256':builder.TRAINING_RUNNER_SHA256,
            'plan_fingerprint':plan['fingerprint'],'index_sha256':builder.file_sha256(self.cache/'index-private.json'),
            'complete_sha256':builder.file_sha256(self.cache/'cache-report.json'),'runtime_environment':{'python':'3.12','device':'cuda'}}
        self.contract['fingerprint']=fingerprint(self.contract)
        self.score=float(np.log(2))
        self.evidence=weak_validation_evidence(plan,list(range(200)),[[.5]*12]*200,logits=[[0.]*12]*200,epoch=6,weak_bce=self.score)
        model=torch.nn.Linear(1,1)
        self.checkpoint={'contract':self.contract,'epoch':6,'weak_bce':self.score,'targets':self.targets,
            'initial_encoder_sha256':builder.INITIAL_ENCODER_SHA256,'weak_validation_evidence':self.evidence,'model':model.state_dict()}
        torch.save(self.checkpoint,self.result/'best.pt')
        self.private={'contract':self.contract,'best_checkpoint_sha256':builder.file_sha256(self.result/'best.pt'),'evidence':self.evidence}
        self.report={'contract':self.contract,'status':'PASS','completed_epochs':12,
            'history':[{'epoch':i,'weak_bce':self.score if i==6 else 1.} for i in range(1,13)],
            'selected_bce':self.score,'initial_encoder_sha256':builder.INITIAL_ENCODER_SHA256}
        self.write(self.result/'training-report.json',self.report)
        self.run={'contract':self.contract,'initial_encoder_sha256':builder.INITIAL_ENCODER_SHA256,
                  'environment':{'code_revision':self.archive['revision'],
                                 'source_sha256':hashlib.sha256(json.dumps(self.archive,sort_keys=True).encode()).hexdigest()}}
        self.write(self.result/'run-contract.json',self.run)
        self.write(self.result/'best-weak-validation-private.json',self.private)
        self.code=types.SimpleNamespace(LABELS=self.targets,KneeNet=lambda *a,**kw:torch.nn.Linear(1,1))

    def tearDown(self):self.temp.cleanup()
    def write(self,path,value):path.write_text(json.dumps(value))
    def write_archive(self):
        sha=hashlib.sha256(json.dumps(self.archive,sort_keys=True).encode()).hexdigest()
        entry=f'CODE={self.bundle!r}\nEXTERNAL={self.external!r}\nCONTRACT={self.archive!r}\n'
        entry+=f"os.environ['KNEE_CODE_REVISION']={self.archive['revision']!r}\nos.environ['KNEE_SOURCE_SHA256']={sha!r}\n"
        self.write(self.training_build/'build-contract.json',self.archive)
        self.write(self.training_build/'notebook.ipynb',{'cells':[{'cell_type':'code','source':entry.splitlines(True)}]})

    def export(self,name='assets'):
        with patch.object(builder,'check_sources'),patch.object(builder,'load_source',return_value=self.code):
            return builder.export_weights(self.sources,self.result,self.cache,self.root/name,training_build=self.training_build)

    def test_export_only_selected_ema_tensors_and_non_sensitive_manifest(self):
        manifest=self.export();asset=self.root/'assets'
        self.assertEqual(set(p.name for p in asset.iterdir()),{'weights.pt','manifest.json'})
        exported=torch.load(asset/'weights.pt',weights_only=True)
        self.assertEqual(set(exported),{'model'})
        for name,value in self.checkpoint['model'].items():torch.testing.assert_close(exported['model'][name],value)
        self.assertEqual(manifest['selected_epoch'],6)
        self.assertNotIn('private-',json.dumps(manifest));self.assertNotIn('weak_labels',json.dumps(manifest))
        self.assertEqual(manifest['weights_sha256'],builder.file_sha256(asset/'weights.pt'))
        with self.assertRaises(ValueError):self.export()

    def test_checkpoint_source_selection_and_private_evidence_mixing_rejected(self):
        cases=('source','epoch','evidence','hash','incomplete')
        for mode in cases:
            with self.subTest(mode=mode):
                ck=copy.deepcopy(self.checkpoint);p=copy.deepcopy(self.private);r=copy.deepcopy(self.report)
                if mode=='source':ck['contract']['config']['image_size']=224
                if mode=='epoch':ck['epoch']=7
                if mode=='evidence':p['evidence']['ids'][0]='wrong'
                if mode=='incomplete':r['completed_epochs']=11
                torch.save(ck,self.result/'best.pt')
                p['best_checkpoint_sha256']=builder.file_sha256(self.result/'best.pt') if mode!='hash' else '0'*64
                self.write(self.result/'best-weak-validation-private.json',p);self.write(self.result/'training-report.json',r)
                with self.assertRaises(ValueError):self.export('bad-'+mode)
                self.assertFalse((self.root/('bad-'+mode)).exists())
        with patch.object(builder,'check_sources',side_effect=ValueError('source')):
            with self.assertRaises(ValueError):builder.export_weights(self.sources,self.result,self.cache,self.root/'bad-public',training_build=self.training_build)

    def test_export_uses_archived_runner_after_current_training_code_changes(self):
        newer=self.root/'new-current-tree';(newer/'knee').mkdir(parents=True)
        (newer/'knee/coverage_cache.py').write_text(self.bundle['knee/coverage_cache.py'])
        (newer/'knee/coverage_training.py').write_text("raise RuntimeError('future objective must not be imported')\n")
        with patch.object(builder,'ROOT',newer):manifest=self.export()
        self.assertEqual(manifest['training_revision'],self.archive['revision'])
        self.assertEqual(manifest['training_source_sha256'],self.run['environment']['source_sha256'])
        self.assertEqual(manifest['training_notebook_sha256'],builder.file_sha256(self.training_build/'notebook.ipynb'))

    def test_archive_source_config_identity_and_duplicate_tamper_rejected(self):
        original=json.loads((self.training_build/'notebook.ipynb').read_text())
        for mode in ('source','external','contract','duplicate','config','runner','identity','run','cache'):
            with self.subTest(mode=mode):
                nb=copy.deepcopy(original);archive=copy.deepcopy(self.archive);run=copy.deepcopy(self.run)
                if mode=='source':nb['cells'][0]['source'][0]=nb['cells'][0]['source'][0].replace('# synthetic helper','# changed helper',1)
                if mode=='external':nb['cells'][0]['source'][1]=nb['cells'][0]['source'][1].replace('# synthetic source','# changed source',1)
                if mode=='contract':archive['revision']='b'*40
                if mode=='duplicate':nb['cells'][0]['source'].append('CODE = {}\n')
                if mode in ('config','runner','cache'):
                    if mode=='config':archive['config']['batch_size']=4
                    if mode=='runner':archive['sources']['knee/coverage_training.py']='b'*64
                    if mode=='cache':archive['cache_pins']['index-private.json']='b'*64
                    nb['cells'][0]['source'][2]=f'CONTRACT={archive!r}\n'
                    sha=hashlib.sha256(json.dumps(archive,sort_keys=True).encode()).hexdigest()
                    nb['cells'][0]['source'][4]=f"os.environ['KNEE_SOURCE_SHA256']={sha!r}\n"
                    run['environment']['source_sha256']=sha
                if mode=='identity':nb['cells'][0]['source'][4]="os.environ['KNEE_SOURCE_SHA256']='wrong'\n"
                if mode=='run':run['environment']['source_sha256']='b'*64
                self.write(self.training_build/'notebook.ipynb',nb);self.write(self.training_build/'build-contract.json',archive)
                self.write(self.result/'run-contract.json',run)
                with self.assertRaises(ValueError):self.export('archive-'+mode)
                self.assertFalse((self.root/('archive-'+mode)).exists())
        with self.assertRaises(ValueError):builder.config_literal(ast.parse("dict(seed=__import__('os').system('false'))",mode='eval').body)
        self.assertEqual(builder.config_literal(ast.parse('dict(seconds=12*3600,bytes=256*1024**2)',mode='eval').body),
                         {'seconds':43200,'bytes':268435456})

    def test_private_offline_rehearsal_then_matching_full_package(self):
        self.export()
        with patch.object(builder,'check_sources'):
            first=builder.build(self.sources,self.root/'assets',self.root/'rehearsal','synthetic-rehearsal',
                                weights_dataset='ozgung/synthetic-own-weights',cache_kernel='ozgung/synthetic-cache')
        metadata=json.loads((self.root/'rehearsal/kernel-metadata.json').read_text())
        self.assertTrue(metadata['is_private']);self.assertFalse(metadata['enable_internet'])
        self.assertEqual(metadata['dataset_sources'],['ozgung/synthetic-own-weights',builder.CONVNEXT_DATASET])
        self.assertEqual(metadata['kernel_sources'],['ozgung/synthetic-cache'])
        self.assertEqual(metadata['competition_sources'],['rsna-knee-abnormality-detection'])
        nb=json.loads((self.root/'rehearsal/notebook.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'synthetic-package','exec')
        self.assertLess((self.root/'rehearsal/notebook.ipynb').stat().st_size,900000)
        report={'status':'PASS','rehearsal':True,'qualification_fingerprint':first['qualification_fingerprint'],
            'policy':POLICY,'train_studies':64,'test_studies':3,'strict_cache_comparisons':64,'input_max_abs_error':0,
            'sample_fingerprint':'a'*64,'empty_studies':0,'resource_errors':0,'prediction_fallbacks':0,
            **dict.fromkeys(('selected_series','missing_slots','unreadable_files','partial_series','short_series',
                           'black_frames','empty_series','failed_series','missing_series_directories'),0),'repeat_max_abs_error':0.,'cache_prediction_max_abs_error':0.,'elapsed_seconds':200,
            'stress_checks':{'status':'PASS','uses_images':False,'uses_labels':False,'cases':dict.fromkeys(STRESS_CASES,'PASS')}}
        path=self.root/'rehearsal.json';self.write(path,report)
        with patch.object(builder,'check_sources'):
            final=builder.build(self.sources,self.root/'assets',self.root/'entry','synthetic-entry',
                weights_dataset='ozgung/synthetic-own-weights',rehearsal_report=path)
        self.assertEqual(first['qualification_fingerprint'],final['qualification_fingerprint'])
        meta=json.loads((self.root/'entry/kernel-metadata.json').read_text())
        self.assertFalse(meta['kernel_sources']);self.assertFalse(meta['enable_internet'])
        report['policy']='old';self.write(path,report)
        with patch.object(builder,'check_sources'),self.assertRaises(ValueError):
            builder.build(self.sources,self.root/'assets',self.root/'bad-entry','synthetic-entry',
                weights_dataset='ozgung/synthetic-own-weights',rehearsal_report=path)
        self.assertFalse((self.root/'bad-entry').exists())
        (self.root/'assets/weights.pt').write_bytes(b'changed')
        with patch.object(builder,'check_sources'),self.assertRaises(ValueError):
            builder.build(self.sources,self.root/'assets',self.root/'changed','changed',
                weights_dataset='ozgung/synthetic-own-weights',cache_kernel='ozgung/cache')


class ArchivedBuildTests(unittest.TestCase):
    def test_actual_preserved_reference_build_without_execution(self):
        parent=os.environ.get('KNEE_REFERENCE_PARENT')
        if not parent:self.skipTest('Optional preserved private training build unavailable')
        path=Path(parent)/'outputs/kaggle/train04-reference-20261009'
        if not path.is_dir():self.skipTest('Optional preserved private training build unavailable')
        build,evidence=builder.read_training_build(path)
        self.assertEqual(build['revision'],'6af890239a4111b1db2dbeab283bde01e90040d8')
        self.assertEqual(evidence['training_source_sha256'],'2b178cdb837b98a6f70c7e50ca448384ec88a0a099eb054d1b9d1a52a7cbd46b')
        self.assertEqual(build['config']['epochs'],12)


if __name__=='__main__':unittest.main()
