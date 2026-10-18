"""Fold extension tests use private pinned membership metadata, never images/outcomes."""
import copy
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import tests.test_source96_training as training_tests
import tests.test_colab_source96_run as run_tests
from knee import source96_training as guard,public_training_smoke as smoke

builder=training_tests.builder
EVIDENCE=training_tests.EVIDENCE
READY=(EVIDENCE/'outputs/analysis/public-train-recipe-20261009/corpus/all_ids.npy').is_file()

@unittest.skipUnless(READY,'Pinned membership metadata required')
class FoldTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.out=self.root/'package-fold1'
        self.receipt=builder.build(self.out,evidence_root=EVIDENCE,fold=1)
        self.split,self.train,self.dev=builder.derive_split(EVIDENCE,1)
        self.ids0train,self.ids0dev=builder.derive_split(EVIDENCE,0)[1:]
        import numpy as np
        ids=[]
        for sub,prefix in [('corpus','all'),('corpus-ext','extra')]:ids.extend(np.load(EVIDENCE/f'outputs/analysis/public-train-recipe-20261009/{sub}/{prefix}_ids.npy',allow_pickle=False).astype(str).tolist())
        eligible=set(self.train+self.dev);self.ids=[u for u in ids if u in eligible]
        self.folds=json.loads((self.out/'folds_v0.json').read_bytes())['folds']
        self.excluded=json.loads((self.out/'excluded_ids.json').read_bytes())['quarantined_study_ids']
    def tearDown(self):self.tmp.cleanup()
    def test_real_fold1_build_and_historical_control_order(self):
        self.assertEqual(self.split,guard.SPLITS[1]);self.assertEqual((len(self.train),len(self.dev)),(3479,866))
        self.assertFalse(set(self.train)&set(self.dev));self.assertEqual(set(self.train)|set(self.dev),set(self.ids));self.assertFalse(set(self.ids)&set(self.excluded))
        path=EVIDENCE/'outputs/kaggle/trackA-coatnet224-fold01-20261009-result/trackA/oof_fold1.csv'
        with path.open() as f:control=[r['StudyInstanceUID'] for r in csv.DictReader(f) if r['set']=='val']
        self.assertEqual(control,self.dev)
        old=EVIDENCE/'outputs/kaggle/trackA-coatnet224-fold01-20261009/train_knee_patched.py'
        self.assertEqual(builder.scientific_ast(old.read_text()),builder.scientific_ast((self.out/'train_knee_patched.py').read_text()))
        args=SimpleNamespace(**{**guard.FROZEN,'fold':1},workers=2)
        guard.require_recipe(args,self.out/'train_knee_patched.py')
        args.fold=0
        with self.assertRaises(ValueError):guard.require_recipe(args,self.out/'train_knee_patched.py')
        self.assertEqual(guard.policy_fold(self.receipt['policy']),1)
    def test_legacy_fold0_stays_fold0_only(self):
        legacy=EVIDENCE/'outputs/kaggle/source96-training-20261012-reviewed-v1'
        self.assertEqual(guard.policy_fold(guard.recipe(legacy/'train_knee_patched.py')),0)
        args=SimpleNamespace(**guard.FROZEN,workers=2);guard.require_recipe(args,legacy/'train_knee_patched.py')
        args.fold=1
        with self.assertRaises(ValueError):guard.require_recipe(args,legacy/'train_knee_patched.py')
        bad=copy.deepcopy(guard.recipe(legacy/'train_knee_patched.py'));bad['frozen']['fold']=1
        with self.assertRaises(ValueError):guard.policy_fold(bad)
        self.assertEqual(guard.experiment_split(self.ids,self.folds,self.excluded,0)[0],guard.SPLITS[0])
    def test_split_mutation_gold_quarantine_and_crossfold_guards(self):
        for ids in (self.ids[::-1],self.ids[:-1],self.ids+[self.ids[0]]):
            with self.assertRaises(ValueError):guard.experiment_split(ids,self.folds,self.excluded,1)
        bad=copy.copy(self.folds);bad[self.ids[0]]=-1
        with self.assertRaises(ValueError):guard.experiment_split(self.ids,bad,self.excluded,1)
        with self.assertRaises(ValueError):guard.experiment_split(self.ids,self.folds,[self.ids[0]]+self.excluded[1:],1)
        script=self.out/'train_knee_patched.py';f=self.out/'folds_v0.json';e=self.out/'excluded_ids.json'
        with patch.object(guard,'_CORPUS',(None,None,self.ids)),patch.object(guard,'_CACHE_IDENTITY',{'fixture':False,'membership':guard.SPLITS[0]}):
            guard.verify_membership(self.train,self.dev,[],1,f,e,script)
            with self.assertRaises(ValueError):guard.verify_membership(self.ids0train,self.ids0dev,[],1,f,e,script)
            with self.assertRaises(ValueError):guard.runtime_split(0,f,e,script)
            with self.assertRaises(ValueError):guard.verify_membership(self.train,self.dev,[],fold=1)
        with self.assertRaises(ValueError):guard.export_guard(self.out,0,self.out/'state.pt','swa_topk',[1,2,3],False,False)
    def test_fold1_runner_success_and_crossfold_backup_failure(self):
        fixture=run_tests.RunnerTests();fixture.setUp()
        try:
            c=fixture.c;c.update(fold=1,name='own224-d96wide-fold1-v1',package_root=str(self.out),package_manifest_sha256=smoke.file_sha(self.out/'manifest.json'),
                workdir=str(fixture.root/'work-fold1'),controller_lock=str(fixture.root/'vm-fold1.lock'))
            c['persistence'].update(dataset='fixture/private-fold1-backup',staging=str(fixture.root/'backup-fold1'),status_file=str(fixture.root/'status-fold1.json'))
            original=fixture.artifacts
            def artifacts():
                status=original();work=Path(c['workdir'])
                for p in list(work.iterdir()):
                    if 'fold0' in p.name:p.rename(p.with_name(p.name.replace('fold0','fold1')))
                record=work/'run_fold1.json';r=json.loads(record.read_text());r.update(fold=1,checkpoint_file='raptor_ft_fold1_swa.pt',source96_experiment_split=guard.SPLITS[1],cache_reference_partition_fold=0,source96_cache_content_sha256=c['content_sha256']);r['export']['n_val']=866;record.write_text(json.dumps(r))
                fixture.published_payload['fold']=1;status['fold']=1
                return status
            fixture.artifacts=artifacts
            receipt=fixture.invoke();self.assertEqual(receipt['status'],'PASS');self.assertEqual(receipt['fold'],1)
            cmd=fixture.calls[0][0];self.assertEqual(cmd[cmd.index('--fold')+1],'1');self.assertEqual(fixture.calls[1][1]['env']['FOLD_A'],'1')
            with self.assertRaises(ValueError):run_tests.runner.primary_complete(c['workdir'],1,866,guard.SPLITS[0],c['content_sha256'])
            c.update(resume=True,resume_sha256=smoke.file_sha(Path(c['workdir'])/'raptor_resume_fold1.pt'))
            status={**fixture.status,'fold':0}
            with self.assertRaises(ValueError):run_tests.runner.final_verified(c,status)
            c['fold']=0
            with self.assertRaises(ValueError):run_tests.runner.validate(c)
            c['fold']=1;c['persistence']['dataset']='fixture/private-fold0-backup'
            with self.assertRaises(ValueError):run_tests.runner.validate(c)
        finally:fixture.tearDown()
