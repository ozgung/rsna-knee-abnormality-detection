import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import subprocess
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('run96',ROOT/'scripts/colab_source96_run.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
PRIMARY=Path(os.environ.get('SOURCE_SPAN_EVIDENCE_ROOT','/nonexistent'))
PACKAGE=PRIMARY/'outputs/kaggle/source96-training-20261012-draft-v2'

@unittest.skipUnless(PACKAGE.exists() and (PRIMARY/'scripts/colab_tracka_persist.py').exists(),'Reviewed package and authoritative persister required')
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        stage=self.root/'cache';stage.mkdir();init=self.root/'initializer.json';weights=self.root/'weights.pt';weights.write_bytes(b'synthetic')
        init.write_text(json.dumps({'weights_file':str(weights),'weights_sha256':runner.sha(weights)}))
        rp=stage/'staging-receipt.json';rp.write_text(json.dumps({'status':'PASS','manifest_sha256':'a'*64,'content_sha256':'b'*64}))
        self.c={'version':runner.VERSION,'name':'own224-d96wide-v1','vm_session_id':'fixtureVM','package_root':str(PACKAGE),'package_manifest_sha256':runner.sha(PACKAGE/'manifest.json'),
            'workdir':str(self.root/'work'),'input_root':str(self.root/'input'),'stage_root':str(stage),'manifest_sha256':'a'*64,'content_sha256':'b'*64,'stage_receipt_sha256':runner.sha(rp),
            'initializer_descriptor':str(init),'initializer_descriptor_sha256':runner.sha(init),'persister':str(PRIMARY/'scripts/colab_tracka_persist.py'),'persister_sha256':runner.PERSISTER_SHA,
            'persistence':{'dataset':'fixture/private-backup','private':True,'staging':str(self.root/'backup'),'status_file':str(self.root/'backup-status.json')},'controller_lock':str(self.root/'vm.lock'),'workers':2,'resume':False}
        self.procs=[];self.calls=[];self.privacy_calls=[]
    def tearDown(self):self.tmp.cleanup()
    def artifacts(self):
        work=Path(self.c['workdir']);work.mkdir(exist_ok=True)
        (work/'raptor_resume_fold0.pt').write_bytes(b'private-resume-epoch9');(work/'raptor_ft_fold0_swa.pt').write_bytes(b'fixed-selected-swa')
        (work/'oof_fold0.csv').write_bytes(b'synthetic schema tested separately')
        meta={'partial':False,'hist':[{'ep':i} for i in range(10)]}
        (work/'raptor_ft_fold0.json').write_text(json.dumps(meta))
        record={'fold':0,'epochs_requested':10,'epochs_completed':10,'stopped_by_time_budget':False,'checkpoint_kind':'swa_topk','comparison_endpoint':'matched K16 primary',
            'history':meta['hist'],'export':{'k_eval':16,'n_val':870,'n_gold':0},'selected_epochs':[3,6,9],'checkpoint_file':'raptor_ft_fold0_swa.pt','checkpoint_sha256':runner.sha(work/'raptor_ft_fold0_swa.pt')}
        (work/'run_fold0.json').write_text(json.dumps(record))
        staging=Path(self.c['persistence']['staging']);staging.mkdir(exist_ok=True)
        payload={'dataset':self.c['persistence']['dataset'],'fold':0,'epoch':9,'resume_sha256':runner.sha(work/'raptor_resume_fold0.pt')}
        (staging/'payload_manifest.json').write_text(json.dumps(payload))
        st=(work/'raptor_resume_fold0.pt').stat()
        return {'dataset':self.c['persistence']['dataset'],'fold':0,'verified_epoch':9,'verified_version':2,'verified_signature':[st.st_size,st.st_mtime_ns],
                'final':{'state':'verified','epoch':9,'version':2}}
    def factory(self,train_rc=0,persist_rc=0,build=True):
        outer=self
        class Process:
            pid=987654321
            def __init__(self,is_persist):
                self.is_persist=is_persist;self.returncode=None if is_persist else train_rc;self.stdout=io.StringIO('[train:fold0] ep0 loss 0.4 | OOF masked macro-AUC 0.8 IDs SECRET\nsecret token must not stream\n');self.signals=[]
                if not is_persist and build:outer.status=outer.artifacts()
            def poll(self):return self.returncode
            def send_signal(self,s):self.signals.append(s)
            def wait(self,timeout=None):self.returncode=persist_rc if self.is_persist else train_rc;return self.returncode
        def popen(cmd,**kw):
            self.calls.append((cmd,kw));p=Process(len(self.procs)==0);self.procs.append(p);return p
        return popen
    def invoke(self,**kw):
        defaults={'popen':self.factory(),'probe':lambda c,e:{'device':'cuda','threads':1,'studies':4345,'timm':'1.0.26'},
            'privacy':lambda d:self.privacy_calls.append(d),'read_status':lambda _:self.status,'install_signals':False,'sleep':lambda _:None}
        defaults.update(kw)
        with patch('builtins.print'):return runner.run(self.c,**defaults)
    def test_success_final_epoch9_keeps_resume_and_old_versions(self):
        r=self.invoke();self.assertEqual(r['status'],'PASS');self.assertTrue(r['resume_preserved'])
        self.assertEqual(self.procs[0].signals,[signal.SIGUSR1]);self.assertIn('--keep-old-versions',self.calls[0][0]);self.assertNotIn('--delete-old-versions',self.calls[0][0])
        env=self.calls[1][1]['env'];self.assertEqual(env['WORKERS'],'2');self.assertEqual(env['RAPTOR_PRELOAD'],'0');self.assertEqual(env['EPOCHS'],'10');self.assertEqual(env['TIME_BUDGET_H'],'0')
        self.assertEqual(len(self.privacy_calls),2)
    def test_training_failure_still_requests_final_backup(self):
        r=self.invoke(popen=self.factory(train_rc=1));self.assertEqual(r['status'],'FAIL');self.assertTrue(r['resume_preserved']);self.assertEqual(self.procs[0].signals,[signal.SIGUSR1])
    def test_backup_failure_and_stale_final_signature_fail(self):
        r=self.invoke(popen=self.factory(persist_rc=3));self.assertEqual(r['status'],'FAIL')
        self.procs=[];self.calls=[];self.c['resume']=True;self.c['resume_sha256']=runner.sha(Path(self.c['workdir'])/'raptor_resume_fold0.pt')
        def stale(_):return {**self.status,'verified_signature':[1,2]}
        r=self.invoke(read_status=stale);self.assertEqual(r['status'],'FAIL');self.assertTrue(r['resume_preserved'])
    def test_artifact_worker_privacy_and_staging_pins_fail_before_process(self):
        for key,value in [('package_manifest_sha256','c'*64),('persister_sha256','c'*64),('stage_receipt_sha256','c'*64),('initializer_descriptor_sha256','c'*64),('workers',10)]:
            old=self.c[key];self.c[key]=value
            with self.assertRaises(ValueError):self.invoke()
            self.c[key]=old
        self.c['persistence']['private']=False
        with self.assertRaises(ValueError):self.invoke()
        self.assertEqual(self.calls,[])
    def test_existing_work_explicit_resume_required(self):
        self.artifacts()
        with self.assertRaises(ValueError):runner.validate(self.c)
        self.c.update(resume=True,resume_sha256=runner.sha(Path(self.c['workdir'])/'raptor_resume_fold0.pt'))
        runner.validate(self.c);self.assertEqual(runner.environment(self.c)['SOURCE96_RESUME_SHA256'],self.c['resume_sha256'])
        self.c['resume_sha256']='c'*64
        with self.assertRaises(ValueError):runner.validate(self.c)
    def test_lock_prevents_duplicate_controller(self):
        with open(self.c['controller_lock'],'w') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(RuntimeError):self.invoke()
        self.assertEqual(self.calls,[])
    def test_primary_incomplete_or_secondary_cannot_pass(self):
        self.artifacts();p=Path(self.c['workdir'])/'run_fold0.json';r=json.loads(p.read_text());r['export']['k_eval']=94;p.write_text(json.dumps(r))
        with self.assertRaises(ValueError):runner.primary_complete(self.c['workdir'])
        r['export']['k_eval']=16;r['epochs_completed']=9;p.write_text(json.dumps(r))
        with self.assertRaises(ValueError):runner.primary_complete(self.c['workdir'])
    def test_shutdown_signal_preserves_checkpoint_and_finalizes_backup(self):
        handlers={};base=self.factory()
        def popen(cmd,**kw):
            p=base(cmd,**kw)
            if not p.is_persist:p.returncode=None
            return p
        def register(sig,handler):handlers[sig]=handler;return signal.SIG_DFL
        def killpg(pid,sig):self.procs[1].returncode=143
        def tick(_):handlers[signal.SIGTERM](signal.SIGTERM,None)
        with patch.object(runner.signal,'signal',side_effect=register),patch.object(runner.os,'killpg',side_effect=killpg):
            r=self.invoke(popen=popen,sleep=tick,install_signals=True)
        self.assertEqual(r['status'],'FAIL');self.assertEqual(r['interrupted_signal'],signal.SIGTERM)
        self.assertTrue(r['resume_preserved']);self.assertEqual(self.procs[0].signals,[signal.SIGUSR1])

    def test_private_tracebacks_retained_but_not_printed(self):
        class Output:
            stdout=io.StringIO('Traceback: private diagnostic\nSECRET report or credential\n')
        target=self.root/'private-diagnostic.log'
        with patch('builtins.print') as print_:runner.stream(Output(),'training',target)
        self.assertEqual(target.read_text(),'Traceback: private diagnostic\nSECRET report or credential\n')
        self.assertEqual(target.stat().st_mode & 0o777,0o600);print_.assert_not_called()
        completed=subprocess.CompletedProcess([],1,'runtime private stdout\n','Traceback: runtime private stderr\n')
        with patch.object(runner.subprocess,'run',return_value=completed),self.assertRaises(RuntimeError):runner.runtime_probe(self.c,{})
        work=Path(self.c['workdir'])
        self.assertEqual((work/'source96-runtime-stdout.log').read_text(),completed.stdout)
        self.assertEqual((work/'source96-runtime-stderr.log').read_text(),completed.stderr)
    def test_dead_persister_ignored_term_escalates_after60(self):
        clock=[0.];base=self.factory(persist_rc=3)
        def popen(cmd,**kw):
            p=base(cmd,**kw)
            if p.is_persist:p.returncode=3
            else:
                p.returncode=None
                def wait(timeout=None):
                    if p.returncode is None:raise subprocess.TimeoutExpired('injected',timeout)
                    return p.returncode
                p.wait=wait
            return p
        signals=[]
        def killpg(pid,sig):
            signals.append(sig)
            if sig==signal.SIGKILL:self.procs[1].returncode=-9
        def tick(_):clock[0]+=30
        with patch.object(runner.os,'killpg',side_effect=killpg):
            r=self.invoke(popen=popen,sleep=tick,monotonic=lambda:clock[0])
        self.assertEqual(clock[0],60);self.assertEqual(signals,[signal.SIGTERM,signal.SIGKILL])
        self.assertEqual(r['training_returncode'],-9);self.assertEqual(r['status'],'FAIL');self.assertTrue(r['resume_preserved'])

    def test_probe_failure_never_launches(self):
        def bad(*_):raise RuntimeError('CUDA unavailable')
        with self.assertRaises(RuntimeError):self.invoke(probe=bad)
        self.assertEqual(self.calls,[])

class SanitizerTests(unittest.TestCase):
    def test_private_metadata_fail_closed(self):
        from types import SimpleNamespace
        class API:
            def authenticate(self):pass
            def dataset_metadata(self,dataset,t):
                p=Path(t)/'dataset-metadata.json';p.write_text(json.dumps({'id':dataset,'isPrivate':True}));return str(p)
        with patch.dict('sys.modules',{'kaggle.api.kaggle_api_extended':SimpleNamespace(KaggleApi=API)}):
            self.assertTrue(runner.private_dataset('fixture/private')['private'])
        class Public(API):
            def dataset_metadata(self,dataset,t):
                p=Path(t)/'dataset-metadata.json';p.write_text(json.dumps({'isPrivate':False}));return str(p)
        with patch.dict('sys.modules',{'kaggle.api.kaggle_api_extended':SimpleNamespace(KaggleApi=Public)}),self.assertRaises(ValueError):runner.private_dataset('fixture/public')

    def test_current_nested_metadata_requires_exact_identity_and_boolean_privacy(self):
        from types import SimpleNamespace
        value={'info':{'ownerUser':'fixture','datasetSlug':'private','isPrivate':True}}
        class API:
            def authenticate(self):pass
            def dataset_metadata(self,dataset,t):
                p=Path(t)/'dataset-metadata.json';p.write_text(json.dumps(value));return str(p)
        with patch.dict('sys.modules',{'kaggle.api.kaggle_api_extended':SimpleNamespace(KaggleApi=API)}):
            self.assertTrue(runner.private_dataset('fixture/private')['private'])
            for bad in [{'info':{'ownerUser':'other','datasetSlug':'private','isPrivate':True}},
                        {'info':{'ownerUser':'fixture','datasetSlug':'private','isPrivate':False}},
                        {'info':{'ownerUser':'fixture','datasetSlug':'private','isPrivate':'true'}},
                        {'isPrivate':True}]:
                value=bad
                with self.assertRaises(ValueError):runner.private_dataset('fixture/private')

    def test_logs_whitelist_only_numeric_progress(self):
        self.assertEqual(runner.sanitized('[train:fold0] ep1 loss 0.2 | OOF masked macro-AUC 0.8 ID private'),'ep1 loss 0.2 | OOF masked macro-AUC 0.8')
        self.assertIsNone(runner.sanitized('Authorization: private-token'))
        self.assertIsNone(runner.sanitized('https://signed.example/secret?id=patient'))
        self.assertEqual(runner.sanitized('[persist 12:12:12] FINAL PERSIST VERIFIED epoch 9 version 2'),'FINAL PERSIST VERIFIED epoch 9 version 2')
