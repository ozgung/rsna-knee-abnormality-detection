"""Synthetic contracts exercise qualification/packaging; no GPU evidence inferred."""
import ast,copy,csv,importlib.util,json,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from knee import own_model_submission as final,public_training_smoke as smoke,raptor_input
SPEC=importlib.util.spec_from_file_location('own_submission_builder',Path(__file__).resolve().parents[1]/'scripts/build_own_model_submission.py')
builder=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(builder)
READER='synthetic-reader';FROZEN='e331776e19800ef584c653cb9bb90698ff708027b0f9f483ce7afb18338a2f35'


def fixture(training_round=None):
    sources={n:(builder.ROOT/n).read_text() for n in builder.FILES};model='pass\n'
    members=[{'fold':f,'kernel':final.KERNELS[0 if f<2 else (1 if f<4 else 2)],'run_file':f'run_fold{f}.json',
              'run_sha256':'a'*64,'checkpoint_file':f'fold{f}.pt','checkpoint_sha256':'b'*64,'selected_epochs':[8,9,7]} for f in range(5)]
    c={'complete_family':True,'settings':final.own.SETTINGS,'members':members,'model_source_sha256':smoke.SOURCE_SHA,
       'reader_sha256':FROZEN,'model_definitions_sha256':smoke.digest(model.encode()),
       'sources':{n:smoke.digest(s.encode()) for n,s in sources.items()},'codecs':builder.CODECS}
    if training_round is not None:
        c.update(training_round=training_round,labels_file=final.LABEL_PINS[training_round][0],labels_sha256=final.LABEL_PINS[training_round][1])
        for m in members:m['kernel']=final.ROUND_KERNELS[training_round][0 if m['fold']<2 else (1 if m['fold']<4 else 2)]
    report={'status':'PASS','contract_sha256':smoke.digest(smoke.canonical(c)),'qualified_folds':list(range(5)),
            'train_studies':5,'test_studies':3,'settings':final.own.SETTINGS,'labels_read':False,
            'training_parity':[{'slots':[{'exact_hash_match':True,'mask_match':True} for _ in range(5)],'windows_exact':True} for _ in range(5)],
            'models':[{'fold':m['fold'],'checkpoint_sha256':m['checkpoint_sha256'],'strict_load':True,'repeat_max_error':0,
                       'train_probability_max_error':0,'missing_plane_finite':True,'cuda_peak_bytes':123} for m in members],
            'stress_checks':{'cases':dict.fromkeys(raptor_input.STRESS_CASES,'PASS')},
            'environment':{'packages':{'torch':'2.10.0+cu128','timm':'1.0.26'}},'elapsed_seconds':60,'visible_csv_sha256':'c'*64}
    return sources,model,c,report


class OwnSubmissionTests(unittest.TestCase):
    def write_csv(self,path,header,rows):
        with path.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)

    def inputs(self,root,ids):
        self.write_csv(root/'test.csv',['StudyInstanceUID'],[[i] for i in ids])
        self.write_csv(root/'sample_submission.csv',['StudyInstanceUID',*smoke.TARGETS],[[i,*[0]*12] for i in ids])
        self.write_csv(root/'test_series.csv',final.SERIES_COLUMNS,[[i,'series'+str(n),'1','0','Sagittal'] for n,i in enumerate(ids)])

    def test_dynamic_ordered_hidden_coverage_and_no_training_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ids=['s4','s1','s8','s2'];self.inputs(root,ids)
            columns,actual,records=final.read_inputs(root)
            self.assertEqual(actual,ids);self.assertEqual(list(records),ids)
            self.assertEqual(columns,['StudyInstanceUID',*smoke.TARGETS])
            self.assertFalse((root/'train_series.csv').exists())
            # Hidden reruns retain the original three-row sample file.
            self.write_csv(root/'sample_submission.csv',columns,[[i,*[0]*12] for i in ['v1','v2','v3']])
            self.assertEqual(final.read_inputs(root)[1],ids)

    def test_reject_coverage_duplicate_and_series_schema(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ids=['a','b'];self.inputs(root,ids)
            self.write_csv(root/'test.csv',['StudyInstanceUID'],[['a']])
            with self.assertRaises(ValueError):final.read_inputs(root)
            self.inputs(root,ids);self.write_csv(root/'test_series.csv',final.SERIES_COLUMNS,[['extra','s1','1','0','Sagittal']])
            with self.assertRaises(ValueError):final.read_inputs(root)
            self.inputs(root,ids);self.write_csv(root/'test.csv',['StudyInstanceUID'],[['../escape'],['b']])
            with self.assertRaises(ValueError):final.read_inputs(root)
            self.inputs(root,ids);self.write_csv(root/'test_series.csv',['StudyInstanceUID'],[['a']])
            with self.assertRaises(ValueError):final.read_inputs(root)
            self.inputs(root,['a','a'])
            with self.assertRaises(ValueError):final.read_inputs(root)

    def test_csv_nan_range_order_and_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'p.csv';columns=['StudyInstanceUID',*smoke.TARGETS]
            self.write_csv(path,columns,[['a',*[.5]*12],['b',*[.2]*12]]);final.validate_csv(path,['a','b'])
            with self.assertRaises(ValueError):final.validate_csv(path,['b','a'])
            for value in ('nan','inf','-0.1','1.1'):
                self.write_csv(path,columns,[['a',*[value]*12]])
                with self.assertRaises(ValueError):final.validate_csv(path,['a'])
            self.write_csv(path,columns[::-1],[['a',*[.5]*12]])
            with self.assertRaises(ValueError):final.validate_csv(path,['a'])

    def test_no_cpu_fallback(self):
        fake=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
        with self.assertRaisesRegex(RuntimeError,'no CPU fallback'):final.require_cuda(fake)

    def test_qualification_rejects_subset_error_and_weight_hash_drift(self):
        _,_,c,r=fixture();final.require_qualification(c,r)
        for mutate in ('subset','status','hash','strict','slots'):
            cc,rr=copy.deepcopy(c),copy.deepcopy(r)
            if mutate=='subset':cc['members']=cc['members'][:2];cc['complete_family']=False
            elif mutate=='status':rr['status']='ERROR'
            elif mutate=='hash':rr['models'][4]['checkpoint_sha256']='d'*64
            elif mutate=='strict':rr['models'][3]['strict_load']=False
            else:rr['training_parity'][0]['windows_exact']=False
            with self.assertRaises(ValueError):final.require_qualification(cc,rr)

    def test_builder_blocks_subset_before_archive_or_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,_,c,r=fixture();c['complete_family']=False
            (root/'build-contract.json').write_text(json.dumps(c));(root/'report.json').write_text(json.dumps(r))
            with patch.object(builder,'archived',side_effect=AssertionError('must not reach archive')):
                with self.assertRaises(ValueError):builder.build(root/'new','own-test',rehearsal_build=root,rehearsal_report=root/'report.json',rehearsal_csv=root/'absent')
            self.assertFalse((root/'new').exists())

    def archive(self,root,training_round=None):
        sources,model,c,report=fixture(training_round)
        code=("import time,sys,hashlib\nfrom pathlib import Path\nSTARTED=time.monotonic()\n"
          f"SOURCES={sources!r}\nCONTRACT={c!r}\nREADER={READER!r}\nMODEL={model!r}\n"
          "base=Path('/kaggle/working/own_model_code')\nfor name,source in SOURCES.items():\n"
          "    if hashlib.sha256(source.encode()).hexdigest()!=CONTRACT['sources'][name]:raise ValueError('Packaged sources differ')\n"
          "    p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)\n"
          "sys.path.insert(0,str(base))\nfrom knee.own_model_rehearsal import run\n"
          "run('/kaggle/input','/kaggle/working/own_model_rehearsal',CONTRACT,READER,MODEL,started=STARTED)\n")
        nb={'cells':[{'cell_type':'code','source':code.splitlines(True)}]};blob=json.dumps(nb).encode()
        receipt={'notebook_sha256':smoke.digest(blob),'notebook_bytes':len(blob),'contract_sha256':smoke.digest(smoke.canonical(c)),
                 'candidate_folds':list(range(5)),'complete_family':True}
        meta={'enable_internet':False,'enable_gpu':True,'is_private':True,'dataset_sources':[*smoke.CORPORA,'goodpjw2008/rsna-knee-2-5d-convnext-reader'],
              'kernel_sources':[smoke.IDENTITY_KERNEL,*final.training_policy(c)],'competition_sources':['rsna-knee-abnormality-detection'],
              'model_sources':[],'docker_image':builder.DOCKER_IMAGE}
        (root/'notebook.ipynb').write_bytes(blob)
        for n,v in [('build-contract.json',c),('build-receipt.json',receipt),('kernel-metadata.json',meta)]:
            (root/n).write_text(json.dumps(v))
        self.write_csv(root/'visible.csv',['StudyInstanceUID',*smoke.TARGETS],[[i,*[.5]*12] for i in ['visible1','visible2','visible3']])
        report['visible_csv_sha256']=smoke.file_sha(root/'visible.csv');(root/'report.json').write_text(json.dumps(report))
        return c

    def test_synthetic_build_attachment_closure_immutable_and_hash_drift(self):
        real_digest=smoke.digest
        def synthetic_digest(v):return FROZEN if v==READER.encode() else real_digest(v)
        with tempfile.TemporaryDirectory() as d,patch.object(smoke,'digest',side_effect=synthetic_digest):
            root=Path(d);self.archive(root);out=root/'new'
            kwargs={'rehearsal_build':root,'rehearsal_report':root/'report.json','rehearsal_csv':root/'visible.csv'}
            receipt=builder.build(out,'own-test',**kwargs);self.assertTrue(receipt['build_only'])
            meta=json.loads((out/'kernel-metadata.json').read_text())
            self.assertEqual(meta['kernel_sources'],final.KERNELS);self.assertEqual(len(meta['dataset_sources']),1)
            self.assertFalse(meta['enable_internet']);self.assertFalse(receipt['production_cuda_tested'])
            nb=json.loads((out/'notebook.ipynb').read_text());entry=''.join(nb['cells'][1]['source']);compile(entry,'test','exec')
            self.assertNotIn('visible1',entry);self.assertIn('setitimer(signal.ITIMER_REAL,32400)',entry)
            assignments={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(entry).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('SOURCES','CONTRACT')}
            packed=root/'packed'
            for name,source in assignments['SOURCES'].items():
                path=packed/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(source)
            check=subprocess.run([sys.executable,'-I','-c','import sys;sys.path.insert(0,sys.argv[1]);import knee.own_model_submission',str(packed)],capture_output=True,text=True)
            self.assertEqual(check.returncode,0,check.stderr)
            self.assertEqual(assignments['CONTRACT']['visible_ids_sha256'],smoke.digest(smoke.canonical(['visible1','visible2','visible3'])))
            self.assertNotEqual(assignments['CONTRACT']['visible_ids_sha256'],smoke.digest(smoke.canonical(['hidden1','hidden2','hidden3'])))
            with self.assertRaises(ValueError):builder.build(out,'own-test',**kwargs)
            (root/'notebook.ipynb').write_bytes((root/'notebook.ipynb').read_bytes()+b' ')
            with self.assertRaises(ValueError):builder.build(root/'new2','own-test',**kwargs)
            self.assertFalse((root/'new2').exists())

    def test_archived_scientific_source_drift_with_rehashed_notebook_rejected(self):
        real_digest=smoke.digest
        with tempfile.TemporaryDirectory() as d,patch.object(smoke,'digest',side_effect=lambda v:FROZEN if v==READER.encode() else real_digest(v)):
            root=Path(d);self.archive(root)
            nb=json.loads((root/'notebook.ipynb').read_text())
            code=''.join(nb['cells'][0]['source'])
            original=(builder.ROOT/'knee/own_model_rehearsal.py').read_text()
            code=code.replace(repr(original),repr(original+'\n# drift\n'))
            nb['cells'][0]['source']=code.splitlines(True);blob=json.dumps(nb).encode();(root/'notebook.ipynb').write_bytes(blob)
            b=json.loads((root/'build-receipt.json').read_text());b.update(notebook_sha256=smoke.digest(blob),notebook_bytes=len(blob))
            (root/'build-receipt.json').write_text(json.dumps(b))
            with self.assertRaisesRegex(ValueError,'scientific dependency changed'):builder.archived(root)

    def test_round_policy_rejects_wrong_round_label_and_filename(self):
        self.assertEqual(final.training_policy({}),final.KERNELS)
        for number in (0,1):
            _,_,c,r=fixture(number);final.require_qualification(c,r)
            for delta in ({'training_round':2},{'training_round':True},{'training_round':'1'},
                          {'labels_file':'wrong.parquet'},{'labels_sha256':'f'*64},
                          {'training_round':1-number}):
                with self.assertRaises(ValueError):final.training_policy(c|delta)
            cc=copy.deepcopy(c);cc['members'][0]['kernel']='ozgung/arbitrary'
            r['contract_sha256']=smoke.digest(smoke.canonical(cc))
            with self.assertRaises(ValueError):final.require_qualification(cc,r)
        with self.assertRaises(ValueError):final.training_policy({'training_round':1})

    def test_round1_record_exact_labels_selection_folds_and_full_epochs(self):
        _,_,c,_=fixture(1);m=c['members'][0]
        r={'fold':0,'smoke':False,'checkpoint_kind':'swa_topk','checkpoint_file':m['checkpoint_file'],
           'checkpoint_sha256':m['checkpoint_sha256'],'selected_epochs':m['selected_epochs'],
           'train_returncode':0,'epochs_completed':10,'epochs_requested':10,'stopped_by_time_budget':False,
           'train_knee_original_sha256':smoke.SOURCE_SHA,'params':{'ARCH':final.own.ARCH,'RES':'256'},'export':{'k_eval':16},
           'labels_file':final.LABEL_PINS[1][0],'labels_sha256':final.LABEL_PINS[1][1],'selection':final.SELECTION,
           'folds_sha256':final.FOLDS_SHA,'excluded_ids_sha256':final.EXCLUDED_SHA}
        final.verify_training_run(r,m,1)
        for delta in ({'labels_sha256':smoke.LABEL_SHA},{'labels_file':'labels_v1_blend.parquet'},
                      {'selection':'gold AUC'},{'epochs_completed':9},{'epochs_requested':11},
                      {'folds_sha256':'f'*64},{'excluded_ids_sha256':'f'*64},{'stopped_by_time_budget':True}):
            with self.assertRaises(ValueError):final.verify_training_run(r|delta,m,1)
        with self.assertRaises(ValueError):final.verify_training_run(r,m,0)
        with self.assertRaises(ValueError):final.verify_training_run(r,m|{'kernel':final.KERNELS[0]},1)

    def test_round1_synthetic_archive_and_final_attach_exact_family(self):
        real_digest=smoke.digest
        with tempfile.TemporaryDirectory() as d,patch.object(smoke,'digest',side_effect=lambda v:FROZEN if v==READER.encode() else real_digest(v)):
            root=Path(d);self.archive(root,1)
            builder.build(root/'new','own-round1-test',rehearsal_build=root,rehearsal_report=root/'report.json',rehearsal_csv=root/'visible.csv')
            meta=json.loads((root/'new/kernel-metadata.json').read_text())
            self.assertEqual(meta['kernel_sources'],final.ROUND_KERNELS[1])
            self.assertFalse(any(k in meta['kernel_sources'] for k in final.KERNELS))
            contract=json.loads((root/'new/build-contract.json').read_text())['rehearsal_contract']
            self.assertEqual(contract['training_round'],1);self.assertEqual(contract['labels_sha256'],final.ROUND1_LABEL_SHA)

if __name__=='__main__':unittest.main()
