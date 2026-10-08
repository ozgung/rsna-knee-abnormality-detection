"""Synthetic identity links, sealed selection, runtime bounds and private packaging."""
from collections import Counter
import ast
import copy
import csv
import importlib.util
import json
import io
import logging
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

from knee import sealed_identity as audit


def write_csv(path, columns, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def fixture(root, count=80):
    targets = ['target' + str(i) for i in range(12)]
    rows = [{'StudyInstanceUID': f'PRIVATE-{i:03}', 'Report': f'PRIVATE REPORT {i}',
             **{t: ('1' if i == count-1 else '') for t in targets}} for i in range(count)]
    series = [{'StudyInstanceUID': row['StudyInstanceUID'], 'SeriesInstanceUID': f'SERIES-{i}-{s}',
               'Fluid_Sensitive': 'True', 'Fat_Suppression': 'True', 'Anatomical_Plane': 'Sagittal'}
              for i, row in enumerate(rows) for s in range(2)]
    write_csv(root/'train.csv', ['StudyInstanceUID', 'Report', *targets], rows)
    write_csv(root/'train_series.csv', ['StudyInstanceUID', 'SeriesInstanceUID', 'Fluid_Sensitive',
                                      'Fat_Suppression', 'Anatomical_Plane'], series)
    write_csv(root/'sample_submission.csv', ['StudyInstanceUID', *targets],
              [{'StudyInstanceUID': 'test', **dict.fromkeys(targets, '.5')}])
    original_records = [dict(uid=r['StudyInstanceUID'], group=audit.report_group(r),
        split='train' if i < 20 else 'weak_validation' if i < 32 else 'gold',
        patient_hashes=[audit.patient_hash('PATIENT-'+str(i))]) for i,r in enumerate(rows) if i < 32 or i == count-1]
    baseline = [{k:r[k] for k in ('uid','split','group')} for r in original_records if r['split'] != 'gold']
    original = audit.stamped(dict(policy=audit.POLICY,records=original_records,targets=targets,study_count=count,
        series_count=count*2, metadata_sha256={n:audit.file_sha256(root/n) for n in audit.METADATA},
        baseline_fingerprint=audit.fingerprint(baseline), original_counts=dict(Counter(r['split'] for r in original_records))))
    for row in series:
        folder=root/'train_series'/row['StudyInstanceUID']/row['SeriesInstanceUID'];folder.mkdir(parents=True)
        (folder/'002.dcm').write_bytes(b'not read by synthetic reader')
        (folder/'001.dcm').write_bytes(b'not read by synthetic reader')
    return original, rows


def synthetic_reader(path, **kwargs):
    assert kwargs == dict(stop_before_pixels=True,specific_tags=['PatientID'])
    assert path.name == '001.dcm'
    i=int(path.parent.parent.name.split('-')[1])
    return {'PatientID': 'PATIENT-'+str(i)}


class SealedIdentityTests(unittest.TestCase):
    def test_normalization_matches_original_conventions(self):
        self.assertEqual(audit.patient_hash('  abc  '),audit.hashlib.sha256(b'knee-patient-v1:abc').hexdigest())
        self.assertEqual(audit.report_group({'Report':' Ａ  B\n', 'StudyInstanceUID':'x'}),
                         audit.hashlib.sha256(b'a b').hexdigest())
        with self.assertRaises(audit.AuditFailure): audit.patient_hash('   ')

    def test_transitive_crosslinks_quarantine_without_reassigning_original(self):
        groups={u:audit.fingerprint(u) for u in ('train','dev','gold','bridge','relative','a','b','c','d')}
        groups['bridge']=groups['dev']
        original=dict(records=[dict(uid='train',split='train',patient_hashes=[audit.patient_hash('P')]),
                               dict(uid='dev',split='weak_validation',patient_hashes=[audit.patient_hash('D')]),
                               dict(uid='gold',split='gold',patient_hashes=[audit.patient_hash('G')])],
                      fingerprint='original',baseline_fingerprint='baseline')
        series=[dict(uid=u,patient_hash=audit.patient_hash(p)) for u,p in
                [('train','P'),('dev','D'),('gold','G'),('bridge','P'),('relative','G'),
                 ('a','A'),('b','A'),('c','C'),('d','C')]]
        before=copy.deepcopy(original)
        manifest=audit.make_manifest(groups,{'gold'},original,series,{'fingerprint':'contract'},target=3)
        by_id={r['uid']:r for r in manifest['records']}
        self.assertEqual(by_id['train']['future_role'],'quarantined')
        self.assertEqual(by_id['train']['quarantine_reasons'],['development'])
        self.assertEqual(by_id['train']['original_split'],'train')
        self.assertEqual(by_id['train']['component'],by_id['dev']['component'])
        self.assertEqual(by_id['relative']['quarantine_reasons'],['gold'])
        self.assertEqual(manifest['sealed_studies'],4);self.assertEqual(manifest['overshoot'],1)
        self.assertEqual(set(manifest['sealed_order']),{'a','b','c','d'})
        self.assertEqual(original,before)
        random.shuffle(series)
        other=audit.make_manifest(dict(reversed(list(groups.items()))),{'gold'},original,series,
                                  {'fingerprint':'contract'},target=3)
        self.assertEqual(other,manifest)

    def test_metadata_exact_coverage_and_original_fingerprint_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original,_=fixture(root)
            groups,gold,keys=audit.load_metadata(root,original)
            self.assertEqual((len(groups),len(gold),len(keys)),(80,1,160))
            for mutation in ('group','split','baseline_fingerprint','series_count'):
                changed=copy.deepcopy(original);changed.pop('fingerprint')
                if mutation in ('group','split'): changed['records'][0][mutation]='changed'
                else: changed[mutation]='changed'
                with self.assertRaises(audit.AuditFailure): audit.load_metadata(root,audit.stamped(changed))
            columns,series=audit.read_csv(root/'train_series.csv')
            series[-1]=series[0]
            write_csv(root/'train_series.csv',columns,series)
            original.pop('fingerprint');original['metadata_sha256']['train_series.csv']=audit.file_sha256(root/'train_series.csv')
            with self.assertRaisesRegex(audit.AuditFailure,'coverage'): audit.load_metadata(root,audit.stamped(original))

    def test_header_only_pilot_receipt_qualification_and_no_private_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original,_=fixture(root)
            out=root/'output'
            with patch.dict('sys.modules',{'pydicom':SimpleNamespace(dcmread=synthetic_reader)}), \
                 patch('importlib.metadata.version',return_value='synthetic'),patch('builtins.print') as output:
                receipt=audit.run_audit(root,original,out)
            logs=''.join(str(call) for call in output.call_args_list)
            self.assertNotIn('PRIVATE',logs);self.assertNotIn('PATIENT',logs);self.assertNotIn('SERIES-',logs)
            self.assertTrue(all(call.kwargs=={'flush':True} for call in output.call_args_list))
            self.assertEqual((receipt['studies'],receipt['series']),(64,128))
            self.assertEqual(receipt['strata'],{'original':{'studies':32,'series':64},
                                               'outside_original':{'studies':32,'series':64}})
            self.assertFalse((out/'sealed-manifest-private.json').exists())
            bundle=dict(receipt=receipt,identity=json.loads((out/'identity-private.json').read_text()))
            groups,gold,keys=audit.load_metadata(root,original);contract=audit.runtime_contract(original)
            self.assertEqual(len(audit.validate_pilot(bundle,contract,groups,original,keys)),128)
            changed=copy.deepcopy(bundle);changed['identity']['series'].pop()
            with self.assertRaises(audit.AuditFailure): audit.validate_pilot(changed,contract,groups,original,keys)
            changed=copy.deepcopy(bundle);changed['receipt']['stage']='full'
            with self.assertRaises(audit.AuditFailure): audit.validate_pilot(changed,contract,groups,original,keys)
            changed_contract=contract|{'fingerprint':'changed'}
            with self.assertRaises(audit.AuditFailure): audit.validate_pilot(bundle,changed_contract,groups,original,keys)

    def test_missing_identity_and_third_party_errors_withhold_private_outputs(self):
        for error in ('empty','unreadable','timeout'):
            with self.subTest(error=error),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);original,_=fixture(root)
                def reader(*args,**kwargs):
                    if error=='empty': return {'PatientID':''}
                    if error=='timeout': raise TimeoutError('PRIVATE-PATH-CANARY')
                    raise OSError('PRIVATE-PATH-CANARY')
                with patch.dict('sys.modules',{'pydicom':SimpleNamespace(dcmread=reader)}), \
                     patch('importlib.metadata.version',return_value='synthetic'),patch('builtins.print') as output:
                    with self.assertRaises(audit.AuditFailure): audit.run_audit(root,original,root/'output')
                files=list((root/'output').iterdir());self.assertEqual([p.name for p in files],['failure-receipt.json'])
                self.assertNotIn('PRIVATE',files[0].read_text()+str(output.call_args_list))

    def test_full_stage_reuses_only_qualified_complete_pilot_and_detects_restamped_manifest_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original,_=fixture(root,count=380)
            with patch.dict('sys.modules',{'pydicom':SimpleNamespace(dcmread=synthetic_reader)}), \
                 patch('importlib.metadata.version',return_value='synthetic'),patch('builtins.print'):
                audit.run_audit(root,original,root/'pilot')
                bundle={k:json.loads((root/'pilot'/name).read_text()) for k,name in
                        [('identity','identity-private.json'),('receipt','receipt.json')]}
                receipt=audit.run_audit(root,original,root/'full',stage='full',pilot=bundle)
            self.assertEqual((receipt['studies'],receipt['series'],receipt['reused_series'],receipt['read_series']),
                             (380,760,128,632))
            manifest=json.loads((root/'full/sealed-manifest-private.json').read_text())
            identity=json.loads((root/'full/identity-private.json').read_text())
            self.assertEqual(manifest['sealed_studies'],300)
            self.assertEqual(len(manifest['records']),380)
            original_weak={r['uid'] for r in original['records'] if r['split']!='gold'}
            self.assertFalse(original_weak & set(manifest['sealed_order']))
            groups,gold,keys=audit.load_metadata(root,original);contract=audit.runtime_contract(original)
            audit.validate_manifest(manifest,groups,gold,original,identity['series'],contract)
            manifest.pop('fingerprint');manifest['records'][0]['future_role']='sealed'
            with self.assertRaises(audit.AuditFailure):
                audit.validate_manifest(audit.stamped(manifest),groups,gold,original,identity['series'],contract)

    def test_expired_budget_never_reads_headers(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(audit,'read_headers') as reader,patch('builtins.print'):
            with self.assertRaises(audit.AuditFailure):
                audit.run_audit(directory,{},directory,started=audit.time.monotonic()-601)
            reader.assert_not_called()
            receipt=json.loads((Path(directory)/'failure-receipt.json').read_text())
            self.assertEqual(receipt['error_code'],'TimeoutError')

    @unittest.skipUnless(importlib.util.find_spec('pydicom'), 'pydicom is optional locally')
    def test_actual_pydicom_requests_no_pixels_and_only_patient_identity(self):
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'train_series'/'study'/'series';folder.mkdir(parents=True)
            meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID='1.2.840.10008.5.1.4.1.1.4';meta.MediaStorageSOPInstanceUID='1.2.3.4'
            ds=FileDataset(str(folder/'001.dcm'),{},file_meta=meta,preamble=b'\0'*128)
            ds.PatientID=' secret ';ds.PatientName='DO NOT RETAIN';ds.PixelData=b'PIXEL-CANARY'
            ds.save_as(folder/'001.dcm',enforce_file_format=True)
            original_read=pydicom.dcmread
            observed=[]
            def read(*args,**kwargs):
                header=original_read(*args,**kwargs);observed.append(header)
                return header
            counts=dict(read_series=0,reused_series=0,header_listing_seconds=0.)
            with patch.object(pydicom,'dcmread',side_effect=read):
                records=audit.read_headers(directory,[('study','series')],counts=counts)
            self.assertEqual(records[0]['patient_hash'],audit.patient_hash('secret'))
            self.assertEqual(list(observed[0].keys()),[pydicom.tag.Tag(0x00100020)])
            self.assertNotIn('PIXEL',json.dumps(records));self.assertNotIn('secret',json.dumps(records))

    def test_labels_never_enter_hash_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original,rows=fixture(root)
            groups,gold,keys=audit.load_metadata(root,original)
            selected=audit.pilot_ids(groups,original)
            for t in original['targets']: rows[-1][t]='0'
            write_csv(root/'train.csv',['StudyInstanceUID','Report',*original['targets']],rows)
            changed=copy.deepcopy(original);changed.pop('fingerprint')
            changed['metadata_sha256']['train.csv']=audit.file_sha256(root/'train.csv');changed=audit.stamped(changed)
            newgroups,newgold,_=audit.load_metadata(root,changed)
            self.assertEqual((newgroups,newgold),(groups,gold))
            self.assertEqual(audit.pilot_ids(newgroups,changed),selected)
            series=[dict(uid=u,patient_hash=audit.patient_hash(u)) for u in groups]
            first=audit.make_manifest(groups,gold,original,series,{'fingerprint':'before'},target=10)
            second=audit.make_manifest(newgroups,newgold,changed,series,{'fingerprint':'after'},target=10)
            self.assertEqual(first['sealed_order'],second['sealed_order'])

    def test_parser_warning_values_are_counted_without_logging_and_logger_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original,_=fixture(root)
            logger=logging.getLogger('pydicom');before=logger.disabled
            def noisy(path,**kwargs):
                logger.warning('PRIVATE RAW VALUE')
                warnings.warn('PRIVATE RAW VALUE')
                return synthetic_reader(path,**kwargs)
            counts=dict(read_series=0,reused_series=0,header_listing_seconds=0.)
            with patch.dict('sys.modules',{'pydicom':SimpleNamespace(dcmread=noisy)}),patch('sys.stderr',new_callable=io.StringIO) as stderr:
                audit.read_headers(root,[('PRIVATE-000','SERIES-0-0')],counts=counts)
            self.assertEqual(stderr.getvalue(),'');self.assertEqual(counts['parser_warnings'],1)
            self.assertEqual(logger.disabled,before)
            with patch.dict('sys.modules',{'pydicom':SimpleNamespace(dcmread=lambda *a,**k: (_ for _ in ()).throw(TimeoutError()))}):
                with self.assertRaises(TimeoutError):audit.read_headers(root,[('PRIVATE-000','SERIES-0-0')],counts=counts)
            self.assertEqual(logger.disabled,before)

    def test_projection_and_output_bounds_and_manifest_tamper(self):
        self.assertEqual(audit.projection(100,1000,10.,5.)['estimated_seconds_with_margin'],155.)
        self.assertFalse(audit.projection(1,1000,3.,1.)['passed'])
        for value in (0.,-1.,float('nan')):
            with self.assertRaises(audit.AuditFailure): audit.projection(1,10,value,1.)
        with tempfile.TemporaryDirectory() as directory,patch.object(audit,'OUTPUT_BYTES',64):
            with self.assertRaises(audit.AuditFailure): audit.write_outputs(directory,{'private.json':{'x':'x'*65}})
            self.assertEqual(list(Path(directory).iterdir()),[])
        value=audit.stamped({'sealed_order':['synthetic']});value['sealed_order'].append('tampered')
        with self.assertRaises(audit.AuditFailure): audit.validate_stamp(value)

    def test_cpu_private_builder_contract_and_notebook_compile(self):
        spec=importlib.util.spec_from_file_location('build_sealed_identity',Path(__file__).parents[1]/'scripts/build_sealed_identity.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original,_=fixture(root)
            with patch.object(builder,'original_contract',return_value=original):
                result=builder.build(root,root,root/'unused',root/'build','knee-identity-test')
            metadata=json.loads((root/'build/kernel-metadata.json').read_text())
            self.assertTrue(metadata['is_private']);self.assertFalse(metadata['enable_gpu']);self.assertFalse(metadata['enable_internet'])
            self.assertEqual(metadata['competition_sources'],['rsna-knee-abnormality-detection'])
            for key in ('dataset_sources','kernel_sources','model_sources'):self.assertEqual(metadata[key],[])
            notebook=json.loads((root/'build/notebook.ipynb').read_text())
            code=''.join(notebook['cells'][-1]['source']);compile(code,'synthetic-notebook','exec')
            # Execute the actual generated outer timer around a blocked mount.
            outer=next(n for n in ast.walk(ast.parse(code)) if isinstance(n,ast.With))
            visited=[]
            def blocked_mount(*args,**kwargs):
                visited.append(True);audit.time.sleep(.3)
            namespace=dict(time=audit.time,_audit_started=audit.time.monotonic()-599.9,
                time_limit=audit.time_limit,json=json,runtime_contract=audit.runtime_contract,
                require=audit.require,Path=Path,exact_root=blocked_mount,run_audit=lambda *a,**k: None)
            with self.assertRaises(TimeoutError):
                exec(compile(ast.Module(body=[outer],type_ignores=[]),'actual-notebook-setup','exec'),namespace)
            self.assertEqual(visited,[True])
            self.assertNotIn('PRIVATE REPORT',code);self.assertNotIn('pip install',code)
            self.assertEqual(result['stage_seconds'],600)
            self.assertEqual(result['notebook_bytes'],(root/'build/notebook.ipynb').stat().st_size)
            with patch.object(builder,'original_contract',return_value=original),patch.object(builder,'KAGGLE_SOURCE_BYTES',100):
                with self.assertRaisesRegex(audit.AuditFailure,'kaggle_source_size_exceeded'):
                    builder.build(root,root,root/'unused',root/'oversize','knee-oversize')
            self.assertFalse((root/'oversize').exists())
            with self.assertRaises(audit.AuditFailure):builder.build(root,root,root/'unused',root/'full','knee-full',stage='full')


if __name__=='__main__': unittest.main()
