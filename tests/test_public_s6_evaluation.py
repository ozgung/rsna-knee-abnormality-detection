import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from knee import public_s6_evaluation as E
from scripts import build_public_s6_evaluation as B
try:
    import numpy as np
    import pandas as pd
except ImportError:
    np=pd=None


class BindingContracts(unittest.TestCase):
    def population(self):
        ids=[f's{i}' for i in range(870)];folds={s:0 for s in ids}
        pre={'targets':E.LABELS,'fingerprints':{'dev':E.fingerprint(ids),'fold_mapping':E.fingerprint(folds)}}
        return ids,folds,[],['component']*870,pre

    def test_exact_membership_and_order(self):
        args=self.population();E.validate_population(*args)
        ids,folds,ex,groups,pre=args
        for bad in [ids[:-1],list(reversed(ids)),ids[:-1]+[ids[0]]]:
            with self.assertRaises(ValueError):E.validate_population(bad,folds,ex,groups,pre)
        with self.assertRaises(ValueError):E.validate_population(ids,folds,[ids[0]],groups,pre)
        folds[ids[0]]=1
        with self.assertRaises(ValueError):E.validate_population(ids,folds,ex,groups,pre)

    def test_schema_groups_and_qualification_fail_closed(self):
        args=list(self.population());args[3][0]=''
        with self.assertRaises(ValueError):E.validate_population(*args)
        pilot={'source_state':'PASS','mechanics':'PASS','prediction':'PASS','completed_studies':32,'optimizer_updates':0,
               'zero_imputed':True,'zero_failed':True,'fp32_recoveries':0,'recipe':E.RECIPE}
        anatomy={'verdict':'PASS_GROSS_VISIBLE_UNION_SCREEN','development_evaluation_planning_allowed':True}
        E.qualification_gate(pilot,anatomy)
        for key,value in [('zero_failed',False),('fp32_recoveries',1),('optimizer_updates',1),('completed_studies',31)]:
            with self.assertRaises(ValueError):E.qualification_gate({**pilot,key:value},anatomy)
        with self.assertRaises(ValueError):E.qualification_gate(pilot,{**anatomy,'verdict':'UNKNOWN'})

    def test_stream_stage_has_only_id_table_and_selected_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp)/'raw';raw.mkdir();(raw/'train.csv').write_text('StudyInstanceUID,report,ACL\na,SECRET,1\nb,SECRET,0\n')
            with (raw/'train_series.csv').open('w',newline='') as h:
                w=csv.DictWriter(h,fieldnames=E.SERIES_COLUMNS+['secret']);w.writeheader()
                for u in ['a','b']:
                    w.writerow(dict(zip(E.SERIES_COLUMNS,[u,'x','1','0','Sagittal']),secret='SECRET'))
                    p=raw/'train_series'/u/'x';p.mkdir(parents=True);(p/'1.dcm').write_bytes(b'fake')
            staged=Path(tmp)/'staged'
            with patch.object(E,'TRAIN_SHA',E.file_sha(raw/'train.csv')):
                receipt=E.stage_studies(raw,staged,['b'],E.file_sha(raw/'train_series.csv'))
            self.assertEqual(receipt['studies'],1);self.assertEqual((staged/'train.csv').read_text(),'StudyInstanceUID\nb\n')
            self.assertNotIn('SECRET',(staged/'train_series.csv').read_text())
            self.assertEqual(set(p.name for p in staged.iterdir()),{'train.csv','train_series.csv','train_series'})
            self.assertFalse((staged/'train_series/a').exists());self.assertTrue((staged/'train_series/b/x').is_symlink())
            (staged/'weak_labels.parquet').write_bytes(b'forbidden')
            with self.assertRaises(ValueError):E.verify_staged(staged,['b'],receipt)

    def test_pinned_artifact_and_build_damage(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);p=d/'notebook.ipynb';p.write_bytes(b'original')
            sha=E.file_sha(p);self.assertEqual(B.pinned(p,sha),b'original')
            (d/'build-receipt.json').write_text(json.dumps({'files':{'notebook.ipynb':sha}}));B.verify_build(d)
            p.write_bytes(b'damaged')
            with self.assertRaises(ValueError):B.verify_build(d)
            with self.assertRaises(ValueError):B.pinned(p,sha)

    def test_literal_closure_rejects_missing_inventory(self):
        with self.assertRaises(ValueError):B.literals('sources={}',['sources','expected_hashes'])
        self.assertEqual(B.literals('sources={"x":"a\\r\\nb"}',['sources'])['sources']['x'],'a\r\nb')

    def test_child_failure_and_no_automatic_resume(self):
        import sys
        with self.assertRaises(Exception):E.run_child([sys.executable,'-c','raise SystemExit(7)'],{})
        self.assertIn('disabled',E.config()['resume']);self.assertEqual(E.config()['blend'],[.7,.3])


@unittest.skipIf(np is None,'NumPy/pandas unavailable')
class PredictionContracts(unittest.TestCase):
    def arms(self):
        a=np.broadcast_to(np.linspace(0,1,870)[:,None],(870,12)).copy();return a,a[::-1].copy()

    def test_all870_average_ties_and_fixed_weights(self):
        c,s=self.arms();s[:,0]=.3;c[:,1]=.2
        q,cr,sr,b=E.fixed_ranks(c,s)
        np.testing.assert_array_equal(cr,pd.DataFrame(c).rank(method='average',pct=True).to_numpy())
        np.testing.assert_array_equal(sr,pd.DataFrame(q).rank(method='average',pct=True).to_numpy())
        np.testing.assert_array_equal(b,.7*cr+.3*sr)
        self.assertEqual(sr[0,0],435.5/870);self.assertEqual(cr[0,1],435.5/870)
        self.assertEqual(sr.shape,(870,12))

    def test_source_csv_boundary_rounding_is_exact(self):
        c,s=self.arms()
        s[:8,0]=[.12345649,.1234565,.12345651,.50000049,.5000005,.50000051,
                 float(np.float32(.7000005)),float(np.float32(.7000015))]
        q,_,_,_=E.fixed_ranks(c,s)
        source_csv=pd.DataFrame(s).to_csv(index=False,header=False,float_format='%.6f')
        oracle=pd.read_csv(io.StringIO(source_csv),header=None).to_numpy()
        np.testing.assert_array_equal(q,oracle)
        self.assertEqual(q[0,0],q[1,0]);self.assertNotEqual(q[1,0],q[2,0])

    def test_control_keeps_full_precision_and_masks_do_not_filter_rank(self):
        c,s=self.arms();c[:2,0]=[.1+.00000001,.1+.00000002];s[:2,0]=c[:2,0]
        q,cr,sr,b=E.fixed_ranks(c,s)
        self.assertNotEqual(cr[0,0],cr[1,0]);self.assertEqual(sr[0,0],sr[1,0])
        mask=np.zeros(870,bool);mask[:2]=True
        self.assertLess(cr[mask,0].max(),1)
        self.assertEqual(b.shape,(870,12))

    def test_incomplete_nonfinite_invalid_arms_rejected(self):
        c,s=self.arms()
        for a,b in [(c[:-1],s[:-1]),(c[:,:11],s[:,:11]),(c,s*np.nan),(c,s+2)]:
            with self.assertRaises(ValueError):E.fixed_ranks(a,b)

    def test_prediction_atomic_checkpoint_survives_write_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'predictions-private.npz';z=np.zeros((1,4,12));v=np.full((1,12),.5)
            E.save_predictions(p,['a'],z,v);original=p.read_bytes()
            with patch.object(np,'savez',side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):E.save_predictions(p,['a','b'],np.zeros((2,4,12)),np.full((2,12),.5))
            self.assertEqual(p.read_bytes(),original)
            with np.load(p,allow_pickle=False) as ar:self.assertEqual(ar['study_ids'].tolist(),['a'])
            with self.assertRaises(ValueError):E.save_predictions(p,['a'],z*np.nan,v)
            with self.assertRaises(ValueError):E.save_predictions(p,['a','a'],np.zeros((2,4,12)),np.full((2,12),.5))

    def test_worker_failure_preserves_partial_count_and_never_opens_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);raw=root/'raw';raw.mkdir()
            for n in ('train.csv','train_series.csv','sample_submission.csv'):(raw/n).write_bytes(b'synthetic')
            labels=root/'labels';labels.write_bytes(b'weak');masks=root/'masks';masks.write_bytes(b'truth')
            saved=root/'control.npy';np.save(saved,np.zeros((870,12),np.float32))
            ids=[f's{i}' for i in range(870)];folds={s:0 for s in ids}
            binding={'label_sha256':E.file_sha(labels),'truth_sha256':E.file_sha(masks),'schema_sha256':E.file_sha(raw/'sample_submission.csv'),
                     'official_train_sha256':E.file_sha(raw/'train.csv'),'series_sha256':E.file_sha(raw/'train_series.csv'),
                     'saved_control_array_sha256':E.array_sha(np.load(saved))}
            cfg={'out':str(root/'out'),'binding':binding,'binding_sha256':E.digest(binding),'raw':str(raw),'labels':str(labels),'masks':str(masks),
                 'saved_control':str(saved),'dev':ids,'folds':folds,'excluded':[],'groups':['g']*870,
                 'preflight':{'targets':E.LABELS,'fingerprints':{'dev':E.fingerprint(ids),'fold_mapping':E.fingerprint(folds)}},
                 'pilot_receipt':{},'coverage':{},'code':'synthetic','source_pins':{},'compact':'synthetic','anatomy':'synthetic','manifest':{},'pilot_binding':{},
                 'staged':str(root/'stage'),'pilot_sentinels':['train-only']}
            path=root/'config.json';E.save(path,cfg)
            def child(args,env):
                phase=args[-2];out=root/'out'/phase;out.mkdir()
                if phase=='control':E.save(out/'receipt.json',{'status':'PASS'});return
                E.save_predictions(out/'predictions-private.npz',ids[:2],np.zeros((2,4,12)),np.full((2,12),.5))
                E.save(out/'receipt.json',{'status':'FAIL','completed':2,'phase':'evaluation','failure_type':'DecoderError'})
                raise RuntimeError('private identifier not for parent receipt')
            with patch.object(E,'qualification_gate'),patch.object(E,'verify_sources'),patch.object(E,'verify_assets'),patch.object(E,'stage_studies',return_value={}),patch.object(E,'run_child',side_effect=child),patch.object(E,'preflight_inputs') as truth:
                with self.assertRaises(RuntimeError):E.run(path)
            truth.assert_not_called()
            receipt=json.loads((root/'out/receipt.json').read_bytes())
            self.assertEqual(receipt['completed_studies'],2);self.assertEqual(receipt['worker_progress']['s6']['failure_type'],'DecoderError')
            self.assertFalse(receipt['intermediate_metrics_opened']);self.assertNotIn('private identifier',json.dumps(receipt))
            with np.load(root/'out/s6/predictions-private.npz') as ar:self.assertEqual(len(ar['study_ids']),2)

    def test_array_binding_includes_dtype_shape_and_values(self):
        x=np.zeros((2,3),dtype=np.float32)
        self.assertNotEqual(E.array_sha(x),E.array_sha(x.astype(np.float64)))
        self.assertNotEqual(E.array_sha(x),E.array_sha(x.reshape(3,2)))
        self.assertNotEqual(E.array_sha(x),E.array_sha(x+1))

    def truth(self):
        y=np.broadcast_to((np.arange(870)%2)[:,None],(870,12)).astype(float).copy();m=np.ones_like(y,bool)
        pre={'targets':E.LABELS,'support':{t:{'dev_observed':870} for t in E.LABELS}}
        return y,m,pre

    def test_missing_truth_support_is_checked_and_train_only_prevalence(self):
        y,m,pre=self.truth();c,s=self.arms();m[0,0]=False;pre['support'][E.LABELS[0]]['dev_observed']=869
        with patch.object(E,'paired_bootstrap',return_value={'ci95':[-.1,.1]}) as boot:
            r=E.analyze(y,m,np.zeros((3,12)),np.ones((3,12),bool),c,s,c,['g']*870,pre)
        self.assertEqual(r['metrics']['control']['targets'][E.LABELS[0]]['positive'],435)
        self.assertEqual(r['metrics']['control']['targets'][E.LABELS[0]]['negative'],434)
        self.assertEqual(r['metrics']['train_prevalence']['macro_auc'],.5)
        self.assertEqual(boot.call_args.kwargs['draws'],2000);self.assertEqual(boot.call_args.kwargs['seed'],42)
        pre['support'][E.LABELS[0]]['dev_observed']=870
        with self.assertRaises(ValueError):E.analyze(y,m,y,m,c,s,c,['g']*870,pre)

    def test_three_supported_damaged_targets_close_even_with_positive_ci(self):
        y,m,pre=self.truth();c,s=self.arms()
        def metric(auc):return {'macro_auc':auc,'defined_targets':12,'targets':{t:{'positive':435,'negative':435,'auc':(.6 if j<3 else .7)} for j,t in enumerate(E.LABELS)}}
        control=metric(.6);control['targets']={t:{'positive':435,'negative':435,'auc':.65} for t in E.LABELS}
        with patch.object(E,'metric_report',side_effect=[control,metric(.68),metric(.5),metric(.5)]),patch.object(E,'paired_bootstrap',return_value={'ci95':[.01,.05]}):
            r=E.analyze(y,m,y,m,c,s,c,['g']*870,pre)
        self.assertEqual(len(r['target_damage']),3);self.assertIn('closed',r['triage']);self.assertFalse(r['generalization_evidence'])

    def test_positive_ci_all12_no_harm_only_separate_rehearsal(self):
        y,m,pre=self.truth();c,s=self.arms()
        def metric(auc):return {'macro_auc':auc,'defined_targets':12,'targets':{t:{'positive':435,'negative':435,'auc':auc} for t in E.LABELS}}
        with patch.object(E,'metric_report',side_effect=[metric(.6),metric(.7),metric(.5),metric(.5)]),patch.object(E,'paired_bootstrap',return_value={'ci95':[.01,.05]}):
            r=E.analyze(y,m,y,m,c,s,c,['g']*870,pre)
        self.assertEqual(r['triage'],'eligible for separate image-only rehearsal')

    def test_all_named_buffers_bound_including_nonpersistent(self):
        try:import torch
        except ImportError:self.skipTest('Torch unavailable')
        model=torch.nn.Linear(3,2);model.register_buffer('grid',torch.zeros(2),persistent=False)
        before=E.model_digest([model]);model.grid[0]=1
        self.assertNotEqual(before,E.model_digest([model]))
        self.assertNotIn('grid',model.state_dict())


if __name__=='__main__':unittest.main()
