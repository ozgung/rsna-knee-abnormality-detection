import ast
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from knee import public_s6_entry as E
from scripts import build_public_s6_entry as B


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def fixture(self,ids,sample=None):
        for filename,columns,rows in [('test.csv',[E.UID],[[s] for s in ids]),('sample_submission.csv',[E.UID]+E.LABELS,[[s]+[.5]*12 for s in (sample or ids)]),
            ('test_series.csv',E.SERIES_COLUMNS,[[s,'series'+s,'1','0','Sagittal'] for s in ids])]:
            with (self.root/filename).open('w',newline='') as h:w=csv.writer(h);w.writerow(columns);w.writerows(rows)
        for s in ids:
            p=self.root/'test_series'/s/('series'+s);p.mkdir(parents=True);(p/'one.dcm').write_bytes(b'synthetic')

    def shards(self,ids,p,arm='old'):
        result=[];mapping=dict(zip(ids,p))
        for board,assigned in enumerate(E.assignments(ids)):
            path=self.root/f'{arm}{board}.npz'
            ps=np.array([mapping[s] for s in assigned])
            if arm=='old':E.old_ledger(path,assigned,ps)
            else:E.save_predictions(path,assigned,np.zeros((len(assigned),4,12),dtype=np.float32),ps)
            r={'status':'PASS','board':board,'arm':arm,'assigned':assigned,'completed':len(assigned),'state_before':'same','state_after':'same',
               'zero_failed':True,'zero_imputed':True,'fp32_recoveries':0,'ledger_sha256':E.file_sha(path)}
            result.append((r,path))
        return result

    def test_ties_uneven_shards_shuffled_completion_match_single_population(self):
        ids=['z','a','x','b','m'];c=np.tile([.3,.1,.3,.7,.1],(12,1)).T;s=np.tile([.2000001,.2000002,.8,.1,.8],(12,1)).T
        cs=self.shards(ids,c);ss=self.shards(ids,s,'s6')
        # Reverse completion time and dictionary iteration does not change canonical output.
        cs[0][0]['completion_time']=9;cs[1][0]['completion_time']=1
        cm=E.merge(ids,cs,'old');sm=E.merge(ids,ss,'s6')
        expected=.7*pd.DataFrame(c).rank(method='average',pct=True).to_numpy()+.3*pd.DataFrame(E.quantize(s)).rank(method='average',pct=True).to_numpy()
        np.testing.assert_array_equal(E.blend(cm,sm),expected)
        self.assertEqual(E.assignments(ids),[['z','x','m'],['a','b']])

    def test_visible_ranks_exclude_probes(self):
        p=np.tile([.2,.7,.4],(12,1)).T;q=np.tile([.8,.2,.2],(12,1)).T
        np.testing.assert_array_equal(E.blend(p,q),.7*pd.DataFrame(p).rank(pct=True).to_numpy()+.3*pd.DataFrame(E.quantize(q)).rank(pct=True).to_numpy())
        self.assertFalse(np.array_equal(E.blend(p,q),E.blend(np.vstack([np.zeros((32,12)),p]),np.vstack([np.ones((32,12)),q]))[-3:]))

    def test_bad_ledgers_and_worker_failure_rejected(self):
        ids=['one','two','three'];p=np.ones((3,12))*.5
        for field,value in [('status','FAIL'),('status','INTERRUPTED'),('completed',1),('state_after','mutated'),('zero_imputed',False),('fp32_recoveries',1),('ledger_sha256','wrong')]:
            shards=self.shards(ids,p);shards[0][0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):E.merge(ids,shards,'old')
        for got in [['one'],['one','one'],['three','one'],['wrong','three']]:
            shards=self.shards(ids,p);path=shards[0][1]
            with path.open('wb') as h:np.savez(h,study_ids=np.asarray(got),probabilities=np.ones((len(got),12))*.5)
            shards[0][0]['ledger_sha256']=E.file_sha(path)
            with self.subTest(ids=got),self.assertRaises(ValueError):E.merge(ids,shards,'old')

    def test_nonfinite_probability_bounds_and_shape(self):
        for p in [np.full((2,12),np.nan),np.full((2,12),np.inf),np.full((2,12),1.01),np.full((2,12),-.01),np.ones((2,11))]:
            with self.assertRaises(ValueError):E.blend(p,np.ones((2,12))*.5)

    def test_hidden_test_csv_authority_and_no_train_metadata(self):
        ids=['hidden3','hidden1','hidden2','hidden4'];self.fixture(ids,sample=['public1','public2','public3'])
        columns,actual,records=E.test_contract(self.root,E.LABELS);self.assertEqual(actual,ids)
        stage=self.root/'stage';r=E.stage(stage,actual,records,{s:str(self.root/'test_series') for s in ids})
        self.assertEqual(r['studies'],4);self.assertFalse((self.root/'train.csv').exists());self.assertFalse((self.root/'train_series.csv').exists())
        with (stage/'train.csv').open() as h:self.assertEqual(next(csv.reader(h)),[E.UID])
        self.assertEqual(set(p.name for p in stage.iterdir()),{'train.csv','train_series.csv','train_series'})

    def test_strict_metadata_and_target_order(self):
        self.fixture(['a','b','c'])
        text=(self.root/'test_series.csv').read_text();(self.root/'test_series.csv').write_text(text.replace('Sagittal','Unknown',1))
        with self.assertRaises(ValueError):E.test_contract(self.root,E.LABELS)
        (self.root/'test_series.csv').write_text(text)
        with self.assertRaises(ValueError):E.test_contract(self.root,list(reversed(E.LABELS)))

    def test_source_six_decimal_boundary_is_gate(self):
        p=np.ones(12)*.12345649;fp={'input':'same'};cfg={'probes':{'sid':{'s6':{'input':fp,'probabilities':p.tolist()}}}}
        self.assertEqual(E.probe_gate('sid','s6',p,fp,cfg),0)
        with self.assertRaises(ValueError):E.probe_gate('sid','s6',np.ones(12)*.12345651,fp,cfg)
        cfg={'probes':{'sid':{'old':{'input':fp,'probabilities':None}}}}
        self.assertIsNone(E.probe_gate('sid','old',p,fp,cfg))
        with self.assertRaises(ValueError):E.probe_gate('sid','old',p,{'input':'changed'},cfg)

    def test_interrupted_atomic_write_retains_previous(self):
        path=self.root/'ledger.npz';E.old_ledger(path,['id'],np.ones((1,12))*.5);before=path.read_bytes()
        with patch('numpy.savez',side_effect=OSError('Interrupted write')):
            with self.assertRaises(OSError):E.old_ledger(path,['id'],np.ones((1,12))*.2)
        self.assertEqual(before,path.read_bytes());self.assertFalse(path.with_name(path.name+'.tmp').exists())

    def test_failed_rehearsal_cannot_gate_entry(self):
        for r in [{'status':'FAIL'},{'status':'PASS','version':E.VERSION,'mode':'rehearsal','binding':{},'completed':38}, {'status':'INTERRUPTED'}]:
            with self.assertRaises(ValueError):E.require_rehearsal(r,{},self.root/'missing.csv')

    def test_worker_failure_never_emits_csv_and_retains_failure(self):
        self.fixture(['a','b','c']);cfg={'mode':'entry','binding':{},'environment':{}}
        boards='0, GPU-zero, Tesla T4\n1, GPU-one, Tesla T4\n'
        with patch.object(E.subprocess,'check_output',return_value=boards),patch.object(E,'launch',side_effect=RuntimeError('Worker died')):
            with self.assertRaises(RuntimeError):E.run(self.root,cfg,self.root/'run',started=E.time.monotonic())
        r=json.loads((self.root/'run/receipt.json').read_bytes());self.assertEqual(r['status'],'FAIL')
        self.assertTrue((self.root/'run/failure-private.json').exists());self.assertFalse((self.root/'run/visible-predictions.csv').exists())

    def test_projection_uses_distribution_and_concurrent_full_phases(self):
        phases={a:{'wall_seconds':10} for a in ['old','s6']};reports={'old':[{'records':[{'production_seconds':.4}]*17},{'records':[{'production_seconds':.4}]*18}],
            's6':[{'records':[{'windows':100,'production_seconds':.4}]*17},{'records':[{'windows':100,'production_seconds':.4}]*18}]}
        r=E.projection({'development_windows':[550]*870},phases,reports,3,2)
        self.assertEqual(len(r['scenarios']),9);self.assertGreater(r['conservative_seconds'],r['scenarios'][0]['seconds'])
        self.assertEqual(r['window_distribution']['total'],478500)

    def test_fixed_method_and_fresh_spawn_contract(self):
        source=Path(E.__file__).read_text();self.assertIn("'CUDA_VISIBLE_DEVICES':str(board)",source);self.assertIn("sys.executable,'-m','knee.public_s6_entry'",source)
        self.assertNotIn('multiprocessing',source);self.assertEqual(E.METHOD['blend'],[.7,.3]);self.assertEqual(E.RECIPE['chunk'],16);self.assertEqual(E.RECIPE['workers'],0)
        self.assertIn("public.eval_windows(v,m,94,320",source);self.assertNotIn('infer_candidate',source)

    def test_closure_transitive_helpers_excludes_truth(self):
        frozen={'knee/a.py':'import json\nfrom knee.b import helper, truth\nC=2\ndef keep():\n    return helper(C)\ndef score():\n    return truth()\n',
                'knee/b.py':'def helper(x):\n    return x\ndef truth():\n    return "labels.parquet"\n'}
        with patch.dict(B.ROOTS,{'knee/a.py':{'keep'}},clear=True):sources,pins=B.inference_closure(frozen)
        self.assertIn('def helper',sources['knee/b.py']);self.assertNotIn('def truth',str(sources));self.assertNotIn('def score',str(sources))
        self.assertIn('knee/a.py:keep',pins)

    def passing_rehearsal(self):
        pids=['p'+str(i) for i in range(32)];oids=['o'+str(i) for i in range(4)];vids=['v'+str(i) for i in range(3)];ids=pids+oids+vids
        sent={'old':oids,'s6':[pids[i] for i in E.SENTINELS]};environment={'torch':'frozen','packages':{'numpy':'frozen'}}
        binding={'version':E.VERSION,'method':E.METHOD,'s6_probe_ids':pids,'old_historical_ids':oids,'environment':environment,
                 'qualified_state':{'old_state':'old','old_grid':'grid','s6_state':'s6'},
                 'sentinel_inputs':{a:{sid:{'input':'same'} for sid in sent[a]} for a in sent},
                 'sentinel_probabilities':{a:{sid:[.5]*12 for sid in sent[a]} for a in sent}}
        csvpath=self.root/'rehearsal.csv';E.write_csv(csvpath,[E.UID]+E.LABELS,vids,np.ones((3,12))*.5)
        boards=[{'name':'Tesla T4','uuid':'GPU'+str(b)} for b in range(2)];workers={}
        for arm in sent:
            workers[arm]=[]
            for board,assigned in enumerate(E.assignments(ids)):
                state={'state':'old','buffers':'all'} if arm=='old' else 's6'
                workers[arm].append({'status':'PASS','board':board,'assigned':assigned,'completed':len(assigned),'state_before':state,'state_after':state,
                    'constructor_grid_sha256':'grid','zero_failed':True,'zero_imputed':True,'fp32_recoveries':0,'peak_cuda_bytes':1,'peak_rss_bytes':1,
                    'environment':{**environment,'physical_uuid':boards[board]['uuid'],'cuda_device_count':1},
                    'sentinels':[{'id':sid,'strict_reload':True,'input':{'input':'same'},'probabilities':[.5]*12,'reloaded_probabilities':[.5]*12,'logits':np.zeros((4,12)).tolist(),'reloaded_logits':np.zeros((4,12)).tolist()} for sid in sent[arm]],
                    'records':[{'id':sid,'source_reference_parity':True,'strict_reload':True,'reload_max_abs':0,'qualified_error':0} for sid in assigned]})
        r={'status':'PASS','version':E.VERSION,'mode':'rehearsal','binding':binding,'completed':39,'completed_visible':3,'completed_probes':36,'completed_s6_probes':32,
           'completed_old_historical':4,'all_input_state_probability_gates':True,'zero_failed':True,'zero_imputed':True,'boards':boards,'elapsed_seconds':300,
           'projection':{'conservative_seconds':20000},'csv_sha256':E.file_sha(csvpath),'visible_ids':vids,'sentinel_ids':sent,'workers':workers,
           'phases':{a:{'wall_seconds':100,'peaks':{'host_rss_bytes':1,'output_disk_bytes':1},'minimum_disk_free_bytes':1} for a in sent}}
        return r,binding,csvpath

    def test_fresh_worker_pythonpath_from_unrelated_directory(self):
        import os
        import subprocess
        import sys
        env=E.worker_environment(1);self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'1')
        expected=Path(E.__file__).resolve()
        command=[sys.executable,'-c',"from pathlib import Path; import knee.public_s6_entry as e,sys; assert Path(e.__file__).resolve()==Path("+repr(str(expected))+"); assert 'torch' not in sys.modules"]
        r=subprocess.run(command,cwd=self.root,env=env,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(env['PYTHONPATH'].split(os.pathsep)[0],str(expected.parents[1]))
        r=subprocess.run([sys.executable,'-m','knee.public_s6_entry','--help'],cwd=self.root,env=env,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

    def test_qualified384_hash_namespace_and_exact_central320_relation(self):
        import torch
        full=[torch.arange(384*384,dtype=torch.float32).reshape(1,1,384,384) for _ in range(2)]
        plain=[x[:,:,32:352,32:352].clone() for x in full]
        fp=E.window_audit(plain,full=full,reference=[x.clone() for x in plain])
        self.assertNotEqual(fp['old320_windows'],fp['qualified_full384_windows'])
        self.assertNotIn('windows',fp)
        plain[0][0,0,0,0]+=1
        with self.assertRaises(ValueError):E.window_audit(plain,full=full)
        source=Path(B.__file__).read_text();self.assertIn("'qualified_full384_windows':cr['qualification'][i]['windows']",source)

    def test_complete_receipt_passes_and_material_mutations_fail(self):
        r,b,path=self.passing_rehearsal();E.require_rehearsal(r,b,path)
        mutations=[lambda q:q.update(completed=38),lambda q:q['projection'].update(conservative_seconds=E.MAX_SECONDS),
                   lambda q:q['workers']['s6'][1].update(state_after='changed'),lambda q:q['workers']['old'][0].update(constructor_grid_sha256='changed'),
                   lambda q:q['workers']['s6'][0]['sentinels'][0].update(probabilities=[.5000006]*12),
                   lambda q:q['workers']['old'][0]['records'][0].update(source_reference_parity=False),
                   lambda q:q['workers']['s6'][1]['records'][0].update(qualified_error=None)]
        for mutate in mutations:
            q=copy.deepcopy(r);mutate(q)
            with self.assertRaises(ValueError):E.require_rehearsal(q,b,path)

    def test_builder_four_attachments_and_pure_entry_fixture_removal(self):
        self.assertEqual(len(B.DATASETS),4);self.assertFalse(any('hpo-assets' in x for x in B.DATASETS))
        source=Path(B.__file__).read_text();tree=ast.parse(source)
        self.assertIn("'s6_probe_ids','old_historical_ids'",source);self.assertIn('require_rehearsal(rr,binding,rehearsal_csv)',source)
        self.assertIn("signal.alarm(9*3600)",source);self.assertIn("'name':'python3'",source);self.assertIn("lzma.compress",source)


if __name__=='__main__':unittest.main()
