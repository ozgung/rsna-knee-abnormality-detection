"""Exercise actual logging blocks without GPU work or private output values."""
import ast
import copy
import inspect
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from knee import fresh_reader
from knee.fresh_reader_setup import progress
from knee.validation import file_sha256


def training_tree():
    return ast.parse(inspect.getsource(fresh_reader.train)).body[0]


def execute(nodes, namespace):
    code = compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[])), '<actual-training-block>', 'exec')
    with patch('builtins.print') as output:
        exec(code, namespace)
    for call in output.call_args_list:
        assert call.kwargs == {'flush': True}
    return [json.loads(call.args[0]) for call in output.call_args_list]


def event_statement(node):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
            isinstance(node.value.func, ast.Name) and node.value.func.id == 'progress')


class TrainingProgressTests(unittest.TestCase):
    def test_real_cache_loop_reports_64_study_cadence_and_no_private_values(self):
        tree = training_tree()
        start = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and
                     any(isinstance(t,ast.Name) and t.id=='records' for t in n.targets))
        end = next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and
                   any(isinstance(t,ast.Name) and t.id=='tr' for t in n.targets))
        with tempfile.TemporaryDirectory(prefix='PRIVATE-PATH-CANARY') as directory:
            root=Path(directory); (root/'arrays').mkdir()
            records=[]; index={}; total=0
            for i in range(135):
                uid=f'PRIVATE-ID-CANARY-{i}'; path=root/'arrays'/f'{uid}.npz'
                data=b'synthetic cache bytes'*(i+1); path.write_bytes(data); total+=len(data)
                records.append(dict(uid=uid,labels=['PRIVATE-LABEL-CANARY']))
                index[uid]=dict(file=path.name,sha256=file_sha256(path),bytes=len(data))
            (root/'cache-index-private.json').write_text(json.dumps(index))
            plan={'records':records}; before=copy.deepcopy(plan); rng=random.getstate()
            namespace=dict(plan=plan,cache_dir=root,json=json,file_sha256=file_sha256,
                           progress=progress,time=SimpleNamespace(monotonic=lambda:10.),
                           time_limit=lambda *a: __import__('contextlib').nullcontext(),
                           remaining_training_budget=fresh_reader.remaining_training_budget,
                           elapsed_before=0.,started=__import__('time').monotonic())
            events=execute(tree.body[start:end],namespace)
            self.assertEqual([e['event'] for e in events],['train_cache_verification_begin',
                'train_cache_verification_progress','train_cache_verification_progress','train_cache_verification_complete'])
            self.assertEqual([e['verified_studies'] for e in events],[0,64,128,135])
            self.assertEqual(events[-1]['verified_bytes'],total)
            self.assertTrue(all(e['total_studies']==135 and e['elapsed_seconds']>=0 for e in events))
            self.assertNotIn('PRIVATE',json.dumps(events)); self.assertEqual(plan,before)
            self.assertEqual(random.getstate(),rng)

    def test_batch_events_use_existing_scalar_trace_even_for_overflow_and_do_not_touch_tensors(self):
        tree=training_tree()
        periodic=next(node for node in ast.walk(tree) if isinstance(node,ast.If) and
                      any(event_statement(child) and child.value.args[0].value=='train_batch_progress' for child in node.body))
        class TensorMustNotBeRead:
            def __getattribute__(self,name): raise AssertionError('Logging read a tensor or GPU method')
        for attempted,expected in ((31,0),(32,1),(64,1)):
            trace=dict(online_training_batches=attempted,successful_optimizer_steps=attempted-1,
                       amp_overflow_skips=1,online_loss_sum=.4*attempted,private='PRIVATE-LABEL-CANARY')
            before=copy.deepcopy(trace); rng=random.getstate()
            namespace=dict(epoch_training_trace=trace,epoch=1,elapsed_before=2.,started=1.,
                           time=SimpleNamespace(monotonic=lambda:5.),progress=progress,
                           batch_loss=TensorMustNotBeRead(),torch=TensorMustNotBeRead(),
                           model=TensorMustNotBeRead(),ema=TensorMustNotBeRead())
            events=execute([periodic],namespace)
            self.assertEqual(len(events),expected); self.assertEqual(trace,before)
            self.assertEqual(random.getstate(),rng); self.assertNotIn('PRIVATE',json.dumps(events))
            if events:
                self.assertEqual(events[0]['attempted_supervised_batches'],attempted)
                self.assertEqual(events[0]['successful_optimizer_steps'],attempted-1)
                self.assertEqual(events[0]['amp_overflow_skips'],1)
                self.assertAlmostEqual(events[0]['online_training_batch_loss_mean'],.4)

    def test_stage_and_checkpoint_events_are_fixed_aggregate_fields_with_flush(self):
        nodes=[node for node in ast.walk(training_tree()) if event_statement(node) and
               node.value.args[0].value in {'gpu_gate_begin','gpu_gate_complete','train_epoch_begin',
                                            'checkpoint_save_begin','checkpoint_save_complete'}]
        namespace=dict(progress=progress,time=SimpleNamespace(monotonic=lambda:10.),tr=[0]*512,va=[0]*200,
            diagnostic_indices=[0]*64,gate_started=1.,gate={'passed':True,'planned_epochs':12,'estimated_epoch_seconds_with_margin':100.},measured=64,epochs=4,
            train_batch_seconds=.25,val_batch_seconds=.1,diagnostic_batch_seconds=.1,epoch_seconds=100.,
            checkpoint_started=7.,epoch=1,elapsed_before=0.,started=0.,
            epoch_training_trace={'online_training_batches':32,'successful_optimizer_steps':31},
            best_payload={'PRIVATE-ID-CANARY':'PRIVATE-LABEL-CANARY'},output_dir=Path('/PRIVATE-PATH-CANARY'))
        events=execute(nodes,namespace)
        self.assertEqual(len(events),9)
        self.assertNotIn('PRIVATE',json.dumps(events))
        saves=[e for e in events if e['event']=='checkpoint_save_complete']
        self.assertEqual({e['kind'] for e in saves},{'best','last'})
        self.assertTrue(all(e['save_seconds']==3. for e in saves))
        self.assertTrue(all(isinstance(v,(str,int,float,bool)) for e in events for v in e.values()))

    def test_resume_runner_checksum_remains_strict(self):
        guard=next(node for node in ast.walk(training_tree()) if isinstance(node,ast.If) and
                   any(isinstance(child,ast.Raise) and isinstance(child.exc,ast.Call) and
                       child.exc.args and isinstance(child.exc.args[0],ast.Constant) and
                       child.exc.args[0].value=='Resume training contract differs' for child in node.body))
        state=dict(plan_fingerprint='synthetic',source_hashes=fresh_reader.SOURCE_HASHES,
                   backbone=fresh_reader.BACKBONE,
                   runner_sha256='072bcd102f3ac231dfe5e19b13f653f9e23c4538b6718c4f298182e9bc38c8ac')
        namespace=dict(state=state,plan={'fingerprint':'synthetic'},SOURCE_HASHES=fresh_reader.SOURCE_HASHES,
                       BACKBONE=fresh_reader.BACKBONE,file_sha256=file_sha256,__file__=fresh_reader.__file__)
        with self.assertRaisesRegex(ValueError,'Resume training contract differs'):
            execute([guard],namespace)
        state['runner_sha256']=file_sha256(fresh_reader.__file__)
        self.assertEqual(execute([guard],namespace),[])


if __name__=='__main__': unittest.main()
