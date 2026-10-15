"""B0′ checkpoint-swap boundaries; synthetic checks are not GPU qualification."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from knee import public_s6_recovery_b0 as B0
from knee import public_s6_recovery_b0prime as P
from scripts import build_public_b0prime as B

CPU=B.PROJECT/'outputs/analysis/b0prime-20261015/cpu-k62.json'
HAS_NUMPY=importlib.util.find_spec('numpy') is not None
HAS_EVIDENCE=HAS_NUMPY and CPU.exists() and (B.RESULT/'receipt.json').exists()


class B0PrimeBoundaries(unittest.TestCase):
    def test_exact_b0_worker_and_blend_bytecode(self):
        for name in ('old_worker','s6_worker','launch','merge','run'):
            with self.subTest(name=name):
                self.assertIs(getattr(P,name).__code__,getattr(B0,name).__code__)
                self.assertIs(getattr(P,name).__globals__,vars(P))
        self.assertEqual(P.RAPTOR_WINDOWS,62)
        self.assertIs(P.recover,B0.recover)
        self.assertEqual(P.METHOD['blend'],[.7,.3])
        self.assertEqual({k for k in P.METHOD if P.METHOD[k]!=B0.METHOD[k]},{'old'})

    def test_source_guards_and_original_modules_stay_frozen(self):
        for name in ('install_raptor_old_guards','public_guard','raptor_windows','raptor_forward'):
            self.assertIs(getattr(P,name),getattr(B0,name))
        for name,pin in B.FROZEN.items():self.assertEqual(B.file_sha(B.PROJECT/name),pin)

    def test_drfull_profile_rejects_original_checkpoint_and_k42(self):
        from knee.reference import RAPTOR_RECIPE
        m={'profile':'raptor','checkpoint_sha256':P.DREAD_CHECKPOINT_SHA256,
           'checkpoint_file':P.CHECKPOINT_FILE,'weight_dataset':P.DATASET,
           'decoded_fastread_sha256':None,'recipe':{**RAPTOR_RECIPE,'windows':62}}
        self.assertEqual(P.raptor_profile_gate(None,m),P.OLD_PROFILE)
        for key,value in [('checkpoint_sha256',B0.DREAD_CHECKPOINT_SHA256),('weight_dataset','wrong')]:
            with self.assertRaises(ValueError):P.raptor_profile_gate(None,{**m,key:value})
        with self.assertRaises(ValueError):P.raptor_profile_gate(None,{**m,'recipe':{**m['recipe'],'windows':42}})

    def test_reader_only_reference_cannot_import_old_probabilities(self):
        counters=dict.fromkeys(P.RAPTOR62_COUNT_KEYS,1)
        cfg={'raptor62_reference':{'synthetic':{'counts':counters}}}
        self.assertEqual(P.raptor62_gate('synthetic',None,counters,cfg),
                         {'decode_counts_equal':True,'checkpoint_probability_reference':False})
        for ref in ({'counts':{**counters,'headers':2}}, {'counts':counters,'probabilities':[.5]*12}):
            with self.assertRaises(ValueError):P.raptor62_gate('synthetic',None,counters,{'raptor62_reference':{'synthetic':ref}})

    def test_projection_gate_and_input_resource_mutations(self):
        rows={'synthetic':{'input':{'hash':'synthetic'},'decode_counters':{'headers':1}}}
        binding={'reader_reference':{a:copy.deepcopy(rows) for a in ('old','s6')}}
        r={'projection':{'conservative_seconds':P.QUALIFICATION_SECONDS},
           'workers':{a:[{'records':[{'id':sid,**v} for sid,v in rows.items()],'peak_rss_bytes':1}] for a in ('old','s6')},
           'phases':{a:{'peaks':{'host_rss_bytes':1}} for a in ('old','s6')}}
        with patch.object(P,'_require_b0prime'):
            P.require_rehearsal(r,binding,'unused')
            q=copy.deepcopy(r);q['projection']['conservative_seconds']+=.001
            with self.assertRaisesRegex(ValueError,'8_3H'):P.require_rehearsal(q,binding,'unused')
            q=copy.deepcopy(r);q['workers']['old'][0]['records'][0]['input']={'hash':'changed'}
            with self.assertRaisesRegex(ValueError,'IMAGE_INPUTS'):P.require_rehearsal(q,binding,'unused')
            q=copy.deepcopy(r);q['phases']['s6']['peaks']['host_rss_bytes']=22e9+1
            with self.assertRaisesRegex(ValueError,'RSS_LIMIT'):P.require_rehearsal(q,binding,'unused')

    def test_entry_requires_real_fresh_rehearsal(self):
        with self.assertRaisesRegex(ValueError,'ENTRY_REQUIRES_FRESH_REHEARSAL'):
            B.build('unused','unused','unused',mode='entry')

    @unittest.skipUnless((B.ARCHIVE/'kernel-metadata.json').exists(),'optional archived metadata unavailable')
    def test_exact_metadata_identity_and_dataset_version(self):
        m=B.metadata('b0prime-test')
        self.assertEqual(m['title'],m['id'].split('/')[1])
        self.assertEqual(m['dataset_sources'][-1],P.DATASET+'/1')
        self.assertNotIn('dreaddevelopment/raptor-knee-widedense',m['dataset_sources'])
        self.assertFalse(m['enable_internet']);self.assertTrue(m['is_private'])
        self.assertEqual(m['machine_shape'],'NvidiaTeslaT4')


@unittest.skipUnless(HAS_NUMPY,'optional numpy unavailable')
class B0PrimeSyntheticContract(unittest.TestCase):
    def setUp(self):
        from test_public_s6_recovery_b0 import passing_rehearsal
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.r,self.binding,self.csv=passing_rehearsal(self.tmp.name)
        b=self.binding;r=self.r
        b.update(version=P.VERSION,method=P.METHOD,old_profile=P.OLD_PROFILE)
        b['qualified_state'].update(old_state=P.DREAD_STATE_SHA256,old_grid=P.DREAD_GRID_SHA256)
        r['version']=P.VERSION
        b['reader_reference']={}
        for arm in ('old','s6'):
            b['reader_reference'][arm]={}
            for w in r['workers'][arm]:
                if arm=='old':
                    w.update(old_profile=P.OLD_PROFILE,state_before={'state':P.DREAD_STATE_SHA256},state_after={'state':P.DREAD_STATE_SHA256})
                for row in w['records']:
                    row.update(input={},decode_counters={'headers':1})
                    b['reader_reference'][arm][row['id']]={'input':{},'decode_counters':{'headers':1}}
                    if 'raptor62' in row:row['raptor62']={'decode_counts_equal':True,'checkpoint_probability_reference':False}

    def test_all_preserved_b0_gates_accept_synthetic_fixture(self):
        P.require_rehearsal(self.r,self.binding,self.csv)

    def test_preserved_state_reload_and_failure_guards_reject_mutations(self):
        def state(r):r['workers']['old'][0]['state_before']={'state':'wrong'}
        def reload(r):r['workers']['old'][0]['sentinels'][0]['strict_reload']=False
        def injection(r):r['workers']['s6'][0]['recovery_injection']['state_healthy']=False
        def reference(r):r['workers']['old'][0]['records'][0]['source_reference_parity']=False
        for change in (state,reload,injection,reference):
            r=copy.deepcopy(self.r);change(r)
            with self.subTest(change=change.__name__),self.assertRaises(ValueError):P.require_rehearsal(r,self.binding,self.csv)

    def test_saved_ledgers_projection_and_blend_replay_rejects_changed_csv(self):
        import csv
        import numpy as np
        from knee.public_s6_recovery_runtime import save_ledger
        root=Path(self.tmp.name)/'saved';root.mkdir()
        r=self.r;b=self.binding;ids=b['s6_probe_ids']+b['old_historical_ids']+r['visible_ids']
        b.update(visible_ids=r['visible_ids'],development_windows=[2]*870)
        p=np.array([.25+.01*i for i in range(12)])
        b['s6_reference']={sid:p.tolist() for sid in ids}
        for arm in ('old','s6'):
            for i,w in enumerate(r['workers'][arm]):
                folder=root/f'board{i}-{arm}'/arm;folder.mkdir(parents=True)
                for row in w['records']:row.update(status='genuine',production_seconds=.01,windows=2,
                    cleanup={'scratch_removed':True,'side_override_removed':True},state_healthy=True)
                logits=np.tile(np.log(p/(1-p)),(len(w['assigned']),4,1)) if arm=='s6' else None
                save_ledger(folder/'predictions.npz',w['assigned'],[p]*len(w['assigned']),['genuine']*len(w['assigned']),logits=logits)
                w.update(version=P.VERSION,arm=arm,genuine=len(w['assigned']),failed=0,ledger_sha256=B.file_sha(folder/'predictions.npz'))
                B.save(folder/'receipt.json',w)
        B.save(root/'assignment.json',{'ids':ids})
        r['setup_seconds']=.1
        r['projection']=P.projection({'development_windows':b['development_windows']},r['phases'],r['workers'],.1,.1)
        path=root/'visible-predictions.csv'
        r['csv_sha256']=P.write_csv(path,['StudyInstanceUID']+P.LABELS,r['visible_ids'],np.full((3,12),2/3))
        B.save(root/'receipt.json',r)
        self.assertTrue(B.verify_result(root,b)['all_four_ledgers_recomputed'])
        with path.open(newline='') as h:rows=list(csv.reader(h))
        rows[1][1]='0.5'
        with path.open('w',newline='') as h:csv.writer(h).writerows(rows)
        r['csv_sha256']=B.file_sha(path);B.save(root/'receipt.json',r)
        with self.assertRaisesRegex(ValueError,'RECOMPUTED_VISIBLE_70_30_BLEND'):B.verify_result(root,b)


@unittest.skipUnless(HAS_EVIDENCE,'optional numpy/private local CPU and B0 evidence unavailable')
class B0PrimeEvidenceChecks(unittest.TestCase):
    def test_cpu_k62_pins_and_no_k42_receipt_substitution(self):
        pack=B.packet(CPU);self.assertEqual(pack['cfg']['binding']['old_qualification']['cpu']['windows'],62)
        old=B.PROJECT/'outputs/analysis/r7-recovery-20261013/cpu-qualification-b3-drfull.json'
        if old.exists():
            with self.assertRaisesRegex(ValueError,'FRESH_CPU_K62'):B.cpu_qualification(old,pack['cfg']['public_manifest'])

    def test_only_new_sources_and_checkpoint_bindings_change(self):
        old,_,_=B.archived();pack=B.packet(CPU)
        self.assertTrue(all(pack['sources'][n]==s for n,s in old['sources'].items()))
        self.assertEqual(set(pack['sources'])-set(old['sources']),{B.RUNTIME,B.EXTENSION})
        a,b=old['cfg'],pack['cfg']
        for name in ('public_source','probes','probe_order','old_sentinels','s6_sentinels','development_windows',
                     'manifest','pilot_state_sha256','source_pins','environment','asset_binding','guard_source_sha256'):
            self.assertEqual(a[name],b[name],name)
        self.assertTrue(all(set(v)=={'counts'} for v in b['raptor62_reference'].values()))

    def test_original_b0_receipt_never_qualifies_as_new_probe(self):
        pack=B.packet(CPU);r=json.loads((B.RESULT/'receipt.json').read_bytes())
        with self.assertRaisesRegex(ValueError,'PASS_EXACT_REHEARSAL_REQUIRED'):
            P.require_rehearsal(r,pack['cfg']['binding'],B.RESULT/'visible-predictions.csv')

    def test_actual_unpacked_runtime_imports_without_checkout_fallback(self):
        import numpy as np
        pack=B.packet(CPU);code=B.bootstrap(pack);tree=ast.parse(code)
        loop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.iter,ast.Call)
                  and ast.unparse(n.iter)=="payload['sources'].items()")
        with tempfile.TemporaryDirectory() as folder:
            project=Path(folder)/'project'
            exec(compile(ast.Module(body=[loop],type_ignores=[]),'<actual-unpack>','exec'),
                 {'payload':pack,'project':project,'code':project/'source'})
            program="""import sys
sys.path.insert(0,sys.argv[1]);sys.path.append(sys.argv[2])
from knee import public_s6_recovery_b0prime as P
assert P.RAPTOR_WINDOWS==62 and P.ENTRY_MODULE.endswith('b0prime')
assert P.old_worker.__globals__ is vars(P)
print('B0PRIME_UNPACKED_IMPORT_PASS')
"""
            r=subprocess.run([sys.executable,'-I','-S','-c',program,str(project),str(Path(np.__file__).parent.parent)],
                             cwd=folder,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr)
            self.assertIn('B0PRIME_UNPACKED_IMPORT_PASS',r.stdout)


if __name__=='__main__':unittest.main()
