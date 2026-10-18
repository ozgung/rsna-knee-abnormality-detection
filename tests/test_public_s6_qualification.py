"""Synthetic integrity contracts; real CUDA/anatomy remains a separate pilot."""
import csv
import ast
import contextlib
import io
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from knee import public_s6_qualification as S


class S6SetupContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('s6_setup_contract_test',cls.root/'scripts/build_public_s6_qualification.py')
        cls.builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.builder)

    def test_codec_filenames_match_successfully_rehearsed_pins(self):
        tree=ast.parse((self.root/'knee/reference_blend.py').read_text())
        published=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODEC_WHEELS' for t in n.targets))
        self.assertEqual(tuple(self.builder.CODECS),published)
        tree=ast.parse((self.root/'knee/coverage_inference.py').read_text())
        hash_table=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODECS' for t in n.targets))
        self.assertEqual(self.builder.CODECS,hash_table)

    def test_large_competition_tree_never_enumerated_modern_and_legacy(self):
        with TemporaryDirectory() as temp:
            p=Path(temp);raw=p/'competitions/rsna-knee-abnormality-detection';raw.mkdir(parents=True)
            (raw/'sample_submission.csv').write_text('synthetic')
            # A large unrelated image corpus containing decoy matching filenames.
            for i in range(256):
                images=raw/'train_series'/str(i)/'series';images.mkdir(parents=True)
                (images/'sample_submission.csv').write_text('decoy')
                (images/'anatomy.pt').write_bytes(b'decoy')
            for modern in (True,False):
                mount=(p/'datasets/heliosli/rsna-knee-compact-v1' if modern else p/'rsna-knee-compact-v1')
                mount.mkdir(parents=True);(mount/'manifest.json').write_text('{}')
                discovery=[]
                with patch.object(Path,'rglob',side_effect=AssertionError('recursive scan forbidden')),patch.object(Path,'glob',side_effect=AssertionError('enumeration forbidden')),patch.object(Path,'iterdir',side_effect=AssertionError('enumeration forbidden')):
                    self.assertEqual(self.builder.mounted_root(p,'rsna-knee-abnormality-detection',['sample_submission.csv'],discovery,competition=True),raw)
                    self.assertEqual(self.builder.mounted_root(p,'heliosli/rsna-knee-compact-v1',['manifest.json'],discovery),mount)
                self.assertEqual(len(discovery),2)
                (mount/'manifest.json').unlink()

    def test_duplicate_partial_missing_or_unscoped_assets_fail_closed(self):
        with TemporaryDirectory() as temp:
            p=Path(temp);a=p/'datasets/heliosli/rsna-knee-compact-v1';b=p/'rsna-knee-compact-v1'
            for root in (a,b):root.mkdir(parents=True);(root/'manifest.json').write_text('{}')
            (a/'model.pt').write_bytes(b'synthetic')
            discovery=[]
            with self.assertRaises(ValueError):self.builder.mounted_root(p,'heliosli/rsna-knee-compact-v1',['manifest.json','model.pt'],discovery)
            self.assertEqual(len(discovery[0]['probes']),2)
            (b/'manifest.json').unlink();(a/'model.pt').unlink()
            with self.assertRaises(ValueError):self.builder.mounted_root(p,'heliosli/rsna-knee-compact-v1',['manifest.json','model.pt'],[])
            with self.assertRaises(ValueError):self.builder.mounted_root(p,'heliosli/rsna-knee-compact-v1',['../outside'],[])
            with self.assertRaises(ValueError):self.builder.mounted_root(p,'../rsna-knee-compact-v1',['manifest.json'],[])

    def test_generated_setup_handler_keeps_private_detail_and_preserves_receipt(self):
        namespace={'Path':Path,'require':S.require,'save':S.save,'json':json,'re':__import__('re')}
        exec(compile(self.builder.setup_handler_source(),'<generated-setup>','exec'),namespace)
        builder_tree=ast.parse((self.root/'scripts/build_public_s6_qualification.py').read_text())
        build=next(n for n in builder_tree.body if isinstance(n,ast.FunctionDef) and n.name=='build')
        template=next(n.value for n in build.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bootstrap' for t in n.targets))
        rendered=eval(compile(ast.Expression(template),'<bootstrap-template>','eval'),dict(vars(self.builder),encoded='',binding={'source_bundle_sha256':'synthetic'},hashes={},manifest={},ids=[]))
        handler=next(n for n in ast.parse(rendered).body if isinstance(n,ast.Try))
        # Execute the exact generated except block; replace only the setup body.
        handler.body=ast.parse("raise ValueError('sensitive path /private/study-example and original cause')").body
        handler_code=compile(ast.fix_missing_locations(ast.Module(body=[handler],type_ignores=[])),'<generated-failure>','exec')
        with TemporaryDirectory() as temp:
            out=Path(temp);discovery=[{'reference':'synthetic','probes':[]}]
            for preserve in (False,True):
                if preserve:S.save(out/'receipt.json',{'failure_stage':'actual_model_gate','original':True})
                stdout=io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    namespace.update(out=out,phase='codec_discovery',discovery=discovery,binding={'pin':'unchanged'})
                    with self.assertRaisesRegex(RuntimeError,'preserve receipt'):
                        exec(handler_code,namespace)
                detail=json.loads((out/'setup-error-private.json').read_bytes())
                self.assertIn('original cause',detail['message']);self.assertIn('ValueError',detail['traceback'])
                self.assertEqual(detail['discovery'],discovery)
                summary=json.loads(stdout.getvalue());self.assertEqual(summary['setup_phase'],'codec_discovery')
                self.assertNotIn('sensitive',stdout.getvalue());self.assertNotIn('/private/',stdout.getvalue())
                receipt=json.loads((out/'receipt.json').read_bytes())
                if preserve:self.assertEqual(receipt,{'failure_stage':'actual_model_gate','original':True})
                else:self.assertEqual(receipt['diagnostic_file'],'setup-error-private.json')


class S6QualificationContracts(unittest.TestCase):
    def test_membership_rejects_changed_order_dev_gold_quarantine(self):
        ids=['synthetic-%02d'%i for i in range(32)];folds={u:1 for u in ids};excluded={'quarantined_study_ids':[]}
        S.validate_membership(ids,folds,excluded,S.digest(ids))
        for badfold in (0,-1):
            with self.subTest(fold=badfold),self.assertRaises(ValueError):
                S.validate_membership(ids,{**folds,ids[0]:badfold},excluded,S.digest(ids))
        with self.assertRaises(ValueError):S.validate_membership(ids[::-1],folds,excluded,S.digest(ids))
        with self.assertRaises(ValueError):S.validate_membership(ids,folds,{'quarantined_study_ids':[ids[2]]},S.digest(ids))
        with self.assertRaises(ValueError):S.validate_membership(ids[:-1]+[ids[0]],folds,excluded,S.digest(ids))

    def test_id_path_traversal_rejected(self):
        ids=['synthetic-%02d'%i for i in range(31)]+['../outside']
        with self.assertRaises(ValueError):S.validate_membership(ids,{u:1 for u in ids},{'quarantined_study_ids':[]},S.digest(ids))

    def test_label_blind_staging_exact_selected_series_symlinks(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'raw';raw.mkdir();ids=['synthetic-%02d'%i for i in range(32)]
            # Deliberately invalid label CSV body: the stage reads its hash, never target values.
            (raw/'train.csv').write_text('StudyInstanceUID,secret_report,target\nnot-an-input\n')
            rows=[]
            for u in ids+['unselected']:
                s='series-'+u;rows.append(dict(zip(S.SERIES_COLUMNS,[u,s,'1','0','Sagittal'])))
                p=raw/'train_series'/u/s;p.mkdir(parents=True);(p/'1.dcm').write_bytes(b'synthetic')
            with (raw/'train_series.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=S.SERIES_COLUMNS+['secret_report']);w.writeheader();w.writerows(rows)
            stage=root/'stage'
            with patch.object(S,'TRAIN_SHA',S.file_sha(raw/'train.csv')):
                receipt=S.stage_inputs(raw,stage,ids,S.file_sha(raw/'train_series.csv'))
            S.verify_staged(stage,ids,receipt)
            self.assertEqual(receipt['studies'],32);self.assertEqual(receipt['series'],32)
            self.assertNotIn('secret_report',(stage/'train_series.csv').read_text())
            self.assertFalse((stage/'train_series/unselected').exists())
            (stage/'weak-labels.csv').write_text('forbidden')
            with self.assertRaises(ValueError):S.verify_staged(stage,ids,receipt)

    def test_staged_schema_or_resolved_path_tampering_rejected(self):
        with TemporaryDirectory() as temp:
            p=Path(temp);(p/'train_series').mkdir();(p/'train.csv').write_text('StudyInstanceUID,target\na,1\n')
            with self.assertRaises(ValueError):S.verify_staged(p,['a'],{})

    def test_official_schema_order_must_be_exact(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)/'sample.csv';p.write_text(','.join([S.UID]+S.LABELS)+'\n')
            self.assertEqual(S.schema_header(p),[S.UID]+S.LABELS)
            p.write_text(','.join([S.UID]+S.LABELS[::-1])+'\n')
            with self.assertRaises(ValueError):S.schema_header(p)

    def test_sentinels_first_two_largest_counts_deduplicated(self):
        counts=[100,90]+[2]*30;self.assertEqual(S.sentinels(counts),[0,1,2,3])
        counts[18]=101;counts[29]=102;self.assertEqual(S.sentinels(counts),[0,1,29,18])
        with self.assertRaises(ValueError):S.sentinels(counts[:-1])
        with self.assertRaises(ValueError):S.sentinels([0]+counts[1:])

    def test_postcrop_geometry_excludes_letterbox_padding(self):
        g=SimpleNamespace(size=384,crop_xyxy=(0,0,192,192),valid_xyxy=(0,0,384,384))
        self.assertEqual(S.retained_rectangle(g),[8.,8.,184.,184.])
        g=SimpleNamespace(size=384,crop_xyxy=(10,20,210,120),valid_xyxy=(0,96,384,288))
        self.assertEqual(S.retained_rectangle(g),[10+16*200/384,20.,10+368*200/384,120.])
        g=SimpleNamespace(size=384,crop_xyxy=(0,0,1,1),valid_xyxy=(0,0,384,10))
        self.assertIsNone(S.retained_rectangle(g))

    def test_physical_order_right_side_duplicate_short_and_thin_slab(self):
        p,g=S.slab_positions({'positions':[7.,3.5,3.5,0.],'plane':'Sagittal','side':'L'})
        self.assertEqual(g,[[3],[1],[2],[0]]);self.assertEqual(p,[0.,3.5,3.5,7.])
        p,g=S.slab_positions({'positions':[0.,3.5,7.],'plane':'Sagittal','side':'R'})
        self.assertEqual(p,[-7.,-3.5,0.]);self.assertEqual(g,[[2],[1],[0]])
        p,g=S.slab_positions({'positions':[4.],'plane':'Axial','side':'R'})
        self.assertEqual((p,g),([4.],[[0]]))
        p,g=S.slab_positions({'positions':list(range(8)),'plane':'Axial','side':'L'})
        self.assertEqual(p,[1.5,5.5]);self.assertEqual(g,[list(range(4)),list(range(4,8))])

    def test_assets_all_hashed_before_any_deserialization(self):
        with TemporaryDirectory() as temp:
            p=Path(temp);names=['E_lrprior_s0_f%d.pt'%i for i in range(4)]+['side_cls.pt','state_spec.json']
            for name in names:(p/name).write_bytes(b'synthetic-'+name.encode())
            (p/'anatomy.pt').write_bytes(b'anatomy');manifest={**S.RECIPE,'models':names[:4],'sha256':{n:S.file_sha(p/n) for n in names},'anchor_weights_sha256':S.file_sha(p/'anatomy.pt')}
            S.save(p/'manifest.json',manifest);binding={'manifest_sha256':S.file_sha(p/'manifest.json')}
            S.verify_assets(p,p/'anatomy.pt',manifest,binding)
            (p/'side_cls.pt').write_bytes(b'tampered')
            with self.assertRaises(ValueError):S.verify_assets(p,p/'anatomy.pt',manifest,binding)

    def test_complete_source_closure_and_tampering(self):
        with TemporaryDirectory() as temp:
            p=Path(temp);pins={}
            for i in range(194):
                name='training/vendor_runtime/monai060/module%d.py'%i;target=p/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text('x=1\n');pins[name]=S.file_sha(target)
            S.verify_sources(p,pins)
            (p/'unexpected.py').write_text('x=1\n')
            with self.assertRaises(ValueError):S.verify_sources(p,pins)
            (p/'unexpected.py').unlink();(p/next(iter(pins))).write_text('x=2\n')
            with self.assertRaises(ValueError):S.verify_sources(p,pins)

    def test_atomic_json_failure_preserves_completed_receipt(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)/'receipt.json';S.save(p,{'completed':1})
            with self.assertRaises(ValueError):S.save(p,{'value':float('nan')})
            self.assertEqual(json.loads(p.read_bytes()),{'completed':1})

    def test_published_pins_cannot_be_replaced_by_mutable_audit_contract(self):
        path=Path(__file__).resolve().parents[1]/'scripts/build_public_s6_qualification.py'
        spec=importlib.util.spec_from_file_location('s6_pins_contract_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with TemporaryDirectory() as temp:
            p=Path(temp);nb=p/'rsna-knee-v2-velciraptor-dinosaur-speed.ipynb'
            S.save(nb,{'cells':[{'cell_type':'code','source':['SOURCE_SHA={"reader.py":"published"}\nCODE_SHA={"native.py":"original"}\n']}]})
            contract={'constants':{'SOURCE_SHA':{'reader.py':'published'},'CODE_SHA':{'native.py':'original'}}}
            with patch.object(module,'NOTEBOOK_SHA',S.file_sha(nb)):
                self.assertEqual(module.published_pins(p,contract),contract['constants'])
                contract['constants']['SOURCE_SHA']['reader.py']='tampered'
                with self.assertRaises(ValueError):module.published_pins(p,contract)
            with self.assertRaises(ValueError):module.published_pins(p,contract)


    def test_builder_requires_complete_collection_and_immutable_artifacts(self):
        path=Path(__file__).resolve().parents[1]/'scripts/build_public_s6_qualification.py'
        spec=importlib.util.spec_from_file_location('s6_builder_contract_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with TemporaryDirectory() as temp:
            p=Path(temp);S.save(p/'collection.json',{'status':'running'})
            with self.assertRaises(ValueError):module.collect_sources(p,p,p/'collection.json')
            (p/'notebook.ipynb').write_text('{}');S.save(p/'build-receipt.json',{'files':{'notebook.ipynb':S.file_sha(p/'notebook.ipynb')}})
            module.verify_build(p);(p/'notebook.ipynb').write_text('changed')
            with self.assertRaises(ValueError):module.verify_build(p)


if __name__=='__main__':unittest.main()
