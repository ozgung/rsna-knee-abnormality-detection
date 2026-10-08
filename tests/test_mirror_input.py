"""Mirror qualification guards and native/public parity on synthetic files only."""
import ast
import copy
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from knee import mirror_input
from knee.reference import (audit_source, canonical_bytes, load_definitions, load_model,
                            RECIPE, sha256, WEIGHT_SHA256)


def fixture():
    source={'recipe':RECIPE,'checkpoint_sha256':WEIGHT_SHA256}
    policy=mirror_input.policy_contract()
    contract={'source':source,'input_policy':policy,'private':True,'internet':False,'train_count':64,'seed':20261008}
    row={'split':'train','valid_slices':96,'resource_errors':0,'eval_windows':94,'public_pixel_max_error':0,
         'window_max_error':0,'repeat_prediction_max_error':0,'strict_comparison':'PASS','decoder_pixel_max_error':0}
    report={'manifest':source,'manifest_sha256':sha256(canonical_bytes(source)),'input_policy':policy,
        'source_bundle_sha256':'a'*64,'checkpoint_verified_sha256':WEIGHT_SHA256,
        'checkpoint_bytes':mirror_input.CHECKPOINT_BYTES,'status':'passed','prediction_fallbacks':0,
        'sampling':{'train_count':64,'test_count':3,'uses_labels':False,'seed':20261008,'ordered_ids_sha256':'b'*64,'selection':'SHA256(seed:StudyInstanceUID) order'},
        'environment':{'python_major_minor':[3,12],'packages':{'torch':'2.10.0+cu128','timm':'1.0.26'}},
        'strict_comparison_counts':{'PASS':67},
        'stress_checks':{'status':'PASS','cases':dict.fromkeys(mirror_input.STRESS_CASES,'PASS'),'uses_raw_data':False,'uses_labels':False},
        'runtime_seconds':200,'studies':[{**row,'ordinal':i,'split':'test' if i<3 else 'train'} for i in range(67)]}
    return contract,report


class MirrorQualificationTests(unittest.TestCase):
    def test_receipt_requires_new67_source_weight_recipe_environment_and_stress_evidence(self):
        contract, report = fixture()
        mirror_input.require_rehearsal(contract,report,bundle_sha256='a'*64)
        for key,value in [('input_policy',{}),('source_bundle_sha256','b'*64),('checkpoint_verified_sha256','b'*64),
                          ('checkpoint_bytes',1),('status','failed'),('prediction_fallbacks',1),('stress_checks',{}),
                          ('runtime_seconds',float('nan')),('runtime_seconds',mirror_input.MAX_SECONDS+1)]:
            with self.subTest(key=key):
                bad=copy.deepcopy(report);bad[key]=value
                with self.assertRaises(ValueError):mirror_input.require_rehearsal(contract,bad,bundle_sha256='a'*64)
        for key,value in [('eval_windows',42),('public_pixel_max_error',1),('window_max_error',1e-7),
                          ('repeat_prediction_max_error',1e-3),('resource_errors',1),('valid_slices',0)]:
            bad=copy.deepcopy(report);bad['studies'][0][key]=value
            with self.assertRaises(ValueError):mirror_input.require_rehearsal(contract,bad,bundle_sha256='a'*64)
        partial=copy.deepcopy(report);partial['studies'][0].update(strict_comparison='ineligible_partial_input',decoder_pixel_max_error=None)
        partial['strict_comparison_counts']={'PASS':66,'ineligible_partial_input':1}
        mirror_input.require_rehearsal(contract,partial,bundle_sha256='a'*64)
        for row in partial['studies']:row.update(strict_comparison='ineligible_partial_input',decoder_pixel_max_error=None)
        with self.assertRaises(ValueError):mirror_input.require_rehearsal(contract,partial,bundle_sha256='a'*64)

    def test_legacy24_receipt_and_changed_recipe_cannot_qualify(self):
        c,r=fixture();r['sampling']['train_count']=24
        with self.assertRaises(ValueError):mirror_input.require_rehearsal(c,r,bundle_sha256='a'*64)
        c,r=fixture();r['sampling']['ordered_ids_sha256']=''
        with self.assertRaises(ValueError):mirror_input.require_rehearsal(c,r,bundle_sha256='a'*64)
        c,r=fixture();r['sampling']['seed']=0
        with self.assertRaises(ValueError):mirror_input.require_rehearsal(c,r,bundle_sha256='a'*64)
        c,r=fixture();c['source']=copy.deepcopy(c['source']);c['source']['recipe']['input_res']=384
        with self.assertRaises(ValueError):mirror_input.require_rehearsal(c,r,bundle_sha256='a'*64)


@unittest.skipUnless(os.environ.get('KNEE_REFERENCE_SOURCE') and importlib.util.find_spec('torch') and
                     importlib.util.find_spec('pydicom') and importlib.util.find_spec('cv2'),
                     'requires pinned source and optional ML/OpenCV environment')
class MirrorSourceTests(unittest.TestCase):
    def source(self):return Path(os.environ['KNEE_REFERENCE_SOURCE']).read_text()
    def objects(self):
        m,d,c=audit_source(self.source());return (*load_definitions(m,d,c),m,d,c)

    def test_decoder_ast_guard_restores_to_identical_scientific_source(self):
        public,decoder,module,source,manifest=self.objects()
        protected,receipt=mirror_input.protect_decoder(source)
        self.assertEqual(receipt,manifest['decoder_exception_policy']);self.assertEqual(receipt['catch_count'],20)
        self.assertEqual(receipt['cleanup_regions'],1)
        tree=ast.parse(protected)
        for h in (n for n in ast.walk(tree) if isinstance(n,ast.ExceptHandler)):
            self.assertEqual(h.name,'_mirror_caught');h.name=None
            prefixes=[n for n in h.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and
                      isinstance(n.value.func,ast.Name) and n.value.func.id=='_mirror_propagate_fatal']
            self.assertEqual(len(prefixes),1);h.body.remove(prefixes[0])
        self.assertEqual(ast.dump(tree),ast.dump(ast.parse(source)))
        with self.assertRaisesRegex(ValueError,'Unreviewed'):mirror_input.protect_decoder(source+'\n# changed')
        bad=copy.deepcopy(manifest);bad['decoder_exception_policy']['sha256']='0'*64
        with self.assertRaises(ValueError):load_definitions(module,source,bad)
        self.assertEqual(public.INPUT_RES,320);self.assertEqual(public.K_EVAL,94);self.assertEqual(public.MAXS,96)

    def test_actual_native_decoder_all_corrupt_missing_and_resource_stresses(self):
        import pydicom
        public,decoder,*_=self.objects();before=(pydicom.dcmread,decoder._records,decoder._result,decoder._Ctx.px,decoder.fast_hdr)
        report=mirror_input.stress_inputs(public,decoder)
        self.assertEqual(report['cases'],dict.fromkeys(mirror_input.STRESS_CASES,'PASS'))
        self.assertFalse(report['uses_labels']);self.assertFalse(report['uses_raw_data'])
        self.assertEqual(before,(pydicom.dcmread,decoder._records,decoder._result,decoder._Ctx.px,decoder.fast_hdr))

    def test_actual_resource_failure_is_counted_and_header_fd_closed(self):
        public,decoder,*_=self.objects();opened=[];original=decoder._fopen
        def track(*args,**kwargs):
            fd=original(*args,**kwargs);opened.append(fd);return fd
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'synthetic'/'series';p.mkdir(parents=True);(p/'one.dcm').touch()
            rows={'synthetic':[dict(SeriesInstanceUID='series',Anatomical_Plane='Sagittal',Fluid_Sensitive=1)]}
            for kind in (MemoryError,TimeoutError):
                opened.clear()
                with patch.object(decoder,'_fopen',track),patch.object(decoder,'_pread',side_effect=kind('synthetic')):
                    with self.assertRaises(kind) as caught:mirror_input.prepare_study(public,decoder,'synthetic',rows,root)
                self.assertEqual(caught.exception.mirror_input_counts['resource_errors'],1)
                self.assertEqual(len(opened),1)
                with self.assertRaises(OSError):os.fstat(opened[0])

    def test_checkpoint_bytes_fail_before_torch_deserialization(self):
        import torch
        public,_,_,_,manifest=self.objects()
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'synthetic.pt';p.write_bytes(b'not checkpoint')
            with patch.dict(sys.modules,{'timm':ModuleType('timm')}),patch.object(torch,'load') as load:
                with self.assertRaisesRegex(ValueError,'hash mismatch'):load_model(public,p,'cpu',manifest)
                load.assert_not_called()

    def test_real_builder_is_offline67_native320_publisher_bound_and_codec_only(self):
        directory=Path(__file__).resolve().parents[1]/'scripts'
        with patch.object(sys,'path',[str(directory),*sys.path]):
            spec=importlib.util.spec_from_file_location('mirror_builder',directory/'build_reference_rehearsal.py')
            builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);publisher=root/'checkpoint-manifest.json'
            publisher.write_text(json.dumps({'accuracy':{'filename':builder.audit_source(self.source())[2]['checkpoint_file'],'sha256':WEIGHT_SHA256}}))
            for kwargs in ({'train_count':24},{'publisher_manifest':None}):
                args={'publisher_manifest':publisher,**kwargs}
                with self.assertRaises(ValueError):builder.build_reference(os.environ['KNEE_REFERENCE_SOURCE'],root/'bad','synthetic',**args)
            receipt=builder.build_reference(os.environ['KNEE_REFERENCE_SOURCE'],root/'valid','synthetic',publisher_manifest=publisher)
            c=json.loads((root/'valid/reference-contract.json').read_text());m=json.loads((root/'valid/kernel-metadata.json').read_text())
            self.assertEqual(c['train_count'],64);self.assertEqual(c['max_seconds'],32400)
            self.assertEqual(c['source']['recipe'],RECIPE);self.assertEqual(c['input_policy'],mirror_input.policy_contract())
            self.assertEqual(m['kernel_sources'],[]);self.assertEqual(m['model_sources'],[])
            self.assertEqual(m['dataset_sources'],[builder.WEIGHT_DATASET,builder.CONVNEXT_DATASET])
            self.assertFalse(m['enable_internet']);self.assertTrue(m['is_private'])
            self.assertEqual(m['docker_image'],builder.DOCKER_IMAGE)
            self.assertLess(receipt['notebook_bytes'],900000)
            nb=json.loads((root/'valid/notebook.ipynb').read_text())
            for cell in nb['cells']:
                if cell['cell_type']=='code':compile(''.join(cell['source']),'<mirror-build>','exec')


if __name__=='__main__':unittest.main()
