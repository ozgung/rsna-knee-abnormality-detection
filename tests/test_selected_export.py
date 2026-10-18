import copy
import ast
import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from knee.selected_export import check_export, sha256_file, validate_expectation


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch unavailable')
class SelectedExportTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = dict(arch='synthetic', res=32, k=2, k_eval=3, bs=1, accum=1,
                           epochs=3, topk=3, seed=42, grad_ckpt=True, bb_lr=1e-5,
                           head_lr=1e-3, wd=.02, ckpt='timm', win_chunk=0,
                           oof_every=4, fp16=False, xla=False, limit=0,
                           freeze_blocks=0, norm='imagenet', smoke=False, folds=5,
                           fold=2, tag='fold2', roi=False, roi_mode='compartment',
                           roi_pad=.06, roi_overlap=.12)
        self.current = dict(script_sha256='a'*64, labels_sha256='b'*64,
                            oof_mask_sha256='c'*64, fold_file_sha256='d'*64,
                            excluded_ids_sha256='e'*64,
                            corpus_ids_sha256={'all_ids.npy':'f'*64,'extra_ids.npy':'1'*64})
        self.labels = ['synthetic0', 'synthetic1']
        topk=[]
        for epoch, value in [(2, 7.), (0, -2.), (1, 4.)]:
            topk.append(dict(epoch=epoch, sel=.9-epoch*.01, arch='synthetic', res=32,
                             lab=self.labels, model={'weight':torch.tensor([value,value/3]),
                             'counter':torch.tensor(epoch)}))
        self.resume = dict(identity=dict(fold=2,n_folds=5,tag='fold2',
                           config=self.config,epochs_requested=3,**self.current),
                           epoch=2,hist=[{'ep':e} for e in range(3)],topk=topk,elapsed=33.)
        # Source arithmetic differs from the independently stacked oracle.
        self.selected = dict(model={'weight':sum(t['model']['weight'].double() for t in topk).div(3).float(),
                             'counter':topk[0]['model']['counter'].clone()},
                             arch='synthetic',res=32,lab=self.labels,swa_over=[2,0,1])
        self.rp=self.root/'resume.pt'; self.sp=self.root/'selected.pt'
        self.save()
        self.expected=dict(dataset_version=7,resume_sha256=sha256_file(self.rp),fold=2,config=copy.deepcopy(self.config))

    def save(self):
        self.torch.save(self.resume,self.rp); self.torch.save(self.selected,self.sp)

    def run_check(self):
        return check_export(self.rp,self.sp,self.expected,self.current,self.labels)

    def test_all_tensors_and_training_metadata_are_bound(self):
        r=self.run_check()
        self.assertEqual((r['status'],r['tensor_count'],r['tensor_cells']),('PASS',2,3))
        self.assertEqual(r['selected_epochs'],[2,0,1])
        self.assertEqual(r['training_config'],self.config)
        self.assertEqual(r['training_wall_seconds'],33.)
        self.assertEqual(r['selected_tensor_sha256'],self.run_check()['selected_tensor_sha256'])

    def test_wrong_source_sha_fails_before_load(self):
        self.expected['resume_sha256']='0'*64
        with mock.patch.object(self.torch,'load',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(ValueError,'mounted resume SHA'):
                self.run_check()

    def test_wrong_fold_config_or_labels_fail(self):
        original=copy.deepcopy(self.resume)
        for change,pattern in [('fold','identity mismatch'),('config','config mismatch'),('labels','target order')]:
            with self.subTest(change=change):
                self.resume=copy.deepcopy(original)
                if change=='fold':self.resume['identity']['fold']=1
                elif change=='config':self.resume['identity']['config']['k']=9
                else:self.resume['topk'][0]['lab']=list(reversed(self.labels))
                self.save(); self.expected['resume_sha256']=sha256_file(self.rp)
                with self.assertRaisesRegex(ValueError,pattern):self.run_check()

    def test_changed_export_tensor_is_rejected(self):
        self.selected['model']['weight'][0]+=0.01;self.save()
        with self.assertRaisesRegex(ValueError,'tensor equality failed'):self.run_check()

    def test_nonfinite_topk_is_rejected(self):
        self.resume['topk'][0]['model']['weight'][0]=float('nan');self.save()
        self.expected['resume_sha256']=sha256_file(self.rp)
        with self.assertRaisesRegex(ValueError,'nonfinite selected'):self.run_check()

    def test_missing_key_integer_buffer_and_epochs_are_checked(self):
        for change in ['key','counter','epochs']:
            with self.subTest(change=change):
                old=copy.deepcopy(self.selected)
                if change=='key':self.selected['model'].pop('weight')
                elif change=='counter':self.selected['model']['counter']+=1
                else:self.selected['swa_over']=[0,1,2]
                self.save()
                with self.assertRaises(ValueError):self.run_check()
                self.selected=old

    def test_incomplete_history_never_qualifies(self):
        self.resume['hist'].pop();self.save();self.expected['resume_sha256']=sha256_file(self.rp)
        with self.assertRaisesRegex(ValueError,'history incomplete'):self.run_check()

    def test_reviewed_selector_reload_and_second_inference_selection_match(self):
        from scripts.build_tracka_export_kaggle import SRC_DIR
        if not (SRC_DIR/'tracka_entry.py').exists():
            self.skipTest('optional reviewed selector artifact unavailable')
        tree=ast.parse((SRC_DIR/'tracka_entry.py').read_text())
        selector=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_checkpoint')
        ns={'Path':Path,'json':json}
        exec(compile(ast.Module(body=[selector],type_ignores=[]),'reviewed-selector','exec'),ns)
        self.torch.save(self.resume,self.root/'raptor_resume_fold2.pt')
        self.expected['resume_sha256']=sha256_file(self.root/'raptor_resume_fold2.pt')
        _,kind,epochs,path,_,_=ns['select_checkpoint'](self.root,2,self.torch)
        self.assertEqual((kind,epochs),('swa_topk_rebuilt_from_resume',[2,0,1]))
        first=check_export(self.root/'raptor_resume_fold2.pt',path,self.expected,self.current,self.labels)
        ns['select_checkpoint'](self.root,2,self.torch)
        self.assertEqual(sha256_file(path),first['selected_file_sha256'])

    def test_builder_embeds_gate_and_expectation_and_rejects_inconsistent_config(self):
        from scripts import build_tracka_export_kaggle as builder
        source=self.root/'synthetic-source';source.mkdir()
        contents={'tracka_entry.py':'# synthetic entry\n','train_knee_original.py':'# synthetic original\n',
                  'train_knee_patched.py':'# synthetic patched\n','train_knee.patch':'synthetic patch\n',
                  'excluded_ids.json':json.dumps({'quarantined_study_ids':['synthetic-id']}),
                  'folds_v0.json':json.dumps({'folds':{'synthetic-id':2}})}
        for name,text in contents.items():(source/name).write_text(text)
        original_sha=sha256_file(source/'train_knee_original.py')
        manifest={'original_sha256':original_sha,'patched_sha256':sha256_file(source/'train_knee_patched.py'),
                  'patch_sha256':sha256_file(source/'train_knee.patch'),
                  'folds_sha256':sha256_file(source/'folds_v0.json'),
                  'excluded_ids_sha256':sha256_file(source/'excluded_ids.json')}
        (source/'manifest.json').write_text(json.dumps(manifest))
        (source/'kernel-metadata.json').write_text(json.dumps({'docker_image':'synthetic-image'}))
        out=self.root/'build'
        args=(out,'owner/synthetic-dataset',2,3,'synthetic-labels.parquet',
              'owner/synthetic-export','Synthetic export','synthetic',32,2,3)
        kwargs=dict(expected_version=7,expected_resume_sha256=self.expected['resume_sha256'],
                    expected_config=self.expected['config'])
        with mock.patch.object(builder,'SRC_DIR',source),mock.patch.object(builder,'ORIGINAL_SHA256',original_sha):
            builder.build(*args,**kwargs)
        notebook=json.loads((out/'notebook.ipynb').read_text())
        for cell in notebook['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'synthetic-notebook','exec')
        runner=''.join(notebook['cells'][-1]['source'])
        self.assertLess(runner.index('audit.validate_resume'),runner.index('subprocess.run'))
        self.assertLess(runner.index('audit.check_export'),runner.index('subprocess.run'))
        self.assertEqual(json.loads((out/'export-expectation.json').read_text()),self.expected)
        bad=copy.deepcopy(kwargs);bad['expected_config']['k']=12
        with self.assertRaisesRegex(ValueError,'export arguments differ'):
            builder.build(*args,**bad)


class ExpectationTests(unittest.TestCase):
    def test_missing_version_or_config_is_rejected(self):
        for expected in [{}, {'dataset_version':True,'resume_sha256':'a'*64},
                         {'dataset_version':3,'resume_sha256':'a'*64,'config':{}}]:
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError):validate_expectation(expected)


if __name__=='__main__':unittest.main()
