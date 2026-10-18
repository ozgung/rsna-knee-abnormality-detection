import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from knee.public_adaptation_fit import fingerprint,fit_config
from knee.public_adaptation_pilot import file_sha
from scripts.build_public_adaptation_evaluation import build_evaluation

class EvaluationBuilderTests(unittest.TestCase):
    def test_immutable_original_binding_and_offline_eval_only_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); sources={'knee/public_adaptation_fit.py':'original operational helper'}
            binding={'config':fit_config(),'source_bundle_sha256':fingerprint(sources)}
            prior={'status':'FAIL','failure_stage':'evaluation','updates':432,'epoch':2,'consumed_cursor':0,
                   'config':fit_config(),'binding':binding,'checkpoint_sha256':'a'*64,'checkpoint_bytes':100}
            terminal={'kernel_source':'ozgung/original-fit','version':1,'status':'ERROR','checkpoint_sha256':'a'*64}
            code=(f"sources={sources!r}\nBINDING={binding!r}\nRESUME_SOURCE=None\nPRIOR=None\n"
                  f"os.environ['KNEE_SOURCE_SHA256']={binding['source_bundle_sha256']!r}\n"
                  "if True:\n    if True:\n        previous=notebook_root('/kaggle/input',RESUME_SOURCE,'public-adaptation-fit',['receipt.json','resume.pt'])\n"
                  "        shutil.copyfile(previous/'resume.pt',out/'resume.pt'); resume=out/'resume.pt'\n"
                  "run(BINDING,resume=resume)\n")
            nb={'cells':[{'cell_type':'markdown','source':[]},{'cell_type':'code','source':code.splitlines(True)}]}
            meta={'id':'ozgung/original-fit','is_private':True,'enable_internet':False,'kernel_sources':[]}
            for name,value in [('notebook',nb),('metadata',meta),('failed',prior),('terminal',terminal)]:
                (p/name).write_text(json.dumps(value))
            manifest={'schema':1,'source_kernel':'ozgung/original-fit','source_version':1,
                      'original_training_binding':binding,'files':{
                          'resume.pt':{'sha256':'a'*64,'bytes':100},
                          'endpoint.pt':{'sha256':'b'*64,'bytes':50},
                          'receipt.json':{'sha256':file_sha(p/'failed'),'bytes':(p/'failed').stat().st_size}}}
            (p/'manifest').write_text(json.dumps(manifest))
            args=[p/'notebook',p/'metadata',p/'failed',p/'terminal',p/'build','eval-only']
            kwargs={f'expected_{name}_sha256':file_sha(p/name) for name in ('notebook','metadata','failed','terminal')}
            kwargs['expected_endpoint_sha256']='b'*64
            kwargs.update(recovery_dataset='ozgung/private-recovery',recovery_manifest=p/'manifest',
                          expected_recovery_manifest_sha256=file_sha(p/'manifest'),expected_dataset_version=1)
            receipt=build_evaluation(*args,**kwargs)
            self.assertEqual(receipt['binding'],binding);self.assertFalse(receipt['launched'])
            self.assertEqual(receipt['optimizer_updates_this_run'],0)
            built=json.loads((p/'build/notebook.ipynb').read_text())
            entry=''.join(built['cells'][1]['source']);compile(entry,'<recovery-test>','exec')
            self.assertIn('evaluation_recovery=EVALUATION_RECOVERY',entry)
            self.assertIn("previous/'endpoint.pt'",entry)
            built_meta=json.loads((p/'build/kernel-metadata.json').read_text())
            self.assertEqual(built_meta['kernel_sources'],[])
            self.assertEqual(built_meta['dataset_sources'],['ozgung/private-recovery'])
            self.assertIn("previous=dataset_root('/kaggle/input'",entry)
            self.assertIn('Recovery dataset file bytes differ',entry)
            self.assertIn('Recovery manifest differs',entry)
            for field,value in [('source_kernel','ozgung/wrong'),('source_version',2)]:
                changed=copy.deepcopy(manifest);changed[field]=value;(p/'manifest').write_text(json.dumps(changed))
                bad_kwargs=dict(kwargs,expected_recovery_manifest_sha256=file_sha(p/'manifest'))
                bad_args=list(args);bad_args[4]=p/('bad-manifest-'+field)
                with self.assertRaisesRegex(ValueError,'Exact recovery dataset manifest'):
                    build_evaluation(*bad_args,**bad_kwargs)
            (p/'manifest').write_text(json.dumps(manifest))
            for field,value in [('expected_dataset_version',0),('expected_dataset_version',True),
                                ('recovery_dataset','other/private-recovery'),('expected_endpoint_sha256',None)]:
                bad_kwargs=dict(kwargs,**{field:value});bad_args=list(args);bad_args[4]=p/('bad-'+field)
                with self.assertRaises(ValueError): build_evaluation(*bad_args,**bad_kwargs)
            for name in ('resume.pt','endpoint.pt','receipt.json'):
                changed=copy.deepcopy(manifest);changed['files'][name]['sha256']='c'*64
                (p/'manifest').write_text(json.dumps(changed))
                bad_kwargs=dict(kwargs,expected_recovery_manifest_sha256=file_sha(p/'manifest'))
                bad_args=list(args);bad_args[4]=p/('bad-file-'+name)
                with self.assertRaisesRegex(ValueError,'Exact recovery dataset manifest'):
                    build_evaluation(*bad_args,**bad_kwargs)
            (p/'manifest').write_text(json.dumps(manifest))
            for field,value in [('epoch',1),('updates',431),('failure_stage','training')]:
                changed=copy.deepcopy(prior);changed[field]=value;(p/'failed').write_text(json.dumps(changed))
                kwargs['expected_failed_sha256']=file_sha(p/'failed');args[4]=p/('bad-'+field)
                with self.assertRaisesRegex(ValueError,'Finished fixed training'):
                    build_evaluation(*args,**kwargs)
            (p/'failed').write_text(json.dumps(prior));kwargs['expected_failed_sha256']=file_sha(p/'failed')
            terminal['version']=0;(p/'terminal').write_text(json.dumps(terminal));kwargs['expected_terminal_sha256']=file_sha(p/'terminal')
            with self.assertRaisesRegex(ValueError,'Exact terminal'):
                build_evaluation(*args,**kwargs)
