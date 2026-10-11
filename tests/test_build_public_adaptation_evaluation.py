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
                  "files=['receipt.json','resume.pt']\n"
                  "if True:\n    if True:\n        shutil.copyfile(previous/'resume.pt',out/'resume.pt'); resume=out/'resume.pt'\n"
                  "run(BINDING,resume=resume)\n")
            nb={'cells':[{'cell_type':'markdown','source':[]},{'cell_type':'code','source':code.splitlines(True)}]}
            meta={'id':'ozgung/original-fit','is_private':True,'enable_internet':False,'kernel_sources':[]}
            for name,value in [('notebook',nb),('metadata',meta),('failed',prior),('terminal',terminal)]:
                (p/name).write_text(json.dumps(value))
            args=[p/'notebook',p/'metadata',p/'failed',p/'terminal',p/'build','eval-only']
            kwargs={f'expected_{name}_sha256':file_sha(p/name) for name in ('notebook','metadata','failed','terminal')}
            kwargs['expected_endpoint_sha256']='b'*64
            receipt=build_evaluation(*args,**kwargs)
            self.assertEqual(receipt['binding'],binding);self.assertFalse(receipt['launched'])
            self.assertEqual(receipt['optimizer_updates_this_run'],0)
            built=json.loads((p/'build/notebook.ipynb').read_text())
            entry=''.join(built['cells'][1]['source']);compile(entry,'<recovery-test>','exec')
            self.assertIn('evaluation_recovery=EVALUATION_RECOVERY',entry)
            self.assertIn("previous/'endpoint.pt'",entry)
            self.assertEqual(json.loads((p/'build/kernel-metadata.json').read_text())['kernel_sources'],['ozgung/original-fit'])
            for field,value in [('epoch',1),('updates',431),('failure_stage','training')]:
                changed=copy.deepcopy(prior);changed[field]=value;(p/'failed').write_text(json.dumps(changed))
                kwargs['expected_failed_sha256']=file_sha(p/'failed');args[4]=p/('bad-'+field)
                with self.assertRaisesRegex(ValueError,'Finished fixed training'):
                    build_evaluation(*args,**kwargs)
            (p/'failed').write_text(json.dumps(prior));kwargs['expected_failed_sha256']=file_sha(p/'failed')
            terminal['version']=0;(p/'terminal').write_text(json.dumps(terminal));kwargs['expected_terminal_sha256']=file_sha(p/'terminal')
            with self.assertRaisesRegex(ValueError,'Exact terminal'):
                build_evaluation(*args,**kwargs)
