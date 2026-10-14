"""VM: write r3 run configs (in-* resume-only recovery with pinned resume SHA; dr-orig fresh) and validate them.

Usage: python vm_raptor64_r3_configs.py SESSION IN_ORIG_RESUME_SHA IN_PLANE_RESUME_SHA
Configs go to /content/raptor64-ARM-run-config.json (fresh VM; never overwritten).
"""
import hashlib, importlib.util, json, os, sys
from pathlib import Path

session, resumes = sys.argv[1], {'in-orig': sys.argv[2], 'in-plane': sys.argv[3]}
os.environ['KAGGLE_CONFIG_DIR'] = '/content'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


vm = load('vm', '/content/raptor64-tools/scripts/colab_raptor64_vm.py')
runner = load('runner', '/content/raptor64-tools/scripts/colab_raptor64_run.py')
out = {}
for arm in ('in-orig', 'in-plane', 'dr-orig'):
    p = vm.paths(arm)
    assert not p['config'].exists(), 'Existing config preserved'
    c = vm.make_config(arm, vm_session_id=session)
    if arm in resumes:
        c.update(resume=True, resume_sha256=resumes[arm])
    runner.validate(c); runner.private_dataset(c['persistence']['dataset'])
    p['config'].write_text(json.dumps(c, indent=2) + '\n')
    out[arm] = {'config': str(p['config']), 'config_sha256': hashlib.sha256(p['config'].read_bytes()).hexdigest(),
                'resume': c['resume'], 'resume_sha256': c.get('resume_sha256'), 'package_manifest_sha256': c['package_manifest_sha256'],
                'content_sha256': c['content_sha256'], 'initializer_descriptor_sha256': c['initializer_descriptor_sha256']}
print(json.dumps(out, indent=1))
