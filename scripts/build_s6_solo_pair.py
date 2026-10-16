"""Build the sealed solo-V2 / solo-S6fall Kaggle pair. Never pushes, launches or submits anything.

  rehearsal  both private/offline 2xT4 frozen-39 qualification packets (default)
  entry      one arm's image-only entry, only from that arm's exact rehearsal build
             plus a collected PASS result that replays independently
  verify     re-derive the emitted rehearsal packets byte for byte

The two notebooks share every source byte and the common config; they differ
only in cfg['arm'] (the B checkpoint binding and its genuine receipts), the
kernel id/title and the V2 weight dataset attachment.
"""
import argparse
import ast
import base64
import hashlib
import json
import lzma
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from knee import s6_solo_pair as PAIR
from knee import s6_solo_runtime as RT
from knee.s6_solo_runtime import canonical, digest, file_sha, require

MAIN = Path(subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                                    cwd=PROJECT, text=True).strip()).parent
OUT = MAIN / 'outputs'
CAPTURED = OUT / 'research/public-0957-20261015/kernels/nartaa_rsna-knee-s6fall-solo/rsna-knee-s6fall-solo.py'
CAPTURED_META = CAPTURED.with_name('kernel-metadata.json')
CAPTURED_META_SHA = 'a355a45d16c4641e7372d633db51a7b40e0cb153c68c538057582b1904b2eaa3'
ACCEPT = OUT / 'analysis/s6v2-final-acceptance-20261016'
ACCEPTANCE_SHA = 'aa7470bed00ea308d17b9f10298720da5eb8c22dc8a97eb45f6ee1c9362b484b'
CPU_SHA = '3dc1da57c52398d53982a4e12ffa404e06a294145776fcab101eeb2d061a948f'
DESCRIPTOR_SHA = {'control': 'b22c80b55c9b84119933641d08090804da1baa2f9a3c57f5d2c4e12f5089f5ea',
                  'v2': '3871320a8a7ec53f93ad728249969aac5e95b33f10475a7b22e8e8aa449a44c0'}
CONTROL_V3 = OUT / 'analysis/s6solo-pair-20261016/control-v3'
B0_NOTEBOOK = OUT / 'kaggle/knee-s6-dread-b0-rehearsal-20261013/notebook.ipynb'
B0_NOTEBOOK_SHA = '9de1fcc114416f773f57d88d62da1b6396f5e1896737838a146d5f46bc216851'
B0_PAYLOAD_SHA = '15f5c50c8c0b406ba1f5ad3bfdd6f4148aeea3872bbf71ebd5de5de7527c40c0'

VERSION = 's6solo-pair-v1'
COMP = 'rsna-knee-abnormality-detection'
OWNER = 'ozgung'
SLUGS = {'v2': 's6solo-v2-qual-20261016', 'control': 's6solo-s6fall-qual-20261016'}
ENTRY_SLUGS = {'v2': 's6solo-v2-entry-20261016', 'control': 's6solo-s6fall-entry-20261016'}
CODE_DATASET, CODE_VERSION, CODE_MOUNT = 'nartaa/rsna-knee-s6fall-weights', 3, 'rsna-knee-s6fall-weights'
# publisher code extras: version 3 = the captured solo kernel's layout (version 1 lacks them)
CODE_FILES = {'code_manifest.json': {'sha256': 'b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22', 'bytes': 52962},
              's6_extras.bin': {'sha256': 'ddc428d9f2cda0e1378b6731ca7efbdd0397978204763f2a2f81169979e50c8e', 'bytes': 256011},
              'checkpoint-manifest.json': {'sha256': 'ed3c6d8d2e69ae08c4a97eab3c860a9ec20c5c69484e35de6a3119d396d731cd', 'bytes': 1128}}
A_WEIGHT = {'dataset': 'nartaa/rsna-knee-publication-swa-weights-20261007', 'file': 'raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt',
            'sha256': '7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef', 'bytes': 292840787}
HELIOS = 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004'
CONTROL_RECEIPTS = {'E_prior_s42_fall.run_config.json': '36e49cf0c8db97e7f995482933911f018c32fca8c5933b4d64399c128c19c886',
                    'E_prior_s42_fall.DONE.json': 'acc99ed61393404feaa4f84a2e20ce27ad9f55a257b8e0d0760e83e105a20c6a',
                    'checkpoint-manifest.json': 'ed3c6d8d2e69ae08c4a97eab3c860a9ec20c5c69484e35de6a3119d396d731cd'}
V2_RECEIPT_NAMES = ('run_config.json', 'SHA256SUMS', 'membership_E_prior_s42_f0.json')
REHEARSAL = {'mounts': [[CODE_MOUNT, 'checkpoint-manifest.json'], [HELIOS.split('/')[1], 'anatomy.pt']],
             'sentinel_per_board': 2, 'sentinel_timeout_s': 1800, 'sentinel_probability_atol': 1e-3,
             'projection_studies': 1300, 'gate_seconds': 29880, 'post_controller_allowance_seconds': 60}
COMMON_SOURCES = ('knee/__init__.py', 'knee/s6_solo_pair.py', 'knee/s6_solo_runtime.py')
REHEARSAL_SOURCES = ('knee/s6_solo_rehearsal.py', 'knee/s6_solo_a_runner.py')
DEVIATIONS = [
    'Control checkpoint is read from nartaa/rsna-knee-s6fall-weights version 3 (the captured solo layout, which carries the '
    'publisher code extras) instead of the accepted version-1 download; identical full SHA256/size is required at runtime. '
    'Both arms mount the same version-3 code extras.',
    'Rehearsal only: component A runs through knee/s6_solo_a_runner.py (exact A bytes, find_test_root returns the fixture); '
    'the controller sees the fixture through its own K1_KAGGLE_ROOT local-test hook; B records pixel hashes (K1_PX_DUMP).',
    'Common prepare_inputs replaces three exact sites: hash-bound inputs (B checkpoint, B receipts, code extras, A weights), '
    'per-arm receipt expectations instead of the publisher DONE/run_config assertions, and stage_checkpoint byte staging '
    'instead of the symlink.',
    'Docker image pinned to the captured publisher kernel image; nartaa publication and heliosli datasets attached unversioned '
    'as in the captured kernel, with their used files hash-pinned at runtime.']


def read(path):
    return json.loads(Path(path).read_bytes())


def captured_public_ids(text):
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PUBLIC_TEST_IDS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('PUBLIC_TEST_IDS')


def fixture(public_ids):
    require(file_sha(B0_NOTEBOOK) == B0_NOTEBOOK_SHA, 'EXACT_B0_FROZEN39_ARCHIVE')
    code = ''.join(read(B0_NOTEBOOK)['cells'][0]['source'])
    encoded = next(ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'encoded' for t in n.targets))
    pack = json.loads(lzma.decompress(base64.b64decode(encoded, validate=True)))
    require(digest(pack) == B0_PAYLOAD_SHA, 'EXACT_B0_FROZEN39_PAYLOAD')
    b = pack['cfg']['binding']
    ids = b['s6_probe_ids'] + b['old_historical_ids'] + b['visible_ids']
    splits = ['train'] * (len(b['s6_probe_ids']) + len(b['old_historical_ids'])) + ['test'] * len(b['visible_ids'])
    require(len(ids) == 39 and len(set(ids)) == 39 and splits.count('train') == 36 and b['visible_ids'] == public_ids,
            'FROZEN39_36_TRAIN_3_TEST')
    return {'ids': ids, 'splits': splits, 'source': {'b0_notebook_sha256': B0_NOTEBOOK_SHA, 'b0_payload_sha256': B0_PAYLOAD_SHA,
                                                     'order': 's6_probe_ids + old_historical_ids + visible_ids'}}


def _json_checked(path, sha, want):
    require(file_sha(path) == sha, 'LOCAL_RECEIPT_PIN_' + Path(path).name)
    data = read(path)
    bad = {k: (data.get(k), v) for k, v in want.items() if data.get(k) != v}
    require(not bad, 'LOCAL_RECEIPT_EXPECTATION_%s_%s' % (Path(path).name, sorted(bad)))


def arms(captured_bytes):
    """Both B bindings from the accepted real descriptors and the genuine local receipts."""
    desc = {}
    for v in ('control', 'v2'):
        path = ACCEPT / ('%s-descriptor.json' % v)
        require(file_sha(path) == DESCRIPTOR_SHA[v], 'ACCEPTED_DESCRIPTOR_' + v)
        desc[v] = read(path)
    plan = PAIR.validate_pair(desc['control'], desc['v2'], source=captured_bytes)
    require(plan['status'] == 'UNSEALED_PREPARATION_ONLY' and plan['source_bytes_verified'], 'PAIR_DESCRIPTOR_VALIDATION')
    require(file_sha(ACCEPT / 'acceptance.json') == ACCEPTANCE_SHA and file_sha(ACCEPT / 'cpu-pair.json') == CPU_SHA, 'ACCEPTED_RECEIPTS')
    acc, cpu = read(ACCEPT / 'acceptance.json'), read(ACCEPT / 'cpu-pair.json')
    require(cpu['status'] == 'PASS_SYNTHETIC_CPU_ONLY', 'CPU_PAIR_PASS')
    targets = PAIR.TARGETS
    out = {}
    # control: publisher's own receipts in the version-3 mount, same values the captured controller asserted
    c = desc['control']
    for name, pin in CODE_FILES.items():
        require(file_sha(CONTROL_V3 / name) == pin['sha256'] and (CONTROL_V3 / name).stat().st_size == pin['bytes'], 'CODE_EXTRAS_' + name)
    c_run = {'size': 320, 'crop': 288, 'views': 'v4', 'slices': 3, 'phys_mm': 3.5, 'backbone': 'convnext_small',
             'spatial_attn': True, 'arms': 'E_prior'}
    c_done = {'model': c['checkpoint_filename'], 'model_sha256': c['checkpoint_sha256'], 'epochs': 16, 'train_studies': 4263}
    _json_checked(CONTROL_V3 / 'E_prior_s42_fall.run_config.json', CONTROL_RECEIPTS['E_prior_s42_fall.run_config.json'], c_run)
    _json_checked(CONTROL_V3 / 'E_prior_s42_fall.DONE.json', CONTROL_RECEIPTS['E_prior_s42_fall.DONE.json'], c_done)
    require(file_sha(CONTROL_V3 / 'checkpoint-manifest.json') == CONTROL_RECEIPTS['checkpoint-manifest.json']
            == acc['files']['control']['checkpoint-manifest.json']['sha256'], 'CONTROL_MANIFEST_UNCHANGED_V1_V3')
    require(read(CONTROL_V3 / 'checkpoint-manifest.json')['sha256'] == c['checkpoint_sha256'], 'CONTROL_MANIFEST_RECORDS_MODEL')
    listing = (CONTROL_V3 / 'official-files-v3.txt').read_text().split()
    require('E_prior_s42_fall.pt' in listing and str(c['checkpoint_size']) == listing[listing.index('E_prior_s42_fall.pt') + 1],
            'CONTROL_V3_LISTS_SAME_SIZE_CHECKPOINT')
    require(cpu['checkpoints']['control']['checkpoint_sha256'] == c['checkpoint_sha256']
            and cpu['checkpoints']['control']['strict_load'] is True, 'CONTROL_CPU_STRICT')
    out['control'] = {
        'variant': 'control', 'label': 'solo-S6fall', 'descriptor': c,
        'b_model': {'file': c['checkpoint_filename'], 'source_file': c['checkpoint_filename'], 'sha256': c['checkpoint_sha256'],
                    'bytes': c['checkpoint_size'], 'arm': 'E_prior', 'dataset': CODE_DATASET, 'dataset_version': CODE_VERSION,
                    'descriptor_dataset_version': c['dataset_version'], 'mount': CODE_MOUNT},
        'receipts': dict(CONTROL_RECEIPTS),
        'json_expect': {'E_prior_s42_fall.run_config.json': c_run, 'E_prior_s42_fall.DONE.json': c_done},
        'text_expect': {},
        'cpu': {'receipt_sha256': CPU_SHA, 'state_sha256': cpu['checkpoints']['control']['state_sha256'], 'strict_load': True},
        'provenance': 'Public nartaa S6fall all-data checkpoint, publisher-reported 4263 training studies/16 epochs; '
                      'public 0.950 is attributed historical context, not our verified control score.'}
    v = desc['v2']
    d = ACCEPT / 'v1'
    receipts = {}
    for name in V2_RECEIPT_NAMES:
        pin = acc['files']['v1'][name]
        require(file_sha(d / name) == pin['sha256'] and (d / name).stat().st_size == pin['size'], 'V2_RECEIPT_' + name)
        receipts[name] = pin['sha256']
    v_run = {'arms': 'E_prior', 'backbone': 'convnext_small', 'crop': 288, 'slices': 3, 'phys_mm': 3.5, 'spatial_attn': True,
             'extent': False, 'state_spec': None, 'freeze_blocks': 16, 'epochs': 16, 'seed_base': 42,
             'include_review_holdout': False, 'pool_studies': 4345, 'gold_studies': 58, 'labels': targets}
    m = acc['membership']
    v_mem = {k: m[k] for k in ('arm', 'seed', 'all_data', 'train', 'train_sha256', 'holdout', 'gold', 'gold_sha256', 'train_gold_overlap')}
    require(v_mem['train'] == 4345 and v_mem['holdout'] == 0 and v_mem['train_gold_overlap'] == 0 and v_mem['all_data'] is True, 'V2_ALL_DATA')
    _json_checked(d / 'run_config.json', receipts['run_config.json'], v_run)
    _json_checked(d / 'membership_E_prior_s42_f0.json', receipts['membership_E_prior_s42_f0.json'], v_mem)
    line = (d / 'SHA256SUMS').read_text().splitlines()[0]
    require(line.split()[0] == v['checkpoint_sha256'] and line.endswith('/' + v['checkpoint_filename']), 'V2_SHA256SUMS_RECORDS_MODEL')
    require(file_sha(d / v['checkpoint_filename']) == v['checkpoint_sha256'] if (d / v['checkpoint_filename']).is_file() else True,
            'LOCAL_V2_CHECKPOINT')
    require(cpu['checkpoints']['v2']['checkpoint_sha256'] == v['checkpoint_sha256'] and cpu['checkpoints']['v2']['strict_load'] is True,
            'V2_CPU_STRICT')
    out['v2'] = {
        'variant': 'v2', 'label': 'solo-V2', 'descriptor': v,
        'b_model': {'file': v['runtime_alias'], 'source_file': v['checkpoint_filename'], 'sha256': v['checkpoint_sha256'],
                    'bytes': v['checkpoint_size'], 'arm': 'E_prior', 'dataset': v['dataset_slug'], 'dataset_version': v['dataset_version'],
                    'descriptor_dataset_version': v['dataset_version'], 'mount': v['dataset_slug'].split('/')[1]},
        'receipts': receipts, 'json_expect': {'run_config.json': v_run, 'membership_E_prior_s42_f0.json': v_mem},
        'text_expect': {'SHA256SUMS': line},
        'cpu': {'receipt_sha256': CPU_SHA, 'state_sha256': cpu['checkpoints']['v2']['state_sha256'], 'strict_load': True},
        'provenance': 'Own post-hoc V2-all ConvNeXt-small S6, all 4345 studies, seed 42, 16 epochs; gold-58 selection-exposed, '
                      'teacher/report exposure retained; original fold-0 BLOCKED verdict unchanged.'}
    for arm in out.values():
        require(PAIR.ALIAS_RE.fullmatch(arm['b_model']['file']) and arm['b_model']['file'] == arm['descriptor'].get(
            'runtime_alias', arm['descriptor']['checkpoint_filename']), 'LOADER_DISCOVERABLE_ALIAS')
    return out


def common(captured_text):
    return {'version': VERSION, 'targets': PAIR.TARGETS, 'captured_source_sha256': RT.CAPTURED_SHA256,
            'embedded_a_sha256': RT.EMBEDDED_A_SHA256, 'contract': PAIR.COMMON,
            'code_mount': {'dataset': CODE_DATASET, 'version': CODE_VERSION, 'mount': CODE_MOUNT, 'files': CODE_FILES},
            'a_weight': A_WEIGHT, 'public_test_ids_count': len(captured_public_ids(captured_text))}


def sources(mode):
    names = COMMON_SOURCES + (REHEARSAL_SOURCES if mode == 'rehearsal' else ())
    out = {n: (PROJECT / n).read_text() for n in names}
    out[RT.CAPTURED] = CAPTURED.read_text()
    require(hashlib.sha256(out[RT.CAPTURED].encode()).hexdigest() == RT.CAPTURED_SHA256, 'CAPTURED_TEXT_ROUNDTRIP')
    for n, s in out.items():
        compile(s, n, 'exec')
    return out


def packet(variant, mode='rehearsal', qualified=None):
    require(variant in SLUGS and mode in ('rehearsal', 'entry'), 'VARIANT_MODE')
    raw = CAPTURED.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == RT.CAPTURED_SHA256, 'CAPTURED_PUBLISHER_SOURCE_SHA256')
    text = raw.decode()
    RT.derived_prepare_inputs_source(text)
    cfg = {'mode': mode, 'common': common(text), 'arm': arms(raw)[variant]}
    if mode == 'rehearsal':
        cfg['rehearsal'] = {**REHEARSAL, 'fixture': fixture(captured_public_ids(text))}
    else:
        require(qualified is not None, 'ENTRY_REQUIRES_QUALIFIED_REHEARSAL')
        cfg['qualified_rehearsal'] = qualified
    src = sources(mode)
    return {'cfg': cfg, 'sources': src, 'source_sha256': {n: hashlib.sha256(s.encode()).hexdigest() for n, s in src.items()}}


BOOTSTRAP = r'''import time
started=time.time()
import os,sys,json,base64,lzma,hashlib,signal,shutil,subprocess
from pathlib import Path
encoded=__ENCODED__
expected_payload=__EXPECTED__
work=Path('/kaggle/working')
def remove_submission():
    for p in (work/'submission.csv',work/'.submission.csv.tmp'):
        try:p.unlink()
        except FileNotFoundError:pass
payload=json.loads(lzma.decompress(base64.b64decode(encoded,validate=True)))
if hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=expected_payload:raise ValueError('S6 solo payload changed')
project=work/'s6solo-project'
if project.exists():shutil.rmtree(project)
for name,content in sorted(payload['sources'].items()):
    rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe source path')
    p=project/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode())
    if hashlib.sha256(p.read_bytes()).hexdigest()!=payload['source_sha256'][name]:raise ValueError('source bytes changed: '+name)
config=project/'runtime-config.json';config.write_text(json.dumps(payload['cfg'],indent=1,sort_keys=True))
env=dict(os.environ,S6SOLO_PAYLOAD=expected_payload,S6SOLO_STARTED=repr(started),HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
for k in ('K1_KAGGLE_ROOT','K1_TMP','K1_PX_DUMP','PYTHONPATH'):env.pop(k,None)
remove_submission()
print('S6SOLO %s %s payload %s'%(payload['cfg']['mode'],payload['cfg']['arm']['label'],expected_payload),flush=True)
child=subprocess.Popen([sys.executable,'-u','-m','knee.s6_solo_runtime','--config',str(config)],cwd=str(project),env=env,
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,text=True,bufsize=1,start_new_session=True)
def backstop(signum,frame):raise TimeoutError('S6 solo notebook backstop (after the controller guard) reached')
signal.signal(signal.SIGALRM,backstop);signal.alarm(8*3600+20*60)
rc=None
try:
    for line in child.stdout:print(line,end='',flush=True)
    rc=child.wait()
finally:
    signal.alarm(0)
    if rc is None:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.wait()
    if rc!=0:remove_submission()
if rc!=0:raise RuntimeError('S6 solo runtime exited %s; submission.csv removed'%rc)
'''


def bootstrap(pack):
    code = BOOTSTRAP.replace('__ENCODED__', repr(base64.b64encode(lzma.compress(canonical(pack), preset=9)).decode()))
    code = code.replace('__EXPECTED__', repr(digest(pack)))
    compile(code, '<s6solo-bootstrap>', 'exec')
    return code


def notebook(pack, cell_id):
    nb = {'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python3', 'language': 'python'}},
          'cells': [{'id': cell_id, 'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                     'source': bootstrap(pack).splitlines(True)}]}
    raw = json.dumps(nb, indent=2) + '\n'
    require(len(raw.encode()) < 900000, 'NOTEBOOK_900K_CAP')
    return raw


def metadata(slug, variant):
    require(isinstance(slug, str) and 3 <= len(slug) <= 50 and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in slug),
            'SAFE_KERNEL_SLUG')
    require(file_sha(CAPTURED_META) == CAPTURED_META_SHA, 'CAPTURED_KERNEL_METADATA')
    sources = ['%s/%d' % (CODE_DATASET, CODE_VERSION), A_WEIGHT['dataset'], HELIOS]
    if variant == 'v2':
        sources.append('%s/%d' % (PAIR.V2_DATASET, 1))
    return {'id': '%s/%s' % (OWNER, slug), 'title': slug, 'code_file': 'notebook.ipynb', 'language': 'python',
            'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
            'competition_sources': [COMP], 'dataset_sources': sources, 'kernel_sources': [], 'model_sources': [],
            'docker_image': read(CAPTURED_META)['docker_image'], 'machine_shape': 'NvidiaTeslaT4'}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def emit(dest, slug, variant, pack, mode):
    dest = Path(dest)
    require(not dest.exists() and dest.resolve().is_relative_to(OUT.resolve()), 'FRESH_IGNORED_OUTPUT')
    raw = notebook(pack, 's6solo-%s-%s' % (variant, mode))
    dest.mkdir(parents=True)
    (dest / 'notebook.ipynb').write_text(raw)
    save_json(dest / 'kernel-metadata.json', metadata(slug, variant))
    changed = subprocess.check_output(['git', 'status', '--porcelain', '--', *COMMON_SOURCES, *REHEARSAL_SOURCES,
                                       'scripts/build_s6_solo_pair.py'], cwd=PROJECT, text=True).strip()
    receipt = {'status': 'PREPARATION_ONLY' if mode == 'rehearsal' else 'QUALIFIED_ENTRY_BUILD', 'mode': mode,
               'variant': variant, 'label': pack['cfg']['arm']['label'], 'slug': slug,
               'binding_sha256': RT.binding_digest(pack['cfg']), 'payload_sha256': digest(pack),
               'common_source_bundle_sha256': digest({n: s for n, s in pack['sources'].items() if n not in REHEARSAL_SOURCES}),
               'source_sha256': pack['source_sha256'], 'b_model': pack['cfg']['arm']['b_model'],
               'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
               'source_state': ('uncommitted implementation' if changed else 'committed implementation') + '; exact file hashes bound',
               'builder_sha256': file_sha(__file__),
               'files': {n: file_sha(dest / n) for n in ('notebook.ipynb', 'kernel-metadata.json')},
               'notebook_bytes': len(raw.encode()), 'runtime_checks_exercised': False, 'deviations': DEVIATIONS}
    save_json(dest / 'build-receipt.json', receipt)
    save_json(dest / 'MANIFEST.json', {'mode': mode, 'payload_sha256': digest(pack), 'files': {
        n: {'sha256': file_sha(dest / n), 'bytes': (dest / n).stat().st_size}
        for n in ('notebook.ipynb', 'kernel-metadata.json', 'build-receipt.json')}})
    return receipt


def build_rehearsal(out_root=OUT / 'kaggle'):
    out = {}
    for variant in ('v2', 'control'):
        out[variant] = emit(Path(out_root) / SLUGS[variant], SLUGS[variant], variant, packet(variant), 'rehearsal')
    return out


def unpack(path):
    code = ''.join(read(path)['cells'][0]['source'])
    vals = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n, ast.Assign)
            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('encoded', 'expected_payload')}
    pack = json.loads(lzma.decompress(base64.b64decode(vals['encoded'], validate=True)))
    require(digest(pack) == vals['expected_payload'], 'NOTEBOOK_PAYLOAD_DIGEST')
    return pack, code


def verify_build(root, variant, mode='rehearsal', qualified=None):
    root = Path(root)
    r, m, meta = read(root / 'build-receipt.json'), read(root / 'MANIFEST.json'), read(root / 'kernel-metadata.json')
    require(set(m['files']) == {'notebook.ipynb', 'kernel-metadata.json', 'build-receipt.json'}, 'MANIFEST_COVERAGE')
    for n, pin in m['files'].items():
        require(pin == {'sha256': file_sha(root / n), 'bytes': (root / n).stat().st_size}, 'IMMUTABLE_BUILD_' + n)
    expected = packet(variant, mode, qualified)
    actual, code = unpack(root / 'notebook.ipynb')
    require(actual == expected and code == bootstrap(expected), 'EXACT_PACKET_REPLAY')
    require(meta == metadata(meta['title'], variant) and meta['id'] == '%s/%s' % (OWNER, meta['title']), 'PRIVATE_OFFLINE_T4_TITLE_ID')
    require(r['mode'] == mode and r['variant'] == variant and r['payload_sha256'] == digest(expected) == m['payload_sha256']
            and r['files'] == {n: file_sha(root / n) for n in ('notebook.ipynb', 'kernel-metadata.json')}, 'BUILD_RECEIPT')
    return r


def build_entry(variant, rehearsal_build, rehearsal_result, destination=None, slug=None):
    from knee import s6_solo_rehearsal as REH
    prepared = verify_build(rehearsal_build, variant)
    rehearsal_cfg = packet(variant)['cfg']
    k1 = RT.load_controller(CAPTURED, name='k1_solo_builder_replay')
    checked = REH.verify_result(rehearsal_result, rehearsal_cfg, k1)
    require(checked['payload_sha256'] == prepared['payload_sha256'], 'RESULT_FROM_THIS_REHEARSAL_PACKET')
    qualified = {**checked, 'rehearsal_build': {'slug': prepared['slug'], 'payload_sha256': prepared['payload_sha256'],
                                                'files': prepared['files']}}
    slug = slug or ENTRY_SLUGS[variant]
    dest = Path(destination or OUT / 'kaggle' / slug)
    return emit(dest, slug, variant, packet(variant, 'entry', qualified), 'entry')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='command')
    r = sub.add_parser('rehearsal'); r.add_argument('--out-root', default=str(OUT / 'kaggle'))
    e = sub.add_parser('entry')
    e.add_argument('--arm', choices=['v2', 'control'], required=True)
    e.add_argument('--rehearsal-build', required=True); e.add_argument('--rehearsal-result', required=True)
    e.add_argument('--destination'); e.add_argument('--slug')
    v = sub.add_parser('verify'); v.add_argument('--out-root', default=str(OUT / 'kaggle'))
    a = p.parse_args(argv)
    if a.command in (None, 'rehearsal'):
        out = build_rehearsal(getattr(a, 'out_root', OUT / 'kaggle'))
    elif a.command == 'entry':
        out = {a.arm: build_entry(a.arm, a.rehearsal_build, a.rehearsal_result, a.destination, a.slug)}
    else:
        out = {x: verify_build(Path(a.out_root) / SLUGS[x], x) for x in ('v2', 'control')}
    print(json.dumps({k: {f: x[f] for f in ('status', 'slug', 'payload_sha256', 'files', 'notebook_bytes')} for k, x in out.items()},
                     indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
