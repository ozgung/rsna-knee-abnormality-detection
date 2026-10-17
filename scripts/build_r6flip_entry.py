"""Build the sealed R6-flip Kaggle packets. Never pushes, launches or submits anything.

  rehearsal  private/offline 2xT4 frozen-39 qualification packet (default slug r6flip-rehearsal-20261017)
  entry      image-only entry, only from this exact rehearsal build plus its collected result, replayed
             independently (projection-only failures need an explicit named waiver file)
  verify     re-derive the emitted rehearsal packet byte for byte

The payload is A-prime's sealed v2 rehearsal payload (ws2-r224-v2-qual-20261015, notebook 412eef8f...) with every
source byte kept, R7-only configuration removed, and four new modules: the R6 profile, the frozen frame, the R6
worker/observer and the controller. R6 assets come from heliosli/rsna-knee-r6-frozen-fe055406-w1 version 1.
"""
import argparse
import ast
import base64
import copy
import hashlib
import json
import lzma
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from knee import public_r6flip as R6F
from knee import public_r6_profile as PROFILE
from knee import public_r6flip_entry as ENTRY
from knee import public_r6flip_worker as WORKER

MAIN = Path(subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                                    cwd=PROJECT, text=True).strip()).parent
OUT = MAIN / 'outputs'
ARCH = OUT / 'kaggle/ws2-r224-v2-qual-20261015'
ARCH_NOTEBOOK_SHA = '412eef8fc280e4596808cc4d4ae0aab4027bb632872125d910782a0105e29620'
ARCH_PAYLOAD_SHA = 'ecb7db28c1989957b83c8c95ec0796bb241427feacc1297511a9c335ecd7008e'
ARCH_META_SHA = '41627021b2480b64a8f434912fe9e6454a5dcd813a3c981b6b20f474b6609885'
APRIME_RESULT = OUT / 'kaggle/ws2-r224-v2-qual-20261015-result/ws2-a-prime/receipt.json'
APRIME_RESULT_SHA = '8b645d41ded79d946b57e1580586c00ca75849db20dda288d1adad5cc7319dd7'   # A-prime's failed-projection v2 receipt
CPU_PINS = OUT / 'analysis/r6flip-20261017/cpu-pins.json'
CPU_PINS_SHA = '3d25aefe93cfa0670087fd66b83e508822d46e5c120f221b713661c260d6a9d7'
R6_ASSETS = OUT / 'research/public-0958-20261017/r6-v1-assets'
PUBLIC_RECIPE_NOTEBOOK_SHA = 'ad0b97aca3361f16e32b0836655dd68bc2a37ff0cc5a44f6d6f4de07880d986d'
OWNER = 'ozgung'
COMP = 'rsna-knee-abnormality-detection'
SLUGS = {'rehearsal': 'r6flip-rehearsal-20261017', 'entry': 'r6flip-entry-20261017'}
NEW_SOURCES = ('knee/public_r6_profile.py', 'knee/public_r6flip.py', 'knee/public_r6flip_worker.py', 'knee/public_r6flip_entry.py')
R7_ONLY_CFG = ('manifest', 'asset_binding', 'pilot_state_sha256', 'heliosli_reference', 'heliosli_visible_atol', 'ws2_binding',
               'guard_source_sha256')
PRIVATE_CFG = ('probes', 'probe_order', 'old_sentinels', 's6_sentinels', 'development_windows')
PRIVATE_BINDING = ('sentinel_inputs', 'sentinel_probabilities', 'visible_ids', 'probe_contract_sha256', 's6_probe_ids',
                   'old_historical_ids')
DATASETS = ['%s/%d' % (R6F.R6_DATASET, R6F.R6_DATASET_VERSION), 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004',
            'goodpjw2008/rsna-knee-2-5d-convnext-reader', 'nartaa/rsna-knee-publication-swa-weights-20261007/1']
DEVIATIONS = [
    'R6 worker = compact_infer.py main() in-process per board with probe_dual.py semantics (CUDA_VISIBLE_DEVICES, one-thread BLAS '
    'env, --amp auto, --workers 2) and the notebook WORKER_ARGS incl. --flip-tta; --data/--split read the staged images-only tree.',
    'Observers wrap torch.load, compact_train.Net.__init__/load_state_dict/forward, the DataLoader (worker_init_fn added) '
    'without changing any source byte; rehearsal sentinel pass 1 injects one no-usable-window failure through window_rows.',
    'Deadline guard (strategy revision): stages stop at T0 + 8.3 h; when every arm is complete A-prime recovery applies '
    '(global genuine mean, floor(3% N) cap), otherwise the public stage-3 fallback (each model ranked over its own rows, a '
    'missing model takes the others\' rank, nothing gives 0.5); the entry never aborts and always writes a complete CSV '
    '(constant 0.5 as last resort); fallback counts are in the receipt. Rehearsal stays strict. compact_infer gets --deadline.',
    'Raptor half = A-prime R384 two-view mirror and R224 one-view workers (our readers), not the public raptor_run scripts, so the '
    'blended predictions are our own; ranks over the complete population after recovery.',
    'Docker image = A-prime v2 rehearsal image (37c64f7d...), not the public notebook image, so A-prime qualified states apply.']


def read(path):
    return json.loads(Path(path).read_bytes())


def file_sha(path):
    return WORKER.file_sha(path)


def unpack_notebook(path):
    nb = read(path)
    code = ''.join(nb['cells'][0]['source'])
    vals = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(code).body if isinstance(n, ast.Assign)
            and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('encoded', 'expected_payload')}
    R6F.require(set(vals) == {'encoded', 'expected_payload'}, 'PAYLOAD_LITERALS')
    pack = json.loads(lzma.decompress(base64.b64decode(vals['encoded'], validate=True)))
    R6F.require(R6F.digest(pack) == vals['expected_payload'], 'NOTEBOOK_PAYLOAD_DIGEST')
    return pack, code


def archived():
    R6F.require(file_sha(ARCH / 'notebook.ipynb') == ARCH_NOTEBOOK_SHA, 'EXACT_APRIME_V2_NOTEBOOK')
    pack, _ = unpack_notebook(ARCH / 'notebook.ipynb')
    R6F.require(R6F.digest(pack) == ARCH_PAYLOAD_SHA and pack['cfg']['mode'] == 'rehearsal', 'EXACT_APRIME_V2_PAYLOAD')
    for name in R7_ONLY_CFG + PRIVATE_CFG:
        R6F.require(name in pack['cfg'], 'APRIME_CFG_' + name)
    for name in NEW_SOURCES:
        R6F.require(name not in pack['sources'], 'NEW_SOURCE_NOT_IN_APRIME_' + name)
    for name in ('compact_infer.py', 'compact_train.py', 'probe_infer.py', 'probe_dual.py', 'px_cache.py', 'knee_anchor_bank_v2.py',
                 'side_common.py'):
        R6F.require(hashlib.sha256(pack['sources'][name].encode()).hexdigest() == PROFILE.SOURCE_SHA256[name], 'R6_SOURCE_IN_APRIME_' + name)
        R6F.require(pack['cfg']['source_pins'][name] == PROFILE.SOURCE_SHA256[name], 'R6_SOURCE_PIN_' + name)
    return pack


def aprime_r224_state():
    R6F.require(file_sha(APRIME_RESULT) == APRIME_RESULT_SHA, 'EXACT_APRIME_V2_RESULT')
    r = read(APRIME_RESULT)
    states = {w['state_before'] for w in r['crop224_workers']} | {w['state_after'] for w in r['crop224_workers']}
    R6F.require(len(states) == 1 and r['completed'] == 39 and r['zero_failed'] is True, 'APRIME_QUALIFIED_R224_STATE')
    return states.pop(), file_sha(APRIME_RESULT)


def cpu_pins():
    R6F.require(file_sha(CPU_PINS) == CPU_PINS_SHA, 'EXACT_CPU_PIN_RECEIPT')
    c = read(CPU_PINS)
    R6F.require(c['status'] == 'PASS_ACTUAL_CHECKPOINT_CPU_SYNTHETIC_INPUTS' and c['environment']['torch'] == '2.10.0'
                and c['environment']['timm'] == '1.0.26' and c['profile']['manifest_sha256'] == PROFILE.MANIFEST_SHA256
                and c['profile']['assets_sha256'] == dict(PROFILE.ASSET_SHA256) and c['profile']['sources_sha256'] == dict(PROFILE.SOURCE_SHA256)
                and c['flip']['equals_pinned_compact_infer'] is True and c['side']['strict'] is True, 'CPU_PIN_RECEIPT_CONTENT')
    keys = ('file', 'file_sha256', 'state', 'buffers', 'constructor', 'constructor_args')
    folds = [{k: f[k] for k in keys} for f in c['folds']]
    R6F.require([f['file'] for f in folds] == ['E_lrprior_s0_f%d.pt' % i for i in range(4)]
                and all(f['file_sha256'] == PROFILE.ASSET_SHA256[f['file']] for f in folds), 'CPU_PIN_FOLDS')
    return {'receipt_sha256': CPU_PINS_SHA, 'folds': folds, 'side': {k: c['side'][k] for k in ('state', 'buffers', 'constructor')},
            'flip': {'perm': c['flip']['perm'], 'bin_perm': c['flip']['bin_perm']}}


def r6_manifest():
    receipt = PROFILE.validate_r6_assets(R6_ASSETS)
    return read(R6_ASSETS / 'manifest.json'), receipt


def binding(pack, sources):
    ab = pack['cfg']['binding']
    manifest, receipt = r6_manifest()
    r224, r224_receipt = aprime_r224_state()
    R6F.require(file_sha(ARCH / 'kernel-metadata.json') == ARCH_META_SHA, 'EXACT_APRIME_METADATA')
    meta = read(ARCH / 'kernel-metadata.json')
    return {
        'version': R6F.VERSION, 'variant': 'R6flip: A-prime R384+R224 Raptor half, heliosli R6 fe055406 flip TTA, per-finding weights',
        'frame': R6F.FRAME, 'frame_digest': R6F.digest(R6F.FRAME), 'raptor_weight_vector': R6F.raptor_weight_vector(),
        'aprime': {'archive_notebook_sha256': ARCH_NOTEBOOK_SHA, 'archive_payload_sha256': ARCH_PAYLOAD_SHA,
                   'archive_binding_sha256': R6F.digest(ab), 'source_bundle_sha256': ab['source_bundle_sha256'],
                   'qualified_state': {k: ab['qualified_state'][k] for k in ('old_state', 'old_grid')},
                   'r224_state': r224, 'r224_result_receipt_sha256': r224_receipt, 'environment': ab['environment'],
                   'docker_image': meta['docker_image'], 'public_manifest_checkpoint_sha256': ab['public_manifest']['checkpoint_sha256'],
                   'crop224_checkpoint': pack['cfg']['ws2_binding']['crop224_checkpoint']},
        'r6': {'dataset': R6F.R6_DATASET, 'dataset_version': R6F.R6_DATASET_VERSION, 'profile_id': PROFILE.PROFILE_ID,
               'bundle_id': PROFILE.BUNDLE_ID, 'tag': PROFILE.BUNDLE_TAG, 'manifest_sha256': PROFILE.MANIFEST_SHA256,
               'asset_sha256': dict(PROFILE.ASSET_SHA256), 'source_sha256': dict(PROFILE.SOURCE_SHA256), 'manifest': manifest,
               'profile_receipt': receipt, 'cpu_pins': cpu_pins(),
               'worker': {'controller': 'probe_dual.py semantics: one compact_infer worker per GPU, loader workers 2, amp auto',
                          'flip_tta': True, 'deadline': 0, 'module': WORKER.WORKER_MODULE, 'env': WORKER.PROBE_DUAL_ENV}},
        'public_recipe_reference': {'kernel': 'heliosli/blend-mine-in-private-lb-luck-to-all-upvoters',
                                    'notebook_sha256': PUBLIC_RECIPE_NOTEBOOK_SHA,
                                    'role': 'recipe facts only (bundle, flip, A224, weights); none of its cells is shipped or executed'},
        'official_columns': ab['columns'], 'labels': R6F.LABELS, 'uid': 'StudyInstanceUID',
        'tolerances': {'r6_cross_process_board_dprob': ENTRY.R6_CROSS_TOLERANCE, 'r6_fresh_reload_dprob': WORKER.RELOAD_DPROB_TOL,
                       'r6_reconstruction_abs': WORKER.RECONSTRUCTION_ATOL, 'aprime_arms_exact': ENTRY.EXACT_TOLERANCE},
        'limits': {'actual_seconds': ENTRY.MAX_SECONDS, 'projection_gate_seconds': ENTRY.PROJECTION_GATE_SECONDS,
                   'projection_population': ENTRY.POPULATION},
        'new_source_sha256': {n: hashlib.sha256(sources[n].encode()).hexdigest() for n in NEW_SOURCES},
        'source_bundle_sha256': R6F.digest({n: hashlib.sha256(s.encode()).hexdigest() for n, s in sorted(sources.items())}),
        'deviations': DEVIATIONS}


def packet(mode='rehearsal', qualified=None):
    R6F.require(mode in ('rehearsal', 'entry'), 'MODE')
    pack = archived()
    sources = dict(pack['sources'])
    for name in NEW_SOURCES:
        text = (PROJECT / name).read_text()
        compile(text, name, 'exec')
        sources[name] = text
    cfg = copy.deepcopy(pack['cfg'])
    b = binding(pack, sources)
    for key in R7_ONLY_CFG:
        cfg.pop(key)
    cfg['mode'] = mode
    cfg['r6flip_binding'] = b
    if mode == 'entry':
        R6F.require(qualified is not None, 'ENTRY_REQUIRES_QUALIFIED_REHEARSAL')
        for key in PRIVATE_CFG:
            cfg.pop(key)
        cfg['binding'] = {k: v for k, v in cfg['binding'].items() if k not in PRIVATE_BINDING}
        cfg['r224_state_pin'] = qualified['r224_state']
        cfg['qualified_rehearsal'] = qualified
    else:
        R6F.require(len(cfg['probes']) == 36 and len(cfg['probe_order']) == 36 and len(cfg['s6_sentinels']) == 4
                    and len(cfg['old_sentinels']) == 4 and len(cfg['development_windows']) == 870, 'FROZEN39_REGISTRY')
    return {'cfg': cfg, 'codecs': pack['codecs'], 'sources': sources}


BOOTSTRAP_HEAD = """import time
started=time.monotonic();started_epoch=time.time()
import os,sys,json,base64,lzma,hashlib,signal,subprocess
from pathlib import Path
def deadline(signum,frame):raise TimeoutError('Actual nine-hour runtime exhausted')
signal.signal(signal.SIGALRM,deadline);signal.alarm(int(8.9*3600))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',BUILD_MONAI='0',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
"""
BOOTSTRAP_TAIL = """
payload=json.loads(lzma.decompress(base64.b64decode(encoded,validate=True)))
if hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()!=expected_payload:raise ValueError('R6flip payload changed')
project=Path('/kaggle/working/r6flip-project');code=project/'source'
for name,content in payload['sources'].items():
    rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe source path')
    p=project/name if name.startswith('knee/') else code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode())
sys.path.insert(0,str(project))
from knee.public_r6flip_entry import run
from knee.public_s6_qualification import file_sha,save,require
cfg=payload['cfg'];codecs=payload['codecs'];out=Path('/kaggle/working/r6flip')
def mount(ref,required):
    owner,slug=ref.split('/')[:2];roots=[Path('/kaggle/input/datasets')/owner/slug,Path('/kaggle/input')/slug]
    matches=[p for p in roots if all((p/n).is_file() for n in required)]
    require(len(matches)==1,'UNIQUE_IMMUTABLE_ATTACHMENT');return matches[0].resolve()
try:
    require(sys.version_info[:2]==(3,12),'PYTHON_ABI')
    b6=cfg['r6flip_binding']
    codec=mount('goodpjw2008/rsna-knee-2-5d-convnext-reader',list(codecs))
    for name,pin in codecs.items():require(file_sha(codec/name)==pin,'EXACT_OFFLINE_CODEC')
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*[str(codec/n) for n in codecs]],check=True)
    r6=mount(b6['r6']['dataset'],['manifest.json',*b6['r6']['asset_sha256']])
    require(file_sha(r6/'manifest.json')==b6['r6']['manifest_sha256'],'EXACT_R6_V1_MANIFEST')
    anatomy=mount('heliosli/rsna-orthofoundation-fold0-epoch7-20261004',['anatomy.pt'])/'anatomy.pt'
    require(file_sha(anatomy)==b6['r6']['manifest']['anchor_weights_sha256'],'EXACT_ANATOMY')
    weights=mount('nartaa/rsna-knee-publication-swa-weights-20261007',[cfg['public_manifest']['checkpoint_file'],b6['aprime']['crop224_checkpoint']['file']])
    roots=[Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')]
    matches=[p for p in roots if all((p/n).is_file() for n in ('test.csv','test_series.csv','sample_submission.csv'))]
    require(len(matches)==1,'UNIQUE_ACTUAL_TEST_ROOT')
    cfg.update(code=str(code),r6_mount=str(r6),anatomy=str(anatomy),
        control_checkpoint=str(weights/cfg['public_manifest']['checkpoint_file']),
        crop224_checkpoint=str(weights/b6['aprime']['crop224_checkpoint']['file']))
    save(project/'runtime-config.json',cfg)
    report=run(matches[0],cfg,out,started=started,started_epoch=started_epoch)
    print('R6FLIP_DONE',json.dumps({'status':report['status'],'mode':report['mode'],'completed':report['completed'],
        'elapsed_seconds':report['elapsed_seconds'],'conservative_seconds':(report.get('projection') or {}).get('conservative_seconds'),
        'point_seconds':(report.get('projection') or {}).get('point_seconds')}),flush=True)
    if report['mode']=='entry' and report['status'] not in ('PASS','PASS_WITH_FALLBACK','PASS_WITH_FALLBACK_AFTER_ERROR','EMERGENCY_CONSTANT'):raise RuntimeError('entry wrote no submission')
except BaseException as e:
    save(out.parent/'r6flip-setup-failure-private.json',{'type':type(e).__name__,'detail':str(e)[:4000]})
    sub=Path('/kaggle/working/submission.csv')
    if payload['cfg']['mode']=='entry' and not sub.exists():
        from knee.public_r6flip_entry import emergency_csv
        for root in (Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),Path('/kaggle/input/rsna-knee-abnormality-detection')):
            try:emergency_csv(root,sub);print('R6FLIP_EMERGENCY_CONSTANT_SUBMISSION',flush=True);break
            except Exception:pass
    if not sub.exists() or payload['cfg']['mode']!='entry':raise
finally:
    signal.alarm(0)
"""


def bootstrap(pack):
    encoded = base64.b64encode(lzma.compress(R6F.canonical(pack), preset=9)).decode()
    script = BOOTSTRAP_HEAD + 'encoded=%r\nexpected_payload=%r\n' % (encoded, R6F.digest(pack)) + BOOTSTRAP_TAIL
    compile(script, '<r6flip-bootstrap>', 'exec')
    return script


def notebook(pack, mode):
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python3', 'language': 'python'}},
          'cells': [{'id': 'r6flip-' + mode, 'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                     'source': bootstrap(pack).splitlines(True)}]}
    raw = json.dumps(nb, indent=2) + '\n'
    R6F.require(len(raw.encode()) < 900000, 'NOTEBOOK_900K_CAP')
    return raw


def metadata(slug):
    R6F.require(isinstance(slug, str) and 3 <= len(slug) <= 50 and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in slug), 'SAFE_SLUG')
    R6F.require(file_sha(ARCH / 'kernel-metadata.json') == ARCH_META_SHA, 'EXACT_APRIME_METADATA')
    arch = read(ARCH / 'kernel-metadata.json')
    R6F.require(arch['machine_shape'] == 'NvidiaTeslaT4' and arch['enable_internet'] is False and arch['is_private'] is True, 'APRIME_METADATA')
    return {'id': '%s/%s' % (OWNER, slug), 'title': slug, 'code_file': 'notebook.ipynb', 'language': 'python', 'kernel_type': 'notebook',
            'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False, 'competition_sources': [COMP],
            'dataset_sources': list(DATASETS), 'kernel_sources': [], 'model_sources': [], 'docker_image': arch['docker_image'],
            'machine_shape': 'NvidiaTeslaT4'}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def emit(dest, slug, pack, mode, extra=None):
    dest = Path(dest)
    R6F.require(not dest.exists() and dest.resolve().is_relative_to(OUT.resolve()), 'FRESH_IGNORED_OUTPUT')
    raw = notebook(pack, mode)
    dest.mkdir(parents=True)
    (dest / 'notebook.ipynb').write_text(raw)
    save_json(dest / 'kernel-metadata.json', metadata(slug))
    tracked = [*NEW_SOURCES, 'scripts/build_r6flip_entry.py', 'scripts/qualify_r6flip_cpu.py']
    changed = subprocess.check_output(['git', 'status', '--porcelain', '--', *tracked], cwd=PROJECT, text=True).strip()
    receipt = {'status': 'PREPARATION_ONLY' if mode == 'rehearsal' else 'QUALIFIED_ENTRY_BUILD', 'mode': mode, 'slug': slug,
               'binding_sha256': R6F.digest(pack['cfg']['r6flip_binding']), 'payload_sha256': R6F.digest(pack),
               'new_source_sha256': pack['cfg']['r6flip_binding']['new_source_sha256'],
               'source_bundle_sha256': pack['cfg']['r6flip_binding']['source_bundle_sha256'], 'source_files': len(pack['sources']),
               'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
               'source_state': ('uncommitted implementation' if changed else 'committed implementation') + '; exact file hashes bound',
               'builder_sha256': file_sha(__file__), 'files': {n: file_sha(dest / n) for n in ('notebook.ipynb', 'kernel-metadata.json')},
               'notebook_bytes': len(raw.encode()), 'runtime_checks_exercised': False, 'deviations': DEVIATIONS, **(extra or {})}
    save_json(dest / 'build-receipt.json', receipt)
    save_json(dest / 'MANIFEST.json', {'mode': mode, 'payload_sha256': R6F.digest(pack), 'files': {
        n: {'sha256': file_sha(dest / n), 'bytes': (dest / n).stat().st_size} for n in ('notebook.ipynb', 'kernel-metadata.json', 'build-receipt.json')}})
    return receipt


def verify_build(root, mode='rehearsal', qualified=None):
    root = Path(root)
    m, r, meta = read(root / 'MANIFEST.json'), read(root / 'build-receipt.json'), read(root / 'kernel-metadata.json')
    R6F.require(set(m['files']) == {'notebook.ipynb', 'kernel-metadata.json', 'build-receipt.json'}, 'MANIFEST_COVERAGE')
    for n, pin in m['files'].items():
        R6F.require(pin == {'sha256': file_sha(root / n), 'bytes': (root / n).stat().st_size}, 'IMMUTABLE_BUILD_' + n)
    expected = packet(mode, qualified)
    actual, code = unpack_notebook(root / 'notebook.ipynb')
    R6F.require(actual == expected and code == bootstrap(expected), 'EXACT_PACKET_REPLAY')
    R6F.require(meta == metadata(meta['title']) and meta['id'] == '%s/%s' % (OWNER, meta['title']) and 'timeout' not in meta,
                'PRIVATE_OFFLINE_T4_TITLE_ID')
    R6F.require(r['mode'] == mode and r['payload_sha256'] == R6F.digest(expected) == m['payload_sha256']
                and r['files'] == {n: file_sha(root / n) for n in ('notebook.ipynb', 'kernel-metadata.json')}, 'BUILD_RECEIPT')
    return r, expected


def verify_result(result_root, rehearsal_cfg):
    """Replay a collected rehearsal (the /kaggle/working/r6flip folder): every gate, ledger and the CSV."""
    root = Path(result_root)
    r = read(root / 'receipt.json')
    q = ENTRY.require_rehearsal(r, rehearsal_cfg, root, root / 'visible-predictions.csv')
    R6F.require(r.get('qualification') == q, 'SAVED_QUALIFICATION_REPLAY')
    R6F.require(r['status'] == ('PASS' if q['projection_gate_pass'] else 'FAIL_PROJECTION_ONLY'), 'SAVED_STATUS')
    crop = {w['state_before'] for w in r['workers']['crop224']}
    return {'status': r['status'], 'receipt_sha256': file_sha(root / 'receipt.json'),
            'csv_sha256': file_sha(root / 'visible-predictions.csv'), 'r224_state': crop.pop(),
            'r6_state': r['workers']['r6'][0]['state_before'], 'anatomy_state': q['r6']['anatomy_state'],
            'projection': {k: r['projection'][k] for k in ('point_seconds', 'conservative_seconds', 'gate_seconds', 'gate_pass')},
            'elapsed_seconds': r['elapsed_seconds']}


def build_rehearsal(dest=None):
    slug = SLUGS['rehearsal']
    return emit(Path(dest or OUT / 'kaggle' / slug), slug, packet('rehearsal'), 'rehearsal')


def build_entry(rehearsal_build, rehearsal_result, waiver=None, dest=None, slug=None):
    prepared, rehearsal_pack = verify_build(rehearsal_build)
    checked = verify_result(rehearsal_result, rehearsal_pack['cfg'])
    if checked['status'] != 'PASS':
        R6F.require(waiver is not None, 'PROJECTION_GATE_FAILED_NAMED_WAIVER_REQUIRED')
        w = read(waiver)
        R6F.require(w.get('scope') == 'projection_only' and w.get('rehearsal_receipt_sha256') == checked['receipt_sha256']
                    and isinstance(w.get('authorized_by'), str) and w['authorized_by'].strip() and w.get('conservative_seconds')
                    == checked['projection']['conservative_seconds'], 'NAMED_PROJECTION_WAIVER')
        checked['waiver'] = {'file_sha256': file_sha(waiver), **w}
    qualified = {**checked, 'rehearsal_build': {'slug': prepared['slug'], 'payload_sha256': prepared['payload_sha256'],
                                                'files': prepared['files']}}
    slug = slug or SLUGS['entry']
    return emit(Path(dest or OUT / 'kaggle' / slug), slug, packet('entry', qualified), 'entry', {'qualified_rehearsal': qualified})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='command')
    r = sub.add_parser('rehearsal')
    r.add_argument('--destination')
    e = sub.add_parser('entry')
    e.add_argument('--rehearsal-build', required=True)
    e.add_argument('--rehearsal-result', required=True)
    e.add_argument('--waiver')
    e.add_argument('--destination')
    e.add_argument('--slug')
    v = sub.add_parser('verify')
    v.add_argument('--build', default=str(OUT / 'kaggle' / SLUGS['rehearsal']))
    a = p.parse_args(argv)
    if a.command in (None, 'rehearsal'):
        out = build_rehearsal(getattr(a, 'destination', None))
    elif a.command == 'entry':
        out = build_entry(a.rehearsal_build, a.rehearsal_result, a.waiver, a.destination, a.slug)
    else:
        out, _ = verify_build(a.build)
    print(json.dumps({k: out[k] for k in ('status', 'slug', 'payload_sha256', 'files', 'notebook_bytes')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
