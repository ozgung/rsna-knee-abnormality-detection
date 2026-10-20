"""Build the sealed R6-flip Kaggle packets. Never pushes, launches or submits anything.

  rehearsal  private/offline 2xT4 frozen-39 qualification packet (default slug r6flip-rehearsal-20261017)
  entry      image-only entry, only from this exact rehearsal build plus its collected result, replayed
             independently (projection-only failures need an explicit named waiver file)
  verify     re-derive the emitted rehearsal packet byte for byte

The payload is A-prime's sealed v2 rehearsal payload (ws2-r224-v2-qual-20261015, notebook 412eef8f...) with every
source byte kept, R7-only configuration removed, and four new modules: the R6 profile, the frozen frame, the R6
worker/observer and the controller. R6 assets come from heliosli/rsna-knee-r6-frozen-fe055406-w1 version 1.
--variant p6 / p6p (dual compact) also ship knee/public_r7_profile.py and mount heliosli/rsna-knee-compact-r7cnxfsss version 1
(A-prime's R7 pins); the rehearsal adds A-prime's qualified R7 probabilities on the frozen 39 as a private reference.
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
from knee import public_s6c as S6C

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
VARIANT_DATE = '20261018'
VARIANT_DATES = {'p6': '20261019', 'p6p': '20261019', 'p6s': '20261020', 'p6sp': '20261020', 'p9': '20261020', 'p9r': '20261020',
                 'p9s': '20261020', 'p9sr': '20261020', 'p10': '20261021'}


def slug_for(variant, mode):
    """The kernel slug (the title must equal it). No variant keeps the existing R6F-pure slugs."""
    R6F.require(variant is None or variant in R6F.VARIANTS, 'KNOWN_VARIANT')
    return SLUGS[mode] if variant is None else 'r6flip-%s-%s-%s' % (variant, mode, VARIANT_DATES.get(variant, VARIANT_DATE))


# P6 dual compact: R7 v1 bundle (A-prime pins), its actual-checkpoint CPU pins and A-prime's qualified R7 run on the frozen 39.
R7_SOURCES = ('knee/public_r7_profile.py',)
R7_ASSETS = OUT / 'analysis/lanea-20261019/r7-v1-assets'
R7_CPU_PINS = OUT / 'analysis/lanea-20261019/r7-cpu-pins.json'
R7_CPU_PINS_SHA = '6d749b29ebca935badab2482602dc40460091ea1b2f2406907feb14518a38190'
APRIME_R7_BASE = OUT / 'kaggle/ws2-r224-v2-qual-20261015-result/ws2-a-prime/base'
APRIME_R7_BASE_SHA = '1ac92b6a7ecf7bd8b90b52f7b2aad7f6f4273a9b319d24744b097e6ab29931f2'     # = A-prime v2 result's base_receipt_sha256
APRIME_R7_BOARD_SHA = ('2527f350813384ecd621db713253c1ce76ae7a70deb274f4cac99c4111c5313d',
                       '0ae1fe71bba29bdb94730e489da604b8065e0766a303fc3cf7fe155883615f71')


P6_REHEARSAL_OUT = OUT / 'kaggle/r6flip-p6-rehearsal-20261019-out/r6flip'
P6_REHEARSAL_RECEIPT_SHA = '5548d5c570f095655960c13f5111783651cf6567eb4a8c7d13ca3968914ada09'     # FAIL_PROJECTION_ONLY, every other gate PASS


def p6_reference():
    """P6s rehearsal fixture: the separate-pass P6 rehearsal's R6 plain and R7 plain probabilities on the frozen 39 (private)."""
    R6F.require(file_sha(P6_REHEARSAL_OUT / 'receipt.json') == P6_REHEARSAL_RECEIPT_SHA, 'EXACT_P6_REHEARSAL_RECEIPT')
    r = read(P6_REHEARSAL_OUT / 'receipt.json')
    R6F.require(r['status'] == 'FAIL_PROJECTION_ONLY' and r['qualification']['all_ledgers_recomputed'] is True, 'P6_REHEARSAL_QUALIFIED')
    out = {}
    for arm in ('r6', 'r7'):
        out[arm] = {}
        for board, w in enumerate(r['workers'][arm]):
            path = P6_REHEARSAL_OUT / ('board%d-%s-production' % (board, arm)) / arm / 'predictions.npz'
            R6F.require(w['ledger_sha256'] == file_sha(path) and w['status'] == 'PASS', 'P6_REFERENCE_LEDGER_BINDING')
            for row in w['records']:
                R6F.require(row['status'] == 'genuine' and row['id'] not in out[arm], 'P6_REFERENCE_GENUINE')
                out[arm][row['id']] = row['probabilities']
    ids = json.loads((P6_REHEARSAL_OUT / 'assignment.json').read_bytes())['ids']
    R6F.require(sorted(out['r6']) == sorted(out['r7']) == sorted(ids) and len(ids) == 39, 'P6_REFERENCE_FROZEN39')
    return {'r6': {s: out['r6'][s] for s in ids}, 'r7': {s: out['r7'][s] for s in ids}, 'receipt_sha256': P6_REHEARSAL_RECEIPT_SHA,
            'source': 'ozgung/r6flip-p6-rehearsal-20261019 v1 (separate R6 plain and R7 plain stages)'}


def has_r7(variant):
    return variant is not None and 'r7' in R6F.compact_arm_names(R6F.VARIANTS[variant])


def own_dual(variant):
    """P9: our S6 members as the second compact arm next to the R6 bundle (knee.public_r6flip.own_compact_arm)."""
    return variant is not None and R6F.own_compact_arm(R6F.VARIANTS[variant])


def has_s6c(variant):
    """Variants that ship the s6c worker and mount our members: P7-prime (compact replaced) and P9 (dual)."""
    return variant in R6F.COMPACT_VARIANTS or own_dual(variant)


# na-full K62 one-view Raptor: the A-triple-prime packet's worker, config and checkpoint binding, taken from its sealed r3 rehearsal.
NAFULL_R3 = OUT / 'kaggle/nafull-aprime-rehearsal-r3-20261016'
NAFULL_R3_NOTEBOOK_SHA = '1d7089e0df4283a45108d27c1ca5cc42604bdcaf49d1a1592b38aec519a519ce'
NAFULL_R3_PAYLOAD_SHA = '68cae8745bfeddb449b541347fc371a083ce92966fee56076d61722ad4d15b98'
NAFULL_SOURCES = ('knee/public_nafull_aprime.py', 'knee/public_nafull_aprime_runtime.py', 'knee/public_nafull_aprime_recovery_adapter.py',
                  'knee/public_nafull_aprime_entry.py', 'knee/public_nafull_aprime_rehearsal.py', 'knee/public_s6_recovery_b0.py',
                  'knee/public_s6_recovery_b0prime.py', 'knee/public_s6_recovery_b0_rehearsal.py',
                  'knee/public_s6_recovery_b0prime_rehearsal.py', 'knee/public_nafull_raptor_input.py', 'knee/public_nafull_reference.py')
NAFULL_WORKER_KEYS = ('public_source', 'decoder_source', 'public_manifest', 'control_checkpoint', 'control_state_sha256',
                      'control_grid_sha256', 'old_sentinels', 'raptor62_reference')
NEW_SOURCES = ('knee/public_r6_profile.py', 'knee/public_r6flip.py', 'knee/public_r6flip_worker.py', 'knee/public_r6flip_entry.py')
# P7-prime (Lane C): our S6 seed ensemble in the compact slot. Shipped only by variants in R6F.COMPACT_VARIANTS; the
# captured solo controller (6e87350d...) is shipped byte for byte for its pure launcher/command helpers.
S6C_SOURCES = ('knee/public_s6c.py', 'knee/public_s6c_worker.py', 'knee/public_s6c_shared.py')
S6C_CAPTURED = OUT / 'research/public-0957-20261015/kernels/nartaa_rsna-knee-s6fall-solo/rsna-knee-s6fall-solo.py'
# Sealed seed set (pre-registered rule in docs/experiments/lane-c-seeds-20261019.md): filled from the verified exports.
# Amendment 1 (parent, 2026-10-20 08:40 UTC): {v3-s42, v3-s43, Lane D fastMRI student}, equal-weight mean rank; all labels v3.
S6C_MODELS = [
    {'seed': 42, 'labels': 'v3', 'file': 'E_prior_s42_v3all.pt', 'sha256': '03da2cacdbc12305519e05c8e9a5626f14eb71f49bc26d56e3f2ced35f997e43',
     'bytes': 200188027, 'dataset': 'ozgung/knee-s6-lanec-seeds-20261019/pyTorch/v3-s42', 'dataset_version': 1},
    {'seed': 42, 'slot': 1, 'labels': 'v3', 'file': 'E_prior_s42_fmall.pt',
     'sha256': '6a13f52e424cbfd4945883273fd081dfb2f25b171adb5b6521fab4d382466b57', 'bytes': 200186923,
     'dataset': 'ozgung/knee-laned-fmall-s42-20261020', 'dataset_version': 1},
    {'seed': 43, 'labels': 'v3', 'file': 'E_prior_s43_v3all.pt', 'sha256': 'a99ad89705f8297ac86659ce8e5a092cd626d851d7d51331c59f1b6920c69690',
     'bytes': 200188027, 'dataset': 'ozgung/knee-s6-lanec-seeds-20261019/pyTorch/v3-s43', 'dataset_version': 1}]
# A100 fp16 per-member read-out of the frozen-36 probes (lanec2, one compact_infer over all three members)
S6C_REFERENCE = {'path': OUT / 'lanec-20261019/p7/reference-p7.json',
                 'sha256': 'd41d7952b8b5e47c78ae431f03eafc2a4f2951772a687235f677f8935ca48504'}
# P9 member set (pre-registered in docs/experiments/lane-p9-20261020.md before any P9 build): the fastMRI student and v3-s43,
# equal weight, both labels v3, loader order; v3-s43 from its private dataset copy (same bytes as the Model export, SHA256SUMS
# read on Oct 20). v3-s42 is added only under the pre-registered runtime rule. The P7 reference covers both aliases.
S6C_MODELS_P9 = [
    {'seed': 42, 'slot': 1, 'labels': 'v3', 'file': 'E_prior_s42_fmall.pt',
     'sha256': '6a13f52e424cbfd4945883273fd081dfb2f25b171adb5b6521fab4d382466b57', 'bytes': 200186923,
     'dataset': 'ozgung/knee-laned-fmall-s42-20261020', 'dataset_version': 1},
    {'seed': 43, 'labels': 'v3', 'file': 'E_prior_s43_v3all.pt', 'sha256': 'a99ad89705f8297ac86659ce8e5a092cd626d851d7d51331c59f1b6920c69690',
     'bytes': 200188027, 'dataset': 'ozgung/knee-s6-v3all-s43-20261020', 'dataset_version': 1}]
S6C_MODELS_P10 = None                           # all-ours compact: sealed when Lane L4 / Lane F exports exist (an unsealed list refuses to build)
S6C_MODELS_BY_VARIANT = {'p7': 'S6C_MODELS', 'p9': 'S6C_MODELS_P9', 'p9r': 'S6C_MODELS_P9', 'p9s': 'S6C_MODELS_P9', 'p9sr': 'S6C_MODELS_P9',
                         'p10': 'S6C_MODELS_P10'}
# P9s shared-decode fixture: the P9 rehearsal's separate s6c worker output on the frozen 39 (per-member probabilities, private).
P9_REHEARSAL_OUT = OUT / 'kaggle/r6flip-p9-rehearsal-20261020-out/r6flip'
P9_REHEARSAL_RECEIPT_SHA = None                 # pinned once the P9 rehearsal result is collected and verified


def p9_reference():
    R6F.require(P9_REHEARSAL_RECEIPT_SHA is not None and file_sha(P9_REHEARSAL_OUT / 'receipt.json') == P9_REHEARSAL_RECEIPT_SHA,
                'EXACT_P9_REHEARSAL_RECEIPT')
    r = read(P9_REHEARSAL_OUT / 'receipt.json')
    R6F.require(r['status'] in ('PASS', 'FAIL_PROJECTION_ONLY') and r['qualification']['all_ledgers_recomputed'] is True, 'P9_REHEARSAL_QUALIFIED')
    aliases = r['binding']['s6c']['aliases']
    out = {}
    for board, w in enumerate(r['workers']['s6c']):
        path = P9_REHEARSAL_OUT / ('board%d-s6c-production' % board) / 's6c' / 'predictions.npz'
        R6F.require(w['ledger_sha256'] == file_sha(path) and w['status'] == 'PASS' and w['compact']['models'] == aliases, 'P9_REFERENCE_LEDGER_BINDING')
        for row in w['records']:
            R6F.require(row['status'] == 'genuine' and row['id'] not in out and len(row['seed_probabilities']) == len(aliases), 'P9_REFERENCE_GENUINE')
            out[row['id']] = row['seed_probabilities']
    ids = json.loads((P9_REHEARSAL_OUT / 'assignment.json').read_bytes())['ids']
    R6F.require(sorted(out) == sorted(ids) and len(ids) == 39, 'P9_REFERENCE_FROZEN39')
    return {'aliases': list(aliases), 'probabilities': {s: out[s] for s in ids}, 'receipt_sha256': P9_REHEARSAL_RECEIPT_SHA,
            'source': 'ozgung/r6flip-p9-rehearsal-20261020 v1 (separate s6c worker stage, two members)'}
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


def nafull_block():
    """(cfg['nafull'] rehearsal block, extra shipped sources) replayed from the sealed A-triple-prime r3 rehearsal packet."""
    from knee import public_nafull_aprime as NA
    R6F.require(file_sha(NAFULL_R3 / 'notebook.ipynb') == NAFULL_R3_NOTEBOOK_SHA, 'EXACT_NAFULL_R3_NOTEBOOK')
    r3, _ = unpack_notebook(NAFULL_R3 / 'notebook.ipynb')
    R6F.require(R6F.digest(r3) == NAFULL_R3_PAYLOAD_SHA and r3['cfg']['mode'] == 'rehearsal', 'EXACT_NAFULL_R3_PAYLOAD')
    sources = {}
    for name in NAFULL_SOURCES:
        text = (PROJECT / name).read_text()
        R6F.require(text == r3['sources'][name], 'NAFULL_SOURCE_EQUALS_SEALED_R3_' + name)
        compile(text, name, 'exec')
        sources[name] = text
    cfg = r3['cfg']
    worker, base = cfg['nafull_worker_cfg'], cfg['nafull_base_binding']
    state = {'old_state': base['qualified_state']['old_state'], 'old_grid': base['qualified_state']['old_grid']}
    R6F.require(state == {'old_state': NA.CHECKPOINT['state_sha256'], 'old_grid': NA.CONSTRUCTOR_GRID_SHA256}
                and worker['control_state_sha256'] == NA.CHECKPOINT['state_sha256']
                and worker['control_grid_sha256'] == NA.CONSTRUCTOR_GRID_SHA256
                and worker['public_manifest']['checkpoint_sha256'] == NA.CHECKPOINT['sha256']
                and worker['control_checkpoint'] == NA.CHECKPOINT['file'], 'NAFULL_R3_CHECKPOINT_PINS')
    w = {k: copy.deepcopy(worker[k]) for k in NAFULL_WORKER_KEYS}
    w['probes'] = {sid: {'old': copy.deepcopy(p['old'])} for sid, p in worker['probes'].items()}
    return ({'profile': NA.VERSION, 'checkpoint': dict(NA.CHECKPOINT), 'worker': w, 'qualified_state': state,
             'reader_reference': copy.deepcopy(base['nafull_reader_reference']), 'cpu_binding': copy.deepcopy(base['nafull_cpu_binding']),
             'final_export_verification_sha256': base['final_export_verification_sha256'],
             'sealed_from': {'kernel': 'ozgung/nafull-aprime-rehearsal-r3-20261016', 'notebook_sha256': NAFULL_R3_NOTEBOOK_SHA,
                             'payload_sha256': NAFULL_R3_PAYLOAD_SHA, 'role': 'worker modules, K62 profile, checkpoint binding and '
                             'reader references of the A-triple-prime packet, unchanged; its R7/R224 parts are not used'}},
            sources)


def variant_frame(variant):
    """The frozen frame with the variant's Raptor arms and weights (the default FRAME is untouched)."""
    arms, weights, default = R6F.resolve_blend(R6F.VARIANTS[variant])
    frame = copy.deepcopy(R6F.FRAME)
    frame.update(frame_id='%s-%s' % (R6F.FRAME_ID, variant), version=R6F.VERSION + '+' + variant)
    frame['raptor_half'] = {'arms': dict(arms), 'source': 'A-prime R384/R224 workers unchanged' + (
        ' plus the A-triple-prime na-full K62 one-view worker' if any(a == 'nafull' for a, _ in arms) else '')}
    frame['raptor_weight_per_finding'] = dict(weights)
    frame['raptor_weight_default'] = default
    frame['excluded'] = [x for x in frame['excluded'] if x != 'na-full']
    compact, flip = R6F.resolve_compact(R6F.VARIANTS[variant])
    if own_dual(variant):
        block = s6c_block(variant)[0]
        frame['compact_half'] = {'arms': dict(compact), 'flip_tta': flip, 'combine': 'weighted sum of average percentile ranks',
                                 'r6': {'bundle_id': PROFILE.BUNDLE_ID, 'dataset': '%s/%d' % (R6F.R6_DATASET, R6F.R6_DATASET_VERSION),
                                        'worker': 'compact_infer.py v4c (captured, unchanged), plain pass, 2 loader workers'},
                                 's6c': {'version': S6C.VERSION, 'members': list(block['aliases']), 'labels': block['labels'],
                                         'aggregation': S6C.AGGREGATION, 'flip_tta': False, 'worker': 'knee.public_s6c_worker, last stage'},
                                 'guard_fallback': 'studies without the own stage keep rank(R6 plain) as the compact slot'}
        if R6F.own_shared(R6F.VARIANTS[variant]):
            frame['compact_half']['s6c'].update(worker='knee.public_s6c_shared inside the R6 plain worker: members rendered at 320 px from '
                                                       'R6\'s native cache (one decode per study), scored after R6\'s last fold')
            frame['compact_half']['guard_fallback'] = ('members stop first near the deadline (R6-only pace x 1.3); studies without members '
                                                       'keep rank(R6 plain) as the compact slot')
    elif compact != R6F.DEFAULT_COMPACT:
        from knee import public_r7_profile as R7P
        frame['compact_half'] = {'arms': dict(compact), 'flip_tta': flip, 'combine': 'weighted sum of average percentile ranks',
                                 'r6': {'bundle_id': PROFILE.BUNDLE_ID, 'dataset': '%s/%d' % (R6F.R6_DATASET, R6F.R6_DATASET_VERSION)},
                                 'r7': {'bundle_id': R7P.BUNDLE_ID, 'dataset': '%s/%d' % (R7P.DATASET, R7P.DATASET_VERSION)},
                                 'worker': 'compact_infer.py v4c (captured, unchanged), plain pass, 2 loader workers; R7 stage last',
                                 'guard_fallback': 'studies without R7 keep rank(R6) as the compact slot'}
        if R6F.compact_shared(R6F.VARIANTS[variant]):
            frame['compact_half'].update(worker='compact_infer.py v4c (captured, unchanged) R6 plain; R7 fold models scored in the same '
                                                'worker on R6\'s rendered windows (one decode per study)',
                                         guard_fallback='R7 stops first near the deadline (R6-only pace x 1.3); studies without R7 '
                                                        'keep rank(R6 plain) as the compact slot (R6 flip is not run in P6s)')
    return frame


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


def r7_cpu_pins():
    from knee import public_r7_profile as R7P
    R6F.require(file_sha(R7_CPU_PINS) == R7_CPU_PINS_SHA, 'EXACT_R7_CPU_PIN_RECEIPT')
    c = read(R7_CPU_PINS)
    R6F.require(c['status'] == 'PASS_ACTUAL_CHECKPOINT_CPU_SYNTHETIC_INPUTS' and c['environment']['torch'] == '2.10.0'
                and c['environment']['timm'] == '1.0.26' and c['profile']['manifest_sha256'] == R7P.MANIFEST_SHA256
                and c['profile']['profile_id'] == R7P.PROFILE_ID and c['profile']['assets_sha256'] == dict(R7P.ASSET_SHA256)
                and c['profile']['sources_sha256'] == dict(PROFILE.SOURCE_SHA256) and c['flip'] is None
                and c['side']['strict'] is True, 'R7_CPU_PIN_RECEIPT_CONTENT')
    keys = ('file', 'file_sha256', 'state', 'buffers', 'constructor', 'constructor_args')
    folds = [{k: f[k] for k in keys} for f in c['folds']]
    R6F.require([f['file'] for f in folds] == ['E_lrprior_s0_f%d.pt' % i for i in range(4)]
                and all(f['file_sha256'] == R7P.ASSET_SHA256[f['file']] for f in folds), 'R7_CPU_PIN_FOLDS')
    return {'receipt_sha256': R7_CPU_PINS_SHA, 'folds': folds, 'side': {k: c['side'][k] for k in ('state', 'buffers', 'constructor')},
            'flip': None}


def r7_block(pack):
    """The binding's R7 profile: A-prime's R7 dataset/manifest pins (asset_binding, manifest of its sealed cfg) and CPU pins."""
    from knee import public_r7_profile as R7P
    receipt = R7P.validate_r7_assets(R7_ASSETS)
    manifest = read(R7_ASSETS / 'manifest.json')
    ab = pack['cfg']['asset_binding']
    R6F.require(manifest == pack['cfg']['manifest'] and ab == {'bundle_id': R7P.BUNDLE_ID, 'dataset': R7P.DATASET,
                                                               'dataset_version': R7P.DATASET_VERSION,
                                                               'manifest_sha256': R7P.MANIFEST_SHA256}, 'APRIME_R7_PINS')
    return {'dataset': R7P.DATASET, 'dataset_version': R7P.DATASET_VERSION, 'profile_id': R7P.PROFILE_ID,
            'bundle_id': R7P.BUNDLE_ID, 'tag': R7P.BUNDLE_TAG, 'manifest_sha256': R7P.MANIFEST_SHA256,
            'asset_sha256': dict(R7P.ASSET_SHA256), 'source_sha256': dict(PROFILE.SOURCE_SHA256), 'manifest': manifest,
            'profile_receipt': receipt, 'cpu_pins': r7_cpu_pins(),
            'worker': {'controller': 'same compact_infer worker as R6, plain pass (no --flip-tta), loader workers 2, amp auto',
                       'flip_tta': False, 'module': WORKER.WORKER_MODULE, 'env': WORKER.PROBE_DUAL_ENV},
            'aprime_pins': {'module': 'knee/public_r7_recovery_entry.py', 'asset_binding': ab}}


def r7_reference(pack):
    """A-prime's qualified R7 probabilities on the frozen 39 (its v2 rehearsal base run), bound by receipt SHA256 (private)."""
    R6F.require(file_sha(APRIME_RESULT) == APRIME_RESULT_SHA and read(APRIME_RESULT)['base_receipt_sha256'] == APRIME_R7_BASE_SHA
                and file_sha(APRIME_R7_BASE / 'receipt.json') == APRIME_R7_BASE_SHA, 'APRIME_R7_BASE_BINDING')
    probabilities = {}
    for board, pin in enumerate(APRIME_R7_BOARD_SHA):
        path = APRIME_R7_BASE / ('board%d-s6' % board) / 's6' / 'receipt.json'
        R6F.require(file_sha(path) == pin, 'APRIME_R7_BOARD_RECEIPT')
        x = read(path)
        R6F.require(x['status'] == 'PASS' and x['slot'] == 'r7' and x['board'] == board
                    and all(row['status'] == 'genuine' for row in x['records']), 'APRIME_R7_BOARD_PASS')
        for row in x['records']:
            R6F.require(row['id'] not in probabilities, 'APRIME_R7_UNIQUE')
            probabilities[row['id']] = row['probabilities']
    ids = pack['cfg']['probe_order'] + pack['cfg']['binding']['visible_ids']
    R6F.require(sorted(probabilities) == sorted(ids) and len(ids) == 39, 'APRIME_R7_FROZEN39')
    return {'probabilities': {s: probabilities[s] for s in ids}, 'receipts_sha256': list(APRIME_R7_BOARD_SHA),
            'base_receipt_sha256': APRIME_R7_BASE_SHA, 'source': 'ozgung A-prime v2 rehearsal (ws2-r224-v2-qual-20261015) base R7 arm'}


def binding(pack, sources, variant=None):
    ab = pack['cfg']['binding']
    manifest, receipt = r6_manifest()
    r224, r224_receipt = aprime_r224_state()
    R6F.require(file_sha(ARCH / 'kernel-metadata.json') == ARCH_META_SHA, 'EXACT_APRIME_METADATA')
    meta = read(ARCH / 'kernel-metadata.json')
    b = {
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
    if variant is not None:
        blend = R6F.VARIANTS[variant]
        arms, weights, default = R6F.resolve_blend(blend)
        frame = variant_frame(variant)
        names = [a for a, _ in arms]
        compact, flip = R6F.resolve_compact(blend)
        compact_text = ('heliosli R6 fe055406 flip TTA' if compact == R6F.DEFAULT_COMPACT else
                        'compact %s (%s, flip TTA %s)' % ('+'.join(a for a, _ in compact), ', '.join('%s %g' % x for x in compact), flip))
        b.update(variant='R6flip-%s: %s Raptor half (%s), %s, per-finding weights' % (
                     variant.upper(), '+'.join(names), ', '.join('%s %g' % x for x in arms), compact_text),
                 frame=frame, frame_digest=R6F.digest(frame), raptor_weight_vector=R6F.raptor_weight_vector(blend=blend),
                 entry_variant={'name': variant, 'blend': blend, 'stages': names + [a for a, _ in compact],
                                'crop224_stage': 'crop224' in names, 'compact_arms': [list(x) for x in compact], 'compact_flip': flip,
                                'nafull_stage': 'nafull' in names, 'weights_digest': R6F.digest(blend),
                                'default_unchanged': 'no --variant: no cfg[blend], no cfg[nafull], original slugs and sources shipped'})
        b['deviations'] = DEVIATIONS + variant_deviations(variant)
        if own_dual(variant):
            block = s6c_block(variant)[0]
            b['variant'] = 'R6flip-%s: %s Raptor half (%s), compact 1/2 R6 plain + 1/2 own S6 %d-member (labels %s) mean rank, per-finding weights' % (
                variant.upper(), '+'.join(names), ', '.join('%s %g' % x for x in arms), len(block['models']), block['labels'])
            b['s6c'] = {'version': S6C.VERSION, 'block_digest': R6F.digest(block), 'labels': block['labels'],
                        'models': [{k: m[k] for k in ('seed', 'file', 'sha256', 'bytes', 'dataset', 'dataset_version')} for m in block['models']],
                        'aliases': list(block['aliases']), 'aggregation': S6C.AGGREGATION,
                        'worker': {'module': S6C.WORKER_MODULE, 'env': WORKER.PROBE_DUAL_ENV},
                        'source_sha256': {n: hashlib.sha256(sources[n].encode()).hexdigest() for n in (*S6C_SOURCES, S6C.CAPTURED)}}
            b['tolerances'].update(s6c_cross_process_board_dprob=S6C.CROSS_TOLERANCE, s6c_a100_reference_dprob=S6C.REFERENCE_ATOL)
        if variant in R6F.COMPACT_VARIANTS:
            block = s6c_block(variant)[0]
            b.pop('r6')
            b['variant'] = 'R6flip-%s: %s Raptor half (%s), own S6 %d-seed ensemble (labels %s) mean rank, per-finding weights' % (
                variant.upper(), '+'.join(names), ', '.join('%s %g' % x for x in arms), len(block['models']), block['labels'])
            b['s6c'] = {'version': S6C.VERSION, 'block_digest': R6F.digest(block), 'labels': block['labels'],
                        'models': [{k: m[k] for k in ('seed', 'file', 'sha256', 'bytes', 'dataset', 'dataset_version')} for m in block['models']],
                        'aggregation': S6C.AGGREGATION, 'worker': {'module': S6C.WORKER_MODULE, 'env': WORKER.PROBE_DUAL_ENV},
                        'source_sha256': {n: hashlib.sha256(sources[n].encode()).hexdigest() for n in (*S6C_SOURCES, S6C.CAPTURED)}}
            b['frame']['compact_half'] = {'s6c': S6C.VERSION, 'seeds': [m['seed'] for m in block['models']], 'labels': block['labels'],
                                          'flip_tta': False, 'aggregation': S6C.AGGREGATION}
            b['frame']['excluded'] = sorted(set(b['frame']['excluded']) | {'heliosli R6 bundle'})
            b['frame_digest'] = R6F.digest(b['frame'])
        if 'nafull' in names:
            b['nafull_source_sha256'] = {n: hashlib.sha256(sources[n].encode()).hexdigest() for n in NAFULL_SOURCES}
        if has_r7(variant):
            b['r7'] = r7_block(pack)
            b['r7_source_sha256'] = {n: hashlib.sha256(sources[n].encode()).hexdigest() for n in R7_SOURCES}
            b['tolerances']['r7_aprime_reference_dprob'] = ENTRY.R7_REFERENCE_TOLERANCE
    return b


def s6c_models(variant='p7'):
    models = globals()[S6C_MODELS_BY_VARIANT[variant]]
    R6F.require(models is not None, 'S6C_MODELS_NOT_SEALED')
    return copy.deepcopy(models)


def s6c_block(variant='p7'):
    """(cfg['s6c'], shipped sources) for the compact stage of P7-prime or the own arm of P9 (the member set is sealed per variant;
    the A100 reference is restricted to the sealed members' aliases)."""
    reference = None
    models = s6c_models(variant)
    if S6C_REFERENCE is not None:
        R6F.require(file_sha(S6C_REFERENCE['path']) == S6C_REFERENCE['sha256'], 'S6C_REFERENCE_PIN')
        reference = read(S6C_REFERENCE['path'])
        aliases = {S6C.model_alias(m) for m in models}
        R6F.require(aliases <= set(reference['probabilities']), 'S6C_REFERENCE_COVERS_SEALED_MEMBERS')
        reference = {'source': reference['source'], 'probabilities': {k: v for k, v in reference['probabilities'].items() if k in aliases}}
    block = S6C.block(models, reference)
    raw = S6C_CAPTURED.read_bytes()
    R6F.require(hashlib.sha256(raw).hexdigest() == S6C.CAPTURED_SHA256, 'S6C_CAPTURED_SOURCE')
    sources = {S6C.CAPTURED: raw.decode()}
    for name in S6C_SOURCES:
        text = (PROJECT / name).read_text()
        compile(text, name, 'exec')
        sources[name] = text
    R6F.require(hashlib.sha256(sources[S6C.CAPTURED].encode()).hexdigest() == S6C.CAPTURED_SHA256, 'S6C_CAPTURED_TEXT_ROUNDTRIP')
    return block, sources


def variant_deviations(variant):
    names = R6F.blend_arm_names(R6F.VARIANTS[variant])
    out = ['Variant %s: Raptor arms and per-finding weights are sealed cfg[blend]; complete populations use A-prime fill/rank rules '
           'generalized to the arm set (recover_arms), otherwise the public fallback ranks.' % variant]
    if 'nafull' in names:
        out.append('na-full Raptor K62 one-view arm: the A-triple-prime worker (public_nafull_aprime_*, B0prime old_worker) run through '
                   'the same two-board deadline stage after crop224 and before r6; gates reuse its own K62 acceptance checks. '
                   'Missing/failed na-full only costs its rank in the entry. Dataset ozgung/knee-raptor64-na-full-fold0-final-20261016 v1.')
    if 'crop224' not in names:
        out.append('No R224 stage: crop224 is not run and its gates are skipped.')
    if has_r7(variant):
        out.append('Dual compact: R6 runs without --flip-tta; a last stage runs heliosli R7 v1 (8c7b666e27dc0e8f, A-prime pins) through '
                   'the same observed compact_infer worker, plain pass; compact slot = 1/2 rank(R6) + 1/2 rank(R7) (pre-registered). '
                   'Guard-cut or failed R7 studies keep rank(R6) as their compact slot. Rehearsal: R7 input must equal R6 input per '
                   'study and R7 probabilities must lie within 5e-3 of A-prime\'s qualified R7 run on the frozen 39.')
    if R6F.compact_shared(R6F.VARIANTS[variant]):
        out.append('Shared decode (P6s): no separate R7 stage. The R6 plain production worker loads the pinned R7 folds (CPU state '
                   'digests), compiles them like compact_infer --compile (warm-up 32..192 windows, first-study compiled-vs-eager '
                   'check 1e-3), and scores them right after R6\'s last fold on the same rendered windows; R7 stackers recomputed '
                   'with knee.public_r7_stackers. R7 stops before R6 near the deadline. Rehearsal: R6 and R7 within 1e-3 of the '
                   'separate-pass P6 rehearsal on the frozen 39.')
    if own_dual(variant):
        out.append('Dual compact with our own members (P9): R6 runs without --flip-tta; the last stage runs our all-data S6 members '
                   '(knee.public_s6c worker, exact nartaa S6 code tree, fp16, no flip TTA, one label table) under the guard; compact '
                   'slot = 1/2 rank(R6 plain) + 1/2 rank(mean over members of per-member average percentile ranks) (pre-registered). '
                   'Guard-cut or failed own-stage studies keep rank(R6 plain) as their compact slot. Rehearsal: two fresh-process '
                   'sentinel passes of the own worker, cross-board/cross-process parity 1e-3 per member, A100 fp16 reference 0.01.')
    if R6F.own_shared(R6F.VARIANTS[variant]):
        out.append('Shared decode (P9s): no separate own stage. The R6 plain worker\'s loader children render the members\' 320 px windows '
                   'from the same native cache (px_cache.StudySlices at 320, identical code in both trees; a study R6\'s side check flipped '
                   'is re-rendered without the forced side, as the s6c worker sees it) and the main process scores every member (S6\'s own '
                   'compact_train.Net, fp16, chunk 192) right after R6\'s last fold. Members stop before R6 near the deadline. Rehearsal: '
                   'members within 1e-3 of the P9 rehearsal\'s separate s6c worker on the frozen 39; R6 sentinel passes repeat the members.')
    if variant in R6F.COMPACT_VARIANTS:
        out.append('Compact slot = our all-data S6 seed ensemble (knee.public_s6c): mean over seeds of per-seed average percentile '
                   'ranks, no flip TTA, one label table; the heliosli R6 bundle is neither attached nor run. One compact_infer per '
                   'board through the captured nartaa solo launcher/B command (exact S6 code tree from nartaa/rsna-knee-s6fall-weights '
                   'v3 + heliosli cloud_code), fp16, two spawn loader workers, eval stride 1, --dump-logits for per-seed logits. '
                   'Rehearsal sentinel passes run the same worker without the R6 source-seam injection.')
    return out


def packet(mode='rehearsal', qualified=None, variant=None):
    R6F.require(mode in ('rehearsal', 'entry'), 'MODE')
    R6F.require(variant is None or variant in R6F.VARIANTS, 'KNOWN_VARIANT')
    pack = archived()
    sources = dict(pack['sources'])
    for name in NEW_SOURCES:
        text = (PROJECT / name).read_text()
        compile(text, name, 'exec')
        sources[name] = text
    cfg = copy.deepcopy(pack['cfg'])
    if has_s6c(variant):
        cfg['s6c'], extra = s6c_block(variant)
        R6F.require(not set(extra) & set(sources), 'S6C_SOURCES_DISJOINT')
        sources.update(extra)
    if variant is not None:
        cfg['blend'] = copy.deepcopy(R6F.VARIANTS[variant])
        if 'nafull' in R6F.blend_arm_names(cfg['blend']):
            cfg['nafull'], extra = nafull_block()
            R6F.require(not set(extra) & set(sources), 'NAFULL_SOURCES_DISJOINT')
            sources.update(extra)
        if has_r7(variant):
            for name in R7_SOURCES:
                R6F.require(name not in sources, 'R7_SOURCES_DISJOINT')
                text = (PROJECT / name).read_text()
                compile(text, name, 'exec')
                sources[name] = text
            if mode == 'rehearsal':
                cfg['r7_reference'] = r7_reference(pack)
                if R6F.compact_shared(cfg['blend']):
                    cfg['p6_reference'] = p6_reference()
        if R6F.own_shared(cfg['blend']) and mode == 'rehearsal':
            cfg['p9_reference'] = p9_reference()
    b = binding(pack, sources, variant)
    for key in R7_ONLY_CFG:
        cfg.pop(key)
    cfg['mode'] = mode
    cfg['r6flip_binding'] = b
    if mode == 'entry':
        R6F.require(qualified is not None, 'ENTRY_REQUIRES_QUALIFIED_REHEARSAL')
        for key in PRIVATE_CFG:
            cfg.pop(key)
        cfg.pop('r7_reference', None)
        cfg.pop('p6_reference', None)
        cfg.pop('p9_reference', None)
        cfg['binding'] = {k: v for k, v in cfg['binding'].items() if k not in PRIVATE_BINDING}
        if 'nafull' in cfg:
            cfg['nafull'] = {k: v for k, v in cfg['nafull'].items() if k != 'reader_reference'}
            cfg['nafull']['worker'] = {k: v for k, v in cfg['nafull']['worker'].items() if k not in ENTRY.NAFULL_WORKER_PRIVATE}
        if variant is None or 'crop224' in R6F.blend_arm_names(cfg['blend']):
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


NAFULL_MOUNT_SITE = "    save(project/'runtime-config.json',cfg)"
NAFULL_MOUNT = """    naweights=mount({dataset!r},[{file!r}])
    nafile=naweights/{file!r}
    require(nafile.stat().st_size=={size} and file_sha(nafile)=={sha!r},'NAFULL_VERSION1_CHECKPOINT_BYTES')
    cfg['nafull']['worker']['control_checkpoint']=str(nafile)
"""


R7_MOUNT = """    r7=mount(b6['r7']['dataset'],['manifest.json',*b6['r7']['asset_sha256']])
    require(file_sha(r7/'manifest.json')==b6['r7']['manifest_sha256'],'EXACT_R7_V1_MANIFEST')
    cfg['r7_mount']=str(r7)
"""
S6C_MOUNT_SITE = """    r6=mount(b6['r6']['dataset'],['manifest.json',*b6['r6']['asset_sha256']])
    require(file_sha(r6/'manifest.json')==b6['r6']['manifest_sha256'],'EXACT_R6_V1_MANIFEST')
    anatomy=mount('heliosli/rsna-orthofoundation-fold0-epoch7-20261004',['anatomy.pt'])/'anatomy.pt'
    require(file_sha(anatomy)==b6['r6']['manifest']['anchor_weights_sha256'],'EXACT_ANATOMY')
"""
S6C_MOUNT = """    s6c=cfg['s6c'];r6=None
    cm=mount(s6c['code']['dataset'],list(s6c['code']['files']))
    for name,pin in s6c['code']['files'].items():require(file_sha(cm/name)==pin['sha256'],'EXACT_S6C_CODE_MOUNT')
    helios=mount('heliosli/rsna-orthofoundation-fold0-epoch7-20261004',['anatomy.pt']);anatomy=helios/'anatomy.pt'
    require(file_sha(anatomy)==s6c['anatomy_sha256'],'EXACT_ANATOMY')
    for m in s6c['models']:
        parts=m['dataset'].split('/')
        md=Path('/kaggle/input/models')/parts[0]/parts[1]/parts[2].lower()/parts[3]/str(m['dataset_version']) if len(parts)==4 else mount(m['dataset'],[m['file']])
        m['path']=str(md/m['file'])
        require((md/m['file']).stat().st_size==m['bytes'] and file_sha(md/m['file'])==m['sha256'],'EXACT_S6C_SEED_BYTES')
    s6c.update(code_mount=str(cm),helios_mount=str(helios))
"""
S6C_R6_MOUNT_ARG = "cfg.update(code=str(code),r6_mount=str(r6),"
S6C_DUAL_MOUNT = """    s6c=cfg['s6c']
    cm=mount(s6c['code']['dataset'],list(s6c['code']['files']))
    for name,pin in s6c['code']['files'].items():require(file_sha(cm/name)==pin['sha256'],'EXACT_S6C_CODE_MOUNT')
    require(file_sha(anatomy)==s6c['anatomy_sha256'],'EXACT_ANATOMY')
    for m in s6c['models']:
        parts=m['dataset'].split('/')
        md=Path('/kaggle/input/models')/parts[0]/parts[1]/parts[2].lower()/parts[3]/str(m['dataset_version']) if len(parts)==4 else mount(m['dataset'],[m['file']])
        m['path']=str(md/m['file'])
        require((md/m['file']).stat().st_size==m['bytes'] and file_sha(md/m['file'])==m['sha256'],'EXACT_S6C_SEED_BYTES')
    s6c.update(code_mount=str(cm),helios_mount=str(anatomy.parent))
"""


def bootstrap(pack):
    tail = BOOTSTRAP_TAIL
    if 'r7' in pack['cfg']['r6flip_binding']:      # P6 dual compact only; every other tail is untouched
        R6F.require(tail.count(NAFULL_MOUNT_SITE) == 1, 'EXACT_R7_MOUNT_SITE')
        tail = tail.replace(NAFULL_MOUNT_SITE, R7_MOUNT + NAFULL_MOUNT_SITE)
    if 's6c' in pack['cfg'] and 'r6' in pack['cfg']['r6flip_binding']:    # P9: the R6 mount stays; our members and the S6 code mount are added
        R6F.require(tail.count(NAFULL_MOUNT_SITE) == 1 and tail.count(S6C_MOUNT_SITE) == 1, 'EXACT_S6C_DUAL_MOUNT_SITE')
        tail = tail.replace(NAFULL_MOUNT_SITE, S6C_DUAL_MOUNT + NAFULL_MOUNT_SITE)
    elif 's6c' in pack['cfg']:         # P7-prime only: no R6 mount; our seed datasets and the S6 code mount instead
        R6F.require(tail.count(S6C_MOUNT_SITE) == 1 and tail.count(S6C_R6_MOUNT_ARG) == 1, 'EXACT_S6C_MOUNT_SITE')
        tail = tail.replace(S6C_MOUNT_SITE, S6C_MOUNT).replace(S6C_R6_MOUNT_ARG, "cfg.update(code=str(code),r6_mount=None,")
    if 'nafull' in pack['cfg']:        # variants with the na-full arm only; the default tail is untouched
        ck = pack['cfg']['nafull']['checkpoint']
        R6F.require(tail.count(NAFULL_MOUNT_SITE) == 1, 'EXACT_NAFULL_MOUNT_SITE')
        tail = tail.replace(NAFULL_MOUNT_SITE, NAFULL_MOUNT.format(dataset=ck['dataset'], file=ck['file'], size=ck['bytes'], sha=ck['sha256'])
                            + NAFULL_MOUNT_SITE)
    encoded = base64.b64encode(lzma.compress(R6F.canonical(pack), preset=9)).decode()
    script = BOOTSTRAP_HEAD + 'encoded=%r\nexpected_payload=%r\n' % (encoded, R6F.digest(pack)) + tail
    compile(script, '<r6flip-bootstrap>', 'exec')
    return script


def notebook(pack, mode):
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python3', 'language': 'python'}},
          'cells': [{'id': 'r6flip-' + mode, 'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                     'source': bootstrap(pack).splitlines(True)}]}
    raw = json.dumps(nb, indent=2) + '\n'
    R6F.require(len(raw.encode()) < 900000, 'NOTEBOOK_900K_CAP')
    return raw


def metadata(slug, variant=None):
    R6F.require(variant is None or variant in R6F.VARIANTS, 'KNOWN_VARIANT')
    R6F.require(isinstance(slug, str) and 3 <= len(slug) <= 50 and all(c in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in slug), 'SAFE_SLUG')
    R6F.require(file_sha(ARCH / 'kernel-metadata.json') == ARCH_META_SHA, 'EXACT_APRIME_METADATA')
    arch = read(ARCH / 'kernel-metadata.json')
    R6F.require(arch['machine_shape'] == 'NvidiaTeslaT4' and arch['enable_internet'] is False and arch['is_private'] is True, 'APRIME_METADATA')
    return {'id': '%s/%s' % (OWNER, slug), 'title': slug, 'code_file': 'notebook.ipynb', 'language': 'python', 'kernel_type': 'notebook',
            'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False, 'competition_sources': [COMP],
            'dataset_sources': variant_datasets(variant),
            'kernel_sources': [], 'model_sources': S6C.model_sources(S6C.block(s6c_models(variant))) if has_s6c(variant) else [],
            'docker_image': arch['docker_image'], 'machine_shape': 'NvidiaTeslaT4'}


def variant_datasets(variant):
    if variant in R6F.COMPACT_VARIANTS:     # no heliosli R6 bundle attached
        return [d for d in DATASETS if not d.startswith(R6F.R6_DATASET + '/')] + S6C.dataset_sources(S6C.block(s6c_models(variant)))
    if own_dual(variant):                   # the R6 bundle and our members
        return list(DATASETS) + S6C.dataset_sources(S6C.block(s6c_models(variant)))
    return list(DATASETS) + (nafull_dataset_sources() if variant is not None
                             and 'nafull' in R6F.blend_arm_names(R6F.VARIANTS[variant]) else []) \
        + (['heliosli/rsna-knee-compact-r7cnxfsss/1'] if has_r7(variant) else [])


def nafull_dataset_sources():
    from knee import public_nafull_aprime as NA
    return ['%s/%d' % (NA.CHECKPOINT['dataset'], NA.CHECKPOINT['version'])]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def emit(dest, slug, pack, mode, extra=None, variant=None):
    dest = Path(dest)
    R6F.require(not dest.exists() and dest.resolve().is_relative_to(OUT.resolve()), 'FRESH_IGNORED_OUTPUT')
    R6F.require(variant is None or pack['cfg']['blend'] == R6F.VARIANTS[variant], 'PACKET_IS_THE_VARIANT')
    raw = notebook(pack, mode)
    dest.mkdir(parents=True)
    (dest / 'notebook.ipynb').write_text(raw)
    save_json(dest / 'kernel-metadata.json', metadata(slug, variant))
    tracked = [*NEW_SOURCES, *(NAFULL_SOURCES if variant is not None else ()), *(R7_SOURCES if has_r7(variant) else ()),
               *(S6C_SOURCES if has_s6c(variant) else ()),
               'scripts/build_r6flip_entry.py', 'scripts/qualify_r6flip_cpu.py']
    changed = subprocess.check_output(['git', 'status', '--porcelain', '--', *tracked], cwd=PROJECT, text=True).strip()
    receipt = {'status': 'PREPARATION_ONLY' if mode == 'rehearsal' else 'QUALIFIED_ENTRY_BUILD', 'mode': mode, 'slug': slug,
               'binding_sha256': R6F.digest(pack['cfg']['r6flip_binding']), 'payload_sha256': R6F.digest(pack),
               'new_source_sha256': pack['cfg']['r6flip_binding']['new_source_sha256'],
               'source_bundle_sha256': pack['cfg']['r6flip_binding']['source_bundle_sha256'], 'source_files': len(pack['sources']),
               'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip(),
               'source_state': ('uncommitted implementation' if changed else 'committed implementation') + '; exact file hashes bound',
               'builder_sha256': file_sha(__file__), 'files': {n: file_sha(dest / n) for n in ('notebook.ipynb', 'kernel-metadata.json')},
               'notebook_bytes': len(raw.encode()), 'runtime_checks_exercised': False,
               'deviations': pack['cfg']['r6flip_binding']['deviations'], **({'variant': variant} if variant is not None else {}),
               **(extra or {})}
    save_json(dest / 'build-receipt.json', receipt)
    save_json(dest / 'MANIFEST.json', {'mode': mode, 'payload_sha256': R6F.digest(pack), 'files': {
        n: {'sha256': file_sha(dest / n), 'bytes': (dest / n).stat().st_size} for n in ('notebook.ipynb', 'kernel-metadata.json', 'build-receipt.json')}})
    return receipt


def verify_build(root, mode='rehearsal', qualified=None):
    root = Path(root)
    m, r, meta = read(root / 'MANIFEST.json'), read(root / 'build-receipt.json'), read(root / 'kernel-metadata.json')
    variant = r.get('variant')          # absent for the default build; replayed below, so a tampered value fails the replay
    R6F.require(set(m['files']) == {'notebook.ipynb', 'kernel-metadata.json', 'build-receipt.json'}, 'MANIFEST_COVERAGE')
    for n, pin in m['files'].items():
        R6F.require(pin == {'sha256': file_sha(root / n), 'bytes': (root / n).stat().st_size}, 'IMMUTABLE_BUILD_' + n)
    expected = packet(mode, qualified, variant)
    actual, code = unpack_notebook(root / 'notebook.ipynb')
    R6F.require(actual == expected and code == bootstrap(expected), 'EXACT_PACKET_REPLAY')
    R6F.require(meta == metadata(meta['title'], variant) and meta['id'] == '%s/%s' % (OWNER, meta['title']) and 'timeout' not in meta
                and (variant is None or meta['title'] == slug_for(variant, mode)), 'PRIVATE_OFFLINE_T4_TITLE_ID')
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
    crop = {w['state_before'] for w in r['workers'].get('crop224', [])}      # none in a no-R224 variant (p5)
    return {'status': r['status'], 'receipt_sha256': file_sha(root / 'receipt.json'),
            'csv_sha256': file_sha(root / 'visible-predictions.csv'), 'r224_state': crop.pop() if crop else None,
            **({'nafull_state': r['workers']['nafull'][0]['state_before']['state']} if 'nafull' in r['workers'] else {}),
            'r6_state': r['workers']['r6'][0]['state_before'], 'anatomy_state': q['r6']['anatomy_state'],
            **({'s6c_state': r['workers']['s6c'][0]['state_before'], 's6c_reference_max_abs': q['s6c']['reference_max_abs_dprob']}
               if 's6c' in r['workers'] else {}),
            **({'r7_state': r['workers']['r7'][0]['state_before'], 'r7_reference_max_abs': q['r7']['aprime_reference_max_abs']}
               if 'r7' in r['workers'] else {}),
            'projection': {k: r['projection'][k] for k in ('point_seconds', 'conservative_seconds', 'gate_seconds', 'gate_pass')},
            'elapsed_seconds': r['elapsed_seconds']}


def build_rehearsal(dest=None, variant=None):
    slug = slug_for(variant, 'rehearsal')
    return emit(Path(dest or OUT / 'kaggle' / slug), slug, packet('rehearsal', variant=variant), 'rehearsal', variant=variant)


def same_stages(a, b):
    """A rehearsal of variant a also qualifies variant b when only per-finding weights differ (same stages, same arm weights)."""
    return (a is not None and b is not None and R6F.VARIANTS[a]['raptor_arms'] == R6F.VARIANTS[b]['raptor_arms']
            and R6F.resolve_compact(R6F.VARIANTS[a]) == R6F.resolve_compact(R6F.VARIANTS[b])
            and R6F.compact_shared(R6F.VARIANTS[a]) == R6F.compact_shared(R6F.VARIANTS[b])       # a shared-decode build has its own gates
            and R6F.own_shared(R6F.VARIANTS[a]) == R6F.own_shared(R6F.VARIANTS[b]))


def build_entry(rehearsal_build, rehearsal_result, waiver=None, dest=None, slug=None, variant=None):
    prepared, rehearsal_pack = verify_build(rehearsal_build)
    rehearsed = prepared.get('variant')
    variant = rehearsed if variant is None else variant
    R6F.require(variant == rehearsed or same_stages(rehearsed, variant), 'REHEARSAL_QUALIFIES_THIS_VARIANT')
    checked = verify_result(rehearsal_result, rehearsal_pack['cfg'])
    if checked['status'] != 'PASS':
        R6F.require(waiver is not None, 'PROJECTION_GATE_FAILED_NAMED_WAIVER_REQUIRED')
        w = read(waiver)
        R6F.require(w.get('scope') == 'projection_only' and w.get('rehearsal_receipt_sha256') == checked['receipt_sha256']
                    and isinstance(w.get('authorized_by'), str) and w['authorized_by'].strip() and w.get('conservative_seconds')
                    == checked['projection']['conservative_seconds'], 'NAMED_PROJECTION_WAIVER')
        checked['waiver'] = {'file_sha256': file_sha(waiver), **w}
    qualified = {**checked, 'rehearsal_build': {'slug': prepared['slug'], 'payload_sha256': prepared['payload_sha256'],
                                                'files': prepared['files']},
                 **({'rehearsal_variant': rehearsed} if rehearsed is not None else {})}
    slug = slug or slug_for(variant, 'entry')
    return emit(Path(dest or OUT / 'kaggle' / slug), slug, packet('entry', qualified, variant), 'entry',
                {'qualified_rehearsal': qualified}, variant=variant)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='command')
    variants = sorted(R6F.VARIANTS)
    r = sub.add_parser('rehearsal')
    r.add_argument('--destination')
    r.add_argument('--variant', choices=variants, help='no flag: the existing R6F-pure build (variant P2)')
    e = sub.add_parser('entry')
    e.add_argument('--rehearsal-build', required=True)
    e.add_argument('--rehearsal-result', required=True)
    e.add_argument('--waiver')
    e.add_argument('--destination')
    e.add_argument('--slug')
    e.add_argument('--variant', choices=variants, help='default: the rehearsed build\'s variant; a rehearsal of p3 also qualifies p4')
    v = sub.add_parser('verify')
    v.add_argument('--build')
    v.add_argument('--variant', choices=variants, help='selects the default --build path; the variant itself is replayed from the build')
    a = p.parse_args(argv)
    if a.command in (None, 'rehearsal'):
        out = build_rehearsal(getattr(a, 'destination', None), getattr(a, 'variant', None))
    elif a.command == 'entry':
        out = build_entry(a.rehearsal_build, a.rehearsal_result, a.waiver, a.destination, a.slug, a.variant)
    else:
        out, _ = verify_build(a.build or str(OUT / 'kaggle' / slug_for(a.variant, 'rehearsal')))
    print(json.dumps({k: out[k] for k in ('status', 'slug', 'payload_sha256', 'files', 'notebook_bytes')}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
