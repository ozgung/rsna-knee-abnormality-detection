"""P7-prime compact slot: our own all-data S6 seed ensemble in the R6F controller's compact stage (no heliosli bundle).

Dependency-free at import; array helpers import NumPy/pandas inside calls.

Frame (Lane C, 2026-10-19, fixed before any P7-prime run):
* Raptor half unchanged from P2: 0.65 rank(R384 two-view) + 0.35 rank(R224 one-view), v1 per-finding weights (frozen
  R6F arithmetic, knee.public_r6flip.recover).
* Compact half = the mean over seeds of each seed's average percentile rank over the scored population. Every seed is an
  all-data S6 fit of ours (V2-all recipe: convnext_small, 320/288, v4, 3 slices, 3.5 mm) on ONE label table (v2 and v3
  seeds are never mixed). No flip TTA. The values enter the frozen recovery in the compact ('r6') slot, where they are
  quantized at '%.9g' and ranked again (a monotone map of the mean rank).
* One compact_infer pass per board scores every seed on the same windows (one decode): exact nartaa S6 code tree
  (nartaa/rsna-knee-s6fall-weights v3 code extras + heliosli cloud_code, compact_infer 34f433fe), launched through the
  captured solo controller's own launcher and B command (fp16, two spawn loader workers, eval stride 1). Per-seed logits
  come from compact_infer's --dump-logits; the CSV is the source's mean-logit probability.
"""
import hashlib
import json
import math
from pathlib import Path

VERSION = 's6c-seed-ensemble-meanrank-v1'
VARIANT = 'p7'
WORKER_MODULE = 'knee.public_s6c_worker'
FOLDER = 's6c'
CAPTURED = 'knee/s6c_publisher/rsna-knee-s6fall-solo.py'
CAPTURED_SHA256 = '6e87350d5ae092b7a889546ca13517303523bc469f7e611ffe945913434d22fe'
CODE_DATASET, CODE_VERSION = 'nartaa/rsna-knee-s6fall-weights', 3
CODE_FILES = {'code_manifest.json': {'sha256': 'b66c574b5c0c13ccaa49d1219d41a8781431166186ef83e78cab6a9ba6b02d22', 'bytes': 52962},
              's6_extras.bin': {'sha256': 'ddc428d9f2cda0e1378b6731ca7efbdd0397978204763f2a2f81169979e50c8e', 'bytes': 256011}}
HELIOS = 'heliosli/rsna-orthofoundation-fold0-epoch7-20261004'
ANATOMY_SHA256 = '5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e'
SOURCE_PINS = {'compact_train.py': '14767c7a8f0daa986477511fe9a83aefe032a208ae978fabf15a5f896992ac80',
               'compact_infer.py': '34f433fe72b732543813165c3397ec80cef01b0d49b25b7b22668714f0b0ef4a'}
GEOMETRY = {'backbone': 'convnext_small', 'size': 320, 'crop': 288, 'views': 'v4', 'slices': 3, 'phys_mm': 3.5,
            'eval_stride': 1, 'chunk': 192, 'item_timeout': 900}
AMP, LOADER_WORKERS = 'fp16', 2
GEOMETRY_CHILD = {'size': GEOMETRY['size'], 'crop': GEOMETRY['crop'], 'views': GEOMETRY['views'], 'slices': GEOMETRY['slices'],
                  'eval_stride': GEOMETRY['eval_stride'], 'phys_mm': GEOMETRY['phys_mm']}      # the shared-decode render recipe
LABEL_TABLES = ('v2', 'v3')                       # P7-prime's tables; later tables match LABEL_PATTERN (v4a, v4b, round1, ...)
LABEL_PATTERN = r'^(v\d[a-z]?|round\d)$'
MAX_MEMBERS = 8                                   # P10/P11 (parent, 2026-10-20 22:55): up to eight own members
AGGREGATIONS = {'mean_rank': 'mean over members of per-member average percentile ranks over the genuine rows of the scored population',
                'mean_logit': 'sigmoid of the mean over members of the per-member logits (the fold-mean convention of every S6 run)'}
SEED_OWNERS = ('ozgung', 'nartaa')   # our exports; nartaa's S6fall control only for smoke runs. Never a heliosli bundle.
RECONSTRUCTION_ATOL = 2e-6       # CSV '%.6f' of sigmoid(mean seed logits) vs the dumped float32 seed logits
CROSS_TOLERANCE = 1e-3           # fresh process / other board, per seed probability
REFERENCE_ATOL = 0.02            # A100 fp16 reference vs T4 fp16, max over 36 probe studies x 12 findings. 0.01 was set from one
                                 # measurement (s42: 0.0047); the P9 v2 / P10 / P11 rehearsals (2026-10-21) measured 0.0087 (v3-s44),
                                 # 0.0105 (r1-f1) and 0.0140 (r1-f3) with means <= 0.0005: fp16 noise across GPUs, while a wrong
                                 # checkpoint or geometry shows >= 0.1. Widened to 0.02 by the dated amendment in the lane note.
LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
          'Synovitis', "Baker's", 'Contusion', 'Fracture']
AGGREGATION = 'mean over seeds of per-seed average percentile ranks over the genuine rows of the scored population'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def alias(seed, slot=0):
    return 'E_prior_s%d_f%d.pt' % (seed, slot)


def model_alias(m):
    """Loader alias; 'slot' separates two members that share a seed number (Lane D's student is also seed 42)."""
    return alias(m['seed'], m.get('slot', 0))


def validate_models(models, allow_mixed_labels=False):
    """The sealed member list: one label table (several only when the variant seals the mix), distinct aliases and files, exact
    file identities; 'no_reference': True marks a member without an A100 probe-36 read-out (explicit, visible in the sealed list).
    Returns the loader-order aliases."""
    import re
    require(isinstance(models, list) and 2 <= len(models) <= MAX_MEMBERS, 'S6C_TWO_TO_EIGHT_MEMBERS')
    keys = {'seed', 'labels', 'file', 'sha256', 'bytes', 'dataset', 'dataset_version'}
    for m in models:
        require(isinstance(m, dict) and set(m) - {'path', 'slot', 'no_reference'} == keys, 'S6C_MODEL_SCHEMA')   # 'path': notebook bootstrap
        require(type(m.get('slot', 0)) is int and 0 <= m.get('slot', 0) <= 9, 'S6C_MODEL_SLOT')
        require(m.get('no_reference', False) in (False, True), 'S6C_MODEL_NO_REFERENCE_FLAG')
        require(type(m['seed']) is int and 0 <= m['seed'] < 10000 and isinstance(m['labels'], str)
                and (m['labels'] in LABEL_TABLES or re.match(LABEL_PATTERN, m['labels'])), 'S6C_MODEL_SEED_LABELS')
        require(isinstance(m['sha256'], str) and len(m['sha256']) == 64 and type(m['bytes']) is int and m['bytes'] > 0, 'S6C_MODEL_IDENTITY')
        require(isinstance(m['dataset'], str) and m['dataset'].split('/')[0] in SEED_OWNERS and len(m['dataset'].split('/')) in (2, 4)
                and type(m['dataset_version']) is int
                and m['dataset_version'] > 0 and isinstance(m['file'], str) and m['file'].endswith('.pt') and '/' not in m['file'],
                'S6C_MODEL_DATASET')
    require(allow_mixed_labels or len({m['labels'] for m in models}) == 1, 'S6C_ONE_LABEL_TABLE_NO_MIXING')
    require(len({model_alias(m) for m in models}) == len(models) and len({m['sha256'] for m in models}) == len(models), 'S6C_DISTINCT_SEEDS')
    names = [model_alias(m) for m in models]
    require(names == sorted(names), 'S6C_SEEDS_IN_LOADER_ORDER')      # compact_infer globs sorted(<arm>_s*_f*.pt)
    return names


def block(models, reference=None, aggregation='mean_rank', allow_mixed_labels=False, eval_stride=1):
    """cfg['s6c'] (paths are added by the notebook bootstrap). aggregation: AGGREGATIONS key (P7-prime: mean_rank). eval_stride: the
    separate worker's window stride (1 = the captured B command; S4 may seal 2 or 3, the rehearsal then also reads the reference
    studies at stride 1 for the A100 reference gate)."""
    names = validate_models(models, allow_mixed_labels)
    require(aggregation in AGGREGATIONS, 'S6C_AGGREGATION')
    require(eval_stride in (1, 2, 3), 'S6C_EVAL_STRIDE')
    tables = sorted({m['labels'] for m in models})
    b = {'version': VERSION, 'labels': tables[0] if len(tables) == 1 else '+'.join(tables), 'models': [dict(m) for m in models], 'aliases': names,
         'code': {'dataset': CODE_DATASET, 'version': CODE_VERSION, 'files': CODE_FILES}, 'helios': HELIOS,
         'anatomy_sha256': ANATOMY_SHA256, 'captured_sha256': CAPTURED_SHA256, 'source_pins': SOURCE_PINS,
         'geometry': GEOMETRY, 'amp': AMP, 'loader_workers': LOADER_WORKERS, 'aggregation': AGGREGATIONS[aggregation],
         'aggregation_key': aggregation, 'mixed_labels_allowed': bool(allow_mixed_labels), 'flip_tta': False}
    if eval_stride != 1:                      # absent key = stride 1: every earlier block (P7, P9, P10, P11) is byte-identical
        b['eval_stride'] = int(eval_stride)
    if reference is not None:
        require(set(reference) == {'source', 'probabilities'} and isinstance(reference['probabilities'], dict), 'S6C_REFERENCE_SCHEMA')
        for name, rows in reference['probabilities'].items():
            require(name in names and isinstance(rows, dict) and rows, 'S6C_REFERENCE_SEED')
            for p in rows.values():
                require(len(p) == 12 and all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in p), 'S6C_REFERENCE_VALUES')
        b['reference'] = reference
    return b


def aggregation_key(s6c):
    return (s6c or {}).get('aggregation_key', 'mean_rank')


def is_model_ref(ref):
    """owner/model/framework/variation = a private Kaggle Model variation (mounted under /kaggle/input/models/...);
    owner/slug = a dataset. Lane C seeds are Models because the private dataset quota is exhausted."""
    return len(ref.split('/')) == 4


def model_mount(ref, version):
    owner, model, framework, variation = ref.split('/')
    return 'models/%s/%s/%s/%s/%d' % (owner, model, framework.lower(), variation, version)


def dataset_sources(s6c):
    return list(dict.fromkeys(['%s/%d' % (CODE_DATASET, CODE_VERSION)] + ['%s/%d' % (m['dataset'], m['dataset_version'])
                                                                          for m in s6c['models'] if not is_model_ref(m['dataset'])]))


def model_sources(s6c):
    return list(dict.fromkeys('%s/%d' % (m['dataset'], m['dataset_version']) for m in s6c['models'] if is_model_ref(m['dataset'])))


def sigmoid(z):
    import numpy as np
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def mean_logit(seed_probabilities, good):
    """[n, S, 12] per-member probabilities, [n] genuine mask -> [n, 12]: sigmoid of the mean member logit; NaN elsewhere."""
    import numpy as np
    p = np.asarray(seed_probabilities, dtype=np.float64)
    good = np.asarray(good, dtype=bool)
    require(p.ndim == 3 and p.shape[2] == 12 and p.shape[1] >= 1 and good.shape == (p.shape[0],), 'S6C_MEAN_LOGIT_SCHEMA')
    out = np.full((p.shape[0], 12), np.nan)
    if good.any():
        g = np.clip(p[good], 1e-12, 1 - 1e-12)
        require(np.isfinite(p[good]).all() and ((p[good] >= 0) & (p[good] <= 1)).all(), 'S6C_GENUINE_SEED_PROBABILITIES')
        out[good] = sigmoid(np.log(g / (1 - g)).mean(axis=1))
    return out


def aggregate(seed_probabilities, good, aggregation='mean_rank'):
    require(aggregation in AGGREGATIONS, 'S6C_AGGREGATION')
    return mean_rank(seed_probabilities, good) if aggregation == 'mean_rank' else mean_logit(seed_probabilities, good)


def mean_rank(seed_probabilities, good):
    """[n, S, 12] per-seed probabilities, [n] genuine mask -> [n, 12]: mean over seeds of each seed's average pct rank
    over the genuine rows; NaN elsewhere."""
    import numpy as np
    import pandas as pd
    p = np.asarray(seed_probabilities, dtype=np.float64)
    good = np.asarray(good, dtype=bool)
    require(p.ndim == 3 and p.shape[2] == 12 and p.shape[1] >= 1 and good.shape == (p.shape[0],), 'S6C_MEAN_RANK_SCHEMA')
    out = np.full((p.shape[0], 12), np.nan)
    if good.any():
        g = p[good]
        require(np.isfinite(g).all() and ((g >= 0) & (g <= 1)).all(), 'S6C_GENUINE_SEED_PROBABILITIES')
        ranks = [pd.DataFrame(g[:, s, :]).rank(method='average', pct=True).to_numpy() for s in range(p.shape[1])]
        out[good] = np.mean(ranks, axis=0)
    return out


def ledger_path(out, board, pass_name='production'):
    return Path(out) / ('board%d-s6c-%s' % (board, pass_name)) / FOLDER / 'predictions.npz'


def _ledger_rows(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as a:
        return a['study_ids'].tolist(), a['statuses'].tolist(), a['probabilities'], a['seed_logits']


def merge(ids, reports, out, n_seeds, pass_name='production', aggregation='mean_rank'):
    """Strict complete-population merge of the two board ledgers -> (aggregated compact values [N, 12] with NaN for failed
    rows, data-local failure mask [N], per-seed probabilities [N, S, 12])."""
    import numpy as np
    from knee.public_s6_recovery_entry import assignments
    from knee.public_s6_qualification import file_sha
    expected = assignments(ids)
    require(len(reports) == 2, 'S6C_TWO_SHARDS')
    by_id, failed = {}, {}
    for board, r in enumerate(reports):
        path = ledger_path(out, board, pass_name)
        require(r.get('version') == VERSION and r.get('status') == 'PASS' and r.get('board') == board and r.get('arm') == FOLDER
                and r.get('pass') == pass_name and r.get('assigned') == expected[board] and r.get('completed') == len(expected[board]),
                'S6C_WORKER_TERMINAL_GATE')
        require(r.get('state_before') == r.get('state_after') and r.get('ledger_sha256') == file_sha(path), 'S6C_STATE_LEDGER_BINDING')
        got, statuses, p, z = _ledger_rows(path)
        require(got == expected[board] and p.shape == (len(got), 12) and z.shape == (len(got), n_seeds, 12), 'S6C_SHARD_ORDER_COVERAGE')
        good = np.asarray(statuses) == 'genuine'
        require(set(statuses) <= {'genuine', 'data_local_failure'} and np.isfinite(p[good]).all() and np.isnan(p[~good]).all()
                and np.isfinite(z[good]).all() and np.isnan(z[~good]).all(), 'S6C_GENUINE_ONLY_LEDGER')
        rows = r['records']
        require([x['id'] for x in rows] == got and [x['status'] for x in rows] == statuses
                and r['genuine'] == int(good.sum()) and r['failed'] == int((~good).sum()), 'S6C_LEDGER_RECORD_BINDING')
        for sid, prob, zz, st in zip(got, p, z, statuses):
            require(sid not in by_id, 'S6C_DUPLICATE_MERGE_ID')
            by_id[sid], failed[sid] = (prob, zz), st != 'genuine'
    require(set(by_id) == set(ids), 'S6C_COMPLETE_POPULATION')
    seed_p = np.array([sigmoid(by_id[s][1]) for s in ids])
    mask = np.array([failed[s] for s in ids], dtype=bool)
    if aggregation == 'mean_logit':           # exactly the worker CSV's arithmetic: sigmoid of the mean member logit
        values = np.full((len(ids), 12), np.nan)
        z = np.array([by_id[s][1] for s in ids], dtype=np.float64)
        values[~mask] = sigmoid(z[~mask].mean(axis=1))
        return values, mask, seed_p
    return aggregate(seed_p, ~mask, aggregation), mask, seed_p


def salvage(ids, out, n_seeds, boards=(0, 1), aggregation='mean_rank'):
    """Entry fallback: every genuine row on disk. Ledger rows give per-seed probabilities; rows only in compact_infer's
    flushed CSV (worker interrupted before its ledger) carry the source mean-logit probability for every seed. Returns
    (mean-rank values [N, 12], statuses [N] of 'genuine'/'unavailable')."""
    import csv
    import numpy as np
    index = {s: i for i, s in enumerate(ids)}
    seed_p = np.full((len(ids), n_seeds, 12), np.nan)
    ok = np.zeros(len(ids), dtype=bool)
    for board in boards:
        rows = []
        try:
            got, statuses, _, z = _ledger_rows(ledger_path(out, board))
            rows = [(sid, sigmoid(zz)) for sid, zz, st in zip(got, z, statuses) if st == 'genuine']
        except Exception:
            try:
                with open(ledger_path(out, board).parent / 'compact' / 'predictions.csv', newline='') as h:
                    for row in list(csv.reader(h))[1:]:
                        if len(row) == 13:
                            rows.append((row[0], np.repeat(np.array([[float(x) for x in row[1:]]]), n_seeds, axis=0)))
            except Exception:
                rows = []
        for sid, sp in rows:
            sp = np.asarray(sp, dtype=np.float64)
            if sid in index and sp.shape == (n_seeds, 12) and np.isfinite(sp).all() and ((sp >= 0) & (sp <= 1)).all():
                seed_p[index[sid]], ok[index[sid]] = sp, True
    return aggregate(seed_p, ok, aggregation), np.where(ok, 'genuine', 'unavailable').tolist()


def _close(a, b, tol):
    return len(a) == len(b) and all(len(x) == len(y) == 12 for x, y in zip(a, b)) and max(
        abs(float(u) - float(v)) for x, y in zip(a, b) for u, v in zip(x, y)) <= tol


def require_worker(x, r, cfg, board, pass_name, ids):
    s6c, ab = cfg['s6c'], cfg['binding']
    require(x.get('version') == VERSION and x.get('status') == 'PASS' and x.get('board') == board and x.get('pass') == pass_name
            and x.get('arm') == FOLDER and x.get('assigned') == list(ids) and x.get('completed') == len(ids), 'S6C_WORKER_PASS')
    env = x['environment']
    require(env['torch'] == ab['environment']['torch'] and env['packages'] == ab['environment']['packages']
            and env['physical_uuid'] == r['boards'][board]['uuid'] and env['cuda_device_count'] == 1, 'S6C_ISOLATED_PINNED_BOARD')
    want = [{'alias': model_alias(m), 'sha256': m['sha256'], 'bytes': m['bytes']} for m in s6c['models']]
    require(x['models'] == want and x['state_before'] == x['state_after'] == digest(want), 'S6C_SEALED_SEED_FILES_UNCHANGED')
    ci = x['compact']
    require(ci['status'] == 'PASS' and ci['amp'] == AMP and ci['models'] == s6c['aliases'] and ci['spatial_models'] == len(want)
            and ci['extent_models'] == 0 and ci['timed_out'] is False and ci['anchor_weights_sha256'] == ANATOMY_SHA256
            and all(ci['sha256'][n] == SOURCE_PINS[n] for n in SOURCE_PINS), 'S6C_SOURCE_RECEIPT')
    require(x['code']['captured_sha256'] == CAPTURED_SHA256 and x['code']['extras_ok'] is True and x['code']['source_sha_ok'] is True,
            'S6C_EXACT_CODE_TREE')
    require(x['peak_cuda_bytes'] > 0 and x['peak_rss_bytes'] > 0, 'S6C_RESOURCES')
    for row in x['records']:
        if row['status'] == 'genuine':
            require(row['reconstruction_max_abs'] <= RECONSTRUCTION_ATOL and isinstance(row['windows'], int) and row['windows'] > 0
                    and len(row['seed_probabilities']) == len(want), 'S6C_STUDY_GATES')
    return x['anatomy_sha256']


def require_compact(r, cfg, passes=('sentinel1', 'sentinel2', 'production'), key='r6', injected=None, reference_key='seed_probabilities'):
    """Rehearsal gates of the compact stage: both boards, every pass PASS, all 39 genuine, sentinel repeats across fresh
    processes and boards within CROSS_TOLERANCE per seed, optional A100 reference within REFERENCE_ATOL.
    key: the controller's worker key ('r6' in P7-prime, where the block replaces the R6 bundle; 's6c' in P9, next to R6).
    injected: (pass name, study id) of R6's rehearsal source-seam injection (P9s shared decode: that study has no R6 forward, so
    the members have no output for it in that pass; it must be the only failure there).
    reference_key: the row field compared with the A100 probe-36 reference ('seed_probabilities_stride1' when the members run at
    eval stride 2: the reference was read at stride 1)."""
    from knee.public_s6_recovery_entry import assignments
    sentinels = cfg['s6_sentinels']
    anatomy, observed = set(), {}
    for pass_name in passes:
        reports = r['workers'][key] if pass_name == 'production' else r['workers'][key + '_sentinels'][pass_name]
        ids_by_board = assignments(cfg['staged_ids']) if pass_name == 'production' else [sentinels, sentinels]
        require(len(reports) == 2, 'S6C_TWO_BOARDS_' + pass_name)
        for board, x in enumerate(reports):
            anatomy.add(require_worker(x, r, cfg, board, pass_name, ids_by_board[board]))
            skip = injected[1] if injected is not None and injected[0] == pass_name else None
            require(x['failed'] == (1 if skip is not None else 0), 'S6C_NO_FAILURE_' + pass_name)
            for row in x['records']:
                if row['id'] == skip:
                    require(row['status'] == 'data_local_failure', 'S6C_INJECTED_ROW_NOT_GENUINE')
                    continue
                require(row['status'] == 'genuine', 'S6C_HEALTHY_MUST_BE_GENUINE')
                observed.setdefault(row['id'], []).append((pass_name, board, row))
    require(anatomy == {ANATOMY_SHA256}, 'S6C_ANATOMY')
    require(set(observed) == set(cfg['staged_ids']), 'S6C_COMPLETE39')
    worst = 0.0
    for sid in sentinels:
        rows = observed[sid]
        require(len(rows) >= 3, 'S6C_SENTINEL_REPEATS')
        for _, _, a in rows:
            for _, _, b in rows:
                require(_close(a['seed_probabilities'], b['seed_probabilities'], CROSS_TOLERANCE), 'S6C_SENTINEL_FRESH_PROCESS_CROSS_BOARD')
                worst = max(worst, max(abs(u - v) for x, y in zip(a['seed_probabilities'], b['seed_probabilities']) for u, v in zip(x, y)))
    ref = cfg['s6c'].get('reference')
    ref_max, ref_n = None, 0
    if ref:
        names = cfg['s6c']['aliases']
        for name, rows in ref['probabilities'].items():
            k = names.index(name)
            for sid, p in rows.items():
                row = [row for pn, _, row in observed.get(sid, []) if pn == 'production']
                require(len(row) == 1 and row[0].get(reference_key) is not None, 'S6C_REFERENCE_STUDY_SCORED')
                d = max(abs(u - v) for u, v in zip(row[0][reference_key][k], p))
                require(d <= REFERENCE_ATOL, 'S6C_A100_REFERENCE')
                ref_max, ref_n = max(ref_max or 0.0, d), ref_n + 1
    return {'sentinel_repeats': {sid: len(observed[sid]) for sid in sentinels}, 'sentinel_max_abs_dprob': worst,
            'reference_studies': ref_n, 'reference_max_abs_dprob': ref_max, 'reference_key': reference_key,
            'members_without_reference': [model_alias(m) for m in cfg['s6c']['models'] if m.get('no_reference')],
            'anatomy_state': ANATOMY_SHA256, 'seeds': len(cfg['s6c']['models']), 'labels': cfg['s6c']['labels'],
            'aggregation': aggregation_key(cfg['s6c'])}


def require_shared(r, cfg, stride=1):
    """P9s gates on the shared-decode members: every report is the shared slot at the sealed eval stride, no cut, every genuine row
    rendered from the shared decode with the R6 worker's own hand-off PASS, and the reference members' probabilities within
    SHARED_PARITY of the separate s6c worker (cfg['p9_reference'], the P9 rehearsal at stride 1) on the frozen 39 -- at stride 2
    from the rehearsal's stride-1 check forwards ('seed_probabilities_stride1'). Also the per-member cost at each measured stride."""
    from knee.public_s6_recovery_entry import assignments
    shards = assignments(cfg['staged_ids'])
    reports, r6_reports = r['workers']['s6c'], r['workers']['r6']
    key = 'seed_probabilities' if stride == 1 else 'seed_probabilities_stride1'
    require(len(reports) == 2 and len(r6_reports) == 2, 'S6C_SHARED_TWO_BOARDS')
    for board, (x, x6) in enumerate(zip(reports, r6_reports)):
        require(x.get('slot') == 'compact-shared' and x.get('cut') is None and x.get('fp32_recoveries') == 0 and x.get('flip') == {'enabled': False}
                and x.get('assigned') == shards[board] and x6.get('s6c_shared', {}).get('status') == 'PASS'
                and (x.get('geometry') or {}).get('eval_stride', 1) == stride, 'S6C_SHARED_WORKER_PASS')
        require(all(row['shared_decode'] is True and row['checksum_equal'] is True and not row['r6_aborted'] and row['fp32_rescues'] == 0
                    and (stride == 1 or row.get(key) is not None) for row in x['records'] if row['status'] == 'genuine'), 'S6C_SHARED_STUDY_GATES')
    rows = {row['id']: row for w in reports for row in w['records']}
    ref = cfg['p9_reference']
    aliases = cfg['s6c']['aliases']
    ref_aliases = [a for a in ref['aliases'] if a in aliases]     # p9s2s2b (four v5 members): no member with a separate-pass read-out;
    require(set(ref['probabilities']) == set(rows) == set(cfg['staged_ids']), 'P9_REFERENCE_COVERAGE')   # parity then not applicable
    require(ref_aliases or not set(ref['aliases']) & set(aliases), 'P9_REFERENCE_COVERAGE')
    worst, per_member = 0.0, {a: 0.0 for a in ref_aliases}
    for sid, row in rows.items():
        for j, a in enumerate(ref_aliases):
            k = aliases.index(a)
            d = max(abs(float(u) - float(v)) for u, v in zip(row[key][k], ref['probabilities'][sid][list(ref['aliases']).index(a)]))
            per_member[a] = max(per_member[a], d)
            worst = max(worst, d)
    require(worst <= SHARED_PARITY, 'P9S_SEPARATE_PASS_PARITY')
    separate = sum(1 for row in rows.values() if (row.get('render') or {}).get('separate_render'))
    n = len(rows)
    cost = {'stride': stride, 'members': len(aliases),
            'per_member_study_seconds': [sum(row['member_seconds'][k] for row in rows.values()) / n for k in range(len(aliases))],
            'windows_per_study': sum(row['windows'] for row in rows.values()) / n}
    if stride != 1:
        cost['stride1_check_per_member_study_seconds'] = [sum(row['stride1_seconds'][k] for row in rows.values()) / n for k in range(len(aliases))]
        cost['stride1_windows_per_study'] = sum(row['windows_stride1'] for row in rows.values()) / n
    return {'studies': n, 'separate_pass_max_abs': worst if ref_aliases else None, 'separate_pass_max_abs_by_member': per_member,
            'parity_applicable': bool(ref_aliases), 'parity_key': key,
            'members_kept_every_study': all(row['status'] == 'genuine' for row in rows.values()) and all(x.get('cut') is None for x in reports),
            'cut_rules': sorted({str(x.get('cut_rule', 'r6_only')) for x in reports}),
            'separate_pass_tolerance': SHARED_PARITY, 'side_flipped_separate_renders': separate,
            'reference_receipt_sha256': ref['receipt_sha256'], 'member_cost': cost}


SHARED_PARITY = 1e-3             # per member, shared decode vs the separate s6c worker (the fresh-process tolerance)


def finite_unit(p):
    return all(isinstance(x, float) and math.isfinite(x) and 0.0 <= x <= 1.0 for x in p)
