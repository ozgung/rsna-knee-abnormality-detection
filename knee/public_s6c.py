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
LABEL_TABLES = ('v2', 'v3')
SEED_OWNERS = ('ozgung', 'nartaa')   # our exports; nartaa's S6fall control only for smoke runs. Never a heliosli bundle.
RECONSTRUCTION_ATOL = 2e-6       # CSV '%.6f' of sigmoid(mean seed logits) vs the dumped float32 seed logits
CROSS_TOLERANCE = 1e-3           # fresh process / other board, per seed probability
REFERENCE_ATOL = 0.01            # A100 fp16 reference vs T4 fp16; measured s42 A100 vs s6solo T4: max 0.0047 on 36 probes
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


def validate_models(models):
    """The sealed seed list: one label table, distinct seeds, exact file identities; returns the loader-order aliases."""
    require(isinstance(models, list) and 2 <= len(models) <= 6, 'S6C_TWO_TO_SIX_SEEDS')
    keys = {'seed', 'labels', 'file', 'sha256', 'bytes', 'dataset', 'dataset_version'}
    for m in models:
        require(isinstance(m, dict) and set(m) - {'path', 'slot'} == keys, 'S6C_MODEL_SCHEMA')     # 'path' is added by the notebook bootstrap
        require(type(m.get('slot', 0)) is int and 0 <= m.get('slot', 0) <= 9, 'S6C_MODEL_SLOT')
        require(type(m['seed']) is int and 0 <= m['seed'] < 10000 and m['labels'] in LABEL_TABLES, 'S6C_MODEL_SEED_LABELS')
        require(isinstance(m['sha256'], str) and len(m['sha256']) == 64 and type(m['bytes']) is int and m['bytes'] > 0, 'S6C_MODEL_IDENTITY')
        require(isinstance(m['dataset'], str) and m['dataset'].split('/')[0] in SEED_OWNERS and len(m['dataset'].split('/')) in (2, 4)
                and type(m['dataset_version']) is int
                and m['dataset_version'] > 0 and isinstance(m['file'], str) and m['file'].endswith('.pt') and '/' not in m['file'],
                'S6C_MODEL_DATASET')
    require(len({m['labels'] for m in models}) == 1, 'S6C_ONE_LABEL_TABLE_NO_MIXING')
    require(len({model_alias(m) for m in models}) == len(models) and len({m['sha256'] for m in models}) == len(models), 'S6C_DISTINCT_SEEDS')
    names = [model_alias(m) for m in models]
    require(names == sorted(names), 'S6C_SEEDS_IN_LOADER_ORDER')      # compact_infer globs sorted(<arm>_s*_f*.pt)
    return names


def block(models, reference=None):
    """cfg['s6c'] (paths are added by the notebook bootstrap)."""
    names = validate_models(models)
    b = {'version': VERSION, 'labels': models[0]['labels'], 'models': [dict(m) for m in models], 'aliases': names,
         'code': {'dataset': CODE_DATASET, 'version': CODE_VERSION, 'files': CODE_FILES}, 'helios': HELIOS,
         'anatomy_sha256': ANATOMY_SHA256, 'captured_sha256': CAPTURED_SHA256, 'source_pins': SOURCE_PINS,
         'geometry': GEOMETRY, 'amp': AMP, 'loader_workers': LOADER_WORKERS, 'aggregation': AGGREGATION, 'flip_tta': False}
    if reference is not None:
        require(set(reference) == {'source', 'probabilities'} and isinstance(reference['probabilities'], dict), 'S6C_REFERENCE_SCHEMA')
        for name, rows in reference['probabilities'].items():
            require(name in names and isinstance(rows, dict) and rows, 'S6C_REFERENCE_SEED')
            for p in rows.values():
                require(len(p) == 12 and all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in p), 'S6C_REFERENCE_VALUES')
        b['reference'] = reference
    return b


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


def merge(ids, reports, out, n_seeds, pass_name='production'):
    """Strict complete-population merge of the two board ledgers -> (mean-rank compact values [N, 12] with NaN for failed
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
    return mean_rank(seed_p, ~mask), mask, seed_p


def salvage(ids, out, n_seeds, boards=(0, 1)):
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
    return mean_rank(seed_p, ok), np.where(ok, 'genuine', 'unavailable').tolist()


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


def require_compact(r, cfg, passes=('sentinel1', 'sentinel2', 'production'), key='r6'):
    """Rehearsal gates of the compact stage: both boards, every pass PASS, all 39 genuine, sentinel repeats across fresh
    processes and boards within CROSS_TOLERANCE per seed, optional A100 reference within REFERENCE_ATOL.
    key: the controller's worker key ('r6' in P7-prime, where the block replaces the R6 bundle; 's6c' in P9, next to R6)."""
    from knee.public_s6_recovery_entry import assignments
    sentinels = cfg['s6_sentinels']
    anatomy, observed = set(), {}
    for pass_name in passes:
        reports = r['workers'][key] if pass_name == 'production' else r['workers'][key + '_sentinels'][pass_name]
        ids_by_board = assignments(cfg['staged_ids']) if pass_name == 'production' else [sentinels, sentinels]
        require(len(reports) == 2, 'S6C_TWO_BOARDS_' + pass_name)
        for board, x in enumerate(reports):
            anatomy.add(require_worker(x, r, cfg, board, pass_name, ids_by_board[board]))
            require(x['failed'] == 0, 'S6C_NO_FAILURE_' + pass_name)
            for row in x['records']:
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
                require(len(row) == 1, 'S6C_REFERENCE_STUDY_SCORED')
                d = max(abs(u - v) for u, v in zip(row[0]['seed_probabilities'][k], p))
                require(d <= REFERENCE_ATOL, 'S6C_A100_REFERENCE')
                ref_max, ref_n = max(ref_max or 0.0, d), ref_n + 1
    return {'sentinel_repeats': {sid: len(observed[sid]) for sid in sentinels}, 'sentinel_max_abs_dprob': worst,
            'reference_studies': ref_n, 'reference_max_abs_dprob': ref_max, 'anatomy_state': ANATOMY_SHA256,
            'seeds': len(cfg['s6c']['models']), 'labels': cfg['s6c']['labels']}


def finite_unit(p):
    return all(isinstance(x, float) and math.isfinite(x) and 0.0 <= x <= 1.0 for x in p)
