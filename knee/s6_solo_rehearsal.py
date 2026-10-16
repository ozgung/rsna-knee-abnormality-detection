"""Rehearsal-only layer of the S6 solo pair: frozen-39 fixture, ledgers, sentinels and projection.

Hooks (installed only when cfg['mode'] == 'rehearsal'):
- the controller's local-test root K1_KAGGLE_ROOT points at a fresh fixture
  whose competition folder lists the frozen 39 studies (36 train + 3 test) as
  "test", with per-study symlinks into the real image folders; the attached
  datasets are symlinked unchanged and ``working`` is /kaggle/working;
- component A runs through knee/s6_solo_a_runner.py (exact bytes, fixture root);
- run_b records per-item pixel/meta hashes (the captured K1_PX_DUMP path,
  predictions unchanged); prepare_inputs is wrapped only to keep ``I``.

After the unchanged controller finishes, its 39-row file is moved away from
submission.csv, two studies per board are re-scored on the other T4, and
``assemble`` derives the ledger, gates and runtime projection purely from the
saved files, so ``verify_result`` can replay them independently.
"""
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

from knee import s6_solo_runtime as RT
from knee.s6_solo_runtime import binding_digest, file_sha, peaks, require, save

COMP = 'rsna-knee-abnormality-detection'
FIXTURE_ROOT = Path('/tmp/s6solo39')
INPUT_ROOT = Path('/kaggle/input')
UID = 'StudyInstanceUID'


def resolve_mount(input_root, slug, marker):
    """The captured find_mount candidate rule, on the real input root."""
    input_root = Path(input_root)
    cands = [input_root / slug, input_root / 'competitions' / slug]
    cands += list(input_root.glob('datasets/*/' + slug)) + list(input_root.glob('datasets/*/' + slug + '/*'))
    cands += list(input_root.glob('*/' + slug)) + list(input_root.glob(slug + '/*'))
    found = []
    for p in cands:
        if (p / marker).is_file() and p.resolve() not in [f.resolve() for f in found]:
            found.append(p)
    require(len(found) == 1, 'UNIQUE_MOUNT_' + slug)
    return found[0]


def _rows(path):
    with open(path, newline='') as f:
        r = csv.reader(f)
        header = next(r)
        return header, list(r)


def _write(path, header, rows):
    with open(path, 'x', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)


def build_fixture(cfg, input_root, root, work):
    """Fresh fixture root; raw images are never copied, only linked per study."""
    fx = cfg['rehearsal']['fixture']
    ids, splits = fx['ids'], fx['splits']
    require(len(ids) == len(splits) == len(set(ids)) and set(splits) <= {'train', 'test'}, 'FROZEN_FIXTURE_SHAPE')
    input_root, root = Path(input_root), Path(root)
    comps = [p for p in (input_root / 'competitions' / COMP, input_root / COMP) if (p / 'test.csv').is_file()]
    require(len(comps) == 1, 'UNIQUE_ACTUAL_COMPETITION_ROOT')
    real = comps[0]
    require(not root.exists() and not root.is_symlink(), 'FRESH_FIXTURE_ROOT')
    comp = root / 'input' / COMP
    comp.mkdir(parents=True)
    test_header, _ = _rows(real / 'test.csv')
    require(test_header == [UID], 'TEST_CSV_SCHEMA')
    sample_header, _ = _rows(real / 'sample_submission.csv')
    require(sample_header[0] == UID and sorted(sample_header[1:]) == sorted(cfg['common']['targets']), 'SAMPLE_SCHEMA')
    series_header = None
    by_split = {}
    for split in sorted(set(splits)):
        header, rows = _rows(real / (split + '_series.csv'))
        series_header = series_header or header
        require(header == series_header and header[0] == UID, 'SERIES_CSV_SCHEMA')
        by_split[split] = {}
        for row in rows:
            by_split[split].setdefault(row[0], []).append(row)
    series = []
    for sid, split in zip(ids, splits):
        rows = by_split[split].get(sid)
        require(rows, 'FIXTURE_STUDY_SERIES_PRESENT')
        series += rows
    _write(comp / 'test.csv', [UID], [[sid] for sid in ids])
    _write(comp / 'test_series.csv', series_header, series)
    _write(comp / 'sample_submission.csv', sample_header, [[sid] + ['0.5'] * (len(sample_header) - 1) for sid in ids])
    (comp / 'test_series').mkdir()
    for sid, split in zip(ids, splits):
        src = real / (split + '_series') / sid
        require(src.is_dir(), 'FIXTURE_STUDY_IMAGES_PRESENT')
        (comp / 'test_series' / sid).symlink_to(src, target_is_directory=True)
    mounts = [tuple(m) for m in cfg['rehearsal']['mounts']]
    b = cfg['arm']['b_model']
    if b['mount'] not in [m[0] for m in mounts]:
        mounts.append((b['mount'], b['source_file']))
    for slug, marker in mounts:
        (root / 'input' / slug).symlink_to(resolve_mount(input_root, slug, marker), target_is_directory=True)
    (root / 'working').symlink_to(Path(work), target_is_directory=True)
    return {'root': str(root), 'comp': str(comp), 'real_competition': str(real), 'studies': len(ids),
            'mounts': sorted(m[0] for m in mounts)}


def prepare(cfg, project):
    require(cfg.get('mode') == 'rehearsal' and 'K1_KAGGLE_ROOT' not in os.environ, 'REHEARSAL_ONLY_FIXTURE')
    t = time.time()
    fixture = build_fixture(cfg, INPUT_ROOT, FIXTURE_ROOT, RT.WORK)
    os.environ['K1_KAGGLE_ROOT'] = fixture['root']
    return {'fixture': fixture, 'fixture_seconds': round(time.time() - t, 3), 'fixture_started': t,
            'runner': str(Path(project) / 'knee' / 's6_solo_a_runner.py')}


def install(k1, cfg, state):
    require(cfg.get('mode') == 'rehearsal' and os.environ.get('K1_KAGGLE_ROOT') == state['fixture']['root']
            and str(k1.KROOT) == state['fixture']['root'], 'REHEARSAL_HOOKS_ON_FIXTURE_ONLY')
    runner, comp = state['runner'], state['fixture']['comp']
    k1.a_cmd = lambda adir: [sys.executable, '-u', runner, str(Path(adir) / 'run.py'), comp]
    run_b = k1.run_b

    def run_b_with_pixel_hashes(I, tag, ids, split, budget_s, px=False):
        return run_b(I, tag, ids, split, budget_s, px=True)
    k1.run_b = run_b_with_pixel_hashes
    prepare_inputs = k1.prepare_inputs

    def keep_inputs():
        I = prepare_inputs()
        state['I'] = I
        return I
    k1.prepare_inputs = keep_inputs
    state['controller_t0'] = k1.T0


# ---- after the unchanged controller -------------------------------------------------------------------------------------------
def run_sentinels(k1, cfg, I):
    """Each board re-scores the first studies the other board scored (same launcher/arguments)."""
    n = cfg['rehearsal']['sentinel_per_board']
    out = Path(k1.OUT)
    parts = [json.loads((out / 'B' / ('part%d' % i) / 'ids.json').read_text()) for i in range(2)]
    root = out / 'sentinel'
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    t0 = time.time()
    deadline = t0 + cfg['rehearsal']['sentinel_timeout_s'] - 120
    specs = []
    for gi, ids in ((1, parts[0][:n]), (0, parts[1][:n])):
        pdir = root / ('board%d' % gi)
        pdir.mkdir()
        (pdir / 'px').mkdir()
        (pdir / 'ids.json').write_text(json.dumps(ids))
        specs.append({'name': 'sentinel.board%d' % gi, 'cmd': k1.b_cmd(I, 'test', pdir / 'ids.json', pdir, k1.TMP / ('sentinel_tmp%d' % gi), deadline),
                      'env': k1.b_env(gi, pdir / 'px'), 'cwd': str(k1.TMP), 'log': pdir / 'log.txt',
                      'done_when': (lambda d=pdir: k1.b_done(d))})
    runs = k1.run_group(specs, cfg['rehearsal']['sentinel_timeout_s'], 'sentinel')
    for gi, ids in ((1, parts[0][:n]), (0, parts[1][:n])):
        k1.collect_part(root / ('board%d' % gi), ids, I)
    return {'seconds': round(time.time() - t0, 1), 'runs': [list(r) for r in runs]}


def finish(k1, cfg, state, sampler, started, out):
    controller_end = time.time()
    final = Path(out) / 'rehearsal-final.csv'
    os.replace(RT.WORK / 'submission.csv', final)       # never leave the 39-row file where Kaggle would score it
    sentinel_run = None
    try:
        sentinel_run = run_sentinels(k1, cfg, state['I'])
    except Exception as e:                            # recorded; the sentinel gate then fails the rehearsal
        sentinel_run = {'error': repr(e)[:2000]}
    sampler.stop()
    timing = {'notebook_started': started, 'fixture_started': state['fixture_started'], 'fixture_seconds': state['fixture_seconds'],
              'controller_t0': state['controller_t0'], 'controller_end': controller_end, 'finished': time.time()}
    resources = sampler.report()
    save(Path(out) / 'resources.json', resources)
    receipt, ledger = assemble(k1, cfg, Path(k1.OUT), final, timing, sentinel_run, resources)
    save(Path(out) / 'ledger.json', ledger)
    receipt['ledger_sha256'] = file_sha(Path(out) / 'ledger.json')
    receipt['resources_sha256'] = file_sha(Path(out) / 'resources.json')
    receipt['final_csv_sha256'] = file_sha(final)
    save(Path(out) / 'receipt.json', receipt)
    return receipt


def a_timing(log_text):
    loaded, gpu_done, done = [], None, None
    for line in log_text.splitlines():
        s = line.strip()
        if s.startswith('[gpu ') and '] loaded ' in s and s.endswith('s'):
            loaded.append(float(s.rsplit('|', 1)[1].strip()[:-1]))
        elif s.startswith('[gpu] done: ') and s.endswith('s'):
            gpu_done = float(s.rsplit('|', 1)[1].strip()[:-1])
        elif s.startswith('DONE ') and s.endswith('s'):
            done = float(s[5:-1])
    require(loaded and gpu_done is not None and done is not None and gpu_done > max(loaded), 'A_LOG_TIMING_LINES')
    return {'loaded_s': max(loaded), 'gpu_done_s': gpu_done, 'done_s': done, 'devices_loaded': len(loaded)}


def b_timing(k1, pdir, n):
    prog = k1.progress_lines(Path(pdir) / 'log.txt')
    first = [p for p in prog if p.get('done') == 1]
    last = [p for p in prog if p.get('done') == n]
    require(first and last and n >= 2, 'B_PROGRESS_LINES')
    rec = json.loads((Path(pdir) / 'receipt.json').read_text())
    return {'n': n, 'first_elapsed': first[0]['elapsed_s'], 'last_elapsed': last[-1]['elapsed_s'], 'seconds': rec['seconds'],
            'windows': rec['stats']['windows']}


def projection(m, k1):
    """Ordinary 1,300-study projection from measured fixed and per-study costs.

    Fixed = every measured second outside per-study production, once. Per-study
    A cost is its whole post-load span over N (pipeline fill included); per-study
    B cost is the slower board's span after its first study. The fixture build and
    sentinels are rehearsal-only and excluded; the rehearsal-only pixel-hash dump
    stays inside measured B production (conservative).
    """
    st, n, target, g = m['stages'], m['studies'], m['target_studies'], m['n_gpu']
    setup = m['pre_controller_seconds'] + m['controller_overhead_seconds'] + st['env'] + st['inputs'] + st['B_probe']
    a_var = m['a']['gpu_done_s'] - m['a']['loaded_s']
    a_fixed = st['A'] - a_var
    spans = [p['last_elapsed'] - p['first_elapsed'] for p in m['b_parts']]
    b_rate = max(s / (p['n'] - 1) for s, p in zip(spans, m['b_parts']))
    b_fixed = st['B'] - max(spans)
    require(a_var > 0 and a_fixed >= 0 and b_rate > 0 and b_fixed >= 0 and g == 2, 'MEASURED_PROJECTION_INPUTS')
    per_board = math.ceil(target / g)
    scenarios = []
    for mult in (1.0, 1.25, 1.5):
        for imb in (1.0, 1.15, 1.3):
            a = a_fixed + a_var / n * target * mult * imb
            b = b_fixed + b_rate * (per_board - 1) * mult * imb
            controller = m['controller_overhead_seconds'] + st['env'] + st['inputs'] + st['B_probe'] + a + b + st['blend']
            scenarios.append({'window_multiplier': mult, 'shard_imbalance': imb, 'A_seconds': a, 'B_seconds': b,
                              'controller_seconds': controller,
                              'seconds': setup + a + b + st['blend'] + m['post_controller_allowance_seconds']})
    c = next(s for s in scenarios if s['window_multiplier'] == 1.25 and s['shard_imbalance'] == 1.15)
    gate = m['gate_seconds']
    inherited = {'A_cap_seconds': k1.A_TIMEOUT_S, 'guard_seconds': k1.GUARD_MAX_S, 'end_slack_seconds': k1.END_SLACK_S,
                 'A_within_cap': c['A_seconds'] < k1.A_TIMEOUT_S,
                 'controller_within_guard': c['controller_seconds'] < k1.GUARD_MAX_S - k1.END_SLACK_S}
    return {'population': target, 'measured_studies': n, 'scenarios': scenarios, 'conservative': c,
            'conservative_seconds': c['seconds'], 'gate_seconds': gate,
            'fixed_seconds': {'setup': setup, 'A': a_fixed, 'B': b_fixed, 'blend': st['blend']},
            'production_seconds': {'A_measured': a_var, 'A_per_study': a_var / n, 'B_slowest_span': max(spans),
                                   'B_per_study_per_board': b_rate},
            'inherited_guards': inherited,
            'passes': c['seconds'] <= gate and inherited['A_within_cap'] and inherited['controller_within_guard'],
            'policy': 'Fixed costs once; 25% more work per study and 15% board imbalance; inherited A cap and 8h guard '
                      'checked separately. Sensitivities, not hidden-population bounds; hidden tails remain unknown.'}


def _csv_matrix(path, labels):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return [r[UID] for r in rows], [[float(r[l]) for l in labels] for r in rows], (list(rows[0].keys()) if rows else [])


def assemble(k1, cfg, k1_dir, final_csv, timing, sentinel_run, resources):
    """Ledger, gates and projection from saved files only (replayed by verify_result)."""
    import numpy as np
    LAB = cfg['common']['targets']
    fx = cfg['rehearsal']['fixture']
    ids = fx['ids']
    k1_dir = Path(k1_dir)
    kr = json.loads((k1_dir / 'k1_receipt.json').read_text())
    checks = {}
    env = kr.get('env', {})
    boards = resources.get('boards', [])
    checks['two_physical_t4'] = (env.get('n_gpu') == 2 and env.get('gpus') == ['Tesla T4', 'Tesla T4'] and len(boards) == 2
                                 and len({b['uuid'] for b in boards}) == 2 and all(b['name'] == 'Tesla T4' for b in boards))
    checks['bound_variant'] = (kr.get('solo', {}).get('binding_sha256') == binding_digest(cfg)
                               and kr['inputs']['model']['sha256'] == cfg['arm']['b_model']['sha256']
                               and kr.get('solo_stage', {}).get('alias_sha256') == cfg['arm']['b_model']['sha256'])
    a_ids, a_vals, a_cols = _csv_matrix(k1_dir / 'A' / 'submission_A.csv', LAB)
    a = kr['A']
    checks['a_all39_genuine'] = (a_ids == ids and a['internal_fallbacks'] == [] and len(a['attempts']) == 1
                                 and np.isfinite(a_vals).all())
    with open(k1_dir / 'B' / 'B_probs.csv', newline='') as f:
        brows = list(csv.DictReader(f))
    b_ids = [r[UID] for r in brows]
    B = np.array([[float(r[l]) for l in LAB] for r in brows], np.float64)
    Z = np.array([[float(r['z|' + l]) for l in LAB] for r in brows], np.float64)
    b = kr['B']
    checks['b_all39_genuine'] = (b_ids == ids and b['failed_after_retry'] == {} and b['retry'] is None
                                 and np.isfinite(B).all() and np.isfinite(Z).all() and b['fail_cap'] == 0)
    f_ids, f_vals, f_cols = _csv_matrix(final_csv, LAB)
    A = np.array(a_vals, np.float64)
    expect = k1.blend_with_failures(A, B, np.zeros(len(ids), bool)).astype(np.float32)
    got = np.array(f_vals, np.float64).astype(np.float32)
    checks['final_full_population_rank_replay'] = (f_ids == ids and got.shape == expect.shape and bool((got == expect).all()))
    checks['schema_and_unit_interval'] = (f_cols == kr['inputs']['sub_cols'] and a_cols == f_cols
                                          and sorted(f_cols[1:]) == sorted(LAB) and np.isfinite(got).all()
                                          and bool(((got >= 0) & (got <= 1)).all()))
    rho = k1.rho_per_finding(A, B)
    checks['blend_receipt_consistent'] = (kr['blend']['b_failed_rank_A_only'] == {} and kr['blend']['rows'] == len(ids)
                                          and k1.sanity_floor(rho, len(ids)) is None)
    px = json.loads((k1_dir / 'B' / 'px_hashes.json').read_text())
    checks['b_input_hashes_all39'] = set(px) == set(ids) and all(px[s].get('error') is None and px[s].get('img_sha256') for s in ids)
    # cross-board sentinels: the same studies through the other physical T4
    sent = {'run': sentinel_run, 'comparisons': []}
    ok = isinstance(sentinel_run, dict) and 'error' not in sentinel_run
    zpos = {s: i for i, s in enumerate(b_ids)}
    parts = []
    for gi in range(2):
        pdir = k1_dir / 'B' / ('part%d' % gi)
        parts.append({'ids': json.loads((pdir / 'ids.json').read_text()), 'dir': pdir})
    if ok:
        n = cfg['rehearsal']['sentinel_per_board']
        for rerun, orig in ((1, 0), (0, 1)):
            sdir = k1_dir / 'sentinel' / ('board%d' % rerun)
            sids = json.loads((sdir / 'ids.json').read_text())
            logits = json.loads((sdir / 'logits.json').read_text())
            spx = {}
            for fp in sorted((sdir / 'px').glob('px_*.jsonl')):
                for line in fp.read_text().splitlines():
                    r = json.loads(line)
                    spx[r['uid']] = r
            ok = ok and sids == parts[orig]['ids'][:n] and set(logits) == set(sids)
            for sid in sids:
                z = np.asarray(logits.get(sid, [[math.nan] * len(LAB)]), np.float64).mean(0)
                dz = float(np.abs(z - Z[zpos[sid]]).max())
                dp = float(np.abs(1 / (1 + np.exp(-z)) - B[zpos[sid]]).max())
                same_input = spx.get(sid, {}).get('img_sha256') == px[sid].get('img_sha256') and \
                    spx.get(sid, {}).get('meta_sha256') == px[sid].get('meta_sha256')
                sent['comparisons'].append({'original_board': orig, 'rerun_board': rerun, 'max_abs_logit': dz,
                                            'max_abs_probability': dp, 'same_input_hashes': same_input})
                ok = ok and math.isfinite(dp) and dp <= cfg['rehearsal']['sentinel_probability_atol'] and same_input
    checks['cross_board_sentinels'] = bool(ok and len(sent['comparisons']) == 2 * cfg['rehearsal']['sentinel_per_board'])
    # projection from measured fixed / per-study costs
    st = {k: v['seconds'] for k, v in kr['stages'].items()}
    m = {'studies': len(ids), 'target_studies': cfg['rehearsal']['projection_studies'], 'n_gpu': env.get('n_gpu'),
         'pre_controller_seconds': round(timing['controller_t0'] - timing['notebook_started'] - timing['fixture_seconds'], 3),
         'controller_overhead_seconds': round(kr['total_seconds'] - sum(st.values()), 3), 'stages': st,
         'post_controller_allowance_seconds': cfg['rehearsal']['post_controller_allowance_seconds'],
         'gate_seconds': cfg['rehearsal']['gate_seconds'],
         'a': a_timing((k1_dir / 'A' / 'log.txt').read_text(errors='replace')),
         'b_parts': [b_timing(k1, p['dir'], len(p['ids'])) for p in parts]}
    proj = projection(m, k1)
    checks['projection_8_3h_and_inherited_guards'] = bool(proj['passes'])
    t0 = timing['controller_t0']
    windows = {name: (t0 + v['start_s'], t0 + v['start_s'] + v['seconds']) for name, v in kr['stages'].items()}
    res = peaks(resources, windows)
    host_total = res.get('host_total_bytes')
    checks['resources_within_host'] = bool(host_total and res['all']['host_used_peak_bytes'] is not None
                                           and res['all']['host_used_peak_bytes'] < 0.95 * host_total
                                           and all(v > 1 << 30 for v in res['all']['disk_free_min_bytes'].values()))
    windows_by_id = {}
    for p in parts:
        rec = json.loads((p['dir'] / 'receipt.json').read_text())
        for row in rec['per_study']:
            windows_by_id[row[0]] = {'windows': row[1], 'decode_s': row[2], 'render_s': row[3], 'gpu_s': row[4]}
    final_rank = got.astype(np.float64)
    ledger = {'ids': ids, 'splits': fx['splits'], 'labels': LAB,
              'rows': [{'id': sid, 'split': sp, 'a_value': A[i].tolist(), 'b_probability': B[i].tolist(), 'b_logit': Z[i].tolist(),
                        'final_rank': final_rank[i].tolist(), 'b_part': next(j for j, p in enumerate(parts) if sid in p['ids']),
                        'b_cost': windows_by_id.get(sid), 'b_input': {k: px[sid].get(k) for k in ('img_sha256', 'meta_sha256', 'n', 'anchor')}}
                       for i, (sid, sp) in enumerate(zip(ids, fx['splits']))]}
    failed = sorted(k for k, v in checks.items() if not v)
    receipt = {'status': 'PASS' if not failed else 'FAIL', 'mode': 'rehearsal', 'variant': cfg['arm']['variant'],
               'binding_sha256': binding_digest(cfg), 'payload_sha256': kr.get('solo', {}).get('payload_sha256'),
               'checks': checks, 'failed_checks': failed, 'studies': len(ids),
               'rho_A_B': dict(zip(LAB, [round(x, 6) for x in rho])), 'sentinel': sent, 'timing': timing,
               'projection_inputs': m, 'projection': proj, 'resource_peaks': res,
               'costs': {'A': {'fixed_seconds': proj['fixed_seconds']['A'], 'production_seconds': m['a']['gpu_done_s'] - m['a']['loaded_s'],
                               'retry_seconds': 0.0, 'attempts': a['attempts']},
                         'B': {'fixed_seconds': proj['fixed_seconds']['B'], 'parts': m['b_parts'], 'retry': b['retry']}},
               'k1_receipt_sha256': file_sha(k1_dir / 'k1_receipt.json'),
               'elapsed_seconds': round(timing['finished'] - timing['notebook_started'], 1),
               'scope': 'Private offline 2xT4 qualification on the frozen 39 (36 train + 3 test); no labels read; '
                        'not a score, not hidden-population evidence.'}
    return receipt, ledger


def verify_result(result_dir, cfg, k1):
    """Replay a collected /kaggle/working tree: s6solo/ receipts plus the controller's k1/ folder."""
    root = Path(result_dir)
    s = root / 's6solo'
    r = json.loads((s / 'receipt.json').read_text())
    require(r['status'] == 'PASS' and r['mode'] == 'rehearsal' and r['variant'] == cfg['arm']['variant']
            and r['binding_sha256'] == binding_digest(cfg), 'PASSING_BOUND_REHEARSAL')
    require(file_sha(s / 'ledger.json') == r['ledger_sha256'] and file_sha(s / 'resources.json') == r['resources_sha256']
            and file_sha(s / 'rehearsal-final.csv') == r['final_csv_sha256'], 'SAVED_FILE_BINDINGS')
    resources = json.loads((s / 'resources.json').read_text())
    again, ledger = assemble(k1, cfg, root / 'k1', s / 'rehearsal-final.csv', r['timing'], r['sentinel']['run'], resources)
    saved_ledger = json.loads((s / 'ledger.json').read_text())
    for key in ('ledger_sha256', 'resources_sha256', 'final_csv_sha256'):
        again[key] = r[key]
    require(json.loads(json.dumps(again, sort_keys=True, default=str)) == json.loads(json.dumps(r, sort_keys=True, default=str))
            and ledger == saved_ledger, 'INDEPENDENT_RECEIPT_LEDGER_REPLAY')
    return {'receipt_sha256': file_sha(s / 'receipt.json'), 'ledger_sha256': r['ledger_sha256'],
            'final_csv_sha256': r['final_csv_sha256'], 'payload_sha256': r['payload_sha256'],
            'conservative_seconds': r['projection']['conservative_seconds'], 'elapsed_seconds': r['elapsed_seconds'],
            'checks': sorted(r['checks'])}
