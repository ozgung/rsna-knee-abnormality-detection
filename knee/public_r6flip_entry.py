"""R6-flip controller: A-prime Raptor half (R384 + R224, unchanged workers) + heliosli R6 with flip TTA.

Phases on two isolated T4 boards, canonical ordinal modulo-2 sharding, fail-closed between phases:
1. old320  : knee.public_r7_recovery_entry old_worker (A-prime bytes, its qualified state/sentinels/stress).
2. crop224 : knee.public_ws2_entry crop224_worker (A-prime bytes, its qualified state/sentinels).
3. r6      : knee.public_r6flip_worker (compact_infer v4c main() unchanged, observed); rehearsal adds two
             fresh-process sentinel passes on the four A-prime S6 sentinels (pass 1 injects one source-seam failure).
Then A-prime's frozen fill/rank with the per-finding blend and the official CSV.
P6 (sealed cfg['blend'] compact_arms): r6 runs without flip TTA and a fourth stage r7 (heliosli R7 v1, plain, same worker)
runs last; the compact slot is 1/2 rank(R6) + 1/2 rank(R7). Studies the guard cuts from R7 keep rank(R6) as their compact slot.
Deadline guard: every stage stops at T0 + 8.3 h (minus a merge reserve); unreached or failed work falls back to the
other models' ranks (public stage-3 semantics, knee.public_r6flip.recover_guarded) and the entry always writes a
complete official CSV (constant 0.5 only as the last resort), recording how many studies fell back. The rehearsal
stays strict (no fallback) and records/gates the ordinary 8.3 h conservative projection (FAIL_PROJECTION_ONLY).
"""
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from knee import public_r6flip as R6F
from knee import public_r6flip_worker as W

VERSION = R6F.VERSION
MAX_SECONDS = 9 * 3600
PROJECTION_GATE_SECONDS = 8.3 * 3600
POPULATION = 1300
R6_CROSS_TOLERANCE = 1e-3      # fresh process / other board recompiles its encoders (A-prime R7 precedent 2.6e-4)
EXACT_TOLERANCE = 1e-6
R6_PASSES = ('sentinel1', 'sentinel2', 'production')
INJECTION_ERROR = 'no usable window'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, value):
    W.save(path, value)


def compact_flip(cfg, arm='r6'):
    """Flip TTA for a compact arm: P2-P5 R6 flip; P6 seals compact_flip False (R7 is always plain)."""
    return arm == 'r6' and R6F.resolve_compact(cfg.get('blend'))[1]


def r6_worker_config(cfg, board, pass_name, ids, folder, inject_uid, arm='r6'):
    """Board config for a compact arm. The 'r6' / 'r6_mount' slots carry the arm's own bundle (R7 for arm 'r7'); P2-P5
    configs are unchanged (no flip/profile keys), P6 adds 'flip': False and, for R7, the r7 profile and arm."""
    require(arm in R6F.compact_arm_names(cfg.get('blend')), 'SEALED_COMPACT_ARM')
    c = {'mode': cfg['mode'], 'board': board, 'physical': cfg['boards'][board], 'environment': cfg['environment'],
         'pass': pass_name, 'ids': list(ids), 'out': str(folder), 'data': cfg['staged'], 'data_ids': cfg['staged_ids'],
         'data_binding': cfg['stage_binding'], 'code': cfg['code'], 'source_pins': cfg['source_pins'],
         'r6_mount': cfg['r6_mount' if arm == 'r6' else 'r7_mount'], 'anatomy': cfg['anatomy'],
         'r6': {'manifest': cfg['r6flip_binding'][arm]['manifest'], 'cpu_pins': cfg['r6flip_binding'][arm]['cpu_pins'],
                'anatomy_state': (cfg.get('qualified_rehearsal') or {}).get('anatomy_state')},
         'inject_uid': inject_uid}
    flip = compact_flip(cfg, arm)
    if not flip:
        c['flip'] = False
    if arm == 'r7':
        c.update(profile='r7', compact_arm='r7')
    return c


NAFULL_ENTRY_MODULE = 'knee.public_nafull_aprime_entry'
NAFULL_REHEARSAL_MODULE = 'knee.public_nafull_aprime_rehearsal'
NAFULL_FOLDER_ARM = 'old'          # the shipped K62 worker (B0prime old_worker) writes <out>/old/{receipt.json,predictions.npz}
NAFULL_WORKER_PRIVATE = ('probes', 'old_sentinels', 'raptor62_reference')


def nafull_worker_config(cfg, board, folder):
    """Per-board config for the A-triple-prime na-full K62 one-view worker, assembled as RT.make_base_facade(...).run does:
    the controller cfg with the sealed na-full profile (source, manifest, checkpoint, state/grid pins, probes) and binding."""
    na = cfg['nafull']
    worker = na['worker']
    base = {k: v for k, v in cfg.items() if k not in ('nafull', 'blend', 'probes', 'old_sentinels', 'raptor62_reference')}
    base.update(board=board, arm='old', dev=cfg['assignment'][board], physical=cfg['boards'][board], out=str(folder))
    for key in ('public_source', 'decoder_source', 'public_manifest', 'control_checkpoint', 'control_state_sha256', 'control_grid_sha256'):
        base[key] = worker[key]
    base['probes'] = {sid: {'old': probe['old']} for sid, probe in worker.get('probes', {}).items()}
    for key in ('old_sentinels', 'raptor62_reference'):
        if key in worker:
            base[key] = worker[key]
    base['binding'] = {'qualified_state': dict(na['qualified_state'])}
    return base


def merge_nafull(ids, reports, out):
    """A-triple-prime's strict old-arm merge: R7's shard merge bound to the na-full worker version."""
    import types
    from knee import public_nafull_aprime as NA
    from knee import public_r7_recovery_entry as R7
    merge = types.FunctionType(R7.merge.__code__, {**vars(R7), 'VERSION': NA.VERSION}, 'merge_nafull', R7.merge.__defaults__)
    return merge(ids, [(x, out / ('board%d-nafull' % i) / NAFULL_FOLDER_ARM / 'predictions.npz') for i, x in enumerate(reports)],
                 NAFULL_FOLDER_ARM)


def merge_r6(ids, reports, out, pass_name='production', arm='r6'):
    """Complete-population R6 (or, P6, R7) ledger: exact shards, statuses, NaN failures, finite genuine logits."""
    import numpy as np
    from knee.public_s6_recovery_entry import assignments
    from knee.public_s6_qualification import file_sha
    expected = assignments(ids)
    by_id, missing = {}, {}
    require(len(reports) == 2, 'TWO_R6_SHARDS')
    for board, r in enumerate(reports):
        path = out / ('board%d-%s-%s' % (board, arm, pass_name)) / arm / 'predictions.npz'
        require(r.get('version') == VERSION and r.get('status') == 'PASS' and r.get('board') == board and r.get('arm') == arm
                and r.get('pass') == pass_name and r.get('assigned') == expected[board] and r.get('completed') == len(expected[board]),
                'R6_WORKER_TERMINAL_GATE')
        require(r.get('state_before') == r.get('state_after') and r.get('zero_imputed') is True, 'R6_WORKER_STATE_GATE')
        require(r.get('ledger_sha256') == file_sha(path), 'R6_ATOMIC_LEDGER_BINDING')
        with np.load(path, allow_pickle=False) as a:
            got, statuses, p, z = a['study_ids'].tolist(), a['statuses'].tolist(), a['probabilities'], a['logits']
        require(got == expected[board] and len(statuses) == len(got) and p.shape == (len(got), 12) and z.shape == (len(got), 4, 12),
                'R6_SHARD_ORDER_COVERAGE')
        good = np.asarray(statuses) == 'genuine'
        require(set(statuses) <= {'genuine', 'data_local_failure'} and np.isfinite(p[good]).all()
                and ((p[good] >= 0) & (p[good] <= 1)).all() and np.isnan(p[~good]).all()
                and np.isfinite(z[good]).all() and np.isnan(z[~good]).all(), 'R6_GENUINE_ONLY_LEDGER')
        rows = r['records']
        require([x['id'] for x in rows] == got and [x['status'] for x in rows] == statuses
                and r['genuine'] == int(good.sum()) and r['failed'] == int((~good).sum()), 'R6_LEDGER_RECORD_BINDING')
        for row, sid, prob, status in zip(rows, got, p, statuses):
            if status == 'genuine':
                require(row['probabilities'] == prob.tolist(), 'R6_LEDGER_EQUALS_RECORD')
            require(sid not in by_id, 'R6_DUPLICATE_MERGE_ID')
            by_id[sid], missing[sid] = prob, status == 'data_local_failure'
    require(set(by_id) == set(ids), 'R6_COMPLETE_POPULATION')
    return np.array([by_id[s] for s in ids]), np.array([missing[s] for s in ids], dtype=bool)


def projection(setup_seconds, phases, workers, merge_seconds, development_windows, population=POPULATION):
    """A-prime's conservative rule over the measured phases; R6 scales with windows like A-prime's compact slot."""
    d = list(development_windows)
    require(len(d) == 870 and all(isinstance(x, int) and x > 0 for x in d), 'FROZEN870_WINDOW_DISTRIBUTION')
    require(all(math.isfinite(float(x)) and x >= 0 for x in (setup_seconds, merge_seconds)), 'MEASURED_SETUP_MERGE')
    mean = sum(d) / len(d)
    production, overhead, studies = {}, {}, {}
    raptor = [arm for arm in ('old', 'crop224', 'nafull') if arm in workers]   # stage order; crop224 absent in p5
    for arm in raptor:
        production[arm] = max(sum(float(x['production_seconds']) for x in w['records']) for w in workers[arm])
        studies[arm] = sum(len(w['records']) for w in workers[arm])
        require(0 < production[arm] <= phases[arm]['wall_seconds'] and studies[arm] > 0, 'MEASURED_PHASE_PRODUCTION_' + arm)
        overhead[arm] = phases[arm]['wall_seconds'] - production[arm]
    compact = [arm for arm in ('r6', 'r7') if arm in workers]                  # r7: P6 only, after r6
    window_counts = {}
    for arm in compact:
        genuine = [[x for x in w['records'] if x['status'] == 'genuine'] for w in workers[arm]]
        production[arm] = max(sum(float(x['production_seconds']) for x in g) for g in genuine)
        window_counts[arm] = sum(x['windows'] for g in genuine for x in g)
        require(0 < production[arm] <= phases[arm]['wall_seconds'] and window_counts[arm] > 0, 'MEASURED_PHASE_PRODUCTION_' + arm)
        overhead[arm] = phases[arm]['wall_seconds'] - production[arm]
    windows = window_counts['r6']
    sentinel_walls = sum(v['wall_seconds'] for k, v in phases.items() if k.startswith('r6_sentinel'))
    overhead['r6'] = overhead['r6'] + sentinel_walls
    scenarios = []
    for multiplier in (1., 1.25, 1.5):
        for imbalance in (1., 1.15, 1.3):
            seconds = setup_seconds + merge_seconds + sum(overhead.values())
            for arm in raptor:
                seconds = seconds + production[arm] * population / studies[arm] * imbalance
            for arm in compact:
                seconds = seconds + production[arm] * (population * mean * multiplier / window_counts[arm]) * imbalance
            scenarios.append({'hidden_window_multiplier': multiplier, 'shard_imbalance': imbalance, 'seconds': seconds})
    pick = lambda m, i: next(x['seconds'] for x in scenarios if x['hidden_window_multiplier'] == m and x['shard_imbalance'] == i)
    conservative = pick(1.25, 1.15)
    return {'population': population, 'window_distribution': {'count': len(d), 'total': sum(d), 'mean': mean, 'max': max(d)},
            'measured_production_wall_seconds': production, 'measured_fixed_phase_overhead_seconds': overhead,
            'measured_studies': studies, 'measured_r6_windows': windows, 'r6_sentinel_pass_wall_seconds': sentinel_walls,
            **({'measured_r7_windows': window_counts['r7']} if 'r7' in compact else {}),
            'setup_seconds': setup_seconds, 'merge_seconds': merge_seconds, 'scenarios': scenarios,
            'point_seconds': pick(1., 1.), 'conservative_seconds': conservative, 'gate_seconds': PROJECTION_GATE_SECONDS,
            'gate_pass': conservative <= PROJECTION_GATE_SECONDS, 'actual_limit_seconds': MAX_SECONDS,
            'policy': 'A-prime rule: slower shard production, every measured phase overhead once (rehearsal-only sentinel passes, '
                      'stress, reference decodes, compile checks and loader start-up included), 25% more R6 windows and 15% '
                      'additional imbalance. Sensitivities are not hidden-population bounds; A-triple-prime timed out under an '
                      '8.32 h projection.'}


def resources(phases, workers):
    return {'peak_host_rss_bytes': max(p['peaks']['host_rss_bytes'] for p in phases.values()),
            'peak_output_disk_bytes': max(p['peaks']['output_disk_bytes'] for p in phases.values()),
            'minimum_disk_free_bytes': min(p['minimum_disk_free_bytes'] for p in phases.values()),
            'peak_cuda_bytes_by_arm': {arm: [x['peak_cuda_bytes'] for x in workers[arm]]
                                       for arm in ('old', 'crop224', 'nafull', 'r6', 'r7') if arm in workers}}


def inventory():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True)
    boards = []
    for line in text.splitlines():
        index, uuid, name = [x.strip() for x in line.split(',')]
        boards.append({'index': int(index), 'uuid': uuid, 'name': name})
    require([b['index'] for b in boards] == [0, 1] and len({b['uuid'] for b in boards}) == 2
            and all(b['name'] == 'Tesla T4' for b in boards), 'EXACT_TWO_T4_INVENTORY')
    return boards


STAGE_RESERVE_SECONDS = 180          # stages stop this long before T0 + 8.3 h, leaving time for merge and CSV
R6_SELF_STOP_MARGIN = 60             # compact_infer's own --deadline lands this much earlier than the stage kill


def launch_guarded(cfg, out, arm, deadline, pass_name='production', ids_by_board=None, inject_uid=None):
    """One stage on both boards with the stage deadline (monotonic). Same worker modules/configs as A-prime's
    launchers. At the deadline every worker gets SIGTERM (A-prime workers stop at the study boundary with an atomic
    ledger), then SIGKILL after 60 s. Rehearsal: any worker failure is fatal. Entry: a failed board never stops the other."""
    from knee import public_r7_recovery_entry as R7
    from knee.public_adaptation_fit import PlatformInterruption
    from knee.public_s6_recovery_entry import resource_snapshot, worker_environment
    rehearsal = cfg['mode'] == 'rehearsal'
    children, folders = [], []
    start = time.monotonic()
    high = {'host_rss_bytes': 0, 'output_disk_bytes': 0}
    minimum = shutil.disk_usage(out).free
    deadline_hit = False
    try:
        with PlatformInterruption() as stop:
            for board in range(2):
                if arm in ('r6', 'r7'):
                    name = 'board%d-%s-%s' % (board, arm, pass_name)
                    c = r6_worker_config(cfg, board, pass_name, ids_by_board[board], out / name, inject_uid, arm)
                    module, env = W.WORKER_MODULE, {**worker_environment(board), **W.PROBE_DUAL_ENV}
                elif arm == 'nafull':
                    name = 'board%d-nafull' % board
                    c = nafull_worker_config(cfg, board, out / name)
                    module = NAFULL_REHEARSAL_MODULE if rehearsal else NAFULL_ENTRY_MODULE
                    env = worker_environment(board)
                else:
                    name = 'board%d-%s' % (board, arm)
                    c = {**{k: v for k, v in cfg.items() if k not in ('nafull', 'blend')}, 'board': board, 'arm': arm,
                         'dev': cfg['assignment'][board], 'physical': cfg['boards'][board], 'out': str(out / name)}
                    module = ('knee.public_ws2_entry' if arm == 'crop224' else
                              R7.REHEARSAL_MODULE if rehearsal else R7.ENTRY_MODULE)
                    env = worker_environment(board)
                (out / name).mkdir()
                path = out / (name + '-config.json')
                save(path, c)
                log = (out / (name + '.log')).open('xb')
                children.append(subprocess.Popen([sys.executable, '-u', '-m', module, '--worker', str(path)], env=env,
                                                 stdout=log, stderr=subprocess.STDOUT))
                log.close()
                folders.append(out / name / ('old' if arm == 'nafull' else arm))   # the A-triple-prime worker writes under old/
            while any(c.poll() is None for c in children):
                s = resource_snapshot(out)
                for k in high:
                    high[k] = max(high[k], s[k])
                minimum = min(minimum, s['disk_free_bytes'])
                if time.monotonic() > deadline and not deadline_hit:
                    deadline_hit = True
                    for c in children:
                        if c.poll() is None:
                            c.send_signal(signal.SIGTERM)
                    kill_at = time.monotonic() + 60
                if deadline_hit and time.monotonic() > kill_at:
                    break
                if stop.requested:
                    raise InterruptedError('platform interruption')
                if rehearsal and any(c.poll() not in (None, 0) for c in children):
                    raise RuntimeError('%s worker failure; no partial merge' % arm)
                time.sleep(.5)
    finally:
        for c in children:
            if c.poll() is None:
                c.kill()
        for c in children:
            c.wait()
    codes = [c.returncode for c in children]
    if rehearsal:
        require(not deadline_hit and all(x == 0 for x in codes), 'REHEARSAL_STAGE_COMPLETE_' + arm)
    reports = []
    for folder in folders:
        try:
            reports.append(json.loads((folder / 'receipt.json').read_bytes()))
        except (OSError, ValueError):
            reports.append({'status': 'NO_RECEIPT'})
    return reports, {'wall_seconds': time.monotonic() - start, 'sample_interval_seconds': .5, 'peaks': high,
                     'minimum_disk_free_bytes': minimum, 'deadline_hit': deadline_hit, 'exit_codes': codes}


def salvage(ids, out, arm, folder_names):
    """Entry fallback: every genuine row any board's atomic ledger (or, for R6, compact_infer's flushed CSV) holds."""
    import csv
    import numpy as np
    p = np.full((len(ids), 12), np.nan)
    status = np.array(['unavailable'] * len(ids), dtype=object)
    index = {s: i for i, s in enumerate(ids)}
    for name in folder_names:
        ledger = out / name / arm / 'predictions.npz'
        rows = []
        try:
            with np.load(ledger, allow_pickle=False) as a:
                rows = list(zip(a['study_ids'].tolist(), a['probabilities'], a['statuses'].tolist()))
        except Exception:
            if arm in ('r6', 'r7'):
                try:
                    with open(out / name / arm / 'compact' / 'predictions.csv', newline='') as h:
                        for row in list(csv.reader(h))[1:]:
                            if len(row) == 13:
                                rows.append((row[0], np.array([float(x) for x in row[1:]]), 'genuine'))
                except Exception:
                    rows = []
        for sid, prob, st in rows:
            prob = np.asarray(prob, dtype=np.float64)
            if sid in index and st == 'genuine' and prob.shape == (12,) and np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all():
                p[index[sid]], status[index[sid]] = prob, 'genuine'
    return p, status.tolist()


def arm_table(ids, workers, out, entry, blend=None):
    """Strict merges (A-prime's for the R384/R224 arms, A-triple-prime's for na-full); in entry, anything incomplete is
    salvaged (missing rows become 'unavailable'). Only the arms of the sealed blend plus R6 are tabulated."""
    import numpy as np
    from knee import public_r7_recovery_entry as R7
    from knee import public_ws2_entry as WS2
    arrays, statuses, notes = {}, {}, {}
    strict = {'old320': lambda: R7.merge(ids, [(x, out / ('board%d-old' % i) / 'old' / 'predictions.npz')
                                               for i, x in enumerate(workers['old'])], 'old'),
              'crop224': lambda: WS2.merged_arm(ids, workers['crop224'], out, 'crop224'),
              'nafull': lambda: merge_nafull(ids, workers['nafull'], out),
              'r6': lambda: merge_r6(ids, workers['r6'], out),
              'r7': lambda: merge_r6(ids, workers['r7'], out, arm='r7')}
    folders = SALVAGE_FOLDERS
    compact = R6F.compact_arm_names(blend)
    for name in R6F.blend_arm_names(blend) + compact:
        try:
            p, missing = strict[name]()
            st = np.where(missing, 'data_local_failure', 'genuine').astype(object)
            if name in compact:
                deadline = {row['id'] for w in workers[name] for row in w['records'] if row.get('deadline')}
                st[[i for i, s in enumerate(ids) if s in deadline]] = 'unavailable'
            arrays[name], statuses[name], notes[name] = p, st.tolist(), 'strict'
        except Exception as e:
            if not entry:
                raise
            arrays[name], statuses[name] = salvage(ids, out, *folders[name])
            notes[name] = 'salvaged after %s' % type(e).__name__
    return arrays, statuses, notes



SALVAGE_FOLDERS = {'old320': ('old', ['board0-old', 'board1-old']),
                   'crop224': ('crop224', ['board0-crop224', 'board1-crop224']),
                   'nafull': (NAFULL_FOLDER_ARM, ['board0-nafull', 'board1-nafull']),
                   'r6': ('r6', ['board0-r6-production', 'board1-r6-production']),
                   'r7': ('r7', ['board0-r7-production', 'board1-r7-production'])}


def salvage_csv(root, out, ids, csv_path, reason, blend=None):
    """Entry after an unexpected error: blend every genuine ledger row on disk (fallback ranks for the rest).
    Only the arms of the sealed blend plus R6 are read; a variant's missing na-full ledger is simply 'unavailable'."""
    from knee.public_fullfield_entry import test_contract, write_csv
    arrays, statuses = {}, {}
    active = [name for name in SALVAGE_FOLDERS if name in R6F.compact_arm_names(blend) or name in R6F.blend_arm_names(blend)]
    for name in active:
        arm, names = SALVAGE_FOLDERS[name]
        arrays[name], statuses[name] = salvage(ids, Path(out), arm, names)
    require(any(s == 'genuine' for st in statuses.values() for s in st), 'SALVAGE_NOTHING_GENUINE')
    rec = R6F.recover_guarded(arrays, statuses, blend=blend)
    columns, _, _ = test_contract(root, R6F.LABELS)
    return {'csv_sha256': write_csv(csv_path, columns, ids, rec['blend']), 'fallback_studies': rec['fallback_studies'],
            'arm_merge': {name: 'salvaged after %s' % reason for name in active},
            'genuine_rows': {name: statuses[name].count('genuine') for name in active}}

def emergency_csv(root, csv_path):
    """Last resort (entry only): a complete, valid constant submission in official order."""
    from knee.public_fullfield_entry import test_contract, write_csv
    columns, ids, _ = test_contract(root, R6F.LABELS)
    return write_csv(csv_path, columns, ids, [[.5] * 12 for _ in ids])


def run(root, cfg, out, *, started, started_epoch=None):
    """Fresh 2xT4 frozen-39 rehearsal or image-only entry. Entry never aborts: T0 + 8.3 h stage deadline, fallback
    ranks for unreached/failed work, and a complete official CSV in every outcome."""
    from knee.public_fullfield_entry import read_series, test_contract, write_csv
    from knee.public_s6_recovery_entry import assignments, stage
    out = Path(out)
    out.mkdir()
    rehearsal = cfg['mode'] == 'rehearsal'
    binding = cfg['r6flip_binding']
    blend = cfg.get('blend')            # sealed variant blend (None: the frozen R6F-pure arithmetic and stages)
    arm_names = R6F.blend_arm_names(blend)
    compact_names = R6F.compact_arm_names(blend)
    started_epoch = time.time() - (time.monotonic() - started) if started_epoch is None else started_epoch
    deadline = started + R6F.DEADLINE_SECONDS - STAGE_RESERVE_SECONDS
    r = {'status': 'RUNNING', 'version': VERSION, 'mode': cfg['mode'], 'binding': binding,
         'deadline': {'t0_relative_seconds': R6F.DEADLINE_SECONDS, 'stage_reserve_seconds': STAGE_RESERVE_SECONDS,
                      'r6_self_stop_margin_seconds': R6_SELF_STOP_MARGIN}}
    save(out / 'receipt.json', r)
    csv_path = out / 'visible-predictions.csv' if rehearsal else Path('/kaggle/working/submission.csv')
    try:
        require(cfg['mode'] in ('rehearsal', 'entry'), 'MODE')
        require(rehearsal or not any(k in cfg for k in ('probes', 'probe_order', 'old_sentinels', 's6_sentinels', 'r7_reference')),
                'PURE_ENTRY_CFG')
        require(('r7' in compact_names) == ('r7' in binding) and (not rehearsal or ('r7' in compact_names) == ('r7_reference' in cfg)),
                'R7_STAGE_CONFIG_MATCHES_BLEND')
        require('raptor_arms' not in cfg, 'RAPTOR_ARMS_ONLY_VIA_SEALED_BLEND')
        require(('nafull' in arm_names) == ('nafull' in cfg), 'NAFULL_STAGE_CONFIG_MATCHES_BLEND')
        require(rehearsal or not any(k in cfg.get('nafull', {}).get('worker', {}) for k in NAFULL_WORKER_PRIVATE)
                and 'reader_reference' not in cfg.get('nafull', {}), 'PURE_ENTRY_NAFULL_CFG')
        columns, visible, records = test_contract(root, R6F.LABELS, visible=rehearsal)
        require(columns == binding['official_columns'], 'OFFICIAL_COLUMNS')
        paths = {s: str(Path(root) / 'test_series') for s in visible}
        ids = list(visible)
        if rehearsal:
            probes = cfg['probes']
            require(len(probes) == 36 and not set(probes) & set(ids), 'FROZEN36_DISTINCT_PROBES')
            training = read_series(Path(root) / 'train_series.csv')
            for sid in probes:
                require(sid in training, 'PROBE_METADATA')
                records[sid], paths[sid] = training[sid], str(Path(root) / 'train_series')
            require(len(cfg['probe_order']) == 36 and set(cfg['probe_order']) == set(probes), 'FROZEN36_CANONICAL_ORDER')
            ids = cfg['probe_order'] + ids
        cfg = {**cfg, 'staged_ids': ids, 'assignment': assignments(ids),
               'deadline_epoch': started_epoch + R6F.DEADLINE_SECONDS - STAGE_RESERVE_SECONDS - R6_SELF_STOP_MARGIN}
        save(out / 'assignment.json', {'ids': ids, 'assignment': cfg['assignment'], 'policy': 'canonical ordinal modulo2'})
        staged = out / 'images-only'
        cfg.update(staged=str(staged), stage_binding=stage(staged, ids, records, paths))
        cfg['boards'] = inventory()
        r.update(boards=cfg['boards'], visible_ids=visible, stage_binding=cfg['stage_binding'], setup_seconds=time.monotonic() - started)
        save(out / 'receipt.json', r)
        phases, workers, skipped = {}, {}, []
        r.update(phases=phases, workers=workers, skipped_stages=skipped)

        def stage_run(key, arm, **kw):
            if time.monotonic() > deadline:
                require(not rehearsal, 'REHEARSAL_DEADLINE')
                skipped.append(key)
                return [{'status': 'SKIPPED_DEADLINE'}] * 2, {'wall_seconds': 0., 'deadline_hit': True, 'skipped': True,
                                                              'peaks': {'host_rss_bytes': 0, 'output_disk_bytes': 0},
                                                              'minimum_disk_free_bytes': shutil.disk_usage(out).free}
            result = launch_guarded(cfg, out, arm, deadline, **kw)
            return result
        workers['old'], phases['old'] = stage_run('old', 'old')
        save(out / 'receipt.json', r)
        if 'crop224' in arm_names:       # p5 omits the R224 stage entirely (time saved, gates skipped)
            crop_cfg_sentinels = cfg.get('s6_sentinels', []) if rehearsal else []
            cfg['crop224_sentinels'] = crop_cfg_sentinels
            workers['crop224'], phases['crop224'] = stage_run('crop224', 'crop224')
            save(out / 'receipt.json', r)
        if rehearsal:
            sentinels = list(cfg['s6_sentinels'])
            workers['r6_sentinels'] = {}
            for name in R6_PASSES[:2]:
                rep, phase = launch_guarded(cfg, out, 'r6', deadline, pass_name=name, ids_by_board=[sentinels, sentinels],
                                            inject_uid=sentinels[0] if name == 'sentinel1' else None)
                workers['r6_sentinels'][name], phases['r6_' + name] = rep, phase
                save(out / 'receipt.json', r)
        workers['r6'], phases['r6'] = stage_run('r6', 'r6', pass_name='production', ids_by_board=cfg['assignment'])
        save(out / 'receipt.json', r)
        if 'nafull' in arm_names:        # last, after r6: at the T0+8.3 h guard it is the arm cut (0.1 of the blend), not R6 (0.5); entry: a failed or missing na-full only costs its rank
            try:
                workers['nafull'], phases['nafull'] = stage_run('nafull', 'nafull')
            except InterruptedError:
                raise
            except Exception as e:
                if rehearsal:
                    raise
                skipped.append('nafull')
                workers['nafull'] = [{'status': 'STAGE_ERROR', 'error_type': type(e).__name__}] * 2
                phases['nafull'] = {'wall_seconds': 0., 'deadline_hit': False, 'skipped': True, 'stage_error': type(e).__name__,
                                    'peaks': {'host_rss_bytes': 0, 'output_disk_bytes': 0},
                                    'minimum_disk_free_bytes': shutil.disk_usage(out).free}
            save(out / 'receipt.json', r)
        if 'r7' in compact_names:        # P6: last of all; at the guard R7 is cut and those studies keep rank(R6) as compact slot
            try:
                workers['r7'], phases['r7'] = stage_run('r7', 'r7', pass_name='production', ids_by_board=cfg['assignment'])
            except InterruptedError:
                raise
            except Exception as e:
                if rehearsal:
                    raise
                skipped.append('r7')
                workers['r7'] = [{'status': 'STAGE_ERROR', 'error_type': type(e).__name__}] * 2
                phases['r7'] = {'wall_seconds': 0., 'deadline_hit': False, 'skipped': True, 'stage_error': type(e).__name__,
                                'peaks': {'host_rss_bytes': 0, 'output_disk_bytes': 0},
                                'minimum_disk_free_bytes': shutil.disk_usage(out).free}
            save(out / 'receipt.json', r)
        merge_start = time.monotonic()
        arrays, statuses, notes = arm_table(ids, workers, out, entry=not rehearsal, blend=blend)
        rec = R6F.recover_guarded(arrays, statuses, blend=blend)
        index = {s: i for i, s in enumerate(ids)}
        require(len(visible) == len(set(visible)) and set(visible) <= set(index), 'VISIBLE_COVERAGE')
        sha = write_csv(csv_path, columns, visible, rec['blend'][[index[s] for s in visible]])
        merge_seconds = time.monotonic() - merge_start
        fallback = rec['fallback_studies'] > 0 or any(p.get('deadline_hit') for p in phases.values())
        r.update(status='PASS_WITH_FALLBACK' if fallback else 'PASS', completed=len(ids), completed_visible=len(visible),
                 csv_sha256=sha, csv_path=str(csv_path), recovery=R6F.guarded_receipt(rec), arm_merge=notes,
                 fallback_studies=rec['fallback_studies'], zero_failed=rec['union_recovered'] == 0, merge_seconds=merge_seconds,
                 elapsed_seconds=time.monotonic() - started)
        try:
            r['resources'] = resources(phases, workers)
        except Exception as e:
            require(not rehearsal, 'RESOURCES')
            r['resources'] = {'unavailable': type(e).__name__}
        if rehearsal:
            require(r['status'] == 'PASS' and r['elapsed_seconds'] < MAX_SECONDS, 'REHEARSAL_COMPLETE_NO_FALLBACK')
            r['projection'] = projection(r['setup_seconds'], phases, workers, merge_seconds, cfg['development_windows'])
            r['qualification'] = require_rehearsal(r, cfg, out, csv_path)
            r['status'] = 'PASS' if r['projection']['gate_pass'] else 'FAIL_PROJECTION_ONLY'
        save(out / 'receipt.json', r)
        return r
    except BaseException as e:
        r.update(failure_type=type(e).__name__, elapsed_seconds=time.monotonic() - started)
        save(out / 'failure-private.json', {'type': type(e).__name__, 'detail': str(e)[:4000]})
        if rehearsal:
            r['status'] = 'INTERRUPTED' if isinstance(e, (KeyboardInterrupt, InterruptedError, TimeoutError)) else 'FAIL'
            save(out / 'receipt.json', r)
            raise
        if not csv_path.exists() and r.get('visible_ids'):
            try:   # salvage every arm ledger already on disk before the constant (57326253 scored 0.500)
                r.update(status='SALVAGED_AFTER_ERROR', **salvage_csv(root, out, r['visible_ids'], csv_path, type(e).__name__, blend))
            except Exception as e3:
                r['salvage_failure'] = type(e3).__name__
        if not csv_path.exists():
            try:
                r.update(status='EMERGENCY_CONSTANT', csv_sha256=emergency_csv(root, csv_path), fallback_studies='all')
            except Exception as e2:
                r.update(status='FAIL', emergency_failure=type(e2).__name__)
        else:
            r['status'] = 'PASS_WITH_FALLBACK_AFTER_ERROR'
        save(out / 'receipt.json', r)
        if r['status'] == 'FAIL':
            raise
        return r


# ---- rehearsal qualification --------------------------------------------------------------------------------------------
def _close(a, b, tol):
    return len(a) == len(b) == 12 and max(abs(float(x) - float(y)) for x, y in zip(a, b)) <= tol


def require_old(r, cfg):
    """A-prime's old320 rehearsal gates, unchanged in substance (knee.public_r7_recovery_entry.require_rehearsal)."""
    from knee.mirror_input import STRESS_CASES
    ab = cfg['binding']
    rr = r['workers']['old']
    require(len(rr) == 2, 'OLD_REHEARSAL_WORKERS')
    old0 = rr[0]
    stress = old0.get('partial_input_stress', {})
    require(stress.get('status') == 'PASS' and stress.get('cases') == dict.fromkeys(STRESS_CASES, 'PASS') and stress.get('case_count') == 17
            and stress.get('uses_raw_data') is False and stress.get('uses_labels') is False, 'OLD_PARTIAL_INPUT_STRESS')
    guarded = old0.get('guarded_partial_input_stress', {})
    require(guarded.get('status') == 'PASS' and guarded.get('guarded') is True and guarded.get('case_count') == 10, 'OLD_GUARDED_PARTIAL_TEN')
    require(all(x.get('recovery_injection', {}).get('status') == 'PASS' and x['recovery_injection'].get('failure_status') == 'data_local_failure'
                and x['recovery_injection'].get('injected_rows_genuine') is False and x['recovery_injection'].get('state_healthy') is True
                for x in rr), 'OLD_CLASSIFIED_RECOVERY_INJECTIONS')
    for board, x in enumerate(rr):
        require(x['status'] == 'PASS' and x['board'] == board and x['state_before'] == x['state_after'] and x['completed'] == len(x['assigned'])
                and x['peak_cuda_bytes'] > 0 and x['peak_rss_bytes'] > 0 and x['zero_failed'] is True and x['zero_imputed'] is True
                and x['fp32_recoveries'] == 0, 'OLD_WORKER_STATE_RESOURCE')
        require(x['state_before']['state'] == ab['qualified_state']['old_state']
                and x.get('constructor_grid_sha256') == ab['qualified_state']['old_grid'], 'OLD_QUALIFIED_STATE_GRID')
        require(x['environment']['torch'] == ab['environment']['torch'] and x['environment']['packages'] == ab['environment']['packages'],
                'OLD_ENVIRONMENT')
        require(x['environment']['physical_uuid'] == r['boards'][board]['uuid'] and x['environment']['cuda_device_count'] == 1, 'OLD_ISOLATED_BOARD')
        require(len(x['sentinels']) == 4 and [s['id'] for s in x['sentinels']] == cfg['old_sentinels']
                and all(s['strict_reload'] for s in x['sentinels']), 'OLD_SENTINELS')
        for s in x['sentinels']:
            require(all(s['input'].get(k) == v for k, v in ab['sentinel_inputs']['old'][s['id']].items()), 'OLD_SENTINEL_INPUTS')
            e = ab['sentinel_probabilities']['old'][s['id']]
            require(_close(s['probabilities'], e, EXACT_TOLERANCE) and _close(s['reloaded_probabilities'], e, EXACT_TOLERANCE),
                    'OLD_SENTINEL_EXACT_REFERENCE')
    for a, b in zip(rr[0]['sentinels'], rr[1]['sentinels']):
        require(a['input'] == b['input'] and _close(a['probabilities'], b['probabilities'], EXACT_TOLERANCE), 'OLD_CROSS_BOARD_SENTINEL')
    rows = {row['id']: row for w in rr for row in w['records']}
    require(len(rows) == 39, 'OLD_COMPLETE39')
    for sid in ab['s6_probe_ids']:
        x = rows[sid]
        require(x.get('source_reference_parity') is True and x.get('strict_reload') is True and 0 <= x.get('reload_max_abs', math.inf) <= EXACT_TOLERANCE,
                'OLD_ALL32_SOURCE_REPEAT_RELOAD')
    return rows


def require_crop224(r, cfg, old_rows):
    """A-prime's R224 rehearsal gates (knee.public_ws2_entry.require_rehearsal), state bound to A-prime's qualified R224."""
    import numpy as np
    from knee import public_ws2_checkpoint as C224
    from knee.public_ws2_entry import crop_parity_ok
    ab, b6 = cfg['binding'], cfg['r6flip_binding']
    workers = r['workers']['crop224']
    require(len(workers) == 2 and len({x['environment']['physical_uuid'] for x in workers}) == 2, 'R224_TWO_PHYSICAL_T4')
    for i, x in enumerate(workers):
        require(x.get('status') == 'PASS' and x.get('slot') == 'native224' and x.get('completed') == len(x['assigned'])
                and x['environment'].get('physical_uuid') == r['boards'][i]['uuid'] and x['environment'].get('torch') == ab['environment']['torch']
                and x['environment'].get('packages') == ab['environment']['packages'] and x.get('state_before') == x.get('state_after')
                and x['model'].get('checkpoint_sha256') == C224.CHECKPOINT_SHA256 and x['model'].get('arch') == C224.ARCH
                and x['model'].get('res') == 224 and x.get('zero_failed') is True and x.get('fp32_recoveries') == 0
                and x.get('peak_cuda_bytes', 0) > 0 and x.get('peak_rss_bytes', 0) > 0 and x.get('ledger_sha256')
                and len(x.get('sentinels', [])) == 4
                and all(s.get('strict_reload') is True and crop_parity_ok(s['input'].get('crop224_cpu_cuda_max_abs'),
                                                                          s['input'].get('crop224_cpu_cuda_mean_abs')) for s in x['sentinels']),
                'R224_STATE_RELOAD_CROP_RESOURCES')
    require(workers[0]['state_before'] == workers[1]['state_before'] == b6['aprime']['r224_state'], 'R224_APRIME_QUALIFIED_STATE')
    for w in workers:
        require([s['id'] for s in w['sentinels']] == cfg['s6_sentinels'], 'R224_SENTINEL_IDENTITY')
        for s in w['sentinels']:
            original = old_rows[s['id']]['input']
            require(all(s['input'][k] == original[k] for k in ('volume_sha256', 'mask_sha256')), 'R224_SENTINEL_GUARDED384_SOURCE')
            p, q = np.asarray(s['probabilities']), np.asarray(s['reloaded_probabilities'])
            require(p.shape == q.shape == (12,) and np.isfinite(p).all() and np.isfinite(q).all() and ((p >= 0) & (p <= 1)).all()
                    and float(np.max(np.abs(p - q))) <= EXACT_TOLERANCE, 'R224_SENTINEL_RELOAD')
    for a, b in zip(workers[0]['sentinels'], workers[1]['sentinels']):
        require(a['id'] == b['id'] and a['input'] == b['input'] and _close(a['probabilities'], b['probabilities'], EXACT_TOLERANCE),
                'R224_CROSS_BOARD_SENTINEL')


def require_nafull(r, cfg, old_rows):
    """The A-triple-prime packet's own na-full acceptance (RT._nafull_rehearsal_checks: stress, typed recovery injection, state and
    grid, all-39 input/decode-count reference parity, 36 strict source repeats and reloads, visible input-only parity), plus the
    exact checkpoint/state/grid pins, two isolated pinned T4 boards, frozen-39 genuine completion in assigned order, atomic ledger
    binding and cross-board sentinel equality. Probabilities of the K62 checkpoint have no external reference by construction."""
    import numpy as np
    from knee import public_nafull_aprime as NA
    from knee import public_nafull_aprime_runtime as RT
    from knee.public_s6_recovery_entry import assignments
    na, ab = cfg['nafull'], cfg['binding']
    workers = r['workers']['nafull']
    ids = cfg['staged_ids']
    expected = assignments(ids)
    manifest = na['worker']['public_manifest']
    require(na['checkpoint'] == NA.CHECKPOINT and manifest['checkpoint_sha256'] == NA.CHECKPOINT['sha256']
            and manifest['checkpoint_file'] == NA.CHECKPOINT['file'] == na['worker']['control_checkpoint'].rsplit('/', 1)[-1]
            and manifest['weight_dataset'] == NA.CHECKPOINT['dataset'] and na['worker']['control_state_sha256'] == NA.CHECKPOINT['state_sha256']
            and na['worker']['control_grid_sha256'] == NA.CONSTRUCTOR_GRID_SHA256
            and na['qualified_state'] == {'old_state': NA.CHECKPOINT['state_sha256'], 'old_grid': NA.CONSTRUCTOR_GRID_SHA256},
            'NAFULL_EXACT_CHECKPOINT_PINS')
    require(len(workers) == 2 and len({x['environment']['physical_uuid'] for x in workers}) == 2, 'NAFULL_TWO_PHYSICAL_T4')
    for board, x in enumerate(workers):
        require(x.get('version') == NA.VERSION and x.get('status') == 'PASS' and x.get('board') == board and x.get('arm') == 'old'
                and x.get('assigned') == expected[board] and x.get('completed') == len(expected[board])
                and x['environment']['torch'] == ab['environment']['torch'] and x['environment']['packages'] == ab['environment']['packages']
                and x['environment']['physical_uuid'] == r['boards'][board]['uuid'] and x['environment']['cuda_device_count'] == 1
                and x.get('state_before') == x.get('state_after') and x['state_before']['state'] == NA.CHECKPOINT['state_sha256']
                and x.get('constructor_grid_sha256') == NA.CONSTRUCTOR_GRID_SHA256
                and x.get('zero_failed') is True and x.get('zero_imputed') is True and x.get('fp32_recoveries') == 0
                and x.get('peak_cuda_bytes', 0) > 0 and x.get('peak_rss_bytes', 0) > 0 and x.get('ledger_sha256'), 'NAFULL_WORKER_STATE_RESOURCE')
        require(all(row['status'] == 'genuine' for row in x['records']) and [row['id'] for row in x['records']] == expected[board],
                'NAFULL_FROZEN39_GENUINE_COMPLETE')
    sentinels = na['worker']['old_sentinels']
    for w in workers:
        require([s['id'] for s in w['sentinels']] == sentinels and len(sentinels) == 4, 'NAFULL_SENTINEL_IDENTITY')
    for a, b in zip(workers[0]['sentinels'], workers[1]['sentinels']):
        require(a['id'] == b['id'] and a['input'] == b['input'] and _close(a['probabilities'], b['probabilities'], EXACT_TOLERANCE)
                and _close(a['reloaded_probabilities'], b['reloaded_probabilities'], EXACT_TOLERANCE), 'NAFULL_CROSS_BOARD_SENTINEL')
    rows = {row['id']: row for w in workers for row in w['records']}
    require(set(rows) == set(ids) and len(rows) == 39, 'NAFULL_COMPLETE39')
    RT._nafull_rehearsal_checks({'workers': {'old': workers}, 'visible_ids': r['visible_ids']},
                                {'nafull_reader_reference': na['reader_reference'], 'qualified_state': na['qualified_state'],
                                 's6_probe_ids': ab['s6_probe_ids'], 'old_historical_ids': ab['old_historical_ids']})
    return {'state': NA.CHECKPOINT['state_sha256'], 'grid': NA.CONSTRUCTOR_GRID_SHA256, 'sentinels': len(sentinels), 'studies': len(rows)}


def require_r6_worker(x, r, cfg, board, pass_name, ids, arm='r6'):
    b6, ab = cfg['r6flip_binding'], cfg['binding']
    pins = b6[arm]['cpu_pins']
    flip = compact_flip(cfg, arm)
    require(x.get('version') == VERSION and x.get('status') == 'PASS' and x.get('board') == board and x.get('pass') == pass_name
            and x.get('arm') == arm and x.get('slot') == ('r6flip' if flip else arm + '-plain')
            and x.get('assigned') == list(ids) and x.get('completed') == len(ids), 'R6_WORKER_PASS')
    env = x['environment']
    require(env['torch'] == ab['environment']['torch'] and env['packages'] == ab['environment']['packages']
            and env['physical_uuid'] == r['boards'][board]['uuid'] and env['cuda_device_count'] == 1, 'R6_ISOLATED_PINNED_BOARD')
    folds = x['folds']
    require(len(folds) == 4, 'R6_FOUR_FOLDS')
    for f, pin in zip(folds, pins['folds']):
        require(f['file'] == pin['file'] and f['file_sha256'] == pin['file_sha256'] and f['strict'] is True
                and all(f[k] == pin[k] for k in ('state', 'buffers', 'constructor', 'constructor_args'))
                and f['reload']['strict'] is True and all(f['reload'][k] == f[k] for k in ('state', 'buffers', 'constructor')),
                'R6_FOLD_BINDING')
        s = f.get('sentinel') or {}
        require(s.get('finite') is True and 0 <= s.get('max_abs_dprob', math.inf) <= W.RELOAD_DPROB_TOL and s.get('compiled_deployed') is True,
                'R6_FRESH_RELOAD_SENTINEL')
    require(x['state_before'] == x['state_after'] == R6F.digest([p['state'] for p in pins['folds']]), 'R6_STATE_BEFORE_AFTER')
    if flip:
        require(x['flip']['perm'] == pins['flip']['perm'] and x['flip']['bin_perm'] == pins['flip']['bin_perm'], 'R6_FLIP_PERMUTATION_PIN')
    else:
        require(x['flip'] == {'enabled': False}, 'PLAIN_PASS_NO_FLIP')
    c = x['compile']
    require(c.get('requested') is True and c.get('active') is True and c.get('fallback') is None, 'R6_COMPILED_NO_FALLBACK')
    require(x['fp32_recoveries'] == 0, 'R6_NO_FP32_RESCUE')
    require(x['loaders'] and all(l['num_workers'] == 2 and l['prefetch_factor'] == 1 and l['timeout'] == 900. and l['context'] == 'spawn'
                                 for l in x['loaders']), 'R6_TWO_SPAWN_LOADER_WORKERS')
    children = x['children']
    require(children and all(ch['before'] is not None and ch['changed'] is False and ch['after'] == ch['before']
                             and ch['before']['side']['state'] == pins['side']['state']
                             and ch['before']['side']['constructor'] == pins['side']['constructor']
                             and ch['before']['anatomy'] is not None for ch in children)
            and sum(ch['items'] for ch in children) >= len(ids), 'R6_LOADER_CHILD_ANATOMY_SIDE_BINDING')
    require(x['peak_cuda_bytes'] > 0 and x['peak_rss_bytes'] > 0, 'R6_RESOURCES')
    for row in x['records']:
        if row['status'] == 'genuine':
            require(row['reconstruction_max_abs'] <= W.RECONSTRUCTION_ATOL and row['mirror_input_equal'] is (True if flip else None)
                    and row['checksum_equal'] is True and row['fluid_status_ok'] is True and row['fp32_rescues'] == 0
                    and not row['stack_fallback'] and row['calls'] == (8 if flip else 4), 'R6_STUDY_GATES')
    return {ch['before']['anatomy']['state'] for ch in children}


def require_r6(r, cfg):
    import numpy as np
    from knee.public_s6_recovery_entry import assignments
    ab = cfg['binding']
    sentinels = cfg['s6_sentinels']
    anatomy = set()
    observed = {}
    for pass_name in R6_PASSES:
        reports = r['workers']['r6'] if pass_name == 'production' else r['workers']['r6_sentinels'][pass_name]
        ids_by_board = assignments(cfg['staged_ids']) if pass_name == 'production' else [sentinels, sentinels]
        require(len(reports) == 2, 'R6_TWO_BOARDS_' + pass_name)
        for board, x in enumerate(reports):
            anatomy |= require_r6_worker(x, r, cfg, board, pass_name, ids_by_board[board])
            for row in x['records']:
                if pass_name == 'sentinel1' and row['id'] == sentinels[0]:
                    require(row['status'] == 'data_local_failure' and row['observed_item'] is True
                            and INJECTION_ERROR in str(row['item_error']), 'R6_INJECTED_SOURCE_FAILURE')
                    continue
                require(row['status'] == 'genuine', 'R6_HEALTHY_MUST_BE_GENUINE')
                observed.setdefault(row['id'], []).append((pass_name, board, row))
            if pass_name == 'sentinel1':
                injected = [i for ch in x['children'] for i in ch['injected']]
                require(len(injected) == 1 and INJECTION_ERROR in str(injected[0]['error']), 'R6_ONE_INJECTION_PER_BOARD')
                require(x['failed'] == 1 and x['genuine'] == 3, 'R6_INJECTION_COUNTS')
            else:
                require(x['failed'] == 0 and all(not ch['injected'] for ch in x['children']), 'R6_NO_INJECTION')
    require(len(anatomy) == 1, 'R6_ANATOMY_STATE_CROSS_BOARD_PASS')
    require(set(observed) == set(cfg['staged_ids']), 'R6_COMPLETE39')
    for sid in sentinels:
        rows = observed[sid]
        require(len(rows) >= 3, 'R6_SENTINEL_REPEATS')
        e = ab['sentinel_inputs']['s6'][sid]
        require(all(row['input'] == rows[0][2]['input'] and row['input']['render_windows'] == e['render_windows']
                    and row['input']['metadata'] == e['metadata'] for _, _, row in rows), 'R6_SENTINEL_INPUT_PARITY')
        for _, _, a in rows:
            for _, _, b in rows:
                require(_close(a['probabilities'], b['probabilities'], R6_CROSS_TOLERANCE), 'R6_SENTINEL_FRESH_PROCESS_CROSS_BOARD')
    parity = 0
    for sid, probe in cfg['probes'].items():
        e = (probe.get('s6') or {}).get('input')
        if not e:
            continue
        row = [row for p, _, row in observed[sid] if p == 'production'][0]
        require(row['input']['render_windows'] == e['render_windows'] and row['input']['metadata'] == e['metadata'], 'R6_APRIME_R7_INPUT_PARITY')
        parity += 1
    require(parity == 32, 'R6_ALL32_INPUT_PARITY')
    return {'sentinel_repeats': {sid: len(observed[sid]) for sid in sentinels}, 'r7_input_parity_studies': parity,
            'anatomy_state': sorted(anatomy)[0]}


R7_REFERENCE_TOLERANCE = 5e-3    # A-prime's heliosli-visible tolerance (HELIOSLI_VISIBLE_ATOL): two R7 implementations, own compiles


def require_r7(r, cfg, anatomy_state):
    """P6 R7 stage gates: pinned R7 folds/side/compile/loaders on both isolated boards, all 39 genuine in assigned order,
    render input byte-identical to the R6 stage's for every study (same decode), loader-child anatomy state equal to R6's, and
    the deployed R7 probabilities within 5e-3 of A-prime's qualified R7 run on the same 39 studies (cfg['r7_reference'])."""
    from knee.public_s6_recovery_entry import assignments
    reports = r['workers']['r7']
    ids_by_board = assignments(cfg['staged_ids'])
    require(len(reports) == 2, 'R7_TWO_BOARDS')
    anatomy = set()
    for board, x in enumerate(reports):
        anatomy |= require_r6_worker(x, r, cfg, board, 'production', ids_by_board[board], 'r7')
        require(x['failed'] == 0 and all(not ch['injected'] for ch in x['children']), 'R7_NO_INJECTION')
    require(anatomy == {anatomy_state}, 'R7_ANATOMY_STATE_EQUALS_R6')
    r6_rows = {row['id']: row for w in r['workers']['r6'] for row in w['records']}
    rows = {row['id']: row for w in reports for row in w['records']}
    require(set(rows) == set(cfg['staged_ids']) == set(r6_rows) and all(row['status'] == 'genuine' for row in rows.values()),
            'R7_COMPLETE39_GENUINE')
    require(all(rows[s]['input'] == r6_rows[s]['input'] and rows[s]['windows'] == r6_rows[s]['windows'] for s in rows),
            'R7_INPUT_EQUALS_R6_INPUT')
    ref = cfg['r7_reference']['probabilities']
    require(set(ref) == set(rows), 'R7_REFERENCE_COVERAGE')
    diffs = {s: max(abs(float(a) - float(b)) for a, b in zip(rows[s]['probabilities'], ref[s])) for s in rows}
    worst = max(diffs.values())
    require(worst <= R7_REFERENCE_TOLERANCE, 'R7_APRIME_REFERENCE_PARITY')
    return {'studies': len(rows), 'input_equals_r6': True, 'aprime_reference_max_abs': worst,
            'aprime_reference_mean_max_abs': sum(diffs.values()) / len(diffs), 'tolerance': R7_REFERENCE_TOLERANCE,
            'reference_receipts_sha256': cfg['r7_reference']['receipts_sha256']}


def require_rehearsal(r, cfg, out, csv_path):
    """Every saved gate of the R6-flip rehearsal except the ordinary projection gate, which is returned."""
    import csv
    import numpy as np
    from knee.public_s6_qualification import file_sha
    b6 = cfg['r6flip_binding']
    require(r.get('version') == VERSION and r.get('mode') == 'rehearsal' and r.get('binding') == b6, 'EXACT_REHEARSAL_BINDING')
    require(r.get('completed') == 39 and r.get('completed_visible') == 3 and r.get('zero_failed') is True
            and r['recovery']['union_recovered'] == 0, 'REHEARSAL_COMPLETE39')
    require(len(r.get('boards', [])) == 2 and len({x['uuid'] for x in r['boards']}) == 2 and all(x['name'] == 'Tesla T4' for x in r['boards']),
            'TWO_PHYSICAL_T4')
    ids = json.loads((Path(out) / 'assignment.json').read_bytes())['ids']
    ab = cfg['binding']
    require(ids == ab['s6_probe_ids'] + ab['old_historical_ids'] + r['visible_ids'] == cfg['probe_order'] + r['visible_ids'],
            'EXACT_FROZEN39_ORDER')
    cfg = {**cfg, 'staged_ids': ids}
    blend = cfg.get('blend')
    arm_names = R6F.blend_arm_names(blend)
    compact_names = R6F.compact_arm_names(blend)
    require(('nafull' in arm_names) == ('nafull' in cfg) and ('nafull' in arm_names) == ('nafull' in r['workers'])
            and ('crop224' in arm_names) == ('crop224' in r['workers']) and ('r7' in compact_names) == ('r7' in r['workers']),
            'REHEARSAL_STAGES_MATCH_SEALED_BLEND')
    old_rows = require_old(r, cfg)
    if 'crop224' in arm_names:
        require_crop224(r, cfg, old_rows)
    nafull = require_nafull(r, cfg, old_rows) if 'nafull' in arm_names else None
    r6 = require_r6(r, cfg)
    r7 = require_r7(r, cfg, r6['anatomy_state']) if 'r7' in compact_names else None
    phases = r.get('phases') or {}
    require(set(phases) == {'old', *[a for a in ('crop224', 'nafull') if a in arm_names], 'r6_sentinel1', 'r6_sentinel2', 'r6',
                            *[a for a in ('r7',) if a in compact_names]}
            and all(p['wall_seconds'] > 0 and p['peaks']['host_rss_bytes'] > 0 and p['peaks']['output_disk_bytes'] > 0
                    and p['minimum_disk_free_bytes'] > 0 for p in phases.values()), 'MEASURED_PHASE_RESOURCES')
    require(r.get('resources') == resources(phases, r['workers']), 'REPLAY_RESOURCES')
    replay = projection(r['setup_seconds'], phases, r['workers'], r['merge_seconds'], cfg['development_windows'])
    require(replay == r.get('projection'), 'REPLAY_PROJECTION')
    require(0 < r['elapsed_seconds'] < MAX_SECONDS, 'ACTUAL_NINE_HOUR_LIMIT')
    arrays, statuses, _ = arm_table(ids, r['workers'], Path(out), entry=False, blend=blend)
    rec = R6F.recover_guarded(arrays, statuses, blend=blend)
    require(rec['mode'] == ('aprime_complete_population' if blend is None else 'variant_complete_population')
            and rec['fallback_studies'] == 0
            and R6F.guarded_receipt(rec) == r['recovery'], 'REPLAY_RECOVERY')
    require(r.get('csv_sha256') == file_sha(csv_path), 'CSV_BINDING')
    with Path(csv_path).open(newline='') as h:
        rows = list(csv.reader(h))
    require(rows[0] == b6['official_columns'] and len(rows) == 4 and [x[0] for x in rows[1:]] == r['visible_ids'], 'CSV_SCHEMA_ORDER')
    actual = np.array([[float(v) for v in row[1:]] for row in rows[1:]])
    expected = rec['blend'][[ids.index(s) for s in r['visible_ids']]]
    require(actual.shape == (3, 12) and np.isfinite(actual).all() and ((actual >= 0) & (actual <= 1)).all()
            and float(np.max(np.abs(actual - expected))) <= 1e-12, 'CSV_RECOMPUTED')
    return {'r6': r6, **({'nafull': nafull} if nafull is not None else {}), **({'r7': r7} if r7 is not None else {}),
            'projection_gate_pass': replay['gate_pass'],
            'conservative_seconds': replay['conservative_seconds'], 'point_seconds': replay['point_seconds'],
            'all_ledgers_recomputed': True}
