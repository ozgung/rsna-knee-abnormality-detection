"""Image-only full384 inference and immutable rehearsal evidence gates."""
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import signal
import shutil
import sys
import time
from collections import Counter

from knee.reference import canonical_bytes, sha256

VERSION = 'full384-image-only-v1'
MAX_SECONDS = 9 * 3600
ATOL = 1e-6
METHOD = {'resolution':384, 'windows':94, 'chunk':16, 'precision':'CUDA fp16', 'tta':'equal original/anatomical-mirror probabilities', 'constructor_grids':True}
SERIES_COLUMNS = ['StudyInstanceUID', 'SeriesInstanceUID', 'Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    """Keep the last complete diagnostics when a write or later phase fails."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('w') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def safe_id(value):
    return isinstance(value, str) and bool(value) and value not in ('.', '..') and '/' not in value and '\\' not in value


def read_series(path):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == SERIES_COLUMNS, 'Official series schema differs')
        records, seen = {}, set()
        for row in reader:
            require(set(row) == set(SERIES_COLUMNS) and all(row[k] is not None for k in SERIES_COLUMNS), 'Malformed series row')
            sid, series = row['StudyInstanceUID'], row['SeriesInstanceUID']
            require(safe_id(sid) and safe_id(series) and series not in seen, 'Invalid/duplicate series ID')
            require(row['Fluid_Sensitive'] in ('0', '1') and row['Fat_Suppression'] in ('0', '1') and row['Anatomical_Plane'] in ('Sagittal', 'Coronal', 'Axial'), 'Invalid series categories')
            seen.add(series); records.setdefault(sid, []).append(row)
    require(records, 'Empty series population')
    return records


def test_contract(root, targets, *, visible=False):
    """Actual test.csv supplies coverage and order; sample supplies target header.

    Public sample rows may remain unchanged under hidden-test replacement.
    """
    root = Path(root)
    with (root / 'sample_submission.csv').open(newline='') as handle:
        reader = csv.reader(handle); columns = next(reader); rows = list(reader)
    require(columns == ['StudyInstanceUID', *targets] and len(columns) == 13 and len(set(columns)) == 13, 'Official target order differs')
    require(all(len(row) == 13 and safe_id(row[0]) for row in rows), 'Malformed sample rows')
    sample_ids = [row[0] for row in rows]
    require(len(sample_ids) == len(set(sample_ids)), 'Duplicate sample IDs')
    records = read_series(root / 'test_series.csv')
    with (root / 'test.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ['StudyInstanceUID'], 'Official test ID schema differs')
        test_rows = list(reader)
    require(all(set(row) == {'StudyInstanceUID'} for row in test_rows), 'Malformed test ID rows')
    ids = [row['StudyInstanceUID'] for row in test_rows]
    require(ids and len(ids) == len(set(ids)) and all(safe_id(x) for x in ids) and set(ids) == set(records), 'Actual test/series coverage differs')
    require(not visible or len(ids) == 3, 'Rehearsal requires three visible studies')
    return columns, ids, records


def write_csv(path, columns, ids, predictions):
    require(len(columns) == 13 and len(set(columns)) == 13 and columns[0] == 'StudyInstanceUID', 'CSV schema')
    require(ids and len(set(ids)) == len(ids) and all(safe_id(x) for x in ids) and len(predictions) == len(ids), 'CSV exact coverage')
    rows = [[float(x) for x in row] for row in predictions]
    require(all(len(row) == 12 and all(math.isfinite(x) and 0 <= x <= 1 for x in row) for row in rows), 'CSV finite probabilities')
    path = Path(path)
    require(not path.exists(), 'Do not overwrite CSV evidence')
    temporary = path.with_name(path.name + '.tmp')
    try:
        with temporary.open('x', newline='') as handle:
            writer = csv.writer(handle); writer.writerow(columns)
            writer.writerows([sid, *row] for sid, row in zip(ids, rows))
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256(path.read_bytes())


def require_rehearsal(report, csv_path, binding):
    require(report.get('status') == 'PASS' and report.get('version') == VERSION and report.get('mode') == 'rehearsal', 'Passing rehearsal required')
    require(binding.get('version') == VERSION and binding.get('method') == METHOD, 'Exact method/version binding')
    require(report.get('binding') == binding and report.get('source_bundle_sha256') == binding['source_bundle_sha256'], 'Exact rehearsal build/source binding required')
    require(report.get('completed_visible') == 3 and report.get('completed_probes') == 32 and report.get('prediction_fallbacks') == 0, 'Complete visible/probe coverage required')
    require(report.get('final_state_and_all_buffers_immutable') is True and report.get('strict_reload_gate') is True and report.get('constructor_oracle_gate') is True and report.get('same_image_hash_gate') is True, 'State/input gates required')
    require(set(report.get('max_prediction_errors', {})) == {'qualified', 'repeat', 'reload'} and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= ATOL for x in report['max_prediction_errors'].values()), 'Prediction gates required')
    require(0 < report.get('elapsed_seconds', float('inf')) <= MAX_SECONDS, 'Actual platform runtime gate')
    require(report.get('environment') == binding['environment'], 'Exact runtime environment binding')
    require(report.get('cuda_boards') and all(isinstance(report.get(k), (int,float)) and math.isfinite(report[k]) and report[k]>0 for k in ('setup_seconds','peak_host_rss_bytes','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','disk_free_bytes','output_disk_bytes')), 'Measured runtime/resource evidence required')
    require(report.get('csv_sha256') == sha256(Path(csv_path).read_bytes()), 'Rehearsal CSV bytes differ')
    with Path(csv_path).open(newline='') as handle:
        reader = csv.reader(handle); columns = next(reader); rows = list(reader)
    require(columns == binding['columns'] and [r[0] for r in rows] == binding['visible_ids'] and len(rows) == 3, 'Rehearsal CSV order differs')
    require(all(len(r) == 13 and all(math.isfinite(float(x)) and 0 <= float(x) <= 1 for x in r[1:]) for r in rows), 'Rehearsal CSV values')
    require(report.get('state_grid_gate') == binding['qualified_grid'] and report.get('candidate_state', {}).get('state') == binding['qualified_grid']['state_sha256'] and len(report.get('candidate_state', {}).get('buffers', '')) == 64, 'Qualified state and constructor grid binding')
    require(all(report.get('decode_counters', {}).get(k, -1) == 0 for k in ('resource_errors', 'invalid_pixels', 'pixel_failures', 'fast_bad_headers', 'header_parse_failures')), 'Decode failures rejected')


def infer_study(public, decoder, candidate, sid, records, image_root, *, audit=False):
    """The exact same function is called by rehearsal visible and pure entry."""
    import numpy as np
    import torch
    from knee.mirror_input import prepare_study
    from knee.public_fullfield import fullfield_windows, infer_candidate
    tick = time.monotonic()
    volume, mask, counts = prepare_study(public, decoder, sid, records, image_root)
    require(not any(counts.get(k, 0) for k in ('resource_errors', 'invalid_pixels', 'pixel_failures', 'fast_bad_headers', 'header_parse_failures')), 'Decoder failed; no imputation')
    decoded = time.monotonic()
    windows = [fullfield_windows(public, volume, mask, 'cuda', mirror=m) for m in (False, True)]
    torch.cuda.synchronize(); windowed = time.monotonic()
    p = np.asarray(infer_candidate(public, candidate, *windows), dtype=np.float32)
    torch.cuda.synchronize(); forwarded = time.monotonic()
    from knee.public_adaptation_pilot import state_fingerprint
    hashes = {'volume_sha256': sha256(volume.tobytes()), 'mask_sha256': sha256(mask.tobytes()), 'windows': {k:state_fingerprint(w) for k,w in zip(('normal', 'mirrored'), windows)}} if audit else None
    return p, counts, hashes, {'decode': decoded-tick, 'windows': windowed-decoded, 'forward': forwarded-windowed, 'hash_audit': time.monotonic()-forwarded, 'total': forwarded-tick}


def run(root, checkpoint, public_source, decoder_source, manifest, contract, out, *, started):
    import numpy as np
    import torch
    from knee.reference import load_definitions, load_model
    from knee.public_fullfield import make_candidate, snapshot, require_immutable, prediction_error
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    mode, binding = contract['mode'], contract['binding']
    require(mode in ('entry', 'rehearsal'), 'Unknown mode')
    report = {'status':'RUNNING', 'version':VERSION, 'mode':mode, 'binding':binding, 'source_bundle_sha256':os.environ.get('KNEE_SOURCE_SHA256'), 'prediction_fallbacks':0, 'completed_visible':0, 'completed_probes':0}
    counters, timings, errors = Counter(), [], {'qualified':0., 'repeat':0., 'reload':0.}
    stage = 'setup'
    def save():
        report.update(elapsed_seconds=time.monotonic()-started, phase=stage, failure_stage=stage if report['status']=='FAIL' else None, decode_counters=dict(counters), max_prediction_errors=errors,
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024), disk_free_bytes=shutil.disk_usage(out).free,
            output_disk_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()))
        if torch.cuda.is_available():
            report.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_json(out/'receipt.json', report)
    save()
    try:
        require(binding.get('version') == VERSION and binding.get('method') == METHOD and binding.get('primitives_sha256') == os.environ.get('KNEE_PRIMITIVES_SHA256'), 'Runtime method/version/primitives binding')
        require(report['source_bundle_sha256'] == binding['source_bundle_sha256'], 'Runtime source binding')
        environment = {'python_major_minor':list(sys.version_info[:2]), 'torch':torch.__version__, 'timm':importlib.metadata.version('timm'), 'numpy':np.__version__, 'cuda':torch.version.cuda}
        require(torch.cuda.is_available() and environment == binding['environment'], 'Pinned CUDA environment; no fallback')
        report['environment'] = environment
        report['cuda_boards'] = [{'ordinal':i, 'name':torch.cuda.get_device_name(i), 'total_memory_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
        torch.set_num_threads(4); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False; torch.use_deterministic_algorithms(True)
        public, decoder = load_definitions(public_source, decoder_source, manifest)
        columns, ids, records = test_contract(root, public.LAB, visible=mode=='rehearsal')
        if mode=='rehearsal':
            require(columns==binding['columns'] and ids==binding['visible_ids'], 'Visible input order binding')
            require(sha256((Path(root)/'train_series.csv').read_bytes())==binding['train_series_sha256'], 'Frozen train series bytes')
        control = load_model(public, checkpoint, 'cuda', manifest)
        candidate, grid = make_candidate(public, control); before=snapshot(candidate)
        require(grid == binding['qualified_grid'] and before['state'] == binding['qualified_grid']['state_sha256'] and (mode == 'rehearsal' or before == contract['expected_state']), 'Exact constructor/state qualification')
        report.update(state_grid_gate=grid, candidate_state=before, constructor_oracle_gate=True, setup_seconds=time.monotonic()-started)
        del control; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        # Only rehearsal preserves strict checkpoint reload diagnostics.
        state = None
        if mode=='rehearsal':
            state = {k:v.detach().cpu().clone() for k,v in candidate.state_dict().items()}
            torch.save(state, out/'frozen-state.pt')
        def infer_checked(sid, rows, images, expected=None):
            p, counts, hashes, timing = infer_study(public, decoder, candidate, sid, rows, images, audit=mode=='rehearsal')
            counters.update({k:v for k,v in counts.items() if isinstance(v, int)}); timings.append(timing)
            if mode=='rehearsal':
                if expected is not None:
                    require(hashes==expected['hashes'], 'Same-image volume/mask/windows hashes differ')
                    errors['qualified']=max(errors['qualified'], prediction_error(p, expected['probabilities'], 'QUALIFIED_PROBABILITIES'))
                repeat, _, repeated_hashes, _ = infer_study(public, decoder, candidate, sid, rows, images, audit=mode=='rehearsal')
                require(hashes==repeated_hashes, 'Repeat input hashes differ')
                errors['repeat']=max(errors['repeat'], prediction_error(p, repeat, 'REPEAT'))
                candidate.load_state_dict(torch.load(out/'frozen-state.pt', map_location='cpu', weights_only=True), strict=True); candidate.eval()
                reloaded, _, reload_hashes, _ = infer_study(public, decoder, candidate, sid, rows, images, audit=mode=='rehearsal')
                require(hashes==reload_hashes, 'Reload input hashes differ')
                errors['reload']=max(errors['reload'], prediction_error(p, reloaded, 'RELOAD'))
            if mode=='rehearsal': require_immutable(candidate, before)
            return p
        stage='visible' if mode=='rehearsal' else 'entry'
        predictions=[]
        for ordinal,sid in enumerate(ids):
            predictions.append(infer_checked(sid, records, Path(root)/'test_series'))
            report['completed_visible']=ordinal+1; save()
        report['csv_sha256']=write_csv(out/('visible-predictions.csv' if mode=='rehearsal' else 'submission.csv'), columns, ids, predictions)
        if mode=='rehearsal':
            stage='probes'; train_records=read_series(Path(root)/'train_series.csv')
            probes=contract['probes']; require(len(probes)==32 and len({p['id'] for p in probes})==32, 'Frozen32 probes')
            for ordinal,probe in enumerate(probes):
                require(probe['id'] in train_records and probe['id'] not in records, 'Probe image path/population split')
                infer_checked(probe['id'], train_records, Path(root)/'train_series', probe)
                report['completed_probes']=ordinal+1; save()
            report.update(strict_reload_gate=True, same_image_hash_gate=True)
        stage='tails'; tails_started=time.monotonic(); require_immutable(candidate, before)
        report.update(final_state_and_all_buffers_immutable=True, study_timings_seconds=timings,
            estimated_1300_seconds_mean=report['setup_seconds']+float(np.mean([t['total'] for t in timings]))*1300,
            estimated_1300_seconds_p95=report['setup_seconds']+float(np.percentile([t['total'] for t in timings],95))*1300,
            runtime_estimate_policy='informational; actual nine-hour alarm includes setup', tails_seconds=time.monotonic()-tails_started)
        require(time.monotonic()-started<MAX_SECONDS, 'Actual nine-hour platform limit')
        report['status']='PASS'; save()
        if mode=='rehearsal': (out/'frozen-state.pt').unlink()
        return report
    except BaseException as exc:
        if hasattr(exc, 'mirror_input_counts'): counters.update(exc.mirror_input_counts)
        report.update(status='FAIL', failure_type=type(exc).__name__); save()
        raise RuntimeError('Full384 image-only inference failed; inspect preserved receipt') from None
