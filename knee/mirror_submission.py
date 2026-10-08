"""Image-only native320/K94 mirror entry; no training or reference recipe changes."""
from collections import Counter
import csv
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
import warnings

from knee import mirror_input
from knee.reference import canonical_bytes, infer_checked, load_definitions, load_model, sha256, validate_predictions
from knee.reference_blend import read_test_contract, write_probabilities

VERSION = 'mirror-standalone-v1'
PARITY_ATOL = 1e-6
CROSS_RUN_ATOL = 1e-4
MAX_SECONDS = 9 * 3600


def read_inputs(root, *, visible):
    """Read only official test metadata/header; sample rows are not test coverage."""
    columns, ids, records = read_test_contract(root, rehearsal=visible)
    expected = ['StudyInstanceUID', 'SeriesInstanceUID', 'Fluid_Sensitive', 'Fat_Suppression', 'Anatomical_Plane']
    with (Path(root) / 'test_series.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected:
            raise ValueError('Official mirror series schema differs')
        rows = list(reader)
    if len({r['SeriesInstanceUID'] for r in rows}) != len(rows):
        raise ValueError('Mirror series identifiers must be unique')
    for row in rows:
        if (set(row) != set(expected) or any(row[k] is None for k in expected) or
                row['Fluid_Sensitive'] not in ('0', '1') or row['Fat_Suppression'] not in ('0', '1') or
                row['Anatomical_Plane'] not in ('Sagittal', 'Coronal', 'Axial')):
            raise ValueError('Malformed official mirror series metadata')
    return columns, ids, records


def read_predictions(path):
    with Path(path).open(newline='') as handle:
        reader = csv.reader(handle)
        columns = next(reader)
        rows = list(reader)
    if (len(columns) != 13 or columns[0] != 'StudyInstanceUID' or len(set(columns)) != 13 or
            not rows or any(len(row) != 13 for row in rows)):
        raise ValueError('Mirror CSV schema/rows differ')
    ids = [row[0] for row in rows]
    if len(set(ids)) != len(ids) or any(not x or '/' in x or '\\' in x or x in ('.', '..') for x in ids):
        raise ValueError('Mirror CSV identifiers invalid')
    values = [[float(x) for x in row[1:]] for row in rows]
    validate_predictions(values[0] if len(rows) == 1 else values, rows=len(rows))
    return columns, ids, values


def compare_csv(reference, actual):
    a, b = read_predictions(reference), read_predictions(actual)
    if a[:2] != b[:2] or len(a[1]) != 3:
        raise ValueError('Visible mirror target/ID order differs from rehearsal')
    error = max(abs(x-y) for ar, br in zip(a[2], b[2]) for x, y in zip(ar, br))
    if error > CROSS_RUN_ATOL:
        raise ValueError('Visible mirror cross-run probabilities differ beyond 1e-4')
    # Process-level CUDA benchmarking can change floating-point reduction order.
    # Keep within-process repeat checks tighter and reject even small rank/tie changes.
    for j in range(12):
        for i in range(3):
            for k in range(i):
                relation = lambda v: (v[i][j] > v[k][j]) - (v[i][j] < v[k][j])
                if relation(a[2]) != relation(b[2]):
                    raise ValueError('Visible mirror cross-run ranking or tie changed')
    return error


def require_visible(report, qualification, csv_path):
    """Bind visible production evidence and its exact ordered output before full build."""
    if (report.get('status') != 'PASS' or report.get('visible') is not True or
            report.get('qualification_sha256') != qualification or report.get('version') != VERSION or
            report.get('test_studies') != 3 or report.get('prediction_fallbacks') != 0 or
            report.get('input_policy') != mirror_input.policy_contract() or
            report.get('production_input_counts', {}).get('resource_errors') != 0 or
            report.get('completed_studies') != 3 or
            any(not isinstance(report.get(k), (int,float)) or not math.isfinite(report[k]) or report[k] <= 0
                for k in ('peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','peak_host_rss_bytes','disk_free_bytes','output_disk_bytes')) or
            not 0 <= report.get('repeat_prediction_max_error', float('inf')) <= PARITY_ATOL or
            not math.isfinite(report.get('elapsed_seconds', float('inf'))) or
            not 0 < report['elapsed_seconds'] <= MAX_SECONDS or
            report.get('csv_sha256') != sha256(Path(csv_path).read_bytes())):
        raise ValueError('Successful exact standalone visible qualification required')
    columns, ids, _ = read_predictions(csv_path)
    if (len(ids) != 3 or report.get('ordered_ids_sha256') != sha256(canonical_bytes(ids)) or
            report.get('target_order_sha256') != sha256(canonical_bytes(columns[1:]))):
        raise ValueError('Visible receipt and CSV order differ')


def run(root, checkpoint, public_source, decoder_source, contract, out, *, started):
    """Same protected decoder, GPU windows and two-view function as the67 rehearsal."""
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Mirror inference requires CUDA; no device fallback')
    q = contract['qualification']
    if (contract.get('version') != VERSION or contract.get('max_seconds') != MAX_SECONDS or
            q['input_policy'] != mirror_input.policy_contract() or
            sha256(canonical_bytes(q)) != contract['qualification_sha256']):
        raise ValueError('Standalone mirror contract differs')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    report = {'status':'running', 'version':VERSION, 'visible':contract['visible'],
        'qualification_sha256':contract['qualification_sha256'], 'contract':contract,
        'input_policy':mirror_input.policy_contract(), 'prediction_fallbacks':0,
        'production_input_counts':{}, 'max_seconds':MAX_SECONDS}
    counts, times, predictions = Counter(), [], []
    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        (out/'mirror-audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    def deadline():
        if time.monotonic() - started >= MAX_SECONDS:
            raise TimeoutError('Actual nine-hour mirror runtime exhausted')
    save()
    try:
        deadline()
        warnings.filterwarnings('ignore', module=r'pydicom(?:\..*)?')
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        public, decoder = load_definitions(public_source, decoder_source, q['manifest'])
        columns, ids, records = read_inputs(root, visible=contract['visible'])
        if columns != ['StudyInstanceUID', *public.LAB]:
            raise ValueError('Official target order differs from pinned model')
        model = load_model(public, checkpoint, 'cuda', q['manifest'])
        torch.cuda.reset_peak_memory_stats()
        report.update(test_studies=len(ids), ordered_ids_sha256=sha256(canonical_bytes(ids)),
            target_order_sha256=sha256(canonical_bytes(public.LAB)),
            checkpoint_verified_sha256=q['manifest']['checkpoint_sha256'],
            checkpoint_bytes=Path(checkpoint).stat().st_size,
            environment={'python':sys.version, 'python_major_minor':list(sys.version_info[:2]),
                'packages':{p:importlib.metadata.version(p) for p in ('torch','timm','numpy','pydicom')},
                'gpu':torch.cuda.get_device_name(0), 'cuda':torch.version.cuda},
            setup_seconds=time.monotonic()-started, repeat_prediction_max_error=None)
        repeat_error = 0.0
        for ordinal, sid in enumerate(ids):
            deadline()
            before = time.monotonic()
            volume, mask, event = mirror_input.prepare_study(public, decoder, sid, records, Path(root)/'test_series')
            counts.update(event)
            windows = [public.gpu_windows(volume, mask, public.K_EVAL, 320, 'cuda', mirror=m) for m in (False, True)]
            if any(tuple(w.shape) != (94, 3, 320, 320) for w in windows):
                raise ValueError('Native320/K94 runtime window shape differs')
            p = validate_predictions(infer_checked(public, model, windows, 'cuda', mirror=True))
            torch.cuda.synchronize()
            times.append(time.monotonic()-before)
            if contract['visible']:
                repeated = validate_predictions(infer_checked(public, model, windows, 'cuda', mirror=True))
                repeat_error = max(repeat_error, float(np.max(np.abs(p-repeated))))
                if repeat_error > PARITY_ATOL:
                    raise ValueError('Visible mirror within-run repetition differs')
            predictions.append(p)
            del volume, mask, windows
            report.update(completed_studies=ordinal+1, production_input_counts=dict(counts))
            if (ordinal+1) % 8 == 0 or ordinal+1 == len(ids):
                save()
                print(json.dumps({'event':'mirror_studies','completed':ordinal+1,'total':len(ids),
                                  'elapsed_seconds':report['elapsed_seconds']}), flush=True)
        values = np.stack(predictions)
        validate_predictions(values[0] if len(ids)==1 else values, rows=len(ids))
        deadline()
        name = 'visible-predictions.csv' if contract['visible'] else 'submission.csv'
        report.update(csv_sha256=write_probabilities(out/name, columns, ids, values),
            repeat_prediction_max_error=repeat_error if contract['visible'] else None,
            estimated_1300_seconds_mean=report['setup_seconds']+float(np.mean(times))*1300,
            estimated_1300_seconds_p95=report['setup_seconds']+float(np.percentile(times,95))*1300,
            runtime_estimate_policy='informational_only; actual competition9h runtime enforced',
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            disk_free_bytes=shutil.disk_usage(out).free,
            output_disk_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()), status='PASS')
        save()
        return report
    except Exception as exc:
        if hasattr(exc, 'mirror_input_counts'):
            counts.update(exc.mirror_input_counts)
        report.update(status='FAIL', failure_type=type(exc).__name__, production_input_counts=dict(counts))
        save()
        # No model/prediction retry. Only sanitized aggregate errors reach notebook logs.
        raise RuntimeError('Mirror entry failed; inspect aggregate audit') from None
