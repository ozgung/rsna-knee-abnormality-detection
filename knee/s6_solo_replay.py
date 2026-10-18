"""Host-only acceptance of sealed solo rehearsal results across NumPy versions.

The sealed producer remains unchanged. Only the derived cross-board probability
difference diagnostic admits double-precision roundoff; every scientific gate is
recomputed and must pass, and all other receipt fields and the ledger are exact.
"""
import json
import math
from pathlib import Path

from knee import s6_solo_rehearsal as producer
from knee.s6_solo_runtime import binding_digest, file_sha, require

PROBABILITY_DIAGNOSTIC_ATOL = 1e-12


def verify_result(result_dir, cfg, k1):
    """Replay original gates without modifying collected bytes or sealed sources."""
    root = Path(result_dir)
    saved = root / 's6solo'
    receipt = json.loads((saved / 'receipt.json').read_text())
    require(receipt['status'] == 'PASS' and receipt['mode'] == 'rehearsal'
            and receipt['variant'] == cfg['arm']['variant']
            and receipt['binding_sha256'] == binding_digest(cfg), 'PASSING_BOUND_REHEARSAL')
    for field, name in (('ledger_sha256', 'ledger.json'), ('resources_sha256', 'resources.json'),
                        ('final_csv_sha256', 'rehearsal-final.csv')):
        require(file_sha(saved / name) == receipt[field], 'SAVED_FILE_BINDINGS')
    resources = json.loads((saved / 'resources.json').read_text())
    again, ledger = producer.assemble(k1, cfg, root / 'k1', saved / 'rehearsal-final.csv',
                                      receipt['timing'], receipt['sentinel']['run'], resources)
    require(again['status'] == 'PASS' and all(again['checks'].values()),
            'PASSING_RECOMPUTED_REHEARSAL')
    for field in ('ledger_sha256', 'resources_sha256', 'final_csv_sha256'):
        again[field] = receipt[field]
    differences = []
    old, new = receipt['sentinel']['comparisons'], again['sentinel']['comparisons']
    require(len(old) == len(new), 'INDEPENDENT_RECEIPT_LEDGER_REPLAY')
    for index, (a, b) in enumerate(zip(old, new)):
        x, y = a['max_abs_probability'], b['max_abs_probability']
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in (x, y)) and abs(x - y) <= PROBABILITY_DIAGNOSTIC_ATOL,
                'SENTINEL_DIAGNOSTIC_NUMERIC_REPLAY')
        if x != y:
            differences.append({'comparison': index, 'saved': x, 'recomputed': y, 'abs_difference': abs(x - y)})
        b['max_abs_probability'] = x  # comparison copy only; original receipt/files stay immutable
    canonical = lambda value: json.loads(json.dumps(value, sort_keys=True, allow_nan=False, default=str))
    require(canonical(again) == canonical(receipt)
            and ledger == json.loads((saved / 'ledger.json').read_text()),
            'INDEPENDENT_RECEIPT_LEDGER_REPLAY')
    return {'receipt_sha256': file_sha(saved / 'receipt.json'), 'ledger_sha256': receipt['ledger_sha256'],
            'final_csv_sha256': receipt['final_csv_sha256'], 'payload_sha256': receipt['payload_sha256'],
            'conservative_seconds': receipt['projection']['conservative_seconds'],
            'elapsed_seconds': receipt['elapsed_seconds'], 'checks': sorted(receipt['checks']),
            'host_replay': {'policy': 'derived-sentinel-probability-roundoff-v1',
                            'source_sha256': file_sha(__file__),
                            'diagnostic_absolute_tolerance': PROBABILITY_DIAGNOSTIC_ATOL,
                            'differences': differences}}
