"""Exact Kaggle attachment paths and bounded setup; never walk input trees."""
from contextlib import contextmanager
import json
from pathlib import Path
import re
import signal
import time


def progress(event, **fields):
    print(json.dumps({'event': event, **fields}), flush=True)


def reference_parts(reference, *, owner=None):
    parts = reference.split('/')
    if len(parts) != 2 or any(not re.fullmatch(r'[A-Za-z0-9_-]+', part) for part in parts):
        raise ValueError('Expected an exact owner/slug attachment reference')
    if owner is not None and parts[0] != owner:
        raise ValueError('Private notebook attachment must belong to the configured owner')
    return parts


def exact_root(candidates, required):
    matches = [root for root in candidates if all((root / name).is_file() for name in required)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one explicit attachment mount, found {len(matches)}')
    return matches[0]


def dataset_root(inputs, reference, required):
    owner, slug = reference_parts(reference)
    inputs = Path(inputs)
    return exact_root([inputs / 'datasets' / owner / slug, inputs / slug], required)


def notebook_root(inputs, reference, directory, required):
    owner, slug = reference_parts(reference, owner='ozgung')
    inputs = Path(inputs)
    return exact_root([inputs / 'notebooks' / owner / slug / directory,
                       inputs / slug / directory], required)


def cache_root(inputs, reference, expected_fingerprint, *, complete=False):
    from knee.fresh_reader import validate_plan
    required = ['plan-private.json', 'cache-index-private.json',
                'cache-complete.json' if complete else 'cache-pilot.json']
    root = notebook_root(inputs, reference, 'fresh_reader_cache', required)
    plan = json.loads((root / 'plan-private.json').read_text())
    if validate_plan(plan) != expected_fingerprint:
        raise ValueError('Exact cache attachment has a different plan fingerprint')
    return root


def resume_checkpoint(inputs, reference, expected_fingerprint):
    # Notebook resume requires the complete own checkpoint pair. The training
    # runner subsequently verifies weights, best metric, code and RNG contract.
    root = notebook_root(inputs, reference, 'fresh_reader_training',
                         ['last.pt', 'best.pt', 'run-contract.json'])
    contract = json.loads((root / 'run-contract.json').read_text())
    if contract.get('plan_fingerprint') != expected_fingerprint:
        raise ValueError('Own resume checkpoint has a different plan fingerprint')
    return root / 'last.pt'


@contextmanager
def time_limit(seconds, label):
    """Main-thread POSIX deadline, including blocking setup/network calls.

    Kaggle Linux supplies SIGALRM. Preserve any enclosing deadline; nesting must
    never extend the outer stage budget. This does not change the data plan.
    """
    if seconds <= 0:
        raise TimeoutError(f'{label} budget exhausted')
    if not hasattr(signal, 'setitimer'):
        raise RuntimeError('Bounded Kaggle setup requires POSIX interval timers')
    started = time.monotonic()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    duration = min(seconds, previous_timer[0]) if previous_timer[0] > 0 else seconds
    def timeout(signum, frame):
        raise TimeoutError(f'{label} deadline exceeded')
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, duration)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            remaining = previous_timer[0] - (time.monotonic() - started)
            if remaining <= 0:
                # Raising synchronously lets enclosing contexts unwind before
                # SIG_DFL is restored. Never schedule a new expired alarm: it
                # can arrive after the outer context restores SIG_DFL and kill
                # the notebook instead of producing a recorded TimeoutError.
                raise TimeoutError(f'Enclosing deadline expired during {label}')
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])
