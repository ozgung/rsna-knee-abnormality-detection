"""Rehearsal only: run component A's exact embedded bytes with the 39-study fixture as its test root.

Component A hard-codes /kaggle/input/{competitions/,}rsna-knee-abnormality-detection
as its test root, so the frozen fixture cannot be presented to it by paths alone.
This runner executes the unchanged, hash-checked A source as a module (its
``__main__`` guard does not fire), replaces only ``find_test_root`` and then
calls A's own ``main``. Weights discovery, preprocessing (fork pool), TTA,
fallbacks and the /kaggle/working/submission.csv write stay A's code. The entry
never uses this file: the captured controller runs A as ``python -u run.py``.
"""
import hashlib
import sys
import types
from pathlib import Path

A_SOURCE_SHA256 = 'bcd864cbf349882e876959b49cb348cbb89b4c62fc204d4e4261049c5e3a79c5'
MODULE = 'k1_component_a'


def run(run_py, test_root, expected_sha256=A_SOURCE_SHA256, module=MODULE):
    run_py, test_root = Path(run_py), str(Path(test_root))
    raw = run_py.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise SystemExit('S6SOLO_A_RUNNER: component A source differs from %s' % expected_sha256)
    if not (Path(test_root) / 'test.csv').is_file() or not (Path(test_root) / 'test_series.csv').is_file():
        raise SystemExit('S6SOLO_A_RUNNER: fixture test root lacks test.csv/test_series.csv')
    m = types.ModuleType(module)
    m.__file__ = str(run_py)
    sys.modules[module] = m          # fork-pool workers resolve pickled A functions through this entry
    exec(compile(raw, str(run_py), 'exec'), m.__dict__)
    if not callable(getattr(m, 'find_test_root', None)) or not callable(getattr(m, 'main', None)):
        raise SystemExit('S6SOLO_A_RUNNER: component A lacks find_test_root/main')
    m.find_test_root = lambda: test_root
    print('S6SOLO_A_RUNNER exact A %s... with the rehearsal fixture as its test root' % expected_sha256[:12], flush=True)
    return m.main()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('usage: s6_solo_a_runner.py RUN_PY FIXTURE_TEST_ROOT')
    run(sys.argv[1], sys.argv[2])
