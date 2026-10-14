"""R7 rehearsal-only faults. Never packaged in a pure entry.

Both arms use the unchanged default injections: the original320 all-empty input
(old arm) and the compact-slot no-usable-window, forced-side cleanup, next-healthy
input parity and source-diagnostic parity (R7 renders through the same pinned
native/geometry/anchor/pixel-cache sources as S6, so S6 pilot inputs apply).
"""
import json
import os
from pathlib import Path


if __name__=='__main__':
    import argparse
    from knee import public_r7_recovery_entry as entry
    from knee.public_s6_recovery_rehearsal import injection_hook
    if Path(__file__).resolve()!=(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_r7_recovery_rehearsal.py').resolve():
        raise ValueError('EXACT_REHEARSAL_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.worker).read_text())
    if cfg['mode']!='rehearsal':raise ValueError('INJECTIONS_REHEARSAL_ONLY')
    entry.rehearsal_hook=injection_hook
    (entry.old_worker if cfg['arm']=='old' else entry.r7_worker)(a.worker)
