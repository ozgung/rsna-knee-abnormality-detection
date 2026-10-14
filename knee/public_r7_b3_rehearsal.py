"""B3 rehearsal-only faults: the B0 Raptor all-empty injection on both Raptor-profile arms (Dread and
dr-full) and the unchanged default compact-slot injections (R7). Never in a pure entry."""
import json
import os
from pathlib import Path


def injection_hook(arm,cfg,context):
    from knee.public_s6_recovery_b0_rehearsal import injection_hook as b0_hook
    return b0_hook('old' if arm=='dr' else arm,cfg,context)


if __name__=='__main__':
    import argparse
    from knee import public_r7_b3_entry as entry
    if Path(__file__).resolve()!=(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_r7_b3_rehearsal.py').resolve():
        raise ValueError('EXACT_REHEARSAL_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.worker).read_text())
    if cfg['mode']!='rehearsal':raise ValueError('INJECTIONS_REHEARSAL_ONLY')
    entry.rehearsal_hook=injection_hook
    entry.WORKERS[cfg['arm']](a.worker)
