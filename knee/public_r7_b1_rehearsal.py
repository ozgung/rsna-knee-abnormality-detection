"""B1 rehearsal-only faults: the unchanged default compact-slot injections. Never in a pure entry."""
import json
import os
from pathlib import Path


if __name__=='__main__':
    import argparse
    from knee import public_r7_b1_entry as entry
    from knee.public_s6_recovery_rehearsal import injection_hook
    if Path(__file__).resolve()!=(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_r7_b1_rehearsal.py').resolve():
        raise ValueError('EXACT_REHEARSAL_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.worker).read_text())
    if cfg['mode']!='rehearsal' or cfg['arm']!='s6':raise ValueError('INJECTIONS_REHEARSAL_R7_ONLY')
    entry.rehearsal_hook=injection_hook
    entry.r7_worker(a.worker)
