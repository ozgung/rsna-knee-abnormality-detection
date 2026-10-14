"""B2 rehearsal-only faults: the B0 Raptor all-empty injection (old arm) and the unchanged default
compact-slot injections (R7). Never in a pure entry."""
import json
import os
from pathlib import Path


if __name__=='__main__':
    import argparse
    from knee import public_r7_b2_entry as entry
    from knee.public_s6_recovery_b0_rehearsal import injection_hook
    if Path(__file__).resolve()!=(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_r7_b2_rehearsal.py').resolve():
        raise ValueError('EXACT_REHEARSAL_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.worker).read_text())
    if cfg['mode']!='rehearsal':raise ValueError('INJECTIONS_REHEARSAL_ONLY')
    entry.rehearsal_hook=injection_hook
    (entry.old_worker if cfg['arm']=='old' else entry.r7_worker)(a.worker)
