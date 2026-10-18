"""B0′ rehearsal-only wrapper; unchanged B0 typed recovery injections."""
import json
import os
from pathlib import Path
from knee.public_s6_recovery_b0_rehearsal import injection_hook


if __name__=='__main__':
    import argparse
    from knee import public_s6_recovery_b0prime as entry
    if Path(__file__).resolve()!=(Path(os.environ['KNEE_ENTRY_PROJECT'])/'knee/public_s6_recovery_b0prime_rehearsal.py').resolve():
        raise ValueError('EXACT_REHEARSAL_SOURCE_IMPORT')
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.worker).read_bytes())
    if cfg['mode']!='rehearsal':raise ValueError('INJECTIONS_REHEARSAL_ONLY')
    entry.rehearsal_hook=injection_hook
    (entry.old_worker if cfg['arm']=='old' else entry.s6_worker)(a.worker)
