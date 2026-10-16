"""A‴ image-only runtime entry assembled from the archived WS2 path."""
from knee.public_nafull_aprime_runtime import run, run_worker


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    import os
    from knee.public_s6_qualification import require
    require(Path(__file__).resolve() == (Path(os.environ['KNEE_ENTRY_PROJECT']) /
            'knee/public_nafull_aprime_entry.py').resolve(), 'EXACT_FRESH_NAFULL_SOURCE_IMPORT')
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    run_worker(args.worker, rehearsal=False)
