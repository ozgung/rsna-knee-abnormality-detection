#!/usr/bin/env python3
"""Check a private prospective packet; exit 2 while qualification is blocked."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knee.combined_rehearsal import EvidenceError, check_packet


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--verify-base", action="store_true",
                        help="lazily run existing immutable R7 verifier/projection (ML environment required)")
    args = parser.parse_args(argv)
    try:
        result = check_packet(args.packet, verify_base=args.verify_base)
    except EvidenceError as exc:
        result = {"status": "REJECTED", "entry_qualified": False, "error": str(exc)}
        code = 1
    except Exception:
        # Malformed structures and backend diagnostics may contain private IDs.
        result = {"status": "REJECTED", "entry_qualified": False, "error": "INVALID_EVIDENCE_STRUCTURE"}
        code = 1
    else:
        code = 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
