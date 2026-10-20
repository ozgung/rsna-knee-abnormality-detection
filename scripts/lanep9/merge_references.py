"""Merge A100 fp16 probe-36 member references into one s6c reference JSON ({'source', 'probabilities': {alias: {study: [12]}}}).

Usage: python scripts/lanep9/merge_references.py <out json> <ref json> [<ref json> ...] [--rename OLD=NEW ...]
Every input is a reference JSON with 'probabilities' keyed by loader alias (knee.public_s6c.block schema; extra keys such as
'receipt' are dropped). Every alias must cover the same probe-study set (the frozen 36). --rename maps an export's alias to the
alias the sealed member list uses (Lane F fold models share seed 42 with other members and get slots 5..8).
Prints the output sha256 and the alias -> study-count table. No study ids are printed.
"""
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from knee import public_s6c as S6C


def main(argv):
    args = [a for a in argv if not a.startswith('--rename')]
    renames = {}
    for i, a in enumerate(argv):
        if a == '--rename':
            old, new = argv[i + 1].split('=')
            renames[old] = new
    args = [a for a in args if '=' not in a or not a.startswith(tuple(renames))]
    out, inputs = args[0], args[1:]
    merged, sources, studies = {}, [], None
    for path in inputs:
        ref = json.loads(Path(path).read_text())
        S6C.require(isinstance(ref.get('probabilities'), dict) and ref['probabilities'], 'REFERENCE_SCHEMA_' + path)
        sources.append('%s (%s)' % (ref.get('source', '?'), hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]))
        for alias, rows in ref['probabilities'].items():
            alias = renames.get(alias, alias)
            S6C.require(alias not in merged, 'DUPLICATE_ALIAS_' + alias)
            S6C.require(all(len(p) == 12 and all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in p) for p in rows.values()), 'REFERENCE_VALUES_' + alias)
            if studies is None:
                studies = set(rows)
            S6C.require(set(rows) == studies, 'PROBE_SET_DIFFERS_' + alias)
            merged[alias] = {k: [float(x) for x in rows[k]] for k in sorted(rows)}
    doc = {'source': ' | '.join(sources), 'probabilities': dict(sorted(merged.items()))}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=0, sort_keys=True) + '\n')
    sha = hashlib.sha256(Path(out).read_bytes()).hexdigest()
    print(json.dumps({'out': out, 'sha256': sha, 'aliases': {a: len(v) for a, v in doc['probabilities'].items()}, 'studies': len(studies)}))


if __name__ == '__main__':
    main(sys.argv[1:])
