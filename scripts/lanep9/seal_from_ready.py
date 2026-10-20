"""Seal the P10/P11 member list from the lanes' READY.json exports (parent, 2026-10-20 23:45 UTC).

Usage: python scripts/lanep9/seal_from_ready.py --out-reference <json> [--members v3-s43,student,v4b-s42,v4b-s43,r1-f0,r1-f1,r1-f2,r1-f3]
                                                 [--fallbacks v3-s44,v3-s45,v3-s42] [--max 8]
Reads outputs/lanef-20261020/members/<name>/READY.json and outputs/lanel4-20261020/members/<name>/READY.json, keeps the members
whose export is READY (fallbacks fill the gaps in order), assigns the sealed (seed, slot) aliases below, prefers the private
dataset copy when its status is ready (else the Kaggle Model ref), merges every available probe-36 reference (alias renamed to
the sealed alias) with scripts/lanep9/merge_references.py and marks the rest 'no_reference'. Prints the S6C_MODELS_P10 literal
(loader order) and the reference sha; nothing is edited. No study ids are printed.

Alias plan (loader alias = E_prior_s<seed>_f<slot>.pt; distinct per member, the slots are bookkeeping only):
  student          seed 42 slot 1   (as in P7-prime / P9)      v3-s42  seed 42 slot 0 (P7-prime / P9S2)
  v4b-s42          seed 42 slot 2                              v3-s43  seed 43 slot 0 (P7-prime / P9S2)
  r1-f0 .. r1-f3   seed 42 slots 5..8 (Lane F hash-fold models, seed_base 42; their exports name them f0..f3)
  v4b-s43          seed 43 slot 1                              v3-s44  seed 44 slot 0    v3-s45  seed 45 slot 0
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from knee import public_s6c as S6C

MAIN = PROJECT.parent.parent.parent if PROJECT.name.startswith('lanep9') else PROJECT
ROOTS = [MAIN / 'outputs/lanef-20261020/members', MAIN / 'outputs/lanel4-20261020/members']
PLAN = {'student': (42, 1, 'v3'), 'v3-s42': (42, 0, 'v3'), 'v3-s43': (43, 0, 'v3'), 'v3-s44': (44, 0, 'v3'), 'v3-s45': (45, 0, 'v3'),
        'v4b-s42': (42, 2, 'v4b'), 'v4b-s43': (43, 1, 'v4b'), 'r1-f0': (42, 5, 'v3'), 'r1-f1': (42, 6, 'v3'), 'r1-f2': (42, 7, 'v3'),
        'r1-f3': (42, 8, 'v3')}
# members already sealed earlier (no READY.json folder of their own)
KNOWN = {'student': {'file': 'E_prior_s42_fmall.pt', 'sha256': '6a13f52e424cbfd4945883273fd081dfb2f25b171adb5b6521fab4d382466b57',
                     'bytes': 200186923, 'dataset': 'ozgung/knee-laned-fmall-s42-20261020', 'dataset_version': 1,
                     'reference': MAIN / 'outputs/lanec-20261019/p7/reference-p7.json', 'reference_alias': 'E_prior_s42_f1.pt'},
         'v3-s43': {'file': 'E_prior_s43_v3all.pt', 'sha256': 'a99ad89705f8297ac86659ce8e5a092cd626d851d7d51331c59f1b6920c69690',
                    'bytes': 200188027, 'dataset': 'ozgung/knee-s6-v3all-s43-20261020', 'dataset_version': 1,
                    'reference': MAIN / 'outputs/lanec-20261019/p7/reference-p7.json', 'reference_alias': 'E_prior_s43_f0.pt'},
         'v3-s42': {'file': 'E_prior_s42_v3all.pt', 'sha256': '03da2cacdbc12305519e05c8e9a5626f14eb71f49bc26d56e3f2ced35f997e43',
                    'bytes': 200188027, 'dataset': 'ozgung/knee-s6-lanec-seeds-20261019/pyTorch/v3-s42', 'dataset_version': 1,
                    'reference': MAIN / 'outputs/lanec-20261019/p7/reference-p7.json', 'reference_alias': 'E_prior_s42_f0.pt'}}


def find_ready(name):
    for root in ROOTS:
        p = root / name / 'READY.json'
        if p.is_file():
            return p
    return None


def entry_from_ready(name, path):
    r = json.loads(path.read_text())
    seed, slot, labels = PLAN[name]
    # Lane L4 layout (v3-s44) or Lane F layout (r1-f*)
    if 'checkpoint_sha256' in r:
        file, sha, size = r['checkpoint_filename'], r['checkpoint_sha256'], int(r['checkpoint_size'])
        ds = (r.get('dataset_copy') or {})
        dataset = ds['dataset'] if ds.get('status') == 'ready' else r['model'].rsplit('/', 1)[0]
        version = 1 if ds.get('status') == 'ready' else int(r['model'].rsplit('/', 1)[1])
        ref = (r.get('probe36_reference') or {})
        ref_path, ref_alias = ref.get('path'), ref.get('alias_key')
        labels = r.get('labels', labels)
    else:
        ck = r['checkpoint']
        file, sha, size = ck['file'], ck['sha256'], int(ck['bytes'])
        ds = r.get('dataset') or {}
        if ds.get('status') == 'ready':
            dataset, version = ds['ref'], int(ds.get('version', 1))
        else:
            dataset, version = r['model']['ref'], int(r['model']['version_with_checkpoint'])
        ref = r.get('probe36_reference') or {}
        ref_path, ref_alias = ref.get('path'), ref.get('alias_key') or ck.get('runtime_alias')
        if not ref_path:                                   # a reference file dropped next to READY.json later
            cands = sorted((path.parent).glob('reference*.json'))
            ref_path = str(cands[0]) if cands else None
            if ref_path and not ref_alias:
                ref_alias = next(iter(json.loads(Path(ref_path).read_text())['probabilities']))
    m = {'seed': seed, 'labels': labels, 'file': file, 'sha256': sha, 'bytes': size, 'dataset': dataset, 'dataset_version': version}
    if slot:
        m['slot'] = slot
    if not (ref_path and Path(ref_path).is_file()):
        m['no_reference'] = True
        ref_path = ref_alias = None
    return m, ref_path, ref_alias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', default='v3-s43,student,v4b-s42,v4b-s43,r1-f0,r1-f1,r1-f2,r1-f3')
    ap.add_argument('--fallbacks', default='v3-s44,v3-s45,v3-s42')
    ap.add_argument('--max', type=int, default=8)
    ap.add_argument('--out-reference', required=True)
    a = ap.parse_args()
    chosen, refs, missing = [], [], []
    for name in a.members.split(',') + a.fallbacks.split(','):
        if len(chosen) >= a.max:
            break
        if name in KNOWN:
            k = KNOWN[name]
            seed, slot, labels = PLAN[name]
            m = {'seed': seed, 'labels': labels, 'file': k['file'], 'sha256': k['sha256'], 'bytes': k['bytes'], 'dataset': k['dataset'],
                 'dataset_version': k['dataset_version']}
            if slot:
                m['slot'] = slot
            chosen.append((name, m))
            refs.append((str(k['reference']), k['reference_alias'], S6C.model_alias(m)))
            continue
        p = find_ready(name)
        if p is None:
            missing.append(name)
            continue
        m, ref_path, ref_alias = entry_from_ready(name, p)
        chosen.append((name, m))
        if ref_path:
            refs.append((ref_path, ref_alias, S6C.model_alias(m)))
    chosen.sort(key=lambda x: S6C.model_alias(x[1]))
    models = [m for _, m in chosen]
    S6C.validate_models(models, allow_mixed_labels=True)
    # merge the references: one input file per (path), renames per alias; the P7 file carries three aliases, keep only the chosen ones
    by_file = {}
    for path, alias, sealed in refs:
        by_file.setdefault(path, {})[alias] = sealed
    args = [a.out_reference]
    for path, mapping in by_file.items():
        doc = json.loads(Path(path).read_text())
        keep = {alias: rows for alias, rows in doc['probabilities'].items() if alias in mapping}
        tmp = Path(a.out_reference).with_name(Path(a.out_reference).stem + '-in-%d.json' % len(args))
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({'source': doc.get('source', path), 'probabilities': keep}))
        args.append(str(tmp))
        for alias, sealed in mapping.items():
            if alias != sealed:
                args += ['--rename', '%s=%s' % (alias, sealed)]
    out = subprocess.run([sys.executable, str(PROJECT / 'scripts/lanep9/merge_references.py'), *args], capture_output=True, text=True)
    print(out.stdout.strip() or out.stderr.strip()[-800:])
    print('MISSING (not READY):', missing)
    print('CHOSEN:', [(n, S6C.model_alias(m), m['labels'], 'noref' if m.get('no_reference') else 'ref') for n, m in chosen])
    print('S6C_MODELS_P10 = [')
    for m in models:
        print('    %r,' % (m,))
    print(']')


if __name__ == '__main__':
    main()
