"""Check the emitted solo pair: exact replay, fresh ``python -I`` imports from emitted sources only, pair difference.

For each packet the payload is unpacked into a new empty directory and a
separate isolated interpreter imports every entry point from those files
alone, loads the captured controller and installs that arm's binding (no
GPU, image or network use). The pair must share every source byte and the
common config; only cfg['arm'], the payload literals, the kernel id/title and
the V2 dataset attachment may differ.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts import build_s6_solo_pair as B
from knee.s6_solo_runtime import digest, require

PROBE = r'''
import base64, hashlib, importlib, json, os, sys
from pathlib import Path
root = os.path.realpath(sys.argv[1]); sys.path.insert(0, root)
cfg = json.loads(Path(root, 'runtime-config.json').read_text())
names = ['knee', 'knee.s6_solo_pair', 'knee.s6_solo_runtime']
if cfg['mode'] == 'rehearsal':
    names += ['knee.s6_solo_rehearsal', 'knee.s6_solo_a_runner']
mods = {n: importlib.import_module(n) for n in names}
for n, m in mods.items():
    assert os.path.realpath(m.__file__).startswith(root + os.sep), (n, m.__file__)
RT, PAIR = mods['knee.s6_solo_runtime'], mods['knee.s6_solo_pair']
k1 = RT.bind(RT.load_controller(Path(root) / RT.CAPTURED), cfg)
assert os.path.realpath(k1.__file__).startswith(root + os.sep)
assert k1.prepare_inputs.__code__.co_filename == '<s6solo-prepare_inputs>'
entry = ['main', 'prepare_inputs', 'probe_env', 'probe_b_stack', 'run_a', 'a_cmd', 'run_b', 'b_cmd', 'collect_part',
         'run_group', 'blend_and_write', 'blend_with_failures', 'rank01', 'sanity_floor', 'write_submission',
         'remove_submission', 'solo_verify_inputs', 'solo_stage_b_model']
assert all(callable(getattr(k1, n)) for n in entry)
consts = {n: getattr(k1, n) for n in ('MODE', 'W_A', 'W_B', 'B_AMP', 'B_WORKERS', 'GUARD_MAX_S', 'A_TIMEOUT_S', 'B_TIMEOUT_S')}
a_sha = hashlib.sha256(base64.b64decode(k1.A_B64)).hexdigest()
b = cfg['arm']['b_model']
assert PAIR.ALIAS_RE.fullmatch(b['file']) and k1.B_MODEL['file'] == b['file'] and k1.B_MODEL['sha256'] == b['sha256']
out = {'mode': cfg['mode'], 'variant': cfg['arm']['variant'], 'imported': sorted(mods), 'entry_points': entry,
       'controller_constants': consts, 'embedded_a_sha256': a_sha, 'b_model': k1.B_MODEL,
       'derived_prepare_inputs_sha256': k1._RECEIPT['solo']['derived_prepare_inputs_sha256'],
       'binding_sha256': k1._RECEIPT['solo']['binding_sha256'], 'isolated': bool(sys.flags.isolated)}
if cfg['mode'] == 'rehearsal':
    R, A = mods['knee.s6_solo_rehearsal'], mods['knee.s6_solo_a_runner']
    assert all(callable(getattr(R, n)) for n in ('prepare', 'install', 'finish', 'assemble', 'projection', 'verify_result'))
    assert A.A_SOURCE_SHA256 == a_sha and callable(A.run)
    out['fixture'] = {'studies': len(cfg['rehearsal']['fixture']['ids']), 'train': cfg['rehearsal']['fixture']['splits'].count('train')}
print('S6SOLO_PROBE ' + json.dumps(out, sort_keys=True))
'''


def probe(pack):
    with tempfile.TemporaryDirectory(prefix='s6solo-unpacked-') as d:
        root = Path(d)
        require(not any(root.iterdir()), 'FRESH_EMPTY_UNPACK_DIRECTORY')
        for name, text in pack['sources'].items():
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(text.encode())
        (root / 'runtime-config.json').write_text(json.dumps(pack['cfg'], indent=1, sort_keys=True))
        r = subprocess.run([sys.executable, '-I', '-c', PROBE, str(root)], cwd=str(root), capture_output=True, text=True, timeout=600)
        line = [x for x in r.stdout.splitlines() if x.startswith('S6SOLO_PROBE ')]
        require(r.returncode == 0 and line, 'FRESH_ISOLATED_IMPORT: %s' % r.stderr[-2000:])
        return json.loads(line[0][len('S6SOLO_PROBE '):])


def pair_difference(roots):
    packs, codes, metas = {}, {}, {}
    for v, root in roots.items():
        packs[v], codes[v] = B.unpack(Path(root) / 'notebook.ipynb')
        metas[v] = json.loads((Path(root) / 'kernel-metadata.json').read_text())
    a, b = packs['v2'], packs['control']
    require(a['sources'] == b['sources'] and a['source_sha256'] == b['source_sha256'], 'IDENTICAL_SOURCE_BYTES')
    require({k: v for k, v in a['cfg'].items() if k != 'arm'} == {k: v for k, v in b['cfg'].items() if k != 'arm'}, 'IDENTICAL_COMMON_CONFIG')
    require(set(a) == set(b) == {'cfg', 'sources', 'source_sha256'}, 'PAYLOAD_KEYS')
    la, lb = codes['v2'].splitlines(), codes['control'].splitlines()
    diff = [i for i, (x, y) in enumerate(zip(la, lb)) if x != y]
    require(len(la) == len(lb) and all(la[i].split('=', 1)[0] in ('encoded', 'expected_payload') for i in diff), 'BOOTSTRAP_DIFFERS_ONLY_IN_PAYLOAD')
    ma, mb = metas['v2'], metas['control']
    require({k: v for k, v in ma.items() if k not in ('id', 'title', 'dataset_sources')}
            == {k: v for k, v in mb.items() if k not in ('id', 'title', 'dataset_sources')}, 'METADATA_COMMON')
    require(ma['dataset_sources'][:-1] == mb['dataset_sources'] and ma['dataset_sources'][-1] == 'ozgung/knee-s6v2-v2all-20261016/1',
            'METADATA_DIFFERS_ONLY_BY_V2_WEIGHTS')
    arm_keys = sorted(k for k in a['cfg']['arm'] if a['cfg']['arm'][k] != b['cfg']['arm'].get(k))
    return {'identical_sources': len(a['sources']), 'common_config_sha256': digest({k: v for k, v in a['cfg'].items() if k != 'arm'}),
            'source_bundle_sha256': digest(a['sources']), 'bootstrap_differing_lines': [la[i].split('=', 1)[0] for i in diff],
            'arm_differing_fields': arm_keys,
            'b_models': {'v2': a['cfg']['arm']['b_model'], 'control': b['cfg']['arm']['b_model']},
            'metadata': {'v2': {k: ma[k] for k in ('id', 'title', 'dataset_sources')},
                         'control': {k: mb[k] for k in ('id', 'title', 'dataset_sources')}}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out-root', default=str(B.OUT / 'kaggle'))
    p.add_argument('--report', default=str(B.OUT / 'analysis/s6solo-pair-20261016/pair-check.json'))
    a = p.parse_args(argv)
    roots = {v: Path(a.out_root) / B.SLUGS[v] for v in ('v2', 'control')}
    report = {'status': 'PASS', 'packets': {}}
    for v, root in roots.items():
        r = B.verify_build(root, v)
        pack, _ = B.unpack(root / 'notebook.ipynb')
        report['packets'][v] = {'root': str(root), 'payload_sha256': r['payload_sha256'], 'files': r['files'], 'probe': probe(pack)}
    require(report['packets']['v2']['probe']['derived_prepare_inputs_sha256']
            == report['packets']['control']['probe']['derived_prepare_inputs_sha256'], 'IDENTICAL_DERIVED_CONTROLLER')
    report['pair'] = pair_difference(roots)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'status': report['status'], 'report': a.report, 'payloads': {v: x['payload_sha256'] for v, x in report['packets'].items()},
                      'arm_differing_fields': report['pair']['arm_differing_fields'],
                      'bootstrap_differing_lines': report['pair']['bootstrap_differing_lines']}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
