"""Produce a sealed s6c member entry from a Kaggle dataset export (SHA256SUMS + file listing) or an explicit sha/bytes.

Usage:
  python scripts/lanep9/member_entry.py --dataset ozgung/knee-s6-v3all-s44-20261020 --version 1 --file E_prior_s44_v3all.pt \
      --seed 44 --labels v3 [--slot 0] [--no-reference] [--work outputs/lanep9-20261020/tmp]
  python scripts/lanep9/member_entry.py --dataset ozgung/<model>/pyTorch/<variation> --version 1 --file X.pt --seed 46 --labels v4b \
      --sha256 <64 hex> --bytes <n>            (Kaggle Models: pass sha/bytes from the parent's export receipt)

Prints one JSON object per call (paste into S6C_MODELS_P9S2 / S6C_MODELS_P10 in scripts/build_r6flip_entry.py, loader order =
sorted aliases E_prior_s<seed>_f<slot>.pt). Only the small SHA256SUMS file is downloaded; the checkpoint is not.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from knee import public_s6c as S6C

KAGGLE = '~/miniconda3/bin/kaggle'


def dataset_sha_and_bytes(dataset, version, file, work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for p in work.glob('SHA256SUMS*'):
        p.unlink()
    subprocess.run([KAGGLE, 'datasets', 'download', '-f', 'SHA256SUMS', '-p', str(work), '--force', dataset], check=True,
                   capture_output=True, text=True)
    sums = work / 'SHA256SUMS'
    if not sums.exists() and (work / 'SHA256SUMS.zip').exists():
        with zipfile.ZipFile(work / 'SHA256SUMS.zip') as z:
            z.extract('SHA256SUMS', work)
    sha = None
    for line in sums.read_text().splitlines():
        m = re.match(r'^([0-9a-f]{64})\s+\*?(.+)$', line.strip())
        if m and m.group(2).strip() == file:
            sha = m.group(1)
    if sha is None:
        raise SystemExit('file %s not in SHA256SUMS of %s' % (file, dataset))
    listing = subprocess.run([KAGGLE, 'datasets', 'files', dataset], check=True, capture_output=True, text=True).stdout
    size = None
    for line in listing.splitlines():
        parts = line.split()
        if parts and parts[0] == file:
            size = int(parts[1])
    if size is None:
        raise SystemExit('file %s not in the listing of %s' % (file, dataset))
    return sha, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='owner/slug (dataset) or owner/model/framework/variation (Kaggle Model)')
    ap.add_argument('--version', type=int, required=True)
    ap.add_argument('--file', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--slot', type=int, default=0)
    ap.add_argument('--no-reference', action='store_true', help='no A100 probe-36 read-out exists for this member (recorded in the sealed list)')
    ap.add_argument('--sha256')
    ap.add_argument('--bytes', type=int)
    ap.add_argument('--work', default=str(PROJECT.parent.parent.parent / 'outputs/lanep9-20261020/tmp'))
    a = ap.parse_args()
    if a.sha256 and a.bytes:
        sha, size = a.sha256, a.bytes
    elif S6C.is_model_ref(a.dataset):
        raise SystemExit('Kaggle Model refs need --sha256 and --bytes (the bundle is not downloaded here)')
    else:
        sha, size = dataset_sha_and_bytes(a.dataset, a.version, a.file, a.work)
    m = {'seed': a.seed, 'labels': a.labels, 'file': a.file, 'sha256': sha, 'bytes': size, 'dataset': a.dataset, 'dataset_version': a.version}
    if a.slot:
        m['slot'] = a.slot
    if a.no_reference:
        m['no_reference'] = True
    S6C.validate_models([m, dict(m, seed=9999, sha256='0' * 64)], allow_mixed_labels=True)   # schema check of the entry itself
    print(json.dumps(m))


if __name__ == '__main__':
    main()
