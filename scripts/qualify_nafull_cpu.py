"""Strict synthetic K62 CPU qualification of the immutable na-full SWA checkpoint.

Reuses the established dr-full qualification procedure with na-full identity and
K62 input. Records actual state/grid/buffer fingerprints; no CUDA or reader claim.
"""
import argparse
import json
from pathlib import Path
import sys
import types
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import qualify_b3_drfull_cpu as Q

CHECKPOINT_SHA256 = 'c5cda6abf3234867d21baf067270b2f45540356802fb556ddc7cf23daa7740c8'
CHECKPOINT_BYTES = 292826627
DATASET = 'ozgung/knee-raptor64-na-full-fold0-final-20261016'
DATASET_VERSION = 1
CHECKPOINT_FILE = 'raptor_ft_fold0_swa.pt'
WINDOWS = 62


def nafull_manifest(source_text):
    from knee.reference import audit_source
    module, decoder, manifest = audit_source(source_text, profile='raptor',
        checkpoint_sha256=CHECKPOINT_SHA256, raptor_windows=WINDOWS)
    manifest.update(checkpoint_file=CHECKPOINT_FILE, weight_dataset=DATASET,
        attribution='Own na-full fixed SWA9-11, nartaa/OAI backbone initialization and Dread teacher; K62 one-view')
    return module, decoder, manifest


def qualify(source_path, checkpoint_path, dread_checkpoint):
    import torch
    import timm
    torch.set_num_threads(2)
    namespace = {**vars(Q), 'DRFULL_CHECKPOINT_SHA256': CHECKPOINT_SHA256,
        'DRFULL_CHECKPOINT_BYTES': CHECKPOINT_BYTES, 'DRFULL_WINDOWS': WINDOWS,
        'drfull_manifest': nafull_manifest}
    run = types.FunctionType(Q.qualify.__code__, namespace, Q.qualify.__name__, Q.qualify.__defaults__)
    receipt = run(source_path, checkpoint_path, dread_checkpoint, forward_windows=WINDOWS)
    if receipt['checkpoint_meta'] != {'epoch': [9, 10, 11], 'swa_over': [9, 10, 11]}:
        raise ValueError('NAFULL_FIXED_SWA_IDENTITY')
    receipt.update(checkpoint_dataset=DATASET, checkpoint_dataset_version=DATASET_VERSION,
        timm=timm.__version__, python=sys.version.split()[0], mirror=False,
        qualifier_sha256=Q.hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_qualifier_sha256=Q.hashlib.sha256(Path(Q.__file__).read_bytes()).hexdigest())
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'checkpoint', 'dread-checkpoint', 'out'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args(); out = Path(a.out)
    if out.exists():
        raise ValueError('FRESH_RECEIPT')
    receipt = qualify(a.source, a.checkpoint, a.dread_checkpoint)
    with out.open('x') as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: receipt[k] for k in ('state_sha256', 'constructor_grid_sha256',
        'windows_shape', 'forward_windows', 'cpu_repeat_max_abs', 'forward_seconds', 'dread_pin_method_check')}))
