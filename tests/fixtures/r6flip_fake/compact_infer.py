"""Synthetic stand-in for compact_infer (tests only): DataLoader with two spawn workers, flip passes, stackers, CSV.
Without --flip-tta (P6 plain arms) it scores the plain pass only with its own arithmetic and writes no flip_tta receipt."""
import argparse, csv, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch


class Studies(torch.utils.data.Dataset):
    def __init__(self, uids, size):
        self.uids, self.size, self.native = list(uids), size, None

    def __len__(self):
        return len(self.uids)

    def _setup(self):
        torch.manual_seed(11)
        self.side_net = torch.nn.Linear(3, 1)
        self.bank = type('Bank', (), {'model': torch.nn.Linear(2, 2)})()
        self.window_rows = lambda *a, **k: [1]
        self.native = True

    def __getitem__(self, i):
        uid = self.uids[i]
        try:
            if self.native is None:
                self._setup()
            if not self.window_rows():
                raise ValueError('no usable window')
            g = torch.Generator().manual_seed(int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16))
            img = torch.randint(0, 256, (4, 3, self.size, self.size), generator=g, dtype=torch.uint8)
            meta = torch.zeros((4, 9))
            meta[:, 7] = torch.tensor([0., 1., 2., 1.])
            meta[:, 1] = torch.tensor([0., 2., 3., 1.])
            meta[:, 2] = torch.tensor([-.5, .25, .5, 1.])
            return {'uid': uid, 'error': None, 'images': img, 'meta': meta, 'seconds': [.1, .1, .1],
                    'fluid': {'status': 'ok', 'eff_sag_p99_open': 4.0, 'bak_sag_p99_open': 1.0}, 'anchor': None, 'side': None}
        except Exception as exc:
            return {'uid': uid, 'error': repr(exc), 'images': torch.zeros((0, 3, self.size, self.size), dtype=torch.uint8),
                    'meta': torch.zeros((0, 9)), 'seconds': [0., 0., 0.]}


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from knee import public_r6flip as F
    from knee import public_r7_stackers as stackers
    import compact_train as CT
    ap = argparse.ArgumentParser()
    for k in ('--assignment', '--output', '--models', '--arm', '--state-spec', '--fluid-stacker', '--state-stacker'):
        ap.add_argument(k)
    ap.add_argument('--size', type=int)
    ap.add_argument('--crop', type=int)
    ap.add_argument('--workers', type=int)
    ap.add_argument('--flip-tta', action='store_true')
    a, _ = ap.parse_known_args()
    out = Path(a.output); out.mkdir(parents=True)
    uids = json.loads(Path(a.assignment).read_text())
    spec = json.loads(Path(a.state_spec).read_bytes())
    models = []
    for p in sorted(Path(a.models).glob(a.arm + '_s*_f*.pt')):
        sd = torch.load(p, map_location='cpu', weights_only=True)
        m = CT.Net('convnext_small', None, freeze_blocks=16, crop=a.crop, state_spec=spec, in_chans=3, spatial=True, extent=True,
                   syn_decomp=True)
        m.load_state_dict(sd, strict=True)
        models.append(m.eval().requires_grad_(False))
    perm, bin_perm = F.mirror_perm(CT.LABELS, spec, models[0].bin_names)
    fluid = stackers.load_fluid(Path(a.fluid_stacker).read_bytes(), CT.LABELS)
    layout = stackers.state_layout(spec, CT.LABELS)
    state = stackers.load_state(Path(a.state_stacker).read_bytes(), CT.LABELS, layout, hashlib.md5(Path(a.state_spec).read_bytes()).hexdigest())
    loader = torch.utils.data.DataLoader(Studies(uids, a.size), batch_size=None, shuffle=False, num_workers=a.workers, prefetch_factor=1,
                                         persistent_workers=False, multiprocessing_context='spawn', timeout=900.)
    off = (a.size - a.crop) // 2
    fallback, rows = {}, []
    with torch.inference_mode():
        for item in loader:
            if item['error']:
                fallback[item['uid']] = item['error']
                continue
            img = item['images'][None, :, :, off:off + a.crop, off:off + a.crop]
            meta = item['meta'][None]
            mask = torch.ones((1, img.shape[1]), dtype=torch.bool)
            models[0](img, meta, mask, chunk=192)
            if not a.flip_tta:          # compact_infer v4c score_v4c with FLIP None: per-model plain z, fold-mean state read-out
                zs, states = [], []
                for m in models:
                    z, sl, bl = m(img, meta, mask, chunk=192, return_aux=True)
                    zs.append(z[0])
                    p_ = sl.float().softmax(-1)
                    states.append(torch.cat([p_[:, c, :k] for c, k in enumerate(m.state_k)] + [torch.sigmoid(bl.float())], 1)[0])
                zs = torch.stack(zs)
                p, _ = stackers.apply(torch.sigmoid(zs.mean(0)).double().numpy(), zs.mean(0).double().numpy(),
                                      torch.stack(states).double().mean(0).numpy(), item['fluid'], fluid, state, CT.LABELS)
                rows.append([item['uid']] + ['%.9g' % float(x) for x in p])
                continue
            img_m, meta_m = F.mirror_inputs(img, meta)
            plain, mirror, pa, ma = [], [], [], []
            for m in models:
                z, sl, bl = m(img, meta, mask, chunk=192, return_aux=True)
                zm, slm, blm = m(img_m, meta_m, mask, chunk=192, return_aux=True)
                plain.append(z[0]); mirror.append(zm[0]); pa.append((sl, bl)); ma.append((slm, blm))
            obs = {'plain': plain, 'mirror': mirror, 'plain_aux': pa, 'mirror_aux': ma, 'state_k': models[0].state_k,
                   'bin_names': models[0].bin_names}
            p, _ = F.reconstruct(obs, item['fluid'], fluid, state, spec)
            rows.append([item['uid']] + ['%.9g' % float(x) for x in p])
    with open(out / 'predictions.csv', 'w', newline='') as h:
        w = csv.writer(h); w.writerow(['StudyInstanceUID'] + CT.LABELS); w.writerows(rows)
    receipt = {'status': 'PASS', 'fallback': fallback, 'timed_out': False, 'amp': 'fp16',
               'compile': {'requested': True, 'active': True, 'fallback': None}, 'state_stacker': {'stack_fallback': {}},
               'fluid_stacker': {'stack_fallback': {}}, 'stats': {}, 'seconds': 1.0}
    if a.flip_tta:
        receipt['flip_tta'] = {'perm': perm, 'bin_perm': bin_perm, 'mcl': 'plain pass only'}
    (out / 'receipt.json').write_text(json.dumps(receipt))
