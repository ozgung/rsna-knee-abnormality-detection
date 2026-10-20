"""Synthetic stand-in for compact_train (tests only): the Net constructor/forward shapes the R6 observer relies on."""
import torch

LABELS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
          'Synovitis', "Baker's", 'Contusion', 'Fracture']


class Net(torch.nn.Module):
    def __init__(self, backbone, init, hidden=4, drop=0.1, freeze_blocks=16, crop=8, state_spec=None, in_chans=3, spatial=False,
                 extent=False, syn_decomp=False):
        super().__init__()
        self.state_k = [len(state_spec['categories'][l]) for l in LABELS]
        self.bin_names = [(b, l) for b, labs in state_spec['binaries'].items() for l in labs]
        self.enc = torch.nn.Linear(in_chans, hidden)
        self.head = torch.nn.Linear(hidden, 12 + 12 * max(self.state_k) + len(self.bin_names))
        self.register_buffer('mean', torch.zeros(1))
        # the head keys compact_infer's load_group detects (sp_q / ext_u / syn_beta), so key-based construction matches
        if spatial:
            self.sp_q = torch.nn.Parameter(torch.zeros(1))
        if extent:
            self.ext_u = torch.nn.Parameter(torch.zeros(1))
        if syn_decomp:
            self.syn_beta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, img, meta, mask, chunk=0, return_aux=False):
        x = (img.float().mean(dim=(-1, -2))[0] + meta[0, :, 2:3]).mean(0, keepdim=True)
        y = self.head(self.enc(x))
        k = max(self.state_k)
        z, sl, bl = y[:, :12], y[:, 12:12 + 12 * k].reshape(1, 12, k), y[:, 12 + 12 * k:]
        return (z, sl, bl) if return_aux else z
