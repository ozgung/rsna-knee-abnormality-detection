"""VM inference passes on the staged raptor64 cache (no training).

  e1      in-orig final SWA (pinned) on the 870 fold-0 studies at K42: full, and with the windows whose centre
          lies in each plane group (sagittal 0-31, coronal 32-51, axial 52-63) dropped from the standard K42 plan.
          Writes /content/raptor64-e1/e1-ablation-private.npz plus a receipt (parity vs the exported K42 OOF).
  teacher Dread v4_full (pinned CC0, full model: backbone + head) K42 probabilities on all 4,345 cached studies.
          Writes /content/raptor64-teacher/teacher-dread-v4-k42-private.npz plus a receipt with its SHA.
Both use the in-orig package's pinned trainer classes (StudyWindows eval mode == public eval_windows) and fp16 autocast.
"""
import csv, hashlib, json, os, subprocess, sys, tempfile, time, zipfile, shutil
from pathlib import Path

mode = sys.argv[1]
ROOT = Path('/content'); PKG = ROOT / 'raptor64-in-orig-pkg'
receipt = json.loads((ROOT / 'raptor64_cache/staging-receipt.json').read_bytes())
os.environ.update(RAPTOR64_STAGE_ROOT=str(ROOT / 'raptor64_cache'), RAPTOR64_MANIFEST_SHA256=receipt['manifest_sha256'],
                  RAPTOR64_CONTENT_SHA256=receipt['content_sha256'], KAGGLE_CONFIG_DIR='/content', RAPTOR_PRELOAD='0',
                  RAPTOR64_STAGE_RECEIPT_SHA256=hashlib.sha256((ROOT / 'raptor64_cache/staging-receipt.json').read_bytes()).hexdigest(),
                  RAPTOR_INPUT=str(ROOT / 'input-raptor64-in-orig'))
sys.path.insert(0, str(PKG)); os.chdir(PKG)
import numpy as np, torch, timm
import train_knee_patched as tk
from knee import raptor64_training as s
IN_ORIG = {'dataset': 'ozgung/knee-raptor64-in-orig-fold0-final-20261013', 'sha256': 'e9e131a15a6673485b048e22ca33e77b5512eeabde915b695d952c1a064f4b90'}
DREAD = ROOT / 'raptor64-init-dread_v4/raptor_ft_coatnet_v4_full.pt'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def fetch(dataset, name, dest):
    with tempfile.TemporaryDirectory(dir='/content') as d:
        r = subprocess.run(['kaggle', 'datasets', 'download', dataset, '-f', name, '-p', d, '--unzip', '-o', '-q'], capture_output=True, text=True)
        assert r.returncode == 0, 'download failed'
        for z in Path(d).glob('*.zip'):
            shutil.unpack_archive(str(z), d); z.unlink()
        shutil.move(str(Path(d) / name), dest)
    return dest


def model_from(state):
    bb = tk.build_backbone(s.ARCH, pretrained=False)
    m = tk.RaptorClassifier(bb, F_dim=bb.num_features)
    m.load_state_dict(state, strict=True); m.win_chunk = 0
    return m.eval().cuda()


s.strict_cuda(torch)
vols, masks, ids = s.staged_corpus(); ids = [str(u) for u in ids]; id2row = {u: i for i, u in enumerate(ids)}
split, train_ids, dev_ids = s.runtime_split(PKG / 'folds_v0.json', PKG / 'excluded_ids.json', PKG / 'train_knee_patched.py')
assert len(dev_ids) == 870 and len(ids) == 4345
start = time.time()
if mode == 'e1':
    out = ROOT / 'raptor64-e1'; out.mkdir(exist_ok=False)
    ck = fetch(IN_ORIG['dataset'], 'raptor_ft_fold0_swa.pt', out / 'in-orig-swa.pt'); assert sha(ck) == IN_ORIG['sha256']
    oof = fetch(IN_ORIG['dataset'], 'oof_fold0.csv', out / 'in-orig-oof_fold0.csv')
    model = model_from(torch.load(ck, map_location='cpu', weights_only=False)['model'])
    slot_of = np.repeat(np.arange(5), [18, 14, 12, 8, 12]); group_of_slot = np.array([0, 0, 1, 1, 2])  # sag, cor, ax
    ds = tk.StudyWindows('', dev_ids, id2row, {u: np.zeros(12, np.float32) for u in dev_ids}, 384, 42, train=False, norm='imagenet')
    res = {k: np.full((870, 12), np.nan, np.float32) for k in ('full', 'no_sagittal', 'no_coronal', 'no_axial')}
    counts = np.zeros((870, 3), np.int16)
    with torch.no_grad():
        for i, u in enumerate(dev_ids):
            x, _ = ds[i]
            centers = [max(1, min(int(c), 62)) for c in s.eval_centers(masks[id2row[u]], 64, 42)]
            g = group_of_slot[slot_of[np.asarray(centers)]]
            counts[i] = [int((g == j).sum()) for j in range(3)]
            with torch.autocast('cuda', dtype=torch.float16):
                f = model.encode(x[None].cuda())
                res['full'][i] = torch.sigmoid(model.head(f).float())[0].cpu().numpy()
                for j, name in enumerate(('no_sagittal', 'no_coronal', 'no_axial')):
                    keep = torch.from_numpy(g != j).cuda()
                    if bool(keep.any()):
                        res[name][i] = torch.sigmoid(model.head(f[:, keep]).float())[0].cpu().numpy()
    with open(oof, newline='') as fh:
        rows = [r for r in csv.DictReader(fh) if r['set'] == 'val']
    assert [r['StudyInstanceUID'] for r in rows] == dev_ids
    exported = np.asarray([[float(r[t]) for t in tk.LAB] for r in rows], np.float32)
    np.savez(out / 'e1-ablation-private.npz', ids=np.asarray(dev_ids), window_counts=counts, **res)
    rec = {'mode': 'e1', 'checkpoint_sha256': IN_ORIG['sha256'], 'studies': 870, 'k': 42,
           'parity_full_vs_exported_k42_max_abs': float(np.nanmax(np.abs(res['full'] - exported))),
           'studies_without_windows_after_drop': {n: int(np.isnan(res[n][:, 0]).sum()) for n in ('no_sagittal', 'no_coronal', 'no_axial')},
           'mean_windows_per_group': {'sagittal': float(counts[:, 0].mean()), 'coronal': float(counts[:, 1].mean()), 'axial': float(counts[:, 2].mean())},
           'method': 'drop windows whose centre lies in the plane group from the standard K42 plan; features computed once, head rerun on the kept windows',
           'npz_sha256': sha(out / 'e1-ablation-private.npz'), 'seconds': round(time.time() - start, 1)}
    (out / 'e1-receipt.json').write_text(json.dumps(rec, indent=2) + '\n'); print(json.dumps(rec))
elif mode == 'teacher':
    out = ROOT / 'raptor64-teacher'; out.mkdir(exist_ok=False)
    assert sha(DREAD) == s.DREAD_PIN['sha256']
    ck = torch.load(DREAD, map_location='cpu', weights_only=True)
    assert ck['arch'] == s.ARCH and list(ck['lab']) == s.LAB
    model = model_from(ck['model'])
    ds = tk.StudyWindows('', ids, id2row, {u: np.zeros(12, np.float32) for u in ids}, 384, 42, train=False, norm='imagenet')
    dl = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=4, collate_fn=tk.collate)
    probs = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast('cuda', dtype=torch.float16):
                probs.append(torch.sigmoid(model(x.cuda()).float()).cpu().numpy())
    probs = np.concatenate(probs).astype(np.float32)
    assert probs.shape == (4345, 12) and np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
    np.savez(out / 'teacher-dread-v4-k42-private.npz', ids=np.asarray(ids), probs=probs)
    rec = {'mode': 'teacher', 'teacher': 'Dread v4_full CC0 (backbone + head), K42 public eval windows, fp16', 'checkpoint_sha256': s.DREAD_PIN['sha256'],
           'studies': 4345, 'ids_order': 'staged corpus order', 'npz_sha256': sha(out / 'teacher-dread-v4-k42-private.npz'),
           'probs_mean': probs.mean(0).round(4).tolist(), 'seconds': round(time.time() - start, 1),
           'exposure': 'Dread saw all non-gold studies; teacher targets carry second-order exposure to fold-0 dev'}
    (out / 'teacher-receipt.json').write_text(json.dumps(rec, indent=2) + '\n'); print(json.dumps(rec))
