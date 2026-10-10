"""Full-state public knee model qualification, not a validation experiment."""
from pathlib import Path
import importlib.metadata
import json
import os
import sys
import time


def run(public_source, decoder_source, manifest, checkpoint, root, out):
    import numpy as np
    import torch
    from knee import mirror_input
    from knee.mirror_submission import read_inputs
    from knee.reference import load_definitions, load_model, validate_predictions

    if not torch.cuda.is_available():
        raise RuntimeError('Full warm-start qualification requires CUDA')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'status': 'running', 'purpose': 'full-state adaptation prerequisite only',
              'seed': 20261010, 'training_steps': 1, 'training_windows': 2,
              'training_targets': 'synthetic alternating binary targets; no labels read',
              'generalization_evidence': False, 'checkpoint_sha256': manifest['checkpoint_sha256'],
              'source_revision': os.environ['KNEE_CODE_REVISION'],
              'source_bundle_sha256': os.environ['KNEE_SOURCE_SHA256']}

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        (out / 'qualification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    save()
    try:
        torch.manual_seed(report['seed'])
        torch.cuda.manual_seed_all(report['seed'])
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        public, decoder = load_definitions(public_source, decoder_source, manifest)
        columns, ids, records = read_inputs(root, visible=True)
        if columns != ['StudyInstanceUID', *public.LAB]:
            raise ValueError('Official target order differs from public checkpoint')
        model = load_model(public, checkpoint, 'cuda', manifest)
        report.update(strict_full_state_load=True, explicit_model_input_size=320,
                      environment={'python': sys.version, 'cuda': torch.version.cuda,
                                   'device': torch.cuda.get_device_name(0),
                                   'packages': {p: importlib.metadata.version(p) for p in ('torch', 'timm', 'numpy', 'pydicom')}})
        torch.cuda.reset_peak_memory_stats()
        frozen_errors, frozen_values, inputs = [], [], None
        # Read only the three visible image studies, never reports or train labels.
        for sid in ids:
            volume, mask, _ = mirror_input.prepare_study(public, decoder, sid, records, Path(root) / 'test_series')
            windows = [public.gpu_windows(volume, mask, 94, 320, 'cuda', mirror=m) for m in (False, True)]
            if any(tuple(w.shape) != (94, 3, 320, 320) for w in windows):
                raise ValueError('Public rich-input geometry changed')
            reference = public.infer_pair_probs(model, *windows, 'cuda')
            # Candidate training wrapper uses the full module's forward, including
            # the inherited norm, target attention and classification parameters.
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                candidate = sum(torch.sigmoid(model(w[None]).float()) for w in windows)[0].cpu().numpy() / 2
            frozen_errors.append(float(np.max(np.abs(reference - candidate))))
            frozen_values.append(validate_predictions(candidate).tolist())
            if inputs is None:
                inputs = windows[0][:2].clone()[None]
            del windows, volume, mask
        report.update(visible_studies=len(ids), frozen_prediction_max_error=max(frozen_errors),
                      frozen_probabilities=frozen_values, input_contract=manifest['recipe'])
        if max(frozen_errors) > 1e-7:
            raise ValueError('Zero-update wrapper differs from public inference')

        model.train()
        groups = {'encoder': [(n, p) for n, p in model.named_parameters() if n.startswith('backbone.')],
                  'head': [(n, p) for n, p in model.named_parameters() if not n.startswith('backbone.')]}
        if any(not ps for ps in groups.values()):
            raise ValueError('Missing full encoder/head parameter group')
        probes = {g: (ps[0][0], ps[0][1].detach().clone()) for g, ps in groups.items()}
        # One fp32 SGD step tests the actual full model, with a tiny view batch.
        # This is a mechanical update check, not the adaptation training recipe.
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        targets = (torch.arange(len(public.LAB), device='cuda') % 2).float()[None]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite warm-start loss')
        loss.backward()
        gradient_counts = {}
        for group, params in groups.items():
            if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in params):
                raise ValueError(f'Missing/nonfinite {group} gradient')
            count = sum(int(torch.count_nonzero(p.grad)) for _, p in params)
            if count == 0:
                raise ValueError(f'Zero {group} gradients')
            gradient_counts[group] = count
        optimizer.step()
        named = dict(model.named_parameters())
        changes = {g: float((named[n].detach() - before).abs().max()) for g, (n, before) in probes.items()}
        if any(v <= 0 or not np.isfinite(v) for v in changes.values()):
            raise ValueError('Encoder/head update probe failed')
        if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
            raise ValueError('Updated full state contains nonfinite values')
        model.eval()
        with torch.no_grad():
            expected = model(inputs).detach().cpu()
        path = out / 'smoke-state.pt'
        torch.save({'model': model.state_dict(), 'synthetic_smoke_only': True}, path)
        # Reload actual saved bytes strictly; no key filtering or shape fallback.
        loaded = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(loaded['model'], strict=True)
        with torch.no_grad():
            actual = model(inputs).detach().cpu()
        error = float((actual - expected).abs().max())
        if not torch.equal(actual, expected):
            raise ValueError('Save/reload prediction parity failed')
        report.update(status='pass', loss=float(loss.detach()), nonzero_gradient_elements=gradient_counts,
                      probe_parameter_max_changes=changes, save_reload_max_error=error,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated())
        # Synthetic smoke weights must never be promoted as a training candidate.
        del loaded
        path.unlink()
        save()
        return report
    except BaseException as error:
        report.update(status='fail', error_type=type(error).__name__, error=str(error))
        save()
        raise
