"""Disposable CUDA32 trainability probe: no validation, selection or saved weights."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

from knee.fresh_reader import (BACKBONE, PREPROCESS_VERSION, SOURCE_HASHES, check_sources,
                               environment, fingerprint, load_cached_batch, load_source,
                               loss_terms, macro_loss, per_target_bce, pilot_order,
                               validate_plan, write_json, tensor_state_sha256)
from knee.fresh_reader_setup import progress, time_limit
from knee.validation import file_sha256

PROBE_STUDIES = 32
PROBE_UPDATES = 64
PROBE_SECONDS = 600


def select_probe_records(plan, index, count=PROBE_STUDIES):
    """Hash-order cached training rows only; target values never select rows."""
    if type(count) is not int or not 1 <= count <= PROBE_STUDIES:
        raise ValueError('Probe count must be in 1..32')
    candidates = [r for r in plan['records'] if r['split'] == 'train' and r['uid'] in index]
    if len(candidates) < count:
        raise ValueError('Insufficient cached training studies for the declared probe')
    selected = pilot_order(candidates, count)
    if len({r['uid'] for r in selected}) != count:
        raise ValueError('Duplicate probe studies')
    return selected


def verify_probe_cache(cache_dir, expected_fingerprint, *, count=PROBE_STUDIES):
    import numpy as np
    cache_dir = Path(cache_dir)
    plan = json.loads((cache_dir / 'plan-private.json').read_text())
    if validate_plan(plan) != expected_fingerprint:
        raise ValueError('Probe plan fingerprint differs')
    if plan['source_hashes'] != SOURCE_HASHES or plan['preprocessing'] != PREPROCESS_VERSION:
        raise ValueError('Probe source/preprocessing contract differs')
    cfg = plan['config']
    if cfg['image_size'] != 224 or cfg['windows'] != 4:
        raise ValueError('Probe requires the original 224px/four-window cache')
    receipt = json.loads((cache_dir / 'cache-pilot.json').read_text())
    if (receipt['status'] != 'PASS' or receipt['plan_fingerprint'] != expected_fingerprint or
            receipt.get('failures', 0) != 0):
        raise ValueError('Probe requires a successful, matching decoding pilot')
    index = json.loads((cache_dir / 'cache-index-private.json').read_text())
    if len(index) != receipt['cached_studies']:
        raise ValueError('Pilot index coverage differs from its receipt')
    records = select_probe_records(plan, index, count)
    for record in records:
        item = index[record['uid']]
        expected_name = hashlib.sha256(record['uid'].encode()).hexdigest() + '.npz'
        if item['file'] != expected_name:
            raise ValueError('Unexpected cached probe filename')
        path = cache_dir / 'arrays' / expected_name
        if path.stat().st_size != item['bytes'] or file_sha256(path) != item['sha256']:
            raise ValueError('Probe cache array changed')
        with np.load(path, allow_pickle=False) as z:
            x, mask, pos = z['x'], z['mask'], z['pos']
            if (x.shape != (6, 4, 3, 224, 224) or x.dtype != np.uint8 or
                    mask.shape != (6,) or mask.dtype != np.bool_ or not mask.any() or
                    pos.shape != (6, 4) or not np.isfinite(pos).all() or
                    (pos < 0).any() or (pos > 1).any() or x[~mask].any()):
                raise ValueError('Invalid probe shapes, sequence masks, positions or padding')
        y = np.asarray(record['labels'])
        if y.shape != (12,) or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
            raise ValueError('Invalid selected training labels')
    return plan, receipt, index, records


def probe(cache_dir, source_dir, output_dir, expected_fingerprint):
    """At most ten minutes, 32 cached training studies, 64 successful updates.

    The original full-cache resource gate may be false: this probe reads only
    already-decoded arrays and does not authorize or perform cache expansion.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {'status': 'RUNNING', 'purpose': 'CUDA/AMP/trainability diagnostic, not model quality or selection',
              'plan_fingerprint': expected_fingerprint, 'max_seconds': PROBE_SECONDS,
              'max_studies': PROBE_STUDIES, 'target_updates': PROBE_UPDATES, 'batch_size': 2,
              'checkpoint_policy': 'No trained weights saved or reused by later training',
              'source_hashes': SOURCE_HASHES, 'runner_sha256': file_sha256(__file__),
              'attempted_steps': 0, 'successful_updates': 0, 'overflow_skips': 0,
              'unknown_batch_skips': 0, 'finite_gradient_steps': 0, 'nonfinite_gradient_steps': 0}
    torch = None
    try:
        with time_limit(PROBE_SECONDS, 'whole CUDA32 probe'):
            import numpy as np
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA probe requires CUDA; no CPU fallback')
            report['environment'] = environment()
            progress('probe_cache_verification_begin')
            check_sources(source_dir)
            plan, receipt, index, records = verify_probe_cache(cache_dir, expected_fingerprint)
            report.update(studies=len(records), selection_fingerprint=fingerprint([r['uid'] for r in records]),
                          original_pilot_gate=receipt['gate'], seed=plan['config']['seed'], config=plan['config'])
            cfg = plan['config']
            random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
            torch.set_num_threads(2)
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
            os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '30')
            progress('probe_imagenet_initialization_begin', backbone=BACKBONE)
            with time_limit(300, 'probe ImageNet initialization'):
                code = load_source(source_dir, 'knee')
                if code.LABELS != plan['targets']:
                    raise ValueError('Probe target order differs from official plan')
                model = code.KneeNet(BACKBONE, pretrained=True, img_size=224).cuda()
            report['backbone'] = BACKBONE
            report['pretrained_cfg'] = model.enc.pretrained_cfg
            report['initial_encoder_sha256'] = tensor_state_sha256(model.enc.state_dict())
            report['encoder_hash_scheme'] = 'fresh-encoder-state-v1'
            # Keep the source head's zero bias: no unsupported-target prior fit
            # or access to labels outside the selected 32 training studies.
            report['head_initialization'] = 'Fresh source head; zero classifier bias; no fitted prior'
            ema = copy.deepcopy(model).eval().requires_grad_(False)
            optimizer = torch.optim.AdamW([
                {'params': model.enc.parameters(), 'lr': cfg['lr_encoder']},
                {'params': [p for n, p in model.named_parameters() if not n.startswith('enc.')], 'lr': cfg['lr_head']}],
                weight_decay=cfg['weight_decay'])
            scaler = torch.amp.GradScaler('cuda')
            encoder_parameter = next(model.enc.parameters())
            initial_encoder = encoder_parameter.detach().flatten()[:128].cpu().clone()
            initial_head = model.cls_w.detach().cpu().clone()
            torch.cuda.reset_peak_memory_stats()
            report['optimizer'] = {k: cfg[k] for k in ('lr_encoder', 'lr_head', 'weight_decay', 'ema')}
            report['ema_role'] = 'Allocated and updated to match training memory; diagnostics use online weights only'

            @torch.inference_mode()
            def diagnostic():
                model.eval()
                numerator, denominator = torch.zeros(12), torch.zeros(12)
                torch.cuda.synchronize(); tick = time.monotonic()
                for start in range(0, len(records), 2):
                    x, mask, pos, y = load_cached_batch(records, index, cache_dir, list(range(start, min(start+2, len(records)))))
                    with torch.autocast('cuda', dtype=torch.float16):
                        logits = model(x, mask, pos, res=224, train=False)
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite unaugmented probe logits')
                    n, d = loss_terms(logits, y)
                    numerator += n.cpu(); denominator += d.cpu()
                torch.cuda.synchronize()
                return {'macro_bce': float(macro_loss(numerator, denominator)),
                        'per_target_bce': per_target_bce(numerator, denominator),
                        'confidence_support': denominator.tolist(),
                        'seconds': time.monotonic() - tick}

            initial = diagnostic()
            report['initial_unaugmented_training'] = initial
            progress('probe_initial_diagnostic', loss=initial['macro_bce'], seconds=initial['seconds'])
            final_reserve = max(45., 2 * initial['seconds'] + 10.)
            order = np.random.permutation(len(records)).tolist()
            cursor = 0
            forward_backward_seconds, total_step_seconds, successful_step_seconds = [], [], []
            # Preserve completed timing evidence even if a later CUDA step fails.
            report.update(forward_backward_seconds=forward_backward_seconds,
                          total_step_seconds=total_step_seconds, successful_step_seconds=successful_step_seconds)
            stop_reason = 'target_updates'
            # A few scaler warmup skips are allowed without counting as updates.
            while report['successful_updates'] < PROBE_UPDATES and report['attempted_steps'] < 96:
                if time.monotonic() - started >= PROBE_SECONDS - final_reserve - 15:
                    stop_reason = 'time_budget'; break
                if cursor >= len(order):
                    order = np.random.permutation(len(records)).tolist(); cursor = 0
                batch_indices = order[cursor:cursor+2]; cursor += 2
                model.train()
                torch.cuda.synchronize(); step_started = time.monotonic()
                x, mask, pos, y = load_cached_batch(records, index, cache_dir, batch_indices)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); compute_started = time.monotonic()
                with torch.autocast('cuda', dtype=torch.float16):
                    n, d = loss_terms(model(x, mask, pos, res=224, train=True), y)
                report['attempted_steps'] += 1
                if not torch.isfinite(n).all():
                    raise ValueError('Nonfinite probe training loss')
                if not (d > 0).any():
                    report['unknown_batch_skips'] += 1; continue
                scaler.scale(macro_loss(n, d)).backward()
                torch.cuda.synchronize()
                forward_backward_seconds.append(time.monotonic() - compute_started)
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
                finite = bool(torch.isfinite(norm))
                report['finite_gradient_steps' if finite else 'nonfinite_gradient_steps'] += 1
                updated = False
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() < old_scale:
                    report['overflow_skips'] += 1
                else:
                    if not finite:
                        raise ValueError('Nonfinite gradients without an AMP skipped update')
                    updated = True
                    report['successful_updates'] += 1
                    step = report['successful_updates']
                    decay = min(cfg['ema'], (1 + step) / (10 + step))
                    with torch.no_grad():
                        for a, b in zip(ema.parameters(), model.parameters()): a.lerp_(b, 1-decay)
                        for a, b in zip(ema.buffers(), model.buffers()): a.copy_(b)
                torch.cuda.synchronize()
                duration = time.monotonic() - step_started
                total_step_seconds.append(duration)
                if updated:
                    successful_step_seconds.append(duration)
                if report['attempted_steps'] % 8 == 0:
                    progress('probe_update', attempted=report['attempted_steps'], updates=report['successful_updates'],
                             overflow_skips=report['overflow_skips'], elapsed_seconds=time.monotonic()-started)
            if report['attempted_steps'] >= 96 and report['successful_updates'] < PROBE_UPDATES:
                stop_reason = 'attempt_limit'
            final = diagnostic()
            encoder_delta = float((encoder_parameter.detach().flatten()[:128].cpu() - initial_encoder).abs().max())
            head_delta = float((model.cls_w.detach().cpu() - initial_head).abs().max())
            report.update(final_unaugmented_training=final,
                          loss_change=final['macro_bce'] - initial['macro_bce'],
                          encoder_parameter_sample_max_change=encoder_delta, head_parameter_max_change=head_delta,
                          stop_reason=stop_reason, final_amp_scale=scaler.get_scale(),
                          forward_backward_seconds=forward_backward_seconds, total_step_seconds=total_step_seconds,
                          successful_step_seconds=successful_step_seconds)
            steady = successful_step_seconds[4:] or successful_step_seconds
            report['steady_step_seconds_mean'] = float(np.mean(steady)) if steady else None
            report['steady_step_seconds_p95'] = float(np.percentile(steady, 95)) if steady else None
            report['status'] = 'PASS' if report['successful_updates'] >= 8 and encoder_delta > 0 and head_delta > 0 else 'PARTIAL'
            # No loss-improvement condition: fitting these same rows is only a
            # numerical diagnostic, never a holdout or model-selection score.
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        if torch is not None and torch.cuda.is_available():
            report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            report['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
        write_json(output_dir/'probe-report.json', report)
    return report
