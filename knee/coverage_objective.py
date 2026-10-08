"""TRAIN04 objective comparison, provenance and bounded training diagnostics."""
import math
import shutil
from pathlib import Path

from knee.fresh_reader import fingerprint, macro_loss, restore_rng
from knee.validation import file_sha256

BATCH_OBJECTIVE = 'batch_macro_v1'
GLOBAL_OBJECTIVE = 'train_global_macro_v1'
EMPTY_POLICY = 'post_forward_skip_v1'
REFERENCE_REVISION = '6af890239a4111b1db2dbeab283bde01e90040d8'
REFERENCE_RUNNER = '81594a1d9b4030cdd4dda1e6d18e36c3642d7b482eaf669f5d15d3b5a9bb4a72'
REFERENCE_BUILD = '2b178cdb837b98a6f70c7e50ca448384ec88a0a099eb054d1b9d1a52a7cbd46b'
REFERENCE_SOURCES = {
    'knee/__init__.py': 'd46f0dce063fdd0d3846fc75f990e3025f9438192f05671e0c1f949d7da691cd',
    'knee/validation.py': '6c846ce49c7b6189e3e21543d32d8db5bc4a381801ccd373e8db5acbeff9ee8b',
    'knee/features.py': '81a44f1dcf6e88f488c2312797518d096328be1e245fbd0915853b7313801890',
    'knee/experiment.py': '35c3ffa7d66acabe6b3d3d8f232e2ab8b5dc1cf874be1ac5ebe22ef4a9be4015',
    'knee/fresh_reader.py': '165104e68fa49d0f43b228643d907e8fd4f23051d42d3c64d59c0d97779dc080',
    'knee/fresh_reader_setup.py': 'b4d701d6de3ddaceb0b837954e37c7620465f9755faf612862ab356592c45bd3',
    'knee/coverage_cache.py': '29716e4130e12934dbbd1f9653345a9b30a43cda57e8f883fbf66eabdda39d1c',
}


def training_mean_weights(weights, training_indices):
    """Fit denominators on the fixed training rows only, in stable float64."""
    import numpy as np
    indices = list(training_indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError('Unique nonempty training indices required')
    array=np.asarray(weights)
    if array.ndim!=2 or any(type(i) is not int or not 0<=i<len(array) for i in indices):
        raise ValueError('Training indices outside confidence matrix')
    values = array[indices].astype(np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid training confidence')
    means = values.mean(0)
    if not (means > 0).all():
        raise ValueError('Training target has no confidence support')
    return means.tolist()


def training_loss(numerator, denominator, *, objective, batch_size, mean_weights):
    """None means post-forward skip; callers must not mutate optimizer/EMA."""
    import torch
    if objective not in (BATCH_OBJECTIVE, GLOBAL_OBJECTIVE):
        raise ValueError('Unknown TRAIN04 objective')
    if (numerator.shape != denominator.shape or numerator.ndim != 1 or
            not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all() or
            (denominator < 0).any()):
        raise ValueError('Nonfinite or invalid training terms')
    if not (denominator > 0).any():
        return None
    if objective == BATCH_OBJECTIVE:
        return macro_loss(numerator, denominator)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Positive batch size required')
    if (mean_weights.shape != numerator.shape or not torch.isfinite(mean_weights).all() or
            not (mean_weights > 0).all()):
        raise ValueError('Invalid training-only denominators')
    return (numerator / batch_size / mean_weights).mean()


def validate_reference_prefix(report, run):
    """Verify finite original-run evidence; deliberately no accuracy/completion gate."""
    contract = report.get('contract', {})
    if (contract != run.get('contract') or contract.get('runner_sha256') != REFERENCE_RUNNER or
            contract.get('variant') != 'v1' or contract.get('version') != 'train04-v1' or
            contract.get('label_sources') != {} or
            run.get('environment', {}).get('code_revision') != REFERENCE_REVISION or
            run.get('environment', {}).get('source_sha256') != REFERENCE_BUILD):
        raise ValueError('Expected the immutable October9 original reference')
    if contract.get('fingerprint') != fingerprint({k:v for k,v in contract.items() if k != 'fingerprint'}):
        raise ValueError('Reference contract fingerprint differs')
    if contract.get('runtime_environment') != {k:run.get('environment',{}).get(k) for k in
                                                ('python','versions','device','cuda_devices')}:
        raise ValueError('Reference runtime environment receipt differs')
    runtime = contract.get('runtime_sources', {})
    if runtime.get('knee/coverage_training.py') != REFERENCE_RUNNER:
        raise ValueError('Reference runner source differs')
    for name in ('knee/coverage_cache.py', 'knee/fresh_reader.py', 'knee/fresh_reader_setup.py', 'knee/validation.py'):
        if runtime.get(name) != REFERENCE_SOURCES[name]:
            raise ValueError('Reference supporting source differs')
    if report.get('status') not in ('RUNNING', 'PASS', 'FAIL', 'PLATFORM_SESSION_STOP'):
        raise ValueError('Unknown reference execution state')
    count = report.get('completed_epochs')
    history = report.get('history', [])
    if (type(count) is not int or not 1 <= count <= 12 or len(history) != count or
            [r.get('epoch') for r in history] != list(range(1,count+1)) or
            (report['status'] == 'PASS' and count != 12)):
        raise ValueError('Reference requires a verified completed-epoch prefix')
    def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    for row in history:
        if not finite(row.get('weak_bce')) or row['weak_bce'] < 0:
            raise ValueError('Nonfinite reference prefix BCE')
        for key in ('per_target_bce', 'confidence_support'):
            values=row.get(key, [])
            if len(values)!=12 or any(not finite(v) or v < 0 for v in values):
                raise ValueError('Invalid reference per-target evidence')
        auc=row.get('thresholded_weak_auc', {})
        score=auc.get('macro_all_targets_auc')
        if score is not None and (not finite(score) or not 0 <= score <= 1):
            raise ValueError('Nonfinite reference prefix AUC')
        if score is None and not auc.get('undefined_targets'):
            raise ValueError('Missing reference AUC evidence')
    if (not finite(report.get('prevalence_weak_bce')) or
            report.get('selected_bce') != min(r['weak_bce'] for r in history) or
            type(report.get('successful_updates')) is not int or report['successful_updates'] <= 0):
        raise ValueError('Reference prefix selection/update evidence differs')
    initial_sha=run.get('initial_model_sha256')
    if (not isinstance(initial_sha,str) or len(initial_sha)!=64 or
            any(c not in '0123456789abcdef' for c in initial_sha)):
        raise ValueError('Reference initialization artifact hash missing')
    return {'reference_contract':contract, 'initial_sha256':initial_sha,
            'reference_status':report['status'], 'completed_epochs':count}


def require_objective_invariants(current, reference, *, environment=True):
    """One explicitly named source/objective change, never a legacy resume waiver."""
    if reference.get('runner_sha256') != REFERENCE_RUNNER or reference.get('variant') != 'v1':
        raise ValueError('Objective comparison requires pinned original runner')
    keys=['config','plan_fingerprint','source_hashes','index_sha256','complete_sha256',
          'labels_fingerprint','weights_fingerprint']
    if environment:keys.append('runtime_environment')
    if any(k not in current or current[k] != reference.get(k) for k in keys):
        raise ValueError('Objective comparison changed a scientific/environment invariant')
    if current.get('objective') != GLOBAL_OBJECTIVE or current.get('variant') != 'v1':
        raise ValueError('Objective comparison must retain v1 supervision')
    for name,digest in REFERENCE_SOURCES.items():
        if current.get('comparison_sources',{}).get(name) != digest:
            raise ValueError('Objective comparison changed an unaffected source')
    expected=set(REFERENCE_SOURCES)|{'knee/coverage_training.py','knee/coverage_objective.py'}
    if set(current.get('comparison_sources',{})) != expected:
        raise ValueError('Unexpected objective comparison source inventory')


def copy_initial(source, destination, expected_sha256):
    if file_sha256(source) != expected_sha256:
        raise ValueError('Original initialization artifact differs')
    shutil.copyfile(source, destination)
    if file_sha256(destination) != expected_sha256:
        raise ValueError('Initialization copy differs')


def load_exact_initial(model, source, *, expected_sha256, plan_fingerprint, config):
    """Our hash-bound private initialization, including head tensors and all RNG."""
    import torch
    if file_sha256(source)!=expected_sha256:
        raise ValueError('Original initialization artifact differs')
    initial=torch.load(source,map_location='cpu',weights_only=False)
    if initial['plan_fingerprint']!=plan_fingerprint or initial['config']!=config:
        raise ValueError('Reference initialization contract differs')
    model.load_state_dict(initial['model'],strict=True)
    restore_rng(initial['rng'])
    return initial


class StepTelemetry:
    """Cumulative scalar statistics; one bounded transfer when reporting."""
    def __init__(self, device, state=None):
        import torch
        self.values=torch.zeros(7,device=device,dtype=torch.float64) if state is None else state.to(device).clone()

    def add(self, loss, preclip_norm, overflow):
        import torch
        loss=loss.detach().to(dtype=torch.float64)
        norm=preclip_norm.detach().to(dtype=torch.float64)
        finite=torch.isfinite(norm)
        safe=torch.where(finite,norm,torch.zeros_like(norm))
        self.values[:4] += torch.stack((loss,torch.ones_like(loss),safe,finite.to(dtype=torch.float64)))
        self.values[4]=torch.maximum(self.values[4],safe)
        self.values[5] += (finite & (norm>3)).to(dtype=torch.float64)
        self.values[6] += int(overflow)

    def report(self):
        loss,count,total,finite,maximum,clipped,overflows=self.values.detach().cpu().tolist()
        return dict(supervised_attempts=int(count),objective_mean=loss/count if count else None,
                    finite_preclip_norms=int(finite),preclip_norm_mean=total/finite if finite else None,
                    preclip_norm_max=maximum if finite else None,
                    clipped_fraction=clipped/finite if finite else None,amp_skips=int(overflows))


def parameter_probes(model):
    names=[next(n for n,_ in model.named_parameters() if n.startswith('enc.')),'cls_w']
    params=dict(model.named_parameters())
    return {n:params[n].detach().flatten()[:32].clone() for n in names}


def probe_displacements(model, probes):
    import torch
    params=dict(model.named_parameters());result={}
    with torch.no_grad():
        for name,baseline in probes.items():
            delta=params[name].detach().flatten()[:len(baseline)].float()-baseline.to(params[name].device).float()
            l2,maximum=torch.stack((delta.norm(),delta.abs().max())).cpu().tolist()
            result[name]={'sampled_parameters':len(baseline),'l2_change':l2,'max_abs_change':maximum}
    return result
