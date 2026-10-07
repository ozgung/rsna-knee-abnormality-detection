"""Compatible prediction heads sharing one encoder feature tensor.

Importing this module does not import ML dependencies. Heads produce probability
averages, never averaged logits. A single head keeps its original numeric path.
"""
import math


def validate_weights(count, weights=None):
    if count < 1:
        raise ValueError("At least one prediction head is required")
    values = [1 / count] * count if weights is None else [float(value) for value in weights]
    if len(values) != count or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Head weights must be finite, nonnegative and match head count")
    total = math.fsum(values)
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError("Head weights must sum to one")
    # Remove harmless representation error only after the sum-to-one check.
    return tuple(value / total for value in values)


class PredictionHeads:
    """Load compatible frozen heads; caller encodes each study exactly once."""

    def __init__(self, paths, targets, preprocessing_version, *, device, weights=None):
        import torch
        from knee.features import StudyHead
        from knee.validation import file_sha256

        paths = list(paths)
        self.weights = validate_weights(len(paths), weights)
        checkpoints = [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
        first = checkpoints[0]
        self.targets = list(targets)
        self.preprocessing = first["preprocessing"]
        self.feature_sha256 = first.get("feature_sha256")
        if len(checkpoints) > 1 and (not isinstance(self.feature_sha256, str) or not self.feature_sha256):
            raise ValueError("Ensemble checkpoint is missing its feature-cache fingerprint")
        for checkpoint in checkpoints:
            if checkpoint["targets"] != self.targets:
                raise ValueError("Checkpoint target order mismatch")
            prep = checkpoint["preprocessing"]
            if prep["preprocessing_version"] != preprocessing_version:
                raise ValueError("Preprocessing version mismatch")
            if prep != self.preprocessing:
                raise ValueError("Ensemble checkpoints have different preprocessing")
            if checkpoint.get("feature_sha256") != self.feature_sha256:
                raise ValueError("Ensemble checkpoints use different feature caches")
        self.heads = []
        priors = []
        for checkpoint in checkpoints:
            state, config = checkpoint["state_dict"], checkpoint["config"]
            head = StudyHead(state["mean"], state["std"], state["prior_logits"].sigmoid(),
                             kind=config["kind"], hidden=config["hidden"], dropout=config.get("dropout", .2))
            head.load_state_dict(state, strict=True)
            self.heads.append((head.cuda() if device == "cuda" else head.to(device)).eval())
            priors.append(state["prior_logits"].sigmoid())
        self.prior_probabilities = self._average(priors)
        self.checkpoint_sha256 = [file_sha256(path) for path in paths]

    def _average(self, probabilities):
        import torch

        if any(not torch.isfinite(value).all() or (value < 0).any() or (value > 1).any()
               for value in probabilities):
            raise ValueError("Nonfinite or invalid head probability")
        if any(value.shape != probabilities[0].shape for value in probabilities):
            raise ValueError("Head probability shape mismatch")
        if len(probabilities) == 1:
            return probabilities[0]
        stacked = torch.stack(probabilities).double()
        weights = torch.tensor(self.weights, dtype=stacked.dtype, device=stacked.device)
        shape = (len(weights),) + (1,) * (stacked.ndim - 1)
        # Clamp only floating-point summation roundoff; inputs/weights validated.
        return (stacked * weights.reshape(shape)).sum(0).clamp(0, 1)

    def __call__(self, features, masks):
        import torch

        with torch.inference_mode():
            return self._average([head(features, masks).sigmoid() for head in self.heads])
