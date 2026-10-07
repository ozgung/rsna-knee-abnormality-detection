"""Lazy hardware selection so metadata tools work without PyTorch."""
import importlib


def select_device(requested="auto"):
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"Unknown device: {requested}")
    if requested == "cpu":
        return "cpu"
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        if requested == "auto":
            return "cpu"
        raise RuntimeError("Install PyTorch before requesting a GPU") from exc
    cuda = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend and mps_backend.is_available())
    available = {"cuda": cuda, "mps": mps}
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if not available[requested]:
        raise RuntimeError(f"Requested device {requested!r} is unavailable")
    return requested
