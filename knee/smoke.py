"""Safe synthetic checks. Does not train, fetch data, or load weights."""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import tomllib
from .device import select_device
from .submission import write_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.toml"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--torch", action="store_true", help="Also run a tiny PyTorch forward pass")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    device = select_device(args.device or config["device"])
    if config["num_targets"] != 12:
        raise ValueError("Competition expects 12 targets")
    with tempfile.TemporaryDirectory() as temp:
        sample = Path(temp) / "synthetic_sample.csv"
        output = Path(temp) / "synthetic_output.csv"
        with sample.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["synthetic_id", *[f"synthetic_target_{i}" for i in range(12)]])
            writer.writerows([["fake_002", *([0.5] * 12)], ["fake_001", *([0.5] * 12)]])
        write_submission(sample, {"fake_001": [0.2] * 12, "fake_002": [0.8] * 12}, output)
        with output.open(newline="") as handle:
            result = list(csv.reader(handle))
        assert result[1][0] == "fake_002" and len(result[1]) == 13
    if args.torch:
        import torch
        from .model import StudyEncoder
        torch.manual_seed(config["seed"])
        model = StudyEncoder().to(device).eval()
        with torch.inference_mode():
            probabilities = model(torch.randn(2, 3, 2, 1, 32, 32, device=device)).sigmoid()
        assert tuple(probabilities.shape) == (2, 12)
        assert torch.isfinite(probabilities).all().item()
        assert ((probabilities >= 0) & (probabilities <= 1)).all().item()
    print(json.dumps({"status": "passed", "device": device, "torch_forward": args.torch,
                      "data": "synthetic only", "training": False}))


if __name__ == "__main__":
    main()
