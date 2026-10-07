"""Kaggle environment preparation and offline image-only inference."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


CODECS = ["pydicom==3.0.2", "pylibjpeg==2.1.0", "pylibjpeg-libjpeg==2.3.0", "pylibjpeg-openjpeg==2.5.0"]


def prepare_codecs(wheel_dir, *, download=False, expected_hashes=None):
    wheel_dir = Path(wheel_dir)
    if download:
        wheel_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--no-deps",
                        "--dest", str(wheel_dir), *CODECS], check=True)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != len(CODECS):
        raise ValueError("Missing codec wheel bundle")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in wheels}
    if expected_hashes is not None and hashes != expected_hashes:
        raise ValueError("Codec wheel hashes differ from the training bundle")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *map(str, wheels)], check=True)
    return hashes


def training_artifacts(root="/kaggle/input", slug="knee-dino-train-20261008"):
    """Known attachment layouts; never recurse through the raw DICOM corpus."""
    root = Path(root)
    candidates = [root / "notebooks" / "ozgung" / slug / "artifacts",
                  root / slug / "artifacts"]
    matches = [p for p in candidates if (p / "extraction.json").exists()
               and (p / "attention/head.pt").exists()]
    if len(matches) != 1:
        raise ValueError(f"Expected one trained artifact attachment, found {len(matches)}")
    return matches[0]


def competition_root(root="/kaggle/input"):
    root = Path(root)
    candidates = [root / "competitions/rsna-knee-abnormality-detection",
                  root / "rsna-knee-abnormality-detection"]
    matches = [p for p in candidates if (p / "sample_submission.csv").exists() and (p / "test_series.csv").exists()]
    if len(matches) != 1:
        raise ValueError(f"Expected one competition attachment, found {len(matches)}")
    return matches[0]


def infer(data_dir, encoder_dir, head_path, output_path, *, max_seconds=28000, repeat=1,
          additional_head_paths=(), head_weights=None):
    """Image-only inference; optional compatible heads share each encoded study.

    The original single-head call is unchanged. For an ensemble pass remaining
    checkpoint paths in additional_head_paths; head_weights covers the first head
    and all additional heads, in order. Omitted weights mean equal probabilities.
    """
    import numpy as np
    import torch
    from knee.dicom import PREPROCESSING_VERSION, StudyLoadError, load_study
    from knee.features import encode_volume, load_encoder
    from knee.ensemble import PredictionHeads
    from knee.experiment import environment, json_write
    from knee.submission import write_submission
    from knee.validation import read_csv, read_schema
    if not torch.cuda.is_available():
        raise RuntimeError("Offline inference explicitly requires CUDA")
    torch.set_num_threads(2)
    start = time.monotonic()
    root = Path(data_dir)
    id_column, targets = read_schema(root / "sample_submission.csv")
    _, test_rows = read_csv(root / "test.csv")
    ids = [r[id_column] for r in test_rows]
    if not ids or len(set(ids)) != len(ids) or any(not uid for uid in ids):
        raise ValueError("Invalid test IDs")
    _, series = read_csv(root / "test_series.csv")
    by_study = defaultdict(list)
    for row in series:
        by_study[row[id_column]].append(row)
    if set(ids) != set(by_study):
        raise ValueError("Test study/series coverage differs")
    heads = PredictionHeads([head_path, *additional_head_paths], targets, PREPROCESSING_VERSION,
                            device="cuda", weights=head_weights)
    prep = heads.preprocessing
    encoder = load_encoder(encoder_dir, "cuda")
    result, totals, errors = {}, Counter(), Counter()
    empty = 0
    timings = []
    for repetition in range(repeat):
        for i, uid in enumerate(ids):
            study_start = time.monotonic()
            if study_start - start > max_seconds:
                raise TimeoutError("Inference reached time bound; refusing a partial submission")
            try:
                volume = load_study(root / "test_series", uid, by_study[uid], image_size=prep["image_size"],
                                    slices_per_plane=prep["slices_per_plane"], crop_mm=prep["crop_mm"])
                features = encode_volume(encoder, volume, "cuda")
                with torch.inference_mode():
                    pred = heads(torch.from_numpy(features)[None].cuda(), torch.from_numpy(volume.slice_mask)[None].cuda())[0].cpu().numpy()
                totals.update(volume.audit["counts"])
                errors.update(volume.audit["errors"])
                totals["missing_planes"] += volume.audit["missing_planes"]
            except StudyLoadError as exc:
                # Predeclared fallback for unavailable imagery only, always counted.
                empty += 1
                errors.update(exc.audit["errors"])
                pred = heads.prior_probabilities.cpu().numpy()
            if not np.isfinite(pred).all():
                raise ValueError("Nonfinite prediction")
            if repetition and not np.allclose(result[uid], pred, atol=1e-5, rtol=1e-4):
                raise ValueError("Repeated offline predictions disagree")
            result[uid] = pred.tolist()
            timings.append(time.monotonic() - study_start)
            if (i + 1) % 50 == 0:
                print(json.dumps({"event":"inference_progress", "completed":i+1, "total":len(ids), "seconds":time.monotonic()-start}), flush=True)
    if empty > max(1, int(len(ids) * repeat * .02)):
        raise RuntimeError("More than 2% of studies lack usable images; inspect pipeline")
    # Hidden reruns may replace test.csv without replacing the three-row example sample.
    # Its header remains authoritative; actual test.csv controls hidden ID coverage/order.
    template = Path(output_path).with_name("runtime-template.csv")
    with template.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([id_column, *targets])
        writer.writerows([[uid, *([.5] * len(targets))] for uid in ids])
    write_submission(template, result, output_path)
    template.unlink()
    report = {"status":"completed", "studies":len(ids), "repeat":repeat, "fallback_studies":empty,
              "runtime_seconds":time.monotonic()-start, "mean_study_seconds":float(np.mean(timings)),
              "p95_study_seconds":float(np.percentile(timings,95)),
              "estimated_1300_seconds":float(np.mean(timings)*1300), "counts":dict(totals), "errors":dict(errors),
              "max_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "cuda_peak_bytes":torch.cuda.max_memory_allocated(), "environment":environment(),
              "head_count":len(heads.heads), "head_weights":list(heads.weights),
              "head_checkpoint_sha256":heads.checkpoint_sha256, "feature_sha256":heads.feature_sha256,
              "output_sha256":hashlib.sha256(Path(output_path).read_bytes()).hexdigest()}
    json_write(Path(output_path).with_name("inference.json"), report)
    print(json.dumps(report), flush=True)
    return report
