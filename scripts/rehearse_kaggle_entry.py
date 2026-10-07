"""Offline gold-image parity and repeated visible-test rehearsal; no fitting."""
import json
import os
from knee.runtime import competition_root, prepare_codecs, training_artifacts
from knee.features import find_encoder
from knee.rehearsal import rehearse

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
artifacts = training_artifacts()
prepare_codecs(artifacts / "wheels",
               expected_hashes=json.loads((artifacts / "wheel-hashes.json").read_text()))
rehearse(competition_root(), find_encoder("/kaggle/input"),
         [artifacts / "attention/head.pt"], artifacts / "attention/gold_predictions.npz",
         "/kaggle/working")
