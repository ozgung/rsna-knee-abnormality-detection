"""Pure image-only inference entry; attach the completed private training run."""
import json
import os
from knee.runtime import competition_root, infer, prepare_codecs, training_artifacts
from knee.features import find_encoder

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
artifacts = training_artifacts()
prepare_codecs(artifacts / "wheels",
               expected_hashes=json.loads((artifacts / "wheel-hashes.json").read_text()))
infer(competition_root(), find_encoder("/kaggle/input"), artifacts / "attention/head.pt",
      "/kaggle/working/submission.csv")
