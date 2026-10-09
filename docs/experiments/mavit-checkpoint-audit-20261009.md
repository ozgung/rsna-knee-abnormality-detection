# MaViT checkpoint-only CPU audit

Implemented and locally tested; **the real checkpoint has not been downloaded, hashed, deserialized or executed here**. This package performs only a CPU checkpoint/architecture audit. Parent reviews, commits and decides when to launch; this task launches no job and creates no competition entry.

The source is [hengck23's MaViT288 notebook](https://www.kaggle.com/code/hengck23/mavit-288-baseline-01-1-5sec-study-1gpu) and its [public dataset](https://www.kaggle.com/datasets/hengck23/hengck23-rnsa-knee). Dataset metadata declares MIT; actual training lineage, labels, cohort and selection metric are unknown. Neither the filename `00000008.pth` nor generic MALA pretraining code establishes training provenance or accuracy. The [MALA project](https://github.com/qhfan/MALA) is the encoder attribution. Research/pre-audit evidence remains under primary ignored `outputs/analysis/post-fusion-research-20261009/`.

The notebook attaches **only** `hengck23/hengck23-rnsa-knee`, with GPU/TPU/internet disabled and the existing immutable Kaggle Docker image pinned. That dataset also contains gold CSVs; the audit never opens them. No competition attachment, images, reports, label CSVs, image forward pass, model fitting, checkpoint copy/export or pretrained download is used. No packages are installed. Python3.12, torch2.10.0+cu128 and timm1.0.26 are required; the actual preinstalled einops/numpy/torchvision versions are recorded. The pinned image and inspected import list bind the dependency environment; real compatibility is still a remote observation to obtain.

All four publisher code files are rehashed before model definitions are used. Only the five literal configuration assignments and17 architecture/function definitions are extracted; their ASTs are unchanged and independently fingerprinted. The fast reader is hashed but never imported. Executable notebook/main blocks, unused factories and imports are omitted. `register_model` is an explicit no-op hook, and `load_mala_pretrain` raises if reached; these are non-math adaptations. The original `KneeModel(pretrained=False)` constructor stays on CPU and in eval mode. A nested `torch.load` during construction is forbidden. No model method that performs a forward pass is called.

The runtime reports the actual checkpoint SHA256 and size, requiring the publisher-listed79,601,008bytes. SHA is a **first-observation result**, not a previously verified identity. It checks ZIP member metadata without extraction, rejecting unsafe paths, duplicate/encrypted/symlink members or an archive expanding beyond a1GiB safety bound. It uses exactly `torch.load(handle, map_location='cpu', weights_only=True)`, with no unsafe global allowlist, pickle retry or strictness relaxation. The file is rehashed after loading.

Every tensor in the safely loaded payload, including mapping keys and stored tensor/container attributes, is checked for CPU location, dense layout and finiteness, including any optimizer/metadata tensors. The full `model_state_dict` must exactly match the public model's keys, shapes and dtypes before `strict=True` loading. Loaded tensors must remain finite and equal the checkpoint tensors. Missing parameters cannot retain random initialization. A mismatch produces a FAIL receipt rather than a partial model.

The result is one aggregate JSON file, capped at64KiB. It records source/architecture/schema fingerprints, actual tensor/parameter counts, dtypes, mismatch counts, elapsed time, memory and environment. Known structural metadata key names/types are reported; values are never exported. Other top-level keys become hashes/types, bounded to64 displayed records plus full count/hash. Failure stage and exception type are retained without the raw exception message, file paths, training IDs or arbitrary metadata values. A PASS establishes storage/source/architecture compatibility only; it does not establish image preprocessing correctness, accuracy, training independence or ensemble utility.

## Build only

From the reviewed repository:

```bash
PROJECT=/Users/ozgun/Dev/rsna-knee-abnormality-detection
"$PROJECT/.venv/bin/python" scripts/build_mavit_checkpoint_audit.py \
  "$PROJECT/outputs/kaggle/mavit-checkpoint-audit-20261009" knee-mavit-checkpoint-audit-20261009 \
  --source-dir "$PROJECT/outputs/analysis/post-fusion-research-20261009/mavit/dataset-code"
```

Parent should review `kernel-metadata.json`, `audit-contract.json` and `build-receipt.json` before its authorized CPU launch. Retrieve only `mavit-checkpoint-audit/checkpoint-audit.json` afterward; leave the large checkpoint remote. A safe-load failure must not be converted into `weights_only=False` or an unsafe-global allowlist. Actual checkpoint identity, metadata and strict compatibility are currently unverified.

## Local verification

Nine focused tests pass: ZIP metadata safety; weights-only loading without retry; exact key/shape/dtype/finite checks; malicious-pickle rejection without executing its side effect; no pretrained load/download/forward; no gold-file reads; bounded private-metadata failure receipt; unchanged public architecture AST; CPU-only offline build and isolated import. A local public-constructor test uses explicitly test-only timm/einops import shims because those packages are absent locally; it verifies the original MALA-T stage depths/dimensions and study-head structure, not real dependency compatibility.

```bash
KNEE_REFERENCE_PARENT=/Users/ozgun/Dev/rsna-knee-abnormality-detection \
/Users/ozgun/Dev/rsna-knee-abnormality-detection/.venv/bin/python \
  -m unittest discover -s tests -p test_mavit_checkpoint.py -v
```

Final fixture suite: **271/271 PASS**, with all source fixtures enabled and no skips. Dependency-free smoke, compile and diff checks PASS; the smoke exercised no tensor forward. Log location is primary ignored `outputs/analysis/mavit-checkpoint-audit-20261009/tests-full.log`. Temporary synthetic checkpoints are removed by tests. A build-only smoke artifact under the same primary analysis directory predates the last reviewed traversal fixes and must not be launched; rebuild from final integrated source. No generated/private artifacts need preservation from the isolated worktree beyond ordinary removable `__pycache__` files.
