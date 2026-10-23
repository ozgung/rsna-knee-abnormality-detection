# MaViT checkpoint-only CPU audit

The real CPU checkpoint audit completed **PASS** on Kaggle. Parent and independent reviewer verified the authentic receipt; the checkpoint stayed remote. This package performs only a storage/architecture audit and creates no competition entry.

The source is [hengck23's MaViT288 notebook](https://www.kaggle.com/code/hengck23/mavit-288-baseline-01-1-5sec-study-1gpu) and its [public dataset](https://www.kaggle.com/datasets/hengck23/hengck23-rnsa-knee). Dataset metadata declares MIT; actual training lineage, labels, cohort and selection metric are unknown. Neither the filename `00000008.pth` nor generic MALA pretraining code establishes training provenance or accuracy. The [MALA project](https://github.com/qhfan/MALA) is the encoder attribution. Research/pre-audit evidence remains under primary ignored `outputs/analysis/post-fusion-research-20261009/`.

The notebook attaches **only** `hengck23/hengck23-rnsa-knee`, with GPU/TPU/internet disabled and the existing immutable Kaggle Docker image pinned. That dataset also contains gold CSVs; the audit never opens them. No competition attachment, images, reports, label CSVs, image forward pass, model fitting, checkpoint copy/export or pretrained download is used. No packages are installed. Python3.12, torch2.10.0+cu128 and timm1.0.26 are required; the actual preinstalled einops/numpy/torchvision versions are recorded. The pinned image and inspected import list bind the dependency environment; real compatibility is still a remote observation to obtain.

All four publisher code files are rehashed before model definitions are used. Only the five literal configuration assignments and17 architecture/function definitions are extracted; their ASTs are unchanged and independently fingerprinted. The fast reader is hashed but never imported. Executable notebook/main blocks, unused factories and imports are omitted. `register_model` is an explicit no-op hook, and `load_mala_pretrain` raises if reached; these are non-math adaptations. The original `KneeModel(pretrained=False)` constructor stays on CPU and in eval mode. A nested `torch.load` during construction is forbidden. No model method that performs a forward pass is called.

The runtime reports the actual checkpoint SHA256 and size, requiring the publisher-listed79,601,008bytes. SHA is a **first-observation result**, not a previously verified identity. It checks ZIP member metadata without extraction, rejecting unsafe paths, duplicate/encrypted/symlink members or an archive expanding beyond a1GiB safety bound. It uses exactly `torch.load(handle, map_location='cpu', weights_only=True)`, with no unsafe global allowlist, pickle retry or strictness relaxation. The file is rehashed after loading.

Every tensor in the safely loaded payload, including mapping keys and stored tensor/container attributes, is checked for CPU location, dense layout and finiteness, including any optimizer/metadata tensors. The full `model_state_dict` must exactly match the public model's keys, shapes and dtypes before `strict=True` loading. Loaded tensors must remain finite and equal the checkpoint tensors. Missing parameters cannot retain random initialization. A mismatch produces a FAIL receipt rather than a partial model.

The result is one aggregate JSON file, capped at64KiB. It records source/architecture/schema fingerprints, actual tensor/parameter counts, dtypes, mismatch counts, elapsed time, memory and environment. Known structural metadata key names/types are reported; values are never exported. Other top-level keys become hashes/types, bounded to64 displayed records plus full count/hash. Failure stage and exception type are retained without the raw exception message, file paths, training IDs or arbitrary metadata values. A PASS establishes storage/source/architecture compatibility only; it does not establish image preprocessing correctness, accuracy, training independence or ensemble utility.

## Build only

From the reviewed repository:

```bash
PROJECT=~/Dev/rsna-knee-abnormality-detection
"$PROJECT/.venv/bin/python" scripts/build_mavit_checkpoint_audit.py \
  "$PROJECT/outputs/kaggle/mavit-checkpoint-audit-20261009" knee-mavit-checkpoint-audit-20261009 \
  --source-dir "$PROJECT/outputs/analysis/post-fusion-research-20261009/mavit/dataset-code"
```

Parent should review `kernel-metadata.json`, `audit-contract.json` and `build-receipt.json` before its authorized CPU launch. Retrieve only `mavit-checkpoint-audit/checkpoint-audit.json` afterward; leave the large checkpoint remote. A safe-load failure must not be converted into `weights_only=False` or an unsafe-global allowlist. Actual checkpoint identity, metadata and strict compatibility are currently unverified.

## Local verification

Nine focused tests pass: ZIP metadata safety; weights-only loading without retry; exact key/shape/dtype/finite checks; malicious-pickle rejection without executing its side effect; no pretrained load/download/forward; no gold-file reads; bounded private-metadata failure receipt; unchanged public architecture AST; CPU-only offline build and isolated import. A local public-constructor test uses explicitly test-only timm/einops import shims because those packages are absent locally; it verifies the original MALA-T stage depths/dimensions and study-head structure, not real dependency compatibility.

```bash
KNEE_REFERENCE_PARENT=~/Dev/rsna-knee-abnormality-detection \
~/Dev/rsna-knee-abnormality-detection/.venv/bin/python \
  -m unittest discover -s tests -p test_mavit_checkpoint.py -v
```

Final fixture suite: **271/271 PASS**, with all source fixtures enabled and no skips. Dependency-free smoke, compile and diff checks PASS; the smoke exercised no tensor forward. Log location is primary ignored `outputs/analysis/mavit-checkpoint-audit-20261009/tests-full.log`. Temporary synthetic checkpoints are removed by tests. A build-only smoke artifact under the same primary analysis directory predates the last reviewed traversal fixes and must not be launched; rebuild from final integrated source. No generated/private artifacts need preservation from the isolated worktree beyond ordinary removable `__pycache__` files.


## Real CPU result

`ozgung/knee-mavit-checkpoint-audit-20261009` v1, sourceeabc93b, completed PASS, verified October9 around02:46UTC. Runtime21.150seconds; peak host RSS969,629,696bytes. Safe weights-only CPU loading succeeded. All341 model entries match: zero missing/unexpected keys, shape or dtype mismatches; strict loading and exact loaded-tensor equality passed. There are333float32 and8int64 tensors,19,861,632parameters and6,344buffer elements; all finite. Payload structural keys are only `epoch` and `model_state_dict`; the epoch value was not exported.

The first-observed checkpoint SHA256 is `7acd2a990a89c7a7d73ec0488badc5058a61b0c77142742c532ebc196749bdbc`, size79,601,008bytes. This identity is now the frozen pin for any subsequent image qualification. Environment matched Python3.12/torch2.10.0+cu128/timm1.0.26, with einops0.8.2/numpy2.0.2/torchvision0.25.0+cu128. GPU visibility was disabled. Zero image forwards or training steps occurred.

Authentic receipt: `outputs/kaggle/mavit-checkpoint-audit-20261009-result/mavit-checkpoint-audit/checkpoint-audit.json`. Independent review: `outputs/analysis/mavit-checkpoint-audit-20261009/result-review.json`. Contract SHA `372d9aba7641ea9b8a937e4a721bc91ddc39da6163c60941d6bbcec309375160`; notebook SHA `9227b11ff561d9c15edb5ab50882098e076d85b4ecf0da25f86a3e3d9777b40a`. Parent independently rechecked source/runtime/build/environment/state bindings. Decoder correctness, real image inference, predictive quality and training lineage remain unverified. The next task is a separate67-study image-only component qualification, not an immediate blend or score claim.
