# Public0.949 mirror reader qualification

October9,2026. Implemented and synthetic-tested; **real checkpoint/CUDA/67-study rehearsal and competitive performance are unverified**. The parent accepts the publicly accessible model interpretation of competition rule2.6 and the publisher's explicit competition permission. This is not explicit host or prize-eligibility clearance; the OAI-lineage uncertainty remains documented in `outputs/analysis/strategy-refresh-20261009/research-note.md`. No raw OAI acquisition is involved.

The pinned [nartaa source](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) and [accuracy checkpoint](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007) remain exact:96stored384px frames,26/22/18/12/18slots,2–98%series sampling,140mm physical crop, native320center crop,94triplet windows, ImageNet normalization, equal original/anatomical-mirror probabilities, CUDAfloat16, sequential backbone chunks16. Sagittal mirror reverses triplet channels; other planes flip image width. No change to resizing, alternative series retry, model blend, fp32 retry or fabricated prediction is introduced. Original public short-series repeated sampling and cross-slot windows remain.

The accuracy file is `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`,292840787bytes; publisher SHA256`7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`. The builder requires the saved publisher manifest and binds its exact bytes. The runtime rehashes the actual mounted checkpoint before `torch.load(weights_only=True)`, verifies architecture/resolution/target order, finite tensors and strict state keys. No weights are downloaded to the Mac. The original manifest's source hash describes its publisher submission; our extracted source snapshot and decoded payload have independent existing pins.

`mirror-study-lut-partial-v1` preserves successful public math while distinguishing handled partial inputs from resource failures. The original two public catches reuse the reviewed Raptor guard. All20pinned decoder catch regions receive only a fatal-resource prefix, retaining the existing header-fd cleanup before rethrowing. Inverse-AST regression restores byte-parsed scientific statements exactly. Missing geometry retains the public InstanceNumber/spacing defaults; unreadable/black selected frames and empty/missing slots retain public masks. Whole-empty studies, inconsistent masks and unsafe pixels fail. MemoryError,TimeoutError and OpenCVStsNoMem propagate through the whole catch regions. The existing Raptor/ConvNeXt blend files and scored packages are unchanged.

The rehearsal selects64training study IDs by SHA256(seed:ID), plus the three visible test IDs in official order. It does not read reports or labels. Every native study_lut volume/mask must equal a separate public pydicom read; strict healthy comparisons are counted separately from source-compatible partial inputs. Both original and mirrored CPU/GPU windows must match exactly, and repeated probabilities within1e-6. Seventeen synthetic production-path cases cover healthy, corrupt, black, missing geometry/series, short series and fatal error paths. Receipt qualification requires current policy/source bundle/checkpoint, exact recipe/environment, stress PASS, cohort fingerprint/seed, per-study windows/repeats and explicit strict-coverage counts; old24-study receipts cannot qualify.

Use the integrated repository for the authoritative package:

```bash
.venv/bin/python scripts/build_reference_rehearsal.py \
  outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py \
  outputs/kaggle/mirror-rehearsal-20261009 \
  knee-mirror-rehearsal-20261009 \
  --profile mirror --train-count 64 --seed 20261008 \
  --publisher-manifest outputs/analysis/strategy-refresh-20261009/mirror-current/checkpoint-manifest.json
```

The command only builds. Metadata is private, GPU-enabled, internet-disabled, with competition plus the nartaa weight dataset and the existing ConvNeXt dataset solely for three hash-pinned Python3.12codec wheels. No training notebook, report-label dataset or other model is attached. The same Docker/Python3.12/Torch2.10.0+cu128/timm1.0.26environment as the qualified ConvNeXt reader is required. Default deadline is the actual9-hour competition ceiling; sample timing extrapolations are informational, never an extra spending veto. Runtime receipts record setup, sequential inference timing, host/CUDA memory and disk. Parent owns launch and any later standalone entry; no fusion is packaged.

Fixture-enabled checks (OpenCV is test-only under `/tmp/coverage-inference-test-deps`; it is not a repository dependency change):

```bash
PYTHONPATH=/tmp/coverage-inference-test-deps \
KNEE_REFERENCE_PARENT=~/Dev/rsna-knee-abnormality-detection \
KNEE_REFERENCE_SOURCE=~/Dev/rsna-knee-abnormality-detection/outputs/research/2026-10-08/mirror-audit/rsna-knee-0949-anatomical-mirror.py \
KNEE_RAPTOR_SOURCE=~/Dev/rsna-knee-abnormality-detection/outputs/research/2026-10-08/raptor-audit/knee-mri-twelve-findings-from-a-single-model.py \
~/Dev/rsna-knee-abnormality-detection/.venv/bin/python -m unittest discover -s tests -v
```

Local verification:230/230fixture-enabled tests PASS, default smoke PASS (no backend forward), source compilation/diff checks PASS;7focused mirror tests PASS, including the actual pinned decoder on synthetic DICOM files and header-descriptor cleanup after resource failure. Full test log is preserved in the primary workspace at `outputs/analysis/mirror-rehearsal-20261009/tests.log`. Actual checkpoint compatibility, compression diversity, CUDA parity/runtime, visible predictions and any score remain pending the parent-run qualification. No commit, remote launch, upload or submission was performed by this implementation task.

## Actual GPU qualification

`ozgung/knee-mirror-rehearsal-20261009`v1, source2a92be1, completedPASS. Authentic receipt downloaded and validated:67/67 strict pixel comparisons PASS; CPU/GPU windows and within-run repeats exact;17synthetic stress cases PASS;6,090unique slices decoded with zero pixel/resource errors. Application elapsed489.952s, peakCUDA3,300,294,656bytes and hostRSS2,527,678,464bytes. Sequential p95 projection for1,300studies is4,377.729s (~73minutes), informational and not a hidden-runtime guarantee. Sixteen missing slots retained public masks. Results: `outputs/kaggle/mirror-rehearsal-20261009-result/reference-rehearsal/`. Standalone visible parity remains a separate qualification; no new public score.
