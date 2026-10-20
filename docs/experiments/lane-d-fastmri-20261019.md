# Lane D — fastMRI noisy student (Oct 19–20)

Owner: Claude (Lane D subagent), branch `claude/laned-20261019`. Plan: [final-push plan §2 Lane D](../research/final-push-plan-20261019.md).
Status labels: **implemented**, **tested on synthetic data**, **tested on real data**, **competitively evaluated**.
No fastMRI pixels, file names or links are in Git; every fastMRI artefact stays in ignored `outputs/laned-20261019/`, on the
VM, or in private Kaggle datasets. Times are UTC.

## Decision and terms

- User decision (Oct 19): "Run Lane D"; fastMRI-derived student weights may enter a submission under NYU's data-use terms.
  This covers Lane D only.
- Data: NYU fastMRI knee DICOM batch 2 (the clinical DICOM batch, not k-space), fetched once onto the Colab VM from the
  user's signed links (kept in a mode-600 file, never echoed or logged); archive SHA256 matched the official list.
- Teacher inference ran in a private Kaggle kernel on a private dataset holding the selected exams in the competition
  layout. Both teacher datasets are deleted after use.

## Gates

| Gate | Result | Time |
|---|---|---|
| Third concurrent Colab A100 session | allowed (B and C held two) | 18:35 |
| Header → plane / fluid classifier ≥ 98 % on 24,371 competition train series | PASS: both 98.84 % (plane 98.95 %, fluid = fat-sat 99.89 %), GroupKFold(5) by study | 18:40 |
| Batch-2 download complete by Oct 20 02:00 | PASS: 30.57 GB, SHA256 matches the official list | 19:17 |
| Cache audit (50-study sample, fail-closed) | PASS (see below) | 19:55 |

## Data

- Batch 2: 2,185 exams / 9,361 series read (headers only); 2,141 usable (sagittal + coronal + a fluid-sensitive series,
  ≥ 3 series, localisers / derived / < 10-file series removed). On fastMRI the classifier's plane agrees with the
  ImageOrientationPatient plane on 99.99 % of series (the IOP plane is used); one series has an uncertain fat-sat probability.
- Selected: 1,600 exams by a content-free hash order (6,915 series, 4.3 per exam). DICOMs are JPEG 2000 lossless
  (decoded with pylibjpeg-openjpeg 2.6.0, the version P2's image installs).
- Pixel cache: the unchanged s6v2 builder (compact_infer `CStudies.rows`, v4 / 320 px / 3 slices / 3.5 mm, lossless PNG
  rows) on local files: 1,600 / 1,600 studies, 0 errors, 710,660 rows, 25 min.
- Audit: store finalised, all studies disposed, 50-study sample with ≥ 2 planes and fluid rows each, no blank rows,
  intensity in range, 5 / 5 re-renders byte-identical; median rows per study 398 vs 564 on the competition cache
  (gate ×0.5–×2), blocks 4 vs 5.

## Teacher

P2's notebook (57340072, R6F-pure v2, sha 3af761f6…) with its frozen payload byte-identical; only the bootstrap's test-root
list points at the private fastMRI dataset (`scripts/laned/build_teacher_kernel.py`). Arms: R384 two-view (`old`), R224
(`crop224`), R6 flip; only exams with all three arms genuine are used.

- Kernel `ozgung/laned-fm-teacher-p0-20261019` v1 (2×T4), queued 19:51, ran 20:40–02:09, 19,272 s, status
  PASS_WITH_FALLBACK_AFTER_ERROR. R384 800 / 800 genuine (6,546 s), R224 800 / 800 (3,715 s), R6 flip 745 / 800: the
  R6 stage died near the end with a FileNotFoundError in compact_infer's temporary geometry folder (board 0 at about
  380 / 400); its flushed CSV rows were salvaged exactly as P2's entry salvage does. R6 on fastMRI: about 18.5 s per study
  per board (JPEG 2000 decode is most of it).
- **Exams used: 745** (all three arms genuine).
- Calibration (deviation from "probabilities, no ranks", decided before any student result): the R384 / R224 outputs are
  not probabilities (fastMRI means 0.35–0.57 for every finding, MCL and fracture included; R6 medians are near 0), so a
  probability mean would raise every rare-finding target. The fastMRI target is P2's own arithmetic on per-arm percentile
  ranks, score_f = w_f (0.65 r_R384 + 0.35 r_R224) + (1 − w_f) r_R6, quantile-mapped onto the competition pool's labels-v3
  marginal per finding (the same calibration labels v3 applies to its R6 teacher). This assumes the fastMRI finding
  prevalence equals the competition's. Arm Spearman on fastMRI: R384 vs R224 0.88–0.98, R384 vs R6 0.57–0.94.

## Student

S6 recipe (`vm_train.sh`: ConvNeXt-small, 16 epochs, batch 2, seed 42, all data) over the competition pool
(labels v3, Lane B READY 22:09, gold-58 0.9254) plus the fastMRI exams. Arm `E_fm=bce:prior:pw`: the same weighted BCE as `E_prior`, with cell
weight 1 on competition rows and a fastMRI weight solved so that, under the trainer's per-batch weight normalisation
with batch 2, a fastMRI study carries an expected 0.5 of a competition study's gradient weight. Competition rows are read
from the restored, preflight-verified cache (unchanged files); fastMRI rows through `--px-override`.

RESULTS_STUDENT

## Deviations from the plan

- Teacher inference ran on Kaggle (P2's own image and code) instead of the Colab A100, because P2's controller is
  pinned to two isolated T4 boards; Colab kept the cache build and the fit.
- Kaggle allows two concurrent batch GPU sessions and another lane held one, so only teacher part 0 (800 exams) ran;
  part 1 was dropped (its dataset deleted) to keep one Kaggle slot free for Lanes A and C and to start the fit early.
- The two teacher datasets (about 22 GB) exhausted the account's private-dataset quota for a while, which blocked Lane C's
  per-epoch checkpoint versions; part 1 was deleted at 20:55 and part 0 after its kernel.

## Reproduction

Scripts: `scripts/laned/` (classifier, fetch, headers, selection, layout, local cache build, audit, merge, targets,
teacher kernel, fit, checkpoint backup, solo builder). Local evidence: `outputs/laned-20261019/` (classifier report,
selection summary, audit receipts, teacher outputs, targets receipt).
