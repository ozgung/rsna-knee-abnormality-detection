# Pipeline review and visual briefing — October 11, 2026

User-requested synthesis of current results, tuning variables, top-ten barriers, localized crops, DICOM/MRI geometry and pretrained MRI models. Deliverable: [29-page presentation-style PDF](../../outputs/briefings/knee-pipeline-review-20261011/knee-pipeline-review.pdf). Reproducible authoring source and private validation evidence are beside it in ignored `outputs/briefings/knee-pipeline-review-20261011/`. The PDF includes existing ordinal-only MRI QA figures, aggregate charts, target support, source links and limitations. No raw images, identifiers, reports, credentials or weights were added to Git.

## Evidence snapshot

Official CLI refresh around **23:06 TRT (20:06 UTC)**: best public score **0.950**, entry 56986393; Full384 entry 57083497 still **PENDING**. The tenth displayed leaderboard row is now **0.963**, compared with older notes targeting roughly 0.961–0.962. Scores are rounded and ties occupy several rows; this is a moving public reference, not a final top-ten guarantee. The S6 v3 job is described only by the existing last-recorded running checkpoint; this review did not collect or audit its terminal results.

Official pages rechecked: macro ROC-AUC over the exact 12 sample-submission targets, offline CPU/GPU notebook runtime ≤ 9 h, approximately 1,300 hidden studies, entry/merge October 15 at 23:59 UTC, final October 22 at 23:59 UTC. Istanbul equivalents are October 16 and October 23 at 02:59 TRT. Local CSV aggregate check: 4,407 studies, 24,371 series, 3–14 series/study with median 5. API page/submission/leaderboard responses and a compact snapshot are retained under the briefing's `private/` directory. Initial sandbox DNS failure and a CLI pagination banner were handled before reading valid results; no failed query was treated as evidence.

The substantive results remain:

- [S6 fixed comparison](../experiments/s6-fixed-evaluation-result-20261011.md): original320 0.951330909 versus fixed 70:30 rank blend 0.962963989, delta +0.011633080, paired 95% CI [+0.009787471,+0.013714779], all 12 positive. Strong exposed development signal, not a public 0.963 score. The intervention combines readers, sampling, local views and fusion, so it does not isolate crop benefit.
- Full384: [fold0](../experiments/public-fullfield-result-20261011.md) delta +0.002646205 and [fold1](../experiments/public-fullfield-replication-result-20261011.md) delta +0.001816829, both intervals positive. Same learned checkpoint with changed retained field/grid. Replication does not remove public all-data checkpoint exposure.
- [Public adaptation](../experiments/public-adaptation-result-20261011.md) rejected at 0.948112635. [80/10/10 public/own blend](../experiments/public-own-fixed-blend-20261011.md) rejected at 0.948295575. [Own224](../experiments/coatnet224-fullfive-20261010.md) public 0.914. These reject particular recipes, not all future own training.

## Review judgment and tuning order

Accept the current fixed S6 image-only rehearsal as the immediate execution priority. Retain the qualified whole-population ranking contract and the shared decision against untested per-study or deadline substitution. Sol remains the execution owner; this review launches no training, new entry, download or runtime change.

Priority after that evidence: (1) anatomical coverage and useful reader diversity; (2) training supervision aligned with image truth and independently evaluated; (3) sequence/depth sampling and masked fusion; (4) a small prespecified ensemble comparison using valid OOF evidence; (5) domain-aware initialization or backbone change; (6) resolution/TTA/optimizer changes driven by a concrete failure. This is an ordinal recommendation, not an estimated AUC gain table.

The immediate operational roadblock is complete offline inference: S6 took21,731.24 s on 870 studies on one T4. Linear scaling to 1,300 is 9.02 h for S6 alone. An ideal two-board half is 4.51 h, before original-arm work, overhead and shared-CPU/tail effects. Neither estimate qualifies a hidden run.

The main scientific roadblock is trusted supervision and generalization. Only 58 studies have official structured truth. Reports omit findings and differ from the image-based rubric. Existing public checkpoints, gold-guided choices and global OOF teachers have exposure that local fold exclusion does not undo. Unknown patient identities further weaken independence. The [label audit](../experiments/label-benchmark-20261009.md) found simple table averaging inconclusive and a copied-gold table. More compute/backbone size alone does not fix these problems.

## Localized crops

The idea is promising and partly present in S6. The source's v4 views preserve a global view and add 100 mm joint/PF/posterior locals or coronal medial/lateral locals, followed by the retained 352/384 field. The [32-study/179-series review](../experiments/s6-anatomical-coverage-review-20261011.md) passed a gross visible union screen, not clinical localization or all-depth population coverage. The [fixed 90 mm center prior](../experiments/native-crop-qualification-20261011.md) visibly clipped anterior/patellar anatomy and remains rejected.

Candidate target regions: ACL course, medial MCL corridor, separate meniscal/osteoarthritis compartments, patella/trochlea, and posterior Baker's region. Effusion/synovitis and bone findings need broad context. These are design hypotheses, not clinically validated crop boundaries. Sagittal compartment separation is chiefly a depth-position problem; a left/right pixel split is not sufficient.

The proposed matched follow-up uses 24 views in every arm: A = 12 common global + 12 extra global; B = 12 global + 12 same-location local crops from a cache; C = same boxes/positions as B but source-DICOM locals. B minus A tests locality/magnification; C minus B tests retained source detail. Hold backbone, initialization, labels, split, optimizer, evaluation budget and endpoint rule fixed. Freeze an image-only localizer after outcome-blind containment review. Require paired macro gain, class support, no material target harm, failure/spacing strata and measured runtime, then second-fold replication. This proposal does not authorize changing the current fixed S6 recipe or restarting the rejected 90 mm prior.

## DICOM/MRI and transferable models

DICOM carries pixels and acquisition/spatial metadata. A study contains series, which contain instances or frames. Reconstructed 3D arrays need spatial ordering and spacing; NIfTI can preserve an array plus an affine. Slice count, stored matrix, contrast, field, thickness/gap and orientation vary. Series may be oblique and are not automatically registered. Resampling cannot recover unacquired detail. Three-channel adjacent-slice input is 2.5D, not full-volume 3D. Reconstruction models operating on raw k-space solve a different problem from this classifier. Primary references: [DICOM geometry](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html), [NiBabel orientation](https://nipy.org/nibabel/dicom/dicom_orientation.html), [SimpleITK](https://simpleitk.readthedocs.io/en/master/fundamentalConcepts.html), [NIfTI](https://nifti.nimh.nih.gov/nifti-1/), [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/) and [fastMRI](https://github.com/facebookresearch/fastMRI).

Verified primary-source model leads, no execution:

- [MedicalNet](https://github.com/Tencent/MedicalNet): released 3D ResNets pretrained on medical volumes, MIT repository. Candidate baseline, not demonstrated competitive knee transfer.
- [RadImageNet](https://github.com/BMEII-AI/RadImageNet): released 2D medical CNNs including MRI pretraining. Easier integration with the present slice pipeline.
- [3DINO](https://github.com/AICONSlab/3DINO): released 3D self-supervised encoder. Repository explicitly states CC BY-NC-ND restrictions; resolve permitted adaptation/reuse before considering an entry.
- [BrainIAC](https://github.com/AIM-KannLab/BrainIAC): released brain MRI encoder; substantial anatomy mismatch and non-commercial academic terms.
- [OrthoFoundation-MSK-5M](https://github.com/ytrsk/OrthoFoundation-MSK-5M): the official repository lists a released DINOv3-L 2D checkpoint continued on musculoskeletal images including knee MRI. Expanded 5M-image release differs from original 1.25M-image paper corpus. Most targeted new initialization lead, but actual checkpoint download/load, release provenance, license and competition eligibility are unverified. Next step is a small metadata/source audit, not a full weight download or training launch.
- [KneePreM](https://arxiv.org/abs/2609.31461): September 2026 preprint describing 3D knee masked-autoencoder pretraining on 19,011 OAI series. Relevant classification/segmentation transfer research; usable public checkpoint not verified from the inspected abstract.

Prior localization evidence comes from the authors' [RSNA 2024 second-place component](https://github.com/brendanartley/RSNA-2024-Competition) and [third-place solution](https://github.com/Moyasii/Kaggle-2024-RSNA-Pub). Both separate localization from classification. They support a hypothesis, not a demonstrated knee gain; spine coordinate supervision is not present in our official knee tables.

## Verification and limits

The PDF's numeric charts use saved audited aggregate results and header counts. The target appendix is generated directly from the independent S6 audit JSON. Existing image provenance is linked in the PDF, and no new diagnostic annotation is asserted. All 29 pages were rendered and visually reviewed. Eleven content, local-link, extraction and table/text overlap checks passed; the receipt is `outputs/briefings/knee-pipeline-review-20261011/private/verification.json`. This is documentation/research work: unit/backend tests and GPU runs are unnecessary and were not rerun. No independent model retraining, raw-image inference, hidden runtime audit, new foundation-model checkpoint verification or competitive evaluation occurred in this chat.

Search scope was a bounded primary-source scan for MRI formats, available pretrained encoders, knee-specific transfer and localization methods. No systematic-review completeness or knowledge of undisclosed top-ten solutions is claimed. Shared coordination is append-only in [the handoff](../claude-astra-handoff.md).
