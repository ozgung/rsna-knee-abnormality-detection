# Cloud research brief: knee MRI abnormality detection, 2026-10-14

A self-contained brief for a cloud research session. The question is in the "Questions" section at the end.

## Competition

- **Competition:** Kaggle *RSNA Knee Abnormality Detection*.
- **Task:** study-level prediction of 12 findings from knee MRI.
- **Metric:** macro AUC over the 12 findings.
- **Inference:** image-only, offline, 9 h limit on 2×T4 for a hidden test of about 1,300 studies.
- **Training data:** about 4,345 labelled studies.
  - Labels are report-derived weak labels, soft and masked where a report does not mention a finding.
  - A small gold set of 58 studies has radiologist labels.
- **The 12 findings:** ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA (patellofemoral), Effusion, Synovitis, Baker's cyst, Contusion (bone bruise), Fracture.
- **Inputs:** each study has several series. Typically:
  - sagittal PD/T2 fat-suppressed and non-fat-suppressed;
  - coronal PD/T1, with or without fat suppression;
  - axial PD fat-suppressed.
- **Frame-of-reference caveat:** the anonymiser regenerated FrameOfReferenceUID per series, so cross-series registration cannot rely on the UIDs. The IPP/IOP geometry itself is probably still real.

## Our best own model (a 2.5D MIL classifier)

- **Architecture:** a CoAtNet-384 backbone, ImageNet init, fed 3-slice windows. Each window is three adjacent slices treated as RGB.
- **Slots:** each study is resampled into 64 slots across series, split 18/14/12/8/12. Slices are taken at 6–94% of each series' extent and resized 336→384.
- **Pooling:** attention multiple-instance pooling per finding.
- **Training:** K8 windows sampled per study in training, K42 windows at evaluation; 12 epochs, top-3 SWA.
- **Validation:** grouped 5-fold split, with fold-0 development of 870 studies. The development truth is the weak labels, masked.

### Per-finding fixed-870 AUC, K42

`in-distil` is the same model trained with 0.5 × soft labels from a stronger public teacher (distillation).

| Finding | in-orig (ImageNet) | in-distil | Positives / observed |
|---|---|---|---|
| ACL | 0.936 | 0.946 | 109 / 825 |
| MCL | 0.878 | 0.891 | 20 / 810 |
| Medial Meniscus | 0.943 | 0.942 | 330 / 826 |
| Lateral Meniscus | 0.862 | 0.878 | 109 / 822 |
| Medial OA | 0.927 | 0.935 | 150 / 812 |
| Lateral OA | 0.840 | 0.860 | 75 / 811 |
| PF OA | 0.848 | 0.864 | 169 / 815 |
| Effusion | 0.917 | 0.921 | 222 / 817 |
| Synovitis | 0.943 | 0.942 | 103 / 440 |
| Baker's | 0.946 | 0.949 | 113 / 547 |
| Contusion | 0.912 | 0.919 | 165 / 799 |
| Fracture | 0.930 | 0.932 | 52 / 794 |
| **Macro** | **0.907** | **0.915** | |

## What we learned

1. **Plane ablation (E1).** We dropped each plane group's windows at inference.
   - **Dropping sagittal:** macro −0.028. The losses are ACL −0.122, Lateral Meniscus −0.103 and PF OA −0.041; Lateral OA moves +0.007.
   - **Dropping coronal:** macro −0.007, MCL −0.037.
   - **Dropping axial:** macro −0.003, inside noise.
   - So PF OA barely uses the axial plane, even though axial is the classic plane for the patellofemoral joint. That points to an untapped signal.
2. **Prediction correlations far exceed label co-occurrence.** Examples are Medial vs Lateral OA at 0.91 and Contusion vs Fracture at 0.92. The model confuses compartments and finding types: it learns a "sick knee" factor more than localised findings.
3. **Closed or null approaches:**
   - a plane/sequence-aware head token (null);
   - stacking (null);
   - LLM relabelling of the reports (no gain);
   - a mirror augmentation that swapped Medial and Lateral labels, which was wrong. A left-right flip changes apparent laterality, not anatomy.
4. **Native-resolution ROI crops** around the joint use a SKM-TEA segmentation model, which is sagittal-only. They are in progress, with a containment problem at the crop edges.
5. **The public state of the art reaches about 0.957 public.** It combines:
   - a CoAtNet trained with external OAI knee data;
   - heliosli's anchored local-view ConvNeXt models (R7/S6), which use high-resolution local views around anatomical anchors plus side and laterality models;
   - stackers for Effusion and Baker's.
   Top teams reach 0.963–0.966 with unknown private methods.

## Questions

1. **Research GitHub and the literature on knee MRI abnormality detection.** Cover MRNet and its successors, fastMRI+ and SKM-TEA, OAI and MOAKS/WORMS-based models, meniscus and ACL tear detectors, bone-marrow-lesion and effusion grading, and any RSNA 2026 knee competition write-ups or code. Report:
   - which architectures and input strategies work best per finding;
   - which public datasets and weights exist.
2. **For each of the 12 findings:**
   - **How a musculoskeletal radiologist diagnoses it:** which sequences, planes and anatomical locations, what signal pattern, and which secondary signs. Examples: meniscal tear signal reaching the articular surface on two or more slices, the ghost meniscus, MCL oedema or discontinuity on coronal, cartilage loss with osteophytes and subchondral changes for OA, joint fluid distension of the suprapatellar recess for effusion, and so on.
   - **Whether it needs true 3D context or cross-plane association,** or whether a single slice or slab is enough.
   - **How our method could imitate that reading:** targeted crops, specific planes or sequences, slice-consistency constraints, per-finding heads with plane gating, compartment-aware pooling, auxiliary tasks such as segmentation or cartilage-thickness proxies, or ordinal or anatomical priors.
3. **Which findings have the most headroom, and why.** Weigh the current AUC, the positive count, label noise and how hard the finding is clinically.
4. **Low-hanging fruit,** given our constraints:
   - about 8 days left;
   - plenty of A100 time for training;
   - a 9 h 2×T4 inference budget, mostly already used by the public arms;
   - image-only inference.
   Rank concrete, testable changes by expected gain per day, with a pre-registered single-factor test against `in-distil` on fold-0.

**Output:** a structured report with a per-finding table (finding | radiologist reading | 3D needed? | method idea | expected headroom) and a ranked list of experiments. Cite sources with links. Label claims [measured], [literature] or [estimate].
