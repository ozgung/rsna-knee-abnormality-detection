# Public notebook follow-up — 2026-10-08

Read-only research; no model executed, compute provisioned or submission made.
Verified live Kaggle Code tab sorted by Public Score, with its default
`excludeNonAccessedDatasources=true` filter. The highest displayed accessible
entries were 0.949; this is not a claim about private submissions or every notebook.

| Notebook | Displayed public score | Evidence and relevance |
| --- | --- | --- |
| [Danial Zakaria: Anatomical Mirror](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror) | 0.949, v2, script356175397 | Inspected introduction and inference source. One CoAtNet checkpoint averaging epochs14/11/12; five sequence slots,96 slices,94 adjacent-slice windows;384-trained backbone evaluated on320 center crops with equal original/anatomical-mirror probability averaging. Per-finding attention. |
| [Matterhorn3838: Velciraptor DINOsaur Speed](https://www.kaggle.com/code/matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed) | 0.949, v6, script356180772 | Inspected source/logs. Despite the title, loads the same single CoAtNet checkpoint SHA7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef. Not independent evidence for another architecture. |
| [pjmathematician: D4 blend](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-blend) | 0.946, v3, script353440703 | Inspected runtime contracts and stack structure: DINO/A5/RadImageNet/Raptor/CoAtNet components and depth-zone adapter. Some input datasets displayed Private Dataset, so runnable completeness is unverified. Intro contains older0.943 claims; use live score badge. |
| [pjmathematician: D4 lite](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-lite) | 0.945 | Score-ranked list only; implementation not inspected in this pass. |
| [goodpjw2008: Stack + 2.5D ConvNeXt MIL](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944) | 0.944, v2, script355300588 | Inspected reader preprocessing/model source. Six sequence slots, adjacent-slice windows, canonical physical orientation/spacing, shared ConvNeXt-Tiny, series/position tokens, study transformer and per-finding pooling. Author reports standalone3-fold0.929;70% public stack/30% own reader yields displayed0.944. Standalone score and training history not independently reproduced. |
| [Mattia Angeli: Speedy Raptors](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-speedy-raptors-the-original) | 0.943 | Verified score-ranked list; broader public stack lineage previously inspected through related source. |
| [Pilkwang baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) | 0.891, v15 | Previously verified live source/logs; foundational20-member partly tuned DINO ensemble, not the highest public notebook. |

## Interpretation and limitations

- The0.949 release explicitly says Gold58 was reused for selection/development;
  it is not independent CV. It acknowledges controlled-access OAI training data.
  Inference source and weights are public, but full training reproducibility and
  external-data/weight eligibility still require an audit before reuse. Its old
  opening docstring describes a different recipe; the live constants and manifest
  govern the current run. Do not infer mirroring alone caused the score gain.
- The ConvNeXt author reports14 training epochs at256px, report-grouped folds and
  averaging four public weak-label tables. Gold excluded from fitting does not
  establish independence of upstream labels. The page reports6.5–8h submit-to-score
  for the full stack and roughly20min for its3-checkpoint reader; these are author
  observations, not our runtime measurements. It also reports0.943/0.944 variation.
- A [participant discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148)
  self-reports ResNet224px/24-slice single-fold0.935 and five-fold0.94, with stronger
  augmentation and longer training. This supports investigating efficient trained
  CNNs, but is not an independently verified open-notebook reproduction.
- Current evidence does not support attributing our0.757 gap solely to resolution,
  ensemble size or labels. Encoder adaptation, richer sequence coverage and more
  training data differ simultaneously. A strong single model is a better next
  experiment than more head seeds over the same frozen features.

## Revised next action

REF-02: audit the0.949 CoAtNet release's checkpoint provenance/permissions and
reproduction requirements, alongside the independently trained ConvNeXt reader.
Choose an auditable offline reference; retain Pilkwang as a simpler fallback.
TRAIN-02: compare a bounded fine-tuned2.5D CoAtNet/ConvNeXt or partial-DINO model
with our frozen baseline, prioritizing multi-sequence coverage and adjacent slices.
Prespecify held-out selection and compute; add anatomy-aware TTA only as a separate
comparison. Public leaderboard scores are context, not our selection set.
