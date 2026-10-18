# Public landscape and strategy review — 2026-10-12

Author: Fable (Claude, Opus 5.5), for Sol and Astra. Read-only review: no compute, training or submission. Notebook sources and forum dumps were pulled with the Kaggle CLI into the Claude scratchpad and were not executed. Leaderboard figures are from `kaggle competitions leaderboard --download` at about 12:20 UTC on October 12.

## 1. Where 0.954 sits

| Public LB rank | Score |
| --- | --- |
| 1 | 0.966 |
| 10 | 0.963 |
| 20 | 0.962 |
| 50 | 0.959 |
| 100 | 0.956 |
| 200 | 0.954 |

5,821 teams; 898 teams at 0.954 or above, 197 strictly above. Our 0.954 ties the crowd at roughly rank 200 to 900. The top 10 needs about +0.009, and the cut has risen daily.

## 2. Our entry is the public recipe

The public 0.954 is a per-finding percentile-rank blend, fixed at 0.7 Raptor plus 0.3 S6. All eight top public notebooks attach the same three datasets:

1. `nartaa/rsna-knee-publication-swa-weights-20261007`: "Raptor", one `coatnet_rmlp_2_rw_384` SWA checkpoint, 0.949 alone. Trained on 4,349 report-labelled studies with LLM weak labels **plus OAI external knees**. Inputs: 96 slices across planes, 140 mm crop, 384 resize, 320 centre crop at inference, 3-slice windows (94 per study), per-finding attention MIL, L/R anatomical mirror TTA.
2. `heliosli/rsna-knee-compact-v1` (identical files in `compact-r6cnxfsss`): "S6", four `E_lrprior_s0_f{0..3}.pt` ConvNeXt-small folds, plus `manifest.json` and `state_spec.json`.
3. `heliosli/rsna-orthofoundation-fold0-epoch7-20261004`: the S6 code, SKM-TEA `anatomy.pt` localizer and OrthoFoundation encoder.

Our original320 arm is Raptor (`knee/reference.py:33`, `raptor_ft_coatnet_v4_full.pt`), and our S6 arm is heliosli's S6. Our fixed 70:30 entry is therefore the public 0.954, with our own capped recovery. heliosli is not our account.

**Consequence for evidence.** Both checkpoints were trained on essentially all labelled studies, so they have seen our fixed 870. The +0.0116 fixed-870 S6 gain became +0.004 public. Any local comparison involving these weights measures exposure as much as skill, and that exposure differs between arms. Fixed-870 cannot rank blends of these components, including 50:50 against 70:30.

## 3. What other public notebooks add

| Notebook | Claimed LB | Addition over the 70:30 base |
| --- | --- | --- |
| heliosli blend-gold-dataset-restored | 0.954 | none (the base itself) |
| heliosli blend-mine-in-private… | not stated | the same two arms at **0.5/0.5** |
| goodpjw2008 coatnet-2-5d-convnext-mil | 0.954 | public 2.5D ConvNeXt-Tiny reader (0.929 alone, 3 folds), per-finding weights |
| goodpjw2008 coatnet-s6-2-5d-mil | 0.954 | the same reader as a second stage; falls back to the base on failure |
| romantamrazov dinosaur-the-last | not stated | bounded rank residuals from the reader on Synovitis, Baker's, Contusion and Lateral OA only |
| evgendvorkin versia-7 | 0.954 | none (reproduction) |
| matterhorn3838 velciraptor | 0.950 | Raptor plus the reader, target-specific |
| bliverrigpdoge velci-kcd25-20 | not stated | 0.8 base plus 0.2 own "KCD25" (knee-centred compartment views, five ImageNet members, private weights) |

goodpjw2008's per-finding reader weights: Baker's 0.35, Contusion 0.30, Medial OA and ACL 0.25, Medial Meniscus and MCL 0.20, Lateral Meniscus, Fracture, Lateral OA and PF OA 0.08, Effusion 0.04, Synovitis 0.03. A flat 30% reader weight scored 0.948, so it hurt. Weights came from 58 gold studies and LB probes. Nobody in the top public set runs DINOv2/v3; a forum report finds DINOv2 adds +0.001 against a 0.002 noise floor.

Note: heliosli published the 0.5/0.5 variant with no stated public score, while the 0.7/0.3 version carries the 0.954 claim. That weakly suggests 0.5/0.5 did not beat it publicly. It is not proof: the score may simply be unreported.

## 4. Forum takeaways (Kaggle CLI `competitions topics`)

- **Validation:** over 15 submissions, weak-label CV correlated with LB at r=0.87 and gold-58 at r=0.66 (#748011). One user reports local OOF 0.937 and gold 0.938 with LB 0.961. That suggests the 0.96 band is reachable by teams validating their own models on weak labels.
- **Labels:** 25% of label cells are "not addressed", and what silence means differs by finding (#733932). An LLM matches gold at 0.878 versus 0.814 for regex. The host permits commercially hosted LLMs for report labelling (#733965). Only 58 of 4,407 studies have gold labels.
- **External data:** OAI is allowed if free and with no institutional-review requirement. The host banned KneeCoT. Whether public weights trained on OAI (Raptor) are allowed is asked in #747455 and #748276 with **no host answer**.
- **Shake-up:** the public/private split size is unstated, as is whether sites are held out. Most medal positions are held by forks of one blend, and the forum expects a large shake-up (#742327, #734681).
- **Hidden test mechanics:** compressed DICOMs need offline decoders; DataParallel hit a system-RAM OOM on the full hidden test; files arrive in random SOP-UID order, so sort by position; header reads dominate runtime. Whether all planes are always present is unanswered (#744519).
- **Licences:** whether CC-BY-NC weights are compatible with the winners' open-licence obligation is unanswered (#735121).

## 5. Answers to Astra's 12:26 questions

**(a) Balanced 50:50 versus another immediate probe.** I accept one 50:50 public probe as cheap: no new inference and only fusion arithmetic changes. Challenge the expectation, though. The +0.0047 fixed-870 result is the expected direction of exposure bias, not evidence that S6 is underweighted. S6 has a larger exposed gain than Raptor on these studies, so any exposed comparison will push weight toward S6. The public 0.954 itself settled on 0.7/0.3, and the published 0.5/0.5 variant carries no score claim. Treat the probe as a coin-flip-sized test worth at most about ±0.002. If it is run, record the outcome as a single LB observation; do not iterate weights on public LB.

The alternative cheap probe I would rank equal or higher: base 70:30 plus the public 2.5D reader as **bounded per-finding residuals on Baker's, Contusion, Synovitis and Lateral OA only**. It is a different model family with real diversity, it already has public evidence of small gains, and it adds one moderate inference arm. Its costs are runtime and one more third-party dataset.

**(b) Frozen-head control on full384 versus a head reusing S6 embeddings.** I challenge both as the main Colab spend. A head trained on frozen features from encoders that already saw the same labelled studies is stacking on in-sample features. Training-set embeddings are more separable than test embeddings, so the head learns a too-confident decision surface. There is no unexposed labelled split to validate it on. The outcome is likely to look good locally and transfer weakly, which is the same failure mode as the 870 gap. If a head pilot runs anyway, use it only as a learning experiment with LB as the judge, not as the main bet.

My preferred direction is **our own model trained on patient- or report-grouped folds, ImageNet- or MRI-pretrained, never initialized from the public competition-trained weights.** Validate it on weak-label OOF (the forum's r=0.87 proxy) plus gold-58, and add it to the public blend as a per-finding rank component at 0.2 to 0.3 (the KCD25 pattern). This is the only direction that produces unexposed evidence, adds information the crowd lacks, and also yields a no-OAI fallback for final selection. Own224/384 were closed as standalone entries; their role here is different (a diverse blend member), so the bar is "adds rank diversity on clean OOF", not "beats 0.954 alone". Concrete levers for that model:
- knee-centred or SKM-TEA-anchored crops, since S6's anchors are the one non-crowd idea already in hand;
- L/R mirror TTA;
- LLM-derived soft labels with an explicit "not addressed" mask;
- multiple Colab jobs in parallel, since units are unconstrained and wall-clock is the constraint.

**(c) Missing S6 targets.** I found no public release. heliosli's three datasets contain the four fold checkpoints, `manifest.json`, `state_spec.json`, and in the orthofoundation dataset the training code `cloud_code/training/labels.py`, `label_source_policy.py`, `fivefold_delivery.py`, `audit_label_quality.py` and `target_anatomy_audit.py`. These may let the likelihood-ratio and state targets be regenerated from the label tables. I did not verify that. Exact 4,344-study membership and the initialization hash are not published.

## 6. Final-selection hedge

Two final selections are allowed. I recommend:
- one strongest-public-LB entry;
- one entry heavy on our own model, ideally with **no OAI-derived weights**.

That hedges the expected shake-up and the unanswered OAI ruling for Raptor. Licences of each third-party dataset used in a final entry should be checked before October 22.

## 7. Sources and one concrete training hypothesis (reply to Astra 12:37)

### Sources (thread JSON pulled with `kaggle competitions topic-messages`; the API returns empty author names, so poster identity is unverified)

- CV versus LB correlation: [https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748011](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748011), posted 2026-10-10. The poster reports, for their own submissions, gold-58 AUC versus public LB at r = 0.66, and AUC on about 4,300 LLM-labelled studies versus public LB at r = 0.87. This is one participant's own submissions (15 by the survey's count), not host data.
- OAI status:
  - [https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819), 2026-09-21: "Using OAI as an external dataset does not violate the … rules if it is generally accessible to all participants without an escalated (i.e. institutional) review … and is free of charge." It reads as a host answer, but the author is unverified. A later reply in the same thread says it is a no because OAI requires institutional sign-off.
  - [https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743416) (94 votes): a participant's status table of external datasets. KneeCoT is prohibited by host ruling in [https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109). Replies from October 8 to 9 report individual OAI approval through NDA with no institution, and an NDA statement of no country exclusions. Participants say the host has not answered since.
  - [https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/747455), October 8, asks directly whether public weights trained on OAI are allowed. [https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748276](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/748276), October 11, asks whether OAI is allowed given that some eligible countries cannot register. Neither had a host reply in the dump.
- Correction to section 4: "the host banned KneeCoT" is sourced. "OAI is allowed if free and without institutional review" is a quoted reply of unverified authorship, not a confirmed host ruling.

### One changed hypothesis relative to own384 and R2

Our own models (`docs/experiments/native-detail-roi-plan-20261010.md`) start from ImageNet weights but train on the public 44-slice, 336-pixel cache with a 140 mm field. Own384 upsamples that cache, so it adds no image detail. That makes geometry, not backbone or labels, the untested factor. The native-ROI plan of October 10 was deferred in favour of public adaptation and never run.

**Hypothesis.** Anchor-centred, native-resolution compartment crops read directly from DICOM improve weak-label OOF macro-AUC over the 140 mm cache input, with backbone, initialization, labels, split and schedule held fixed. The largest effect is expected on compartment-specific targets: Lateral OA, PF OA, menisci and Contusion.

- **Single changed factor:** input geometry. The treatment uses about 100 mm crops centred by the fixed SKM-TEA localizer (`anatomy.pt`, trained on external segmentation data with no competition labels, so it is unexposed to our folds). Crops sit on the medial and lateral tibiofemoral, patellofemoral and posterior compartments, taken from native DICOM at the stored pixel spacing with no upsampling. Control: the existing 140 mm cache views, with the same number of encoded views.
- **Held fixed:** ImageNet initialization only, never the public competition checkpoints; own384's CoAtNet configuration; `train_labels_v1` with its masks; own384's fixed fold file and exclusions; seed, epochs, optimizer, augmentation and selection policy.
- **Primary endpoint:** fold-0 masked report-label macro-AUC against own384 fold 0 on the same 870 studies, plus a paired bootstrap. Neither arm is exposed, because both start from ImageNet and train on the same folds. Secondary: gold-58 descriptive, and per-target deltas on the four weakest targets.
- **Stop condition:** fold 0 must beat the control by at least +0.005 macro with the CI excluding zero, or the idea closes. If it passes, finish five folds on parallel Colab sessions.
- **Then a separate step:** one LB probe at 0.8 public 70:30 plus 0.2 new model, with weights fixed in advance; this is the KCD25 pattern. Swapping in a label change (for example `steven_v4_blend` soft labels with a not-addressed mask, the best table on gold-58 at 0.893) is a second, separately run comparison and is not mixed into this one.
- **Feasibility to confirm (parent):** anchor inference and DICOM decoding cost on Colab, and the hidden-test runtime of an added arm alongside the current entry (5.97 h base, 8.19 h conservative). If runtime does not fit, the arm would have to replace S6 in one of the two final selections rather than being added.

## Parent review — 12:40 UTC

This is Fable’s attributed research synthesis; participant score/correlation/shake-up claims are not independently established causal findings. Source inspection confirms the latest goodpjw three-reader notebook keeps its improved stage4 reader/code/weights private; a public copy falls back to the original70:30. The earlier released reader is distinct. Absence of a published50:50 score establishes no direction of performance. Section7 supersedes the categorical OAI wording in section4.

Native-ROI feasibility is accepted, with protocol qualifications in the shared handoff12:40: retain full context, match encoded views and training contracts, measure actual resampling, and replicate on a second fold before full expansion. No training, final-selection change or score forecast follows from this review.
