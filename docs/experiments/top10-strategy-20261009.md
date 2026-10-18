# Top-10 strategy revision — October 9, 2026

The user replaces the previous >0.950 milestone with **top 10 in the competition**. Current public AUC **0.962+** is a provisional waypoint, not a guaranteed final rank or private score. Our verified best remains 0.950 from public checkpoints. No additional training or submission was launched for this research update.

## Current priority after the own-model result (2026-10-09 18:45 UTC)

Own round0 entry57016778 scored **0.878**, versus the fixed public-method best0.950. The five-fold raw/corpus and final-visible checks passed exactly, and the public result is broadly consistent with the model's descriptive gold results. This makes an undetected gross inference mismatch less likely; it does not prove every hidden input is handled identically or establish the causal source of the accuracy gap. The five report-mask validation AUCs0.879–0.890 were not evidence of a0.95-class clinical model.

1. **Finish the already-running fixed round1 comparison.** Exact bootstrap labels passed the artifact audit. Keep its recipe, folds and parameters fixed; package its actual fitted weights with explicit kernel/label binding and image-only qualification. Report its validation as biased by globalOOF teacher exposure. No recursive bootstrap expansion or gold-selected label switch. One October9 entry remains; assess round1 as a standalone comparison if qualified.
2. **Cancel the blind round0/public equal-rank fusion probe.** A0.071 standalone deficit against the0.949 public member does not support spending the remaining entry without complementarity evidence. DINO likewise stays deferred after both its standalone weakness and the measured negative blend result.
3. **Test the corrected CoAtNet geometry before full fits.** Fable owns the native224 model replacement. Require a real forward/backward smoke; then use the fixed family comparison already planned. Larger architecture or resolution may help, but neither is established as the explanation for the gap.
4. **Prioritize a controlled domain-initialization comparison over another generic backbone sweep.** The existing0.949 public MRI member is a stronger starting point than our0.878 ImageNet-initialized model. Audit compatible backbone/state mapping and public checkpoint provenance, then design a small warm-start adaptation with the same preprocessing and a fixed label/loss policy. Public all-data/OAI initialization must be a separately identified competition member, never an independent OOF teacher or clean local-validation claim. This is the next hypothesis to prepare, not a launched run or an explanation attributed to top teams.
5. **Require predictive evidence before adding cost or model count.** Use the observed round1 and CoAtNet results to choose the next controlled change. Keep the0.950 artifact intact. Top10 remains the target; there is no evidence yet that the current own-training recipe is close to0.962.

The earlier plan below is historical context. The user has since released the former600reserved studies for new CV, yielding4,345 eligible nongold/nonrelated studies; historical TRAIN04 splits remain frozen. TRAIN04 and MaViT expansion were retired after their recorded outcomes. These current priorities supersede the older pending-job and reserved-split statements below.

## Current evidence

A fresh authenticated leaderboard read shows leaders at 0.964, ranks 7–9 at 0.962, and rank 10 at 0.961 (displayed precision). The user reports our rank near 500. The threshold will move; use rank and current official results rather than treating0.962 as a permanent success criterion. [Leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard).

Concrete public clues, all **participant reports rather than reproduced experiments**:

- [October 9 report](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3535687): ConvNeXt-T 2.5D at 0.953, DINOv2-S 2.5D at 0.951, their ensemble at 0.955; pseudo-label training, author says no external data. This is the most relevant new clue for our current model families.
- [Five-fold 224px ResNet report](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3529410): 0.954. A fold ensemble of one architecture is not one independently trained checkpoint.
- [Five-fold single-family report](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3533599): 0.955 without a complete released recipe.
- [Out-of-fold soft supervision](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3526342): mix image predictions with report-derived targets; in-fold predictions were ineffective for that author. [Another author](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3529062) reports local gains that did not transfer to the leaderboard. This is a testable hypothesis, not an established universal improvement.
- [Teachers from different weak-label sources](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3527279) reportedly provide target-specific complementary supervision. No clean training lineage or OOF artifacts were supplied in the comment.

This bounded refresh did not uncover a reproducible 0.962+ recipe or establish what the current top 10 teams use. Claims about their architectures, external data, supervision or ensembles would be speculation. New topic titles about top scores and label benchmarks were found, but the CLI topic endpoint omitted their root post body; titles are not method evidence. Previously inspected Fuad0957 was a local diagnostic score, not a verified0.957 public result.

## Assessment

Closing the gap by extending public methods is plausible; reaching top 10 through only extra windows, more TTA or repeated blend-weight tuning is not a credible primary plan. Our 62-window change yielded 0.001, and the current fusion improved its strongest component from 0.949 to 0.950. Those observations do not prove an inference ceiling, but favor investing in model quality and error diversity.

Our central gap is that the scored 0.950 contains no successful own-trained model. The existing reference fit's near-prevalence behavior is a training failure to resolve, not evidence that the public model families cannot improve. The corrected global-objective fit remains pending. Existing-label CV measures a proxy; agreement with the training label source is not an independent estimate of radiologist truth. Reports may omit findings, so stronger extraction alone cannot recover all image findings.

## Revised priority

1. **Finish the active evidence.** Inspect TRAIN04 selected-checkpoint and per-target results against the exact same-split prevalence/frozen baselines. Inspect the running MaViT train2000 diagnostic; qualify a repaired standalone only after PASS. Treat MaViT as a possible complement, not the main top 10 thesis. Preserve failures and current jobs.
2. **Establish a competent trainable reference.** If TRAIN04 remains near prevalence, diagnose the learning pipeline before adding epochs, folds or a large model. Use the current 2,000/200 comparison first. Once useful, train on the eligible training pool with fixed group folds, retaining the 300 sealed studies, original development roles and gold-related exclusions. Do not silently turn held-out rows into training data.
3. **One controlled supervision experiment.** Compare current targets with explicitly defined soft image/report supervision using group-disjoint out-of-fold teacher predictions. Keep validation targets fixed and never validate against teacher-filled labels. Unknown report findings remain unknown unless a separately identified pseudo-label policy supplies training-only supervision; no blanket negative fill. Public checkpoints of unknown training lineage cannot establish clean OOF evidence. Define weight/confidence policy prospectively; no weight grid tuned on gold or public scores.
4. **One complementary trained family.** Adapt generic DINOv2-S or a comparable efficient backbone on the same folds, keeping the public-method recipe as a starting point. Evaluate per-target held-out improvement and ensemble gain, not model count or low correlation alone. More folds follow evidence; larger resolution or anatomy-specific views are later controlled tests, not assumed top-team secrets.
5. **Promote on evidence.** Require reliable optimization and useful own-model validation before spending on expansion. Seek a 0.953–0.955-class standalone/complement as an intermediate competitive milestone, not a forecast. Then test a small predeclared ensemble. Top10 remains a stretch objective with no justified success probability or predicted score today.

No new arbitrary runtime or GPU reserves. Use existing authorized compute and actual platform limits. Preserve image-only offline rehearsal, target/ID schema checks, provenance restrictions and failure handling. No raw external data acquisition or new terms acceptance follows from this strategy.

## Evidence and verification

Fresh CLI leaderboard, active/recent topic lists and selected discussion snapshots are saved under ignored `outputs/analysis/top10-strategy-20261009/`. CLI leaderboard output includes a pagination prefix; topic lists include warning/suffix text. JSON was parsed with raw decoding rather than assuming the whole output is JSON. Direct web browsing corroborated discussion availability but did not expose newer root posts. Prior inspected public-source evidence remains linked in `outputs/analysis/post-fusion-research-20261009/research-note.md`.

This is a documentation and planning change: content/link checks and `git diff --check`; runtime tests not applicable. Existing running jobs were not altered. The previous >0.950 objective is superseded, while historical experiment decisions and results remain unchanged.
