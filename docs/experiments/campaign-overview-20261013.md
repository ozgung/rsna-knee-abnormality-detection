# RSNA knee campaign overview — October 13, 2026

Snapshot prepared at 16:35 TRT (13:35 UTC); submission scores were freshly retrieved through the authenticated official Kaggle API. Later training recovery/qualification actions may still be running. This document is a history summary, not a new experiment protocol or a claim of top-ten placement.

There are **18 competition submissions: 12 scored and 6 hidden-rerun failures**. The best public score is **0.954** from released public models. The best submitted model family trained by us is **0.914**, original-label five-fold CoAtNet224. Our own weights still trail the strongest released knee models.

Public AUC is Kaggle's public test score. Development AUC is against weak/report-derived labels on reused study/report folds; gold58 is a small, previously used development set. These values are not interchangeable. Public knee checkpoints can have trained on our development studies. AUC measures ranking, not percent correct. ImageNet and DINOv2 pretraining in our fresh models is general natural-image pretraining; nartaa/Dread/S6 checkpoints are already knee-specific trained models.

The table lists all actual competition entries, including operational failures. Public origin means we used released weights/source without training a new MRI model for that entry; all entries were wrapped or qualified for offline image-only execution. Our trained models also reuse public architectures/training recipes, so “own” means our fitted weights, not an invented architecture.

## All competition submissions

| Date/time (TRT) | Submission | Model / entry | Public AUC | Origin | What changed | What was tested |
| --- | --- | --- | ---: | --- | --- | --- |
| Oct 08 01:13 | 56922773 | Frozen DINOv2 + attention head | 0.756 | Our head; generic DINOv2 features | First image-only baseline; masked weak-label loss | Raw/cache gold parity; repeatability; finite ordered CSV |
| Oct 08 01:36 | 56923149 | Four-head DINOv2 ensemble | 0.757 | Our heads; generic DINOv2 features | Attention seeds42/43/44 + mean/max42; equal probabilities | Weak-BCE-only selection; repeated raw-DICOM parity |
| Oct 08 02:50 | 56924358 | ConvNeXt three-fold reader | No score: rerun error | Public goodpjw checkpoint; no retraining | Standalone audited inference wrapper | 19-study offline reader/probability parity; hidden rerun failed |
| Oct 08 03:13 | 56925712 | 50:50 ConvNeXt + plain Raptor | No score: rerun error | Two public model families; no retraining | Fixed per-target rank blend, K42 Raptor | Offline same-environment and blend parity; hidden rerun failed |
| Oct 08 04:37 | 56927932 | ConvNeXt partial-input repair | 0.929 | Public goodpjw checkpoint; no retraining | Preserve usable partial series; mask unusable slots | 67-study/seven-stress checks; healthy output unchanged |
| Oct 08 05:18 | 56929332 | Repaired 50:50 public blend | 0.936 | Two public model families; no retraining | Source-compatible partial inputs; fatal resource propagation | 67-study component/stress and repeated blend checks |
| Oct 08 20:15 | 56964291 | Public blend with Raptor K62 | 0.937 | Two public model families; no retraining | Only Raptor windows K42→K62; fixed50:50 rank weights | 67-study/sixteen-stress and common-blend/runtime checks |
| Oct 09 02:44 | 56978342 | Anatomical Mirror Raptor | 0.949 | Public nartaa recipe/checkpoint; no retraining | Native320/K94 mirrored recipe reproduced; audited runtime | 67-study native/window/probability/repeat parity |
| Oct 09 05:35 | 56986393 | Mirror Raptor + ConvNeXt fusion | 0.950 | Published public fusion; no retraining | Frozen published rank-fusion recipe; inference safeguards | Both components67-study and common offline qualification |
| Oct 09 06:40 | 56989116 | MaViT standalone | No score: rerun error | Public MaViT checkpoint; no retraining | Six-view, eight-triplet,288px source recipe; no blend/TTA | 67-study and sixteen-stress parity; hidden rerun failed |
| Oct 09 12:28 | 57003785 | MaViT malformed-DICOM repair | No score: rerun error | Same public MaViT; no retraining | Repair malformed-DICOM handling; model unchanged | Fixed67 + 2000-study image-only qualification; hidden rerun failed |
| Oct 09 20:56 | 57016778 | Own ConvNeXt round0 | 0.878 | Our five trained folds; public MIL trainer + ImageNet | 44-slice/res256/K16; original blended report labels | Allfive raw/cache/window/model parity; exact final-visible CSV |
| Oct 10 05:45 | 57028725 | Own ConvNeXt round1 | 0.874 | Our five trained folds; same public MIL recipe | Change only to fixed teacher-bootstrap labels | Same five-fold reader/SWA/CSV checks; fixed round0 comparison |
| Oct 10 11:52 | 57037296 | Own CoAtNet224 | 0.914 | Our five trained folds; public MIL trainer + ImageNet | CoAtNet224/original labels; five-fold equal probability mean | Fullfive checkpoint/reader/probability and visible CSV qualification |
| Oct 11 22:07 | 57083497 | Public full-field384 Raptor | No score: rerun error | Frozen public nartaa checkpoint; no retraining | Central320→full384, constructor-compatible grids | Positive fixed two-fold development tests + offline rehearsal; hidden rerun failed |
| Oct 12 00:40 | 57086107 | Strict original320 + S6 blend | No score: rerun error | Public nartaa + released S6 bundle; no retraining | Fixed70:30 rank blend; dual-T4 implementation | 870-study development;39-study dual-T4 parity/runtime; hidden rerun failed |
| Oct 12 07:53 | 57096002 | Capped-recovery original320 + S6 | 0.954 | Same public models/70:30 recipe; no retraining | Classify local input failures; global arm means;3% union cap | 39-study dual-T4 rehearsal; failure/state injections;95-check audit |
| Oct 13 00:47 | 57119585 | B0 Dread + S6 no-OAI blend | 0.940 | Public Dread/S6 checkpoints; no retraining | Replace nartaa arm with native Dread v4/K62; same70:30/recovery | 39-study rehearsal;123-check audit; unchanged S6 and exact blend oracle |

## All training families and substantive fits

Fold lists below are ordered fold0 through fold4. They are report-masked macro AUCs from selected models, except where explicitly labelled gold, weak200/300, training-only or final single-epoch. A pooled OOF score is recomputed across predictions and is not the arithmetic mean of fold scores. “Our fit” may reuse a public architecture/trainer; it does not mean a new architecture.

| Date | Training / status | Origin and changed factor | Recorded score | What we tested / outcome |
| --- | --- | --- | --- | --- |
| Oct8 | Frozen DINO head42; complete | Our finding-attention head on generic frozen DINOv2;2000train/300weakdev | Gold58 AUC **0.7201**; weak BCE **0.4632**; public **0.756** | Masked unknowns, train-only normalization, raw/cache parity, offline inference. |
| Oct8 | Extra DINO heads43/44 and meanmax42; complete |Three new heads on same frozen features; encoder unchanged | Gold **0.7165 /0.7159 /0.7016**; four-head mean **0.7207**, public **0.757** | Weak-BCE-only equal-ensemble selection; no per-target fitting. |
| Oct8 | Fresh ConvNeXt512, four epochs; complete |Our ImageNet-initialized six-slot reader;512train/200weakdev |Selected EMA weak AUC **0.8055**, BCE **0.4959** |Trainability, image masks/reader, finite gradients; no submission. |
| Oct8 | Same512 twelve-epoch fit; complete |Fresh initialization; longer schedule on identical cohort |Selected EMA weak AUC **0.8040**, BCE **0.4884** |Duration comparison; BCE improved, ranking did not clearly improve. |
| Oct8 | Matched512 frozen-DINO head; complete |Own CPU head; same512/200 as ConvNeXt |Weak AUC **0.8272**, BCE **0.4764** |Controlled training-count difference; no encoder update or public entry. |
| Oct8–9 | TRAIN04 reference; failed after11epochs |Own generic ConvNeXt;2000train/200weakdev, richer six-slot coverage |Selected weak AUC **0.5278**, BCE **0.59647** |Unknown-only batch exposed objective failure; discrimination near chance. |
| Oct9 | TRAIN04 global-objective recovery; complete |Fresh fit with corrected loss normalization/unknown handling |Weak AUC **0.5236**, BCE **0.59456** |11995 accepted updates; objective corrected but useful ranking not established. Closed. |
| Oct9 | Own ConvNeXt256 round0, five folds; complete |Our weights; public Raptor MIL trainer, generic ImageNet,44slice cache/original blended labels |Folds **.8809/.8823/.8814/.8785/.8895**; pooled **.8821**; public **.878** |Top3SWA, heldout report support, allfive raw/cache/model/CSV parity. |
| Oct9 | Track-A DINOv2-Small224, five folds; complete |Our adapted generic DINO backbone in same MIL framework |Folds **.8420/.8112/.8377/.7312/.8337**; no public entry |Representation/complementarity test; fixed fusion with ConvNeXt worsened development. |
| Oct9–10 | Own ConvNeXt256 round1, five folds; complete |Fresh fit; fixed teacher-bootstrapv1 target table |Folds **.8885/.8961/.8935/.8908/.9002**; pooled **.8937**; public **.874** |Matched reader/recipe; weak AUC rose while public score fell versus .878. |
| Oct9–10 | Own CoAtNet224 original, five folds; complete |Our ImageNet12k→1k weights adapted using public MIL recipe/original labels |Folds **.9018/.9020/.8969/.8976/.9066**; pooled **.9003**; public **.914** |Architecture switch, top3SWA and fullfive image-only qualification. Best submitted own family. |
| Oct10 | CoAtNet224 bootstrapR1, folds0/1; complete |Same recipe, bootstrapv1 labels |Folds **.8998/.9096**; pooled pair **.9048** vs original pair **.9017** |Mixed fold effects; MCL worsened. Other R1 folds were not fitted. |
| Oct10–11 | CoAtNet224 R2, five folds; complete |Bootstrapv2 plus changed class weights/global teachers |Folds **.9090/.9142/.9090/.9090/.9141**; pooled **.9108**; no public entry |Label/weight recipe combined; indirect teacher and prior gold exposure remain. |
| Oct9–10 | CoAtNet384 bootstrap fold0; complete after earlier losses |Our generic initialization;14epochs,K12train/K24eval,bootstrapv1 |Recovered selected SWA **.918185**; no public entry |Strict recovered tensor/epoch/export checks; not a clean resolution-only comparison. |
| Oct10–11 | CoAtNet384 original, five folds; complete |Our ImageNet model;44×336cache resized384;12epochs,K12/K24,original labels |Folds **.9128/.9142/.9120/.9071/.9210**; pooled **.9132**; no public entry |Higher-resolution recipe, persisted/resumed runs, selected-state provenance and OOF export. |
| Oct10–11 | Public full-predictor adaptation, two epochs; fit complete/evaluation recovered |Fine-tuned released nartaa knee backbone **and head**, frozen BN, low-LR AdamW |Exposed development **.951331 frozen→.948113 adapted** |432 updates; fixed two-epoch average; paired CI negative. Rejected, not submitted. |
| Oct12 | Source96 fold0; complete/bookkeeping failure reconciled |Our pinned ImageNet CoAtNet224;96slots,2–98%span,10epochs |Selected K16 SWA **.894530**; denseK94 **.899102**; no public entry |Input-coverage recipe; selected-weight/backup validation and same-checkpoint density test. Closed. |
| Oct12 | Source96 fold1; complete |Same96recipe, separate fixed fold |Selected K16 SWA **.898941**; no public entry |Second-fold training/backup/export; bulk verification failure recovered per-file. |
| Oct13 | Raptor64 in-orig;12epochs complete, final export recovery |Own generic ImageNet CoAtNet384;64slots,6–94%span,original head |Final single-epoch K16 **.905286**; selected-SWA/K42 **pending** |New input recipe, initialization and persistence; final export lost with VM, resumes preserved. |
| Oct13 | Raptor64 in-plane;12epochs complete, final export recovery |Same input/init; add plane/FS/depth embeddings in attention/classifier |Final single-epoch K16 **.905070**; selected-SWA/K42 **pending** |Matched head comparison; metadata parity. No final head-benefit conclusion yet. |
| Oct13 | E4 per-target stacking; CPU fits complete |Nested-fold logistic regression/GBM on own OOF outputs; no image encoder fit |Best36-input LR **.9166**, versus best simple blend **.9174** |Nested fitting and leakage checks; no gain over best simple identity blend. Closed. |

## Small optimization pilots and failed attempts

These did execute optimization where noted, but were mechanics tests rather than competitive fitted models. Failed setup attempts with zero updates are listed so they are not mistaken for additional completed trainings.

| Run | Actual learning / result | What was tested |
| --- | --- | --- |
| Fresh CUDA probe, Oct8 |64 accepted updates on32studies; no heldout score or retained model |Generic ConvNeXt encoder/head movement, AMP and fit mechanics. |
| TRAIN04 diagnostic, Oct9 |One ordinary update plus64 deterministic32-study updates; no development score |Loss, gradients, EMA and unknown handling. Earlier attachment failure had zero updates. |
| Public MIL smoke40, Oct9 |Two epochs/10 accepted updates; training-only AUC **.6615→.7185**; no exported model |Generic ConvNeXt in the public MIL trainer; finite encoder/head updates and exact save/reload. |
| Track-A smoke v2, Oct9 |Two one-epoch40-study fits, folds0/1; exploratory gold roughly **.52/.54** |Trainer wiring before full folds. v1 label-column error occurred before first batch. |
| Early CoAtNet224 attempts |Position-grid mismatch and OOM failures; no accepted final model score |Correct native224 architecture, tensor geometry and memory recipe before completed runs. |
| Earlier CoAtNet384 attempts1/2 |Attempt1 fitted epochs0–3 (latest report **.8882**); attempt2 fitted through epoch11; sessions lost |Persistence/recovery lessons. These are distinct incomplete attempts, not proven exact continuations of the final14epoch fit. |
| Public full-model warmstart smoke |One synthetic-target SGD update on real visible images; no accuracy score |Full backbone/head load, frozen-reference parity, gradient movement and save/reload. |
| AdamW adaptation pilot v1/v2/v3 |v1zero updates failed;v2two updates failed;v3three updates failed |Zero-mask rows, accumulation tail gradient and deterministic continuation; disposable states. |
| Streamed adaptation pilot and external recovery |Three qualified updates plus replay, then one recovery-tail update; no accuracy score |Raw producer/consumer queue, exact inline gradient oracle, restored RNG/optimizer/state. |
| Adaptation evaluation recovery |Zero optimizer updates |Recovered and evaluated the original fit; no retraining after corrected binary-truth assumption. |
| Source96 and384 selected exports; current raptor64 recovery |Zero intended optimizer updates |Reconstruct/persist selected checkpoints and OOF outputs; export jobs are not new fits. |


## Important tests that were not MRI-model training

| Test | Result | What it tells us / decision |
| --- | --- | --- |
| Original own224 dense-window test, same fold0 checkpoint | K16 **0.901842** → K42 **0.905877**; delta **+0.004035**, paired CI [+0.001769,+0.006344] | More inference windows helped this checkpoint. This is development evidence, not a submission. |
| Source96 secondary evaluation | K16 **0.894530** → K94 **0.899102**; gated mirror **0.899450** | Density helped, but dense96 still trailed own44/K42 by0.006775 with negative CI. This96 recipe was closed; mirror gain was inconclusive. |
| Frozen public Raptor central320→full384 | Fold0 **0.951331→0.953977**; fold1 **0.953150→0.954966** | Positive paired development improvement on both folds; public checkpoint exposure persists. Qualified entry57083497 failed hidden execution, so no public accuracy result. |
| Original320 + S6 fixed70:30 | **0.951331→0.962964** on870 development studies; S6 alone0.969494 descriptively | Positive exposed development evidence; the published70:30 recipe eventually scored0.954 with our recovery wrapper. |
| Own384 + R2-224 fixed50:50 probabilities | Pooled report AUC **0.913156 /0.910751→0.917421** | Complementarity between own models on reused report folds. Not submitted or competitively established. |
| 80% public +10% own384 +10% R2-224 | **0.951331→0.948296**, negative paired CI | Adding these own arms hurt this fixed development blend. Closed without weight search. |
| Public original320/S6 balanced50:50 | **0.962964→0.967675** compared with70:30 on exposed870 | Positive diagnostic, deferred because this reuses exposed development and a similar public recipe already exists. No submission. |
| LLM relabel pilot |358 reports; gold58 label-AUC **0.889**, versus public Stevenv4 **0.893** | No compelling improvement; synovitis regression and borderline state agreement. This is a label-extraction metric, not image-model CV. Relabeling closed. |
| Label/provenance benchmark | Several tables around0.89; one table copied official gold labels exactly | Audited label independence and unknown handling; rejected leaked gold as validation evidence. No new MRI training. |
| Center-only native90mm crop | Mechanics passed; anatomy-containment review failed | A geometric image-center crop was not a qualified joint ROI. This route was rejected. |
| All-corpus orientation audit |4407 studies/677292 headers; universal99% mirror criterion failed | Frozen per-study mirror eligibility/bypass required; no universal mirror claim. |
| Current native joint100mm reader qualification | Private CPU/offline frozen32 study run launched; no classifier or fitting | Tests localizer projection, native sampling, matched global fallback and containment. It must pass before ROI training. |

## Planned and conditional training

| Next item | Source / comparison | Current state | Gate / purpose |
| --- | --- | --- | --- |
| Recover raptor64 ImageNet original/plane final models |Restore final epoch11 resumes, rebuild top3 SWA and K42 exports |Recovery under Claude; **no retraining needed** |Recover lost VM-only exports and perform matched per-target evaluation. |
| dr-orig fold0 |Dread competition-trained384 backbone; fresh original MIL head; declared64-slot recipe |Prepared; original queued client did **not** launch; follows recovery |Measure mechanics and transfer behavior; development is exposed to Dread's prior training. |
| dr-full |Dread-backbone fit on all4345 eligible studies |Package prepared, **not launched**; epoch/SWA policy must be frozen |Full-data candidate for a qualified public test only if prior evidence warrants it. No heldout-CV claim. |
| na-orig |nartaa competition/OAI backbone plus our fresh original head |Pin/tests prepared, **held; not launched** |Optional separately named transfer arm; not an approved result or matched clean control. |
| Native global/local controlled pair |Two fresh ImageNet models:24 matched triplets; allglobal control versus12global+12native100mm joint crops |Codex plan; reader qualification running; **not trained** |Keep split/labels/backbone/optimization/view budget fixed; test locality/magnification with exact fallback. |
| Second-fold replication |Winning unexposed new input/head recipe |Conditional; **not launched** |Replicate a supported fixed first-fold gain before scaling. |
| Final own fit and three-member entry |Best supported own model alongside original320/S6 |Conditional; no such new entry submitted |Need genuine incremental benefit, T4 runtime/memory qualification and valid image-only rehearsal. |
| Dread plane-aware arm |dr-plane |Dropped before fit |The clean ImageNet original-vs-plane comparison answers the head question without another exposed arm. |
| New OAI pretraining, more blanket relabeling/stacking, repeated source96 recipe |External OAI collection or previously weak routes |Not scheduled / closed or deferred |No OAI acquisition/training by us; preserve negative results and spend compute on a distinct hypothesis. |

## Reading the campaign so far

1. **We reproduced strong public assets before attempting to improve them.** The largest leaderboard jumps came from better public knee checkpoints and their existing ensemble recipes.0.954 does not show that we trained a new0.954 model.
2. **Our own backbone choice mattered.** Generic-pretrained CoAtNet224 at0.914 outperformed our five-fold ConvNeXt0.878. More resolution or a higher weak-label score has not yet demonstrated a better public entry.
3. **Several plausible interventions were negative.** ConvNeXt teacher bootstrapping fell0.878→0.874; the public full-model adaptation lost development AUC; the wider96-input recipe underperformed; relabeling and stacking did not yield persuasive gains. These are specific recipes, not proof that the whole method category can never work.
4. **Coverage and locality remain distinct hypotheses.** Dense windows improve our saved own models; public full-field rendering improved exposed development. The new joint-crop pair will test magnified local anatomy while preserving global context.
5. **Operational validity remains part of the result.** Six entries have no score because hidden execution failed despite visible rehearsals. Checkpoint persistence saved completed training when Colab reclaimed sessions. Recovering an export is not a new training run.
6. **A top-ten outcome is not established.** Current rank must be checked on the live leaderboard; the older recorded0.963 tenth-row score is not a current threshold. None of the current unsubmitted development scores provides a top-ten forecast.

Sources: official Kaggle submission snapshot `outputs/analysis/campaign-ledger-20261013/official-submissions.json`; linked per-experiment documents and training ledger below. Raw identifiers, predictions, caches, reports and weights remain ignored/private. Routine cache jobs, source audits, rehearsals and recovery exports are grouped by experiment rather than counted as independent model fits.

## Exact fold appendix

These33 rows represent six five-fold families, a two-fold bootstrap arm and one384 bootstrap fold. Values are read from receipts; this report did not rerun image inference. Gold columns remain exposed/exploratory.

| Family / receipt directory | Fold | Report AUC | Gold58 AUC | Epochs | Selected SWA |
| --- | ---: | ---: | ---: | ---: | --- |
| trackA-coatnet224-fold01-20261009-result | 0 | 0.901842101 | 0.897156911 | 10 | [8, 9, 7] |
| trackA-coatnet224-fold01-20261009-result | 1 | 0.902007790 | 0.890379215 | 10 | [9, 8, 7] |
| trackA-coatnet224-fold23-20261009-result | 2 | 0.896920731 | 0.879271977 | 10 | [8, 9, 7] |
| trackA-coatnet224-fold23-20261009-result | 3 | 0.897643212 | 0.896758816 | 10 | [9, 8, 7] |
| trackA-coatnet224-fold4-20261009-result | 4 | 0.906643414 | 0.897501275 | 10 | [8, 9, 7] |
| trackA-coatnet224-r1-fold01-20261010-result | 0 | 0.899827877 | 0.888276788 | 10 | [8, 9, 7] |
| trackA-coatnet224-r1-fold01-20261010-result | 1 | 0.909633436 | 0.886904500 | 10 | [9, 8, 7] |
| trackA-coatnet224-r2-fold01-20261010-result | 0 | 0.909008784 | 0.900834052 | 10 | [9, 8, 7] |
| trackA-coatnet224-r2-fold01-20261010-result | 1 | 0.914247352 | 0.905754914 | 10 | [8, 9, 7] |
| trackA-coatnet224-r2-fold23-20261010-result | 2 | 0.909005324 | 0.901848521 | 10 | [8, 9, 7] |
| trackA-coatnet224-r2-fold23-20261010-result | 3 | 0.909010545 | 0.908420502 | 10 | [8, 9, 7] |
| trackA-coatnet224-r2-fold4-20261010-result | 4 | 0.914071141 | 0.899827207 | 10 | [8, 9, 7] |
| trackA-convnext-20261009-result | 0 | 0.880946170 | 0.865673810 | 10 | [8, 9, 7] |
| trackA-convnext-20261009-result | 1 | 0.882340870 | 0.863233975 | 10 | [8, 7, 9] |
| trackA-convnext-fold23-20261009-result | 2 | 0.881449365 | 0.856512243 | 10 | [8, 9, 6] |
| trackA-convnext-fold23-20261009-result | 3 | 0.878533561 | 0.855896328 | 10 | [8, 9, 7] |
| trackA-convnext-fold4-20261009-result | 4 | 0.889544519 | 0.866255770 | 10 | [9, 8, 7] |
| trackA-convnext-r1-fold01-20261009-result | 0 | 0.888543843 | 0.868783101 | 10 | [9, 8, 7] |
| trackA-convnext-r1-fold01-20261009-result | 1 | 0.896111612 | 0.870473212 | 10 | [8, 9, 7] |
| trackA-convnext-r1-fold23-20261009-result | 2 | 0.893513865 | 0.863961828 | 10 | [8, 9, 7] |
| trackA-convnext-r1-fold23-20261009-result | 3 | 0.890753046 | 0.864592130 | 10 | [9, 8, 7] |
| trackA-convnext-r1-fold4-20261009-result | 4 | 0.900206501 | 0.867052074 | 10 | [9, 8, 7] |
| trackA-dinov2s-fold01-20261009-result | 0 | 0.841957056 | 0.827336956 | 10 | [9, 8, 7] |
| trackA-dinov2s-fold01-20261009-result | 1 | 0.811173326 | 0.797507587 | 10 | [9, 8, 7] |
| trackA-dinov2s-fold23-20261009-result | 2 | 0.837672815 | 0.821207027 | 10 | [9, 8, 7] |
| trackA-dinov2s-fold23-20261009-result | 3 | 0.731155637 | 0.662239332 | 10 | [9, 8, 7] |
| trackA-dinov2s-fold4-20261009-result | 4 | 0.833655558 | 0.809909919 | 10 | [9, 8, 7] |
| trackA-export-c384-fold0-20261010-result | 0 | 0.918184624 | 0.902109991 | 14 | [13, 12, 11] |
| trackA-export-c384r-fold0-20261010-result | 0 | 0.912771053 | 0.907974467 | 12 | [10, 11, 9] |
| trackA-export-c384r-fold1-20261010-result | 1 | 0.914180850 | 0.907143201 | 12 | [8, 9, 10] |
| trackA-export-c384r-fold2-20261010-result | 2 | 0.912045846 | 0.901528398 | 12 | [9, 11, 10] |
| trackA-export-c384r-fold3-provenance-20261011-result | 3 | 0.907117421 | 0.904314727 | 12 | [10, 11, 8] |
| trackA-export-c384r-fold4-provenance-20261011-result | 4 | 0.920959462 | 0.904515676 | 12 | [10, 9, 11] |

Primary evidence: [first baseline](20261008-first-baseline.md), [own224 fullfive](coatnet224-fullfive-20261010.md), [bootstrap arm](coatnet224-label-arm-20261010.md), [384 recovery](coat384-recovery-export-20261010.md), [family blend](fixed-family-blend-20261011.md), [public adaptation](public-adaptation-result-20261011.md), [source96 secondary](source96-fold0-secondary-result-20261012.md), and current STATUS/shared handoff for raptor64. Detailed aggregate receipt extraction: `outputs/analysis/campaign-ledger-20261013/training-ledger-exact-folds.json`.
