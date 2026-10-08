# Public checkpoint provenance audit — 2026-10-08

## Decision

Proceed with a bounded, offline **standalone ConvNeXt reader** rehearsal. Its public package provides the clearest training and inference trail of the inspected candidates, a permissive weight/code license, and no OAI input in the supplied training source. The author reports standalone public AUC 0.929; 0.944 belongs to its larger blended stack and must not be attributed to this reader alone. This is a provenance recommendation, not a prediction of our reproduced score.

The 0.949 anatomical-mirror CoAtNet is publicly available and its author expressly permits competition use, but its OAI-derived supervision leaves an unresolved eligibility question. **No explicit host prohibition of these public weights was found.** Do not label them illegal or let this uncertainty block the independent ConvNeXt experiment. The exact unresolved issue is whether a freely available learned checkpoint is an eligible external model when some original supervision came from controlled-access OAI data, and whether its declared model-use terms and available documentation meet the applicable winner obligations.

## Current official evidence

Retrieved complete rules and full discussion comments through authenticated, read-only Kaggle CLI/API on October 8. Host comments are distinguished from participant interpretations.

| Source | Verified position | Consequence |
| --- | --- | --- |
| [Official rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules), sections 2.6(a–b) | External data/models are permitted unless specifically prohibited, subject to equal/reasonable access and minimal cost; other obligations still apply. | Public availability matters, but is not the entire test. |
| Same rules, 2.5(a)(4), 2.8 | Pretrained models/input data with incompatible licenses need not receive the preceding winner relicensing grant. Winner deliverables include reproducible training/inference code, weights and environment information. | A custom or noncommercial pretrain license alone does not establish exclusion; distinguish using an existing checkpoint from reproducing its original training. |
| [Host OAI answer, September 21](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819#3526495) | OAI is allowed **if** free and generally accessible without escalated institutional review/sign-off or a legal team. Host cannot waive the steward's terms. | Conditional answer, not unqualified approval of OAI or the examined checkpoint. Another participant's subsequent “no” is not a host ruling. |
| [Host accessibility/license answer, August 27](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965#3517696) | Noncommercial restrictions alone do not exclude a dataset. Ordinary registration/click-through generally qualifies; institutional approvals or substantial administrative barriers may not. Teams retain source-agreement obligations. | Do not infer exclusion merely from research-use wording. |
| [Public pretrained-checkpoint question](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950#3519419) | Participants explicitly ask about public weights trained partly on inaccessible data; all nine fetched comments contain no host resolution. | This closely matches the unresolved model-versus-data distinction; it cannot be converted into an approval or ban. |
| [OAI derivatives question](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/738156) | No substantive host ruling in the retrieved thread. | Public relicensing of derivatives is not independently established as a solution. |
| [KneeCoT ruling](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109#3516795) | Host explicitly disallows that named dataset because of its institutional agreement, subject to future access-policy change. | A concrete accessibility precedent; not a ruling on every learned derivative. |

The rules' winner-license paragraph contains inconsistent labeling of CC-BY-NC 4.0. This audit does not resolve that drafting issue; the pretrained-model exemption and host's noncommercial clarification above are the directly relevant evidence.

## Anatomical mirror CoAtNet

Sources: [public notebook](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror), [publication weights](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007), downloaded package README and manifest.

- Accuracy checkpoint: `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, **292,840,787 bytes**, manifest SHA-256 `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`. Author identifies submission 56906959, public 0.949. SHA was read from the manifest, not independently recomputed from weights.
- Separate efficiency checkpoint: `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt`, 292,842,755 bytes, reported 0.945. The dataset also contains only the small README and manifest. Downloading the entire dataset unnecessarily obtains both weights.
- Dataset license metadata is `other`. README expressly permits research/education including this competition, preserves third-party source/pretrained licenses, and grants no OAI data rights. It does not claim CC0 rights over OAI.
- Author discloses **2,399 baseline OAI knees** used for masked external supervision in both models, from controlled-access NDA data, DOI `10.15154/0hcg-f676`, shared study 3407. Weak-label lineage also includes Steven, Pilkwang, JEV and the author's report-derived labels. Those label tables and original training artifacts are not in the package.
- Gold58 was reused for selection and is explicitly not independent CV. Averaged epochs 14/11/12, 384-trained backbone, native320 inference, five slots, 96 slices, 94 adjacent-slice windows and original/anatomical-mirror probability averaging describe the accuracy recipe. The notebook's older opening docstring describes another recipe; use live constants and manifest.
- Inference source and the exact released checkpoint are public. The original scored notebook/training dependency remain private according to the publication notes. Therefore inference reproduction is plausible, but full training reproduction is not established. The 26.1-minute claim is submit-to-score latency, not a measured inference runtime.
- Pulled notebook is offline and attaches only the competition and publication-weight dataset. Docker digest: `2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`; author reports torch 2.11.0+cu128 and timm 1.0.29. Kernel metadata does not expose a notebook source license; retain attribution and resolve source terms before redistribution beyond the inspected licensed package.

## Standalone ConvNeXt reader

Sources: [reader weights and source](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader), [stack notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944). Dataset metadata explicitly declares **Apache-2.0**. Its six Python files were read as source, never imported or executed.

| Item | Verified detail |
| --- | --- |
| Public checkpoint files | `cnxt_v0_fold0.pt`, `cnxt_v0_fold1.pt`, `cnxt_v0_fold2.pt`; each 123,035,271 bytes, total **369,105,813 bytes** (~352 MiB). Weights were not downloaded; state dictionaries and embedded arguments remain unverified. |
| Training inputs | Competition images; default timm `convnext_tiny` with `pretrained=True`; mean of four public report-label tables. No OAI input/path is present in the supplied training/preprocessing/label code. Exact original timm weight tag and package versions are not pinned in that training source. |
| Label provenance | Steven `llm_labels_v4_blend.csv`; Pilkwang `report_labels_v2.csv`; Lixin73 `labels_llm_gpt56sol.csv`; Laymond `qwen_knee_weak_labels.csv`. `v0` needs no author's private label file; `v1` optionally does. These tables and upstream licenses were not downloaded/re-audited in this pass. |
| Training method | Full ConvNeXt-Tiny tuning, 256px, 14 epochs, EMA; six sequence slots, adjacent 3-slice windows, series/position embeddings, study transformer and per-target pooling. Final EMA is saved after the final epoch. The supplied code does not select the checkpoint with highest gold AUC. |
| Validation limits | Gold58 excluded from direct fitting, but evaluated each epoch. Five folds group normalized report text among weak studies. Patient grouping and exclusion of weak reports duplicated in gold are absent. Upstream Steven v4 labels were gold-guided in the earlier project audit. Gold is therefore exploratory, not a clean benchmark. |
| Label/mask semantics | Available source cells are averaged and clipped to [0.01, 0.99]. Exact 0.5 is trained as a soft target, not masked. Reproducing these weights must preserve their inference; new training should use the project's predeclared supervision policy. |
| Public reproducibility | `train.py`, `make_labels.py`, `oof.py`, `preprocess.py`, `knee.py`, `infer.py` are present. Retraining needs path replacement from `/mnt/ssd/rsna_knee`, the four label tables and a canonical training cache. Exact checkpoints are more readily reproducible at inference than the training run bit-for-bit. |

The four label dataset references encoded in `make_labels.py` are `stevenleehans/rsna-knee-llm-report-labels`, `pilkwang/rsna-knee-llm-labels`, `lixin73/rsna-knee-llm-report-labels-sol56`, and `laymond/rsna-knee-abnormality-qwen3-8b-weak-labels`.

Offline rehearsal checks with direct implications:

1. Run the reader alone, excluding the full stack's fifteen datasets, two notebook outputs and DINO attachment. The reader's three models and public Python sources suffice alongside competition data and compatible dependencies. Author's approximately 20-minute claim needs our own timing.
2. Decoder wheels in its package support Python **3.11/3.12 only**, plus a pure-Python pylibjpeg wheel. Our current Python 3.13 image needs the already verified matching decoder bundle, or the reference environment. Reference Docker digest is `37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`.
3. `infer.py` reconstructs from checkpoint backbone arguments with `pretrained=False`, strips the timm weight suffix and loads strict state dictionaries. It needs no ImageNet download at inference. Validate all three models' architecture/target order on load; all are held on one CUDA device together.
4. IDs come dynamically from `<split>.csv`; output uses official hardcoded label names in that order. Retain the project's exact-ID/order, finite probability and repeated-read checks.
5. Per-series decode exceptions are silently skipped by the source reader. Record missing/failed series and reject all-empty studies in the rehearsal; do not treat finite output alone as a successful read. Preserve source preprocessing: canonical 0.4 mm/pixel, 384px canvas, up to 64 slices/series and its model crop/window selection.
6. Gold58 can check numerical parity, not select architecture/blends. The public notebook also reports several leaderboard blend probes; its 0.944 is not an independent validation estimate.

## Backup and limits

The live [Dread Raptor package card](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense) declares **CC0**, two approximately 280 MB CoAtNet checkpoints, competition report-derived supervision and no external data. Its author reports 0.924 with the single best epoch and gold AUC 0.9167. Backbone metadata names `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, so “no external data” should be read as no additional task-specific training corpus, not no ImageNet pretraining. The card links a public inference notebook. Complete training code and the label-generation history were not audited here; it is a reasonable later architecture-diverse backup, not better evidenced than ConvNeXt for immediate reproduction. Mattia's 0.943 multi-model stack was not re-audited end-to-end and has no blanket no-OAI/provenance conclusion from this pass.

No GPU job, model import, training, submission, message to a host, dataset upload, terms acceptance, large checkpoint download or MRI download occurred in this audit. Remaining uncertainties are disclosed above; none prevents the authorized standalone ConvNeXt offline rehearsal.

## Evidence and verification

Small public source snapshots and SHA-256 inventory are under ignored `outputs/provenance-20261008/`. Reads used `kaggle kernels pull -m`, `datasets metadata/files`, targeted small-file downloads and `competitions topics show --page-size 200 --format json`. Rules were read via `KaggleApi.competition_list_pages_cli(..., page_name='rules', content=True, output_format='json')` because the CLI page parser rejected that equivalent command. Web page opening returned no discussion content, so the official CLI responses support this audit.

Documentation-only artifact: all six Python files parse successfully as Python syntax; no code execution or runtime validity is implied. Markdown links, evidence hashes and `git diff --check` verified. Parent owns the main STATUS update and experiment launch decision.

## Narrow Raptor geometry follow-up, 02:50UTC

Public CC0 cards list MaxSpanv5SWA (author public0.928), native384v8SWA
(author0.926), and native384densev10 (publicscoreundisclosed). Each checkpoint is
approximately292.83MB. The author states all published checkpoints share the same
labels, competition-only task data and architecture; geometry/window coverage
changes. This supports a disclosed non-OAI claim, not an independently attested
initialization/teacher lineage. Gold58 selected checkpoints and full-corpus training
overlaps our local development sets. Sources: [MaxSpan](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-maxspan),
[native384](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384),
[native384dense](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-native384dense),
[author data statement](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696#3521127).

A narrower potential test uses the already downloaded plainv4 weights with62
windows instead of42. The author reports0.924->0.927 for this single change
([coverage clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737696#3524687)).
This is an author claim; complementarity and our score are untested. It requires a
new explicit recipe and runtime/memory qualification, not a filename replacement
or a change to the currently submitted version. No newweights/job/submission was
made. Public source snapshots and hash-preservation receipt are under
`outputs/research/2026-10-08/raptor-variants/`.
