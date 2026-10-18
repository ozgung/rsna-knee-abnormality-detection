# Public S6 source and weight eligibility check — 2026-10-11 (read-only)

Requested by Sol (handoff 07:46 UTC). Scope: official rules text via the Kaggle CLI, dataset metadata via the Kaggle CLI,
public web pages. No weights downloaded, no terms accepted, no messages sent, no compute. Kaggle web pages render only a
title to a fetcher, so Kaggle facts come from the CLI; discussion threads could not be read this way.

## Official rules (competition `rules` page, read with `kaggle competitions pages --content`)

- Specific rules 6 External Data and Tools: External Data may be used if "publicly available and equally accessible to use by
  all Participants of the Competition for purposes of the competition at no cost" or meets the Host's Reasonableness standard;
  "The use of external data and models is acceptable unless specifically prohibited by the Host"; use must be "reasonably
  accessible to all" and of "minimal cost".
- Code Requirements page: "Freely & publicly available external data is allowed, including pre-trained models".
- Specific rules 5.2 Winner license: winning Submission and source code licensed under CC-BY-NC 4.0. 5.4: "In the event that
  input data or pretrained models with an incompatible license are used to generate your winning solution, you do not need to
  grant an open source license in the preceding Section for that data and/or model(s)."
- General rules, Use of Open Source: open source code used in the model must be OSI-approved and must not limit commercial use
  of code or a model depending on it. (This clause governs code; the 5.4 carve-out governs data/pretrained models.)
- Private sharing outside teams prohibited; public sharing must be on the competition forum/notebooks.
- Host discussion positions previously recorded in the handoff (not re-read today): non-commercial licenses not disqualifying;
  OAI allowed if generally accessible without institutional review and free (D/741819); eligibility of the OAI-trained public
  CoAtNet checkpoint asked in D/747660 and D/747455 with no host answer.

## The three attached public datasets (owner, Kaggle license field, description)

| Dataset | License field | Content (author description) |
|---|---|---|
| `heliosli/rsna-knee-compact-v1` | other | four ConvNeXt-small fold models + side classifier, fine-tuned on report-derived targets; "No images, reports or labels inside" |
| `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` | other | EMA weights, inference source, "unchanged SKM-TEA/KneeXNet auxiliary weights. Upstream weights retain their original licensing; no new open-source license is granted" |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | other | the original public CoAtNet SWA checkpoints (0.949 / 0.945) |

All three are public Kaggle datasets attached to public competition notebooks, so they are "publicly available and equally
accessible ... at no cost" in the sense of rule 6. The Kaggle license field "other" is a declaration by the uploader, not
a grant; the compact-v1 README says only: "The dataset has been restored. I removed it earlier because someone told me it
wasn't allowed, but it seems to be allowed after all." That is the uploader's opinion, not a host determination.

## The learned-localizer weights (`anatomy.pt`)

- Hosted in `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` as `anatomy.pt`, 112,496,037 bytes, created 2026-10-04.
- The Matterhorn anchor code loads it as "the SKM-TEA V-Net" with default path `pretrained/skm_tea/model-raw.ckpt` and binds
  it by sha256 to the training manifest's `anchor_weights_sha256`.
- Stanford MIMI publishes the SKM-TEA pretrained models on Hugging Face `arjundd/skm-tea-models`, model card license
  **apache-2.0**. `neurips2021/segmentation-models/V-Net_E1/model-raw.ckpt` and `model.ckpt` are **exactly 112,496,037 bytes**;
  no other file in that repository has this size. Byte identity is not proven (no sha256 published for the Hugging Face file
  and none downloaded), but the size match plus the loader's name make it very likely that `anatomy.pt` is the unmodified
  SKM-TEA V-Net E1 checkpoint.
- Consequence: the localizer weights are a publicly hosted, Apache-2.0 model (OSI-approved, commercial use allowed). Using
  or re-uploading them is compatible with the rules; attribution to Stanford MIMI / Desai et al. (NeurIPS 2021 Datasets
  track) is required by Apache-2.0 notice terms and expected by AIMI's acknowledgment request.

## SKM-TEA *data* terms (relevant only if someone trains on the data, which the S6 path does not)

- The SKM-TEA paper states the data is distributed under Stanford University School of Medicine terms and the terms listed
  for the LERA dataset. The LERA research use agreement (AIMI page) is "for personal, non-commercial research purposes only",
  "YOU MAY NOT DISTRIBUTE, PUBLISH, OR REPRODUCE A COPY of any portion", "must not ... create derivative works", and
  prohibits sharing the download link. AIMI's shared-datasets page: free for "non-commercial research use"; commercial use
  needs a license. Neither page addresses models trained on the data; Stanford released the trained models separately under
  Apache-2.0, which settles the weights question without needing to interpret the data agreement.
- The S6 reader itself was fine-tuned on competition images with report-derived targets (per manifest: 4,344 pool studies,
  58 gold excluded, official labels unread). Its ImageNet-22k→1k ConvNeXt-small initialisation is a timm/Facebook weight
  under a permissive license. The restoration README's "someone told me it wasn't allowed" most plausibly refers to the
  earlier private-sharing concern, not to a licensing finding; no host statement was found either way.

## Remaining unknowns (exact)

1. Whether the Host regards the OAI-trained public CoAtNet checkpoint (`nartaa` weights, used by every 0.95+ public
   notebook including Matterhorn's first arm) as eligible: D/747660 and D/747455 had no host answer as of the last read;
   today's check could not open discussion threads. This is the only eligibility question with real exposure, and it applies
   equally to our current best public entry.
2. Byte identity of `anatomy.pt` with the Hugging Face V-Net E1 `model-raw.ckpt` (size-only match).
3. Whether `heliosli` fine-tuned KneeXNet weights are used anywhere in the S6 inference path: the Matterhorn source contains
   no KneeXNet reference, so apparently not, but the orthofoundation bundle ships them.
4. The gold montages (competition data) informed the author's anchor/offset development; that is use of competition data for
   development by a participant and is permitted; it does not make the weights contain gold labels (manifest: 58 gold
   excluded from fitting, official labels unread). Not independently verifiable without the training logs.

## Bottom line

Nothing found makes the S6 reader, its Apache-2.0 SKM-TEA localizer, or the public datasets ineligible under the written
rules: they are free, public, equally accessible, and attached to public notebooks. The one open exposure is the unanswered
host question about the OAI-trained CoAtNet, which we already share with every leading public entry. If the team wants a
definitive answer, the only authoritative route is a host reply in the competition discussion; I have not posted anything.

Parent integration07:54UTC: accepted the read-only rules/publisher evidence as support in principle. The exact mountedlocalizer is not yet bound to the upstream Apache-2.0 artifact; size-only agreement is an inference, not a license or identity proof. A narrow existingmanifest versus upstreamLFS SHA comparison is assigned to Fable without downloading weights. The statements about all leading notebooks and the cause of restoration are not independently verified. No new model collection or inference rehearsal follows this review yet.

## Follow-up (Sol 07:54 UTC): localizer byte identity via published hashes — RESOLVED as a match

- S6 training manifest (`heliosli/rsna-knee-compact-v1` → `manifest.json`, bundle `209a32c8b50a6b1c`, built 2026-10-09 02:35:47):
  `anchor_weights_sha256 = 5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e`.
- Hugging Face `arjundd/skm-tea-models` (license apache-2.0), Git LFS pointers read from the `raw/main` endpoint and the tree API:
  `neurips2021/segmentation-models/V-Net_E1/model-raw.ckpt` → `oid sha256:5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e`, `size 112496037`;
  `V-Net_E1/model.ckpt` carries the same oid and size. No other V-Net_E1 file has this oid.
- The two hashes are identical. The training-time anchor checkpoint is therefore byte-for-byte Stanford MIMI's SKM-TEA V-Net E1
  release, published under Apache-2.0.
- Attached runtime file: `heliosli/rsna-orthofoundation-fold0-epoch7-20261004/anatomy.pt`, 112,496,037 bytes, created
  2026-10-04 07:54:50 (same size as the upstream file). The Matterhorn notebook asserts at inference, on the `views in (v3k, v4)`
  path that the S6 reader uses, `sha256(anatomy.pt) == manifest['anchor_weights_sha256']` and aborts otherwise. Its posted
  run completed, so the attached bytes matched the manifest hash in that execution. I have not hashed the attached file
  myself (no download); the binding is manifest hash = upstream LFS oid (verified here) plus the notebook's own runtime assertion.
- Restoration bindings: the Kaggle CLI cannot read version numbers of datasets I do not own (`datasets status` → 404,
  metadata has no version field). What is readable: file names, sizes and creation timestamps (compact-v1 files and
  `anatomy.pt` listed in the scratch receipts), and the README statement that restored latest files equal version 10.
  The S6 fold files are bound by sha256 in the same manifest (`sha256` map), so a byte audit at use time can rely on the
  manifest rather than on Kaggle version numbers.
- Correction to the first write-up: I wrote that "every leading public entry" shares the OAI-trained CoAtNet exposure.
  Verified only for the three inspected notebooks (Matterhorn, Heliosli restored, Jenil) and our own replication; not an
  exhaustive audit of all 0.95+ notebooks.
- Restoration file timestamps (CLI listing, read-only): `heliosli/rsna-knee-compact-v1` current files were all created
  2026-10-10 19:50:55–19:51:00 (manifest.json 2,457 B, README 684 B, state_spec.json 7,814 B, side_cls.pt 85,255,211 B,
  four fold files 200,379,129 B each), i.e. the restoration upload; `anatomy.pt` in the orthofoundation dataset dates from
  2026-10-04 07:54:50 and was not part of that restoration.

Parent independent verification 2026-10-11 08:28 UTC:134-byte [upstream LFS pointer](https://huggingface.co/arjundd/skm-tea-models/raw/main/neurips2021/segmentation-models/V-Net_E1/model-raw.ckpt) matches the existing S6 manifest's5904dd16eb9a4fd7a1caf90ac977f8fe2ef364908598c23c209db3fe7fd1563e and112496037-byte size. The31-byte [publisher model card](https://huggingface.co/arjundd/skm-tea-models/raw/main/README.md) declares Apache-2.0.3 parent metadata checks PASS, without downloading model bytes. Receipt `outputs/analysis/s6-provenance-parent-20261011/receipt.json`. This resolves the former size-only upstream identity uncertainty; our actual mountedanatomy.pt must still match that SHA before use. Public-source OAI/validation exposure limits remain unchanged.
