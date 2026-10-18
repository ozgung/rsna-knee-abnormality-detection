# MRI domain checkpoint audit — October 11, 2026

## Decision

Neither lead is ready to replace the qualified S6 predictor. Preserve the pending submission and prioritize its actual result. This metadata audit used no GPU, downloaded no weights, ran no external code and accepted no terms. It narrows the next checkpoint-selection decision; it does not establish performance or competition eligibility.

## OrthoFoundation-MSK-5M

Pinned repository revision **df8f943460ffc452974869d33f15afe3674aa9f3** (commit July 21). The untruncated recursive tree contains 43 entries and no LICENSE, COPYING or NOTICE path; GitHub's repository-license field is null. Thus model-use permission remains unresolved, rather than verified permissive.

The 135-byte Git LFS pointer binds an advertised **1,307,719,878-byte** checkpoint to SHA256 **01395a5326c9ddf7756eded2ce106db50268c7087d61bfb5f95ea0124666bde9**. The artifact itself was not fetched or loaded. Configuration selects `vitl16` for teacher and student; the README describes DINOv3-L continuation on musculoskeletal images. Corpus claims include public OAI/fastMRI and private clinical images. Actual checkpoint architecture, preprocessing, expanded-corpus provenance and competition-data overlap remain unverified. [Pinned repository](https://github.com/ytrsk/OrthoFoundation-MSK-5M/tree/df8f943460ffc452974869d33f15afe3674aa9f3).

**Next gate:** find explicit weight-use permission and adequate provenance through published sources, then verify artifact hash and bounded load/input compatibility. Do not start pretraining from scratch or presume DINOv3-L will improve our competition score.

## KneePreM

The paper links to FuseMedML's OAI example. Current API revision **fcb882bd7548073270853cffd7e06174fb4d8c79** (October 5) has Apache-2.0 repository metadata. The current README explicitly describes KneePreM's 3D masked-autoencoder workflow, `40 × 224 × 224` inputs, and loading user-supplied MAE checkpoints. It states that model checkpoints are not distributed with the repository. The SuPreM download link refers to a comparator, not KneePreM weights. This current README is more informative than the older browser-cached DINO description. [Pinned example](https://github.com/BiomedSciAI/fuse-med-ml/tree/fcb882bd7548073270853cffd7e06174fb4d8c79/fuse_examples/imaging/oai_example).

A usable KneePreM artifact, its license and provenance remain unverified. The paper's transfer results motivate an experiment; they cannot predict RSNA leaderboard gains. [Paper and reproducibility section](https://arxiv.org/pdf/2609.31461).

**Next gate:** verify an independently published, accessible pretrained checkpoint with documented permission. Until then, retain this as research rather than an immediate training candidate.

## Evidence and verification

Saved API receipts, untruncated tree, selected configurations, LFS pointer and current KneePreM README: `outputs/analysis/mri-domain-source-audit-20261011/`. Only small metadata/text files were retrieved. Local content checks verify immutable revisions, pointer size/hash and README checkpoint statement. No model, image preprocessing, dataset overlap or downstream accuracy test was performed. No messages sent to authors.
