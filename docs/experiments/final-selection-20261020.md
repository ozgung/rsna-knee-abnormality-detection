# Final-selection evidence, October 20, 2026

All lanes of the [final-push plan](../research/final-push-plan-20261019.md) have finished. Times are UTC.

## Scored candidates

| Entry | Submission | Public | Compact slot | Provenance of the compact slot |
|---|---|---|---|---|
| **P2** (R6F-pure v2) | 57340072 | **0.958** | heliosli R6, flip TTA | heliosli bundle (R6 auxiliary data unresolved) |
| P6s′ (dual compact, no R224) | 57414571 | 0.957 | ½ R6 plain + ½ R7 plain, shared decode | heliosli R6 + R7 (fastMRI auxiliary) |
| A′-w | 57324282 | 0.957 | heliosli R7, no flip | heliosli R7 |
| P3 / P4 / P5 | 57366765 / 57368819 / 57368388 | 0.957 | R6 flip + na-full 0.1 | heliosli R6 |
| P7′ (all-ours compact) | 57432784 | 0.955 | mean rank of S6 v3-s42, v3-s43 and the fastMRI student | our fits; Raptor half still nartaa (OAI) |
| C1 solo v3-s43 | 57432734 | 0.955 | single S6 fit | ours |
| Lane D student solo | 57426580 | 0.954 | single S6 fit + fastMRI soft targets | ours (NYU fastMRI terms) |

Every Raptor half is byte-identical across these entries (R384 + R224 at 0.65/0.35 with the v1 per-finding weights; P6s′ has R384 only).

## Similarity to P2 and the value of a second pick

Per-finding Spearman on the 39 rehearsal studies (3 visible + 36 train probes), blends reproduced from the per-board ledgers with the sealed weights (`outputs/analysis/final-selection-20261020.json`):

| Pair | Mean ρ | Min ρ (finding) |
|---|---|---|
| P6s′ vs P2 | 0.991 | 0.972 (Lateral Meniscus) |
| P7′ vs P2 | 0.967 | 0.932 (ACL) |
| P7′ vs P6s′ | 0.962 | 0.918 (Lateral Meniscus) |

Expected gain in max(final 1, final 2) under the plan's model (private SD 0.0023, paired SD = 0.0023·√(2(1−ρ)), expected deficit = public difference):

| Final-2 candidate | Deficit | ρ | Paired SD | E[max] gain |
|---|---|---|---|---|
| P6s′ | −0.001 | 0.991 | 0.00032 | 0.000000 |
| A′-w | −0.001 | 0.97 (assumed) | 0.00056 | 0.000009 |
| P7′ | −0.003 | 0.967 | 0.00059 | 0.000000 (0.000011 if the true deficit were −0.001) |
| P3/P4/P5 | −0.001 | 0.995 | 0.00023 | 0.000000 |

No second pick adds a measurable chance of beating P2. The second slot is therefore insurance, not upside.

## Lane results that bear on the choice

- Lane B: on 4,349 held-out training studies, dual − R6 flip = +0.0029 [+0.0019, +0.0039] with single fold models. P6s′ nevertheless displayed 0.957, not 0.958: it dropped R224 (estimated −0.0003 to −0.0005), its hidden run may have had R7 guard-cut for the last studies, and 0.001 on the public board is inside noise. The out-of-fold gain is not refuted, but it did not show.
- Lane C/D: our own compact slot reaches 0.955 in the blend, 0.003 below P2. Seeds and the fastMRI student are individually 0.954–0.955 as solos (S6 v2 was 0.950).
- The October 20 provenance recheck was not adverse ([note](../research/provenance-refresh-20261020.md)), so the plan's override (rule 3) is not triggered. The per-asset questions on heliosli's and nartaa's weights remain unanswered by the host.

## Recommendation

- **Final 1 = P2 (57340072).** No challenger displayed ≥ 0.958.
- **Final 2:** the pre-registered rule 2 gives **P6s′** (≥ 0.957). Its E[max] contribution is zero, and it shares P2's heliosli compact bundles, so it insures nothing. **P7′ (57432784)** is the only scored entry whose compact slot is entirely ours; it is the better insurance if heliosli-bundle entries were ever set aside, at the same (zero) E[max] cost. This is a deviation from rule 2 on hedge grounds, not on score, and the user decides it on October 21.
- Nothing after this changes the recommendation unless a new entry displays ≥ 0.958 with every gate passed.
