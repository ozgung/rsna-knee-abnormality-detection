# Blinded LLM relabel pilot (WS-A, L1) — result, 2026-10-13

Pilot: 358 reports (gold-58 hidden among 300 non-gold, stratified by language, hashed IDs), rubric `relabel-rubric-v1-20261013.md`. Two independent blinded labelers ("opus" and "sonnet" Claude subagents) read only the rubric and their batches. All 12 label files are complete and valid. Scorer: `scripts/relabel_pilot_score.py`; receipt `outputs/labels/relabel-pilot-20261013/score_receipt.json`. This is the first of at most two pre-registered gold scorings.

| Check | Value | Threshold | Pass |
| --- | ---: | ---: | --- |
| Gold-58 macro AUC (labeler mean) against steven_v4 0.893 | 0.889 | ≥ 0.883 | yes |
| Worst finding against steven (findings with ≥ 12 positives) | Synovitis −0.060 | ≥ −0.050 | **no** |
| Directional agreement on 2,983 consensus cells | 0.988 | ≥ 0.90 | yes |
| Inter-rater state agreement | 0.950 | ≥ 0.95 | **no** (borderline) |
| Inter-rater p within 0.2 | 0.981 | ≥ 0.95 | yes |
| Minority-language agreement gap against English | −0.004 | ≥ −0.030 | yes |

**Decision: NO-GO.**
- **Per finding:** the Claude labels are statistically indistinguishable from the best public table. The SE of gold macro is about 0.017. Claude is ahead on MCL, Lateral OA and PF OA, and behind on Synovitis (0.730 against 0.790) and Contusion (0.825 against 0.860).
- **Synovitis:** the shortfall is rubric-driven. The rule "Hoffa fat-pad edema → synovitis positive" appears inconsistent with the gold convention.
- **Minority languages:** these are not a weakness of the public tables. Agreement is about equal across languages.
- **Conclusion:** report-derived labels look saturated at about 0.89 against gold, as the Fable review predicted. Relabeling is closed as a lever for a large gain. The second gold scoring is deliberately not spent, since any rubric revision could not show a meaningful difference at this SE.
