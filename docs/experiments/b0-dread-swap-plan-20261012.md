# B0 plan: no-OAI Raptor arm swap (declared 2026-10-12, before execution)

B0 keeps the scored 0.954 entry (`ozgung/knee-public-s6-recovery-entry-20261012` v1, build `outputs/kaggle/public-s6-recovery-entry-20261012-reviewed-v1/`) and replaces only its OAI-trained original320 arm with a CC0, competition-only Dread Raptor checkpoint; S6 arm, 70:30 rank blend and capped recovery are unchanged. Evidence: `outputs/analysis/b0-dread-swap-20261012/` (cards, pulled sources, `candidates-summary.json`). Nothing was pushed, downloaded or executed.

## Candidates and provenance

All `dreaddevelopment` CoAtNet checkpoints are **CC0-1.0**, about 293 MB, same architecture (`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` plus per-finding attention over 3-slice windows) and same twelve-label order as our code. The author states (discussion 737696, 2026-09-04) that every checkpoint used the same label set and "same comp data, nothing external"; only geometry and window count change. Labels are LLM soft labels from competition reports (`rsna-knee-labels`, CC0). No OAI, NDA or acknowledgement text appears on any card or in either public notebook (SHA-256 unchanged since the October 8 pins). This is a declaration, not an attestation; ImageNet-12k timm pretraining is the only external component.

| Checkpoint | Author LB | Slices / span / stored px / windows |
|---|---|---|
| widedense `v4_full.pt` (SHA `89606f05…1ded`, local copy verified) | 0.924 @42; **0.927 @62** | 64 (18/14/12/8/12) / 6–94% / 336 / 42 or 62 |
| maxspan `v5_full_swa.pt` | 0.928 | 64 / 2–98% / 336 / 62 |
| native384 `v8_full_swa.pt` | 0.926 | 44 / 6–94% / 384 / 42 |
| native384dense `v10_full.pt` | untested | 64 / 2–98% / 384 / 62 |
| finespacing `v9_full.pt` | 0.932 | 80 (22/18/15/10/15) / 2–98% / 336 / 78 |
| fullspan v7, widefov v6, arms-x, arms, cnn336, EfficientNet | 0.69–0.917 | not candidates |

**Recommendation: `raptor_ft_coatnet_v4_full.pt` at 62 windows.** It is the only Dread checkpoint our code already runs: `knee/reference.py` has the `raptor` profile (`RAPTOR_RECIPE`, 42/62 windows), pinned source `859aae12…`, `raptor_input.py` guards, 16 stress cases and T4 receipts (`raptor62-rehearsal-result`; LB 0.937 in the ConvNeXt blend). The claimed 0.001 gap to v5 is not worth a new pin before one rehearsal. v5 is the follow-up: its span constant sits inside the pinned public `build_study`, so it needs a source patch, a new `extracted_module_sha256` pin and a test. v9 needs an 80-slot plan with no published inference source; defer.

## Contract comparison

| Field | original320 arm (nartaa, `mirror` profile) | Dread v4 (`raptor` profile) |
|---|---|---|
| Weights / dataset | `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`, nartaa, licence "other", OAI | `raptor_ft_coatnet_v4_full.pt`, `dreaddevelopment/raptor-knee-widedense`, CC0 |
| timm arch, head, outputs | same CoAtNet RMLP-2, RaptorClassifier, 12, same `LAB` order | same |
| Stored image / input | 384 px, 140 mm, **320 centre crop** | **336 px, 140 mm, bilinear resize to 384** |
| Slots / depth | 26/22/18/12/18 = **96**, span **2–98%** | 18/14/12/8/12 = **64**, span **6–94%** |
| Windows K | 94 | 42 or **62** |
| TTA | original + anatomical mirror, averaged | none |
| Precision | CUDA fp16 | CUDA fp16, no fp32 fallback |
| Decoder | pinned `study_lut` fastread | memoized pydicom |
| Normalization | ImageNet | ImageNet |

A weight-path swap inside the mirror arm **cannot** work, and fails loudly: `load_model` rejects the SHA and `CHECKPOINT_BYTES` (292,840,787 vs 292,829,402). If those pins were relaxed, the strict state load would succeed and the model would run on a 320-crop, 96-slot, mirrored input it never saw (14% scale change, cropped margins): silently wrong probabilities. B0 must run the old arm through the existing `raptor` profile, which is a code change, not a config change.

## Exact changes for a B0 build

1. `knee/public_s6_recovery_entry.py` `old_worker`: branch on `public_manifest['profile']`. For `raptor`: `raptor_input.prepare_study(public, sid, records, root)` (no decoder), one pass `public.eval_windows(v, m, 62, 384)`, `infer_checked(..., mirror=False)`, `raptor_input.stress_inputs` instead of `mirror_stress`, drop the full384 window audit; `install_old_guards` in `public_s6_recovery_runtime.py` must patch `raptor_input.validate_production_volume` and skip the `decoder.NeedMore` check. Bump `VERSION`, `METHOD['old']`.
2. `scripts/build_public_s6_recovery_entry.py`: slug `knee-public-s6-recovery-dread-b0-20261013`; `DATASETS` swaps `WEIGHT_DATASET` for `RAPTOR_WEIGHT_DATASET`; `public_manifest = audit_source(dread_source, profile='raptor', checkpoint_sha256='89606f05849838529e1b4658d28fb049623205d9853371548403bca460361ded', raptor_windows=62)`; `control_checkpoint` path under `/kaggle/input/datasets/dreaddevelopment/raptor-knee-widedense/`; new `control_state_sha256` and `control_grid_sha256` computed on CPU from the local copy; replace `CONTROL_*` pins and the 870-row `historical` old-sentinel array (nartaa-specific) with the `raptor62-rehearsal-result` receipt plus repeat/strict-source parity. S6 pins (`EVALUATION_*`, `PILOT_*`, `AUDIT_SHA`) and the frozen CPU gate stay, as `knee/public_s6_recovery.py` is untouched.
3. `knee/reference.py`: no change for v4@62.
4. Tests: extend `tests/test_public_s6_recovery_runtime.py` for the raptor-profile old arm; existing `test_reference` 62-window and non-OAI-profile tests cover the manifest.

## Verification before submission

1. **CPU fixture check:** unit tests plus one script against the local `v4_full.pt`: SHA match, strict load, `lab` order, windows shape `(62, 3, 384, 384)` from a synthetic 64x336x336 volume, finite 12 outputs, state fingerprint equal to the new pin. Dependency-free smoke.
2. **One Kaggle rehearsal** (`...-dread-b0-rehearsal-20261013`, 36 probes + 3 visible, T4x2, about 26 min, under 0.5 GPU h of the 15.02 h left before the October 17 reset): all 39 genuine in both arms, S6 raw probabilities byte-identical to the 0.954 rehearsal, old-arm strict comparisons and repeat parity pass, model state unchanged, projected conservative runtime under 9 h.
3. **Pure entry** v1 (visible, about 3 min), parent collects, then **one submission**. Record in a result doc and `STATUS.md`.

## Runtime

Measured on T4: original320 5.39 s/study (20-study shard); Dread v4 at 62 windows 4.32 s mean, 5.84 s p95 (42 windows: 2.88 s). Base projection 5.97 h drops to about 5.8 h; conservative 8.19 h to about 8.0 h. Peak CUDA rises to 11.8 GB on the old board (fits 15 GB). Decoders differ, so hidden-tail behaviour is not identical.

## Risks

- Score: Dread 0.927 against nartaa 0.949 on the 0.7 arm; B0 is a hedge, not an A candidate.
- Provenance is declared, not attested; the weights saw all 4,349 non-gold studies, so fixed-870 cannot rank B0.
- The S6 arm's external inputs (SKM-TEA localizer) are not re-audited; B0 is "no OAI", not "no external".
- 70:30 was tuned by the public for the stronger OAI arm; keep it fixed for B0 and treat S6-heavier weights as a separately qualified follow-up.
- The `raptor` profile has no fp32 fallback; a cuDNN fp16 failure aborts the arm and the recovery cap applies.

## Expected LB (speculation only)

Roughly 0.945 to 0.951, based on the two public arm scores and the earlier 0.937 Dread plus ConvNeXt blend. No forecast is claimed.
