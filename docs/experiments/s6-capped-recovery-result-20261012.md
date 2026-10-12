# S6 capped recovery CPU sensitivity — October 12, 2026

All **13 prespecified scenarios PASS** against intact original320 (macro AUC **0.951330909**). The worst macro delta is **+0.006614209**, with paired 95% CI **[+0.002551465, +0.010174061]**, for 26 coincident ordinal failures. No scenario triggers the supported-target damage rule. This is controlled synthetic sensitivity on exposed development data, not natural-failure accuracy, independent generalization, hidden runtime qualification or a public-score forecast.

The zero-failure blend/rank arrays match the saved strict comparison exactly. The policy uses float64 target means from each arm’s genuine successful rows across the entire population. Genuine S6 values are source-six-decimal first; recovered S6 means are also six-decimal. Old320 precision remains intact. Whole missing rows are filled before whole-population average percentile ranks and fixed70:30 blending. Union cap is floor(0.03*N); explicit missing masks do not excuse genuine nonfinite predictions or fatal errors.

Masks were saved and hashed before truth extraction: protocol SHA256 `e11e611c1089f63bd2bfd90b7659ba276bf5b7c9ddf97d18c178df8922b99def`. Seed20261012 uses one NumPy default_rng permutation, prefixes8/26, and first-half old / second-half S6 for disjoint cases. Workload tail selects top26 by saved S6 window count descending with canonical ordinal ties. All policies, masks and implementation/source hashes are frozen; no outcome-dependent selection occurred.

| Scenario | Macro AUC | Delta | Paired 95% CI | Recovered union | Gate |
| --- | ---: | ---: | --- | ---: | --- |
| zero | 0.962963989 | +0.011633080 | [+0.009787471, +0.013714779] | 0 | PASS |
| ordinal_8_old_only | 0.961641421 | +0.010310512 | [+0.007412824, +0.012941111] | 8 | PASS |
| ordinal_8_s6_only | 0.962970063 | +0.011639154 | [+0.009753881, +0.013710385] | 8 | PASS |
| ordinal_8_coincident | 0.961237949 | +0.009907040 | [+0.006419129, +0.012814551] | 8 | PASS |
| ordinal_8_disjoint | 0.962629653 | +0.011298744 | [+0.009359766, +0.013431946] | 8 | PASS |
| ordinal_26_old_only | 0.959348297 | +0.008017388 | [+0.004781598, +0.010863508] | 26 | PASS |
| ordinal_26_s6_only | 0.963010684 | +0.011679776 | [+0.009772599, +0.013791953] | 26 | PASS |
| ordinal_26_coincident | 0.957945118 | +0.006614209 | [+0.002551465, +0.010174061] | 26 | PASS |
| ordinal_26_disjoint | 0.961213255 | +0.009882346 | [+0.007003828, +0.012496910] | 26 | PASS |
| workload_tail_26_old_only | 0.960621477 | +0.009290568 | [+0.006481001, +0.011993990] | 26 | PASS |
| workload_tail_26_s6_only | 0.963460334 | +0.012129426 | [+0.010203925, +0.014301474] | 26 | PASS |
| workload_tail_26_coincident | 0.959748463 | +0.008417555 | [+0.004978866, +0.011588843] | 26 | PASS |
| workload_tail_26_disjoint | 0.962124652 | +0.010793744 | [+0.008484314, +0.013196180] | 26 | PASS |

Each scenario uses the existing paired whole-component bootstrap, seed42 / 2,000 draws; all2,000 draws are valid and all843 original report components are preserved. Each passes positive macro delta, positive lower CI and fewer than three targets with at least20 observed positives and20 negatives declining by at least0.01. All12 targets are defined in every scenario. Macro policy: unweighted arithmetic mean over targets with both observed classes; unknown cells stay masked. Supports are identical across scenarios:

| Target | Positive | Negative | Observed | Missing | Intact old320 AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| ACL | 109 | 716 | 825 | 45 | 0.982022961 |
| MCL | 20 | 790 | 810 | 60 | 0.953037975 |
| Medial Meniscus | 330 | 496 | 826 | 44 | 0.966532258 |
| Lateral Meniscus | 109 | 713 | 822 | 48 | 0.963881776 |
| Medial OA | 150 | 662 | 812 | 58 | 0.957562941 |
| Lateral OA | 75 | 736 | 811 | 59 | 0.919021739 |
| PF OA | 169 | 646 | 815 | 55 | 0.922289190 |
| Effusion | 222 | 595 | 817 | 53 | 0.931364978 |
| Synovitis | 103 | 337 | 440 | 430 | 0.957218173 |
| Baker's | 113 | 434 | 547 | 323 | 0.958341829 |
| Contusion | 165 | 634 | 799 | 71 | 0.942873530 |
| Fracture | 52 | 742 | 794 | 76 | 0.961823554 |

All per-target scenario AUCs (same order as the scenario table):

| Target | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACL | 0.988327098 | 0.987090616 | 0.988352724 | 0.986930449 | 0.988263031 | 0.982971144 | 0.988314284 | 0.981689816 | 0.986309005 | 0.978864487 | 0.988641023 | 0.975923838 | 0.984105120 |
| MCL | 0.965063291 | 0.965316456 | 0.965316456 | 0.965569620 | 0.965253165 | 0.966139241 | 0.965696203 | 0.966898734 | 0.965949367 | 0.966772152 | 0.965727848 | 0.967405063 | 0.966518987 |
| Medial Meniscus | 0.974465420 | 0.974187439 | 0.974447092 | 0.973927786 | 0.974175220 | 0.973490958 | 0.974792278 | 0.971829179 | 0.974230205 | 0.974019428 | 0.974795332 | 0.972690616 | 0.974242424 |
| Lateral Meniscus | 0.971627829 | 0.970238172 | 0.971730767 | 0.969839289 | 0.971659997 | 0.969511175 | 0.972232587 | 0.968912850 | 0.970630621 | 0.964312827 | 0.971659997 | 0.962730162 | 0.968893550 |
| Medial OA | 0.967854985 | 0.966495468 | 0.967935549 | 0.966289023 | 0.968157100 | 0.962713998 | 0.968217523 | 0.960720040 | 0.965961732 | 0.966299094 | 0.967970796 | 0.965579053 | 0.966868077 |
| Lateral OA | 0.936530797 | 0.933668478 | 0.936503623 | 0.932934783 | 0.936730072 | 0.929365942 | 0.936467391 | 0.928052536 | 0.934085145 | 0.937780797 | 0.937137681 | 0.938315217 | 0.937500000 |
| PF OA | 0.944327404 | 0.943118325 | 0.944519757 | 0.942115339 | 0.943800722 | 0.939197978 | 0.944084672 | 0.936170700 | 0.942344331 | 0.941524539 | 0.944698371 | 0.940384157 | 0.942293953 |
| Effusion | 0.950511015 | 0.950147627 | 0.950541298 | 0.949685820 | 0.950897116 | 0.945313801 | 0.950594292 | 0.943818609 | 0.948660005 | 0.948315542 | 0.951347566 | 0.947819668 | 0.951533046 |
| Synovitis | 0.965918585 | 0.964996687 | 0.965774538 | 0.964506929 | 0.965803348 | 0.962504682 | 0.965400017 | 0.960776123 | 0.963786696 | 0.969102014 | 0.968367376 | 0.968410590 | 0.968237734 |
| Baker's | 0.962756413 | 0.959442926 | 0.962368990 | 0.958321439 | 0.960921251 | 0.957750500 | 0.962195669 | 0.955211859 | 0.957974797 | 0.958525346 | 0.963286571 | 0.957546593 | 0.960095428 |
| Contusion | 0.957303317 | 0.957054775 | 0.957657012 | 0.957585317 | 0.957967690 | 0.954593251 | 0.957394131 | 0.953670777 | 0.956376063 | 0.954143963 | 0.957035656 | 0.953130676 | 0.954507217 |
| Fracture | 0.970881713 | 0.967940079 | 0.970492950 | 0.967149596 | 0.967927120 | 0.968626892 | 0.970739166 | 0.967590193 | 0.968251089 | 0.967797533 | 0.970855795 | 0.967045926 | 0.970700290 |

CPU runtime: **4.942 seconds**. Saved exact array hashes, canonical870 membership, original truth fingerprint, source bindings, and component identities were checked. Existing miniconda PyArrow extracted truth after freeze; existing Python3.12 NumPy/Pandas environment scored it. No installs, GPU, training, submission or production strict-code edits.

Checks: focused `tests.test_public_s6_recovery` **8 PASS**, dependency-free smoke PASS, `git diff --check` PASS. Coverage includes union cap/actual-count floor, whole-row means and S6 rounding, ties/permutations, canonical two-shard merge/duplicates/coverage, both-arm failures, all-missing rejection, genuine nonfinite/out-of-range values and explicit fatal-error abort. These are array-policy tests; runtime failure classification and cleanup are not yet implemented or qualified.

Evidence under ignored `outputs/analysis/s6-capped-recovery-20261012/`: `protocol-private.json`, `protocol-freeze.json`, `truth-extraction.json`, `truth-components-private.npz`, and `result.json`. Full results include all AUC/support/undefined fields, arm genuine/recovered counts, means, mask hashes and paired bootstrap policies. New recipe `s6-fixed70-30-capped-arm-mean-v1` / receipt `s6-recovery-cpu-sensitivity-v1` cannot inherit an old strict PASS.

Next: parent numerical review, then explicitly classified data-local runtime recovery and fresh image-only injection rehearsal. Preserve fatal source/state/schema/path/resource errors, subsequent-study correctness and model-state parity. Public-source training/report/gold exposure and unavailable patient identities remain limitations.


Parent independent audit: **22 PASS / 0 FAIL**. Separate NumPy whole-population ranks and direct positive-negative pair comparisons reproduce all13 macro/per-target AUCs with maximum error0, exact zero-failure array parity, identical supports and explicit union/status counts. The worst scenario's full2,000-draw paired bootstrap repeated exactly. Frozen implementation bytes and seed/prefix/half masks verified. Receipt `outputs/analysis/s6-capped-recovery-20261012/parent-independent-audit.json`. Parent accepts CPU qualification only; separate runtime implementation remains in progress, with actual image-only injection rehearsal still required.


## Runtime candidate integrated — October12,03:33UTC

Separate version `s6-dual-capped-recovery-v1` preserves the strict production files and frozen CPU policy. Parent reran **58 focused tests, no skips**, smoke and diff checks; **18 integration checks PASS**, including16 unchanged numerical/input/device helper ASTs and frozen CPU/strict source bytes. Actual guarded CPU checks: mirror10 partial-input cases and native2 missing/empty companion cases. These do not exercise CUDA model inference.

Recovery is limited to audited source-site typed data errors and terminal unusable-image/window outcomes. Unknown/programming/generated-artifact/resource errors stay fatal before upstream stringification. The reader's pinned NeedMore control signal is preserved only at its original parser boundaries. Two pydicom3.0.2 pixel-validation raise sites are tagged after exact file/function verification; package files and successful numerical code are unchanged. Source/class identity, causes, decorators, reset/reinstall behavior and cleanup are checked. Baseline mirror17/native2 remain explicitly preguard checks, not claimed as new guarded coverage.

Fresh CUDA rehearsal must keep all39 main predictions genuine and historically unchanged, while each arm/board separately demonstrates a declared data failure, cleanup, next-healthy correctness and unchanged model state. Only the rehearsal extension contains injected faults; pure entry removes it and fixtures. No new competition submission until actual qualification and visible parity pass. Build review from isolated source passed16 checks/233 compiled sources; parent will rebuild after commit and verify accepted source before one launch. Evidence `outputs/analysis/s6-capped-recovery-20261012/runtime-handoff.json`, `parent-integration-review.json`, `guarded-mirror-cpu.json`, `guarded-native-cpu.json`.
