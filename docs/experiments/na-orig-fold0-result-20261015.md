# na-orig fold-0 result, Oct 15, 2026: R1's global-view control

**What:** nartaa's R384 backbone (OAI, sha 7e5315da…) with the original raptor64 head and the in-distil loss (labels_v1_blend + 0.5 × the Dread v4_full K42 teacher). Global raptor64 cache, fold 0, frozen 12-epoch recipe, seed 42. It differs from `in-distil` only in the init.
- **Run:** Colab A100 `raptor64-na-orig-r2-20261015`, launched 19:15 UTC. Terminal at 22:26 UTC (phase `exported`); VM stopped.
- **Code:** branch `claude/na-orig-20261015` (8e4fd9c, d6209fe). The first session never trained because Kaggle now serves raw `.pt` files, which the setup step rejected; that's fixed in d6209fe.
- **Load:** missing 0, unexpected 0, 478 tensors, `res == 384`.

## Result: fold-0 K42 masked report macro

Paired whole-component bootstrap vs own44 (0.90588):

| Arm | Init | Teacher | Macro | Δ vs own44 [95% CI] |
|---|---|---|---|---|
| in-distil | ImageNet | Dread 0.5 | 0.91490 | +0.0090 [+0.0045, +0.0138] |
| **na-orig** | **nartaa R384** | Dread 0.5 | **0.94088** | +0.0350 [+0.0284, +0.0423] |
| dr-orig | Dread v4_full | none | 0.94708 | +0.0412 [+0.0344, +0.0484] |

Evaluation file: `outputs/analysis/raptor64-20261013/endpoint-na-orig-v1.json`.

## Reading
- **Both knee-pretrained inits are exposed.** They were trained on all data, including these 870 studies, so their margin over in-distil (+0.026 for na-orig) mixes real transfer with memorisation. It cannot be read as generalisation.
- **Report-label fold-0 scores don't track the leaderboard** (see `s6-labels-v2-main-lane-protocol-20261015.md`). This number is descriptive only.
- **na-orig is now R1's equal-exposure control.** The `na-local − na-orig` difference isolates the effect of the local view, once the local64 cache exists.
- R1 is the secondary lane. The main lane is S6 on labels v2.
