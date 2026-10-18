# CoAtNet384 recovery export — October 10, 2026

Selected model recovery and export succeeded without retraining. Kaggle `ozgung/knee-tracka-export-c384-fold0-20261010` v1 rebuilt the epoch13 backup's SWA [13,12,11] and scored all870 held-out fold0 studies. Raw-report masked macro AUC is **0.918185**; all12 targets are defined. This model used bootstrap-v1 targets and remains diagnostic rather than a default ensemble member.

Parent reproduced the28-check artifact/metric audit, then independently downloaded the292,832,595-byte exported selected checkpoint and verified its recorded file hash. All486 tensors exactly equal the previously recovered local model, with finite values and matching architecture,384px geometry,target order and selected epochs. Ten tensor/metadata checks PASS.

- Exported file SHA256: `03422ebdadbb44cd70a6262afefef174b91bf91a6e703e9c87a6418c84b89926`.
- Canonical tensor SHA256: `af33707bb08e4fc2bcdf76745e4cb208c89cbec029868e9bbff7c781972a2141`.
- Input resume SHA matches the parent epoch13/version14 recovery.
- Evidence and scripts: `outputs/analysis/coat384-export-audit-20261010/{audit.py,audit.json,verify_tensors.py,tensor-review.json}`.
- Result directory: `outputs/kaggle/trackA-export-c384-fold0-20261010-result/trackA/`.

| Report target | AUC | Positive | Negative | Unaddressed |
| --- | ---: | ---: | ---: | ---: |
| ACL | 0.952066 | 109 | 716 | 45 |
| MCL | 0.877785 | 20 | 790 | 60 |
| Medial Meniscus | 0.941422 | 330 | 496 | 44 |
| Lateral Meniscus | 0.892752 | 109 | 713 | 48 |
| Medial OA | 0.938177 | 150 | 662 | 58 |
| Lateral OA | 0.873043 | 75 | 736 | 59 |
| PF OA | 0.870830 | 169 | 646 | 55 |
| Effusion | 0.928458 | 222 | 595 | 53 |
| Synovitis | 0.958284 | 103 | 337 | 430 |
| Baker's | 0.955956 | 113 | 434 | 323 |
| Contusion | 0.912456 | 165 | 634 | 71 |
| Fracture | 0.916986 | 52 | 742 | 76 |

Macro aggregation is the unweighted mean over defined targets. Unknown report labels remain masked. The evaluated fold selected checkpoints and the global teacher adds indirect exposure, so this is not clean outer-fold evidence. No gold outcomes were recomputed or used in this audit's decision.

Preserve the original export receipt, but interpret its training metadata with these corrections: actual training used k12/24 and14epochs; exported defaults report k8 and10 requested epochs, zero training time, and a partial-fit flag. Its source hashes describe the export package, not verified original training sources. The original state omits full experiment identity hashes. Exact selected-weight recovery and evaluation are now verified; optimizer continuation and missing provenance remain unverified.

The original-label384 run is a separately identified12epoch experiment with other recipe differences from224. Do not describe comparisons as isolating resolution or labels alone. Next: obtain its finished selected checkpoint/export and compare report metrics under the documented limitations, without promoting models through gold-based selection.
