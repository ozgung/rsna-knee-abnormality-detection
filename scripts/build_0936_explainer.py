"""Build a private, illustrated notebook from the immutable scored v2 payload.

Requires matplotlib/numpy for original figures; nbformat/nbconvert are optional QA.
Generated notebook includes competition-derived MRI examples and MUST stay ignored.
No inference, model fitting, uploading, or submission is performed by this builder.
"""
from pathlib import Path
import base64
import csv
import hashlib
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/notebooks/rsna-knee-0936'
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = OUT / 'figures'
ASSETS.mkdir(exist_ok=True)
SOURCE = ROOT / 'outputs/kaggle/reference-blend-submission-v2/notebook.ipynb'
original = json.loads(SOURCE.read_text())
score_receipt = json.loads((ROOT / 'outputs/research/2026-10-08/blend-v2-scored-1257.json').read_text())
entry = next(x for x in score_receipt['submissions'] if x['ref'] == 56929332)
assert entry['public_score'] == '0.936' and not entry['error_description']
parity = json.loads((ROOT / 'outputs/kaggle/reference-blend-submission-v2-result/qualification-parity.json').read_text())
assert parity['status'] == 'PASS' and parity['csv_sha256'] == '2aa5bde70b03e720b21435b9137d9b7560ff4256c4b6ae7059e12538f1fb9d25'
with (ROOT / 'data/raw/sample_submission.csv').open() as f:
    targets = next(csv.reader(f))[1:]
assert len(targets) == 12
INK, TEAL, ORANGE, PALE = '#172d3b', '#087f8c', '#d77236', '#edf4f6'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'text.color': INK,
    'axes.labelcolor': INK, 'axes.edgecolor': '#d1dce1', 'xtick.color': INK,
    'ytick.color': INK, 'figure.facecolor': 'white', 'axes.facecolor': 'white'})

def save(fig, name):
    fig.savefig(ASSETS / f'{name}.png', dpi=115, bbox_inches='tight', facecolor='white')
    fig.savefig(ASSETS / f'{name}.svg', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def box(ax, xy, w, h, title, detail, color=TEAL):
    x,y=xy
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.018',
        linewidth=1.2,edgecolor=color,facecolor=PALE if color==TEAL else '#fcf2eb'))
    ax.text(x+w/2,y+h*.68,title,ha='center',va='center',weight='bold',fontsize=11,color=color)
    ax.text(x+w/2,y+h*.29,detail,ha='center',va='center',fontsize=9,linespacing=1.5)

def arrow(ax, a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=INK,lw=1.5))

# Real MRI illustrations from the two already-verified local QA caches. Never embed IDs.
qa = []
for p in sorted((ROOT/'outputs/kaggle/fresh-cache-pilot-result/fresh_reader_cache/arrays').glob('*.npz')):
    with np.load(p,allow_pickle=False) as z:
        if z['mask'][:3].all(): qa.append(z['x'].copy())
assert qa, 'A verified sample with all three planes is required'
x=qa[0]
fig,axes=plt.subplots(1,3,figsize=(9.4,3.15))
for i,(ax,name,view) in enumerate(zip(axes,['Sagittal · side view','Coronal · front view','Axial · cross-section'],['side','front','cross'])):
    ax.imshow(x[i,1,1],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    ax.set_title(name,fontsize=11,pad=9);ax.axis('off')
fig.suptitle('One knee, three complementary imaging planes',x=.5,y=1.02,fontsize=16,weight='bold')
fig.text(.5,-.015,'Real competition MRI · illustrative 224px QA cache · no diagnosis or lesion annotation',ha='center',fontsize=9)
save(fig,'mri-planes')

fig,ax=plt.subplots(figsize=(10.8,5.8));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.02,.96,'From a study to 12 ranked confidence scores',fontsize=18,weight='bold')
box(ax,(.02,.40),.19,.23,'MRI study','DICOM slices\n+ series metadata')
box(ax,(.30,.67),.30,.20,'ConvNeXt-Tiny × 3','6 slots · 12 windows/slot\n256px · fold probability mean')
box(ax,(.30,.25),.30,.20,'Raptor / CoAtNet × 1','5 slots · 64 sampled frames\n42 windows · 384px',ORANGE)
box(ax,(.70,.45),.27,.23,'Equal rank blend','Rank each target across studies\n50% ConvNeXt + 50% Raptor')
arrow(ax,(.21,.57),(.30,.77));arrow(ax,(.21,.48),(.30,.35))
arrow(ax,(.60,.77),(.70,.60));arrow(ax,(.60,.35),(.70,.51))
ax.text(.43,.60,'12 sigmoid scores per study',ha='center',fontsize=9,color=TEAL)
ax.text(.45,.18,'12 sigmoid scores per study',ha='center',fontsize=9,color=ORANGE)
arrow(ax,(.835,.45),(.835,.23));ax.text(.835,.16,'submission.csv',ha='center',fontsize=12,weight='bold')
ax.text(.835,.10,'One row per actual test study',ha='center',fontsize=9)
ax.text(.02,.035,'Image-only inference · fixed weights · no test reports · no inference-time fitting',fontsize=10)
save(fig,'pipeline')

fig,ax=plt.subplots(figsize=(10,3.35));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.015,.91,'Why “2.5D”?',fontsize=18,weight='bold')
for i,label in enumerate(['Slice z − 1','Slice z','Slice z + 1']):
    ax.add_patch(FancyBboxPatch((.035+i*.085,.28+i*.065),.18,.37,boxstyle='round,pad=.008',
        facecolor=['#e3eef2','#bfdae2','#8fc1ce'][i],edgecolor=TEAL,lw=1))
    ax.text(.125+i*.085,.34+i*.065,label,ha='center',fontsize=10)
arrow(ax,(.40,.48),(.50,.48))
box(ax,(.51,.29),.20,.36,'2D encoder','Three neighboring frames\nact as input channels')
arrow(ax,(.72,.48),(.80,.48))
ax.text(.90,.53,'Window feature',ha='center',weight='bold',fontsize=11)
ax.text(.90,.38,'Repeat across study;\npool by finding',ha='center',fontsize=10)
ax.text(.02,.035,'Channels are neighboring processed frames, not RGB colors. Schematic; stacks may be subsampled.',fontsize=10)
save(fig,'windows')

fig,ax=plt.subplots(figsize=(9.8,3.5))
labels=['Frozen DINO heads\nEarlier project baseline','ConvNeXt-Tiny × 3\nOur reproduced standalone','ConvNeXt + Raptor\nOur fixed equal-rank blend']
values=[.757,.929,.936]
ys=np.arange(3)
ax.barh(ys,values,color=['#a8b9c3',TEAL,ORANGE],height=.52)
for y,v in zip(ys,values):ax.text(v+.004,y,f'{v:.3f}',va='center',weight='bold',fontsize=12)
ax.set_yticks(ys,labels);ax.invert_yaxis();ax.set_xlim(.5,1.0)
ax.axvline(.950,color=INK,ls='--',lw=1.2);ax.text(.953,0,'0.950 target',fontsize=9,va='center')
ax.set_xlabel('Public leaderboard macro ROC-AUC · axis shown from 0.50 to 1.00')
ax.set_title('Verified submissions: 0.936 is the current best',loc='left',fontsize=16,weight='bold',pad=18)
ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
fig.text(.01,-.09,'Observed scores, not controlled architecture ablations. Hidden per-target AUCs and uncertainty are unavailable.',fontsize=9)
save(fig,'results')

# A deliberately synthetic, exact-tie example; not extracted from any patient predictions.
a=np.array([.1,.9,.3,.7]);b=np.array([.4,.6,.2,.6])
def rankpct(v):
    return np.array([(np.sum(v<t)+1+np.sum(v<=t))/(2*len(v)) for t in v])
ra,rb=rankpct(a),rankpct(b);blend=(ra+rb)/2
fig,ax=plt.subplots(figsize=(9.4,3.25));ax.axis('off')
ax.set_title('Rank blending, one target and four synthetic studies',loc='left',fontsize=16,weight='bold',pad=16)
rows=[[f'Study {chr(65+i)}',f'{a[i]:.2f}',f'{b[i]:.2f}',f'{ra[i]:.3f}',f'{rb[i]:.3f}',f'{blend[i]:.4f}'] for i in range(4)]
t=ax.table(cellText=rows,colLabels=['Synthetic row','ConvNeXt p','Raptor p','ConvNeXt rank/N','Raptor rank/N','50/50 blend'],
    cellLoc='center',loc='center',colWidths=[.15,.14,.14,.20,.19,.16])
t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,1.9)
for (r,c),cell in t.get_celld().items():
    cell.set_edgecolor('white')
    cell.set_facecolor(INK if r==0 else (PALE if r%2 else '#f7fafb'))
    if r==0:cell.set_text_props(color='white',weight='bold',fontsize=9)
    if c==5 and r:cell.set_text_props(color=ORANGE,weight='bold')
fig.text(.02,-.025,'Raptor ties at 0.60 receive average rank 3.5/4 = 0.875. No labels are used.',fontsize=10)
save(fig,'rank-example')

cells=[]
def md(body, **metadata):
    body=textwrap.dedent(body).strip()+'\n'
    cells.append({'cell_type':'markdown','id':f'text-{len(cells):02d}','metadata':metadata,'source':body.splitlines(True)})
def figure(name,alt,caption):
    data=base64.b64encode((ASSETS/f'{name}.png').read_bytes()).decode()
    cells.append({'cell_type':'markdown','id':f'figure-{name}','metadata':{},
        'attachments':{f'{name}.png':{'image/png':data}},
        'source':[f'![{alt}](attachment:{name}.png)\n\n',caption+'\n']})

md('''
# Knee MRI abnormality detection
## An illustrated guide to our 0.936 public-score ensemble

**Result verified October 8, 2026 · Submission 56929332 · Private research notebook**

Given a knee MRI examination, predict twelve findings. Our current best entry combines
three public **ConvNeXt-Tiny** fold checkpoints with one public **Raptor CoAtNet** checkpoint,
then averages their per-target percentile ranks with equal weights.
The scored entry achieved **0.936 public macro ROC-AUC**, versus **0.929** for our reproduced
ConvNeXt component alone. The goal of exceeding 0.950 remains unmet.

This notebook explains the problem and exact submitted method, then preserves the scored
inference code for reproduction. We reused community-trained weights; **we did not train
these four checkpoints**. Our separate fresh-training experiments are future work and did
not produce this score.

**Reading guide:** task → MRI views → model branches → rank blend → measured results →
reliability and limitations → executable reproduction → conclusion and next directions.
''')
md('''
## 1. What must the model predict?

A **study** is one MRI examination. It contains multiple **series** (acquisitions in different
planes or with different contrast settings); each series contains individual DICOM slices.
The output is one row per study with twelve confidence scores. Several findings can coexist,
so this is **multi-label classification**, not a choice among twelve exclusive classes.

The audited training metadata contains **4,407 studies and 24,371 series**. Only **58 studies**
have the full set of official binary labels. Training reports can provide additional, noisy
supervision. **Reports are unavailable at test time:** the submitted pipeline reads images
and series metadata only. The visible test has three example studies; scoring replaces them
with the actual hidden test, described by the organizers as approximately 1,300 studies.

Sources: [official task](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview),
[official data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data).
Counts above were checked against the real competition CSVs on October 8.
''')
definitions=[('ACL','Anterior cruciate ligament injury'),('MCL','Medial collateral ligament injury'),
('Medial Meniscus','Medial meniscus tear'),('Lateral Meniscus','Lateral meniscus tear'),
('Medial OA','Osteoarthritis in the medial tibiofemoral compartment'),('Lateral OA','Osteoarthritis in the lateral tibiofemoral compartment'),
('PF OA','Patellofemoral osteoarthritis'),('Effusion','Excess joint fluid'),('Synovitis','Inflammation of the joint lining'),
("Baker's","Baker’s cyst"),('Contusion','Bone bruise'),('Fracture','Fracture')]
assert [x[0] for x in definitions]==targets
md('### Exact target order\n\n| Submission column | Finding |\n|---|---|\n'+'\n'.join(f'| `{k}` | {v} |' for k,v in definitions)+
   '\n\nThe leading identifier column is `StudyInstanceUID`. These names and their order come from the actual sample submission.')
md(r'''
### The metric rewards ranking

The competition metric is the unweighted mean of the twelve ROC-AUCs:

$$\mathrm{Score}=\frac{1}{12}\sum_{c=1}^{12}\mathrm{AUC}_c.$$

For a given finding, AUC measures how often an affected study is ranked above an unaffected
study, with half-credit for ties. Every target contributes equally, regardless of prevalence.
A score of **0.936 is not 93.6% diagnostic accuracy**. AUC does not establish sensitivity at a
chosen threshold, calibrated disease probabilities, or clinical performance.

Source: [official evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation).
''')
md('''
## 2. Why MRI needs more than a single image

Sagittal, coronal and axial acquisitions offer complementary views. Fluid-sensitive sequences
and fat suppression emphasize different aspects of the scan; the metadata flags
`Fluid_Sensitive` and `Fat_Suppression` are related but **not interchangeable**. Protocols,
image orientations, slice counts, pixel spacing and missing acquisitions vary across studies.

A useful reader must combine evidence across slices and series while retaining enough detail
for small structures. It must also handle absent or partly unreadable inputs without producing
a confident-looking default prediction.
''')
figure('mri-planes','Sagittal, coronal and axial MRI examples from one anonymized study',
    '**Figure 1. Real MRI context.** These unlabeled examples come from our already-verified 224px training QA cache. They illustrate the imaging planes; they are not exact tensors from either scored branch, diagnostic annotations, or model explanations. No study identifiers or reports are displayed. Keep this competition-derived illustration within the authorized private workspace.')
figure('windows','Three adjacent grayscale slices become three input channels to a two-dimensional encoder',
    '**Figure 2. 2.5D representation.** A 2D backbone receives three neighboring grayscale frames in a processed stack as channels. They need not be consecutive acquired slices: the stack may be subsampled, and Raptor windows can cross series slots. Repeating this over the examination adds local depth context without a full 3D backbone. Sampling and aggregation determine which evidence reaches the classifier.')
md('''
## 3. The exact ensemble that scored 0.936

A few terms make the architecture easier to follow:

- A **checkpoint** is a file containing the model's learned weights.
- An **encoder** converts an image window into a numerical feature vector.
- **Attention pooling** learns how much each window should contribute to the feature summary
  for a particular finding. It is not a verified lesion explanation.
- A **sigmoid** maps a model's raw score (logit) into the range 0–1.
- **EMA** means exponential moving average: a smoothed copy of weights accumulated during training.
- **BCE**, binary cross-entropy, is a training/evaluation loss; lower is better.

Both branches are trained MRI readers with finding-specific attention pooling. They differ in
backbone, spatial preprocessing, series selection and coverage, giving a plausible source of
complementary errors. The measured blend gain supports trying the combination; it does not
identify which difference caused the gain.

The blend weights were fixed at **50/50 before submitting**. There is no fitted meta-model,
threshold tuning, test-time augmentation, or report processing in this inference pipeline.
''')
figure('pipeline','Two independent MRI readers feed an equal per-target percentile-rank blend',
    '**Figure 3. Submitted inference flow.** ConvNeXt first averages the sigmoid probabilities from its three checkpoints. Those averaged probabilities and the single Raptor output are separately ranked across the full actual test cohort for each target, then combined equally.')
md('''
## 4. Branch A — ConvNeXt-Tiny with study-level attention

**Input selection.** The reader organizes series into six slots: sagittal, coronal and axial,
first with fat suppression and then without. It selects one candidate per slot by the reviewed
slice-count ordering. Missing slots are explicitly masked. Usable partially decoded series are
retained if at least three slices remain; the wrapper does not try a different candidate after
a selected series fails.

**Spatial and depth processing.** Public preprocessing uses DICOM geometry for canonical
ordering/orientation when available, a **153.6 mm** field of view represented at **384 × 384**
(**0.4 mm/pixel**), and at most **64 slices per series**. Intensities are normalized per series
using the 1st and 99.5th percentiles. The model samples **12 three-slice windows per slot** over
the 4–96% depth interval, takes a centered **0.92** field-of-view crop, and uses **256px** model
inputs with ImageNet channel normalization. Six present slots yield 72 windows per checkpoint.

**Model.** Each window passes through ConvNeXt-Tiny. Its feature is projected to 384 dimensions,
then tagged with learned slot and slice-position embeddings. A two-layer, eight-head transformer
mixes window tokens across the study. Each of the twelve findings has its own attention pooling
and classification weights, followed by a sigmoid.

**Ensemble within this branch.** We load `cnxt_v0_fold0.pt`, `cnxt_v0_fold1.pt` and
`cnxt_v0_fold2.pt`, and average their **probabilities**, not logits. These are three selected
folds of the author's five-fold normalized-report grouping scheme—not a complete three-fold
cross-validation estimate. The public recipe saved the final EMA after 14 epochs. We inspected
and reproduced inference, not the full upstream training process.

Credit: [goodpjw2008’s ConvNeXt notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944)
and [Apache-2.0 model/code asset](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader).
The source title’s **0.944** refers to its larger stack; our three-Tiny-checkpoint reproduction scored **0.929**.
''')
md('''
## 5. Branch B — plain Raptor CoAtNet

This branch uses one `raptor_ft_coatnet_v4_full.pt` checkpoint with backbone
`coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, a convolution/attention hybrid. It is the **plain Raptor**
reader, not the separate OAI-derived mirror model.

**Different coverage.** Five slots use `Fluid_Sensitive` preferences: sagittal fluid-sensitive
**18 frames**, sagittal non-fluid-sensitive **14**, coronal fluid-sensitive **12**, coronal
non-fluid-sensitive **8**, and axial **12**, for **64 sampled frames** in a concatenated volume.
The public series-selection preferences are retained. Within each selected series, frame indices
span approximately **6–94%** of its ordered stack. Intensities are normalized using the
selected-series **2nd/98th percentiles**; frames receive a nominal **140 mm** central crop,
limited by available image dimensions, and are prepared at **336px**. Three-frame windows
are then resized to **384px** for inference.

**Windows and pooling.** The scorer selects **42 three-frame windows** over the global valid
range of the concatenated volume. This detail matters: the public sampler can cross slot
boundaries, include interior zero slots, or repeat centers. We preserve that behavior rather
than describing it as 42 independent anatomical locations. CoAtNet features pass through
layer normalization and twelve separate attention-pooling heads, then sigmoid classification.
Unlike the ConvNeXt branch, this head has no added study-level transformer or slot embeddings.

Our audited wrapper runs the qualified CUDA half-precision path with finite-output checks.
It does not use the original notebook's error-to-prior behavior or silently retry in float32.

Credit: [Dread Development / Johnathan Wagner’s source](https://www.kaggle.com/code/dreaddevelopment/knee-mri-twelve-findings-from-a-single-model)
and [CC0 Raptor weights](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense).
The author reports competition-image/report-label training; a complete independent reconstruction
of upstream training history is unavailable.
''')
md(r'''
## 6. Why average ranks instead of raw probabilities?

Different readers can assign different numerical probability scales. Because ROC-AUC depends
on ordering, we convert each reader's scores into percentile ranks **separately for each target**.
For $N$ actual test studies, let $r^A_{ic}$ and $r^B_{ic}$ be ascending ranks, starting at 1,
with average ranks for exact ties. The submitted confidence is:

$$s_{ic}=\tfrac12\frac{r^A_{ic}}{N}+\tfrac12\frac{r^B_{ic}}{N}.$$

The rank is over studies, never over the twelve findings within one study. A completely tied
column receives $(N+1)/(2N)$; a one-study cohort receives 1. The resulting numbers satisfy the
submission range but are **rank-derived confidence scores, not calibrated disease probabilities**.
Rank transformation preserves each component's ordering; averaging the two can change it.
There is no guarantee that a rank blend improves AUC.

The full actual test cohort defines the ranks. Recomputing them for a subset can change both
numerical outputs and the combined ordering, so inference must collect all test predictions
before blending. No labels are used to compute these ranks.
''')
figure('rank-example','Worked numerical rank blend for four synthetic studies, including a tie',
    '**Figure 4. Synthetic worked example.** Ties use their average rank. These rows are invented solely to explain the calculation; they are not patient predictions or evaluation results.')
md('''
### Minimal implementation of the blend

The following educational excerpt expresses the same ranking rule. The preserved executable
payload below contains the validated production implementation and its shape/range checks.

```python
import numpy as np
from scipy.stats import rankdata

# Arrays have shape [number_of_actual_test_studies, 12].
# ConvNeXt probabilities have already been averaged across its three folds.
n = len(convnext_probabilities)
convnext_ranks = rankdata(convnext_probabilities, axis=0, method="average") / n
raptor_ranks = rankdata(raptor_probabilities, axis=0, method="average") / n
submission_scores = 0.5 * convnext_ranks + 0.5 * raptor_ranks
```

This excerpt is explanatory, not an additional dependency or replacement for the pinned runtime.
''')
md('''
## 7. Results we actually measured

| Entry | Public macro AUC | Evidence |
|---|---:|---|
| Earlier frozen DINO four-head baseline | 0.757 | Submission 56923149 |
| Reproduced three-fold ConvNeXt-Tiny reader | 0.929 | Submission 56927932 |
| Fixed 50/50 ConvNeXt + plain Raptor rank blend | **0.936** | Submission **56929332** |

The blend improves the observed public score by **0.007** over the standalone ConvNeXt entry.
Both corrected entries completed without an error field in the authenticated submission receipt.
This is a public leaderboard result, not a private leaderboard score. We do not have hidden
labels, per-target hidden AUCs or confidence intervals; this notebook therefore does not invent
ROC curves, target breakdowns, significance claims or diagnostic example predictions.
''')
figure('results','Three verified public scores: 0.757, 0.929 and 0.936, below a target line at 0.950',
    '**Figure 5. Verified project progression.** Models, training data and preprocessing differ across entries; this is a record of submissions, not a controlled ablation of individual design choices. The plotted axis begins at 0.50.')
md('''
### Execution and robustness evidence

| Check | Observed result | What it establishes |
|---|---|---|
| Final blend visible run | **61.77 s**, 3 example studies | Successful offline execution on the small visible test |
| Peak CUDA allocated memory | **8.18 GB** / 7.62 GiB | Measured device allocation for that visible run; not total RAM or hidden-run peak |
| Expanded ConvNeXt rehearsal | 67 studies, 319 series, 7 stress cases; **403.31 s** | Broader decoder/missing-input checks |
| Expanded Raptor rehearsal | 67 studies, 4,130 unique slices, 16 stress cases; **368.30 s** | Production decoding, strict healthy-input comparisons and failure policy |
| Visible final CSV parity | **Byte-identical** to the qualified blend output | Packaging preserved predictions on the visible examples |
| Corrected hidden submission | **Completed; public score 0.936** | The scoring rerun completed successfully |

The 61.77-second figure is **not** the hidden-test runtime or a full-test benchmark. Rehearsal
projections included runtime margin, but the hidden duration and memory peak are unavailable.
The qualified environment was Python 3.12, PyTorch 2.10.0+cu128, timm 1.0.26 and T4 CUDA,
with attached, hash-checked offline DICOM codec wheels and internet disabled.
''')
md('''
## 8. What our engineering changed

Initial public-reader wrappers passed the three visible studies but failed hidden reruns with
generic notebook exceptions. The exact hidden trace was unavailable. The audit identified
concrete input-handling defects and repaired them without changing healthy-input model math:

- ConvNeXt retains usable partial series and masks unusable selected slots. It does not silently
  switch to another candidate or return a prior for a completely unreadable examination.
- Raptor preserves the public geometry defaults, partial-frame behavior and sampling semantics.
  Resource failures propagate rather than being swallowed by broad decoder exception handlers.
- Source code, weights, codec wheels and the common runtime are pinned. Both branches are
  independently rehearsed and checked for finite outputs before the blend can be packaged.
- Submission generation preserves official target order and exact actual-test study coverage,
  rejects invalid scores, and refuses to overwrite an existing final CSV.

The corrected entries scored successfully. That confirms the repaired versions ran on the hidden
set; it does not prove which repaired defect caused each earlier hidden failure.

Within-run repeats were exact in the rehearsals. Small Raptor probability differences across
separate runs were observed (up to about **4.04 × 10⁻⁵**); the three visible-study rankings were
unchanged. Hidden rank invariance across runs is not established.
''')
md('''
## 9. Interpretation and remaining limitations

**The gain is promising but modest.** A +0.007 public improvement is evidence for this fixed
combination on the current leaderboard split, not proof of an optimal blend or future generalization.
Repeated leaderboard-driven tuning would make that estimate increasingly optimistic.

**Public weights have a different validation history from our own experiments.** Their authors
used report-derived labels and public labeled studies during development. We cannot turn those
same studies into an independent clinical validation set for these already-trained readers.
Our upstream provenance inspection is not a full audit of every training decision.

**MRI inputs are heterogeneous.** Missing series, imperfect geometry, compressed DICOMs and
small findings remain practical failure modes. Two-dimensional windows may miss useful depth
context, and the Raptor concatenation can mix adjacent slots. More coverage costs time and memory.

**AUC alone is insufficient for deployment.** Calibration, clinically chosen operating points,
site/patient independence and external validation remain unestablished. This is a competition
research result, not a validated clinical diagnostic system.
''')
md('''
## 10. Reproduce the scored inference

The code cell below is **byte-for-byte identical** to the executable cell from our scored v2
notebook. It contains the audited source bundle, component definitions, build contract and
entrypoint. Keeping this appendix intact avoids changing preprocessing while improving the
explanation. The new notebook itself has **not** been submitted or rerun on GPU; the 0.936 result
belongs to the source v2 entry. Markdown and attached illustrations do not execute.

On Kaggle, use a **private GPU notebook**, disable internet, and attach:

1. [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
2. [ConvNeXt reader dataset](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader)
3. [Plain Raptor dataset](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense)

The accompanying `kernel-metadata.json` retains the source notebook's exact Docker image and
attachments, under a distinct private notebook ID. Runtime checks intentionally fail if the
qualified Python/PyTorch/timm versions or pinned files have changed. Start from a clean working
directory: the entry refuses to overwrite `/kaggle/working/submission.csv`. No training reports,
weak-label dataset, external model download or internet access is required for this inference.

Expected outputs are `/kaggle/working/submission.csv` and the audit artifacts under
`/kaggle/working/reference-blend/`. Visible execution covers only the three supplied examples;
Kaggle replaces the inputs during a competition submission.
''')
md('''
### Provenance and integrity

- Scored source: [our fixed-blend notebook, version 2](https://www.kaggle.com/code/ozgung/knee-reference-blend-submission-20261008) — select saved **version 2** in Kaggle. The local preserved v2 payload is the authority for this artifact.
- Scored entry: `56929332`; score checked **2026-10-08 12:57:55 UTC**.
- Scored source revision: `962ab80`; branch recipe: equal per-target rank average, no TTA.
- Qualification SHA-256: `1dee9b7992a4e0e0af350c62cbef29a4f8b4615eb310ccb326872eda363360ca`.
- Visible CSV SHA-256: `2aa5bde70b03e720b21435b9137d9b7560ff4256c4b6ae7059e12538f1fb9d25`.

The companion `build-receipt.json` records the original and illustrated notebook hashes,
executable-cell hash, illustration hashes and exact executable parity. The MRI illustration is
competition-derived and this artifact stays private. Source/model licenses retain their authors'
terms; attaching explanatory prose does not change them.
''')
for c in original['cells']:
    if c['cell_type']=='code':
        preserved=json.loads(json.dumps(c))
        preserved['metadata'].update({'jupyter':{'source_hidden':True},'tags':['scored-payload','hide-input']})
        preserved['outputs']=[];preserved['execution_count']=None
        cells.append(preserved)
md('''
## 11. Conclusion and next steps

**A fixed blend of complementary public MRI readers reached 0.936 public macro ROC-AUC.**
The reproducible ingredients are three ConvNeXt-Tiny checkpoints, one plain Raptor checkpoint,
faithful preprocessing, robust image-only inference and equal per-target percentile-rank averaging.
The strongest verified comparison is the **0.007** gain over our 0.929 ConvNeXt reproduction.
Exceeding **0.950** remains a research target, not a promised outcome.

The next experiments should earn additional compute through bounded comparisons:

1. **Establish useful learning in our own reader.** A separate fresh four-epoch ConvNeXt pilot
   improved throughout training but remained below our frozen-feature weak-validation baseline.
   A reviewed twelve-epoch experiment keeps the same 512 training / 200 development studies,
   inputs and labels, changing the training horizon and its cosine schedule. It has a measured
   timing gate and a 90-minute application cap. It is not part of the 0.936 submission.
2. **Scale only after a validation gate passes.** The prospective gate uses the same BCE-selected
   checkpoint: weak-validation BCE ≤0.456285 and AUC ≥0.850340. These are spending heuristics,
   not clinical or leaderboard guarantees. A full metadata/header audit has already frozen 300
   additional studies in whole report/patient-link groups for later evaluation of fixed recipes;
   those images have not been used for candidate scoring. Unique study-level anonymized patient
   IDs still cannot prove repeat-patient independence.
3. **Test coverage at small scale.** A Raptor 42→62-window variant is a concrete hypothesis, with
   author-reported gains elsewhere. First measure runtime, memory, output integrity and paired
   behavior with unchanged weights. Any new score must come from a separately qualified entry.
4. **Add diversity only when justified.** Partial adaptation of a generic DINOv2 backbone could
   provide a complementary representation. Compare it on the same development protocol before
   paying for larger runs or adding an ensemble branch. Improve weak-label uncertainty handling
   and leakage controls alongside architecture work.

Preserve the scored baseline, spend the remaining submission budget on justified candidates,
and report negative results as carefully as improvements. The immediate objective is a stronger
validated reader and a reliable offline entry; the leaderboard will determine whether that work
moves beyond 0.950.
''')
nb={'nbformat':4,'nbformat_minor':5,'metadata':original['metadata'],'cells':cells}
nb['metadata']['title']='Knee MRI — the 0.936 equal-rank ensemble'
nb['metadata']['description']='Illustrated private research notebook; exact scored v2 executable preserved.'
original_code=''.join(''.join(c['source']) for c in original['cells'] if c['cell_type']=='code')
new_code=''.join(''.join(c['source']) for c in cells if c['cell_type']=='code')
assert original_code==new_code
compile(new_code,'<preserved-scored-v2>','exec')
path=OUT/'rsna-knee-0936-explained.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
metadata=json.loads((SOURCE.parent/'kernel-metadata.json').read_text())
metadata.update(id='ozgung/rsna-knee-0936-explained',title='Knee MRI: Our 0.936 Ensemble Explained',code_file=path.name,is_private=True)
(OUT/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'status':'PASS','source_notebook_sha256':sha(SOURCE),'illustrated_notebook_sha256':sha(path),
    'executable_sha256':hashlib.sha256(new_code.encode()).hexdigest(),'executable_byte_parity':True,
    'code_cells':sum(c['cell_type']=='code' for c in cells),'total_cells':len(cells),
    'bytes':path.stat().st_size,'scored_submission':56929332,'verified_public_score':entry['public_score'],
    'illustrated_notebook_gpu_rerun':False,'public_upload':False,'contains_competition_mri':True,
    'figures':{p.name:sha(p) for p in sorted(ASSETS.glob('*.png'))}}
(OUT/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
assert path.stat().st_size<900000,'Notebook exceeds conservative Kaggle source-size budget'
print(json.dumps(receipt,indent=2))
