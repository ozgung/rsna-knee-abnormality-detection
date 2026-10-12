# Private96-wide cache preparation — October12

Purpose: prepare the accepted combined96-source/2–98% representation for the next own224 training candidate. This is preprocessing, not a trained model or accuracy result. Existing CPU32pilot qualified the reader and exact44control; see [pilot](source-span-pilot-20261012.md).

Source revision `ca3cc5b81437e548132fd5181f14a0e2052c56db`. Fixed4345 eligible studies in historical order, excluding58gold and4quarantined; original3475train/870development membership retained. Four disjoint shards1087/1086/1086/1086.336px/140mm, slots26/22/18/12/18,2–98% span. Pinned helper/source/CPU environment from real pilot. Only small corpus identity files are read; no repeated22GB historical volume hashing.

One-study streaming compressedNPZ, private aligned identity index, raw volume/mask and file hashes, atomic writes, bounded shape/header verification. Resume requires exact contract and valid completed files; prior receipts preserved. Failure preserves completed outputs and emits sanitized aggregate receipt. Actual disk preflight includes worst-case remaining output and temporary files; actual12h platform timeout, no assistant runtime cap. Source image arrays/index remain private and outsideGit.

Parent review:26focused testsPASS/0skip across source96cache, source-span pilot and corpus44 parity, using existing cv2 overlay and explicit pinned evidence; synthetic CPU smokePASS. Tests cover real4345 membership/build, corruption, failure/resume/import, disk refusal, path safety and existing syntheticDICOM/control behavior. These are not full-corpus execution results.

## Active shard0

`ozgung/knee-source96-cache-shard0-20261012` version1/kernel138342688 RUNNING14:29UTC;17accepted-source/private/offline/CPUsettingschecksPASS. First1087studies; at least12,410,943,584freebytes required before decoding. Runtime unmeasured at this scale. Shards1–3 not launched yet; verify first shard startup before expansion.

Build `outputs/kaggle/source96-cache-shard0-20261012-reviewed-v1/`. NotebookSHA `ec15c9ee9af9d43884b86483ed872931656aba488dfc87baac69a72519638639`; contractSHA `96e10c1ced34db0d365fe5ed8544afb4c2d3b051087f8f4ffab12284c9aea5ee`; bundleSHA `53787be974e0750296d6f9347a34c2e0ef60680b6cbd6961a22a224235678ed8`.

Launch executed once: `/Users/ozgun/miniconda3/bin/kaggle kernels push -p outputs/kaggle/source96-cache-shard0-20261012-reviewed-v1`. Accepted evidence and small-output collector: `outputs/analysis/source96-cache-shard0-20261012/`. Run `collect_terminal_v1.py` with miniconda Python when terminal; collector downloads aggregate receipt/log only, not private image corpus. Remote output `/kaggle/working/source96_cache/`; retain `run-contract.json`, `index-private.json`, receipt/history andNPZs for exact-contract recovery. Stop via Kaggle session control if required; no automatic restart. No local controller.

Next: inspect terminal counts/source/environment/index hash and storage/runtime; prepare remaining shards and verified direct-to-Colab staging. Training integration still pending; no96training launched.

## Parallel shards and loader —14:53UTC

Loader commit `0f0507c`: parent33focusedtestsPASS/0skip (including exact RNG-restored dense/proxy K8train/K16eval tensors, fork/spawn, corruption and membership); syntheticCPU smokePASS. Full-corpus staging/training still unverified. All file/raw hashes checked at staging; subsequent access checks file signatures and bounded schema/CRC, one pair per worker. Read-only trusted staging is required.

Shard0 remainsRUNNING14:49. To reduce wall time, launched remaining disjoint shards after synthetic/real32 qualification, without claiming fullshard startup/decoding verified: live output endpoint returned no files/log; explicit version_label=1 returned404. Each scientific contract matches shard0. Each new shard accepted17checksPASS/privateCPU/offline; no GPUquota used for cache.

- Shard1: `ozgung/knee-source96-cache-shard1-20261012` v1/kernel138345304, RUNNING14:52UTC,1086studies. NotebookSHA `c19d7c68bd2a7b63e1a5ddce4e6ef08344bbf0b68fab686be71e7f18823a04b4`. Build `outputs/kaggle/source96-cache-shard1-20261012-reviewed-v1/`; source0f0507c.
- Shard2: `ozgung/knee-source96-cache-shard2-20261012` v1/kernel138345324, RUNNING14:52UTC,1086studies. NotebookSHA `342238bb2c6f644d45b0887d42a2cfba462773098c06dd78a678a1c9a8ad14c3`. Build `outputs/kaggle/source96-cache-shard2-20261012-reviewed-v1/`; source0f0507c.
- Shard3: `ozgung/knee-source96-cache-shard3-20261012` v1/kernel138345351, RUNNING14:52UTC,1086studies. NotebookSHA `67a9c3ebe0e9e83e102d2b4d408757fa99b94bd00f19bde98e2802d528bf8462`. Build `outputs/kaggle/source96-cache-shard3-20261012-reviewed-v1/`; source0f0507c.

For each shardN, launch executed once as `kaggle kernels push -p outputs/kaggle/source96-cache-shardN-20261012-reviewed-v1`; collector `outputs/analysis/source96-cache-shardN-20261012/collect_terminal_v1.py` with minicondaPython fetches aggregate receipt/log only. Exact contracts/build hashes/accepted sources stored adjacent. Same remote output/recovery paths as shard0; no localcontrollers. Do not rerun. Collect terminal receipts and qualify all4before training.

## Colab staging implemented —15:02UTC

Commit `6fe954b`: VM-side `scripts/colab_source96_prepare.py`, parent15focused staging/loader testsPASS/0skip. Injected transport covers metadata-first exact4shard coverage, source/version races, path/URL bounds, corrupted bytes, interruptions and disk refusal. Live VM authentication/URLs/transfer remain unverified. Stage requires all4terminalPASS receipts and reviewed accepted notebook/contract/privateindex/receipt hashes. Streams1MiB chunks directly toVM; no laptop image relay, no logged signedURLs. Actualdisk preflight reserves full remaining compressed bytes, two largest temporary buffers and5GiB operational space. Completed files and failed receipts preserved; readiness published only after full loader assembly and finalsource check.

One helper now implements immutable96training package adaptation; no training launched. Historical generic initializer hash is not yet reconciled; use recipe-level comparison language. Exact sampler/model/scientific recipe preservation, everyepoch checkpoints and explicit persistent-worker RNG resume limitation remain required.

## Training package implemented —15:23UTC

Commit `cdad021`: runtime safeguards, immutable builder and CPUtests; parent14focusedtestsPASS/0skip including loader, syntheticCPU smokePASS. Historical StudyWindows/RaptorClassifier, optimizer/loss/scheduler, update loop, top3 selection and SWA AST unchanged. Entry uses verified96 cache, fixed10epochs/K8train/K16selection, workers2 default, no assistant timecap, strictCUDA. Per-epoch model/optimizer/scaler/scheduler/mainRNG recovery binds exactcache/supervision/initializer/config. Persistent-worker RNG/prefetch remains unrestored; no bit-identical continuation claim. K94 only exports same completed primaryK16 checkpoint. No GPU execution/training yet.

Reviewed build `outputs/kaggle/source96-training-20261012-reviewed-v1/`: manifestSHAc7dffd7365e555387bc1936ef2f0ee9156d795eba2f4fd490647739b707ab2ff, trainerSHA4a6381363db8c24013692ff386522aabe74658c3544fb08e20552ee2e834d4f0, entrySHA47d28fe77c639a395f695969aa691dce0d1b5563a41209b5f138001cd36a3f1e. Build command `.venv/bin/python scripts/build_source96_training.py outputs/kaggle/source96-training-20261012-reviewed-v1 --evidence-root .`.

Initializer primary source: https://huggingface.co/timm/coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k identifies generic ImageNet12k pretraining/ImageNet1k fine-tuning and Apache2 license. Official API metadata saved `outputs/analysis/source96-initializer-20261012/hf-model-metadata.json`: public/ungated revision50b0fc810164980f0e494198ed3da32f5abf6dbe; model.safetensors295695280bytes, SHA8558fd38c0c67be0425f3ca19752a7baa4356314940fa4954633a6395a97d973. Metadata only fetched, no weights. Old initialization exactbytes still unresolved. Next helper implements fixedrevision VM normalization/descriptor with strictkeycoverage; parent owns realVMvalidation, staging and orchestration/persistence.
