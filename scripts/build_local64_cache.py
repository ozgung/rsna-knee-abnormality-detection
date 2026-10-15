"""Build-only private CPU local64 shard notebooks; never push or launch them."""
import argparse
import ast
import base64
import json
from pathlib import Path
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee import local64_cache as cache, public_training_smoke as smoke, reference, raptor64_cache as raptor
from knee.coverage_inference import CODECS
from knee.reference_blend import DOCKER_IMAGE

S6_BUILD = "outputs/kaggle/public-s6-qualification-20261011-implementation-draft"
S6_BUNDLE_SHA = "eb63239f3b9ec2615f2cf7ea5e7371a255782ff93cdd645973bca39eeba4dcad"
S6_NOTEBOOK_SHA = "e91ff45bb1ba76c4308940cb78fa0b488f2ad80b5697edd07f857ed05a57b0d7"
ENVIRONMENT_RECEIPT = "outputs/kaggle/source96-cache-shard0-20261012-result-v1/source96_cache/receipt.json"
ENVIRONMENT_RECEIPT_SHA = "898c6a24271a48490ecfcef0adc5aa43a9a11343d9dde0bc8a116d2ad5a1cc45"
TRAIN_CSV_SHA = "8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33"
S6_MANIFEST_SHA = "1c1211e64b73f1f6433ade804816572db1a2aa5422ed9c57365184bc6a9517e6"
OWNER = "ozgung"
SOURCES = (
    "knee/__init__.py", "knee/public_training_smoke.py", "knee/source96_cache.py",
    "knee/source_span_pilot.py", "knee/corpus44_parity.py", "knee/public_rich_cache.py",
    "knee/reference.py", "knee/fresh_reader_setup.py", "knee/raptor_input.py",
    "knee/local64_cache.py", "knee/local64_runtime.py",
    "knee/local64_loader.py",
)


def _bundle(notebook_path):
    if smoke.file_sha(notebook_path) != S6_NOTEBOOK_SHA:
        raise ValueError("Immutable S6 qualification notebook bytes differ")
    notebook = json.loads(Path(notebook_path).read_bytes())
    encoded = None
    manifest = None
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "encoded" for t in node.targets):
                value = ast.literal_eval(node.value)
                if encoded is not None:
                    raise ValueError("Ambiguous S6 embedded source bundle")
                encoded = value
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "manifest" for t in node.targets):
                try:
                    manifest = ast.literal_eval(node.value)
                except Exception:
                    continue
    if not encoded:
        raise ValueError("Pinned S6 notebook source bundle missing")
    sources = json.loads(zlib.decompress(base64.b64decode(encoded, validate=True)))
    if smoke.digest(smoke.canonical(sources)) != S6_BUNDLE_SHA:
        raise ValueError("Pinned S6 source closure differs")
    if not isinstance(manifest, dict) or not manifest.get("anchor_weights_sha256") or not manifest.get("sha256", {}).get("side_cls.pt"):
        raise ValueError("Pinned S6 runtime manifest/checkpoint pins missing")
    return sources, manifest


def _eligible(evidence, raw, np):
    evidence = Path(evidence)
    identity_path = evidence / "outputs/kaggle/sealed-identity-full-result/sealed_identity/sealed-manifest-private.json"
    smoke.require_file(identity_path, smoke.MANIFEST_SHA)
    identity = json.loads(identity_path.read_bytes())
    historical = []
    for sub, prefix in (("corpus", "all"), ("corpus-ext", "extra")):
        path = evidence / f"outputs/analysis/public-train-recipe-20261009/{sub}/{prefix}_ids.npy"
        smoke.require_file(path, smoke.PINS[prefix + "_ids.npy"])
        historical.extend(np.load(path, allow_pickle=False).astype(str).tolist())
    folds_path = evidence / "outputs/kaggle/trackA-coatnet224-fold01-20261009/folds_v0.json"
    folds_sha = smoke.file_sha(folds_path)
    folds = json.loads(folds_path.read_bytes())["folds"]
    ids, summary = cache.membership(historical, Path(raw) / "train.csv", folds, identity)
    gold = [uid for uid in historical if uid in {r["uid"] for r in identity["records"] if r["future_role"] == "gold"}]
    shards, gold = cache.partition_population(ids, gold)
    ordinal_parts = cache.make_shards(ids, 4)
    parts = [{"studies": len(rows), "ordinals_sha256": smoke.digest(smoke.canonical(ordinals)),
              "ordered_ids_sha256": smoke.digest(smoke.canonical(rows))}
             for rows, ordinals in zip(shards, ordinal_parts)]
    return ids, gold, summary, parts, folds_sha


def build(destination, slug, *, evidence_root, public_source, shard_index, shard_count=4):
    import numpy as np
    out = Path(destination)
    evidence = Path(evidence_root)
    if out.exists():
        raise ValueError("Fresh immutable build directory required")
    cache.slug(slug)
    if len(OWNER + "/" + slug) > 64:
        raise ValueError("Kernel reference too long")
    if type(shard_index) is not int or type(shard_count) is not int or (shard_index, shard_count) not in ((0, 4), (1, 4), (2, 4), (3, 4)):
        raise ValueError("local64 requires one of four fixed CPU shards")
    raw = evidence / "data/raw"
    smoke.require_file(raw / "train.csv", TRAIN_CSV_SHA)
    s6root = evidence / S6_BUILD
    s6receipt = json.loads((s6root / "build-receipt.json").read_bytes())
    if s6receipt.get("binding", {}).get("source_bundle_sha256") != S6_BUNDLE_SHA:
        raise ValueError("Accepted S6 source closure receipt differs")
    s6sources, s6manifest = _bundle(s6root / "notebook.ipynb")
    public_path = Path(public_source) if Path(public_source).is_absolute() else evidence / public_source
    module, decoder, reader_manifest = reference.audit_source(public_path.read_text(), profile="raptor",
                                               checkpoint_sha256=raptor.V4_CHECKPOINT_SHA256, raptor_windows=42)
    if decoder or raptor.load_public(module, reader_manifest).IMG != 336:
        raise ValueError("Pinned Dread v4 source reader contract differs")
    ids, gold_ids, membership, partitions, folds_sha = _eligible(evidence, raw, np)
    shards, _ = cache.partition_population(ids, gold_ids)
    frozen_path = evidence / "outputs/analysis/native-crop-qualification-20261011/sample-private.json"
    frozen_ids = json.loads(frozen_path.read_bytes())["StudyInstanceUID"]
    if len(frozen_ids) != 32 or smoke.digest(smoke.canonical(frozen_ids)) != "1e4cf23b048d1ced8c960ee8035abbfe1f43aba4992feb6c4d42d4949748c414" or not set(frozen_ids) <= set(ids):
        raise ValueError("Frozen32 K0 subset/order differs from the accepted training membership")
    proof_path = evidence / "outputs/analysis/local64-20261015/frozen32-membership-proof.json"
    proof = json.loads(proof_path.read_bytes())
    if proof.get("all_in_eligible_training") is not True or proof.get("frozen32_order_sha256") != smoke.digest(smoke.canonical(frozen_ids)):
        raise ValueError("Frozen32 membership proof is missing or differs")
    frozen_hashes = [__import__("hashlib").sha256(uid.encode()).hexdigest() for uid in frozen_ids]
    frozen_counts = [sum(__import__("hashlib").sha256(uid.encode()).hexdigest() in set(frozen_hashes) for uid in [ids[n] for n in part]) for part in cache.make_shards(ids, 4)]
    env_path = evidence / ENVIRONMENT_RECEIPT
    smoke.require_file(env_path, ENVIRONMENT_RECEIPT_SHA)
    env_receipt = json.loads(env_path.read_bytes())
    if env_receipt.get("status") != "PASS":
        raise ValueError("Qualified CPU environment receipt is not PASS")
    sources = dict(s6sources)
    for name in SOURCES:
        text = (ROOT / name).read_text()
        compile(text, name, "exec")
        sources[name] = text
    for name, text in sources.items():
        compile(text, name, "exec")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    input_pins = {
        "competition": {name: smoke.file_sha(raw / name) for name in ("train.csv", "train_series.csv")},
        "historical_ids": {k: v for k, v in smoke.PINS.items() if k.endswith("_ids.npy")},
        "folds_sha256": folds_sha, "identity_sha256": smoke.MANIFEST_SHA,
        "codecs": CODECS, "environment_receipt_sha256": ENVIRONMENT_RECEIPT_SHA,
        "anatomy_sha256": s6manifest["anchor_weights_sha256"],
        "side_cls_sha256": s6manifest["sha256"]["side_cls.pt"],
        "s6_manifest_sha256": S6_MANIFEST_SHA,
    }
    asset_manifest_sha256 = smoke.digest(smoke.canonical({
        "s6_manifest_sha256": S6_MANIFEST_SHA,
        "anatomy_sha256": input_pins["anatomy_sha256"],
        "side_cls_sha256": input_pins["side_cls_sha256"], "codecs": CODECS,
    }))
    source_bundle_sha256 = smoke.digest(smoke.canonical(sources))
    contract = {
        "version": cache.VERSION, "config": {"depth": cache.DEPTH, "storage": cache.STORAGE,
            "render": cache.RENDER_SIZE, "span": list(cache.SPAN), "slots": [list(x) for x in cache.SLOTS],
            "views": cache.VIEWS, "mask_schema": cache.MASK_SCHEMA, "cpu_threads": 1},
        "membership": membership, "shard_index": shard_index, "shard_count": 4,
        "partitions": partitions, "reader": {"module_sha256": smoke.digest(module.encode()),
            "manifest_sha256": smoke.digest(smoke.canonical(reader_manifest)), "source_sha256": reference.RAPTOR_SOURCE_SHA256,
            "input_policy": __import__("knee.raptor_input", fromlist=["policy_contract"]).policy_contract()},
        "s6_manifest": s6manifest,
        "s6_manifest_canonical_sha256": smoke.digest(smoke.canonical(s6manifest)),
        "code_revision": revision, "sources": {n: smoke.digest(t.encode()) for n, t in sources.items()},
        "source_bundle_sha256": source_bundle_sha256, "asset_manifest_sha256": asset_manifest_sha256,
        "input_pins": input_pins, "qualified_environment": env_receipt["environment"],
        "training_membership_sha256": smoke.digest(smoke.canonical(ids)),
        "gold_membership_sha256": smoke.digest(smoke.canonical(gold_ids)),
        "frozen32": {"ordered_membership_sha256": smoke.digest(smoke.canonical(frozen_ids)),
                     "uid_sha256": frozen_hashes, "counts_by_shard": frozen_counts,
                     "proof_sha256": smoke.file_sha(proof_path)},
        "population": {"eligible": 4345, "train": 3475, "development": 870, "gold_scoring_only": 58,
                       "patient_identity_available": False},
        "shard_studies": len(shards[shard_index]),
        "runtime_limit": "Actual CPU platform limit 12h; source preparation plus native local-render pass",
        "policy": "Image-only private cache; no report text, labels, training, submission or gold mixing",
    }
    encoded_sources = base64.b64encode(zlib.compress(smoke.canonical(sources), 9)).decode()
    encoded_module = base64.b64encode(module.encode()).decode()
    encoded_manifest = base64.b64encode(smoke.canonical(reader_manifest)).decode()
    bootstrap = f'''import base64,json,hashlib,os,sys,zlib,subprocess,time
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUDA_VISIBLE_DEVICES='',BUILD_MONAI='0')
SOURCES=json.loads(zlib.decompress(base64.b64decode({encoded_sources!r},validate=True)))
CONTRACT={contract!r}
module=base64.b64decode({encoded_module!r},validate=True).decode()
manifest=json.loads(base64.b64decode({encoded_manifest!r},validate=True))
from hashlib import sha256
canon=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
project=Path('/kaggle/working/local64-project');code=project/'source'
for name,text in SOURCES.items():
    if sha256(text.encode()).hexdigest()!=CONTRACT['sources'].get(name):raise ValueError('Packaged source differs: '+name)
    p=(project/name) if name.startswith('knee/') else (code/name)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
sys.path[:0]=[str(project),str(code)]
from knee.fresh_reader_setup import dataset_root,notebook_root
from knee.public_training_smoke import require_file,file_sha
from knee import local64_cache as cache
from knee.local64_runtime import run
from knee import public_training_smoke as smoke
from knee.public_s6_qualification import verify_assets
from knee import source96_cache as membership_cache
from pathlib import Path
if sys.version_info[:2]!=(3,12):raise ValueError('Pinned Python 3.12 environment differs')
inputs=Path('/kaggle/input')
competition=inputs/'competitions/rsna-knee-abnormality-detection'
if not competition.is_dir():competition=inputs/'rsna-knee-abnormality-detection'
for name,pin in CONTRACT['input_pins']['competition'].items():require_file(competition/name,pin)
identity=notebook_root(inputs,smoke.IDENTITY_KERNEL,'sealed_identity',['sealed-manifest-private.json'])
require_file(identity/'sealed-manifest-private.json',smoke.MANIFEST_SHA)
identity_doc=json.loads((identity/'sealed-manifest-private.json').read_bytes())
historical=[]
for ref,prefix in zip(smoke.CORPORA,('all','extra')):
    root=dataset_root(inputs,ref,[prefix+'_ids.npy'])
    require_file(root/(prefix+'_ids.npy'),smoke.PINS[prefix+'_ids.npy'])
    import numpy as np
    historical.extend(np.load(root/(prefix+'_ids.npy'),allow_pickle=False).astype(str).tolist())
hpo=dataset_root(inputs,smoke.ASSETS,['folds_v0.json'])
require_file(hpo/'folds_v0.json',CONTRACT['input_pins']['folds_sha256'])
folds=json.loads((hpo/'folds_v0.json').read_bytes())['folds']
ids,membership=membership_cache.membership(historical,competition/'train.csv',folds,identity_doc)
gold_set={{r['uid'] for r in identity_doc['records'] if r['future_role']=='gold'}}
gold=[uid for uid in historical if uid in gold_set]
if membership!=CONTRACT['membership'] or cache.digest(ids)!=CONTRACT['training_membership_sha256'] or cache.digest(gold)!=CONTRACT['gold_membership_sha256']:raise ValueError('Historical membership differs')
frozen_hashes=CONTRACT['frozen32']['uid_sha256']
if sum(hashlib.sha256(uid.encode()).hexdigest() in set(frozen_hashes) for uid in ids)!=32:raise ValueError('Frozen32 hashed membership differs')
anatomy_root=dataset_root(inputs,'heliosli/rsna-orthofoundation-fold0-epoch7-20261004',['anatomy.pt'])
compact=dataset_root(inputs,'heliosli/rsna-knee-compact-v1',['manifest.json','side_cls.pt'])
require_file(anatomy_root/'anatomy.pt',CONTRACT['input_pins']['anatomy_sha256'])
require_file(compact/'side_cls.pt',CONTRACT['input_pins']['side_cls_sha256'])
verify_assets(compact,anatomy_root/'anatomy.pt',CONTRACT['s6_manifest'],{{'manifest_sha256':CONTRACT['input_pins']['s6_manifest_sha256']}})
from knee.fresh_reader_setup import dataset_root as find_dataset
codec_root=find_dataset(inputs,'goodpjw2008/rsna-knee-2-5d-convnext-reader',list(CONTRACT['input_pins']['codecs']))
wheels=[]
for name,pin in CONTRACT['input_pins']['codecs'].items():
    p=codec_root/name;require_file(p,pin);wheels.append(str(p))
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*wheels],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
from knee.reference import load_definitions
public,decoder=load_definitions(module,'',manifest)
if decoder is not None:raise ValueError('Unexpected embedded decoder')
shard=CONTRACT['shard_index'];members=cache.make_shards(ids,4)[shard]
cfg={{'data_root':str(competition),'code':str(code),'out':'/kaggle/working/local64_cache',
 'shard_index':shard,'shard_count':4,'training_ids':ids,'gold_ids':gold if shard==0 else [],
 'contract':CONTRACT,'dread_module':module,'dread_manifest':manifest,'frozen32_uid_sha256':frozen_hashes,
 'anatomy_path':str(anatomy_root/'anatomy.pt'),'side_model_path':str(compact/'side_cls.pt')}}
receipt=run(cfg)
print(json.dumps({{'event':'local64_done','status':receipt['status'],'studies':receipt['completed_studies'],'k0':'UNKNOWN'}}))
'''
    compile(bootstrap, "local64-cache-notebook", "exec")
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python3", "language": "python"}},
          "cells": [{"cell_type": "markdown", "id": "protocol", "metadata": {}, "source": [
              "# Private local64 CPU cache shard\n", f"Shard {shard_index}/4 of 4,345 fixed historical-eligible studies.\n",
              "Exact Dread source mask/slot order plus pinned S6-native local crops; 384 render then center 352 crop.\n",
              "Gold58 is isolated under a separate scoring-only output on shard zero. No labels/reports/training.\n",
              "K0 remains UNKNOWN until bound montage and signed visual review.\n"]},
              {"cell_type": "code", "id": "run", "metadata": {}, "execution_count": None, "outputs": [], "source": bootstrap.splitlines(True)}]}
    blob = (json.dumps(nb, indent=2) + "\n").encode()
    metadata = {"id": OWNER + "/" + slug, "title": slug, "code_file": "notebook.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [*smoke.CORPORA, smoke.ASSETS, "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
                            "goodpjw2008/rsna-knee-2-5d-convnext-reader"],
        "kernel_sources": [smoke.IDENTITY_KERNEL], "model_sources": [], "docker_image": DOCKER_IMAGE}
    receipt = {"status": "BUILD_ONLY", "notebook_sha256": smoke.digest(blob), "notebook_bytes": len(blob),
        "contract_sha256": smoke.digest(smoke.canonical(contract)), "source_bundle_sha256": source_bundle_sha256,
        "code_revision": revision, "kernel": metadata["id"], "shard_index": shard_index,
        "shard_count": 4, "studies": len(shards[shard_index]), "K0": "UNKNOWN"}
    out.mkdir(parents=True)
    (out / "notebook.ipynb").write_bytes(blob)
    for name, value in (("kernel-metadata.json", metadata), ("build-contract.json", contract), ("build-receipt.json", receipt)):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path); parser.add_argument("slug")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--public-source", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
