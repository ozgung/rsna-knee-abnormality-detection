"""Verify four private local64 cache shards and gate promotion on bound K0 review."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from knee import local64_cache as cache, public_training_smoke as smoke


def _json(path):
    return json.loads(Path(path).read_bytes())


def verify_identities(rows, expected_sha256):
    """Bind metadata to exact ordered membership without reading label values."""
    ids = [row["StudyInstanceUID"] for row in rows]
    if len(set(ids)) != len(ids) or cache.digest(ids) != expected_sha256:
        raise ValueError("Cache study identities/order differ from the sealed population")
    return set(ids)


def verify_cache_files(root, rows, *, verify_pixels):
    """Optional remote/staging pixel check; default collector stays metadata-only."""
    for row in rows:
        cache.validate_private_entry(row)
        if row.get("file") != cache.filename(row["ordinal"]):
            raise ValueError("Cache filename/ordinal differs")
        for field in ("file_sha256", "volume_sha256", "mask_sha256", "joint_mask_sha256"):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Cache metadata content hash missing")
    if not verify_pixels:
        return
    if {p.name for p in root.glob("*.npz")} != {row["file"] for row in rows}:
        raise ValueError("Cache file inventory differs")
    for row in rows:
        path = cache.safe_file(root, row["file"])
        if smoke.file_sha(path) != row["file_sha256"]:
            raise ValueError("Cache NPZ file identity differs")
        arrays = cache.read_npz(path)
        hashes = cache.array_hashes(*arrays)
        if any(hashes[k] != row.get(k) for k in hashes):
            raise ValueError("Cache array/index hashes differ")
        cache.validate_slot_mask(arrays[1], row["slots"])


def qualify(build_root, result_root, *, review_path=None, verify_pixels=False):
    """Return a metadata-only receipt; no cache pixels are copied or promoted here."""
    build_root, result_root = Path(build_root), Path(result_root)
    contracts, all_entries, all_indices = [], [], []
    shards = []
    for shard in range(4):
        build = build_root / f"shard{shard}"
        result = result_root / f"shard{shard}" / "local64_cache"
        contract = _json(build / "build-contract.json")
        run_contract = _json(result / "run-contract.json")
        receipt = _json(result / "receipt.json")
        index = _json(result / "index-private.json")
        if contract != run_contract or contract.get("shard_index") != shard:
            raise ValueError(f"Shard {shard} immutable build/run contract differs")
        members = cache.make_shards([str(i) for i in range(cache.TRAIN_COUNT)], 4)[shard]
        expected_ids_sha = contract.get("partitions", [])[shard].get("ordered_ids_sha256")
        if (receipt.get("status") != "PASS" or receipt.get("version") != cache.VERSION or
                receipt.get("contract_sha256") != cache.digest(contract) or
                receipt.get("completed_studies") != len(members) or
                receipt.get("private_index_sha256") != smoke.file_sha(result / "index-private.json") or
                index.get("contract_sha256") != cache.digest(contract) or
                [row.get("ordinal") for row in index.get("entries", [])] != members):
            raise ValueError(f"Shard {shard} receipt/index coverage differs")
        rows = index["entries"]
        verify_identities(rows, expected_ids_sha)
        verify_cache_files(result, rows, verify_pixels=verify_pixels)
        common = {k: v for k, v in contract.items() if k not in {"shard_index", "shard_studies"}}
        contracts.append(common)
        all_entries.extend(rows)
        all_indices.append(index)
        shards.append({"shard_index": shard, "studies": len(rows), "contract_sha256": cache.digest(contract),
                       "index_sha256": smoke.file_sha(result / "index-private.json"),
                       "receipt_sha256": smoke.file_sha(result / "receipt.json"),
                       "cache_files_recorded": len(rows), "k0_montage_pages": sum(len(r.get("k0_montage", {}).get("pages", [])) for r in rows)})
    if any(doc != contracts[0] for doc in contracts[1:]):
        raise ValueError("Four shard scientific/source/input contracts differ")
    if len(all_entries) != cache.TRAIN_COUNT or sorted(e["ordinal"] for e in all_entries) != list(range(cache.TRAIN_COUNT)):
        raise ValueError("Training membership is incomplete or duplicated")
    ordered = sorted(all_entries, key=lambda row: row["ordinal"])
    training_ids = verify_identities(ordered, contracts[0]["training_membership_sha256"])
    verify_identities(ordered, contracts[0]["membership"]["ordered_eligible_sha256"])
    # Shard zero's scoring-only population has an independent contract, index,
    # and directory. It is validated for completeness and never enters K0 or
    # the training membership/stage candidate.
    gold_root = result_root / "shard0" / "local64_cache" / "gold58"
    gold_contract = _json(gold_root / "run-contract.json")
    gold_receipt = _json(gold_root / "receipt.json")
    gold_index = _json(gold_root / "index-private.json")
    if (gold_contract.get("population") != "scoring-only-gold58" or
            gold_contract.get("training_membership_sha256") is not None or
            gold_contract.get("gold_ids_sha256") != contracts[0].get("gold_membership_sha256") or
            gold_receipt.get("status") != "PASS" or
            gold_receipt.get("training_promotion_allowed") is not False or
            gold_receipt.get("completed_studies") != cache.GOLD_COUNT or
            gold_receipt.get("contract_sha256") != cache.digest(gold_contract) or
            gold_index.get("contract_sha256") != cache.digest(gold_contract) or
            gold_receipt.get("private_index_sha256") != smoke.file_sha(gold_root / "index-private.json")):
        raise ValueError("Separate scoring-only gold58 cache contract/receipt differs")
    gold_rows = gold_index.get("entries", [])
    if [row.get("ordinal") for row in gold_rows] != list(range(cache.GOLD_COUNT)):
        raise ValueError("Scoring-only gold58 index coverage differs")
    gold_ids = verify_identities(gold_rows, contracts[0]["gold_membership_sha256"])
    if training_ids & gold_ids:
        raise ValueError("Scoring-only gold overlaps training cache")
    verify_cache_files(gold_root, gold_rows, verify_pixels=verify_pixels)
    contract = contracts[0]
    if contract.get("membership", {}).get("eligible") != cache.TRAIN_COUNT:
        raise ValueError("Historical eligible population differs")
    # Bind rows to the hashed frozen32 manifest, keeping raw study IDs out of the aggregate receipt.
    expected_frozen = contract["frozen32"]["uid_sha256"]
    by_uid_hash = {__import__("hashlib").sha256(e["StudyInstanceUID"].encode()).hexdigest(): e for e in all_entries}
    visual_entries = []
    montage_pages = []
    for uid_hash in expected_frozen:
        row = by_uid_hash.get(uid_hash)
        if row is None or not row.get("k0_montage"):
            raise ValueError("Frozen32 K0 cache/montage membership is incomplete")
        visual_entries.append(row)
        montage = row["k0_montage"]
        source = result_root / f"shard{row['ordinal'] % 4}" / "local64_cache" / "k0-montage" / f"shard{row['ordinal'] % 4}" / montage["study_key"]
        if montage.get("study_key") != uid_hash[:24] or montage.get("frame_count") != sum(s["count"] for s in row["slots"] if s["selected"]):
            raise ValueError("Frozen32 K0 montage frame/source linkage differs")
        for page in montage.get("pages", []):
            path = source / page["file"]
            if not path.is_file() or smoke.file_sha(path) != page.get("sha256"):
                raise ValueError("Frozen32 K0 montage page hash differs")
            montage_pages.append({"study_key": montage["study_key"], **page})
    montage_manifest = {"version": cache.VERSION, "frozen32_order_sha256": contract["frozen32"]["ordered_membership_sha256"],
        "source_bundle_sha256": contract["source_bundle_sha256"],
        "asset_manifest_sha256": contract["asset_manifest_sha256"],
        "contract_set_sha256": smoke.digest(smoke.canonical([s["contract_sha256"] for s in shards])),
        "pages": montage_pages, "frame_count": sum(p["frame_count"] for p in montage_pages)}
    montage_sha = smoke.digest(smoke.canonical(montage_manifest))
    contract_set_sha = montage_manifest["contract_set_sha256"]
    review = _json(review_path) if review_path is not None else None
    review_binding = {"source_bundle_sha256": contract["source_bundle_sha256"],
        "asset_manifest_sha256": contract["asset_manifest_sha256"],
        "contract_set_sha256": contract_set_sha, "montage_sha256": montage_sha}
    if review is not None:
        annotations = review.get("studies")
        if not isinstance(annotations, dict) or set(annotations) != {e["k0_montage"]["study_key"] for e in visual_entries}:
            raise ValueError("K0 visual review must annotate every frozen32 study exactly once")
        for row in visual_entries:
            observation = annotations[row["k0_montage"]["study_key"]]
            for key in ("wrong_side", "axis_error", "sagittal_truncated", "axial_patella_retained"):
                if key not in observation or (key != "axial_patella_retained" and type(observation[key]) is not bool):
                    raise ValueError("K0 visual review annotations are incomplete")
                row["side"]["wrong_side"] = observation["wrong_side"]
                row["anchor"]["axis_error"] = observation["axis_error"]
                row["anchor"]["sagittal_truncated"] = observation["sagittal_truncated"]
                row["anchor"]["axial_patella_retained"] = observation["axial_patella_retained"]
    k0 = cache.k0_aggregate(all_entries, visual_entries=visual_entries,
        montage_review=review, expected_binding=review_binding)
    k0["montage_sha256"] = montage_sha
    k0["contract_set_sha256"] = contract_set_sha
    receipt = {"status": "PASS", "version": cache.VERSION, "shards": shards,
        "membership_counts": {"eligible": 4345, "training": 3475, "development": 870, "gold_scoring_only": 58},
        "gold_scoring_cache": {"status": "PASS", "studies": cache.GOLD_COUNT,
            "contract_sha256": cache.digest(gold_contract), "index_sha256": smoke.file_sha(gold_root / "index-private.json"),
            "separate_from_training": True},
        "grouping": "Historical fixed fold0 membership; patient identities unavailable; cache makes no new split",
        "aggregate_contains_ids": False, "images_copied": False,
        "k0": k0, "cache_qualification_pass": k0["status"] == "PASS",
        "pixels_verified_by_collector": verify_pixels,
        "pixel_evidence": "all file/raw hashes replayed" if verify_pixels else "bound worker receipts/index; staging verification still required",
        "training_promotion_allowed": False,
        "integration_status": "LOADER_STAGE_GATE_REQUIRED"}
    stage_config = None
    if k0["status"] == "PASS":
        stage_config = {"version": cache.VERSION, "membership": contract["membership"],
            "shards": [{"root": str(result_root / f"shard{i}" / "local64_cache"),
                        "contract_sha256": shards[i]["contract_sha256"],
                        "index_sha256": shards[i]["index_sha256"]} for i in range(4)],
            "k0_review_binding": review_binding, "training_promotion_allowed": False,
            "integration_status": "CACHE_QUALIFIED_LOADER_STAGE_GATE_REQUIRED"}
    return receipt, stage_config, montage_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--montage-manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--verify-pixels", action="store_true",
                        help="Verify full cache files on the remote staging host; default collects metadata only")
    parser.add_argument("--stage-config", type=Path)
    args = parser.parse_args(argv)
    if args.receipt.exists() or args.montage_manifest.exists() or (args.stage_config and args.stage_config.exists()):
        raise SystemExit("Fresh output paths required")
    receipt, stage_config, montage = qualify(args.build_root, args.result_root, review_path=args.review,
                                           verify_pixels=args.verify_pixels)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.montage_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    args.montage_manifest.write_text(json.dumps(montage, indent=2, allow_nan=False) + "\n")
    args.receipt.chmod(0o600); args.montage_manifest.chmod(0o600)
    if stage_config is not None:
        if args.stage_config is None:
            raise SystemExit("K0 passed but --stage-config path was not supplied")
        args.stage_config.parent.mkdir(parents=True, exist_ok=True)
        args.stage_config.write_text(json.dumps(stage_config, indent=2, allow_nan=False) + "\n")
        args.stage_config.chmod(0o600)
    print(json.dumps({"status": receipt["status"], "k0": receipt["k0"]["status"],
                      "training_promotion_allowed": receipt["training_promotion_allowed"],
                      "shards": len(receipt["shards"]), "montage_sha256": receipt["k0"]["montage_sha256"]}))


if __name__ == "__main__":
    main()
