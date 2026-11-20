"""Prepare a user-prefix SASRec cohort from Tenrec's official SBR data."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

SOURCE = Path(r"E:\Tenrec\Tenrec\sbr_data_1M.csv")


def groups(reader, max_users):
    current, rows = None, []
    closed, count = set(), 0
    for row in reader:
        uid = row["user_id"]
        if current is None:
            current = uid
        if uid != current:
            closed.add(current)
            yield current, rows
            count += 1
            if count >= max_users:
                return
            if uid in closed:
                raise ValueError("SBR source has non-contiguous user histories; cannot make sequence splits safely.")
            current, rows = uid, []
        rows.append(row["item_id"])
    if current is not None and count < max_users:
        yield current, rows


def eligible_targets(targets, train, item_map, previous=None):
    eligible = {}
    excluded = Counter({"repeat": 0, "outside_catalog": 0})
    for uid, target in targets.items():
        if target in train[uid] or (previous is not None and target == previous[uid]):
            excluded["repeat"] += 1
        elif target not in item_map:
            excluded["outside_catalog"] += 1
        else:
            eligible[uid] = item_map[target]
    return eligible, dict(excluded)


def test_histories_for(train, test, context_items):
    return {
        uid: train[uid] + ([context_items[uid]] if uid in context_items else [])
        for uid in test
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--max-users", type=int, default=10000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    src, out = args.source.resolve(), args.out.resolve()
    if args.max_users < 1:
        raise ValueError("--max-users must be positive")
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"Output directory is not empty: {out}; use a fresh run directory.")
    train, valid, test = {}, {}, {}
    counts = Counter()
    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not {"user_id", "item_id", "click"}.issubset(reader.fieldnames or []):
            raise ValueError("SBR source must contain user_id, item_id, and click columns")
        for uid, rows in groups(reader, args.max_users):
            counts["users_read"] += 1
            clicks = [item for item in rows]
            counts["raw_click_interactions"] += len(clicks)
            if len(clicks) < 5:
                counts["users_too_short"] += 1
                continue
            train[uid] = clicks[:-2]
            valid[uid] = clicks[-2]
            test[uid] = clicks[-1]
    if not train:
        raise ValueError("No eligible user sequences were found")

    item_map = {item: idx + 1 for idx, item in enumerate(sorted({i for seq in train.values() for i in seq}))}
    val_out, valid_excluded = eligible_targets(valid, train, item_map)
    test_out, test_excluded = eligible_targets(test, train, item_map, previous=valid)
    if not val_out or not test_out:
        raise ValueError("Validation and test each need at least one new target in the training catalog")
    train_out = {u: [item_map[i] for i in seq] for u, seq in train.items()}
    test_context_items = {u: item_map[valid[u]] for u in test_out if valid[u] in item_map}
    payload = {"train": train_out, "valid": val_out, "test": test_out,
               "test_context_items": test_context_items, "item_map": item_map}
    (out / "cohort.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    (out / "item_map.json").write_text(json.dumps(item_map), encoding="utf-8")
    source_stat = src.stat()
    manifest = {
        "source": str(src),
        "source_size_bytes": source_stat.st_size,
        "source_mtime": datetime.fromtimestamp(source_stat.st_mtime).isoformat(),
        "source_description": "Tenrec official SBR derived dataset: click-positive histories, items with corpus frequency below 2 or above 200000 removed, first 1M grouped user histories",
        "selection": f"first {args.max_users} user groups from the official SBR file",
        "user_group_contiguity_checked": True,
        "split": "per-user next-item: all but last two events train; penultimate validation target; final event test target",
        "candidate_catalog": "items present in training prefixes of all users with at least five clicks",
        "evaluation": "split-specific, first-time targets in the training catalog. At test time use the observed penultimate click when it is in the catalog, regardless of its validation eligibility.",
        "counts": {**dict(counts), "train_users": len(train_out), "train_items": len(item_map),
                   "train_clicks": sum(map(len, train_out.values())),
                   "train_next_item_pairs": sum(len(s) - 1 for s in train_out.values()),
                   "validation_targets": len(val_out), "validation_excluded": valid_excluded,
                   "test_targets": len(test_out), "test_excluded": test_excluded,
                   "test_histories_skipping_unknown_penultimate": len(test_out) - len(test_context_items)},
        "cohort_sha256": hashlib.sha256((out / "cohort.json").read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
