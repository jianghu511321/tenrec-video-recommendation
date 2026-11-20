"""Evaluate the same full-catalog, seen-item-filtered ranking with popularity."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from prepare_sasrec_cohort import test_histories_for


def score(histories, targets, popularity, k=20):
    order = [i for i, _ in sorted(popularity.items(), key=lambda pair: (-pair[1], pair[0]))]
    ndcg, hits, hits100 = [], [], []
    for uid, target in targets.items():
        seen = set(histories[uid])
        candidate_pool = order[:max(k, 100) + len(seen)]
        filtered = [i for i in candidate_pool if i not in seen]
        top = filtered[:k]
        hits100.append(target in filtered[:100])
        rank = top.index(target) if target in top else None
        found = rank is not None
        hits.append(float(found))
        ndcg.append(1 / math.log2(rank + 2) if found else 0.0)
    return {f"recall{k}": float(np.mean(hits)), f"hit{k}": float(np.mean(hits)),
            f"ndcg{k}": float(np.mean(ndcg)), "recall100": float(np.mean(hits100)), "users": len(targets),
            "candidate_catalog": len(popularity), "model": "training item popularity; exact full-catalog ranking"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    payload = json.loads((args.data.resolve() / "cohort.json").read_text(encoding="utf-8"))
    train, valid, test = payload["train"], payload["valid"], payload["test"]
    counts = Counter(item for seq in train.values() for item in seq)
    report = {
        "validation": score(train, valid, counts),
        "test": score(test_histories_for(train, test, payload.get("test_context_items", valid)), test, counts),
        "protocol": "same per-user temporal targets and train-seen candidate catalog as SASRec; prior clicks filtered",
    }
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "popularity_baseline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
