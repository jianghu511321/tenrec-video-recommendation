"""Fit collaborative retrieval on click prefixes and select blending on validation."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from baseline_popularity import score as popularity_metrics
from prepare_sasrec_cohort import SOURCE, groups, test_histories_for


DEPTH = 100
DISCOUNT = 1 / np.log2(np.arange(2, 22))
NEIGHBORS = (40, 160)
IDF_POWERS = (0.0, 0.5)
HISTORY_DECAYS = (False, True)
FUSION_WEIGHTS = (0.25, 0.5, 0.75, 1.0)


def sparse_histories(histories, size, recent=False):
    items, values, offsets = [], [], [0]
    for history in histories:
        weights = {}
        for distance, item in enumerate(reversed(history)):
            weights.setdefault(item, 1 / (1 + distance / 8) if recent else 1.0)
        items.extend(weights)
        values.extend(weights.values())
        offsets.append(len(items))
    return csr_matrix((np.asarray(values, dtype=np.float32), items, offsets),
                      shape=(len(offsets) - 1, size))


def normalize(matrix):
    lengths = np.sqrt(np.asarray(matrix.power(2).sum(axis=1)).ravel())
    return matrix.multiply((1 / np.maximum(lengths, 1e-8))[:, None]).tocsr()


class NeighborRetriever:
    def __init__(self, train, num_items, idf_power):
        self.size = num_items + 1
        self.row_for_user = {uid: row for row, uid in enumerate(train)}
        self.pop = Counter(item for seq in train.values() for item in seq)
        self.popular = np.array(sorted(self.pop, key=lambda item: (-self.pop[item], item)))
        self.click_counts = np.array([self.pop[item] for item in range(self.size)])
        self.clicks = sparse_histories(train.values(), self.size)
        users_per_item = np.asarray(self.clicks.sum(axis=0)).ravel()
        self.idf = np.log1p(len(train) / np.maximum(1, users_per_item)) ** idf_power
        self.user_vectors_t = normalize(self.clicks.multiply(self.idf).tocsr()).T.tocsc()

    def candidate_lists(self, histories, recent, neighbors):
        users = list(histories)
        for start in range(0, len(users), 96):
            batch = users[start:start + 96]
            query = sparse_histories((histories[u] for u in batch), self.size, recent)
            query = normalize(query.multiply(self.idf).tocsr())
            similarities = (query @ self.user_vectors_t).tocsr()
            for row, uid in enumerate(batch):
                left, right = similarities.indptr[row:row + 2]
                ids, sims = similarities.indices[left:right], similarities.data[left:right]
                keep = ids != self.row_for_user.get(uid, -1)
                ids, sims = ids[keep], sims[keep]
                order = np.lexsort((ids, -sims))[:max(neighbors)]
                ids, sims = ids[order], sims[order]
                seen = set(histories[uid])
                pop = [int(item) for item in self.popular[:DEPTH + len(seen)] if item not in seen][:DEPTH]
                ranks = {}
                for n in neighbors:
                    votes = np.asarray(self.clicks[ids[:n]].T @ (sims[:n] ** 2)).ravel()
                    if seen:
                        votes[list(seen)] = 0
                    available = np.flatnonzero(votes > 0)
                    # Compare the vote total with a version that discounts very popular videos.
                    for penalty in (0.0, 0.25):
                        scores = votes[available] / self.click_counts[available] ** penalty
                        ranked = np.lexsort((available, -scores))[:DEPTH]
                        ranks[n, penalty] = available[ranked].tolist()
                yield uid, pop, ranks


def blend(pop, similar, weight):
    if weight == 1:
        covered = set(similar)
        return (similar + [item for item in pop if item not in covered])[:DEPTH]
    scores = {item: (1 - weight) / (20 + rank) for rank, item in enumerate(pop, 1)}
    for rank, item in enumerate(similar, 1):
        scores[item] = scores.get(item, 0.0) + weight / (20 + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))[:DEPTH]


def target_rank(items, target):
    try:
        return items.index(target) + 1
    except ValueError:
        return 0


def summarize(ranks):
    ranks = np.asarray(ranks)
    top20 = (ranks > 0) & (ranks <= 20)
    return {
        "users": len(ranks), "hits20": int(top20.sum()), "hits100": int((ranks > 0).sum()),
        "recall20": float(top20.mean()), "recall100": float((ranks > 0).mean()),
        "ndcg20": float(DISCOUNT[ranks[top20] - 1].sum() / len(ranks)),
    }


def develop(train, valid, size):
    histories = {uid: train[uid] for uid in valid}
    trials = []
    for idf_power in IDF_POWERS:
        model = NeighborRetriever(train, size, idf_power)
        for recent in HISTORY_DECAYS:
            ranks = {}
            for uid, pop, candidates in model.candidate_lists(histories, recent, NEIGHBORS):
                for (neighbors, penalty), similar in candidates.items():
                    for weight in FUSION_WEIGHTS:
                        key = (neighbors, penalty, weight)
                        ranks.setdefault(key, []).append(target_rank(blend(pop, similar, weight), valid[uid]))
            for (neighbors, penalty, weight), values in ranks.items():
                config = dict(idf_power=idf_power, recent=recent, neighbors=neighbors,
                              popularity_penalty=penalty, collaborative_weight=weight)
                trials.append({"config": config, "metrics": summarize(values)})
            print(f"validation finished: idf_power={idf_power}, recent={recent}", flush=True)
    return sorted(trials, key=lambda trial: -trial["metrics"]["ndcg20"])


def evaluate(model, histories, targets, config):
    pop_ranks, model_ranks = [], []
    key = config["neighbors"], config["popularity_penalty"]
    for uid, pop, candidates in model.candidate_lists(histories, config["recent"], (key[0],)):
        pop_ranks.append(target_rank(pop, targets[uid]))
        model_ranks.append(target_rank(blend(pop, candidates[key], config["collaborative_weight"]), targets[uid]))
    result = {"popularity": summarize(pop_ranks), "model": summarize(model_ranks)}
    pop = np.asarray(pop_ranks)
    ranked = np.asarray(model_ranks)
    discounts = np.concatenate(([0.0], DISCOUNT, np.zeros(DEPTH - len(DISCOUNT))))
    delta = discounts[ranked] - discounts[pop]
    rng = np.random.default_rng(20261114)
    differences = [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(2000)]
    result["ndcg20_difference"] = float(delta.mean())
    result["ndcg20_paired_bootstrap_95_percent_interval"] = np.quantile(differences, [0.025, 0.975]).tolist()
    return result


def fresh_holdout(source, start, limit, item_map, development_users):
    histories, targets = {}, {}
    counts = Counter()
    with source.open(encoding="utf-8", newline="") as handle:
        for row, (uid, seq) in enumerate(groups(csv.DictReader(handle), start + limit)):
            if row < start:
                continue
            if uid in development_users:
                raise ValueError(f"Holdout user overlaps with the fit cohort: {uid}")
            counts["users_scanned"] += 1
            if len(seq) < 5:
                counts["history_too_short"] += 1
            elif seq[-1] in seq[:-1]:
                counts["repeat_target"] += 1
            elif seq[-1] not in item_map:
                counts["target_outside_catalog"] += 1
            else:
                history = [item_map[item] for item in seq[:-1] if item in item_map]
                if not history:
                    counts["no_known_history"] += 1
                    continue
                histories[uid] = history
                targets[uid] = item_map[seq[-1]]
    counts["evaluated_users"] = len(targets)
    return histories, targets, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("develop", "evaluate"))
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--holdout-start", type=int, default=10000)
    parser.add_argument("--holdout-users", type=int, default=10000)
    args = parser.parse_args()
    if args.holdout_start < 0 or args.holdout_users < 1:
        parser.error("holdout-start must be nonnegative and holdout-users must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.cohort.read_bytes()
    fingerprint = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    train, valid, size = data["train"], data["valid"], len(data["item_map"])
    frozen = args.out / "selection.json"
    if args.phase == "develop":
        if frozen.exists():
            raise FileExistsError(frozen)
        trials = develop(train, valid, size)
        baseline = popularity_metrics(train, valid, Counter(i for seq in train.values() for i in seq))
        with frozen.open("x", encoding="utf-8") as handle:
            json.dump({"cohort_sha256": fingerprint, "validation_popularity": baseline,
                       "selected": trials[0], "trials": trials}, handle, indent=2)
        print(json.dumps({"baseline": baseline, "selected": trials[0]}, indent=2))
        return
    selection = json.loads(frozen.read_text(encoding="utf-8"))
    if selection["cohort_sha256"] != fingerprint:
        raise ValueError("The frozen configuration was selected on a different cohort")
    config = selection["selected"]["config"]
    contexts, targets, counts = fresh_holdout(args.source, args.holdout_start, args.holdout_users,
                                               data["item_map"], train)
    if not targets:
        raise ValueError("No eligible targets in the new cohort")
    test = data["test"]
    test_contexts = test_histories_for(train, test, data.get("test_context_items", valid))
    model = NeighborRetriever(train, size, config["idf_power"])
    report = {
        "cohort_sha256": fingerprint, "fit_users": len(train), "catalog_items": size, "config": config,
        "fresh_cohort": {"source": str(args.source), "offset": args.holdout_start,
                         "requested_users": args.holdout_users, "counts": dict(counts),
                         **evaluate(model, contexts, targets, config)},
        "previous_test_cohort": evaluate(model, test_contexts, test, config),
    }
    with (args.out / "evaluation.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
