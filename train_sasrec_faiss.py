"""Train a SASRec-style next-item model and evaluate exact FAISS Top-K retrieval."""
import argparse
import json
import math
import random
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from prepare_sasrec_cohort import test_histories_for


class PrefixPairs(Dataset):
    def __init__(self, sequences, max_len, num_items, seed, negative_sampling="uniform",
                 sequence_batches=False):
        self.sequences = sequences
        self.max_len = max_len
        self.num_items = num_items
        self.rng = random.Random(seed)
        self.sequence_batches = sequence_batches
        self.stride = max(1, max_len // 2) if sequence_batches else 1
        self.pairs = [(u, t) for u, seq in sequences.items() for t in range(1, len(seq), self.stride)]
        self.seen = {u: set(seq) for u, seq in sequences.items()}
        self.mix_rng = random.Random(seed + 1) if negative_sampling == "mixed" else None
        if self.mix_rng is not None:
            counts = Counter(item for seq in sequences.values() for item in seq)
            self.popularity_cdf = np.cumsum([counts[i] for i in range(1, num_items + 1)]).tolist()

    def sample_negative(self, user):
        use_popularity = self.mix_rng is not None and self.mix_rng.random() < 0.5
        for _ in range(101):
            if use_popularity:
                draw = self.rng.randrange(self.popularity_cdf[-1])
                item = bisect_right(self.popularity_cdf, draw) + 1
            else:
                item = self.rng.randint(1, self.num_items)
            if item not in self.seen[user]:
                return item
        return 0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        user, target_pos = self.pairs[idx]
        seq = self.sequences[user]
        if self.sequence_batches:
            stop = min(target_pos + self.stride, len(seq))
            start = max(1, stop - self.max_len)
            context = seq[start - 1:stop - 1]
            warmup = target_pos - start
            padding = [0] * (self.max_len - len(context))
            positives = [0] * warmup + seq[target_pos:stop] + padding
            negatives = [0] * warmup + [self.sample_negative(user) for _ in range(target_pos, stop)] + padding
            return torch.tensor(context + padding), torch.tensor(positives), torch.tensor(negatives)
        context = seq[max(0, target_pos - self.max_len):target_pos]
        neg = self.sample_negative(user)
        padded = context + [0] * (self.max_len - len(context))
        return torch.tensor(padded), torch.tensor(seq[target_pos]), torch.tensor(neg)


class SASRec(nn.Module):
    def __init__(self, num_items, max_len=50, dim=64, heads=2, dropout=0.1):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                                           dropout=dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def encode_positions(self, seq):
        positions = torch.arange(self.max_len, device=seq.device).unsqueeze(0)
        x = self.item_embedding(seq) * math.sqrt(self.item_embedding.embedding_dim)
        x = self.dropout(x + self.position_embedding(positions))
        causal = torch.triu(torch.ones(self.max_len, self.max_len, device=seq.device, dtype=torch.bool), diagonal=1)
        return self.norm(self.encoder(x, mask=causal, src_key_padding_mask=(seq == 0)))

    def encode(self, seq):
        x = self.encode_positions(seq)
        last = (seq != 0).sum(dim=1).clamp(min=1) - 1
        return x[torch.arange(len(seq), device=seq.device), last]


def make_index(model):
    items = model.item_embedding.weight.detach().cpu().numpy()[1:].astype("float32", copy=True)
    index = faiss.IndexFlatIP(items.shape[1])
    index.add(items)
    return index


@torch.no_grad()
def retrieval_metrics(model, seqs, targets, max_len, device, k=20, batch_users=256):
    model.eval()
    index = make_index(model)
    ndcg, hit, hit100 = [], [], []
    users = list(targets)
    depth = max(k, 100)
    for start in range(0, len(users), batch_users):
        chunk = users[start:start + batch_users]
        contexts = []
        max_search = depth
        for uid in chunk:
            hist = seqs[uid][-max_len:]
            contexts.append(hist + [0] * (max_len - len(hist)))
            max_search = max(max_search, len(seqs[uid]) + depth)
        query = model.encode(torch.tensor(contexts, dtype=torch.long, device=device)).cpu().numpy().astype("float32")
        n_search = min(index.ntotal, max_search)
        _, item_ids = index.search(query, n_search)
        for uid, ranked in zip(chunk, item_ids):
            target = targets[uid] - 1
            seen = set(seqs[uid])
            filtered = [int(i) for i in ranked if i >= 0 and (i + 1) not in seen]
            rank = next((r for r, item in enumerate(filtered[:k]) if item == target), None)
            found = rank is not None
            hit.append(float(found))
            hit100.append(target in filtered[:100])
            ndcg.append(1.0 / math.log2(rank + 2) if found else 0.0)
    return {f"recall{k}": float(np.mean(hit)), f"hit{k}": float(np.mean(hit)),
            f"ndcg{k}": float(np.mean(ndcg)), "recall100": float(np.mean(hit100)),
            "users": len(users), "catalog_items": index.ntotal, "retrieval": "FAISS IndexFlatIP exact inner-product search"}


def load_data(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (payload["train"], payload["valid"], payload["test"], len(payload["item_map"]),
            payload.get("test_context_items", payload["valid"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--initial-checkpoint", type=Path)
    ap.add_argument("--negative-sampling", choices=("uniform", "mixed"), default="uniform")
    ap.add_argument("--cpu-threads", type=int)
    ap.add_argument("--sequence-batches", action="store_true",
                    help="Train on overlapping causal windows and predict several positions per forward pass")
    args = ap.parse_args()
    if min(args.epochs, args.max_len, args.batch_size) < 1 or (
        args.cpu_threads is not None and args.cpu_threads < 1
    ):
        ap.error("epochs, max-len, batch-size, and cpu-threads must be positive")
    if args.cpu_threads is not None:
        torch.set_num_threads(args.cpu_threads)
        faiss.omp_set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    train, valid, test, num_items, test_context_items = load_data(args.data.resolve() / "cohort.json")
    if not train or not valid or not test:
        raise ValueError("Cohort needs warm train, validation, and test sequences")
    train_ds = PrefixPairs(train, args.max_len, num_items, args.seed, args.negative_sampling,
                           args.sequence_batches)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = SASRec(num_items, args.max_len, args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98))
    best_ndcg, best_epoch, records = -1.0, None, []
    epoch_offset = 0
    if args.initial_checkpoint:
        prior = torch.load(args.initial_checkpoint.resolve(), map_location=device, weights_only=True)
        model.load_state_dict(prior["model"])
        epoch_offset = int(prior["epoch"])
        prior_metrics = retrieval_metrics(model, train, valid, args.max_len, device)
        best_ndcg, best_epoch = prior_metrics["ndcg20"], epoch_offset
        torch.save(prior, out / "best_sasrec.pt")
        records.append({"epoch": epoch_offset, "train_pairwise_loss": None,
                        **{f"validation_{k}": v for k, v in prior_metrics.items()},
                        "optimizer_state": "fresh Adam state for continuation"})
        print(json.dumps(records[-1]), flush=True)
    for epoch in range(1, args.epochs + 1):
        absolute_epoch = epoch_offset + epoch
        model.train()
        total_loss, examples = 0.0, 0
        for seq, pos, neg in loader:
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            if args.sequence_batches:
                has_target = pos != 0
                query = model.encode_positions(seq)[has_target]
                pos, neg = pos[has_target], neg[has_target]
            else:
                query = model.encode(seq)
            pos_emb = model.item_embedding(pos)
            neg_emb = model.item_embedding(neg)
            pos_score = (query * pos_emb).sum(-1)
            neg_score = (query * neg_emb).sum(-1)
            valid_neg = (neg != 0).float()
            loss_each = torch.nn.functional.softplus(neg_score - pos_score) * valid_neg
            loss = loss_each.sum() / valid_neg.sum().clamp(min=1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(pos)
            examples += len(pos)
        val_metrics = retrieval_metrics(model, train, valid, args.max_len, device)
        row = {"epoch": absolute_epoch, "train_pairwise_loss": total_loss / max(1, examples),
               **{f"validation_{k}": v for k, v in val_metrics.items()}}
        records.append(row)
        print(json.dumps(row), flush=True)
        if val_metrics["ndcg20"] > best_ndcg:
            best_ndcg, best_epoch = val_metrics["ndcg20"], absolute_epoch
            torch.save({"model": model.state_dict(), "epoch": absolute_epoch}, out / "best_sasrec.pt")
        (out / "validation_by_epoch.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    checkpoint = torch.load(out / "best_sasrec.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_metrics = retrieval_metrics(model, test_histories_for(train, test, test_context_items), test,
                                     args.max_len, device)
    result = {"selected_epoch": best_epoch, "validation_ndcg20": best_ndcg,
              "test": test_metrics,
              "training": {"seed": args.seed, "total_epochs_run": epoch_offset + args.epochs,
                           "epochs_this_invocation": args.epochs,
                           "initial_checkpoint": str(args.initial_checkpoint.resolve()) if args.initial_checkpoint else None,
                           "optimizer_state": "fresh Adam state for continuation" if args.initial_checkpoint else "single run",
                           "batch_size": args.batch_size,
                           "max_len": args.max_len, "embedding_dim": args.embedding_dim,
                           "learning_rate": args.learning_rate, "device": str(device),
                           "cpu_threads": args.cpu_threads,
                           "negative_sampling": args.negative_sampling,
                           "sequence_batches": args.sequence_batches,
                           "objective": "sampled-negative pairwise next-item loss"}}
    (out / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    config = {"seed": args.seed, "total_epochs_run": epoch_offset + args.epochs,
              "epochs_this_invocation": args.epochs, "batch_size": args.batch_size,
              "max_len": args.max_len, "embedding_dim": args.embedding_dim,
              "learning_rate": args.learning_rate, "device": str(device),
              "cpu_threads": args.cpu_threads,
              "negative_sampling": args.negative_sampling,
              "sequence_batches": args.sequence_batches,
              "initial_checkpoint": str(args.initial_checkpoint.resolve()) if args.initial_checkpoint else None,
              "optimizer_state": "fresh Adam state for continuation" if args.initial_checkpoint else "single run"}
    (out / "training_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
