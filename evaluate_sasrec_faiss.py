"""Evaluate the selected SASRec checkpoint once on validation and test."""
import argparse
import json
from pathlib import Path

import torch

from prepare_sasrec_cohort import test_histories_for
from train_sasrec_faiss import SASRec, load_data, retrieval_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    train, valid, test, num_items, test_context_items = load_data(args.data.resolve() / "cohort.json")
    checkpoint = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=True)
    model = SASRec(num_items, args.max_len, args.embedding_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    validation = retrieval_metrics(model, train, valid, args.max_len, device)
    test_histories = test_histories_for(train, test, test_context_items)
    test_results = retrieval_metrics(model, test_histories, test, args.max_len, device)
    result = {"selected_epoch": checkpoint["epoch"], "validation": validation, "test": test_results,
              "training_device": str(device), "candidate_scope": "items observed in the selected training cohort"}
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
