"""Choose the frozen-probe reply threshold on a held-out video split."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, help="Checkpoint from train_probe.py")
    parser.add_argument("--val_features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold_decimals", type=int, default=2,
                        help="Select and save the best reply threshold on this decimal grid (default: 2).")
    args = parser.parse_args()
    if args.threshold_decimals < 0:
        raise ValueError("--threshold_decimals must be non-negative.")

    probe = torch.load(args.probe, map_location="cpu", weights_only=True)
    data = torch.load(args.val_features, map_location="cpu", weights_only=True)
    x = data["features"].float().numpy()
    y = data["labels"].long().numpy()
    if len(np.unique(y)) != 2:
        raise RuntimeError("Validation split must contain reply and silence labels.")

    scaler = StandardScaler()
    scaler.mean_ = probe["scaler_mean"].numpy()
    scaler.scale_ = probe["scaler_scale"].numpy()
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(scaler.mean_)
    x = scaler.transform(x)
    logits = x @ probe["coef"].numpy() + probe["intercept"]
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    _, _, thresholds = precision_recall_curve(y, probabilities)
    if len(thresholds) == 0:
        raise RuntimeError("No threshold candidates returned by precision_recall_curve.")

    # The saved value is the value used at inference.  Search directly over
    # the rounded grid, so the displayed two-decimal threshold and reported
    # validation metrics are exactly consistent.
    candidate_thresholds = np.unique(np.round(thresholds.astype(np.float64), args.threshold_decimals))
    best_threshold, best_f1, best_precision, best_recall = None, -np.inf, 0.0, 0.0
    for threshold in candidate_thresholds:
        predicted_reply = probabilities >= threshold
        true_positive = int(np.logical_and(predicted_reply, y == 1).sum())
        false_positive = int(np.logical_and(predicted_reply, y == 0).sum())
        false_negative = int(np.logical_and(~predicted_reply, y == 1).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), float(f1)
            best_precision, best_recall = float(precision), float(recall)

    result = {
        "threshold": best_threshold,
        "threshold_decimals": args.threshold_decimals,
        "selection_metric": "reply_f1",
        "reply_f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "n_validation_decisions": int(len(y)),
        "positive_reply_decisions": int(y.sum()),
        "probe": args.probe,
        "val_features": args.val_features,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
