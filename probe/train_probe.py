"""Train a frozen response probe with When2Tool's estimator and settings."""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def load(path):
    d = torch.load(path, map_location="cpu", weights_only=True)
    return d["features"].float().numpy(), d["labels"].long().numpy()

def evaluate(x, y, clf):
    prob = clf.predict_proba(x)[:, 1]
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, clf.predict(x))),
            "auroc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else None}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reg", type=float, default=10.0,
                        help="When2Tool convention: sklearn C=1/reg.")
    parser.add_argument("--eval_features", default=None,
                        help="Optional diagnostic only; no neuron/threshold selection uses it.")
    args = parser.parse_args()
    if args.reg <= 0: raise ValueError("reg must be positive")
    x, y = load(args.train_features)
    if len(np.unique(y)) != 2: raise RuntimeError("Need reply and silence training labels.")
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # Exact When2Tool/src/train_probe.py classifier settings.
    clf = LogisticRegression(C=1.0/args.reg, solver="lbfgs", max_iter=2000, random_state=42)
    clf.fit(x, y)
    summary = {"reg": args.reg, "C": 1.0/args.reg, "n_features": int(x.shape[1]),
               "train": evaluate(x, y, clf)}
    if args.eval_features:
        ex, ey = load(args.eval_features)
        summary["eval"] = evaluate(scaler.transform(ex), ey, clf)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"coef": torch.from_numpy(clf.coef_[0]), "intercept": float(clf.intercept_[0]),
                "scaler_mean": torch.from_numpy(scaler.mean_),
                "scaler_scale": torch.from_numpy(scaler.scale_), "reg": args.reg,
                "C": 1.0/args.reg, "feature_order": "layer-major selected MLP activations"}, out)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); print(f"Saved probe: {out}")
if __name__ == "__main__":
    main()
