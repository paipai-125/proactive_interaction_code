"""Visualize Response-G1 frozen-probe selected MLP neurons.

Adapted from WAFL/visualize_top_neuron_heatmaps.py. The input is the
``neuron_map.pt`` written by ``probe/select_neurons.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import torch


def load_neuron_map(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"indices", "scores", "layer_names", "topk", "topk_ratio"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"{path} is not a probe neuron map; missing {sorted(missing)}")
    indices = payload["indices"].detach().cpu().numpy()
    scores = payload["scores"].detach().cpu().numpy().astype(np.float64)
    names = list(payload["layer_names"])
    if indices.shape != scores.shape or indices.ndim != 2 or len(names) != indices.shape[0]:
        raise ValueError("indices, scores, and layer_names have incompatible shapes")
    return payload, names, indices, scores


def score_norm(scores: np.ndarray):
    low, high = float(np.nanmin(scores)), float(np.nanmax(scores))
    if low < 0 < high:
        bound = max(abs(low), abs(high))
        return TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound), "coolwarm"
    if high <= 0:
        return Normalize(vmin=low, vmax=0.0 if low < 0 else 1.0), "coolwarm_r"
    return Normalize(vmin=0.0, vmax=high), "magma"


def robust_positive_norm(scores: np.ndarray, percentile: float):
    finite = scores[np.isfinite(scores)]
    vmax = float(np.percentile(finite, percentile))
    if vmax <= 0:
        vmax = float(np.max(finite))
    return Normalize(vmin=0.0, vmax=max(vmax, np.finfo(np.float64).eps))


def render_heatmap(matrix, layer_names, title, xlabel, color_label, output, cmap, norm):
    fig, axis = plt.subplots(
        figsize=(max(10, matrix.shape[1] * 0.12), max(5, matrix.shape[0] * 0.34)),
        dpi=180,
    )
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    axis.set_yticks(np.arange(len(layer_names)), layer_names, fontsize=7)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Qwen3-VL MLP layer")
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label=color_label)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_layer_summary(scores, output, title):
    layer = np.arange(scores.shape[0])
    top1 = scores[:, 0]
    mean = scores.mean(axis=1)
    fig, axis = plt.subplots(figsize=(12, 4.5), dpi=180)
    axis.bar(layer - 0.2, top1, width=0.4, label="top-1 selected score")
    axis.bar(layer + 0.2, mean, width=0.4, label="mean selected score")
    axis.set_xticks(layer)
    axis.set_xlabel("Qwen3-VL MLP layer index")
    axis.set_ylabel("signed reply-sensitive score")
    axis.set_title(f"{title}: layer-level selected-neuron strength")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_csv(output: Path, names, indices, scores):
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "rank", "mlp_neuron_index", "signed_reply_score"])
        for layer, name in enumerate(names):
            for rank, (neuron, score) in enumerate(zip(indices[layer], scores[layer]), start=1):
                writer.writerow([name, rank, int(neuron), float(score)])


def main():
    parser = argparse.ArgumentParser(description="Visualize selected Response-G1 probe neurons")
    parser.add_argument("--neuron-map", type=Path, required=True,
                        help="neuron_map.pt from probe/select_neurons.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Response-G1 probe selected neurons")
    parser.add_argument("--display-topk", type=int, default=100,
                        help="Show this many highest-ranked neurons per layer in the signed and robust heatmaps")
    parser.add_argument("--robust-percentile", type=float, default=99.0,
                        help="Upper color limit percentile for the Top-k robust heatmap")
    args = parser.parse_args()

    payload, names, indices, scores = load_neuron_map(args.neuron_map)
    if args.display_topk < 1:
        parser.error("--display-topk must be positive")
    if args.display_topk > scores.shape[1]:
        print(f"[visualization] requested top {args.display_topk}, but only {scores.shape[1]} selected per layer; using all.")
        args.display_topk = scores.shape[1]
    if not 0 < args.robust_percentile <= 100:
        parser.error("--robust-percentile must be in (0, 100]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "selected_neurons.csv", names, indices, scores)

    # 1) Signed view over the same top ranks as the focused figure.  Limiting
    # columns prevents the long low-score tail of all selected neurons from
    # compressing the visually informative leading ranks.
    displayed = scores[:, :args.display_topk]
    norm, cmap = score_norm(displayed)
    render_heatmap(displayed, names,
                   f"{args.title}: top {args.display_topk} global signed scores",
                   "selected-neuron rank within layer", "C[l, i] (signed reply score)",
                   args.output_dir / "signed_score_heatmap.png", cmap, norm)

    # 2) Focused display: avoids a few extreme top-1 neurons hiding all other ranks.
    focused = displayed
    render_heatmap(focused, names,
                   f"{args.title}: top {args.display_topk} ranks (robust {args.robust_percentile:g}th percentile scale)",
                   "selected-neuron rank within layer", "C[l, i], clipped for display",
                   args.output_dir / "robust_top_rank_heatmap.png", "magma",
                   robust_positive_norm(focused, args.robust_percentile))

    # 3) Within-layer relative rank profile: highlights decay/structure, not magnitude.
    denominator = np.maximum(np.max(np.abs(scores), axis=1, keepdims=True), np.finfo(np.float64).eps)
    row_normalized = scores / denominator
    render_heatmap(row_normalized, names, f"{args.title}: within-layer normalized scores",
                   "selected-neuron rank within layer", "C[l, i] / max_j |C[l, j]|",
                   args.output_dir / "row_normalized_score_heatmap.png", "magma", Normalize(vmin=0.0, vmax=1.0))

    # 4) Compact layer-level comparison.
    render_layer_summary(scores, args.output_dir / "layer_score_summary.png", args.title)

    render_heatmap(indices, names, f"{args.title}: selected MLP dimensions",
                   "selected-neuron rank within layer", "MLP intermediate dimension index",
                   args.output_dir / "neuron_index_heatmap.png", "viridis",
                   Normalize(vmin=0, vmax=float(indices.max())))

    summary = {
        "source": str(args.neuron_map), "formula": payload.get("formula"),
        "selection": payload.get("selection"), "topk_ratio": float(payload["topk_ratio"]),
        "topk_per_layer": int(payload["topk"]), "display_topk": args.display_topk,
        "robust_percentile": args.robust_percentile, "layers": len(names),
        "score_min": float(scores.min()), "score_max": float(scores.max()),
        "paired_question_count": int(payload.get("paired_question_count", 0)),
        "skipped_episodes": payload.get("skipped_episodes", {}),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved visualizations to: {args.output_dir}")


if __name__ == "__main__":
    main()
