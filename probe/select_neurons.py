"""Question-paired, signed response-neuron selection.

For every valid question episode q, Delta[q,l,i] is its reply activation mean
minus its pre-reply NO REPLY activation mean. The score is:

    C[l,i] = mean_q Delta[q,l,i] * ||W_down[l,:,i]||_2.

Scores retain their sign. Selecting the largest values therefore chooses only
neurons whose mean MLP intermediate value is higher when a reply is due.
"""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--topk", type=float, default=0.03,
                        help="Fraction of positive-score neurons to select independently per MLP layer (default: 0.03).")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    s = torch.load(args.stats, map_location="cpu", weights_only=True)
    paired_count = int(s["paired_question_count"])
    if paired_count == 0:
        raise RuntimeError("No valid question episodes with both reply and prior NO REPLY.")
    delta = s["episode_delta_sum"] / paired_count
    down_norm = s["down_projection_norms"]
    if delta.shape != down_norm.shape:
        raise RuntimeError("Activation and down-projection shapes differ.")
    if not 0 < args.topk <= 1:
        raise ValueError("topk must be a fraction in (0, 1].")
    topk_count = max(1, int(delta.shape[1] * args.topk))

    # No abs(): topk(largest=True) keeps only reply-higher, signed neurons.
    scores = delta * down_norm
    values, indices = torch.topk(scores, topk_count, dim=1, largest=True, sorted=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "formula": "mean_q(mean_reply_q-mean_pre_reply_no_reply_q)*l2(W_down[:,i])",
        "selection": "top_positive_signed_scores",
        "topk_ratio": args.topk,
        "topk": topk_count,
        "indices": indices,
        "scores": values,
        "layer_names": s["layer_names"],
        "paired_question_count": paired_count,
        "skipped_episodes": s.get("skipped_episodes", {}),
    }, args.output)
    print(f"Saved top {args.topk:.2%} = {topk_count} signed reply-sensitive neurons "
          f"in each of {indices.shape[0]} layers from {paired_count} paired questions: {args.output}")


if __name__ == "__main__":
    main()
