"""Video-grouped split for frozen Response-G1 probe features.

MMDuet2 supplies one label per original assistant turn.  All decision points of
one ``metadata.video_id`` stay in one partition, preventing near-identical video
prefixes from appearing in both training and validation data.
"""
import argparse
import json
from pathlib import Path

import torch
from sklearn.model_selection import GroupShuffleSplit


def subset(data, indices):
    return {
        "features": data["features"][indices],
        "labels": data["labels"][indices],
        "metadata": [data["metadata"][int(i)] for i in indices],
        "neuron_map": data["neuron_map"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--val_output", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error("--val_ratio must be in (0, 1)")

    data = torch.load(args.features, map_location="cpu", weights_only=True)
    metadata = data["metadata"]
    if len(metadata) != len(data["labels"]):
        raise RuntimeError("Feature, label and metadata lengths differ.")
    groups = [entry["video_id"] for entry in metadata]
    if len(set(groups)) < 2:
        raise RuntimeError("Need at least two video_id groups for validation.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio,
                                 random_state=args.seed)
    indices = torch.arange(len(groups)).numpy()
    train_idx, val_idx = next(splitter.split(indices, data["labels"].numpy(), groups))
    train, val = subset(data, train_idx), subset(data, val_idx)
    if len(torch.unique(train["labels"])) != 2 or len(torch.unique(val["labels"])) != 2:
        raise RuntimeError("Both reply and silence must occur in train and validation splits.")

    for path, payload in ((Path(args.train_output), train), (Path(args.val_output), val)):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
    manifest = {
        "source": args.features, "group_key": "metadata.video_id",
        "val_ratio": args.val_ratio, "seed": args.seed,
        "train_decisions": len(train_idx), "val_decisions": len(val_idx),
        "train_videos": len({groups[int(i)] for i in train_idx}),
        "val_videos": len({groups[int(i)] for i in val_idx}),
    }
    Path(args.val_output).with_suffix(".split.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
