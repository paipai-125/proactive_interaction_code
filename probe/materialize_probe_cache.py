"""Turn sharded FP16 data-parallel activation caches into probe features on CPU."""
import argparse
import json

import torch


def main():
    parser = argparse.ArgumentParser(description="Slice neuron-map features from data-parallel FP16 cache shards")
    parser.add_argument("--cache_manifest", required=True)
    parser.add_argument("--neuron_map", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.cache_manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    neuron_map = torch.load(args.neuron_map, map_location="cpu", weights_only=True)
    if list(manifest["layer_names"]) != list(neuron_map["layer_names"]):
        raise RuntimeError("Cache layer order does not match the neuron map.")
    indices = neuron_map["indices"].long()
    feature_parts, label_parts, metadata = [], [], []
    for path in manifest["shards"]:
        if path is None:
            continue
        cache = torch.load(path, map_location="cpu", weights_only=True)
        activations = cache["activations"].float()
        selected = torch.cat([activations[:, layer, ids] for layer, ids in enumerate(indices)], dim=1)
        feature_parts.append(selected)
        label_parts.append(cache["labels"].long())
        metadata.extend(cache["metadata"])
    width = indices.numel()
    torch.save({
        "features": torch.cat(feature_parts, dim=0) if feature_parts else torch.empty(0, width),
        "labels": torch.cat(label_parts, dim=0) if label_parts else torch.empty(0, dtype=torch.long),
        "metadata": metadata,
        "neuron_map": args.neuron_map,
        "cache_manifest": args.cache_manifest,
    }, args.output)
    print(f"Saved {sum(x.shape[0] for x in feature_parts)} cached decision features to {args.output}")


if __name__ == "__main__":
    main()
