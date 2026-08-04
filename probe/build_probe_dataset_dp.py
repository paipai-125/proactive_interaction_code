"""Single-node data-parallel Response-G1 probe-data construction.

Launch with torchrun.  Each rank owns one complete Qwen3-VL model on one GPU
and processes a disjoint modulo shard of the annotation JSONL.  This contrasts
with probe.multigpu's model-parallel loader, which splits one model across GPUs.
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm.auto import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probe.build_probe_dataset import process_record  # noqa: E402
from probe.runtime import Qwen3VLMLPCollector  # noqa: E402


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    elif not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    return rank, local_rank, world_size


def load_single_gpu_model(ckpt_path, local_rank):
    """Use the same Qwen3-VL/FlashAttention settings as probe.multigpu, on one GPU."""
    device = torch.device(f"cuda:{local_rank}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        ckpt_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).to(device).eval()
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 3
    print(f"[data-parallel rank={local_rank}] full model on {device}; allocated={allocated:.2f} GiB", flush=True)
    return model


def reduce_stats(stats, world_size):
    if world_size == 1:
        return stats
    delta = stats["episode_delta_sum"].to(torch.cuda.current_device())
    dist.all_reduce(delta, op=dist.ReduceOp.SUM)
    stats["episode_delta_sum"] = delta.cpu()
    counts = torch.tensor(
        [stats["paired_question_count"], stats["candidate_pairs_selected"]],
        dtype=torch.long,
        device=torch.cuda.current_device(),
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    stats["paired_question_count"] = int(counts[0].item())
    stats["candidate_pairs_selected"] = int(counts[1].item())
    all_skipped = [None] * world_size
    dist.all_gather_object(all_skipped, stats["skipped_episodes"])
    merged = defaultdict(int)
    for item in all_skipped:
        for reason, count in item.items():
            merged[reason] += count
    stats["skipped_episodes"] = dict(merged)
    return stats


def save_rank_cache(cache, path, collector, width, rank):
    if cache is None:
        return None
    rank_path = Path(f"{path}.rank{rank}.pt")
    payload = {
        "activations": torch.stack(cache["activations"]) if cache["activations"] else torch.empty(
            0, len(collector.modules), width, dtype=torch.float16
        ),
        "labels": torch.tensor(cache["labels"], dtype=torch.long),
        "metadata": cache["metadata"],
        "layer_names": cache["layer_names"],
    }
    torch.save(payload, rank_path)
    with open(str(rank_path) + ".jsonl", "w", encoding="utf-8") as handle:
        for item in cache["metadata"]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return str(rank_path)


def merge_feature_shards(output, rank_paths):
    features, labels, metadata, neuron_map = [], [], [], None
    for path in rank_paths:
        shard = torch.load(path, map_location="cpu", weights_only=True)
        features.append(shard["features"])
        labels.append(shard["labels"])
        metadata.extend(shard["metadata"])
        neuron_map = shard["neuron_map"]
    width = features[0].shape[1] if features else 0
    torch.save({
        "features": torch.cat(features, dim=0) if features else torch.empty(0, width),
        "labels": torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.long),
        "metadata": metadata,
        "neuron_map": neuron_map,
    }, output)


def select_video_ids(annotation_path, max_videos):
    """Return the first ``max_videos`` distinct MMDuet2 video IDs in JSONL order.

    This is intentionally performed identically by every torchrun rank before
    the modulo sharding below.  Thus --max_videos is a global video-level cap,
    not a per-rank cap and not a cap on question episodes.
    """
    selected = []
    seen = set()
    with open(annotation_path, encoding="utf-8") as handle:
        for record_index, line in enumerate(handle):
            record = json.loads(line)
            try:
                video_id = record["metadata"]["video_id"]
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"Record {record_index} has no metadata.video_id; "
                    "--max_videos requires MMDuet2-style annotations."
                ) from exc
            if video_id not in seen:
                seen.add(video_id)
                selected.append(video_id)
                if len(selected) == max_videos:
                    break
    if len(selected) < max_videos:
        raise ValueError(
            f"Requested --max_videos {max_videos}, but annotations contain only "
            f"{len(selected)} distinct video IDs."
        )
    return set(selected)


def count_local_records(annotation_path, selected_video_ids, max_records, rank, world_size):
    """Count this rank's input records once, so the tqdm total is exact."""
    total = 0
    with open(annotation_path, encoding="utf-8") as handle:
        for record_index, line in enumerate(handle):
            if max_records is not None and record_index >= max_records:
                break
            if record_index % world_size != rank:
                continue
            if selected_video_ids is not None:
                record = json.loads(line)
                if record.get("metadata", {}).get("video_id") not in selected_video_ids:
                    continue
            total += 1
    return total


def build_parser():
    parser = argparse.ArgumentParser(description="Data-parallel probe stats/features; one full Qwen3-VL per GPU")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=("stats", "features"))
    parser.add_argument("--neuron_map")
    parser.add_argument("--decision_cache", default=None,
                        help="Stats: prefix for one FP16 cache shard per GPU. Features: unsupported; use materialize_probe_cache.py.")
    parser.add_argument("--max_records", type=int, default=None,
                        help="Global JSONL-row cap; use --max_videos for a video-level experiment cap.")
    parser.add_argument("--max_videos", type=int, default=None,
                        help="Global cap on distinct metadata.video_id values; all questions in each selected video remain.")
    parser.add_argument("--max_question_pairs_per_rank", type=int, default=None,
                        help="Optional cap independently applied on each GPU rank; total is at most this value times world size.")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--min_pixels", type=int, default=448 * 448)
    parser.add_argument("--max_pixels", type=int, default=448 * 448)
    parser.add_argument("--visual_context_frames", type=int, default=None)
    return parser


def main():
    args = build_parser().parse_args()
    if args.fps != 1 or args.frame_interval != 1:
        raise ValueError("MMDuet2 Live-WhisperX is fixed to fps=1 and frame_interval=1.")
    if args.visual_context_frames is not None and args.visual_context_frames < 1:
        raise ValueError("--visual_context_frames must be positive.")
    if args.mode == "features" and not args.neuron_map:
        raise ValueError("--neuron_map is required in features mode.")
    if args.mode == "features" and args.decision_cache:
        raise ValueError("Use probe/materialize_probe_cache.py for sharded FP16 cache replay.")
    if args.max_question_pairs_per_rank is not None and args.max_question_pairs_per_rank < 1:
        raise ValueError("--max_question_pairs_per_rank must be positive.")
    if args.max_videos is not None and args.max_videos < 1:
        raise ValueError("--max_videos must be positive.")
    if args.max_videos is not None and args.max_records is not None:
        raise ValueError("Use only one of --max_videos and --max_records.")

    selected_video_ids = (select_video_ids(args.annotations, args.max_videos)
                          if args.max_videos is not None else None)
    rank, local_rank, world_size = distributed_context()
    # Keep terminal output readable under torchrun: rank 0 owns the visible
    # progress bar, while nonzero ranks keep writing their result shards/logs.
    if rank != 0:
        _rank_output_sink = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = _rank_output_sink
        sys.stderr = _rank_output_sink
    if rank == 0 and selected_video_ids is not None:
        print(f"[data-parallel] globally selected {len(selected_video_ids)} videos; "
              "all question episodes in those videos will be processed.", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    # process_record expects this name; it is intentionally a per-rank cap.
    args.max_question_pairs = args.max_question_pairs_per_rank
    model = load_single_gpu_model(args.ckpt_path, local_rank)
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels, trust_remote_code=True
    )
    collector = Qwen3VLMLPCollector(model)
    neuron_indices = None
    if args.mode == "features":
        neuron_map = torch.load(args.neuron_map, map_location="cpu", weights_only=True)
        neuron_indices = neuron_map["indices"].long()
        if list(neuron_map["layer_names"]) != collector.layer_names:
            raise RuntimeError("Neuron-map layer order differs from the loaded Qwen3-VL model.")
    width = collector.down_projection_norms().shape[1]
    stats = {
        "episode_delta_sum": torch.zeros(len(collector.modules), width),
        "paired_question_count": 0,
        "candidate_pairs_selected": 0,
        "skipped_episodes": {},
        "down_projection_norms": collector.down_projection_norms(),
        "layer_names": collector.layer_names,
    }
    decision_cache = ({"activations": [], "labels": [], "metadata": [], "layer_names": collector.layer_names}
                      if args.mode == "stats" and args.decision_cache else None)
    features, labels, metadata = [], [], []
    decisions = 0
    local_total = count_local_records(
        args.annotations, selected_video_ids, args.max_records, rank, world_size
    )
    progress = tqdm(
        total=local_total,
        desc=f"rank {rank} input videos",
        position=rank,
        leave=True,
        dynamic_ncols=True,
        unit="video",
    )
    with open(args.annotations, encoding="utf-8") as handle:
        for record_index, line in enumerate(handle):
            if args.max_records is not None and record_index >= args.max_records:
                break
            if record_index % world_size != rank:
                continue
            record = json.loads(line)
            if selected_video_ids is not None:
                video_id = record.get("metadata", {}).get("video_id")
                if video_id not in selected_video_ids:
                    continue
            decisions += process_record(
                record, args, model, processor, collector, neuron_indices,
                stats, features, labels, metadata, decision_cache,
            )
            progress.update(1)
            progress.set_postfix(decisions=decisions)
    progress.close()

    if args.mode == "stats":
        cache_path = save_rank_cache(decision_cache, args.decision_cache, collector, width, rank)
        stats = reduce_stats(stats, world_size)
        cache_paths = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(cache_paths, cache_path)
        else:
            cache_paths[0] = cache_path
        if rank == 0:
            torch.save(stats, args.output)
            if args.decision_cache:
                manifest = {"format": "response_g1_probe_dp_cache_v1", "layer_names": collector.layer_names,
                            "shards": cache_paths}
                with open(str(args.decision_cache) + ".manifest.json", "w", encoding="utf-8") as handle:
                    json.dump(manifest, handle, ensure_ascii=False, indent=2)
            print(f"Saved merged stats from {stats['paired_question_count']} paired questions to {args.output}")
    else:
        rank_path = f"{args.output}.rank{rank}.pt"
        torch.save({
            "features": torch.stack(features) if features else torch.empty(0, neuron_indices.numel()),
            "labels": torch.tensor(labels, dtype=torch.long), "metadata": metadata,
            "neuron_map": args.neuron_map,
        }, rank_path)
        rank_paths = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(rank_paths, rank_path)
        else:
            rank_paths[0] = rank_path
        if rank == 0:
            merge_feature_shards(args.output, rank_paths)
            print(f"Saved merged selected features to {args.output}")

    collector.close()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
