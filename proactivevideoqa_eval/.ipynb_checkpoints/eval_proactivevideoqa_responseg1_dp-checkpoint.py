
#!/usr/bin/env python3
"""Data-parallel ProactiveVideoQA evaluation for Response-G1 / frozen probe.

Reference implementation:
  - probe/build_probe_dataset_dp.py: one full Qwen3-VL per LOCAL_RANK and
    deterministic modulo sharding.
  - proactivevideoqa_eval/eval_proactivevideoqa_responseg1.py: unchanged
    MMDuet2 frame-input stream, Response-G1 graph/trigger, and output format.

Launch with torchrun.  ``WORLD_SIZE`` GPUs load WORLD_SIZE independent full
models; example i is evaluated by rank ``i % WORLD_SIZE``.  Rank 0 merges the
rank JSONL shards into ``--output`` after all ranks finish.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe.build_probe_dataset_dp import distributed_context, load_single_gpu_model
from probe.runtime import Qwen3VLMLPCollector, load_frozen_probe
from proactivevideoqa_eval.eval_proactivevideoqa_responseg1 import run_one


def build_parser():
    p = argparse.ArgumentParser(
        "Data-parallel Response-G1 ProactiveVideoQA evaluation: one full model per GPU"
    )
    p.add_argument("--frame_input", required=True, help="MMDuet2 *-frame_input_format.json")
    p.add_argument("--gold_file", required=True, help="MMDuet2 *-proactivevideoqa_format.json")
    p.add_argument("--video_root", required=True, help="Extracted ProactiveVideoQA subset videos directory")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output", required=True, help="Merged MMDuet2-compatible prediction JSONL")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Global sample cap before rank sharding; omit for all samples")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--frame_interval", type=float, default=1.0)
    p.add_argument("--visual_context_frames", type=int, default=None)
    p.add_argument("--min_pixels", type=int, default=448 * 448)
    p.add_argument("--max_pixels", type=int, default=448 * 448)
    p.add_argument("--probe_path", default=None,
                   help="Omit for the original Response-G1 generated trigger baseline")
    p.add_argument("--neuron_map", default=None)
    p.add_argument("--probe_threshold", type=float, default=None)
    return p


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if "question_id" in row:
                    ids.add(str(row["question_id"]))
    return ids


def merge_rank_outputs(output: Path, shard_paths: list[Path], existing_output: Path | None):
    """Same ID-preserving JSONL merge policy as existing Response-G1 DP evaluators."""
    by_id = {}
    for path in ([existing_output] if existing_output and existing_output.exists() else []) + shard_paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if "question_id" in row:
                        by_id[str(row["question_id"])] = row
    with output.open("w", encoding="utf-8") as handle:
        for qid in sorted(by_id):
            handle.write(json.dumps(by_id[qid], ensure_ascii=False) + "\n")
    return len(by_id)


def main():
    args = build_parser().parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.visual_context_frames is not None and args.visual_context_frames < 1:
        raise ValueError("--visual_context_frames must be >= 1")
    if args.probe_path and (not args.neuron_map or args.probe_threshold is None):
        raise ValueError("--probe_path requires --neuron_map and --probe_threshold")

    rank, local_rank, world_size = distributed_context()
    args.rank, args.local_rank, args.world_size = rank, local_rank, world_size
    # Non-zero ranks keep their own per-rank log file instead of discarding
    # output: a crash traceback must stay recoverable for diagnosis.
    rank_log = None
    if rank != 0:
        rank_log = open(Path(f"{args.output}.rank{rank}.log"), "w", encoding="utf-8")
        sys.stdout = rank_log
        sys.stderr = rank_log

    frame_data = json.load(open(args.frame_input, encoding="utf-8"))
    gold = {x["question_id"]: x for x in json.load(open(args.gold_file, encoding="utf-8"))}
    end = len(frame_data) if args.end_idx is None else args.end_idx
    if args.max_samples is not None:
        end = min(end, args.start_idx + args.max_samples)
    selected = frame_data[args.start_idx:end]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = read_existing_ids(output)
    pending = [item for item in selected if str(item["question_id"]) not in completed]
    local_items = [item for local_index, item in enumerate(pending)
                   if local_index % world_size == rank]
    shard = Path(f"{args.output}.rank{rank}.jsonl")
    # The rank shard belongs exclusively to this invocation.  Previous merged
    # output remains untouched until rank 0 has all new shards.
    with shard.open("w", encoding="utf-8"):
        pass

    if rank == 0:
        print(f"[data-parallel] world_size={world_size}; selected={len(selected)}; "
              f"already_complete={len(completed)}; pending={len(pending)}", flush=True)
        if world_size > 1:
            print(f"[data-parallel] rank logs -> {output}.rank*.log | "
                  f"failed samples -> {output}.rank*.errors.jsonl", flush=True)

    # Exact single-GPU loader used by the existing probe DP data construction.
    model = load_single_gpu_model(args.ckpt_path, local_rank)
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    collector = Qwen3VLMLPCollector(model) if args.probe_path else None
    probe = load_frozen_probe(args.probe_path, args.neuron_map) if args.probe_path else None

    errors_path = Path(f"{args.output}.rank{rank}.errors.jsonl")
    # Reset the per-run error log before appending below: stale failures from
    # a previous invocation must not pollute this run's failure summary.
    with errors_path.open("w", encoding="utf-8"):
        pass
    with shard.open("a", encoding="utf-8") as handle, torch.no_grad():
        error_handle = errors_path.open("a", encoding="utf-8")
        iterator = tqdm(local_items, total=len(local_items), desc="ProactiveVideoQA",
                        unit="sample", dynamic_ncols=True, disable=(rank != 0))
        for item in iterator:
            qid = item["question_id"]
            if qid not in gold:
                raise KeyError(f"{qid} absent from {args.gold_file}")
            try:
                result = run_one(item, gold, args, model, processor, probe, collector)
            except Exception as exc:
                # A single bad sample (missing/corrupt video, frame decode
                # failure, transient CUDA error) must not take down the whole
                # process group: log it, skip it, and keep going.
                message = (f"[rank {rank}] question_id={qid} FAILED: "
                           f"{type(exc).__name__}: {exc}")
                print(message, flush=True)
                error_handle.write(json.dumps(
                    {"question_id": str(qid), "error": message}, ensure_ascii=False) + "\n")
                error_handle.flush()
                torch.cuda.empty_cache()
                continue
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
        error_handle.close()

    if collector:
        collector.close()
    # Every rank has closed its shard before rank 0 begins the merge.
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        shards = [Path(f"{args.output}.rank{i}.jsonl") for i in range(world_size)]
        previous = output if output.exists() else None
        merged = merge_rank_outputs(output, shards, previous)
        print(f"[data-parallel] merged {merged} ProactiveVideoQA results into {output}", flush=True)
        error_files = [Path(f"{args.output}.rank{i}.errors.jsonl") for i in range(world_size)]
        error_files = [p for p in error_files if p.exists()]
        if error_files:
            n_errors = sum(1 for p in error_files for _ in p.open(encoding="utf-8"))
            print(f"[data-parallel] WARNING: {n_errors} samples failed and were skipped; "
                  f"see {output}.rank*.errors.jsonl", flush=True)
    if world_size > 1:
        dist.destroy_process_group()
    if rank_log is not None:
        rank_log.close()


if __name__ == "__main__":
    main()
