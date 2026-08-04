"""Build Response-G1 probe data from MMDuet2's author-provided SFT labels.

The stream, frame encoding, graph generation/retrieval, trigger prompt and
assistant-prefix decision position are reused from eval_ovobench_proactive.py.
MMDuet2 supplies only the supervision: assistant ``NO REPLY`` is silence=0;
every other assistant response is reply=1.

For Live-WhisperX, MMDuet2's supplied HDF5 asset name contains its source FPS
(``..._2.0fps.hdf5``), while the SFT selects indices ``0, 2, 4, ...``. Thus an
image at HDF5 index ``i`` is the 1-FPS stream frame at ``i / 2`` seconds. An
assistant label is consumed only at the final image of its original user turn;
no label is fabricated for intermediate stream frames.

Run twice: ``--mode stats`` creates class activation sums for neuron selection;
then ``--mode features --neuron_map ...`` replays the same frozen stream and
saves only selected-neuron features.  The two passes avoid storing all MLP
activations for the full training set.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_ovobench_proactive import (  # noqa: E402
    Memory_Bank_Naive, build_prompt, extract_frame, text_kwargs,
)
from qwen_online.utils import encode_images, whole_current_frame_tokens  # noqa: E402
from response_graph.Scene_Graph_ovo import (  # noqa: E402
    Query_Graph_generation, Scene_Graph_generation_CRR,
)
from probe.runtime import Qwen3VLMLPCollector  # noqa: E402
from probe.multigpu import add_model_parallel_args, load_qwen3vl  # noqa: E402

IMAGE_TAG = re.compile(r"<image>")
H5_TIME = re.compile(r"_(?P<fps>\d+(?:\.\d+)?)fps\.hdf5:::(?P<index>\d+)$")
EGOEXO_JPG_INDEX = re.compile(r"/(?P<index>\d+)\.jpg$")


def h5_frame_info(item):
    """Return the source FPS and frame index encoded by an SFT image path."""
    match = H5_TIME.search(item["path"])
    if match is None:
        raise ValueError(f"Missing HDF5 FPS/index in {item!r}")
    fps = float(match.group("fps"))
    if fps <= 0:
        raise ValueError(f"Invalid HDF5 FPS in {item!r}")
    return fps, int(match.group("index"))


def h5_time_seconds(item):
    """Return the timestamp encoded by MMDuet2's HDF5 path itself."""
    fps, index = h5_frame_info(item)
    return index / fps


def images_follow_live_whisperx_grid(items, start_image_index):
    """Validate the released Live-WhisperX 2-FPS / SFT 1-FPS image grid."""
    if not items:
        return False
    for offset, item in enumerate(items):
        try:
            fps, index = h5_frame_info(item)
        except (KeyError, TypeError, ValueError):
            return False
        if fps != 2.0 or index != 2 * (start_image_index + offset):
            return False
    return True


def egoexolearn_time_seconds(item):
    """Return the MMDuet2 EgoExoLearn timestamp from ``%06d.jpg``.

    The official MMDuet2 preprocessing extracts each clip at 1 FPS using
    ffmpeg's ``%06d.jpg`` numbering.  Therefore 000001.jpg is the clip's
    first (0-second) visual observation, and image n represents n - 1 seconds.
    """
    match = EGOEXO_JPG_INDEX.search(item["path"])
    if match is None:
        raise ValueError(f"Missing EgoExoLearn JPG index in {item!r}")
    index = int(match.group("index"))
    if index <= 0:
        raise ValueError(f"Invalid EgoExoLearn JPG index in {item!r}")
    return float(index - 1)


def images_follow_egoexolearn_grid(items):
    """Check the monotonic 1-FPS JPG sequence released by MMDuet2."""
    if not items:
        return False
    previous = None
    for item in items:
        try:
            current = egoexolearn_time_seconds(item)
        except (KeyError, TypeError, ValueError):
            return False
        if previous is not None and current <= previous:
            return False
        previous = current
    return True


def turn_events(record):
    """Parse MMDuet2 turns into question episodes without relabeling them.

    A question episode starts at a non-empty user question and ends only when a
    later user turn introduces another question.  Assistant replies do not close
    an episode: MMDuet2's system prompt permits incremental replies for the same
    question.  An episode contributes to neuron selection only when it has both
    NO REPLY and reply turns and all message/image alignments are valid.  A
    text-only question is bound to the most recent valid video-frame timestamp.
    The feature pass keeps all alignment-valid turns.
    """
    image_cursor = 0
    last_visual_time = None
    active_episode = None
    pending = None
    events = []
    episodes = []
    next_episode_id = 0

    def invalidate(episode, reason):
        if episode is not None and episode["invalid_reason"] is None:
            episode["invalid_reason"] = reason

    for message_index, message in enumerate(record["messages"]):
        role, content = message["role"], message["content"]
        if role == "user":
            if pending is not None:
                invalidate(pending["episode"], "missing_assistant_after_user_turn")
                pending = None
            count = len(IMAGE_TAG.findall(content))
            question = IMAGE_TAG.sub("", content).strip()
            if count == 0:
                if question:
                    active_episode = {
                        "id": next_episode_id,
                        "question": question,
                        "events": [],
                        "invalid_reason": None if last_visual_time is not None
                        else "question_before_first_visual_frame",
                    }
                    next_episode_id += 1
                    episodes.append(active_episode)
                    if last_visual_time is not None:
                        # No new visual observation arrived with this question:
                        # use the already available visual prefix unchanged.
                        pending = {
                            "time": last_visual_time,
                            "episode": active_episode,
                            "user_message_index": message_index,
                        }
                continue
            image_start = image_cursor
            used = record["images"][image_start:image_start + count]
            image_cursor += count
            if question:
                active_episode = {
                    "id": next_episode_id,
                    "question": question,
                    "events": [],
                    "invalid_reason": None,
                }
                next_episode_id += 1
                episodes.append(active_episode)
            if len(used) != count:
                invalidate(active_episode, "message_image_count_mismatch")
                continue
            if all(H5_TIME.search(item["path"]) is not None for item in used):
                grid_valid = images_follow_live_whisperx_grid(used, image_start)
                try:
                    end_time = h5_time_seconds(used[-1])
                except (KeyError, TypeError, ValueError):
                    invalidate(active_episode, "unparseable_hdf5_timestamp")
                    continue
                grid_error = "hdf5_image_time_mismatch"
            elif all(EGOEXO_JPG_INDEX.search(item["path"]) is not None for item in used):
                grid_valid = images_follow_egoexolearn_grid(used)
                try:
                    end_time = egoexolearn_time_seconds(used[-1])
                except (KeyError, TypeError, ValueError):
                    invalidate(active_episode, "unparseable_egoexolearn_timestamp")
                    continue
                grid_error = "egoexolearn_image_time_mismatch"
            else:
                invalidate(active_episode, "mixed_or_unparseable_image_timestamp")
                continue
            if grid_valid:
                last_visual_time = end_time
            if active_episode is None:
                continue
            if not grid_valid:
                invalidate(active_episode, grid_error)
            pending = {
                "time": end_time,
                "episode": active_episode,
                "user_message_index": message_index,
            }
        elif role == "assistant" and pending is not None:
            episode = pending["episode"]
            event = {
                "time": pending["time"],
                "question": episode["question"],
                "label": 0 if content.strip() == "NO REPLY" else 1,
                "assistant": content,
                "episode_id": episode["id"],
                "user_message_index": pending["user_message_index"],
                "assistant_message_index": message_index,
            }
            pending = None
            episode["events"].append(event)
            events.append(event)

    if pending is not None:
        invalidate(pending["episode"], "missing_assistant_at_end_of_record")
    if image_cursor != len(record["images"]):
        for episode in episodes:
            invalidate(episode, "unused_or_unaligned_annotation_images")

    paired_episodes = []
    skipped = defaultdict(int)
    for episode in episodes:
        labels = [event["label"] for event in episode["events"]]
        if episode["invalid_reason"] is not None:
            skipped[episode["invalid_reason"]] += 1
        elif 1 not in labels:
            skipped["no_reply_only"] += 1
        elif 0 not in labels:
            skipped["no_prior_no_reply"] += 1
        else:
            paired_episodes.append(episode)

    paired_ids = {episode["id"] for episode in paired_episodes}
    valid_ids = {episode["id"] for episode in episodes
                 if episode["invalid_reason"] is None}
    for event in events:
        event["alignment_valid"] = event["episode_id"] in valid_ids
        event["use_for_neuron_stats"] = event["episode_id"] in paired_ids
    return events, paired_episodes, dict(skipped)


EGOEXO_SEGMENT = re.compile(
    r"^(?P<uid>.+)-(?P<start>\d+(?:\.\d+)?)s_(?P<end>\d+(?:\.\d+)?)s$"
)


def _first_existing(candidates):
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def resolve_video_source(record, args):
    """Resolve one SFT record to (MP4 path, source start time, dataset name).

    Live-WhisperX records use their video ID directly. EgoExoLearn records use
    ``<source_uid>-<start>s_<end>s`` while the downloaded MP4 is named only
    ``<source_uid>.mp4``.  Their annotation timestamps remain clip-relative;
    the returned offset is applied only when decoding source video frames.
    """
    metadata = record["metadata"]
    video_id = metadata["video_id"]
    declared_source = metadata.get("dataset") or metadata.get("data_source")
    ego_match = EGOEXO_SEGMENT.match(video_id)
    is_ego = declared_source == "egoexolearn" or (
        declared_source is None and args.egoexolearn_video_root is not None and ego_match is not None
    )
    if is_ego:
        if args.egoexolearn_video_root is None:
            raise ValueError(
                "EgoExoLearn record encountered but --egoexolearn_video_root was not supplied"
            )
        if ego_match is None:
            raise ValueError(f"Invalid EgoExoLearn clip video_id: {video_id}")
        uid = ego_match.group("uid")
        start = float(ego_match.group("start"))
        root = Path(args.egoexolearn_video_root)
        path = _first_existing((root / f"{uid}.mp4", root / "videos" / f"{uid}.mp4"))
        return path, start, "egoexolearn"

    if args.video_root is None:
        raise ValueError(
            "Encountered a Live-WhisperX record but --video_root was not supplied."
        )
    root = Path(args.video_root)
    path = _first_existing((root / "videos" / f"{video_id}.mp4", root / f"{video_id}.mp4"))
    return path, 0.0, "live_whisperx"


def build_decision_embeddings(model, processor, memories, question, query_graph, visual_context_frames=None):
    context_graphs = memories.get_graphs_top_new(model, processor, query_graph)
    annotation = {"question": question, "answer": ""}
    trigger_prompt = build_prompt("CRR", None, None, annotation, 0, context_graphs)
    response_ids = processor.tokenizer(
        ["<|im_start|>user\n" + trigger_prompt + "<|im_end|>\n"], **text_kwargs)
    response_ids = torch.tensor(response_ids["input_ids"], device=model.device)
    response_embeddings = model.get_input_embeddings()(response_ids)
    raw_visual_window = 600 if visual_context_frames is None else visual_context_frames
    frame_embeddings = torch.cat(memories.get_context_frames(context_window=raw_visual_window), dim=1).to(model.device)
    assistant_ids = processor.tokenizer(["<|im_start|>assistant\n"], **text_kwargs)
    assistant_ids = torch.tensor(assistant_ids["input_ids"], device=model.device)
    assistant_embeddings = model.get_input_embeddings()(assistant_ids)
    input_embeddings = torch.cat(
        [frame_embeddings, response_embeddings, assistant_embeddings], dim=1
    ).to(model.device)
    attention_mask = torch.ones(input_embeddings.shape[:2], dtype=torch.long, device=model.device)
    return input_embeddings, attention_mask


def process_record(record, args, model, processor, collector, neuron_indices, stats,
                   features, labels, metadata, decision_cache=None):
    video_id = record["metadata"]["video_id"]
    video_path, source_start_time, dataset_name = resolve_video_source(record, args)
    if not video_path.exists():
        print(f"[skip missing] dataset={dataset_name} path={video_path}")
        return 0
    events, paired_episodes, skipped_episodes = turn_events(record)
    if args.mode == "stats":
        for reason, count in skipped_episodes.items():
            stats["skipped_episodes"][reason] = stats["skipped_episodes"].get(reason, 0) + count
        # Cap valid question-level reply-vs-NO-REPLY pairs, not JSONL rows or turns.
        if args.max_question_pairs is not None:
            remaining = args.max_question_pairs - stats["candidate_pairs_selected"]
            if remaining <= 0:
                return 0
            paired_episodes = paired_episodes[:remaining]
        selected_ids = {episode["id"] for episode in paired_episodes}
        stats["candidate_pairs_selected"] += len(paired_episodes)
        events = [event for event in events if event["episode_id"] in selected_ids]
    else:
        # Feature/probe training keeps all original labels whose image/message
        # timestamp is valid, but excludes malformed question episodes.
        events = [event for event in events if event["alignment_valid"]]
    if not events:
        return 0
    events_by_frame = defaultdict(list)
    for event in events:
        event["frame"] = int(round(event["time"] * args.fps))
        if abs(event["time"] * args.fps - event["frame"]) > 1e-6:
            raise ValueError(f"MMDuet2 event {event['time']} is off the configured {args.fps} FPS grid")
        events_by_frame[event["frame"]].append(event)
    start_frame, end_frame = 0, max(events_by_frame)
    ask_frame = min(events_by_frame)
    total_frames = end_frame - ask_frame + 1
    # Same adaptive scene-graph interval as Response-G1's OVO evaluator.
    sg_interval = 2 if total_frames <= 50 else (total_frames - 1) // 50 + 2
    memories = Memory_Bank_Naive(memory_size=end_frame - start_frame + 1,
                                  visual_memory_size=args.visual_context_frames)
    active_question, query_graph = None, None
    written = 0
    episode_activations = {
        episode["id"]: {"reply": [], "silence": []}
        for episode in paired_episodes
    }

    for frame_idx in range(start_frame, end_frame + 1, args.frame_interval):
        timestamp = frame_idx / args.fps
        decode_timestamp = source_start_time + timestamp
        frame = extract_frame(str(video_path), decode_timestamp)
        if frame is None:
            frame = extract_frame(str(video_path), max(source_start_time, decode_timestamp - 1))
        if frame is None:
            continue
        memories.reserve_next_frame()
        frame_inputs = processor.image_processor(frame)
        frame_inputs["pixel_values"] = frame_inputs["pixel_values"].to(model.device)
        frame_inputs["image_grid_thw"] = frame_inputs["image_grid_thw"].to(model.device)
        frame_tokens, _, _ = encode_images(
            model, frame_inputs["pixel_values"], frame_inputs["image_grid_thw"]
        )
        frame_tokens = whole_current_frame_tokens(processor, model, timestamp, frame_tokens)
        del frame_inputs
        if frame_tokens is None:
            continue
        memories.update(frame_tokens.unsqueeze(0).to(model.device))

        # Process same-timestamp turns in their original dialogue order.  A
        # text-only question is attached to the preceding frame and can therefore
        # share a timestamp with an earlier image-conditioned decision.
        if frame_idx in events_by_frame:
            first = events_by_frame[frame_idx][0]
            if first["question"] != active_question:
                active_question = first["question"]
                query_graph = Query_Graph_generation(processor, model, active_question)

        relative = frame_idx - start_frame
        should_sg = relative == 0 or (sg_interval > 0 and relative % sg_interval == 0)
        scene_graph_question = None
        if should_sg and active_question is not None:
            clip = torch.cat(memories.get_context_frames(sg_interval), dim=1).to(model.device)
            graph = Scene_Graph_generation_CRR(processor, model, clip, active_question, query_graph)
            if len(graph.get("scene_graph", [])):
                memories.update_graph(timestamp, graph)
            scene_graph_question = active_question

        decision_ran = False
        for event in events_by_frame.get(frame_idx, []):
            if event["question"] != active_question:
                active_question = event["question"]
                query_graph = Query_Graph_generation(processor, model, active_question)
            if should_sg and scene_graph_question != active_question:
                clip = torch.cat(memories.get_context_frames(sg_interval), dim=1).to(model.device)
                graph = Scene_Graph_generation_CRR(processor, model, clip, active_question, query_graph)
                if len(graph.get("scene_graph", [])):
                    memories.update_graph(timestamp, graph)
                scene_graph_question = active_question
            inputs, mask = build_decision_embeddings(
                model, processor, memories, active_question, query_graph,
                visual_context_frames=args.visual_context_frames,
            )
            activations = collector.collect(model, inputs, mask)
            del inputs, mask
            label = event["label"]
            event_metadata = {"video_id": video_id, "time": event["time"],
                              "dataset": dataset_name, "source_start_time": source_start_time,
                              "question": active_question, "label": label,
                              "episode_id": event["episode_id"],
                              "question_id": record["metadata"].get("question_id"),
                              "assistant": event["assistant"]}
            if args.mode == "stats":
                if event["use_for_neuron_stats"]:
                    key = "reply" if label else "silence"
                    episode_activations[event["episode_id"]][key].append(activations)
                    if decision_cache is not None:
                        decision_cache["activations"].append(activations.to(torch.float16))
                        decision_cache["labels"].append(label)
                        decision_cache["metadata"].append(event_metadata)
            elif event["alignment_valid"]:
                selected = torch.stack(
                    [activations[layer, ids] for layer, ids in enumerate(neuron_indices)]
                ).reshape(-1)
                features.append(selected)
                labels.append(label)
                metadata.append(event_metadata)
            written += 1
            decision_ran = True
            del activations
        # Response-G1 clears transient generation tensors after a decision, not
        # after every incoming frame. Do the same for probe-forward tensors.
        if decision_ran:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    if args.mode == "stats":
        for episode in paired_episodes:
            collected = episode_activations[episode["id"]]
            if not collected["reply"] or not collected["silence"]:
                stats["skipped_episodes"]["unreadable_event_frame"] = (
                    stats["skipped_episodes"].get("unreadable_event_frame", 0) + 1
                )
                continue
            reply_mean = torch.stack(collected["reply"]).mean(dim=0)
            silence_mean = torch.stack(collected["silence"]).mean(dim=0)
            stats["episode_delta_sum"].add_(reply_mean - silence_mean)
            stats["paired_question_count"] += 1
    return written


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="MMDuet2 Live SFT JSONL or its downloaded subset")
    parser.add_argument("--video_root", default=None,
                        help="Optional Live-WhisperX root; required only when annotations contain Live records.")
    parser.add_argument("--egoexolearn_video_root", default=None,
                        help="Optional EgoExoLearn root containing <source_uid>.mp4 files.")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", required=True, choices=["stats", "features"])
    parser.add_argument("--neuron_map", default=None, help="Required in features mode")
    parser.add_argument("--max_records", type=int, default=None,
                        help="Optional JSONL-record cap applied before pair filtering.")
    parser.add_argument("--max_question_pairs", type=int, default=None,
                        help="Stats mode: use at most this many valid question-level reply-vs-NO-REPLY pairs.")
    parser.add_argument("--decision_cache", default=None,
                        help="Stats mode: save FP16 MLP activations + per-decision metadata here. Features mode: reuse it and skip model/video replay.")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--min_pixels", type=int, default=448 * 448)
    parser.add_argument("--max_pixels", type=int, default=448 * 448)
    parser.add_argument("--visual_context_frames", type=int, default=None,
                        help="Keep only the most recent K raw visual frames for each LLM decision; omit to preserve the original full/600-frame prefix. Older history remains available through Response-G1 scene-graph retrieval text.")
    add_model_parallel_args(parser)
    args = parser.parse_args()
    if args.fps != 1 or args.frame_interval != 1:
        parser.error("This MMDuet2 Live-WhisperX protocol is fixed to 1 FPS and frame_interval=1.")
    if args.visual_context_frames is not None and args.visual_context_frames < 1:
        parser.error("--visual_context_frames must be a positive integer when provided")
    if args.mode == "features" and not args.neuron_map:
        parser.error("--neuron_map is required in features mode")

    if args.max_question_pairs is not None and args.max_question_pairs < 1:
        parser.error("--max_question_pairs must be a positive integer")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "features" and args.decision_cache:
        cache = torch.load(args.decision_cache, map_location="cpu", weights_only=True)
        neuron_map = torch.load(args.neuron_map, map_location="cpu", weights_only=True)
        if list(cache["layer_names"]) != list(neuron_map["layer_names"]):
            raise RuntimeError("Decision-cache layer order does not match neuron_map.")
        indices = neuron_map["indices"].long()
        activations = cache["activations"].float()
        selected = torch.cat([activations[:, layer, ids] for layer, ids in enumerate(indices)], dim=1)
        torch.save({"features": selected, "labels": cache["labels"].long(),
                    "metadata": cache["metadata"], "neuron_map": args.neuron_map,
                    "decision_cache": args.decision_cache}, args.output)
        print(f"Saved selected features for {selected.shape[0]} cached decisions to {args.output}")
        return

    model = load_qwen3vl(args.ckpt_path, args)
    processor = AutoProcessor.from_pretrained(args.ckpt_path, min_pixels=args.min_pixels,
                                               max_pixels=args.max_pixels, trust_remote_code=True)
    collector = Qwen3VLMLPCollector(model)
    neuron_indices = None
    if args.mode == "features":
        neuron_map = torch.load(args.neuron_map, map_location="cpu", weights_only=True)
        neuron_indices = neuron_map["indices"].long()
        if list(neuron_map["layer_names"]) != collector.layer_names:
            raise RuntimeError("Neuron-map layer order does not match the loaded Qwen3-VL model.")

    width = collector.down_projection_norms().shape[1]
    stats = {"episode_delta_sum": torch.zeros(len(collector.modules), width),
             "paired_question_count": 0,
             "candidate_pairs_selected": 0,
             "skipped_episodes": {},
             "down_projection_norms": collector.down_projection_norms(),
             "layer_names": collector.layer_names}
    decision_cache = ({"activations": [], "labels": [], "metadata": [],
                       "layer_names": collector.layer_names}
                      if args.mode == "stats" and args.decision_cache else None)
    features, labels, metadata = [], [], []
    total = 0
    with open(args.annotations, encoding="utf-8") as f:
        for record_index, line in enumerate(f):
            if args.max_records is not None and record_index >= args.max_records:
                break
            total += process_record(json.loads(line), args, model, processor, collector,
                                    neuron_indices, stats, features, labels, metadata, decision_cache)
            if (record_index + 1) % 10 == 0:
                print(f"processed records={record_index + 1}, decisions={total}")
    if args.mode == "stats":
        torch.save(stats, args.output)
        print(f"Saved statistics from {stats['paired_question_count']} paired questions "
              f"({total} processed decisions) to {args.output}")
        print(f"Candidate pairs selected by cap: {stats['candidate_pairs_selected']}")
        if decision_cache is not None:
            payload = {"activations": torch.stack(decision_cache["activations"]) if decision_cache["activations"] else torch.empty(0, len(collector.modules), width, dtype=torch.float16),
                       "labels": torch.tensor(decision_cache["labels"], dtype=torch.long),
                       "metadata": decision_cache["metadata"],
                       "layer_names": decision_cache["layer_names"]}
            torch.save(payload, args.decision_cache)
            with open(str(args.decision_cache) + ".jsonl", "w", encoding="utf-8") as f:
                for item in decision_cache["metadata"]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Saved FP16 decision cache and metadata JSONL to {args.decision_cache}")
        print(f"Skipped question episodes: {stats['skipped_episodes']}")
    else:
        torch.save({"features": torch.stack(features) if features else torch.empty(0, neuron_indices.numel()),
                    "labels": torch.tensor(labels, dtype=torch.long), "metadata": metadata,
                    "neuron_map": args.neuron_map}, args.output)
        print(f"Saved selected features for {total} decisions to {args.output}")
    collector.close()

if __name__ == "__main__":
    main()
