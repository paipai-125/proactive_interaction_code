from transformers import AutoProcessor
import torch
import json
import os
import os.path as osp
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import re
import logging
import time
from collections import defaultdict
import argparse
import ffmpeg
from collections import deque
import sys
sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '..')))

import torch.nn.functional as F

from qwen_online.utils import encode_images, whole_current_frame_tokens
from response_graph.Scene_Graph_ovo import Query_Graph_generation, Scene_Graph_generation_CRR, format_scene_graphs_for_prompt_offline
from probe.runtime import Qwen3VLMLPCollector, load_frozen_probe
from probe.build_probe_dataset_dp import distributed_context, load_single_gpu_model

# Do not overwrite CUDA_VISIBLE_DEVICES: the launcher controls all visible GPUs.
# NPROC_PER_NODE is intentionally not used; this is one model-parallel process.

# Path and hyperparameter defaults are defined in build_parser() / resolved on args in main().

forward_tasks = ["CRR"]

text_kwargs = {
    "padding": True,
    "return_token_type_ids": False
}

generate_trigger_kwargs = {
    "do_sample": False,
    "num_beams": 1, 
    "min_length": 1,
    "num_return_sequences": 1,
    "max_new_tokens": 128,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt_str = "%(asctime)s %(levelname)7s | %(message)s"
fmt = logging.Formatter(fmt_str)


class Memory_Bank_Naive:
    def __init__(self, memory_size, visual_memory_size=None):
        self.memory_size = memory_size
        # Keep K GPU-resident raw frames; retain the full scene-graph history.
        self.visual_memory_size = memory_size if visual_memory_size is None else visual_memory_size
        self.memory_bank = deque(maxlen=self.visual_memory_size) # Memory bank size (frame number)
        self.Scene_Graph_memory_bank = deque(maxlen=memory_size) # Scene graph memory bank size (frame number)

    def reserve_next_frame(self):
        # Evict before encoding so K+1 raw tensors never coexist.
        if len(self.memory_bank) >= self.visual_memory_size:
            self.memory_bank.popleft()

    def update(self, current_frame):
        self.memory_bank.append(current_frame) # Automatically FIFO

    def update_graph(self, timestamp, current_Scene_Graph):
        self.Scene_Graph_memory_bank.append((timestamp, current_Scene_Graph)) # Automatically FIFO

    def get_context_frames(self, context_window):
        context_window = int(context_window)
        start_index = max(0, len(self.memory_bank) - context_window)
        recent_frames = list(self.memory_bank)[start_index:]

        return recent_frames

    def get_graphs_top_new(self, model, processor, Query_Graph):
        # Retrieve Context_Scene_Graphs based on the Query_Graph
        if len(self.Scene_Graph_memory_bank) == 0:
            return None

        if isinstance(Query_Graph, str):
            query_graph_text = Query_Graph
        else:
            query_graph_text = format_scene_graphs_for_prompt_offline([Query_Graph])
        query_graph_ids = processor.tokenizer(query_graph_text, **text_kwargs)
        query_graph_ids_tensor = torch.tensor(query_graph_ids['input_ids']).to(model.device)
        query_graph_embeddings = model.get_input_embeddings()(query_graph_ids_tensor)
        # query_graph_embeddings: [n1, 4096], n1 is the sequence length
        query_graph_embeddings = query_graph_embeddings.mean(dim=0)  # [4096]

        # Similarity Calculation of each Context_Scene_Graph with the Query_Graph
        context_Scene_Graphs_list = list(self.Scene_Graph_memory_bank)
        similarities = []
        for timestamp, Scene_Graph in context_Scene_Graphs_list:
            context_Scene_Graph_text = format_scene_graphs_for_prompt_offline([Scene_Graph])
            context_Scene_Graph_ids = processor.tokenizer(context_Scene_Graph_text, **text_kwargs)
            context_Scene_Graph_ids_tensor = torch.tensor(context_Scene_Graph_ids['input_ids']).to(model.device)
            context_Scene_Graph_embeddings = model.get_input_embeddings()(context_Scene_Graph_ids_tensor)
            # context_Scene_Graph_embeddings: [n2, 4096], n2 is the sequence length
            context_Scene_Graph_embeddings = context_Scene_Graph_embeddings.mean(dim=0)  # [4096]

            # Calculate cosine similarity: cosine similarity of two [4096] vectors
            similarity = self.cosine_similarity(context_Scene_Graph_embeddings, query_graph_embeddings)
            similarities.append(similarity.item())

        # top-k similar Scene Graphs by similarity
        if len(similarities) == 0:
            return None
        k = min(3, len(similarities))  # Ensure k is not greater than the list length
        similarities_tensor = torch.tensor(similarities)
        top_k_similarities = torch.topk(similarities_tensor, k=k)
        top_k_indices = top_k_similarities.indices.tolist()
        top_k_Scene_Graphs = [context_Scene_Graphs_list[i] for i in top_k_indices]
        newest_indices = [len(context_Scene_Graphs_list) - 1]
        newest_Scene_Graphs = [context_Scene_Graphs_list[i] for i in newest_indices]
        if newest_indices[0] not in top_k_indices:
            top_k_Scene_Graphs.extend(newest_Scene_Graphs)

        return top_k_Scene_Graphs

    def cosine_similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity of two embedding vectors
        Args:
            embedding1: [4096] or [1, 4096] vector
            embedding2: [4096] or [1, 4096] vector
        Returns:
            similarity: scalar or [1] tensor
        """
        # Ensure two vectors are 1D or 2D (batch_size=1)
        if embedding1.dim() == 1:
            embedding1 = embedding1.unsqueeze(0)  # [1, 4096]
        if embedding2.dim() == 1:
            embedding2 = embedding2.unsqueeze(0)  # [1, 4096]
        
        # Calculate cosine similarity, dim=1 means calculate on the feature dimension
        similarity = F.cosine_similarity(embedding1, embedding2, dim=1)  # [1]
        
        return similarity


def build_prompt(task, question, options, _anno_, index, context_Scene_Graphs=None):
    question = _anno_["question"]
    answer = _anno_["answer"]
    if context_Scene_Graphs is not None:
        context_Scene_Graphs_text = "The relative scene graph shows:\n"
        for timestamp, Scene_Graph in context_Scene_Graphs:
            frame_second_placeholder = f"<{timestamp:.1f} seconds>"
            context_Scene_Graph_text = format_scene_graphs_for_prompt_offline([Scene_Graph])
            context_Scene_Graphs_text += f"{frame_second_placeholder}:{context_Scene_Graph_text}\n"
        prompt = f"""
            You're responsible of answering questions based on the video content. 

            {context_Scene_Graphs_text}

            The following question are relevant to the latest frames, i.e. the end of the video.
            {question}
            Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """
    else:
        prompt = f"""
            You're responsible of answering questions based on the video content. 
            The following question are relevant to the latest frames, i.e. the end of the video.
            {question}
            Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
            Answer only with "Yes" or "No".
            Do not include any additional text or explanation in your response.
        """
    return prompt


def eval_r(args):
    # Load model and processor (Load model-official model)
    torch.manual_seed(1234)
    logger.info(f"Set manual seed to 1234")
    model = load_single_gpu_model(args.ckpt_path, args.local_rank)
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    logger.info(f"Load model and processor from {args.ckpt_path}")
    model.eval()
    # Optional frozen external probe.  Without these arguments this copied
    # evaluator remains byte-for-byte equivalent in its trigger behavior.
    probe = None
    collector = None
    if args.probe_path:
        if not args.neuron_map:
            raise ValueError("--neuron_map is required with --probe_path")
        if args.probe_threshold is None:
            raise ValueError("--probe_threshold from select_probe_threshold.py is required with --probe_path")
        collector = Qwen3VLMLPCollector(model)
        probe = load_frozen_probe(args.probe_path, args.neuron_map)
        logger.info("Loaded frozen response probe: %s", args.probe_path)

    # Load dataset
    with open(args.task_file, "r", encoding="utf-8") as f:
        task_list = json.load(f)
    task_list = [task for task in task_list if task.get('task') in ["CRR"]]
    task_list = [item for task_index, item in enumerate(task_list)
                 if task_index % args.world_size == args.rank]

    # Test: each rank processes a disjoint subset of CRR samples.
    with torch.no_grad():
        for item in tqdm(task_list, desc=f"rank {args.rank} CRR", position=args.rank,
                         dynamic_ncols=True, unit="video"):
            try:
                # Get task
                id, video, task, test_info, ask_time, clue_time = \
                    item['id'], item['video'], item['task'], item['test_info'], item['ask_time'], item['clue_time']

                video_path = osp.join(args.video_dir, video)
                # Get video
                # Use defaultdict(list) to handle repeated realtime values
                realtime_index_map = defaultdict(list)
                for i in range(len(test_info)):
                    realtime_index_map[test_info[i]['realtime']].append(i)
                start_frame = 0
                ask_frame = ask_time / args.fps
                clue_frame = clue_time / args.fps
                end_frame = max(int(i) for i in realtime_index_map.keys()) * args.fps
                # Build Memory Bank
                memories = Memory_Bank_Naive(memory_size=end_frame - start_frame + 1,
                                            visual_memory_size=args.visual_context_frames)

                question = item["question"]
                Query_Graph = Query_Graph_generation(processor, model, question)

                sg_force_first_frame = args.sg_force_first_frame

                # Calculate sg_generation_interval based on end_frame - start_frame
                # Within 50 frames: generate every two frames (interval = 2), every 50 frames or more: generate every frame + 1
                total_frames = end_frame - ask_frame + 1
                if total_frames <= 50:
                    sg_generation_interval = 2  # Generate every two frames within 50 frames
                else:
                    sg_generation_interval = (total_frames - 1) // 50 + 2  # Generate every frame + 1 every 50 frames or more

                # Streaming Pipeline
                for frame_idx in range(start_frame, end_frame + 1):
                    timestamp = frame_idx / args.fps

                    current_frame = extract_frame(video_path, timestamp)
                    if current_frame == None:
                        current_frame = extract_frame(video_path, timestamp - 1)
                        # continue
                    memories.reserve_next_frame()
                    current_frame_embeddings = processor.image_processor(current_frame)
                    current_frame_embeddings["pixel_values"] = current_frame_embeddings["pixel_values"].to(model.device)
                    current_frame_embeddings['image_grid_thw'] = current_frame_embeddings['image_grid_thw'].to(model.device)
                    current_frame_tokens, _, token_per_frame = encode_images(model, current_frame_embeddings["pixel_values"], current_frame_embeddings['image_grid_thw'])
                    current_frame_tokens = whole_current_frame_tokens(processor, model, timestamp, current_frame_tokens)
                    current_frame_tokens = current_frame_tokens.unsqueeze(0).to(model.device)
                    # Release current frame embeddings memory
                    del current_frame_embeddings
                    if current_frame_tokens is None:
                        continue
                    else:
                        memories.update(current_frame_tokens)

                    # * Online Scene Graph Generation (Optimized: interval generation + cache reuse) *#
                    relative_frame_idx = frame_idx - start_frame
                    should_generate_sg = (
                        (sg_force_first_frame and relative_frame_idx == 0) or  # Force generate the first frame
                        (sg_generation_interval > 0 and relative_frame_idx % sg_generation_interval == 0 and relative_frame_idx != 0) or  # Interval generation
                        (sg_generation_interval <= 0)  # If interval <= 0, generate every frame (keep original behavior)
                    )
                    if should_generate_sg and frame_idx >= ask_frame:  # Gate scene graph after ask time
                        current_clip_tokens = memories.get_context_frames(context_window=sg_generation_interval if sg_generation_interval > 0 else 1)
                        current_clip_tokens = torch.cat(current_clip_tokens, dim=1).to(model.device)
                        current_Scene_Graph = Scene_Graph_generation_CRR(processor, model, current_clip_tokens, question, Query_Graph)
                        if len(current_Scene_Graph.get('scene_graph', [])) != 0:
                            memories.update_graph(timestamp, current_Scene_Graph)

                    frame_realtime = frame_idx / args.fps
                    if frame_realtime in realtime_index_map:
                        for test_idx in realtime_index_map[frame_realtime]:
                            # --- Retrieve graphs for prompt context ---
                            context_Scene_Graphs = memories.get_graphs_top_new(model, processor, Query_Graph)
                            # context_Scene_Graphs = None

                            # --- Build user / assistant prompt embeddings ---
                            prompt_response = build_prompt(
                                task=task,
                                question=None,
                                options=None,
                                _anno_=item,
                                index=realtime_index_map[frame_idx],
                                context_Scene_Graphs=context_Scene_Graphs,
                            )

                            response_ids = processor.tokenizer(
                                ["<|im_start|>user\n" + prompt_response +"<|im_end|>\n"], **text_kwargs)
                            response_ids_tensor = torch.tensor(response_ids['input_ids']).to(model.device)
                            response_prompt_embeddings = model.get_input_embeddings()(response_ids_tensor)

                            # Raw visuals are capped only when explicitly requested;
                            # scene-graph retrieval above still spans all saved history.
                            raw_visual_window = 600 if args.visual_context_frames is None else args.visual_context_frames
                            past_frame_tokens = memories.get_context_frames(context_window=raw_visual_window)
                            past_frame_tokens = torch.cat(past_frame_tokens, dim=1).to(model.device)

                            input_embeddings = torch.cat([past_frame_tokens, response_prompt_embeddings], dim=1).to(model.device)

                            generation_ids = processor.tokenizer(["<|im_start|>assistant\n"], **text_kwargs)
                            generation_ids_tensor = torch.tensor(generation_ids['input_ids']).to(model.device)
                            generation_prompt_embeddings = model.get_input_embeddings()(generation_ids_tensor)
                            input_embeddings = torch.cat([input_embeddings, generation_prompt_embeddings], dim=1).to(model.device)

                            attention_masks = torch.ones(input_embeddings.shape[:2], dtype=torch.long).to(model.device)

                            if probe is None:
                                generated_ids = model.generate(
                                    inputs_embeds=input_embeddings,
                                    attention_mask=attention_masks,
                                    **generate_trigger_kwargs,
                                )
                                output_text = processor.batch_decode(
                                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                )
                                response = output_text[0]
                            else:
                                activations = collector.collect(model, input_embeddings, attention_masks)
                                reply_probability = probe.probability(activations)
                                response = "Yes" if reply_probability >= args.probe_threshold else "No"
                                item['test_info'][test_idx]['probe_reply_probability'] = reply_probability
                                item['test_info'][test_idx]['probe_threshold'] = args.probe_threshold
                                del activations

                            item['test_info'][test_idx]['response'] = response

                            # Release CUDA memory
                            del input_embeddings, attention_masks, past_frame_tokens
                            if probe is None:
                                del generated_ids
                            del response_prompt_embeddings, generation_prompt_embeddings
                            del response_ids_tensor, generation_ids_tensor
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                    output_dict = item

                with open(args.output_jsonl, 'a' if osp.exists(args.output_jsonl) else 'w') as f:
                    f.write(json.dumps(output_dict) + '\n')
            except Exception as e:
                logger.error(f"Error in processing {item}: {e}")
                print(f"Error in processing {item}: {e}")


# helper functions
def time_to_seconds(time_str):
    """Convert time string to seconds (compatible with HH:MM:SS and MM:SS)"""
    if len(time_str) == 5:
        time_obj = datetime.strptime(time_str, '%M:%S')
    else:
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return total_seconds

def seconds_to_time(total_seconds):
    """Convert seconds to time string (fixed HH:MM:SS format)"""
    total_seconds = int(total_seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def extract_frame(video_path, timestamp):
    """Extract frame from video at specified timestamp"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        video_fps = cap.get(cv2.CAP_PROP_FPS) # Note: here must use the video's frame (not the model's frame args.fps)
        frame_number = int(timestamp * video_fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # Convert to RGB PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # import numpy as np
        # image_array = np.array(pil_image)
        # print(f"Array shape: {image_array.shape}")

        return pil_image

    except Exception as e:
        print(f"❌ Extract frame failed: {e}")
        return None


def build_parser():
    parser = argparse.ArgumentParser(description="OVO-Bench proactive (CRR-style) evaluation")
    parser.add_argument("--run_name", type=str, default="proactive_output_0d5")
    parser.add_argument("--drop_method", type=str, default="feature")
    parser.add_argument("--drop_threshold", type=float, default=0.5)
    parser.add_argument("--drop_relative", action="store_true", help="If set, use relative drop mode (default: absolute)")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Local folder with Qwen3-VL weights")
    parser.add_argument(
        "--task_json",
        "--task_csv",
        dest="task_file",
        type=str,
        required=True,
        help="Path to ovo_bench_new.json (either flag name is accepted)",
    )
    parser.add_argument("--video_dir", type=str, required=True, help="Folder containing benchmark videos (e.g. src_videos)")
    parser.add_argument("--result_dir", type=str, required=True, help="Directory for logs, output/, drop/")
    parser.add_argument("--min_pixels", type=int, default=448 * 448)
    parser.add_argument("--max_pixels", type=int, default=448 * 448)
    parser.add_argument("--visual_context_frames", type=int, default=None,
                        help="Keep only the most recent K raw visual frames for each LLM decision; omit to preserve the original 600-frame prefix. Scene-graph retrieval text remains global.")
    parser.add_argument("--min_frames", type=int, default=4)
    parser.add_argument("--max_frames", type=int, default=720)
    parser.add_argument("--tolerance_time", type=int, default=2)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--probe_path", type=str, default=None,
                        help="Frozen probe from probe/train_probe.py; omit for the unmodified Response-G1 trigger.")
    parser.add_argument("--neuron_map", type=str, default=None,
                        help="Neuron map from probe/select_neurons.py.")
    parser.add_argument("--probe_threshold", type=float, default=None,
                        help="Reply threshold selected by select_probe_threshold.py; required with --probe_path.")
    parser.add_argument(
        "--sg_generation_interval",
        type=int,
        default=-1,
        help="Reserved for future use; CRR loop uses adaptive interval from clip length",
    )
    parser.add_argument(
        "--sg_force_first_frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force scene graph on first frame of stream (use --no-sg_force_first_frame to disable)",
    )
    return parser


def merge_rank_outputs(final_output_jsonl, shard_paths):
    """Merge one JSONL shard per data-parallel rank deterministically by OVO id."""
    records = []
    for shard_path in shard_paths:
        if not osp.exists(shard_path):
            raise FileNotFoundError(f"Missing rank output shard: {shard_path}")
        with open(shard_path, encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(key=lambda item: item.get("id", -1))
    with open(final_output_jsonl, "w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[data-parallel] merged {len(records)} CRR results into {final_output_jsonl}", flush=True)


def main():
    args = build_parser().parse_args()
    if args.visual_context_frames is not None and args.visual_context_frames < 1:
        raise ValueError("--visual_context_frames must be a positive integer when provided")

    rank, local_rank, world_size = distributed_context()
    # Keep terminal output readable under torchrun: rank 0 owns the visible
    # progress bar, while nonzero ranks keep writing their result shards/logs.
    if rank != 0:
        _rank_output_sink = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = _rank_output_sink
        sys.stderr = _rank_output_sink
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    if world_size > 1:
        import torch.distributed as dist
        timestamp_holder = [datetime.now().strftime("%Y%m%d_%H%M%S") if rank == 0 else None]
        dist.broadcast_object_list(timestamp_holder, src=0)
        curr_time = timestamp_holder[0]
    else:
        curr_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.curr_time = curr_time
    final_output_jsonl = osp.join(args.result_dir, "output", f"{args.run_name}_{curr_time}.jsonl")
    output_stem, output_ext = osp.splitext(final_output_jsonl)
    args.output_jsonl = f"{output_stem}.rank{rank}{output_ext}"
    args.final_output_jsonl = final_output_jsonl
    args.log_path = osp.join(args.result_dir, "log", f"{args.run_name}_{curr_time}.rank{rank}.log")
    args.dr_save_path = osp.join(args.result_dir, "drop", f"{args.run_name}_{curr_time}.rank{rank}.json")
    args.drop_absolute = not args.drop_relative

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(osp.join(args.result_dir, "output"), exist_ok=True)
    os.makedirs(osp.join(args.result_dir, "drop"), exist_ok=True)
    os.makedirs(osp.join(args.result_dir, "log"), exist_ok=True)
    if osp.exists(args.output_jsonl):
        os.remove(args.output_jsonl)
    # Create the shard before evaluation so rank-0 can merge valid empty shards
    # even when every item assigned to a rank fails independently.
    open(args.output_jsonl, "w", encoding="utf-8").close()

    file_handler = logging.FileHandler(args.log_path)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    logger.info("Running %s on OVO-Bench proactive (CRR), rank %d/%d", args.run_name, rank, world_size)
    logger.info("Checkpoint path: %s", args.ckpt_path)
    logger.info("Rank output jsonl: %s", args.output_jsonl)
    eval_r(args)

    if world_size > 1:
        import torch.distributed as dist
        dist.barrier()
    shard_paths = [f"{output_stem}.rank{worker_rank}{output_ext}" for worker_rank in range(world_size)]
    if rank == 0:
        merge_rank_outputs(final_output_jsonl, shard_paths)
    # The pre-merge barrier is the only required collective.  Do not issue
    # any NCCL operation after rank-0-only filesystem merging: nonzero ranks
    # may already return.  Let process exit release the process group.


if __name__ == "__main__":
    main()
