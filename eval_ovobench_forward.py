"""Response-G1 runner for all three OVO-Bench forward tasks.

It reuses the published CRR streaming implementation and the official
OVO-Bench REC/SSR prompt templates.  The output schema is the official
``test_info[i]['response']`` schema consumed by ``eval_paper_metrics.py``.
"""

import argparse
import copy
import json
import logging
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime

import torch
from tqdm import tqdm
from transformers import AutoProcessor

import eval_ovobench_proactive as base
from probe.multigpu import add_model_parallel_args, load_qwen3vl
from response_graph.Scene_Graph_ovo import (
    Query_Graph_generation,
    Scene_Graph_generation_CRR,
    format_scene_graphs_for_prompt_offline,
)


FORWARD_TASKS = ("REC", "SSR", "CRR")
TEXT_KWARGS = {"padding": True, "return_token_type_ids": False}
GENERATE_KWARGS = base.generate_trigger_kwargs
LOGGER = logging.getLogger(__name__)

REC_PROMPT = """You're watching a video in which people may perform a certain type of action repetively.
The person performing this kind of action are referred to as 'they' in the following statement. You're task is to count how many times have different people in the video perform this kind of action in total. One complete motion counts as one. Now, answer the following question: {question} Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count. Do not include any additional text or explanation in your response."""
SSR_PROMPT = """You're watching a tutorial video which contain a sequential of steps.
The following is one step from the whole procedures: {step} Your task is to determine if the man or woman in the video is currently performing this step. Answer only with “Yes” or “No”. Do not include any additional text or explanation in your response."""


def query_for_point(item, point):
    if item["task"] == "REC":
        return "How many times did they " + item["activity"] + "?"
    if item["task"] == "SSR":
        return "Is the person currently performing this step: " + point["step"] + "?"
    return item["question"]


def answer_prompt(task, query, point, graphs):
    graph_text = ""
    if graphs:
        graph_text = "The relative scene graph shows:\n"
        for timestamp, graph in graphs:
            graph_text += f"<{timestamp:.1f} seconds>:"
            graph_text += format_scene_graphs_for_prompt_offline([graph]) + "\n"
    if task == "REC":
        body = REC_PROMPT.format(question=query)
    elif task == "SSR":
        body = SSR_PROMPT.format(step=point["step"])
    else:
        # This is the published Response-G1 CRR trigger template.
        return base.build_prompt(task, query, None, {"question": query, "answer": ""}, None, graphs)
    return graph_text + "\n" + body


def generate_text(model, processor, frame_tokens, prompt):
    text_ids = processor.tokenizer(["<|im_start|>user\n" + prompt + "<|im_end|>\n"], **TEXT_KWARGS)
    text_ids = torch.tensor(text_ids["input_ids"]).to(model.device)
    text_embeddings = model.get_input_embeddings()(text_ids)
    assistant_ids = processor.tokenizer(["<|im_start|>assistant\n"], **TEXT_KWARGS)
    assistant_ids = torch.tensor(assistant_ids["input_ids"]).to(model.device)
    assistant_embeddings = model.get_input_embeddings()(assistant_ids)
    inputs_embeds = torch.cat([frame_tokens, text_embeddings, assistant_embeddings], dim=1).to(model.device)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=model.device)
    generated = model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **GENERATE_KWARGS)
    response = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    del text_ids, text_embeddings, assistant_ids, assistant_embeddings, inputs_embeds, attention_mask, generated
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return response


def process_query_group(item, indices, query, args, model, processor):
    """Run one query over its prefix stream and fill the requested test points.

    SSR has a different query for each step.  We therefore group test points by
    step and replay the published CRR stream once per distinct step, instead of
    leaking a future step description into an earlier decision.
    """
    realtime_map = defaultdict(list)
    for index in indices:
        realtime_map[item["test_info"][index]["realtime"]].append(index)
    end_frame = int(max(realtime_map) * args.fps)
    memory = base.Memory_Bank_Naive(
        memory_size=end_frame + 1,
        visual_memory_size=args.visual_context_frames,
    )
    query_graph = Query_Graph_generation(processor, model, query)
    total_frames = end_frame + 1
    interval = 2 if total_frames <= 50 else (total_frames - 1) // 50 + 2

    for frame_idx in range(0, end_frame + 1, args.frame_interval):
        timestamp = frame_idx / args.fps
        frame = base.extract_frame(osp.join(args.video_dir, item["video"]), timestamp)
        if frame is None and timestamp > 0:
            frame = base.extract_frame(osp.join(args.video_dir, item["video"]), timestamp - 1)
        if frame is None:
            continue
        memory.reserve_next_frame()
        image = processor.image_processor(frame)
        image["pixel_values"] = image["pixel_values"].to(model.device)
        image["image_grid_thw"] = image["image_grid_thw"].to(model.device)
        current, _, _ = base.encode_images(model, image["pixel_values"], image["image_grid_thw"])
        current = base.whole_current_frame_tokens(processor, model, timestamp, current).unsqueeze(0).to(model.device)
        del image
        memory.update(current)

        should_graph = (
            (args.sg_force_first_frame and frame_idx == 0)
            or (interval > 0 and frame_idx % interval == 0 and frame_idx != 0)
            or interval <= 0
        )
        if should_graph:
            clip = torch.cat(memory.get_context_frames(interval if interval > 0 else 1), dim=1).to(model.device)
            graph = Scene_Graph_generation_CRR(processor, model, clip, query, query_graph)
            del clip
            if graph.get("scene_graph"):
                memory.update_graph(timestamp, graph)

        # The official annotations are evaluated at their listed timestamps.
        if timestamp in realtime_map:
            graphs = memory.get_graphs_top_new(model, processor, query_graph)
            raw_window = 600 if args.visual_context_frames is None else args.visual_context_frames
            frames = torch.cat(memory.get_context_frames(raw_window), dim=1).to(model.device)
            for index in realtime_map[timestamp]:
                point = item["test_info"][index]
                item["test_info"][index]["response"] = generate_text(
                    model, processor, frames, answer_prompt(item["task"], query, point, graphs)
                )
            del frames


def run(args):
    torch.manual_seed(1234)
    model = load_qwen3vl(args.ckpt_path, args)
    processor = AutoProcessor.from_pretrained(
        args.ckpt_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels, trust_remote_code=True
    )
    model.eval()
    with open(args.task_file, "r", encoding="utf-8") as handle:
        items = [item for item in json.load(handle) if item.get("task") in args.tasks]

    with torch.no_grad(), open(args.output_jsonl, "w", encoding="utf-8") as output:
        for source_item in tqdm(items, desc="OVO forward tasks"):
            try:
                item = copy.deepcopy(source_item)
                groups = defaultdict(list)
                for index, point in enumerate(item["test_info"]):
                    groups[query_for_point(item, point)].append(index)
                for query, indices in groups.items():
                    process_query_group(item, indices, query, args, model, processor)
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
                output.flush()
            except Exception as error:
                LOGGER.exception("skip OVO item id=%s: %s", source_item.get("id"), error)


def build_parser():
    parser = argparse.ArgumentParser(description="Response-G1 evaluation for OVO REC/SSR/CRR")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--task_json", "--task_csv", dest="task_file", required=True)
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--run_name", default="forward_output")
    parser.add_argument("--tasks", nargs="+", choices=FORWARD_TASKS, default=list(FORWARD_TASKS))
    parser.add_argument("--min_pixels", type=int, default=448 * 448)
    parser.add_argument("--max_pixels", type=int, default=448 * 448)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--visual_context_frames", type=int, default=None)
    parser.add_argument("--sg_force_first_frame", action=argparse.BooleanOptionalAction, default=True)
    add_model_parallel_args(parser)
    return parser


def main():
    args = build_parser().parse_args()
    if args.visual_context_frames is not None and args.visual_context_frames < 1:
        raise ValueError("--visual_context_frames must be positive")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_jsonl = osp.join(args.result_dir, "output", f"{args.run_name}_{timestamp}.jsonl")
    os.makedirs(osp.dirname(args.output_jsonl), exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
