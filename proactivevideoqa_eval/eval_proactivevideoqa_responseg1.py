
#!/usr/bin/env python3
# MMDuet2 ProactiveVideoQA stream/JSONL protocol + Response-G1 inference adapter.
from __future__ import annotations
import argparse, json, logging, re, sys
from pathlib import Path
from typing import Any
import cv2
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen_online.utils import encode_images, whole_current_frame_tokens
from response_graph.Scene_Graph_ovo import Query_Graph_generation, Scene_Graph_generation_CRR, format_scene_graphs_for_prompt_offline
from probe.eval_ovobench_proactive_probe import Memory_Bank_Naive, text_kwargs, generate_trigger_kwargs
from probe.runtime import Qwen3VLMLPCollector, load_frozen_probe
from probe.multigpu import add_model_parallel_args, load_qwen3vl

NO_REPLY = "NO REPLY"
LOGGER = logging.getLogger("pvqa")


def graph_text(graphs):
    if not graphs:
        return ""
    out = "The relative scene graph shows:\n"
    for timestamp, graph in graphs:
        out += f"<{timestamp:.1f} seconds>:{format_scene_graphs_for_prompt_offline([graph])}\n"
    return out


def trigger_prompt(question, graphs):
    # Verbatim Response-G1 OVO proactive trigger wording.
    return ("\nYou're responsible of answering questions based on the video content.\n\n"
            + graph_text(graphs)
            + "\nThe following question are relevant to the latest frames, i.e. the end of the video.\n"
            + question + "\nDecide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.\n"
            + 'Answer only with "Yes" or "No".\nDo not include any additional text or explanation in your response.\n')


def answer_prompt(question, graphs, context_text):
    # This is MMDuet2's public proactive-answer semantic, reached only after gate=yes.
    supplemental = "\n".join(context_text)
    extra = "\nAdditional user-provided streaming context:\n" + supplemental + "\n" if supplemental else ""
    return ("\nYou are a helpful assistant. Answer the active question based on continuously incoming video frames. "
            "Your answer must include only information supported by the observed video since the last reply.\n\n"
            + graph_text(graphs) + "\nQuestion: " + question + "\n" + extra
            + "Evidence is sufficient. Answer the active question directly and concisely.\n")


def embed_prompt(model, processor, prompt):
    ids = processor.tokenizer(["<|im_start|>user\n" + prompt + "<|im_end|>\n"], **text_kwargs)
    ids = torch.tensor(ids["input_ids"], device=model.device)
    text = model.get_input_embeddings()(ids)
    assistant = processor.tokenizer(["<|im_start|>assistant\n"], **text_kwargs)
    assistant = torch.tensor(assistant["input_ids"], device=model.device)
    return torch.cat((text, model.get_input_embeddings()(assistant)), dim=1)


def make_inputs(model, processor, memories, prompt):
    visual = torch.cat(memories.get_context_frames(len(memories.memory_bank)), dim=1).to(model.device)
    inputs = torch.cat((visual, embed_prompt(model, processor, prompt)), dim=1).to(model.device)
    mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=model.device)
    return inputs, mask


def generate_answer(model, processor, memories, prompt):
    inputs, mask = make_inputs(model, processor, memories, prompt)
    try:
        ids = model.generate(inputs_embeds=inputs, attention_mask=mask, **{**generate_trigger_kwargs, "max_new_tokens": 128})
        text = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        del ids
        return text
    finally:
        del inputs, mask


def get_parts(turn):
    images, texts = [], []
    content = turn.get("content", [])
    if not isinstance(content, list):
        return images, [str(content)] if content else []
    for part in content:
        if part.get("type") == "image": images.append(part["image"])
        elif part.get("type") == "text" and part.get("text", "").strip(): texts.append(part["text"].strip())
    return images, texts


def initial_question(conversation):
    for turn in conversation:
        images, texts = get_parts(turn)
        if not images and texts: return "\n".join(texts)
    raise ValueError("frame_input example has no text-only initial question")


def video_path(root, video):
    root = Path(root)
    for path in (root / video, root / "videos" / video):
        if path.exists(): return path
    raise FileNotFoundError(f"Video {video!r} absent below {root}; extract the subset videos.zip first")


def source_timestamp(image_ref, fallback):
    try: index = int(Path(image_ref).stem)
    except ValueError: return fallback
    # MMDuet2 manifest identifies WEB source frames as magqa-2fps; all other
    # released PVQA frame directories are 1fps.
    return index / (2.0 if "magqa-2fps" in image_ref.replace("\\", "/") else 1.0)


def read_frame(path, timestamp):
    """Read the frame nearest to ``timestamp``, degrading gracefully.

    A timestamp past the end of the video is clamped to the last frame, and a
    failed decode retries a few neighbouring frame indices before giving up.
    This keeps one bad/corrupt video from dropping the whole sample.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target = int(timestamp * fps)
        if total > 0:
            target = max(0, min(target, total - 1))
        for pos in (target, target - 1, target + 1, target - 2, target + 2):
            if total > 0:
                pos = max(0, min(pos, total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ok, frame = cap.read()
            if ok:
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return None
    finally:
        cap.release()


def append_frame(model, processor, memories, image, timestamp):
    memories.reserve_next_frame()
    x = processor.image_processor(image)
    x["pixel_values"] = x["pixel_values"].to(model.device)
    x["image_grid_thw"] = x["image_grid_thw"].to(model.device)
    tokens, _, _ = encode_images(model, x["pixel_values"], x["image_grid_thw"])
    del x
    tokens = whole_current_frame_tokens(processor, model, timestamp, tokens)
    memories.update(tokens.unsqueeze(0).to(model.device))


def response_is_yes(text):
    return re.match(r"^\s*yes\b", text, flags=re.I) is not None


def run_one(example, gold, args, model, processor, probe, collector):
    qid = example["question_id"]
    question = initial_question(example["conversation"])
    video = video_path(args.video_root, gold[qid]["video"])
    turns = [get_parts(t) for t in example["conversation"]]
    turns = [(ims, txt) for ims, txt in turns if ims]
    if not turns: raise ValueError(f"{qid}: no visual turns")
    n_images = sum(len(x[0]) for x in turns)
    memories = Memory_Bank_Naive(memory_size=n_images, visual_memory_size=args.visual_context_frames)
    query_graph = Query_Graph_generation(processor, model, question)
    sg_interval = 2 if n_images <= 50 else (n_images - 1) // 50 + 2
    seen_images, stream_time, context = 0, 0.0, []
    result_turns = [{"role": "user", "content": question, "time": 0.0}]
    for image_refs, texts in turns:
        for image_ref in image_refs:
            stream_time += args.frame_interval  # MMDuet2 ProactiveInferenceClient convention.
            image = read_frame(video, source_timestamp(image_ref, stream_time))
            if image is None: raise RuntimeError(f"Unreadable frame: {video} / {image_ref}")
            append_frame(model, processor, memories, image, source_timestamp(image_ref, stream_time))
            seen_images += 1
            if seen_images == 1 or (seen_images - 1) % sg_interval == 0:
                clip = torch.cat(memories.get_context_frames(sg_interval), dim=1).to(model.device)
                graph = Scene_Graph_generation_CRR(processor, model, clip, question, query_graph)
                if len(graph.get("scene_graph", [])): memories.update_graph(stream_time, graph)
                del clip
        context.extend(texts)
        for text in texts: result_turns.append({"role": "user", "content": text, "time": stream_time})
        graphs = memories.get_graphs_top_new(model, processor, query_graph)
        inputs, mask = make_inputs(model, processor, memories, trigger_prompt(question, graphs))
        try:
            if probe is None:
                ids = model.generate(inputs_embeds=inputs, attention_mask=mask, **generate_trigger_kwargs)
                gate_text = processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
                del ids
                reply, probability = response_is_yes(gate_text), None
            else:
                activations = collector.collect(model, inputs, mask)
                probability = probe.probability(activations)
                reply = probability >= args.probe_threshold
                del activations
        finally:
            del inputs, mask
        if reply:
            content = generate_answer(model, processor, memories, answer_prompt(question, graphs, context))
            if content:
                turn = {"role": "assistant", "content": content, "time": stream_time}
                if probability is not None: turn["probe_reply_probability"] = round(probability, 8)
                result_turns.append(turn)
        torch.cuda.empty_cache()
    return {"question_id": qid, "model_response_list": result_turns}


def parser():
    p = argparse.ArgumentParser("Response-G1 ProactiveVideoQA using MMDuet2 protocol")
    p.add_argument("--frame_input", required=True, help="MMDuet2 *-frame_input_format.json")
    p.add_argument("--gold_file", required=True, help="MMDuet2 *-proactivevideoqa_format.json")
    p.add_argument("--video_root", required=True, help="Extracted PVQA subset videos directory")
    p.add_argument("--ckpt_path", required=True)
    p.add_argument("--output", required=True, help="MMDuet2-compatible prediction JSONL")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--frame_interval", type=float, default=1.0)
    p.add_argument("--visual_context_frames", type=int, default=None)
    p.add_argument("--min_pixels", type=int, default=448*448)
    p.add_argument("--max_pixels", type=int, default=448*448)
    p.add_argument("--probe_path", default=None, help="omit for original Response-G1 trigger baseline")
    p.add_argument("--neuron_map", default=None)
    p.add_argument("--probe_threshold", type=float, default=None)
    add_model_parallel_args(p)
    return p


def main():
    args = parser().parse_args()
    if args.visual_context_frames is not None and args.visual_context_frames < 1: raise ValueError("--visual_context_frames must be >=1")
    if args.probe_path and (not args.neuron_map or args.probe_threshold is None): raise ValueError("probe needs --neuron_map and --probe_threshold")
    data = json.load(open(args.frame_input, encoding="utf-8"))
    gold = {x["question_id"]: x for x in json.load(open(args.gold_file, encoding="utf-8"))}
    end = len(data) if args.end_idx is None else args.end_idx
    if args.max_samples is not None: end = min(end, args.start_idx + args.max_samples)
    data = data[args.start_idx:end]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if Path(args.output).exists(): done = {json.loads(x)["question_id"] for x in open(args.output, encoding="utf-8") if x.strip()}
    model = load_qwen3vl(args.ckpt_path, args).eval()
    processor = AutoProcessor.from_pretrained(args.ckpt_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels, trust_remote_code=True)
    collector = Qwen3VLMLPCollector(model) if args.probe_path else None
    probe = load_frozen_probe(args.probe_path, args.neuron_map) if args.probe_path else None
    with open(args.output, "a", encoding="utf-8") as f, torch.no_grad():
        for example in tqdm(data, desc="ProactiveVideoQA", unit="sample"):
            if example["question_id"] in done: continue
            if example["question_id"] not in gold: raise KeyError(example["question_id"])
            output = run_one(example, gold, args, model, processor, probe, collector)
            f.write(json.dumps(output, ensure_ascii=False) + "\n"); f.flush()
    if collector: collector.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
    main()
