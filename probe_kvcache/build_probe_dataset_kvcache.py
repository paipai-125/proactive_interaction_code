#!/usr/bin/env python3
"""Isolated sliding-KV Response-G1 neuron-statistics builder.

Reuses Response-G1 scene graphs and MMDuet2 labels. It never edits probe/.
The cache stores only recent serialized visual-frame turns. Trigger text is
temporarily appended, activations are read, then trigger KV is removed.

Important: window eviction makes this a bounded-KV approximation. Validate it
against probe/build_probe_dataset.py on the same small subset before treating
its measurements as a main result.
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict, deque
from pathlib import Path
import torch
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_ovobench_proactive import Memory_Bank_Naive, build_prompt, extract_frame, text_kwargs
from probe.build_probe_dataset import raw_video_path, turn_events
from probe.multigpu import add_model_parallel_args, load_qwen3vl
from qwen_online.utils import encode_images, whole_current_frame_tokens
from response_graph.Scene_Graph_ovo import Query_Graph_generation, Scene_Graph_generation_CRR

class Collector:
    def __init__(self, model):
        self.names, self.modules = [], []
        for name, module in model.named_modules():
            if name.endswith(".mlp") and all(hasattr(module, x) for x in ("gate_proj","up_proj","down_proj","act_fn")):
                self.names.append(name); self.modules.append(module)
        self.values, self.active = [], False
        self.handles = [m.register_forward_pre_hook(self.hook) for m in self.modules]
    def hook(self, m, inputs):
        if self.active:
            x = inputs[0][:, -1:, :]
            self.values.append((m.act_fn(m.gate_proj(x))*m.up_proj(x))[0,0].detach().float().cpu())
    def norms(self):
        return torch.stack([m.down_proj.weight.detach().float().cpu().norm(p=2, dim=0) for m in self.modules])
    def close(self):
        for h in self.handles: h.remove()

class SlidingKV:
    def __init__(self, model, frames):
        self.model, self.frames, self.cache, self.lengths = model, frames, None, deque()
    def n(self): return 0 if self.cache is None else int(self.cache.get_seq_length())
    def trim(self, a, b=None):
        for layer in self.cache.layers:
            if getattr(layer, "is_initialized", False):
                layer.keys = layer.keys[..., a:b, :].contiguous()
                layer.values = layer.values[..., a:b, :].contiguous()
    def append(self, x):
        while len(self.lengths) >= self.frames:
            self.trim(self.lengths.popleft(), None)
        n = self.n()
        mask = torch.ones((1,n+x.shape[1]), dtype=torch.long, device=x.device)
        with torch.no_grad():
            out = self.model.model(inputs_embeds=x, attention_mask=mask, past_key_values=self.cache, use_cache=True, return_dict=True)
        self.cache = out.past_key_values; del out
        if self.cache is None: raise RuntimeError("Qwen3-VL returned no past_key_values")
        self.lengths.append(x.shape[1])
    def decision(self, suffix, collector):
        base = self.n()
        mask = torch.ones((1,base+suffix.shape[1]), dtype=torch.long, device=suffix.device)
        collector.values.clear(); collector.active = True
        try:
            with torch.no_grad():
                out = self.model.model(inputs_embeds=suffix, attention_mask=mask, past_key_values=self.cache, use_cache=True, return_dict=True)
            self.cache = out.past_key_values; del out
        finally:
            collector.active = False
        if len(collector.values) != len(collector.modules):
            raise RuntimeError(f"Expected {len(collector.modules)} activations, got {len(collector.values)}")
        ans = torch.stack(collector.values)
        self.trim(0, base)
        if self.n() != base: raise RuntimeError("temporary trigger KV was not restored")
        return ans

def suffix(model, processor, mem, question, qgraph):
    graphs = mem.get_graphs_top_new(model, processor, qgraph)
    prompt = build_prompt("CRR", None, None, {"question":question,"answer":""}, 0, graphs)
    ids = processor.tokenizer(["<|im_start|>user\n"+prompt+"<|im_end|>\n"], **text_kwargs)
    a = model.get_input_embeddings()(torch.tensor(ids["input_ids"], device=model.device))
    ids = processor.tokenizer(["<|im_start|>assistant\n"], **text_kwargs)
    b = model.get_input_embeddings()(torch.tensor(ids["input_ids"], device=model.device))
    return torch.cat([a,b], dim=1)

def process(record, args, model, processor, col, stats):
    path = raw_video_path(args.video_root, record["metadata"]["video_id"])
    if not path.exists(): print("[skip missing]",path); return 0
    events, episodes, skipped = turn_events(record)
    for key, value in skipped.items(): stats["skipped_episodes"][key] = stats["skipped_episodes"].get(key,0)+value
    events = [e for e in events if e["use_for_neuron_stats"]]
    if not events: return 0
    by = defaultdict(list)
    for e in events:
        e["frame"] = int(round(e["time"]*args.fps)); by[e["frame"]].append(e)
    end, ask = max(by), min(by)
    interval = 2 if end-ask+1 <= 50 else (end-ask)//50+2
    mem = Memory_Bank_Naive(memory_size=end+1, visual_memory_size=args.kv_window_frames)
    kv, active, qgraph, count = SlidingKV(model,args.kv_window_frames), None, None, 0
    acts = {e["id"]:{"reply":[],"silence":[]} for e in episodes}
    for i in range(end+1):
        t=i/args.fps; frame=extract_frame(str(path),t) or extract_frame(str(path),max(t-1,0))
        if frame is None: continue
        mem.reserve_next_frame()
        fi=processor.image_processor(frame)
        fi["pixel_values"]=fi["pixel_values"].to(model.device); fi["image_grid_thw"]=fi["image_grid_thw"].to(model.device)
        ft,_,_=encode_images(model,fi["pixel_values"],fi["image_grid_thw"]); del fi
        ft=whole_current_frame_tokens(processor,model,t,ft)
        if ft is None: continue
        ft=ft.unsqueeze(0).to(model.device); mem.update(ft); kv.append(ft)
        if i in by and by[i][0]["question"] != active:
            active=by[i][0]["question"]; qgraph=Query_Graph_generation(processor,model,active)
        need_sg = i==0 or i%interval==0; graph_q=None
        if need_sg and active is not None:
            clip=torch.cat(mem.get_context_frames(interval),dim=1).to(model.device)
            graph=Scene_Graph_generation_CRR(processor,model,clip,active,qgraph)
            if len(graph.get("scene_graph",[])): mem.update_graph(t,graph)
            graph_q=active
        for e in by.get(i,[]):
            if e["question"] != active:
                active=e["question"]; qgraph=Query_Graph_generation(processor,model,active)
            if need_sg and graph_q != active:
                clip=torch.cat(mem.get_context_frames(interval),dim=1).to(model.device)
                graph=Scene_Graph_generation_CRR(processor,model,clip,active,qgraph)
                if len(graph.get("scene_graph",[])): mem.update_graph(t,graph)
                graph_q=active
            h=kv.decision(suffix(model,processor,mem,active,qgraph),col)
            acts[e["episode_id"]]["reply" if e["label"] else "silence"].append(h); count+=1; del h
            torch.cuda.empty_cache(); torch.cuda.synchronize()
    for e in episodes:
        x=acts[e["id"]]
        if x["reply"] and x["silence"]:
            stats["episode_delta_sum"].add_(torch.stack(x["reply"]).mean(0)-torch.stack(x["silence"]).mean(0))
            stats["paired_question_count"]+=1
        else: stats["skipped_episodes"]["unreadable_event_frame"]=stats["skipped_episodes"].get("unreadable_event_frame",0)+1
    return count

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--annotations",required=True);p.add_argument("--video_root",required=True);p.add_argument("--ckpt_path",required=True);p.add_argument("--output",required=True)
    p.add_argument("--max_records",type=int,default=None);p.add_argument("--fps",type=int,default=1);p.add_argument("--frame_interval",type=int,default=1)
    p.add_argument("--min_pixels",type=int,default=448*448);p.add_argument("--max_pixels",type=int,default=448*448);p.add_argument("--kv_window_frames",type=int,default=16)
    add_model_parallel_args(p);args=p.parse_args()
    if args.fps!=1 or args.frame_interval!=1: p.error("only 1 FPS / frame_interval=1 is supported")
    if args.kv_window_frames<1:p.error("--kv_window_frames must be positive")
    model=load_qwen3vl(args.ckpt_path,args)
    processor=AutoProcessor.from_pretrained(args.ckpt_path,min_pixels=args.min_pixels,max_pixels=args.max_pixels,trust_remote_code=True)
    col=Collector(model); width=col.norms().shape[1]
    stats={"episode_delta_sum":torch.zeros(len(col.modules),width),"paired_question_count":0,"skipped_episodes":{},"down_projection_norms":col.norms(),"layer_names":col.names,"execution":"sliding_visual_kv_approximation","kv_window_frames":args.kv_window_frames}
    total=0
    with open(args.annotations,encoding="utf-8") as f:
        for n,line in enumerate(f):
            if args.max_records is not None and n>=args.max_records:break
            total+=process(json.loads(line),args,model,processor,col,stats)
            print(f"processed records={n+1}, decisions={total}",flush=True)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);torch.save(stats,args.output)
    print(f"Saved KV-cache statistics from {stats['paired_question_count']} paired questions ({total} processed decisions) to {args.output}")
    print("Skipped question episodes:",stats["skipped_episodes"]);col.close()
if __name__=="__main__":main()
