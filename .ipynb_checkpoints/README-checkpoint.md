# 主动响应：实验交接说明

本文所有命令均从 `proactive_interaction_code` 运行。


## 1. 环境配置

```bash
cd proactive_interaction_code

conda create -n response python=3.10 -y
conda activate response
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

配置API评测环境

```bash
conda deactivate
conda create -n pvqa-judge-cu124 python=3.10 -y
conda activate pvqa-judge-cu124

pip install -U pip

pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install vllm==0.8.5.post1

pip install --force-reinstall --no-deps \
  huggingface_hub==0.30.2 \
  transformers==4.51.3 \
  tokenizers==0.21.1

conda deactivate
conda activate response
```

## 2. 模型下载

模型统一放在代码目录同级的 `../Qwen/`，已有的模型无需重复下载。

```bash
hf download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir ../Qwen/Qwen3-VL-8B-Instruct \
  --max-workers 4

hf download Qwen/Qwen3-8B \
  --local-dir ../Qwen/Qwen3-8B \
  --max-workers 4
```


## 3. 数据前置条件

数据集下载过程不在本文重复。以下路径应已存在：

```text
../proactive_interaction_data/data/MMDuet2-data/sft/
  live_whisperx-half_multi_half_single_question-2_sec_per_frame-max_180s-sft-h5_images.jsonl

../proactive_interaction_data/data/Live-WhisperX-selected/
  videos/<metadata.video_id>.mp4

../proactive_interaction_data/data/OVO-Bench/data/
  ovo_bench_new.json
  src_videos/

../proactive_interaction_data/data/StreamingBench-PO/
  StreamingBench/Proactive_Output.csv
  videos/sample_*/...

../proactive_interaction_data/data/ProactiveVideoQA/{WEB,EGO,TV,VAD}/videos/
../proactive_interaction_data/data/MMDuet2-data/evaluate/
  {web,ego,tv,vad}-frame_input_format.json
  {web,ego,tv,vad}-proactivevideoqa_format.json
```

**若缺少上述数据，需补充**


## 4. 构建可用训练样本

```bash
python probe/extract_video_subset_jsonl.py \
  --annotations ../proactive_interaction_data/data/MMDuet2-data/sft/live_whisperx-half_multi_half_single_question-2_sec_per_frame-max_180s-sft-h5_images.jsonl \
  --video_root ../proactive_interaction_data/data/Live-WhisperX-selected \
  --output ../proactive_interaction_data/runs/probe/main/live_sft_available.jsonl \
  --num_videos 756 \
  --manifest_output ../proactive_interaction_data/runs/probe/main/live_sft_available_videos.txt
```

若要选择不同规模，只修改 `--num_videos`，756是总训练样本数；其余后续路径保持不变。


## 5. 构造问题级神经元统计与 FP16 决策缓存

该步骤同时完成：
1. 解析 MMDuet2 流式 turn；
2. 构造每个问题的 `reply` / `NO REPLY` 配对；
3. 对每个原始决策点运行 Response-G1，读取各层 MLP 中间激活；
4. 聚合问题级 `reply - silence` 激活差；
5. 保存 FP16 激活缓存，后续特征阶段无需再次跑 Qwen/场景图。

```bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone \
  --nproc_per_node=8 \
  probe/build_probe_dataset_dp.py \
  --annotations ../proactive_interaction_data/runs/probe/main/live_sft_available.jsonl \
  --video_root ../proactive_interaction_data/data/Live-WhisperX-selected \
  --ckpt_path ../Qwen/Qwen3-VL-8B-Instruct \
  --output ../proactive_interaction_data/runs/probe/main/activation_stats_dp.pt \
  --mode stats \
  --decision_cache ../proactive_interaction_data/runs/probe/main/decision_cache_dp \
  --fps 1 \
  --frame_interval 1
```

若显存不够，需要加上 `--visual_context_frames 128`，表示仅将最近 128 帧原始视觉 token 输入 LLM；场景图记忆与检索仍保留完整历史。


## 6. 选择神经元并可视化

```bash
python probe/select_neurons.py \
  --stats ../proactive_interaction_data/runs/probe/main/activation_stats_dp.pt \
  --topk 0.03 \
  --output ../proactive_interaction_data/runs/probe/main/neuron_map_dp.pt
```

可视化结果：

```bash
python probe/visualize_probe_neurons.py \
  --neuron-map ../proactive_interaction_data/runs/probe/main/neuron_map_dp.pt \
  --output-dir ../proactive_interaction_data/runs/probe/main/neuron_visualization_dp \
  --display-topk 100 \
  --robust-percentile 99
```

## 7. 从 FP16 缓存提取 probe 特征、划分数据、训练

下面步骤不再运行视觉编码、场景图和 Qwen 前向；仅从第 6 节的 FP16 决策缓存按 `neuron_map_dp.pt` 切出特征。

```bash
python probe/materialize_probe_cache.py \
  --cache_manifest ../proactive_interaction_data/runs/probe/main/decision_cache_dp.manifest.json \
  --neuron_map ../proactive_interaction_data/runs/probe/main/neuron_map_dp.pt \
  --output ../proactive_interaction_data/runs/probe/main/probe_features_dp.pt

python probe/split_probe_features.py \
  --features ../proactive_interaction_data/runs/probe/main/probe_features_dp.pt \
  --train_output ../proactive_interaction_data/runs/probe/main/probe_train_dp.pt \
  --val_output ../proactive_interaction_data/runs/probe/main/probe_val_dp.pt \
  --val_ratio 0.1 \
  --seed 42

python probe/train_probe.py \
  --train_features ../proactive_interaction_data/runs/probe/main/probe_train_dp.pt \
  --eval_features ../proactive_interaction_data/runs/probe/main/probe_val_dp.pt \
  --output ../proactive_interaction_data/runs/probe/main/frozen_probe_dp.pt \
  --reg 10.0
```

选择回复阈值。保存值默认保留两位小数，供所有后续 probe 评测共用：

```bash
python probe/select_probe_threshold.py \
  --probe ../proactive_interaction_data/runs/probe/main/frozen_probe_dp.pt \
  --val_features ../proactive_interaction_data/runs/probe/main/probe_val_dp.pt \
  --output ../proactive_interaction_data/runs/probe/main/probe_threshold_dp.json \
  --threshold_decimals 2
```

查看最终阈值：

```bash
cat ../proactive_interaction_data/runs/probe/main/probe_threshold_dp.json
```

后续命令中的 `--probe_threshold` 填写该 JSON 中的 `threshold`；以下示例使用 `0.32`，请以实际输出替换。


## 8. OVO-Bench：baseline 与 probe

### 8.1 baseline

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone \
  --nproc_per_node=8 \
  probe/eval_ovobench_forward_probe_dp.py \
  --ckpt_path ../Qwen/Qwen3-VL-8B-Instruct \
  --task_json ../proactive_interaction_data/data/OVO-Bench/data/ovo_bench_new.json \
  --video_dir ../proactive_interaction_data/data/OVO-Bench/data/src_videos \
  --result_dir ../proactive_interaction_data/runs/eval/ovo_baseline_dp \
  --run_name baseline \
  --tasks REC SSR CRR \
  --fps 1 \
  --frame_interval 1
```

### 8.2 probe

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone \
  --nproc_per_node=8 \
  probe/eval_ovobench_forward_probe_dp.py \
  --ckpt_path ../Qwen/Qwen3-VL-8B-Instruct \
  --task_json ../proactive_interaction_data/data/OVO-Bench/data/ovo_bench_new.json \
  --video_dir ../proactive_interaction_data/data/OVO-Bench/data/src_videos \
  --result_dir ../proactive_interaction_data/runs/eval/ovo_probe_dp \
  --run_name probe \
  --tasks REC SSR CRR \
  --fps 1 \
  --frame_interval 1 \
  --probe_path ../proactive_interaction_data/runs/probe/main/frozen_probe_dp.pt \
  --neuron_map ../proactive_interaction_data/runs/probe/main/neuron_map_dp.pt \
  --probe_threshold 0.32
```

汇总论文表格式结果。评测脚本的输出文件名带时间戳，下面自动选择最新结果：

```bash
python eval_paper_metrics.py \
  --benchmark ovo \
  --result_files "$(ls -t ../proactive_interaction_data/runs/eval/ovo_baseline_dp/output/baseline_*.jsonl | head -n 1)" \
  --output_file ../proactive_interaction_data/runs/eval/ovo_baseline_dp/metrics.json

python eval_paper_metrics.py \
  --benchmark ovo \
  --result_files "$(ls -t ../proactive_interaction_data/runs/eval/ovo_probe_dp/output/probe_*.jsonl | head -n 1)" \
  --output_file ../proactive_interaction_data/runs/eval/ovo_probe_dp/metrics.json
```

## 9. StreamingBench-PO：baseline 与 probe

### 9.1 baseline

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone \
  --nproc_per_node=8 \
  probe/eval_streamingbench_proactive_probe_dp.py \
  --ckpt_path ../Qwen/Qwen3-VL-8B-Instruct \
  --task_csv ../proactive_interaction_data/data/StreamingBench-PO/StreamingBench/Proactive_Output.csv \
  --video_dir ../proactive_interaction_data/data/StreamingBench-PO/videos \
  --result_dir ../proactive_interaction_data/runs/eval/streaming_baseline_dp \
  --run_name baseline \
  --fps 1 \
  --frame_interval 1
```

### 9.2 probe

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone \
  --nproc_per_node=8 \
  probe/eval_streamingbench_proactive_probe_dp.py \
  --ckpt_path ../Qwen/Qwen3-VL-8B-Instruct \
  --task_csv ../proactive_interaction_data/data/StreamingBench-PO/StreamingBench/Proactive_Output.csv \
  --video_dir ../proactive_interaction_data/data/StreamingBench-PO/videos \
  --result_dir ../proactive_interaction_data/runs/eval/streaming_probe_dp \
  --run_name probe \
  --fps 1 \
  --frame_interval 1 \
  --probe_path ../proactive_interaction_data/runs/probe/main/frozen_probe_dp.pt \
  --neuron_map ../proactive_interaction_data/runs/probe/main/neuron_map_dp.pt \
  --probe_threshold 0.32
```

汇总指标：

```bash
python eval_paper_metrics.py \
  --benchmark streaming \
  --result_files "$(ls -t ../proactive_interaction_data/runs/eval/streaming_baseline_dp/output/baseline_*.jsonl | head -n 1)" \
  --output_file ../proactive_interaction_data/runs/eval/streaming_baseline_dp/metrics.json

python eval_paper_metrics.py \
  --benchmark streaming \
  --result_files "$(ls -t ../proactive_interaction_data/runs/eval/streaming_probe_dp/output/probe_*.jsonl | head -n 1)" \
  --output_file ../proactive_interaction_data/runs/eval/streaming_probe_dp/metrics.json
```

## 10. ProactiveVideoQA：四个子任务

### 10.1 baseline

```bash
for subset in web ego tv vad; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --standalone \
    --nproc_per_node=8 \
    proactivevideoqa_eval/eval_proactivevideoqa_responseg1_dp.py \
    --frame_input "../proactive_interaction_data/data/MMDuet2-data/evaluate/${subset}-frame_input_format.json" \
    --gold_file "../proactive_interaction_data/data/MMDuet2-data/evaluate/${subset}-proactivevideoqa_format.json" \
    --video_root "../proactive_interaction_data/data/ProactiveVideoQA/${subset^^}/videos" \
    --ckpt_path ../Qwen/Qwen3-VL-8B-Instruct \
    --output "../proactive_interaction_data/runs/eval/pvqa_${subset}_dp/baseline_pred.jsonl" \
    --frame_interval 1 \
    --visual_context_frames 8
done
```

### 10.2 probe

```bash
for subset in web ego tv vad; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --standalone \
    --nproc_per_node=8 \
    proactivevideoqa_eval/eval_proactivevideoqa_responseg1_dp.py \
    --frame_input "../proactive_interaction_data/data/MMDuet2-data/evaluate/${subset}-frame_input_format.json" \
    --gold_file "../proactive_interaction_data/data/MMDuet2-data/evaluate/${subset}-proactivevideoqa_format.json" \
    --video_root "../proactive_interaction_data/data/ProactiveVideoQA/${subset^^}/videos" \
    --ckpt_path ../Qwen/Qwen3-VL-8B-Instruct \
    --output "../proactive_interaction_data/runs/eval/pvqa_${subset}_dp/probe_pred.jsonl" \
    --frame_interval 1 \
    --visual_context_frames 8 \
    --probe_path ../proactive_interaction_data/runs/probe/main/frozen_probe_dp.pt \
    --neuron_map ../proactive_interaction_data/runs/probe/main/neuron_map_dp.pt \
    --probe_threshold 0.32
done
```

### 10.3 MMDuet2 官方评分与统计

**另开一个终端**，启动本地 Qwen3-8B judge，保持该终端运行，不要关闭。

```bash
conda activate pvqa-judge-cu124

CUDA_VISIBLE_DEVICES=0 vllm serve \
  ../proactive_interaction_data/models/Qwen3-8B \
  --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85
```

**返回response环境的终端**，计算最终指标。

baseline

```bash
for subset in web ego tv vad; do
  LLM_EVALUATOR_URL="http://127.0.0.1:8000/v1/chat/completions" \
  LLM_EVALUATOR_MODEL="Qwen/Qwen3-8B" \
  LLM_EVALUATOR_API_KEY="EMPTY" \
  python proactivevideoqa_eval/mmduet2_evaluate.py \
    --func evaluate \
    --input_file "../proactive_interaction_data/runs/eval/pvqa_${subset}_dp/baseline_pred.jsonl" \
    --gold_file "../proactive_interaction_data/data/MMDuet2-data/evaluate/${subset}-proactivevideoqa_format.json" \
    --output_file "../proactive_interaction_data/runs/eval/pvqa_${subset}_dp/baseline_judged.jsonl" \
    --num_workers 10

  python proactivevideoqa_eval/mmduet2_evaluate.py \
    --func stat_scores \
    --input_file "../proactive_interaction_data/runs/eval/pvqa_${subset}_dp/baseline_judged.jsonl"
done
```

probe

```bash
for subset in web ego tv vad; do
  LLM_EVALUATOR_URL="http://127.0.0.1:8000/v1/chat/completions" \
  LLM_EVALUATOR_MODEL="Qwen/Qwen3-8B" \
  LLM_EVALUATOR_API_KEY="EMPTY" \
  python proactivevideoqa_eval/mmduet2_evaluate.py \
    --func evaluate \
    --input_file "../proactive_interaction_data/runs/eval/pvqa_${subset}_dp/probe_pred.jsonl" \
    --gold_file "../proactive_interaction_data/data/MMDuet2-data/evaluate/${subset}-proactivevideoqa_format.json" \
    --output_file "../proactive_interaction_data/runs/eval/pvqa_${subset}_dp/probe_judged.jsonl" \
    --num_workers 10

  python proactivevideoqa_eval/mmduet2_evaluate.py \
    --func stat_scores \
    --input_file "../proactive_interaction_data/runs/eval/pvqa_${subset}_dp/probe_judged.jsonl"
done
```
