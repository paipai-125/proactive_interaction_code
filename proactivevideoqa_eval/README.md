# ProactiveVideoQA evaluation

`eval_proactivevideoqa_responseg1.py` reuses MMDuet2’s released `*-frame_input_format.json` as the stream schedule and outputs MMDuet2-compatible `model_response_list` JSONL. It reuses Response-G1's Qwen visual encoding, graph creation/retrieval, and original Yes/No trigger. With `--probe_path`, only the trigger is replaced by the frozen probe; all subsequent answer generation and the evaluator remain identical.

`mmduet2_evaluate.py` is the unmodified public MMDuet2 `proactive_eval/evaluate.py`.
