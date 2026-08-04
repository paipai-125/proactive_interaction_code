"""Paper-table scorer for Response-G1 benchmark outputs.

This scorer intentionally follows the official OVO-Bench scoring rules and
the per-record correctness convention already used by Response-G1's two
StreamingBench runners.  It is a post-processing tool: model inference stays
in the original Response-G1 scripts.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


OVO_REALTIME = ("OCR", "ACR", "ATR", "STU", "FPD", "OJR")
OVO_BACKWARD = ("EPM", "ASI", "HLD")
OVO_FORWARD = ("REC", "SSR", "CRR")
OVO_TASKS = OVO_REALTIME + OVO_BACKWARD + OVO_FORWARD

STREAMING_TASKS = (
    "Object Perception",
    "Causal Reasoning",
    "Clips Summarize",
    "Attribute Perception",
    "Event Understanding",
    "Text-Rich Understanding",
    "Prospective Reasoning",
    "Spatial Understanding",
    "Action Perception",
    "Counting",
)
STREAMING_SHORT = {
    "Object Perception": "OP",
    "Causal Reasoning": "CR",
    "Clips Summarize": "CS",
    "Attribute Perception": "ATP",
    "Event Understanding": "EU",
    "Text-Rich Understanding": "TR",
    "Prospective Reasoning": "PR",
    "Spatial Understanding": "SU",
    "Action Perception": "ACP",
    "Counting": "CT",
}


def _read_json_or_jsonl(path):
    """Read either Response-G1 JSONL output or the official OVO JSON bundle."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and {"backward", "realtime", "forward"} & set(payload):
            rows = []
            for split in ("backward", "realtime", "forward"):
                rows.extend(payload.get(split, []))
            return rows
        if isinstance(payload, list):
            return payload
        return [payload]

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def _percent(correct, total):
    return None if total == 0 else 100.0 * correct / total


def _mean(values):
    values = [value for value in values if value is not None]
    return None if not values else sum(values) / len(values)


def _as_yes(value):
    return str(value).strip().lower() in {"yes", "y"}


def _ovo_choice_correct(row):
    # This is exactly the official OVO-Bench rule: the expected option letter
    # must occur in the generated response.  Do not replace it with a custom
    # answer normalizer or an LLM judge.
    response = row.get("response")
    target = row.get("ground_truth")
    return response is not None and target is not None and str(target) in str(response)


def _ovo_forward_correct(task, point):
    response = point.get("response")
    if response is None:
        return False
    response = str(response).strip()
    if task == "REC":
        # Official OVO scorer concatenates every digit in the answer.
        return "".join(re.findall(r"\d+", response)) == str(point.get("count"))
    expected_type = point.get("type")
    if (response == "N" and expected_type == 0) or (response == "Y" and expected_type == 1):
        return True
    expected = "No" if expected_type == 0 else "Yes"
    return expected in response


def score_ovo(rows, require_full=False):
    """Reproduce the OVO official task scoring and Table-1 macro averages."""
    task_hits = defaultdict(int)
    task_total = defaultdict(int)
    seen = set()
    duplicates = 0

    for row in rows:
        task = row.get("task")
        if task not in OVO_TASKS:
            continue  # aggregate rows or unrelated logs
        key = (task, row.get("id", row.get("question_id")))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)

        if task in OVO_FORWARD:
            points = row.get("test_info", [])
            for point in points:
                task_total[task] += 1
                task_hits[task] += int(_ovo_forward_correct(task, point))
        else:
            task_total[task] += 1
            task_hits[task] += int(_ovo_choice_correct(row))

    missing = [task for task in OVO_TASKS if task_total[task] == 0]
    if require_full and missing:
        raise ValueError("incomplete OVO coverage; missing: " + ", ".join(missing))

    per_task = {task: _percent(task_hits[task], task_total[task]) for task in OVO_TASKS}
    groups = {
        "real_time_visual_perception": _mean([per_task[task] for task in OVO_REALTIME]),
        "backward_tracing": _mean([per_task[task] for task in OVO_BACKWARD]),
        "forward_active_responding": _mean([per_task[task] for task in OVO_FORWARD]),
    }
    overall = _mean(list(groups.values()))
    return {
        "benchmark": "OVO-Bench",
        "score_source": "official OVO-Bench task rules; Table-1 macro averages",
        "task_order": list(OVO_TASKS),
        "per_task_percent": per_task,
        "correct": {task: task_hits[task] for task in OVO_TASKS},
        "total": {task: task_total[task] for task in OVO_TASKS},
        "group_average_percent": groups,
        "overall_percent": overall,
        "missing_tasks": missing,
        "duplicate_rows_skipped": duplicates,
    }


def score_streaming(rows, require_full=False):
    """Score the Table-2 columns: 10 RTVU types, All, PO, and Overall."""
    reactive_hits = defaultdict(int)
    reactive_total = defaultdict(int)
    po_hits = 0
    po_total = 0
    seen = set()
    duplicates = 0

    for row in rows:
        question_id = row.get("question_id")
        task = row.get("task_type")
        if task in STREAMING_TASKS and "answer" in row and "response" in row:
            key = ("reactive", question_id)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            reactive_total[task] += 1
            reactive_hits[task] += int(str(row["response"]).strip() == str(row["answer"]).strip())
        elif "ground_truth_output" in row and "correct" in row:
            key = ("po", question_id)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            po_total += 1
            po_hits += int(_as_yes(row["correct"]))

    missing = [task for task in STREAMING_TASKS if reactive_total[task] == 0]
    if po_total == 0:
        missing.append("PO")
    if require_full and missing:
        raise ValueError("incomplete StreamingBench coverage; missing: " + ", ".join(missing))

    per_task = {
        STREAMING_SHORT[task]: _percent(reactive_hits[task], reactive_total[task])
        for task in STREAMING_TASKS
    }
    reactive_hits_all = sum(reactive_hits.values())
    reactive_total_all = sum(reactive_total.values())
    all_score = _percent(reactive_hits_all, reactive_total_all)
    po_score = _percent(po_hits, po_total)
    overall = _percent(reactive_hits_all + po_hits, reactive_total_all + po_total)
    return {
        "benchmark": "StreamingBench",
        "score_source": "Response-G1 per-record correctness convention; Table-2 micro averages",
        "task_order": [STREAMING_SHORT[task] for task in STREAMING_TASKS] + ["All", "PO", "Overall"],
        "per_task_percent": per_task,
        "correct": {STREAMING_SHORT[task]: reactive_hits[task] for task in STREAMING_TASKS},
        "total": {STREAMING_SHORT[task]: reactive_total[task] for task in STREAMING_TASKS},
        "all_percent": all_score,
        "po_percent": po_score,
        "overall_percent": overall,
        "po_correct": po_hits,
        "po_total": po_total,
        "missing_tasks": missing,
        "duplicate_rows_skipped": duplicates,
    }


def _fmt(value):
    return "N/A" if value is None else f"{value:.1f}"


def print_ovo(metrics):
    print("OVO-Bench (paper Table 1 order, percent)")
    print(" | ".join(f"{task}={_fmt(metrics['per_task_percent'][task])}" for task in OVO_TASKS))
    groups = metrics["group_average_percent"]
    print(
        "RTVP Avg={}; BT Avg={}; FAR Avg={}; Overall={}".format(
            _fmt(groups["real_time_visual_perception"]),
            _fmt(groups["backward_tracing"]),
            _fmt(groups["forward_active_responding"]),
            _fmt(metrics["overall_percent"]),
        )
    )


def print_streaming(metrics):
    print("StreamingBench (paper Table 2 order, percent)")
    print(" | ".join(f"{task}={_fmt(metrics['per_task_percent'][task])}" for task in metrics["per_task_percent"]))
    print("All={}; PO={}; Overall={}".format(
        _fmt(metrics["all_percent"]), _fmt(metrics["po_percent"]), _fmt(metrics["overall_percent"])
    ))


def main():
    parser = argparse.ArgumentParser(description="Score Response-G1 outputs in the two paper table formats")
    parser.add_argument("--benchmark", choices=("ovo", "streaming"), required=True)
    parser.add_argument("--result_files", nargs="+", required=True, help="One or more JSON/JSONL model-output files")
    parser.add_argument("--output_file", required=True, help="Metrics JSON to write")
    parser.add_argument("--require_full", action="store_true", help="Fail unless every paper-table task has output")
    args = parser.parse_args()

    rows = []
    for result_file in args.result_files:
        rows.extend(_read_json_or_jsonl(result_file))
    metrics = score_ovo(rows, args.require_full) if args.benchmark == "ovo" else score_streaming(rows, args.require_full)
    print_ovo(metrics) if args.benchmark == "ovo" else print_streaming(metrics)
    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print("Wrote metrics to:", args.output_file)


if __name__ == "__main__":
    main()
