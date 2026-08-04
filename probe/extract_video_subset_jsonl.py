#!/usr/bin/env python3
"""Select complete local Live-WhisperX and/or EgoExoLearn MMDuet2 SFT records.

Live-only invocations keep their original JSONL schema. Ego-only and joint
invocations tag output records with metadata.dataset so the probe builder can
resolve the corresponding downloaded video root.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterator

EGOEXO_SEGMENT = re.compile(
    r"^(?P<uid>.+)-(?P<start>\d+(?:\.\d+)?)s_(?P<end>\d+(?:\.\d+)?)s$"
)


def iter_records(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {error}") from error


def get_video_id(record: dict, line_number: int) -> str:
    metadata = record.get("metadata")
    video_id = metadata.get("video_id") if isinstance(metadata, dict) else None
    if not isinstance(video_id, str) or not video_id:
        raise ValueError(f"Line {line_number} has no metadata.video_id.")
    return video_id


def live_video_path(root: Path, video_id: str) -> Path:
    candidates = (root / "videos" / f"{video_id}.mp4", root / f"{video_id}.mp4")
    return next((path for path in candidates if path.is_file()), candidates[0])


def ego_video_path(root: Path, video_id: str) -> Path:
    match = EGOEXO_SEGMENT.match(video_id)
    if match is None:
        raise ValueError(f"Invalid EgoExoLearn clip video_id: {video_id}")
    uid = match.group("uid")
    candidates = (root / f"{uid}.mp4", root / "videos" / f"{uid}.mp4")
    return next((path for path in candidates if path.is_file()), candidates[0])


def tagged(record: dict, dataset: str) -> dict:
    result = dict(record)
    result["metadata"] = dict(record["metadata"])
    result["metadata"]["dataset"] = dataset
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select complete local MMDuet2 Live-WhisperX and/or EgoExoLearn SFT records."
    )
    parser.add_argument("--annotations", type=Path, default=None,
                        help="Optional Live-WhisperX SFT JSONL.")
    parser.add_argument("--video_root", type=Path, default=None,
                        help="Optional Live-WhisperX root containing videos/<video_id>.mp4.")
    parser.add_argument("--ego_annotations", type=Path, default=None,
                        help="Optional EgoExoLearn SFT JSONL.")
    parser.add_argument("--egoexolearn_video_root", type=Path, default=None,
                        help="Optional EgoExoLearn root containing <source_uid>.mp4.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSONL; existing path may be retained.")
    parser.add_argument("--num_videos", type=int, default=20,
                        help="Backward-compatible default cap for each configured source.")
    parser.add_argument("--num_live_videos", type=int, default=None,
                        help="Live-WhisperX cap; overrides --num_videos for Live only.")
    parser.add_argument("--num_egoexolearn_videos", type=int, default=None,
                        help="EgoExoLearn cap; overrides --num_videos for Ego only.")
    parser.add_argument("--all_available", action="store_true",
                        help="Select every locally available video from each configured source.")
    parser.add_argument("--manifest_output", type=Path, default=None,
                        help="Optional selected source/video-id manifest.")
    return parser.parse_args()


def select_source(path: Path, video_root: Path, dataset: str, all_available: bool, num_videos: int):
    resolver = live_video_path if dataset == "live_whisperx" else ego_video_path
    selected, selected_set, missing = [], set(), Counter()
    for line_number, record in iter_records(path):
        video_id = get_video_id(record, line_number)
        if video_id in selected_set:
            continue
        if resolver(video_root, video_id).is_file():
            selected.append(video_id)
            selected_set.add(video_id)
            if not all_available and len(selected) == num_videos:
                break
        else:
            missing[video_id] += 1
    return selected, selected_set, missing


def main() -> None:
    args = parse_args()
    if args.num_videos <= 0:
        raise ValueError("--num_videos must be positive.")
    for name, value in (("--num_live_videos", args.num_live_videos),
                        ("--num_egoexolearn_videos", args.num_egoexolearn_videos)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when supplied.")
    if (args.annotations is None) != (args.video_root is None):
        raise ValueError("--annotations and --video_root must be supplied together.")
    if args.annotations is not None:
        if not args.annotations.is_file():
            raise FileNotFoundError(args.annotations)
        if not args.video_root.is_dir():
            raise NotADirectoryError(args.video_root)
    if (args.ego_annotations is None) != (args.egoexolearn_video_root is None):
        raise ValueError("--ego_annotations and --egoexolearn_video_root must be supplied together.")
    if args.ego_annotations is not None:
        if not args.ego_annotations.is_file():
            raise FileNotFoundError(args.ego_annotations)
        if not args.egoexolearn_video_root.is_dir():
            raise NotADirectoryError(args.egoexolearn_video_root)

    sources = []
    if args.annotations is not None:
        sources.append(("live_whisperx", args.annotations, args.video_root,
                        args.num_live_videos or args.num_videos))
    if args.ego_annotations is not None:
        sources.append(("egoexolearn", args.ego_annotations, args.egoexolearn_video_root,
                        args.num_egoexolearn_videos or args.num_videos))
    if not sources:
        raise ValueError("Configure Live (--annotations/--video_root), Ego, or both.")

    selected_by_source, missing_by_source, cap_by_source = {}, {}, {}
    for dataset, annotation_path, video_root, cap in sources:
        selected, selected_set, missing = select_source(
            annotation_path, video_root, dataset, args.all_available, cap
        )
        if not selected:
            raise RuntimeError(f"No locally available {dataset} videos matched its annotations.")
        selected_by_source[dataset] = selected_set
        missing_by_source[dataset] = missing
        cap_by_source[dataset] = cap

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = Counter()
    tag_records = any(dataset == "egoexolearn" for dataset, *_ in sources)
    with args.output.open("w", encoding="utf-8") as handle:
        for dataset, annotation_path, _, _ in sources:
            selected = selected_by_source[dataset]
            for line_number, record in iter_records(annotation_path):
                if get_video_id(record, line_number) in selected:
                    item = tagged(record, dataset) if tag_records else record
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                    written[dataset] += 1

    if args.manifest_output is not None:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest_output.open("w", encoding="utf-8") as handle:
            for dataset, _, _, _ in sources:
                for video_id in sorted(selected_by_source[dataset]):
                    handle.write(f"{dataset}\t{video_id}\n")

    for dataset, _, _, _ in sources:
        selected_count = len(selected_by_source[dataset])
        cap_desc = "all available" if args.all_available else f"requested {cap_by_source[dataset]}"
        print(f"{dataset}: selected {selected_count} ({cap_desc}); written records={written[dataset]}; "
              f"missing IDs encountered={len(missing_by_source[dataset])}")
    print(f"Output: {args.output}")
    if args.manifest_output is not None:
        print(f"Manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
