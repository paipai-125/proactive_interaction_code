#!/usr/bin/env python3
"""Extract complete MMDuet2 JSONL records for a fixed number of local videos."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator


def iter_records(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            yield line_number, record


def video_path(video_root: Path, video_id: str) -> Path:
    """Resolve the Live-WhisperX layout without a recursive filesystem scan."""
    relative = Path(video_id + ".mp4")
    candidates = (video_root / "videos" / relative, video_root / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select complete MMDuet2 JSONL records for N locally available videos."
    )
    parser.add_argument("--annotations", required=True, type=Path,
                        help="Original MMDuet2 Live SFT JSONL.")
    parser.add_argument("--video_root", required=True, type=Path,
                        help="Root containing videos/<metadata.video_id>.mp4.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output JSONL containing complete selected records.")
    parser.add_argument("--num_videos", type=int, default=20,
                        help="Number of distinct local videos to select (default: 20).")
    parser.add_argument("--manifest_output", type=Path, default=None,
                        help="Optional text file containing selected video IDs, one per line.")
    return parser.parse_args()


def get_video_id(record: dict, line_number: int) -> str:
    metadata = record.get("metadata")
    video_id = metadata.get("video_id") if isinstance(metadata, dict) else None
    if not isinstance(video_id, str) or not video_id:
        raise ValueError(f"Line {line_number} has no metadata.video_id.")
    return video_id


def main() -> None:
    args = parse_args()
    if args.num_videos <= 0:
        raise ValueError("--num_videos must be positive.")
    if not args.annotations.is_file():
        raise FileNotFoundError(args.annotations)
    if not args.video_root.is_dir():
        raise NotADirectoryError(args.video_root)

    # Select source-file-order videos in pass 1; then retain every complete
    # matching trajectory in pass 2, so no multi-turn record is truncated.
    selected: list[str] = []
    selected_set: set[str] = set()
    missing = Counter()
    for line_number, record in iter_records(args.annotations):
        video_id = get_video_id(record, line_number)
        if video_id in selected_set:
            continue
        if video_path(args.video_root, video_id).is_file():
            selected.append(video_id)
            selected_set.add(video_id)
            if len(selected) == args.num_videos:
                break
        else:
            missing[video_id] += 1

    if not selected:
        raise RuntimeError("No annotation video IDs had a locally available MP4.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written_records = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for line_number, record in iter_records(args.annotations):
            if get_video_id(record, line_number) in selected_set:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written_records += 1

    if args.manifest_output is not None:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text("\n".join(selected) + "\n", encoding="utf-8")

    print(f"Selected videos: {len(selected)} / requested {args.num_videos}")
    print(f"Written complete JSONL records: {written_records}")
    print(f"Output: {args.output}")
    if args.manifest_output is not None:
        print(f"Manifest: {args.manifest_output}")
    if len(selected) < args.num_videos:
        print(f"Warning: only {len(selected)} local videos were found; "
              f"encountered {len(missing)} unavailable video IDs before EOF.")


if __name__ == "__main__":
    main()
