"""Parse DMD gaze annotations (ASAM OpenLABEL JSON) into a flat manifest.

The DMD gaze JSON stores temporal actions of the form
``gaze_zone/<label>`` with explicit frame_start / frame_end intervals.
We expand those intervals into a per-frame label table and pair each
session with its co-located ``rgb_face.mp4`` (the single-camera face
view we will feed to MediaPipe).

The output manifest is a single CSV with one row per labelled frame:

    group,subject,session_key,video_path,frame_idx,label,on_road

This is intentionally simple so the feature-extraction stage can be
parallelised by row and resumed cleanly.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path

from .config import DMD_ROOT, GAZE_ZONES, INVALID_LABELS, MANIFEST_CSV, ON_ROAD_ZONES

GAZE_PREFIX = "gaze_zone/"


def _iter_gaze_jsons(root: Path) -> Iterator[Path]:
    """Yield every ``*_rgb_ann_gaze.json`` under the DMD root."""
    yield from sorted(root.glob("g*/*/s6/*_rgb_ann_gaze.json"))


def _face_video_for(json_path: Path) -> Path | None:
    """Return the matching rgb_face.mp4 for a given gaze annotation file."""
    stem = json_path.name.replace("_rgb_ann_gaze.json", "_rgb_face.mp4")
    candidate = json_path.parent / stem
    return candidate if candidate.is_file() else None


def _expand_intervals(annotation: dict) -> dict[int, str]:
    """Map frame_idx -> gaze-zone label for one OpenLABEL document."""
    root = annotation.get("openlabel", annotation)
    actions = root.get("actions", {}) or {}
    frame_to_label: dict[int, str] = {}
    for action in actions.values():
        action_type = action.get("type", "")
        if not action_type.startswith(GAZE_PREFIX):
            continue
        label = action_type[len(GAZE_PREFIX):]
        if label in INVALID_LABELS or label not in GAZE_ZONES:
            continue
        for interval in action.get("frame_intervals", []) or []:
            start = int(interval["frame_start"])
            end = int(interval["frame_end"])
            for frame_idx in range(start, end + 1):
                # Last writer wins; intervals should not overlap for gaze
                # but we keep the behaviour deterministic anyway.
                frame_to_label[frame_idx] = label
    return frame_to_label


def _session_key(json_path: Path) -> tuple[str, str, str]:
    """(group, subject, session_key) derived from filename + path."""
    # Filename: gA_1_s6_2019-03-08T09;15;15+01;00_rgb_ann_gaze.json
    # Path:     .../gA/1/s6/<file>
    parts = json_path.name.split("_")
    group, subject = parts[0], parts[1]
    session_key = json_path.stem.replace("_rgb_ann_gaze", "")
    return group, subject, session_key


def build_manifest(
    dmd_root: Path = DMD_ROOT, out_csv: Path = MANIFEST_CSV
) -> Path:
    """Walk DMD, expand gaze intervals, and write the manifest CSV."""
    rows = 0
    skipped_no_video = 0
    per_label: dict[str, int] = {label: 0 for label in GAZE_ZONES}

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "group", "subject", "session_key",
            "video_path", "frame_idx", "label", "on_road",
        ])

        for json_path in _iter_gaze_jsons(dmd_root):
            video = _face_video_for(json_path)
            if video is None:
                skipped_no_video += 1
                continue

            with json_path.open("r", encoding="utf-8") as jf:
                doc = json.load(jf)
            frame_to_label = _expand_intervals(doc)
            if not frame_to_label:
                continue

            group, subject, session_key = _session_key(json_path)
            video_rel = video.relative_to(dmd_root.parent.parent.parent).as_posix()

            for frame_idx, label in sorted(frame_to_label.items()):
                writer.writerow([
                    group, subject, session_key,
                    video_rel, frame_idx, label,
                    int(label in ON_ROAD_ZONES),
                ])
                rows += 1
                per_label[label] += 1

    print(f"manifest: {out_csv}  rows={rows}  skipped_no_video={skipped_no_video}")
    for label in GAZE_ZONES:
        print(f"  {label:<16} {per_label[label]:>8}")
    return out_csv


if __name__ == "__main__":
    build_manifest()
