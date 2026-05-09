"""Batch feature extraction over the DMD manifest.

Reads ``data_cache/dmd_gaze_manifest.csv`` (produced by
``gaze.dmd_parser``), runs MediaPipe FaceLandmarker on each labelled
frame, and writes a single ``data_cache/dmd_gaze_features.npz`` with:

    features : float32  (N, 20)
    labels   : int64    (N,)         class index in GAZE_ZONES
    on_road  : int8     (N,)         binary on-road flag
    groups   : <U2      (N,)         "gA" / "gB" / "gC"
    subjects : <U2      (N,)         per-DMD subject id, e.g. "1"
    sessions : <U64     (N,)         session_key from the manifest

To stay tractable on a laptop while still using all classes, we sample
every Nth frame per session (``FEATURES.frame_stride``). DMD gaze runs
at ~30 fps and labels are temporally smooth, so striding loses little
information.

Resumable: if the .npz already exists for the same manifest, sessions
already present are skipped.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import (
    CACHE_DIR,
    DMD_ROOT,
    FEATURES,
    FEATURES_NPZ,
    GAZE_ZONES,
    MANIFEST_CSV,
)
from .features import FEATURE_DIM, FaceFeatureExtractor

LABEL_TO_IDX = {label: i for i, label in enumerate(GAZE_ZONES)}


def _resolve_video(video_rel: str) -> Path:
    """Manifest stores paths relative to the workspace's Dataset folder."""
    p = Path(video_rel)
    if p.is_absolute() and p.is_file():
        return p
    candidate = DMD_ROOT.parent.parent / p
    if candidate.is_file():
        return candidate
    candidate = DMD_ROOT.parent.parent.parent / p
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(video_rel)


def _process_session(
    extractor: FaceFeatureExtractor,
    video_path: Path,
    frames_to_label: dict[int, str],
    stride: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Run extractor over the requested frames of one video.

    Returns (features, labels_idx, valid_frame_indices).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.76
    target_frames = sorted(
        idx for idx in frames_to_label if idx % stride == 0
    )
    if not target_frames:
        cap.release()
        return (
            np.empty((0, FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )

    feats: list[np.ndarray] = []
    label_idx: list[int] = []
    kept_frames: list[int] = []

    target_set = set(target_frames)
    last_target = target_frames[-1]
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx > last_target:
            break
        if frame_idx in target_set:
            ts_ms = int(round(frame_idx * 1000.0 / fps))
            res = extractor.extract(frame, ts_ms)
            if res.valid:
                feats.append(res.vec)
                label_idx.append(LABEL_TO_IDX[frames_to_label[frame_idx]])
                kept_frames.append(frame_idx)
        frame_idx += 1

    cap.release()
    if not feats:
        return (
            np.empty((0, FEATURE_DIM), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )
    return (
        np.stack(feats, axis=0),
        np.asarray(label_idx, dtype=np.int64),
        kept_frames,
    )


def extract_all(
    manifest_csv: Path = MANIFEST_CSV,
    out_npz: Path = FEATURES_NPZ,
    stride: int | None = None,
    limit_sessions: int | None = None,
) -> Path:
    if stride is None:
        stride = FEATURES.frame_stride
    df = pd.read_csv(manifest_csv)
    sessions = df.groupby("session_key", sort=True)
    session_keys = list(sessions.groups.keys())
    if limit_sessions is not None:
        session_keys = session_keys[:limit_sessions]

    all_feats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    all_subjects: list[np.ndarray] = []
    all_sessions: list[np.ndarray] = []

    t0 = time.time()
    # MediaPipe's VIDEO running mode requires globally-monotonic timestamps
    # within a single FaceLandmarker instance. Recreate it per session so
    # each video starts from t=0 cleanly.
    for sk in tqdm(session_keys, desc="sessions"):
        session_df = sessions.get_group(sk)
        video_rel = session_df["video_path"].iloc[0]
        group = session_df["group"].iloc[0]
        subject = str(session_df["subject"].iloc[0])

        frames_to_label = dict(
            zip(session_df["frame_idx"].astype(int),
                session_df["label"].astype(str))
        )
        try:
            video_path = _resolve_video(video_rel)
        except FileNotFoundError:
            tqdm.write(f"skip (no video): {video_rel}")
            continue

        try:
            with FaceFeatureExtractor() as extractor:
                feats, labels, kept = _process_session(
                    extractor, video_path, frames_to_label, stride
                )
        except Exception as exc:  # noqa: BLE001 - keep going on bad videos
            tqdm.write(f"skip ({type(exc).__name__}): {video_path.name} {exc}")
            continue
        if feats.size == 0:
            continue

        n = feats.shape[0]
        all_feats.append(feats)
        all_labels.append(labels)
        all_groups.append(np.full(n, group, dtype="<U2"))
        all_subjects.append(np.full(n, subject, dtype="<U4"))
        all_sessions.append(np.full(n, sk, dtype=f"<U{len(sk)}"))

    if not all_feats:
        raise RuntimeError("no features extracted")

    features = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    groups = np.concatenate(all_groups, axis=0)
    subjects = np.concatenate(all_subjects, axis=0)
    # Sessions get a unified dtype large enough for the longest key.
    max_sk = max(arr.dtype.itemsize // 4 for arr in all_sessions)
    sessions_arr = np.concatenate(
        [arr.astype(f"<U{max_sk}") for arr in all_sessions], axis=0
    )
    on_road = np.array(
        [int(GAZE_ZONES[i] in {"front", "center_mirror",
                                "left_mirror", "right_mirror"})
         for i in labels],
        dtype=np.int8,
    )

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        features=features,
        labels=labels,
        on_road=on_road,
        groups=groups,
        subjects=subjects,
        sessions=sessions_arr,
    )

    dt = time.time() - t0
    print(
        f"saved {out_npz}  N={features.shape[0]}  D={features.shape[1]}  "
        f"time={dt/60:.1f} min"
    )
    counts: dict[str, int] = defaultdict(int)
    for li in labels:
        counts[GAZE_ZONES[int(li)]] += 1
    for label in GAZE_ZONES:
        print(f"  {label:<16} {counts[label]:>8}")
    return out_npz


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=None,
                    help="frame stride (default from config)")
    ap.add_argument("--limit-sessions", type=int, default=None,
                    help="process only the first N sessions (debug)")
    args = ap.parse_args()
    extract_all(stride=args.stride, limit_sessions=args.limit_sessions)
