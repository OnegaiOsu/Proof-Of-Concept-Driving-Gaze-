"""PyTorch dataset over the cached DMD gaze features.

The dataset is intentionally trivial - all features fit comfortably in
RAM as float32 (~5 MB for ~250k samples). Splitting is **subject-aware**
to avoid identity leakage between train / val / test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FEATURES_NPZ, GAZE_ZONES, TRAIN, TrainConfig


@dataclass
class GazeArrays:
    features: np.ndarray  # (N, D) float32
    labels: np.ndarray    # (N,) int64
    groups: np.ndarray    # (N,) "<U2"
    subjects: np.ndarray  # (N,) "<U4"
    sessions: np.ndarray  # (N,) <U?


def load_arrays(path: Path = FEATURES_NPZ) -> GazeArrays:
    data = np.load(path, allow_pickle=False)
    return GazeArrays(
        features=data["features"].astype(np.float32, copy=False),
        labels=data["labels"].astype(np.int64, copy=False),
        groups=data["groups"],
        subjects=data["subjects"],
        sessions=data["sessions"],
    )


def make_subject_splits(
    arr: GazeArrays, cfg: TrainConfig = TRAIN
) -> dict[str, np.ndarray]:
    """Return dict of split_name -> sample indices, split by *subject*.

    Test = all subjects in ``cfg.test_groups``. From the remaining
    groups we randomly hold out ``cfg.val_fraction`` of subjects for
    validation. Everyone else is train.
    """
    rng = np.random.default_rng(cfg.seed)

    # Unique (group, subject) tuples preserve identity even if subject
    # ids repeat across groups.
    pair = np.char.add(np.char.add(arr.groups, b"_" if arr.groups.dtype.kind == "S" else "_"), arr.subjects)
    test_mask = np.isin(arr.groups, np.array(cfg.test_groups))
    test_idx = np.flatnonzero(test_mask)

    other_pairs = np.unique(pair[~test_mask])
    rng.shuffle(other_pairs)
    n_val = max(1, int(round(len(other_pairs) * cfg.val_fraction)))
    val_pairs = set(other_pairs[:n_val].tolist())

    val_mask = np.array([p in val_pairs for p in pair]) & ~test_mask
    train_mask = ~test_mask & ~val_mask

    return {
        "train": np.flatnonzero(train_mask),
        "val": np.flatnonzero(val_mask),
        "test": test_idx,
    }


def build_temporal_features(arr: GazeArrays, window: int) -> np.ndarray:
    """Return ``(N, window * D)`` array. Each row is the concatenation of
    the previous ``window - 1`` per-frame feature vectors and the current
    one, clamped at session boundaries (the first frame of a session is
    repeated as needed). Row order matches ``arr.features`` so existing
    split indices remain valid.
    """
    feats = arr.features
    sessions = arr.sessions
    n, d = feats.shape
    if window <= 1:
        return feats.copy()

    # First-row index of each contiguous session run.
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = sessions[1:] != sessions[:-1]
    starts = np.flatnonzero(change)
    first_of_session = np.empty(n, dtype=np.int64)
    j = 0
    for i in range(n):
        if j + 1 < len(starts) and i >= starts[j + 1]:
            j += 1
        first_of_session[i] = starts[j]

    out = np.empty((n, window * d), dtype=np.float32)
    arange = np.arange(n, dtype=np.int64)
    for off in range(window):
        lag = window - 1 - off  # off=window-1 -> current frame
        idx = np.maximum(arange - lag, first_of_session)
        out[:, off * d:(off + 1) * d] = feats[idx]
    return out


def build_delta_features(arr: GazeArrays) -> np.ndarray:
    """Return ``(N, 2 * D)`` array: [v_t | v_t - v_{t-1}].

    The delta channel encodes per-frame motion (gaze velocity) and is
    clamped to zero at session boundaries so cross-session leakage is
    impossible.
    """
    feats = arr.features
    sessions = arr.sessions
    n, d = feats.shape
    if n == 0:
        return feats.copy()

    # First-row index of each contiguous session run.
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = sessions[1:] != sessions[:-1]
    prev_idx = np.arange(n, dtype=np.int64) - 1
    prev_idx[change] = np.flatnonzero(change)  # clamp at session start
    deltas = (feats - feats[prev_idx]).astype(np.float32)

    return np.concatenate([feats, deltas], axis=1)


class GazeDataset(Dataset):
    def __init__(
        self,
        arr: GazeArrays,
        indices: np.ndarray,
        features: np.ndarray | None = None,
    ) -> None:
        src = arr.features if features is None else features
        self.x = torch.from_numpy(src[indices])
        self.y = torch.from_numpy(arr.labels[indices])

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]


def class_weights(arr: GazeArrays, indices: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights for cross-entropy."""
    counts = np.bincount(arr.labels[indices], minlength=len(GAZE_ZONES))
    counts = np.maximum(counts, 1)
    inv = counts.sum() / (len(GAZE_ZONES) * counts)
    return torch.tensor(inv, dtype=torch.float32)
