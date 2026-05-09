"""Compute and save the per-class feature means used for calibration.

Run once after training::

    python -m gaze.compute_class_means

This produces ``models/gaze_class_means.npz`` containing the mean
feature vector of each class on the *training* split. The "front" class
mean is the population reference an individual user is calibrated
against at deployment - we subtract ``user_front_mean - dataset_front_mean``
from every live feature vector to remove user-specific bias.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import GAZE_ZONES, MODELS_DIR, TRAIN
from .dataset import load_arrays, make_subject_splits


def main() -> None:
    arr = load_arrays()
    splits = make_subject_splits(arr, TRAIN)
    train_idx = splits["train"]

    means = np.zeros((len(GAZE_ZONES), arr.features.shape[1]), dtype=np.float32)
    counts = np.zeros(len(GAZE_ZONES), dtype=np.int64)
    for c in range(len(GAZE_ZONES)):
        sel = train_idx[arr.labels[train_idx] == c]
        if sel.size == 0:
            continue
        means[c] = arr.features[sel].mean(axis=0)
        counts[c] = int(sel.size)

    out = MODELS_DIR / "gaze_class_means.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, means=means, counts=counts, labels=np.array(GAZE_ZONES))
    print(f"saved {out}")
    for name, n in zip(GAZE_ZONES, counts):
        print(f"  {name:<16} n={n}")


if __name__ == "__main__":
    main()
