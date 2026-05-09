"""Per-user calibration for the live gaze classifier.

A single 5-second neutral pose ("look straight at the road") is enough
to remove most of the cross-subject bias that hurts our test metric.

Mechanics
---------

At training time we cached the mean feature vector of every class on
the training split (``gaze.compute_class_means``). The ``front`` class
mean is the population's "looking at the road" centroid in feature
space.

During calibration we collect the user's own feature vectors while
they stare at the road for ``seconds`` seconds, average them, and
compute::

    offset = user_front_mean - dataset_front_mean

At runtime every incoming feature vector is corrected by::

    vec_calibrated = vec - offset

This is a translation in feature space - cheap, reversible, and
mathematically benign for the iris and blendshape components. The
6D rotation block is *not* a Euclidean quantity but for the small
person-to-person posture deltas we see in DMD-style seated drivers
the linear correction is a good first-order fix.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import GAZE_ZONES, MODELS_DIR


CALIBRATION_PATH = MODELS_DIR / "user_calibration.npz"
CLASS_MEANS_PATH = MODELS_DIR / "gaze_class_means.npz"


@dataclass
class Calibration:
    offset: np.ndarray            # (D,) float32, subtract from raw feature vec
    n_frames: int                 # how many frames went into the user mean
    user_front_mean: np.ndarray   # (D,) float32
    dataset_front_mean: np.ndarray  # (D,) float32

    def apply(self, vec: np.ndarray) -> np.ndarray:
        return (vec - self.offset).astype(np.float32, copy=False)

    def save(self, path: Path = CALIBRATION_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            offset=self.offset,
            n_frames=np.int64(self.n_frames),
            user_front_mean=self.user_front_mean,
            dataset_front_mean=self.dataset_front_mean,
        )

    @classmethod
    def load(cls, path: Path = CALIBRATION_PATH) -> "Calibration":
        data = np.load(path, allow_pickle=False)
        return cls(
            offset=data["offset"].astype(np.float32, copy=False),
            n_frames=int(data["n_frames"]),
            user_front_mean=data["user_front_mean"].astype(np.float32, copy=False),
            dataset_front_mean=data["dataset_front_mean"].astype(np.float32, copy=False),
        )


def load_dataset_front_mean(path: Path = CLASS_MEANS_PATH) -> np.ndarray:
    """Return the population mean feature vector of the ``front`` class."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m gaze.compute_class_means` first."
        )
    data = np.load(path, allow_pickle=False)
    means = data["means"]
    front_idx = GAZE_ZONES.index("front")
    return means[front_idx].astype(np.float32, copy=False)


def identity(dim: int) -> Calibration:
    z = np.zeros(dim, dtype=np.float32)
    return Calibration(offset=z, n_frames=0, user_front_mean=z, dataset_front_mean=z)
