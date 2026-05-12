"""Standalone evaluator for a saved gaze MLP checkpoint.

Loads the best checkpoint produced by ``gaze.train`` and reports
classification metrics on the held-out test split, plus a coarser
on-road / off-road binary view (matches the thesis Objective 2 metric).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .config import FEATURES_NPZ, FEATURES_NPZ_V2, GAZE_ZONES, ON_ROAD_ZONES, RUNS_DIR
from .dataset import (
    GazeDataset,
    build_delta_features,
    build_temporal_features,
    load_arrays,
    make_subject_splits,
)
from .model import GazeMLP


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=RUNS_DIR / "gaze_mlp" / "best.pt")
    ap.add_argument("--features-npz", type=Path, default=None,
                    help="override feature cache path (default: auto-pick v1/v2 by ckpt)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    is_v2 = ckpt.get("version") == "v2"
    use_deltas = bool(ckpt.get("use_deltas", False))

    if args.features_npz is not None:
        npz_path = args.features_npz
    else:
        npz_path = FEATURES_NPZ_V2 if is_v2 else FEATURES_NPZ
    print(f"loading features from {npz_path}  (v2={is_v2}, deltas={use_deltas})")

    arr = load_arrays(npz_path)
    splits = make_subject_splits(arr)
    if use_deltas:
        feats = build_delta_features(arr)
    else:
        window = int(ckpt.get("temporal_window", 1))
        feats = build_temporal_features(arr, window)
    feats = ((feats - ckpt["mean"]) / ckpt["std"]).astype(np.float32)
    test = GazeDataset(arr, splits["test"], features=feats)

    model = GazeMLP(
        in_dim=ckpt["in_dim"],
        hidden=tuple(ckpt.get("hidden", (64, 64))),
        dropout=float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.inference_mode():
        x = test.x.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
    y = test.y.numpy()

    print("=== 9-class gaze zones ===")
    print(classification_report(y, preds, target_names=list(GAZE_ZONES), zero_division=0))
    print("confusion matrix:")
    print(confusion_matrix(y, preds))

    on_road_idx = {GAZE_ZONES.index(z) for z in ON_ROAD_ZONES}
    y_bin = np.array([1 if int(v) in on_road_idx else 0 for v in y])
    p_bin = np.array([1 if int(v) in on_road_idx else 0 for v in preds])
    print("\n=== binary on-road vs off-road (thesis Objective 2) ===")
    print(classification_report(
        y_bin, p_bin, target_names=["off_road", "on_road"], zero_division=0,
    ))


if __name__ == "__main__":
    main()
