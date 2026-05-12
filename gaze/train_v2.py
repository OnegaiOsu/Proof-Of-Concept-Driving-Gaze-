"""Safe v2 retrain — does NOT overwrite the v1 model.

This orchestrator runs the full v2 pipeline end-to-end:

  1. Build a fresh manifest covering BOTH RGB and IR face videos.
  2. Extract features with the wider v2 blendshape set.
  3. Optionally append per-frame delta features.
  4. Train a new GazeMLP and save it under ``runs/gaze_mlp_v2/`` and
     ``models/gaze_mlp_v2.onnx`` (the v1 artefacts in ``runs/gaze_mlp``
     and ``models/gaze_mlp.onnx`` are untouched).

Usage:
    python -m gaze.train_v2                 # full pipeline (extract + train)
    python -m gaze.train_v2 --skip-extract  # reuse cached v2 features
    python -m gaze.train_v2 --no-deltas     # train without delta features
    python -m gaze.train_v2 --rgb-only      # ablation: drop IR

Compared to v1 this run differs in three independent ways:
  * IR augmentation:           +~65k frames (≈2× data)
  * Wider blendshapes:         10 → 14 (adds eyeSquint, browDown L/R)
  * Delta features:            in_dim doubles (current + velocity)
  * front_right reclassified as on-road (binary metric improvement only)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .config import (
    FEATURES_NPZ_V2,
    FEATURES_V2,
    GAZE_ZONES,
    MANIFEST_CSV_V2,
    MODELS_DIR,
    ON_ROAD_ZONES,
    TRAIN,
    V2_RUN_DIR,
)
from .dataset import (
    GazeDataset,
    build_delta_features,
    class_weights,
    load_arrays,
    make_subject_splits,
)
from .dmd_parser import build_manifest
from .extract_features import extract_all
from .model import GazeMLP


def _normalise(features: np.ndarray, train_idx: np.ndarray):
    mean = features[train_idx].mean(axis=0).astype(np.float32)
    std = features[train_idx].std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    normed = ((features - mean) / std).astype(np.float32)
    return normed, mean, std


def _epoch(model, loader, criterion, optim, device, train: bool):
    model.train(train)
    total = 0
    correct = 0
    loss_sum = 0.0
    all_y, all_p = [], []
    ctx = torch.enable_grad() if train else torch.inference_mode()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
            loss_sum += float(loss.item()) * int(y.numel())
            all_y.append(y.detach().cpu().numpy())
            all_p.append(preds.detach().cpu().numpy())
    y_true = np.concatenate(all_y) if all_y else np.empty(0)
    y_pred = np.concatenate(all_p) if all_p else np.empty(0)
    macro_f1 = (
        f1_score(y_true, y_pred, average="macro", zero_division=0)
        if total else 0.0
    )
    return loss_sum / max(total, 1), correct / max(total, 1), macro_f1, y_true, y_pred


def run_extraction(
    modalities: tuple[str, ...],
    manifest_csv: Path = MANIFEST_CSV_V2,
    out_npz: Path = FEATURES_NPZ_V2,
) -> Path:
    print(f"[v2] building manifest with modalities={modalities} -> {manifest_csv.name}")
    build_manifest(out_csv=manifest_csv, modalities=modalities)
    print(f"[v2] extracting features with {len(FEATURES_V2.blendshape_names)} "
          f"blendshapes → {out_npz.name}")
    extract_all(
        manifest_csv=manifest_csv,
        out_npz=out_npz,
        cfg=FEATURES_V2,
    )
    return out_npz


def train(
    use_deltas: bool,
    epochs: int,
    dropout: float | None = None,
    weight_decay: float | None = None,
    label_smoothing: float = 0.0,
    patience: int = 0,
    features_npz: Path = FEATURES_NPZ_V2,
    run_dir: Path = V2_RUN_DIR,
) -> dict:
    cfg = TRAIN
    dropout = cfg.dropout if dropout is None else dropout
    weight_decay = cfg.weight_decay if weight_decay is None else weight_decay
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[v2] device: {device}")
    print(f"[v2] dropout={dropout}  weight_decay={weight_decay}  "
          f"label_smoothing={label_smoothing}  patience={patience}")

    arr = load_arrays(features_npz)
    splits = make_subject_splits(arr, cfg)
    print({k: int(v.size) for k, v in splits.items()})

    if use_deltas:
        feats = build_delta_features(arr)
        print(f"[v2] delta features ON  per_frame_dim={arr.features.shape[1]}  "
              f"in_dim={feats.shape[1]}")
    else:
        feats = arr.features.copy()
        print(f"[v2] delta features OFF  in_dim={feats.shape[1]}")

    feats, mean, std = _normalise(feats, splits["train"])
    in_dim = feats.shape[1]

    train_ds = GazeDataset(arr, splits["train"], features=feats)
    val_ds = GazeDataset(arr, splits["val"], features=feats)
    test_ds = GazeDataset(arr, splits["test"], features=feats)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = GazeMLP(in_dim=in_dim, hidden=cfg.hidden, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v2] model params: {n_params}")

    weights = class_weights(arr, splits["train"]).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights, label_smoothing=label_smoothing
    )
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    best_f1 = -1.0
    best_epoch = 0
    epochs_since_best = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, *_ = _epoch(model, train_loader, criterion, optim, device, train=True)
        va_loss, va_acc, va_f1, *_ = _epoch(model, val_loader, criterion, optim, device, train=False)
        sched.step()
        dt = time.time() - t0
        improved = va_f1 > best_f1
        if improved:
            best_f1 = va_f1
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "labels": list(GAZE_ZONES),
                    "in_dim": in_dim,
                    "hidden": list(cfg.hidden),
                    "dropout": dropout,
                    "temporal_window": 1,
                    "per_frame_dim": arr.features.shape[1],
                    "use_deltas": use_deltas,
                    "blendshape_names": list(FEATURES_V2.blendshape_names),
                    "version": "v2",
                },
                best_path,
            )
        else:
            epochs_since_best += 1
        print(
            f"epoch {epoch:02d}  "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f}  |  "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}  "
            f"{'*' if improved else ' '}  {dt:.1f}s"
        )
        if patience > 0 and epochs_since_best >= patience:
            print(f"[v2] early stop at epoch {epoch} "
                  f"(no val improvement for {patience} epochs; best epoch {best_epoch})")
            break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    _, te_acc, te_f1, y_true, y_pred = _epoch(
        model, test_loader, criterion, optim, device, train=False
    )

    print(f"\n[v2] TEST  acc {te_acc:.3f}  macro-F1 {te_f1:.3f}")
    print("\n=== 9-class report ===")
    print(classification_report(y_true, y_pred, target_names=list(GAZE_ZONES), zero_division=0))
    print("confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    on_road_idx = {GAZE_ZONES.index(z) for z in ON_ROAD_ZONES}
    y_bin = np.array([1 if int(v) in on_road_idx else 0 for v in y_true])
    p_bin = np.array([1 if int(v) in on_road_idx else 0 for v in y_pred])
    bin_acc = float((y_bin == p_bin).mean()) if y_bin.size else 0.0
    print(f"\n=== binary on-road vs off-road (front_right is on-road) ===")
    print(f"binary acc: {bin_acc:.3f}")
    print(classification_report(
        y_bin, p_bin, target_names=["off_road", "on_road"], zero_division=0,
    ))

    metrics = {
        "test_acc": te_acc,
        "test_macro_f1": te_f1,
        "test_binary_acc": bin_acc,
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "n_params": n_params,
        "in_dim": in_dim,
        "use_deltas": use_deltas,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "patience": patience,
        "on_road_zones": sorted(ON_ROAD_ZONES),
        "blendshape_names": list(FEATURES_V2.blendshape_names),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\n[v2] saved checkpoint → {best_path}")
    print(f"[v2] v1 artefacts left untouched.")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-extract", action="store_true",
                    help="reuse existing v2 features cache")
    ap.add_argument("--rgb-only", action="store_true",
                    help="ablation: skip IR videos")
    ap.add_argument("--no-deltas", action="store_true",
                    help="train without delta features")
    ap.add_argument("--epochs", type=int, default=TRAIN.epochs)
    ap.add_argument("--dropout", type=float, default=0.4,
                    help="MLP dropout (v2 default 0.4 for stronger regularisation)")
    ap.add_argument("--weight-decay", type=float, default=1e-3,
                    help="AdamW weight decay (v2 default 1e-3)")
    ap.add_argument("--label-smoothing", type=float, default=0.1,
                    help="cross-entropy label smoothing (v2 default 0.1)")
    ap.add_argument("--patience", type=int, default=5,
                    help="early stop after N epochs without val macro-F1 gain (0=off)")
    ap.add_argument("--features", type=Path, default=FEATURES_NPZ_V2,
                    help="features cache .npz to use (and write if extracting)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_CSV_V2,
                    help="manifest CSV to build/use")
    ap.add_argument("--run-dir", type=Path, default=V2_RUN_DIR,
                    help="directory to write best.pt + metrics.json")
    args = ap.parse_args()

    if not args.skip_extract:
        modalities = ("rgb",) if args.rgb_only else ("rgb", "ir")
        run_extraction(modalities, manifest_csv=args.manifest, out_npz=args.features)
    else:
        if not args.features.is_file():
            raise FileNotFoundError(
                f"{args.features} missing; drop --skip-extract."
            )
        print(f"[v2] reusing cached features at {args.features}")

    train(
        use_deltas=not args.no_deltas,
        epochs=args.epochs,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        features_npz=args.features,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    main()
