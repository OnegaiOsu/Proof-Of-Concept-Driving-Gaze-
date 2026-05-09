"""Train the gaze-zone MLP on cached DMD features.

Run with:

    python -m gaze.train

This will:
  * Load ``data_cache/dmd_gaze_features.npz``.
  * Make subject-disjoint train/val/test splits.
  * Train ``GazeMLP`` with class-balanced cross-entropy.
  * Save the best checkpoint (by val macro-F1) to
    ``runs/gaze_mlp/best.pt`` along with feature normaliser stats and
    the label list.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from .config import GAZE_ZONES, RUNS_DIR, TRAIN
from .dataset import (
    GazeArrays,
    GazeDataset,
    build_temporal_features,
    class_weights,
    load_arrays,
    make_subject_splits,
)
from .model import GazeMLP


def _normalise(
    features: np.ndarray, train_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if total
        else 0.0
    )
    return loss_sum / max(total, 1), correct / max(total, 1), macro_f1, y_true, y_pred


def main() -> None:
    cfg = TRAIN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})")

    arr = load_arrays()
    splits = make_subject_splits(arr, cfg)
    print({k: int(v.size) for k, v in splits.items()})

    # Temporal stacking: classifier input = window * per-frame dim.
    feats = build_temporal_features(arr, cfg.temporal_window)
    feats, mean, std = _normalise(feats, splits["train"])
    in_dim = feats.shape[1]
    print(f"temporal_window={cfg.temporal_window}  in_dim={in_dim}")

    train_ds = GazeDataset(arr, splits["train"], features=feats)
    val_ds = GazeDataset(arr, splits["val"], features=feats)
    test_ds = GazeDataset(arr, splits["test"], features=feats)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = GazeMLP(in_dim=in_dim, hidden=cfg.hidden, dropout=cfg.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}")

    weights = class_weights(arr, splits["train"]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    out_dir = RUNS_DIR / "gaze_mlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_f1 = -1.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1, *_ = _epoch(model, train_loader, criterion, optim, device, train=True)
        va_loss, va_acc, va_f1, *_ = _epoch(model, val_loader, criterion, optim, device, train=False)
        sched.step()
        dt = time.time() - t0
        improved = va_f1 > best_f1
        if improved:
            best_f1 = va_f1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "labels": list(GAZE_ZONES),
                    "in_dim": in_dim,
                    "hidden": list(cfg.hidden),
                    "dropout": cfg.dropout,
                    "temporal_window": cfg.temporal_window,
                    "per_frame_dim": arr.features.shape[1],
                },
                best_path,
            )
        print(
            f"epoch {epoch:02d}  "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} f1 {tr_f1:.3f}  |  "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} f1 {va_f1:.3f}  "
            f"{'*' if improved else ' '}  {dt:.1f}s"
        )

    # Final test eval on the best checkpoint.
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    _, te_acc, te_f1, y_true, y_pred = _epoch(
        model, test_loader, criterion, optim, device, train=False
    )
    print(f"\nTEST  acc {te_acc:.3f}  macro-F1 {te_f1:.3f}")
    print(classification_report(y_true, y_pred, target_names=list(GAZE_ZONES), zero_division=0))

    metrics = {
        "test_acc": te_acc,
        "test_macro_f1": te_f1,
        "best_val_macro_f1": best_f1,
        "n_params": n_params,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
