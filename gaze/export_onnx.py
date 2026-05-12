"""Export the trained MLP to ONNX for Raspberry Pi 5 deployment.

Bakes the feature normaliser (mean / std) into the graph so the Pi
runtime only has to feed raw 20-D feature vectors. Output graph runs
in <1 ms with onnxruntime on the Pi 5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import GAZE_ZONES, MODELS_DIR, RUNS_DIR
from .features import FEATURE_DIM
from .model import GazeMLP


class _NormalisedGazeModel(nn.Module):
    def __init__(self, mlp: nn.Module, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.mlp = mlp
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp((x - self.mean) / self.std)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=RUNS_DIR / "gaze_mlp" / "best.pt")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "gaze_mlp.onnx")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    in_dim = int(ckpt["in_dim"])
    mlp = GazeMLP(
        in_dim=in_dim,
        hidden=tuple(ckpt.get("hidden", (64, 64))),
        dropout=float(ckpt.get("dropout", 0.0)),
    )
    mlp.load_state_dict(ckpt["state_dict"])
    mlp.eval()

    mean = torch.tensor(ckpt["mean"], dtype=torch.float32)
    std = torch.tensor(ckpt["std"], dtype=torch.float32)
    full = _NormalisedGazeModel(mlp, mean, std).eval()

    dummy = torch.zeros(1, in_dim, dtype=torch.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        full,
        dummy,
        args.out,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    meta = {
        "labels": list(GAZE_ZONES),
        "feature_dim": in_dim,
        "per_frame_dim": int(ckpt.get("per_frame_dim", FEATURE_DIM)),
        "temporal_window": int(ckpt.get("temporal_window", 1)),
        "use_deltas": bool(ckpt.get("use_deltas", False)),
        "blendshape_names": list(ckpt.get("blendshape_names", [])),
        "version": str(ckpt.get("version", "v1")),
        "input": f"features  (B, {in_dim})  float32",
        "output": "logits    (B, 9)   float32  (argmax -> labels[i])",
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    # Sanity check the export.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    raw = np.random.randn(4, in_dim).astype(np.float32)
    out = sess.run(None, {"features": raw})[0]
    print(f"exported {args.out}  out shape={out.shape}")


if __name__ == "__main__":
    main()
