"""Tiny MLP for gaze-zone classification.

Sized for Raspberry Pi 5 inference: ~5K parameters total. Trains in
seconds on the RTX 5070 Ti, runs in <1 ms on the Pi via ONNX Runtime.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import GAZE_ZONES, TRAIN, TrainConfig
from .features import FEATURE_DIM


class GazeMLP(nn.Module):
    def __init__(
        self,
        in_dim: int = FEATURE_DIM,
        n_classes: int = len(GAZE_ZONES),
        hidden: tuple[int, ...] | None = None,
        dropout: float | None = None,
        cfg: TrainConfig = TRAIN,
    ) -> None:
        super().__init__()
        h_dims = tuple(hidden) if hidden is not None else cfg.hidden
        drop = cfg.dropout if dropout is None else dropout
        layers: list[nn.Module] = []
        last = in_dim
        for h in h_dims:
            layers += [nn.Linear(last, h), nn.GELU(), nn.Dropout(drop)]
            last = h
        layers.append(nn.Linear(last, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
