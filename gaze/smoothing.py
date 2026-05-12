"""Streaming smoothers used at inference time.

EMA is the cheapest stable choice and runs in microseconds on a Pi.
The 1-Euro filter is included as a slightly smarter alternative for
deployment - it adapts smoothing strength to the speed of motion,
giving low lag on fast head turns and heavy smoothing when still.
"""

from __future__ import annotations

import math

import numpy as np


class EMA:
    """Exponential moving average over a fixed-length feature vector."""

    def __init__(self, alpha: float, dim: int) -> None:
        self.alpha = float(alpha)
        self._state: np.ndarray | None = None
        self._dim = dim

    def reset(self) -> None:
        self._state = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._state is None:
            self._state = x.astype(np.float32, copy=True)
        else:
            self._state = self.alpha * x + (1.0 - self.alpha) * self._state
        return self._state


class OneEuroFilter:
    """Casiez et al. 2012 - low-lag smoothing for noisy 1-D signals."""

    def __init__(
        self,
        freq: float,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, freq: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float) -> float:
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat


class HysteresisCounter:
    """Trip when ``cond`` holds for >= ``frames_required`` consecutive frames."""

    def __init__(self, frames_required: int) -> None:
        self.frames_required = int(frames_required)
        self._count = 0
        self._tripped = False

    def update(self, cond: bool) -> bool:
        if cond:
            self._count += 1
        else:
            self._count = 0
            self._tripped = False
        if self._count >= self.frames_required:
            self._tripped = True
        return self._tripped

    def reset(self) -> None:
        self._count = 0
        self._tripped = False


class TimedHysteresisCounter:
    """Trip when ``cond`` holds for >= ``seconds_required`` wall-clock seconds.

    Frame-count hysteresis silently stretches to 2-3x real time when the
    capture/inference loop runs below the configured target FPS. This
    counter accumulates the per-frame dt so the trip threshold is always
    in real seconds.
    """

    def __init__(self, seconds_required: float) -> None:
        self.seconds_required = float(seconds_required)
        self._elapsed = 0.0
        self._tripped = False

    def update(self, cond: bool, dt: float) -> bool:
        if cond:
            self._elapsed += max(0.0, float(dt))
        else:
            self._elapsed = 0.0
            self._tripped = False
        if self._elapsed >= self.seconds_required:
            self._tripped = True
        return self._tripped

    @property
    def elapsed(self) -> float:
        return self._elapsed

    def reset(self) -> None:
        self._elapsed = 0.0
        self._tripped = False
