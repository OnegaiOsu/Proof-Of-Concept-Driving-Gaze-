"""HUD overlay primitives for the live demo.

Kept separate from ``live.py`` to keep the main loop short. All
functions take a BGR image and draw in place (with a few that return
the modified image for chaining). Drawings are deliberately cheap -
under 1 ms total on the Pi 5 for the full HUD.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from .config import GAZE_ZONES, ON_ROAD_ZONES


# Outline ring of FaceMesh vertices around each eye (subset; we only
# need enough points to read a clear contour).
LEFT_EYE_RING = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
RIGHT_EYE_RING = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
LEFT_IRIS = (468, 469, 470, 471, 472)
RIGHT_IRIS = (473, 474, 475, 476, 477)
NOSE_TIP = 1
CHIN = 152

# Spatial layout of the 9 gaze zones in the dashboard reference frame
# (row, col) in a 3x3 grid. Used by ``draw_zone_panel``.
ZONE_GRID: dict[str, tuple[int, int]] = {
    "left_mirror":   (0, 0),
    "front":         (0, 1),
    "right_mirror":  (0, 2),
    "left":          (1, 0),
    "center_mirror": (1, 1),
    "front_right":   (1, 2),
    "steering_wheel":(2, 0),
    "infotainment":  (2, 1),
    "right":         (2, 2),
}


def put(img, text, org, color=(255, 255, 255), scale=0.6, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


# --- face overlay -------------------------------------------------------

def draw_landmarks(
    img: np.ndarray,
    lm_xy: np.ndarray,
) -> None:
    """Draw eye contours + iris centres + nose tip + chin marker."""
    h, w = img.shape[:2]

    def to_px(idxs: Iterable[int]) -> np.ndarray:
        pts = lm_xy[list(idxs)]
        return (pts * np.array([w, h], dtype=np.float32)).astype(np.int32)

    cv2.polylines(img, [to_px(LEFT_EYE_RING)], True, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.polylines(img, [to_px(RIGHT_EYE_RING)], True, (180, 220, 255), 1, cv2.LINE_AA)

    # Iris centroids.
    for ring in (LEFT_IRIS, RIGHT_IRIS):
        pts = to_px(ring)
        cx, cy = pts.mean(axis=0).astype(np.int32)
        cv2.circle(img, (int(cx), int(cy)), 3, (0, 255, 255), -1, cv2.LINE_AA)

    nose = to_px([NOSE_TIP])[0]
    cv2.circle(img, tuple(nose.tolist()), 3, (255, 0, 255), -1, cv2.LINE_AA)


def draw_pose_axes(
    img: np.ndarray,
    lm_xy: np.ndarray,
    rot: np.ndarray,
    length: int = 70,
) -> None:
    """Draw a 3-axis triad anchored at the nose tip from the head rotation
    matrix. X (right) red, Y (up) green, Z (forward) blue. Z is the most
    informative axis for head-pose readability.
    """
    h, w = img.shape[:2]
    origin = (lm_xy[NOSE_TIP] * np.array([w, h], dtype=np.float32)).astype(np.int32)
    # MediaPipe rotation maps model-space axes into the image. The first
    # row of R applied to a unit vector yields its image-x projection.
    axes_world = np.eye(3, dtype=np.float32) * length
    proj = rot @ axes_world  # (3, 3) - columns are projected axes
    for i, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 128, 0)]):
        # Image y is inverted relative to world y.
        end = (
            int(origin[0] + proj[0, i]),
            int(origin[1] - proj[1, i]),
        )
        cv2.arrowedLine(img, tuple(origin.tolist()), end, color, 2,
                        cv2.LINE_AA, tipLength=0.2)


# --- side panels --------------------------------------------------------

def draw_zone_panel(
    img: np.ndarray,
    probs: np.ndarray | None,
    pred: str | None,
    org: tuple[int, int],
    cell: int = 56,
) -> None:
    """Mini 3x3 gaze-zone map. Cells coloured by probability, predicted
    cell outlined, on-road cells get a green border.
    """
    x0, y0 = org
    panel_w = cell * 3 + 16
    panel_h = cell * 3 + 36
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (90, 90, 90), 1)
    put(img, "model view", (x0 + 6, y0 + 18), (200, 200, 200), 0.5, 1)

    for name, (r, c) in ZONE_GRID.items():
        cx0 = x0 + 8 + c * cell
        cy0 = y0 + 28 + r * cell
        cx1 = cx0 + cell - 4
        cy1 = cy0 + cell - 4
        if probs is not None:
            p = float(probs[GAZE_ZONES.index(name)])
            shade = int(40 + 200 * p)
            fill = (shade, shade, 60) if name in ON_ROAD_ZONES else (40, 80, shade)
        else:
            fill = (40, 40, 40)
        cv2.rectangle(img, (cx0, cy0), (cx1, cy1), fill, -1)
        border = (0, 200, 0) if name in ON_ROAD_ZONES else (60, 60, 120)
        cv2.rectangle(img, (cx0, cy0), (cx1, cy1), border, 1)
        if name == pred:
            cv2.rectangle(img, (cx0 - 1, cy0 - 1), (cx1 + 1, cy1 + 1),
                          (0, 255, 255), 2)
        # Short label.
        short = name.replace("_", " ").split()[0][:6]
        put(img, short, (cx0 + 4, cy0 + cell - 8), (230, 230, 230), 0.4, 1)


def draw_top_probs(
    img: np.ndarray,
    probs: np.ndarray | None,
    org: tuple[int, int],
    width: int = 220,
    k: int = 3,
) -> None:
    """Horizontal bar chart of top-k softmax probabilities."""
    x0, y0 = org
    line_h = 20
    total_h = line_h * k + 30
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + total_h), (0, 0, 0), -1)
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + total_h), (90, 90, 90), 1)
    put(img, "top probabilities", (x0 + 6, y0 + 18), (200, 200, 200), 0.5, 1)
    if probs is None:
        put(img, "(no face)", (x0 + 6, y0 + 40), (180, 180, 180), 0.5, 1)
        return
    top = np.argsort(probs)[::-1][:k]
    for i, idx in enumerate(top):
        name = GAZE_ZONES[int(idx)]
        p = float(probs[int(idx)])
        bar_x0 = x0 + 6
        bar_y0 = y0 + 28 + i * line_h
        bar_w = int((width - 90) * p)
        col = (0, 220, 0) if name in ON_ROAD_ZONES else (0, 165, 255)
        cv2.rectangle(img, (bar_x0, bar_y0),
                      (bar_x0 + bar_w, bar_y0 + line_h - 4), col, -1)
        put(img, f"{name}", (bar_x0 + 4, bar_y0 + 14), (255, 255, 255), 0.45, 1)
        put(img, f"{p*100:5.1f}%", (x0 + width - 60, bar_y0 + 14),
            (255, 255, 255), 0.45, 1)


# --- buttons ------------------------------------------------------------

class Button:
    """A simple rectangular button rendered on the HUD and hit-tested by
    the mouse callback. Click events go through ``Button.hit``.
    """

    def __init__(self, label: str, rect: tuple[int, int, int, int],
                 color: tuple[int, int, int] = (60, 130, 200)) -> None:
        self.label = label
        self.rect = rect
        self.color = color
        self._flash_until = 0.0

    def draw(self, img: np.ndarray, now: float) -> None:
        x, y, w, h = self.rect
        col = self.color if now >= self._flash_until else (40, 200, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
        # Centred label.
        ((tw, th), _) = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, 2)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(img, self.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, self.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def hit(self, x: int, y: int) -> bool:
        bx, by, bw, bh = self.rect
        return bx <= x <= bx + bw and by <= y <= by + bh

    def flash(self, now: float, duration: float = 0.15) -> None:
        self._flash_until = now + duration


# --- session metrics ----------------------------------------------------

class Counter:
    """Edge-triggered counter from a hysteresis flag (counts 0->1 transitions)."""

    def __init__(self) -> None:
        self.count = 0
        self._prev = False

    def update(self, flag: bool) -> None:
        if flag and not self._prev:
            self.count += 1
        self._prev = flag

    def reset(self) -> None:
        self.count = 0
        self._prev = False


def draw_metrics_panel(
    img: np.ndarray,
    org: tuple[int, int],
    width: int,
    rows: Sequence[tuple[str, str]],
    title: str = "session metrics",
) -> None:
    x0, y0 = org
    line_h = 18
    h = line_h * len(rows) + 30
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + h), (0, 0, 0), -1)
    cv2.rectangle(img, (x0, y0), (x0 + width, y0 + h), (90, 90, 90), 1)
    put(img, title, (x0 + 6, y0 + 18), (200, 200, 200), 0.5, 1)
    for i, (label, value) in enumerate(rows):
        y = y0 + 36 + i * line_h
        put(img, label, (x0 + 6, y), (180, 180, 180), 0.45, 1)
        put(img, value, (x0 + width - 6 - 8 * len(value), y),
            (255, 255, 255), 0.45, 1)
