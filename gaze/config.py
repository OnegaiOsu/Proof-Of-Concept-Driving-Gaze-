"""Central configuration for the gaze pipeline.

All paths are resolved relative to the repo root so the package works
regardless of where it is invoked from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --- Paths --------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DMD_ROOT = REPO_ROOT / "Dataset" / "DMD" / "dmd"
CACHE_DIR = REPO_ROOT / "data_cache"
RUNS_DIR = REPO_ROOT / "runs"
MODELS_DIR = REPO_ROOT / "models"

CACHE_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

MANIFEST_CSV = CACHE_DIR / "dmd_gaze_manifest.csv"
FEATURES_NPZ = CACHE_DIR / "dmd_gaze_features.npz"

# --- v2 artefacts (safe retrain — do NOT overwrite v1) ------------------
# v2 adds IR videos, wider blendshapes, optional delta features, and
# reclassifies front_right as on-road. Kept on separate paths so v1
# (the currently-deployed model) is preserved untouched.
MANIFEST_CSV_V2 = CACHE_DIR / "dmd_gaze_manifest_v2.csv"
FEATURES_NPZ_V2 = CACHE_DIR / "dmd_gaze_features_v2.npz"
V2_RUN_DIR = RUNS_DIR / "gaze_mlp_v2"

# --- Label space --------------------------------------------------------
# DMD gaze-zone labels (mutually exclusive within a frame).
GAZE_ZONES: tuple[str, ...] = (
    "left_mirror",
    "left",
    "front",
    "center_mirror",
    "front_right",
    "right_mirror",
    "right",
    "infotainment",
    "steering_wheel",
)

# Frames labelled `not_valid` or under occlusion are dropped from training.
INVALID_LABELS: frozenset[str] = frozenset({"not_valid"})

# Coarse binary mapping used by the temporal "eyes-off-road" detector
# (Thesis Objective 2). Mirrors are considered on-road glances per the
# SPIDER scanning model (Strayer & McDonnell, 2025). front_right is
# included because the right windshield region is part of the forward
# scanning arc (right A-pillar / forward-right traffic) rather than a
# distraction zone — this reclassification was validated in v2 ablation.
ON_ROAD_ZONES: frozenset[str] = frozenset({
    "front", "center_mirror", "left_mirror", "right_mirror",
    "front_right",
})


def label_to_index(label: str) -> int:
    return GAZE_ZONES.index(label)


def is_on_road(label: str) -> bool:
    return label in ON_ROAD_ZONES


# --- Feature extraction -------------------------------------------------

@dataclass(frozen=True)
class FeatureConfig:
    # Frame stride when sampling videos for training (29.76 fps source).
    # stride=3 -> ~10 fps effective, plenty for gaze and ~3x less compute.
    frame_stride: int = 3
    # Drop frames whose face detection score is below this.
    min_face_score: float = 0.5
    # MediaPipe FaceLandmarker task file (downloaded by tools/setup.py).
    landmarker_task: Path = field(
        default_factory=lambda: REPO_ROOT / "models" / "face_landmarker.task"
    )
    # Subset of MediaPipe blendshape names used as gaze features.
    # These ten describe eye direction and lid state without depending on
    # camera intrinsics or background.
    blendshape_names: tuple[str, ...] = (
        "eyeLookInLeft", "eyeLookOutLeft",
        "eyeLookUpLeft", "eyeLookDownLeft",
        "eyeLookInRight", "eyeLookOutRight",
        "eyeLookUpRight", "eyeLookDownRight",
        "eyeBlinkLeft", "eyeBlinkRight",
    )

    @property
    def feature_dim(self) -> int:
        # 6 rotation-6D + 4 iris + N blendshapes
        return 6 + 4 + len(self.blendshape_names)


# v2 widens the blendshape window with brow + squint signals. Mirror
# checks tend to engage brow and eyelid muscles in addition to gaze
# direction; including these gives the MLP more separating power
# between front_right vs center_mirror (the dominant v1 confusion).
FEATURES_V2 = FeatureConfig(
    blendshape_names=(
        "eyeLookInLeft", "eyeLookOutLeft",
        "eyeLookUpLeft", "eyeLookDownLeft",
        "eyeLookInRight", "eyeLookOutRight",
        "eyeLookUpRight", "eyeLookDownRight",
        "eyeBlinkLeft", "eyeBlinkRight",
        "eyeSquintLeft", "eyeSquintRight",
        "browDownLeft", "browDownRight",
    ),
)


# --- Training -----------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    # Subjects held out for test (group-level split prevents identity leak).
    test_groups: tuple[str, ...] = ("gC",)
    val_fraction: float = 0.15  # of remaining (gA+gB) subjects
    seed: int = 42

    batch_size: int = 256
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden: tuple[int, ...] = (64, 64)
    dropout: float = 0.2

    # Temporal context: 1 = single-frame classifier (best on this dataset).
    # Larger windows did not generalise on subject-disjoint test in our
    # ablation; we lean on inference-time hysteresis instead.
    temporal_window: int = 1

    # EMA smoothing alpha applied to features at inference time. Higher =
    # more responsive, lower = smoother. 0.3 ~= ~3-frame effective window.
    ema_alpha: float = 0.3


# --- Inference / temporal threshold (Thesis Objective 3) ----------------

@dataclass(frozen=True)
class InferenceConfig:
    fps_target: int = 30
    # Driver must be classified off-road for >= this many seconds before
    # an alert fires. Thesis specifies 1.5 - 2.0 s.
    off_road_seconds: float = 1.5
    # The v2 model was trained with frame_stride=3 on ~30 fps DMD video,
    # so its delta features encode ~100 ms gaps. Running the classifier
    # at the camera's full FPS would shrink that gap to ~33 ms and make
    # deltas dominated by head jitter. We throttle the classifier (and
    # the delta tracker) to this cadence; MediaPipe + blink/yawn signals
    # still update every frame.
    gaze_infer_hz: float = 10.0

    @property
    def off_road_frames(self) -> int:
        return int(round(self.fps_target * self.off_road_seconds))


FEATURES = FeatureConfig()
TRAIN = TrainConfig()
INFER = InferenceConfig()
