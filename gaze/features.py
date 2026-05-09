"""View-invariant feature extraction with MediaPipe FaceLandmarker.

For each frame we produce a fixed-length float32 vector composed of
features that are continuous and invariant to translation, scale, and
moderate camera placement changes:

    * 6 numbers     : first two columns of the head rotation matrix
                      (the "6D rotation" representation; continuous,
                      no gimbal lock, no PnP needed - we read the
                      ``facial_transformation_matrix`` MediaPipe
                      already computes internally).
    * 4 numbers     : iris-relative-to-eye-corner gaze vectors
                      (x, y for each eye), normalised by eye width.
    * 10 numbers    : selected eye-related blendshape activations
                      (look in/out/up/down for both eyes + blink).

Total = 20 features. All values live in roughly [-1, 1] without any
further normalisation, which keeps the downstream MLP small and stable.

The extractor is *deliberately* PnP-free: thesis discussion noted past
issues with jittery Euler angles. MediaPipe's facial-transformation
matrix is solved jointly with the landmarks and is far more stable than
a per-frame ``cv2.solvePnP`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import FEATURES, FeatureConfig

# MediaPipe FaceMesh landmark indices (468-point topology).
# Eye corners follow Google's official annotation map.
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_IRIS_CENTER = 468

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_IRIS_CENTER = 473

FEATURE_DIM = 6 + 4 + 10  # rotation (6) + iris (4) + blendshapes (10) = 20


@dataclass
class FrameFeatures:
    vec: np.ndarray  # shape (FEATURE_DIM,), float32
    valid: bool      # False if no face detected
    score: float     # face presence score, 0 if invalid
    blendshapes: dict[str, float] | None = None  # full activation map (live use)
    landmarks_xy: np.ndarray | None = None  # (468, 2) normalised [0,1], live use
    rotation: np.ndarray | None = None      # (3, 3) head rotation matrix


class FaceFeatureExtractor:
    """Wraps MediaPipe FaceLandmarker for offline batch use (VIDEO mode).

    Use one instance per video so timestamps stay monotonic.
    """

    def __init__(self, cfg: FeatureConfig = FEATURES) -> None:
        # Imported lazily so importing the module is cheap.
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not cfg.landmarker_task.is_file():
            raise FileNotFoundError(
                f"FaceLandmarker task not found at {cfg.landmarker_task}. "
                "Download it before extracting features."
            )

        base = mp_python.BaseOptions(model_asset_path=str(cfg.landmarker_task))
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=cfg.min_face_score,
            min_face_presence_confidence=cfg.min_face_score,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        self._blendshape_names = cfg.blendshape_names

    # -- main API --------------------------------------------------------

    def extract(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FrameFeatures:
        """Run landmarker on one BGR frame and return the feature vector."""
        import mediapipe as mp

        # MediaPipe expects RGB, not BGR.
        rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return FrameFeatures(
                np.zeros(FEATURE_DIM, dtype=np.float32), False, 0.0,
                None, None, None,
            )

        landmarks = result.face_landmarks[0]
        matrix = result.facial_transformation_matrixes[0]
        blendshapes = result.face_blendshapes[0]

        rot6d = _rotation_6d(matrix)
        iris = _iris_features(landmarks)
        bsh = _selected_blendshapes(blendshapes, self._blendshape_names)

        vec = np.concatenate([rot6d, iris, bsh]).astype(np.float32, copy=False)
        bsh_map = {b.category_name: float(b.score) for b in blendshapes}
        lm_xy = np.array([[p.x, p.y] for p in landmarks], dtype=np.float32)
        rot = np.asarray(matrix, dtype=np.float32)[:3, :3]
        return FrameFeatures(vec, True, 1.0, bsh_map, lm_xy, rot)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceFeatureExtractor":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


# --- pure-numpy helpers -------------------------------------------------

def _rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """Take MediaPipe's 4x4 transform, return first two columns of R.

    The 6D rotation representation (Zhou et al. 2019) is continuous and
    avoids gimbal lock and quaternion sign ambiguity, which is exactly
    what we need to feed into a small MLP.
    """
    m = np.asarray(matrix, dtype=np.float32)
    r = m[:3, :3]
    # Columns 0 and 1 of the rotation matrix.
    return np.concatenate([r[:, 0], r[:, 1]])


def _iris_features(landmarks) -> np.ndarray:
    """Iris position relative to the eye box, normalised per eye.

    Returns ``[gx_left, gy_left, gx_right, gy_right]``. Each component
    lives roughly in [-1, 1] - 0 means the iris is centred in its eye.
    """
    def lm(idx: int) -> np.ndarray:
        p = landmarks[idx]
        return np.array([p.x, p.y], dtype=np.float32)

    feats = np.zeros(4, dtype=np.float32)
    for i, (iris, inner, outer, top, bottom) in enumerate([
        (LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER,
         LEFT_EYE_TOP, LEFT_EYE_BOTTOM),
        (RIGHT_IRIS_CENTER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
         RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM),
    ]):
        iris_p = lm(iris)
        eye_center = 0.5 * (lm(inner) + lm(outer))
        width = float(np.linalg.norm(lm(outer) - lm(inner)) + 1e-6)
        height = float(np.linalg.norm(lm(top) - lm(bottom)) + 1e-6)
        delta = iris_p - eye_center
        feats[2 * i + 0] = 2.0 * delta[0] / width   # gaze_x
        feats[2 * i + 1] = 2.0 * delta[1] / height  # gaze_y
    return feats


def _selected_blendshapes(blendshape_list, names: tuple[str, ...]) -> np.ndarray:
    """Pull the configured blendshape activations in deterministic order."""
    by_name = {b.category_name: b.score for b in blendshape_list}
    return np.array([by_name.get(n, 0.0) for n in names], dtype=np.float32)
