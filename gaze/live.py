"""Live inference: webcam -> MediaPipe -> ONNX gaze head + drowsiness rules.

This is the runtime that will be packaged onto the Raspberry Pi 5 in
deployment. It is deliberately written against ``onnxruntime`` (CPU
provider) and a single MediaPipe pass per frame so the laptop demo and
the Pi build use exactly the same code path.

Pipeline per frame:

    1. Capture BGR frame from webcam.
    2. ``FaceFeatureExtractor.extract`` -> 20-D feature vec + full
       blendshape activation map + raw landmarks + head rotation.
    3. Subtract per-user calibration offset, then EMA smooth.
    4. ONNX gaze classifier -> softmax -> argmax -> zone label.
    5. Hysteresis counter on "off-road" predictions trips an alert
       after ``InferenceConfig.off_road_seconds`` of sustained gaze
       away from the road (Thesis Objective 2).
    6. Drowsiness rules over blendshapes (Thesis Objective 3):
         - microsleep: mean(eyeBlinkL, eyeBlinkR) > eye_closed_thr
           sustained for ``microsleep_seconds``.
         - yawn: jawOpen > yawn_thr sustained for ``yawn_seconds``.
    7. HUD overlay with calibrate / reset buttons, "model view" panel
       (3x3 zone map), top-3 probability bars, head-pose triad,
       per-frame latencies, and session-level event counters.

Run with::

    python -m gaze.live                          # default webcam 0
    python -m gaze.live --camera 1 --mirror      # alt webcam, mirrored
    python -m gaze.live --no-display             # headless smoke test
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibration import (
    CALIBRATION_PATH,
    Calibration,
    identity,
    load_dataset_front_mean,
)
from .config import (
    FEATURES,
    FEATURES_V2,
    FeatureConfig,
    GAZE_ZONES,
    INFER,
    MODELS_DIR,
    TRAIN,
    is_on_road,
)
from .features import FEATURE_DIM, FaceFeatureExtractor
from .overlay import (
    Button,
    Counter,
    draw_landmarks,
    draw_metrics_panel,
    draw_pose_axes,
    draw_top_probs,
    draw_zone_panel,
    put,
    softmax,
)
from .smoothing import EMA, HysteresisCounter


WINDOW_NAME = "VisDrive - live"


# --- drowsiness thresholds ---------------------------------------------

@dataclass(frozen=True)
class DrowsinessConfig:
    eye_closed_thr: float = 0.5      # blendshape activation
    microsleep_seconds: float = 0.4  # eyes closed continuously => microsleep
    yawn_thr: float = 0.5
    yawn_seconds: float = 1.5
    perclos_window_seconds: float = 60.0
    perclos_alert: float = 0.20      # >20% closed-eye fraction over 60 s
    blink_min_seconds: float = 0.10  # short closed -> just a normal blink


# --- helpers ------------------------------------------------------------

def _bsh(extras: dict[str, float] | None, name: str) -> float:
    if not extras:
        return 0.0
    return float(extras.get(name, 0.0))


def _load_session(onnx_path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"]
    )


def _load_model_meta(onnx_path: Path) -> dict:
    """Read the sidecar JSON next to an ONNX file (written by export_onnx)."""
    import json as _json

    meta_path = onnx_path.with_suffix(".json")
    if not meta_path.is_file():
        return {}
    try:
        return _json.loads(meta_path.read_text())
    except Exception:
        return {}


def _feature_config_for(meta: dict) -> FeatureConfig:
    """Pick the FeatureConfig that matches a model's blendshape set."""
    names = tuple(meta.get("blendshape_names") or ())
    if not names:
        return FEATURES
    if names == FEATURES_V2.blendshape_names:
        return FEATURES_V2
    if names == FEATURES.blendshape_names:
        return FEATURES
    # Custom set: construct a matching config so the extractor outputs
    # exactly what the model was trained on.
    return FeatureConfig(blendshape_names=names)


def _calibrate(
    cap: cv2.VideoCapture,
    extractor: FaceFeatureExtractor,
    seconds: float,
    fps: int,
    mirror: bool,
    display: bool,
    t_start: float,
) -> Calibration:
    """Run a 5-second neutral-pose calibration and return a Calibration.

    The user is asked to look straight at the (virtual) road during a
    short countdown plus capture window. Frames where no face is
    detected are skipped. We require at least 30 valid frames before
    accepting; otherwise we fall back to identity (no correction).
    """
    dataset_front = load_dataset_front_mean()
    target_frames = max(30, int(round(seconds * fps)))
    samples: list[np.ndarray] = []

    countdown_until = time.monotonic() + 3.0
    capture_until: float | None = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)

        now = time.monotonic()
        ts_ms = int(round((now - t_start) * 1000.0))
        ff = extractor.extract(frame_bgr, ts_ms)

        in_countdown = now < countdown_until
        if not in_countdown and capture_until is None:
            capture_until = now + seconds

        if not in_countdown and ff.valid:
            samples.append(ff.vec.copy())

        if display:
            hud = frame_bgr
            if in_countdown:
                remaining = countdown_until - now
                put(hud, "CALIBRATION", (16, 40), (0, 200, 255), 1.0, 2)
                put(hud, "Look straight at the road", (16, 80),
                    (255, 255, 255), 0.8, 2)
                put(hud, f"Starting in {remaining:.1f}s", (16, 120),
                    (255, 255, 255), 0.8, 2)
                put(hud, "Press 's' to skip", (16, 160),
                    (180, 180, 180), 0.6, 1)
            else:
                remaining = max(0.0, (capture_until or now) - now)
                put(hud, "CALIBRATION - hold still", (16, 40),
                    (0, 200, 255), 1.0, 2)
                put(hud, f"Capturing... {remaining:.1f}s left", (16, 80),
                    (255, 255, 255), 0.8, 2)
                put(hud, f"valid frames: {len(samples)}/{target_frames}",
                    (16, 120), (255, 255, 255), 0.7, 2)
                if not ff.valid:
                    put(hud, "NO FACE - move into frame", (16, 160),
                        (0, 0, 255), 0.8, 2)
            cv2.imshow(WINDOW_NAME, hud)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                print("calibration skipped by user")
                return identity(extractor.feature_dim)
            if key == ord("q"):
                raise KeyboardInterrupt

        if capture_until is not None and now >= capture_until:
            break
        if len(samples) >= target_frames * 2:  # safety
            break

    if len(samples) < 30:
        print(f"calibration: only {len(samples)} valid frames - skipping correction")
        return identity(extractor.feature_dim)

    user_front = np.mean(np.stack(samples, axis=0), axis=0).astype(np.float32)
    offset = (user_front - dataset_front).astype(np.float32)
    print(f"calibration: {len(samples)} frames, |offset|={np.linalg.norm(offset):.4f}")
    return Calibration(
        offset=offset,
        n_frames=len(samples),
        user_front_mean=user_front,
        dataset_front_mean=dataset_front,
    )


# --- main loop ----------------------------------------------------------

def run(
    camera: int = 1,
    onnx_path: Path = MODELS_DIR / "gaze_mlp.onnx",
    mirror: bool = False,
    display: bool = True,
    max_frames: int | None = None,
    calibrate: bool = True,
    calibration_seconds: float = 5.0,
    load_calibration: Path | None = None,
    save_calibration: bool = True,
) -> None:
    drowsy = DrowsinessConfig()
    fps = INFER.fps_target

    sess = _load_session(onnx_path)
    in_name = sess.get_inputs()[0].name
    meta = _load_model_meta(onnx_path)
    feat_cfg = _feature_config_for(meta)
    per_frame_dim = int(meta.get("per_frame_dim", feat_cfg.feature_dim))
    model_in_dim = int(meta.get("feature_dim", per_frame_dim))
    use_deltas = bool(meta.get("use_deltas", model_in_dim == 2 * per_frame_dim))
    print(f"live: model={onnx_path.name}  version={meta.get('version', 'v1')}  "
          f"per_frame_dim={per_frame_dim}  in_dim={model_in_dim}  "
          f"deltas={'ON' if use_deltas else 'OFF'}")

    extractor = FaceFeatureExtractor(cfg=feat_cfg)
    ema = EMA(alpha=TRAIN.ema_alpha, dim=per_frame_dim)
    prev_vec: np.ndarray | None = None  # for delta features

    off_road_hyst = HysteresisCounter(INFER.off_road_frames)
    microsleep_hyst = HysteresisCounter(max(1, int(round(drowsy.microsleep_seconds * fps))))
    yawn_hyst = HysteresisCounter(max(1, int(round(drowsy.yawn_seconds * fps))))

    perclos_window = deque(maxlen=int(round(drowsy.perclos_window_seconds * fps)))

    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    t_start = time.monotonic()
    fps_window: deque[float] = deque(maxlen=30)
    mp_lat_ms: deque[float] = deque(maxlen=60)
    nn_lat_ms: deque[float] = deque(maxlen=60)
    total_lat_ms: deque[float] = deque(maxlen=60)
    last_frame_time = time.monotonic()
    frame_count = 0

    # Edge-triggered session counters.
    off_road_events = Counter()
    microsleep_events = Counter()
    yawn_events = Counter()
    blink_hyst = HysteresisCounter(max(1, int(round(drowsy.blink_min_seconds * fps))))
    blink_events = Counter()
    off_road_seconds_total = 0.0  # cumulative seconds with off-road alert active

    # --- calibration ----------------------------------------------------
    if load_calibration is not None and load_calibration.is_file():
        cal = Calibration.load(load_calibration)
        print(f"loaded calibration from {load_calibration} "
              f"(n={cal.n_frames}, |offset|={np.linalg.norm(cal.offset):.4f})")
    elif calibrate:
        cal = _calibrate(
            cap, extractor, calibration_seconds, fps,
            mirror=mirror, display=display, t_start=t_start,
        )
        if save_calibration and cal.n_frames > 0:
            cal.save(CALIBRATION_PATH)
            print(f"saved calibration to {CALIBRATION_PATH}")
    else:
        cal = identity(per_frame_dim)

    # --- buttons + mouse input -----------------------------------------
    btn_calibrate = Button("CALIBRATE", (1280 - 16 - 140, 12, 140, 36),
                           color=(60, 130, 200))
    btn_reset = Button("RESET", (1280 - 16 - 140 - 100, 12, 90, 36),
                       color=(110, 110, 110))
    pending_action: dict[str, bool] = {"calibrate": False, "reset": False}

    def _mouse(event, x, y, flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if btn_calibrate.hit(x, y):
            pending_action["calibrate"] = True
            btn_calibrate.flash(time.monotonic())
        elif btn_reset.hit(x, y):
            pending_action["reset"] = True
            btn_reset.flash(time.monotonic())

    if display:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, _mouse)

    print(f"live: gaze={onnx_path.name}  off_road_frames={INFER.off_road_frames}  "
          f"microsleep_frames={microsleep_hyst.frames_required}  "
          f"yawn_frames={yawn_hyst.frames_required}")
    print("buttons: CALIBRATE / RESET   keys: q=quit r=reset c=recalibrate")

    def _do_reset() -> None:
        nonlocal prev_vec
        ema.reset()
        prev_vec = None
        off_road_hyst.reset()
        microsleep_hyst.reset()
        yawn_hyst.reset()
        blink_hyst.reset()
        perclos_window.clear()
        off_road_events.reset()
        microsleep_events.reset()
        yawn_events.reset()
        blink_events.reset()
        nonlocal_metrics["session_t0"] = time.monotonic()
        nonlocal_metrics["off_road_seconds_total"] = 0.0

    nonlocal_metrics: dict[str, float] = {
        "session_t0": time.monotonic(),
        "off_road_seconds_total": 0.0,
    }

    try:
        while True:
            t_iter0 = time.monotonic()
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            now = time.monotonic()
            dt = now - last_frame_time
            last_frame_time = now
            fps_window.append(1.0 / dt if dt > 0 else 0.0)
            ts_ms = int(round((now - t_start) * 1000.0))

            t_mp0 = time.monotonic()
            ff = extractor.extract(frame_bgr, ts_ms)
            mp_lat_ms.append((time.monotonic() - t_mp0) * 1000.0)

            zone: str = "<no face>"
            on_road = True
            eye_closed = False
            yawning = False
            probs: np.ndarray | None = None

            if ff.valid:
                vec = ema(cal.apply(ff.vec))
                if use_deltas:
                    delta = vec - prev_vec if prev_vec is not None else np.zeros_like(vec)
                    prev_vec = vec
                    model_in = np.concatenate([vec, delta]).astype(np.float32)
                else:
                    model_in = vec.astype(np.float32)
                t_nn0 = time.monotonic()
                logits = sess.run(None, {in_name: model_in[None, :]})[0]
                nn_lat_ms.append((time.monotonic() - t_nn0) * 1000.0)
                probs = softmax(logits[0])
                cls_idx = int(np.argmax(probs))
                zone = GAZE_ZONES[cls_idx]
                on_road = is_on_road(zone)

                eye_avg = 0.5 * (_bsh(ff.blendshapes, "eyeBlinkLeft")
                                 + _bsh(ff.blendshapes, "eyeBlinkRight"))
                eye_closed = eye_avg > drowsy.eye_closed_thr
                jaw_open = _bsh(ff.blendshapes, "jawOpen")
                yawning = jaw_open > drowsy.yawn_thr
            else:
                ema.reset()
                prev_vec = None

            perclos_window.append(1 if eye_closed else 0)
            perclos = sum(perclos_window) / max(1, len(perclos_window))

            off_road_alert = off_road_hyst.update(ff.valid and not on_road)
            microsleep_alert = microsleep_hyst.update(eye_closed)
            yawn_alert = yawn_hyst.update(yawning)
            perclos_alert = perclos > drowsy.perclos_alert and len(perclos_window) > fps
            blink_flag = blink_hyst.update(eye_closed)

            # Edge-triggered counters.
            off_road_events.update(off_road_alert)
            microsleep_events.update(microsleep_alert)
            yawn_events.update(yawn_alert)
            # Blink = short eye-closure that did NOT escalate to microsleep.
            blink_events.update(blink_flag and not microsleep_alert)

            if off_road_alert:
                nonlocal_metrics["off_road_seconds_total"] += dt

            total_lat_ms.append((time.monotonic() - t_iter0) * 1000.0)

            # ---------------- HUD ----------------
            if display:
                hud = frame_bgr

                if ff.valid and ff.landmarks_xy is not None:
                    draw_landmarks(hud, ff.landmarks_xy)
                    if ff.rotation is not None:
                        draw_pose_axes(hud, ff.landmarks_xy, ff.rotation)

                # Top-left status.
                color_zone = (0, 220, 0) if on_road else (0, 165, 255)
                put(hud, f"gaze: {zone}", (16, 32), color_zone, 0.8, 2)
                put(hud, f"on-road: {'yes' if on_road else 'NO'}", (16, 60), color_zone)
                put(hud, f"PERCLOS: {perclos*100:5.1f}%", (16, 84),
                    (0, 0, 255) if perclos_alert else (255, 255, 255))
                put(hud, f"FPS: {np.mean(fps_window):4.1f}", (16, 108))
                cal_tag = (f"calibrated (n={cal.n_frames})"
                           if cal.n_frames > 0 else "no calibration")
                put(hud, cal_tag, (16, 132), (180, 180, 180), 0.5, 1)

                # Active alerts directly under the status block.
                alerts: list[str] = []
                if off_road_alert:
                    alerts.append("OFF-ROAD >1.5s")
                if microsleep_alert:
                    alerts.append("MICROSLEEP")
                if yawn_alert:
                    alerts.append("YAWN")
                if perclos_alert:
                    alerts.append("DROWSY (PERCLOS)")
                if not ff.valid:
                    alerts.append("NO FACE")
                for i, msg in enumerate(alerts):
                    put(hud, msg, (16, 162 + 28 * i), (0, 0, 255), 0.8, 2)

                # Right column: model view, probabilities, latencies, counters.
                col_x = 1280 - 16 - 240
                draw_zone_panel(hud, probs, zone if ff.valid else None,
                                (col_x, 60))
                draw_top_probs(hud, probs, (col_x, 60 + 220))

                session_t = now - nonlocal_metrics["session_t0"]
                blink_rate = (
                    blink_events.count / max(session_t / 60.0, 1e-6)
                    if session_t > 5.0 else 0.0
                )
                latency_rows: list[tuple[str, str]] = [
                    ("MediaPipe ms", f"{np.mean(mp_lat_ms):5.1f}" if mp_lat_ms else "-"),
                    ("ONNX ms",      f"{np.mean(nn_lat_ms):5.1f}" if nn_lat_ms else "-"),
                    ("frame ms",     f"{np.mean(total_lat_ms):5.1f}" if total_lat_ms else "-"),
                    ("eye blink",    f"{(_bsh(ff.blendshapes, 'eyeBlinkLeft') + _bsh(ff.blendshapes, 'eyeBlinkRight'))/2:.2f}"),
                    ("jaw open",     f"{_bsh(ff.blendshapes, 'jawOpen'):.2f}"),
                ]
                draw_metrics_panel(hud, (col_x, 60 + 220 + 110), 240,
                                   latency_rows, title="latency / signals")

                metric_rows: list[tuple[str, str]] = [
                    ("session",        f"{session_t:6.1f}s"),
                    ("off-road events", f"{off_road_events.count}"),
                    ("off-road time",  f"{nonlocal_metrics['off_road_seconds_total']:5.1f}s"),
                    ("microsleeps",    f"{microsleep_events.count}"),
                    ("yawns",          f"{yawn_events.count}"),
                    ("blinks",         f"{blink_events.count}"),
                    ("blink/min",      f"{blink_rate:5.1f}"),
                ]
                draw_metrics_panel(hud, (col_x, 60 + 220 + 110 + 130), 240,
                                   metric_rows, title="session metrics")

                btn_calibrate.draw(hud, now)
                btn_reset.draw(hud, now)

                cv2.imshow(WINDOW_NAME, hud)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    pending_action["reset"] = True
                if key == ord("c"):
                    pending_action["calibrate"] = True

                if pending_action["reset"]:
                    pending_action["reset"] = False
                    _do_reset()
                if pending_action["calibrate"]:
                    pending_action["calibrate"] = False
                    cal = _calibrate(
                        cap, extractor, calibration_seconds, fps,
                        mirror=mirror, display=display, t_start=t_start,
                    )
                    if save_calibration and cal.n_frames > 0:
                        cal.save(CALIBRATION_PATH)
                    _do_reset()

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        extractor.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--onnx", type=Path, default=MODELS_DIR / "gaze_mlp.onnx")
    ap.add_argument("--mirror", action="store_true",
                    help="flip frame horizontally (selfie webcam)")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after N frames (smoke test)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip the per-user neutral-pose calibration")
    ap.add_argument("--calibration-seconds", type=float, default=5.0)
    ap.add_argument("--load-calibration", type=Path, default=None,
                    help="reuse a saved calibration .npz instead of capturing")
    ap.add_argument("--no-save-calibration", action="store_true",
                    help="don't persist a freshly-captured calibration")
    args = ap.parse_args()

    run(
        camera=args.camera,
        onnx_path=args.onnx,
        mirror=args.mirror,
        display=not args.no_display,
        max_frames=args.max_frames,
        calibrate=not args.no_calibrate,
        calibration_seconds=args.calibration_seconds,
        load_calibration=args.load_calibration,
        save_calibration=not args.no_save_calibration,
    )


if __name__ == "__main__":
    main()

