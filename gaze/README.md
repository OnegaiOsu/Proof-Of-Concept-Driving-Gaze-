# Gaze pipeline

VisDrive's gaze-zone classifier: a tiny MLP trained on view-invariant
MediaPipe features extracted from the
[DMD gaze-estimation subset](https://dmd.vicomtech.org/), designed to
run real-time on a Raspberry Pi 5.

## Why this design

Pixel-based CNNs over-fit to backgrounds and lighting and are too heavy
for an embedded target. We instead feed only **geometric features** that
MediaPipe FaceLandmarker already computes:

* **6D rotation** (first two columns of the head rotation matrix) -
  continuous, no gimbal lock, no `solvePnP` jitter.
* **Iris-relative-to-eye-corner** vectors normalised by eye width -
  the actual gaze signal, invariant to face position and scale.
* **Eye blendshapes** (look in/out/up/down + blink for both eyes) -
  Google trained these to be camera-agnostic by design.

= 20 floats per frame. The MLP on top is ~5K parameters and runs in
sub-millisecond time on the Pi. Everything before it is one MediaPipe
pass, which dominates the budget at ~15-25 ms / frame on Pi 5.

## Pipeline

```
Dataset/DMD/dmd/g{A,B,C}/*/s6/*_rgb_ann_gaze.json
         │
         │  gaze.dmd_parser
         ▼
data_cache/dmd_gaze_manifest.csv      one row per labelled frame
         │
         │  gaze.extract_features    (MediaPipe FaceLandmarker)
         ▼
data_cache/dmd_gaze_features.npz      features (N, 20), labels, etc.
         │
         │  gaze.train               (PyTorch, GPU)
         ▼
runs/gaze_mlp/best.pt                  best checkpoint by val macro-F1
         │
         │  gaze.export_onnx
         ▼
models/gaze_mlp.onnx                   ready for Raspberry Pi
```

## Run from scratch

```powershell
# 1. Build the manifest from DMD JSONs (~1 s).
.venv\Scripts\python.exe -m gaze.dmd_parser

# 2. Extract features for every labelled frame (~10 min on a laptop).
#    Override frame stride with --stride; use --limit-sessions for a smoke test.
.venv\Scripts\python.exe -m gaze.extract_features

# 3. Train the MLP on GPU (a few minutes).
.venv\Scripts\python.exe -m gaze.train

# 4. Re-evaluate the best checkpoint (subject-disjoint test split).
.venv\Scripts\python.exe -m gaze.evaluate

# 5. Export ONNX for the Pi.
.venv\Scripts\python.exe -m gaze.export_onnx
```

## Splits

We split **by subject (group, subject_id)**, never by frame. By default:

* test  = group `gC` (subjects 11-15)
* val   = 15 % of subjects from `gA` + `gB`, randomly picked with seed 42
* train = the remainder

This avoids the classic mistake of training and testing on the same
person in different frames, which inflates accuracy by 10-30 points.

## Robustness to camera placement

Because every feature is either pose- or scale-relative, the model
generalises across moderate camera mount differences. Two extra knobs
in `gaze.config`:

* `INFER.off_road_seconds` - the temporal threshold (default 1.5 s)
  used to debounce alerts (Thesis Objective 3).
* `TRAIN.ema_alpha` - smoothing alpha applied at inference to the
  feature vector before classification.

For deployment we also recommend a 5-second neutral calibration:
record mean feature vector while the driver looks straight, then
subtract it as a per-driver bias. That eliminates the residual offset
caused by mount differences and seat position.

## Drowsiness

The drowsiness side of VisDrive is rule-based (PERCLOS + yawn ratio
from the same MediaPipe pass) and lives outside this package. Trained
gaze classification + hardcoded drowsiness rules together cover both
thesis Objectives 1-3 with a single MediaPipe inference per frame.
