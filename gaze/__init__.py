"""VisDrive gaze classification pipeline.

Trains a tiny MLP on view-invariant features (MediaPipe FaceLandmarker
blendshapes + iris-relative geometry + 6D rotation) extracted from the
DMD gaze-estimation dataset. Designed to run real-time on Raspberry Pi 5.
"""
