from pathlib import Path

import numpy as np

from dance_pose_coach.config import AnalysisConfig
from dance_pose_coach.models import PoseSequence
from dance_pose_coach.pose import extract_pose
from dance_pose_coach.pose_opencv import OpenCVPoseDetector, model_path, probe_opencv_pose


def _sequence(backend: str) -> PoseSequence:
    return PoseSequence(
        times=np.arange(10, dtype=np.float32),
        landmarks=np.zeros((10, 33, 4), dtype=np.float32),
        source_fps=30,
        analyzed_fps=10,
        low_confidence_ratio=0,
        backend=backend,
    )


def test_bundled_cpu_model_is_present_and_loadable():
    assert model_path().is_file()
    assert model_path().stat().st_size > 1_000_000
    ok, message = probe_opencv_pose()
    assert ok, message


def test_cpu_detector_returns_33_landmarks_for_real_frame():

    detector = OpenCVPoseDetector(confidence_threshold=0.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    landmarks, confidence = detector.infer(frame)
    assert confidence >= 0
    assert landmarks is not None
    assert landmarks.shape == (33, 4)


def test_auto_backend_falls_back_to_legacy(monkeypatch):
    monkeypatch.setattr(
        "dance_pose_coach.pose._extract_pose_opencv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("CPU unavailable")),
    )
    monkeypatch.setattr(
        "dance_pose_coach.pose._extract_pose_mediapipe",
        lambda *_args, **_kwargs: _sequence("mediapipe-legacy"),
    )
    result = extract_pose(Path("unused.mp4"), AnalysisConfig())
    assert result.backend == "mediapipe-legacy"


def test_auto_backend_prefers_opencv(monkeypatch):
    monkeypatch.setattr(
        "dance_pose_coach.pose._extract_pose_opencv",
        lambda *_args, **_kwargs: _sequence("opencv-cpu"),
    )
    monkeypatch.setattr(
        "dance_pose_coach.pose._extract_pose_mediapipe",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    result = extract_pose(Path("unused.mp4"), AnalysisConfig())
    assert result.backend == "opencv-cpu"
