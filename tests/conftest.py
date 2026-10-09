from __future__ import annotations

import numpy as np
import pytest

from dance_pose_coach.models import PoseSequence


def synthetic_landmarks(frames: int = 30, phase: float = 0.0) -> np.ndarray:
    data = np.zeros((frames, 33, 4), dtype=np.float32)
    data[:, :, 3] = 1.0
    # A simple upright body with arms and legs; selected joints are non-collinear.
    xy = {
        11: (-0.25, -0.55),
        12: (0.25, -0.55),
        13: (-0.45, -0.20),
        14: (0.45, -0.20),
        15: (-0.55, 0.05),
        16: (0.55, 0.05),
        23: (-0.16, 0.0),
        24: (0.16, 0.0),
        25: (-0.20, 0.50),
        26: (0.20, 0.50),
        27: (-0.22, 1.0),
        28: (0.22, 1.0),
    }
    for joint, (x, y) in xy.items():
        data[:, joint, 0] = x
        data[:, joint, 1] = y
    wave = np.sin(np.linspace(phase, phase + 2 * np.pi, frames))
    data[:, 15, 1] += wave * 0.25
    data[:, 16, 1] -= wave * 0.25
    data[:, 23:25, 0] += wave[:, None] * 0.08
    return data


@pytest.fixture
def pose_sequence() -> PoseSequence:
    landmarks = synthetic_landmarks()
    return PoseSequence(
        times=np.arange(len(landmarks), dtype=np.float32) / 10,
        landmarks=landmarks,
        source_fps=30,
        analyzed_fps=10,
        low_confidence_ratio=0.0,
    )

