from __future__ import annotations

import numpy as np

from dance_pose_coach.alignment import align_sequences
from dance_pose_coach.pose import LEFT_WRIST, RIGHT_WRIST, mirror_pose, normalize_pose


def test_normalize_pose_removes_translation(pose_sequence):
    shifted = pose_sequence.landmarks.copy()
    shifted[:, :, :3] += np.array([4.0, -2.0, 0.3])
    assert np.allclose(
        normalize_pose(pose_sequence.landmarks)[:, :, :3],
        normalize_pose(shifted)[:, :, :3],
        atol=1e-4,
    )


def test_mirror_swaps_sides(pose_sequence):
    original = pose_sequence.landmarks
    mirrored = mirror_pose(original)
    assert np.allclose(mirrored[:, LEFT_WRIST, 0], -original[:, RIGHT_WRIST, 0])
    assert np.allclose(mirrored[:, RIGHT_WRIST, 0], -original[:, LEFT_WRIST, 0])


def test_dtw_returns_monotonic_path(pose_sequence):
    user = pose_sequence.landmarks
    reference = np.repeat(user, 2, axis=0)
    path = align_sequences(user, reference)
    assert path.shape[1] == 2
    assert np.all(np.diff(path[:, 0]) >= 0)
    assert np.all(np.diff(path[:, 1]) >= 0)
    assert tuple(path[0]) == (0, 0)
    assert tuple(path[-1]) == (len(user) - 1, len(reference) - 1)

