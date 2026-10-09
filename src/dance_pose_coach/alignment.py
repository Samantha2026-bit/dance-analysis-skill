from __future__ import annotations

import librosa
import numpy as np

from .models import PoseSequence
from .pose import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    mirror_pose,
    normalize_pose,
)

KEY_JOINTS = [
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]


def _features(landmarks: np.ndarray) -> np.ndarray:
    normalized = normalize_pose(landmarks)[:, KEY_JOINTS, :2]
    position = normalized.reshape(len(normalized), -1)
    velocity = np.diff(position, axis=0, prepend=position[:1])
    return np.concatenate([position, velocity * 0.5], axis=1)


def align_sequences(user_landmarks: np.ndarray, reference_landmarks: np.ndarray) -> np.ndarray:
    """Return monotonically matched frame pairs using constrained DTW."""
    if len(user_landmarks) < 2 or len(reference_landmarks) < 2:
        raise ValueError("At least two frames are required for alignment")
    user_features = _features(user_landmarks)
    ref_features = _features(reference_landmarks)
    try:
        _, path = librosa.sequence.dtw(
            X=user_features.T,
            Y=ref_features.T,
            metric="euclidean",
            global_constraints=True,
            band_rad=0.25,
        )
    except Exception:
        # A large intro/outro duration mismatch can make the diagonal band impossible.
        # At the 120 s / 10 FPS cap, unconstrained DTW remains practical on an Intel Mac.
        _, path = librosa.sequence.dtw(
            X=user_features.T,
            Y=ref_features.T,
            metric="euclidean",
        )
    return np.asarray(path[::-1], dtype=np.int32)


def choose_mirror(
    user: PoseSequence,
    reference: PoseSequence,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    original = user.landmarks
    if mode == "always":
        mirrored = mirror_pose(original)
        return mirrored, align_sequences(mirrored, reference.landmarks), True
    direct_path = align_sequences(original, reference.landmarks)
    if mode == "never":
        return original, direct_path, False
    mirrored = mirror_pose(original)
    mirrored_path = align_sequences(mirrored, reference.landmarks)
    direct_cost = _path_cost(original, reference.landmarks, direct_path)
    mirrored_cost = _path_cost(mirrored, reference.landmarks, mirrored_path)
    if mirrored_cost < direct_cost * 0.98:
        return mirrored, mirrored_path, True
    return original, direct_path, False


def _path_cost(user: np.ndarray, reference: np.ndarray, path: np.ndarray) -> float:
    user_norm = normalize_pose(user)[:, KEY_JOINTS, :2]
    ref_norm = normalize_pose(reference)[:, KEY_JOINTS, :2]
    distances = user_norm[path[:, 0]] - ref_norm[path[:, 1]]
    return float(np.mean(np.linalg.norm(distances, axis=-1)))
