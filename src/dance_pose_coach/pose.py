from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import AnalysisConfig
from .models import PoseSequence

# MediaPipe Pose landmark indexes. Keeping them here avoids leaking framework objects
# into the rest of the analysis code.
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

LEFT_RIGHT_PAIRS = [
    (1, 4),
    (2, 5),
    (3, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
    (25, 26),
    (27, 28),
    (29, 30),
    (31, 32),
]


class PoseError(RuntimeError):
    pass


def _interpolate_missing(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    x = np.arange(len(result))
    for landmark in range(result.shape[1]):
        for channel in range(3):
            series = result[:, landmark, channel]
            valid = np.isfinite(series)
            if valid.sum() >= 2:
                result[:, landmark, channel] = np.interp(x, x[valid], series[valid])
            elif valid.sum() == 1:
                result[:, landmark, channel] = series[valid][0]
            else:
                result[:, landmark, channel] = 0.0
    return result


def _extract_pose_mediapipe(video_path: str | Path, config: AnalysisConfig) -> PoseSequence:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise PoseError("缺少 OpenCV 或 MediaPipe，请重新安装项目依赖。") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PoseError(f"无法读取视频：{video_path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if source_fps <= 0:
        cap.release()
        raise PoseError("视频帧率无效。")
    sample_every = max(1, round(source_fps / config.analysis_fps))
    max_source_frames = int(config.max_duration_seconds * source_fps)
    values: list[np.ndarray] = []
    times: list[float] = []
    low_confidence = 0

    pose_api = mp.solutions.pose
    with pose_api.Pose(
        static_image_mode=False,
        model_complexity=config.pose_model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as detector:
        source_index = 0
        while source_index < max_source_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if source_index % sample_every == 0:
                result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if result.pose_landmarks:
                    frame_values = np.array(
                        [[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks.landmark],
                        dtype=np.float32,
                    )
                    if np.mean(frame_values[:, 3] >= config.min_visibility) < 0.65:
                        low_confidence += 1
                else:
                    frame_values = np.full((33, 4), np.nan, dtype=np.float32)
                    frame_values[:, 3] = 0.0
                    low_confidence += 1
                values.append(frame_values)
                times.append(source_index / source_fps)
            source_index += 1
    cap.release()
    if len(values) < 10:
        raise PoseError("有效姿态帧不足；请确保全身入镜、光线充足且视频至少 2 秒。")
    landmarks = _interpolate_missing(np.stack(values))
    return PoseSequence(
        times=np.asarray(times, dtype=np.float32),
        landmarks=landmarks,
        source_fps=source_fps,
        analyzed_fps=source_fps / sample_every,
        low_confidence_ratio=low_confidence / len(values),
        backend="mediapipe-legacy",
    )


def _extract_pose_opencv(video_path: str | Path, config: AnalysisConfig) -> PoseSequence:
    try:
        import cv2

        from .pose_opencv import OpenCVPoseDetector
    except ImportError as exc:
        raise PoseError("缺少 OpenCV，无法启动 CPU 姿态识别。") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PoseError(f"无法读取视频：{video_path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    if source_fps <= 0:
        cap.release()
        raise PoseError("视频帧率无效。")
    sample_every = max(1, round(source_fps / config.analysis_fps))
    max_source_frames = int(config.max_duration_seconds * source_fps)
    detector = OpenCVPoseDetector(confidence_threshold=0.5)
    values: list[np.ndarray] = []
    times: list[float] = []
    low_confidence = 0
    source_index = 0
    while source_index < max_source_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if source_index % sample_every == 0:
            frame_values, _ = detector.infer(frame)
            if frame_values is None:
                frame_values = np.full((33, 4), np.nan, dtype=np.float32)
                frame_values[:, 3] = 0.0
                low_confidence += 1
            elif np.mean(frame_values[:, 3] >= config.min_visibility) < 0.65:
                low_confidence += 1
            values.append(frame_values)
            times.append(source_index / source_fps)
        source_index += 1
    cap.release()
    if len(values) < 10:
        raise PoseError("有效姿态帧不足；请确保全身入镜、光线充足且视频至少 2 秒。")
    landmarks = _interpolate_missing(np.stack(values))
    return PoseSequence(
        times=np.asarray(times, dtype=np.float32),
        landmarks=landmarks,
        source_fps=source_fps,
        analyzed_fps=source_fps / sample_every,
        low_confidence_ratio=low_confidence / len(values),
        backend="opencv-cpu",
    )


def extract_pose(video_path: str | Path, config: AnalysisConfig) -> PoseSequence:
    """Extract pose without requiring a graphical session.

    OpenCV DNN is the default because it is CPU-only and works in terminals,
    desktop launchers, and headless AI sandboxes. The legacy MediaPipe graph is
    retained as an explicit compatibility option for existing installations.
    """
    if config.pose_backend in {"auto", "opencv"}:
        try:
            return _extract_pose_opencv(video_path, config)
        except Exception as opencv_exc:
            if config.pose_backend == "opencv":
                raise PoseError(str(opencv_exc)) from opencv_exc
            try:
                return _extract_pose_mediapipe(video_path, config)
            except Exception as mediapipe_exc:
                raise PoseError(
                    "CPU 姿态识别和 MediaPipe 兼容模式均失败。"
                    f" CPU：{opencv_exc}；MediaPipe：{mediapipe_exc}"
                ) from mediapipe_exc
    return _extract_pose_mediapipe(video_path, config)


def normalize_pose(landmarks: np.ndarray) -> np.ndarray:
    """Remove camera translation and scale while retaining body proportions."""
    points = landmarks.copy().astype(np.float32)
    hip_center = (points[:, LEFT_HIP, :3] + points[:, RIGHT_HIP, :3]) / 2
    shoulder_center = (points[:, LEFT_SHOULDER, :3] + points[:, RIGHT_SHOULDER, :3]) / 2
    scale = np.linalg.norm(shoulder_center - hip_center, axis=1)
    scale = np.maximum(scale, 1e-4)
    points[:, :, :3] = (points[:, :, :3] - hip_center[:, None, :]) / scale[:, None, None]
    return points


def mirror_pose(landmarks: np.ndarray) -> np.ndarray:
    mirrored = landmarks.copy()
    mirrored[:, :, 0] *= -1
    for left, right in LEFT_RIGHT_PAIRS:
        mirrored[:, [left, right]] = mirrored[:, [right, left]]
    return mirrored


def motion_energy(landmarks: np.ndarray, fps: float) -> np.ndarray:
    normalized = normalize_pose(landmarks)
    important = [LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]
    velocity = np.diff(normalized[:, important, :3], axis=0, prepend=normalized[:1, important, :3])
    energy = np.mean(np.linalg.norm(velocity * fps, axis=-1), axis=1)
    if len(energy) >= 7:
        from scipy.signal import savgol_filter

        window = min(11, len(energy) if len(energy) % 2 else len(energy) - 1)
        energy = savgol_filter(energy, max(window, 5), 2)
    return np.maximum(energy, 0)


def angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    cosine = np.sum(ba * bc, axis=-1) / np.maximum(denominator, 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
