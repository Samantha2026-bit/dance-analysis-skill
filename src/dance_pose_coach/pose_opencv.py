from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np

MODEL_NAME = "pose_estimation_mediapipe_2023mar.onnx"
MODEL_SHA256 = "9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f"
INPUT_SIZE = 256


class OpenCVPoseError(RuntimeError):
    pass


def model_path() -> Path:
    return Path(str(files("dance_pose_coach.assets").joinpath(MODEL_NAME)))


class OpenCVPoseDetector:
    """CPU-only 33-point pose inference using OpenCV DNN.

    The bundled model is OpenCV Zoo's Apache-2.0 MediaPipe Pose ONNX export.
    Keeping inference inside OpenCV avoids MediaPipe's macOS OpenGL graph service.
    """

    def __init__(self, confidence_threshold: float = 0.5, path: Path | None = None):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise OpenCVPoseError("缺少 OpenCV，无法启动 CPU 姿态识别。") from exc

        self.cv2 = cv2
        self.confidence_threshold = confidence_threshold
        self.path = path or model_path()
        if not self.path.is_file():
            raise OpenCVPoseError(f"CPU 姿态模型不存在：{self.path}")
        try:
            self.net = cv2.dnn.readNet(str(self.path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as exc:
            raise OpenCVPoseError(f"CPU 姿态模型无法载入：{exc}") from exc

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))

    def infer(self, frame: np.ndarray) -> tuple[np.ndarray | None, float]:
        if frame is None or frame.ndim != 3:
            return None, 0.0
        height, width = frame.shape[:2]
        side = max(height, width)
        pad_x = (side - width) // 2
        pad_y = (side - height) // 2
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        canvas[pad_y : pad_y + height, pad_x : pad_x + width] = frame

        resized = self.cv2.resize(
            canvas,
            (INPUT_SIZE, INPUT_SIZE),
            interpolation=self.cv2.INTER_AREA,
        )
        blob = self.cv2.cvtColor(resized, self.cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        try:
            self.net.setInput(blob[np.newaxis, ...])
            outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        except self.cv2.error as exc:
            raise OpenCVPoseError(f"CPU 姿态推理失败：{exc}") from exc

        if len(outputs) < 2:
            raise OpenCVPoseError("CPU 姿态模型输出格式不受支持。")
        raw = outputs[0][0].reshape(-1, 5)
        confidence = float(outputs[1].reshape(-1)[0])
        if confidence < self.confidence_threshold or len(raw) < 33:
            return None, confidence

        raw = raw[:33].copy()
        x_pixels = raw[:, 0] / INPUT_SIZE * side - pad_x
        y_pixels = raw[:, 1] / INPUT_SIZE * side - pad_y
        landmarks = np.empty((33, 4), dtype=np.float32)
        landmarks[:, 0] = x_pixels / max(width, 1)
        landmarks[:, 1] = y_pixels / max(height, 1)
        # Match MediaPipe's approximate normalized depth convention.
        landmarks[:, 2] = raw[:, 2] / INPUT_SIZE * (side / max(width, 1))
        visibility = self._sigmoid(raw[:, 3])
        presence = self._sigmoid(raw[:, 4])
        landmarks[:, 3] = np.minimum(visibility, presence) * confidence
        return landmarks, confidence


def probe_opencv_pose() -> tuple[bool, str]:
    try:
        detector = OpenCVPoseDetector(confidence_threshold=0.0)
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        detector.infer(blank)
    except Exception as exc:
        return False, str(exc)
    return True, "OpenCV CPU 姿态模型可用（不需要 OpenGL）。"
