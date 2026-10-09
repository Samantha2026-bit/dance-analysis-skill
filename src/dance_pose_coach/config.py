from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    """Runtime knobs chosen conservatively for supported desktop computers."""

    analysis_fps: float = 10.0
    max_duration_seconds: float = 120.0
    min_visibility: float = 0.45
    pose_model_complexity: int = 1
    pose_backend: str = "auto"  # auto/opencv use CPU; mediapipe keeps legacy behavior
    mirror_mode: str = "auto"  # auto, never, always
    output_root: Path = Path("runs")
    keep_intermediates: bool = False
    allow_online_services: bool = False

    def validate(self) -> None:
        if not 2 <= self.analysis_fps <= 30:
            raise ValueError("analysis_fps must be between 2 and 30")
        if not 5 <= self.max_duration_seconds <= 600:
            raise ValueError("max_duration_seconds must be between 5 and 600")
        if self.mirror_mode not in {"auto", "never", "always"}:
            raise ValueError("mirror_mode must be auto, never, or always")
        if self.pose_backend not in {"auto", "opencv", "mediapipe"}:
            raise ValueError("pose_backend must be auto, opencv, or mediapipe")
