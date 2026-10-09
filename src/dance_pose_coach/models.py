from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MediaInfo:
    path: Path
    duration: float
    fps: float
    width: int
    height: int


@dataclass
class RecognitionResult:
    query: str
    title: str = ""
    artist: str = ""
    confidence: str = "low"
    method: str = "filename"
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchCandidate:
    title: str
    url: str
    uploader: str = ""
    duration: float | None = None


@dataclass
class PoseSequence:
    times: np.ndarray
    landmarks: np.ndarray  # [frames, 33, 4] => x, y, z, visibility
    source_fps: float
    analyzed_fps: float
    low_confidence_ratio: float
    backend: str = "unknown"


@dataclass
class AudioFeatures:
    tempo: float
    beat_times: np.ndarray
    onset_times: np.ndarray
    duration: float


@dataclass
class RegionMetric:
    name: str
    score: float
    mean_angle_error: float
    max_angle_error: float
    weak_intervals: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class AnalysisResult:
    overall_score: float
    pose_score: float
    rhythm_score: float
    flow_score: float
    timing_offset_seconds: float
    tempo_ratio: float
    mirror_used: bool
    confidence: float
    regions: list[RegionMetric]
    differences: list[str]
    rhythm_issues: list[str]
    recommendations: list[str]
    detailed_issues: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SoloAnalysisResult:
    duration_seconds: float
    source_fps: float
    resolution: tuple[int, int]
    analysis_fps: float
    pose_backend: str
    pose_confidence: float
    estimated_tempo_bpm: float
    detected_beats: int
    motion_accents: int
    timing_offset_seconds: float
    median_absolute_offset_seconds: float
    accents_within_120ms: float
    accents_within_200ms: float
    left_wrist_range: float
    right_wrist_range: float
    hip_horizontal_range: float
    high_energy_intervals: list[tuple[float, float]]
    abrupt_transition_intervals: list[tuple[float, float]]
    low_visibility_intervals: list[tuple[float, float]]
    segments: list[dict[str, float]]
    arm_metrics: dict[str, float]
    hip_metrics: dict[str, float]
    frame_metrics: dict[str, float]
    lower_body_metrics: dict[str, float]
    power_metrics: dict[str, float]
    rhythm_metrics: dict[str, float]
    flow_metrics: dict[str, float]
    groove_metrics: dict[str, float]
    coach_summary: str
    issues: list[dict[str, Any]]
    recommendations: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
