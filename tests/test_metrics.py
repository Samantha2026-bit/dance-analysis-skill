from __future__ import annotations

import numpy as np

from dance_pose_coach.metrics import build_result, compute_region_metrics
from dance_pose_coach.models import AudioFeatures, PoseSequence


def test_identical_pose_scores_near_100(pose_sequence):
    path = np.column_stack([np.arange(30), np.arange(30)])
    metrics = compute_region_metrics(
        pose_sequence,
        pose_sequence,
        pose_sequence.landmarks,
        path,
    )
    assert len(metrics) == 6
    assert min(item.score for item in metrics) >= 99.9
    assert all(item.mean_angle_error == 0 for item in metrics)


def test_arm_error_lowers_arm_score(pose_sequence):
    changed = pose_sequence.landmarks.copy()
    changed[:, 15, 0] += 0.7
    path = np.column_stack([np.arange(30), np.arange(30)])
    metrics = compute_region_metrics(pose_sequence, pose_sequence, changed, path)
    scores = {item.name: item.score for item in metrics}
    assert scores["左臂"] < 99


def test_comparison_report_has_detailed_body_power_and_training_fields(pose_sequence):
    changed = pose_sequence.landmarks.copy()
    changed[:, 15, 0] += 1.4
    changed[:, 16, 1] -= 0.9
    user = PoseSequence(
        times=pose_sequence.times,
        landmarks=changed,
        source_fps=pose_sequence.source_fps,
        analyzed_fps=pose_sequence.analyzed_fps,
        low_confidence_ratio=0.0,
        backend="test",
    )
    path = np.column_stack([np.arange(30), np.arange(30)])
    audio = AudioFeatures(120.0, np.arange(0, 3, 0.5), np.array([]), 3.0)
    result = build_result(
        user,
        pose_sequence,
        changed,
        path,
        audio,
        audio,
        False,
        {},
    )
    assert result.detailed_issues
    assert any(item["category"] == "胳膊" for item in result.detailed_issues)
    assert "power_energy_ratio" in result.metadata
    assert "power_correlation" in result.metadata
    assert "comparison_segments" in result.metadata
    assert len(result.recommendations) >= 3
