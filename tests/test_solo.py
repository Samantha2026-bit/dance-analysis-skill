from __future__ import annotations

import json

import numpy as np

from dance_pose_coach.models import AudioFeatures, MediaInfo
from dance_pose_coach.solo import build_solo_result
from dance_pose_coach.solo_report import write_solo_reports


def test_solo_result_has_no_reference_score(pose_sequence, tmp_path):
    audio = AudioFeatures(
        tempo=120.0,
        beat_times=np.arange(0, 3, 0.5),
        onset_times=np.arange(0, 3, 0.25),
        duration=3.0,
    )
    media = MediaInfo(tmp_path / "video.mp4", 3.0, 30.0, 1080, 1920)
    result = build_solo_result(pose_sequence, audio, media, {"version": "test"})
    payload = result.to_dict()
    assert result.estimated_tempo_bpm == 120.0
    assert result.pose_confidence == 1.0
    assert "overall_score" not in payload
    assert "pose_score" not in payload
    assert result.segments
    assert set(result.arm_metrics) >= {"wrist_range_balance", "elbow_range_balance"}
    assert set(result.frame_metrics) >= {"torso_lean_p90_degrees", "shoulder_tilt_p90_degrees"}
    assert set(result.hip_metrics) >= {"horizontal_range_frame_widths", "center_motion_share"}
    assert set(result.lower_body_metrics) >= {"knee_range_balance", "leg_motion_share"}
    assert set(result.power_metrics) >= {"accent_sharpness_proxy", "arm_motion_share"}
    assert set(result.groove_metrics) >= {
        "vertical_bounce_range_torso_units",
        "chest_pulse_range_torso_units",
        "arm_frame_p75_torso_units",
    }
    assert result.coach_summary
    assert result.issues
    expected_issue_fields = {
        "category",
        "level",
        "evidence",
        "suggestion",
        "impact",
        "drill_steps",
    }
    assert all(expected_issue_fields <= set(item) for item in result.issues)
    assert all(item["drill_steps"] for item in result.issues)


def test_solo_reports_explain_that_no_total_score_exists(pose_sequence, tmp_path):
    audio = AudioFeatures(120.0, np.arange(0, 3, 0.5), np.array([]), 3.0)
    media = MediaInfo(tmp_path / "video.mp4", 3.0, 30.0, 1080, 1920)
    result = build_solo_result(pose_sequence, audio, media, {"version": "test"})
    paths = write_solo_reports(result, tmp_path / "report")
    html = paths["html"].read_text(encoding="utf-8")
    markdown = paths["markdown"].read_text(encoding="utf-8")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert "不提供姿态正确性" in html
    assert "不提供姿态正确性" in markdown
    assert "最需要先改的问题" in html
    assert "具体练法" in html
    assert "上下律动" in html
    assert "胸腔收放" in html
    assert "分析依据（想看数据时再展开）" in html
    assert html.index("最需要先改的问题") < html.index("分析依据（想看数据时再展开）")
    assert "训练顺序" in html
    assert payload["estimated_tempo_bpm"] == 120.0


def test_solo_pipeline_is_local_by_default(monkeypatch, pose_sequence, tmp_path):
    from dance_pose_coach.config import AnalysisConfig
    from dance_pose_coach.pipeline import analyze_solo

    audio = AudioFeatures(120.0, np.arange(0, 3, 0.5), np.array([]), 3.0)
    media = MediaInfo(tmp_path / "video.mp4", 3.0, 30.0, 1080, 1920)
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"test")
    monkeypatch.setattr("dance_pose_coach.pipeline.inspect_video", lambda _path: media)
    monkeypatch.setattr("dance_pose_coach.pipeline.validate_duration", lambda *_args: None)
    monkeypatch.setattr("dance_pose_coach.pipeline.extract_audio", lambda *_args: audio_path)
    monkeypatch.setattr("dance_pose_coach.pipeline.extract_pose", lambda *_args: pose_sequence)
    monkeypatch.setattr("dance_pose_coach.pipeline.analyze_audio", lambda *_args: audio)
    result, paths = analyze_solo(
        media.path,
        AnalysisConfig(output_root=tmp_path / "runs"),
    )
    assert result.metadata["privacy"] == "local"
    assert paths["html"].exists()
