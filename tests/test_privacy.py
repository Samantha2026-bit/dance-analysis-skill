from __future__ import annotations

import pytest

from dance_pose_coach.config import AnalysisConfig
from dance_pose_coach.pipeline import analyze, identify_and_search


def test_online_services_are_disabled_by_default():
    assert AnalysisConfig().allow_online_services is False


def test_search_fails_before_touching_video_when_offline(tmp_path):
    missing_video = tmp_path / "not-present.mp4"
    with pytest.raises(RuntimeError, match="默认关闭"):
        identify_and_search(missing_video)


def test_analysis_requires_local_reference_when_offline(tmp_path):
    missing_video = tmp_path / "not-present.mp4"
    with pytest.raises(RuntimeError, match="本地参考视频"):
        analyze(missing_video)
