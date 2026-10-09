from __future__ import annotations

import json

from dance_pose_coach.models import AnalysisResult, RegionMetric
from dance_pose_coach.report import write_reports


def test_reports_are_self_contained_and_serializable(tmp_path):
    result = AnalysisResult(
        overall_score=82.0,
        pose_score=80.0,
        rhythm_score=88.0,
        flow_score=79.0,
        timing_offset_seconds=-0.09,
        tempo_ratio=1.02,
        mirror_used=False,
        confidence=0.91,
        regions=[RegionMetric("左臂", 75.0, 9.2, 26.0, [(1.0, 1.5)])],
        differences=["左臂角度偏差"],
        rhythm_issues=["动作略抢拍"],
        recommendations=["慢速分解练习"],
        metadata={"version": "test", "reference_source": "local"},
    )
    paths = write_reports(result, tmp_path)
    html = paths["html"].read_text(encoding="utf-8")
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert "舞蹈对比分析报告" in html
    assert "<style>" in html
    assert payload["overall_score"] == 82.0
    assert paths["markdown"].exists()

