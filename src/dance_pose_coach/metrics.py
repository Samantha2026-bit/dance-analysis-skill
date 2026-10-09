from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.signal import find_peaks

from .models import AnalysisResult, AudioFeatures, PoseSequence, RegionMetric
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
    angle_degrees,
    motion_energy,
    normalize_pose,
)

REGION_ANGLES: dict[str, list[tuple[int, int, int]]] = {
    "左臂": [
        (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
        (LEFT_ELBOW, LEFT_SHOULDER, LEFT_HIP),
    ],
    "右臂": [
        (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
        (RIGHT_ELBOW, RIGHT_SHOULDER, RIGHT_HIP),
    ],
    "肩部与躯干": [
        (LEFT_SHOULDER, LEFT_HIP, RIGHT_SHOULDER),
        (RIGHT_SHOULDER, RIGHT_HIP, LEFT_SHOULDER),
    ],
    "髋部": [
        (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
        (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    ],
    "左腿": [
        (LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),
        (LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE),
    ],
    "右腿": [
        (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE),
        (RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE),
    ],
}


def _angle_matrix(landmarks: np.ndarray, definitions: Iterable[tuple[int, int, int]]) -> np.ndarray:
    return np.stack(
        [
            angle_degrees(landmarks[:, a, :3], landmarks[:, b, :3], landmarks[:, c, :3])
            for a, b, c in definitions
        ],
        axis=1,
    )


def _weak_intervals(
    times: np.ndarray,
    errors: np.ndarray,
    threshold: float = 25.0,
) -> list[tuple[float, float]]:
    bad_times = times[errors >= threshold]
    if not len(bad_times):
        return []
    intervals: list[tuple[float, float]] = []
    start = previous = float(bad_times[0])
    for current_value in bad_times[1:]:
        current = float(current_value)
        if current - previous > 0.6:
            intervals.append((round(start, 2), round(previous, 2)))
            start = current
        previous = current
    intervals.append((round(start, 2), round(previous, 2)))
    return sorted(intervals, key=lambda item: item[1] - item[0], reverse=True)[:3]


def compute_region_metrics(
    user: PoseSequence,
    reference: PoseSequence,
    user_landmarks: np.ndarray,
    path: np.ndarray,
) -> list[RegionMetric]:
    user_indexes, ref_indexes = path[:, 0], path[:, 1]
    times = user.times[user_indexes]
    metrics: list[RegionMetric] = []
    for name, definitions in REGION_ANGLES.items():
        user_angles = _angle_matrix(user_landmarks, definitions)[user_indexes]
        ref_angles = _angle_matrix(reference.landmarks, definitions)[ref_indexes]
        per_frame = np.mean(np.abs(user_angles - ref_angles), axis=1)
        mean_error = float(np.mean(per_frame))
        metrics.append(
            RegionMetric(
                name=name,
                score=round(float(100 * np.exp(-mean_error / 32)), 1),
                mean_angle_error=round(mean_error, 1),
                max_angle_error=round(float(np.percentile(per_frame, 95)), 1),
                weak_intervals=_weak_intervals(times, per_frame),
            )
        )
    return metrics


def _nearest_signed(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if not len(values) or not len(targets):
        return np.array([], dtype=float)
    indexes = np.searchsorted(targets, values)
    indexes = np.clip(indexes, 1, len(targets) - 1)
    left = targets[indexes - 1]
    right = targets[indexes]
    nearest = np.where(np.abs(values - left) <= np.abs(values - right), left, right)
    return values - nearest


def _motion_accents(sequence: PoseSequence, landmarks: np.ndarray) -> np.ndarray:
    energy = motion_energy(landmarks, sequence.analyzed_fps)
    distance = max(1, int(sequence.analyzed_fps * 0.25))
    prominence = max(float(np.std(energy) * 0.35), 1e-5)
    peaks, _ = find_peaks(energy, distance=distance, prominence=prominence)
    return sequence.times[peaks]


def compute_rhythm(
    user_pose: PoseSequence,
    reference_pose: PoseSequence,
    user_landmarks: np.ndarray,
    user_audio: AudioFeatures,
    reference_audio: AudioFeatures,
) -> tuple[float, float, float, list[str]]:
    tempo_ratio = user_audio.tempo / reference_audio.tempo if reference_audio.tempo > 0 else 1.0
    user_accents = _motion_accents(user_pose, user_landmarks)
    ref_accents = _motion_accents(reference_pose, reference_pose.landmarks)
    user_offsets = _nearest_signed(user_accents, user_audio.beat_times)
    ref_offsets = _nearest_signed(ref_accents, reference_audio.beat_times)
    user_median = float(np.median(user_offsets)) if len(user_offsets) else 0.0
    ref_median = float(np.median(ref_offsets)) if len(ref_offsets) else 0.0
    timing_offset = user_median - ref_median
    accent_spread = float(np.median(np.abs(user_offsets))) if len(user_offsets) else 0.35
    tempo_penalty = abs(np.log(max(tempo_ratio, 1e-4)))
    score = 100 * np.exp(-abs(timing_offset) / 0.18 - accent_spread / 0.35 - tempo_penalty / 0.18)
    issues: list[str] = []
    if abs(tempo_ratio - 1) > 0.06:
        direction = "更快" if tempo_ratio > 1 else "更慢"
        issues.append(f"整体动作/音频速度比参考{direction}约 {abs(tempo_ratio - 1) * 100:.0f}%")
    if timing_offset > 0.08:
        issues.append(f"动作重拍平均慢约 {timing_offset:.2f} 秒")
    elif timing_offset < -0.08:
        issues.append(f"动作重拍平均抢约 {abs(timing_offset):.2f} 秒")
    if accent_spread > 0.18:
        issues.append("动作重音分散，卡点稳定性不足")
    if not issues:
        issues.append("未发现明显的整体抢拍或慢拍；可继续检查分段细节")
    return round(float(np.clip(score, 0, 100)), 1), timing_offset, tempo_ratio, issues


def compute_flow(
    user: PoseSequence,
    reference: PoseSequence,
    user_landmarks: np.ndarray,
    path: np.ndarray,
) -> float:
    user_norm = normalize_pose(user_landmarks)
    ref_norm = normalize_pose(reference.landmarks)
    joints = [LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]
    user_energy = motion_energy(user_landmarks, user.analyzed_fps)[path[:, 0]]
    ref_energy = motion_energy(reference.landmarks, reference.analyzed_fps)[path[:, 1]]
    if np.std(user_energy) > 1e-6 and np.std(ref_energy) > 1e-6:
        correlation = float(np.corrcoef(user_energy, ref_energy)[0, 1])
    else:
        correlation = 0.0
    user_velocity = np.diff(user_norm[:, joints, :3], axis=0) * user.analyzed_fps
    ref_velocity = np.diff(ref_norm[:, joints, :3], axis=0) * reference.analyzed_fps
    user_jerk = np.diff(user_velocity, n=2, axis=0)
    ref_jerk = np.diff(ref_velocity, n=2, axis=0)
    user_rms = float(np.sqrt(np.mean(user_jerk**2))) if user_jerk.size else 0.0
    ref_rms = float(np.sqrt(np.mean(ref_jerk**2))) if ref_jerk.size else 0.0
    jerk_ratio_error = abs(np.log((user_rms + 1e-4) / (ref_rms + 1e-4)))
    score = 100 * (0.55 * np.clip((correlation + 1) / 2, 0, 1) + 0.45 * np.exp(-jerk_ratio_error))
    return round(float(np.clip(score, 0, 100)), 1)


def _comparison_details(
    user: PoseSequence,
    reference: PoseSequence,
    user_landmarks: np.ndarray,
    path: np.ndarray,
    regions: list[RegionMetric],
    rhythm_score: float,
    timing_offset: float,
    tempo_ratio: float,
    flow_score: float,
    confidence: float,
) -> tuple[list[dict], dict]:
    user_indexes, ref_indexes = path[:, 0], path[:, 1]
    times = user.times[user_indexes]
    user_energy = motion_energy(user_landmarks, user.analyzed_fps)[user_indexes]
    ref_energy = motion_energy(reference.landmarks, reference.analyzed_fps)[ref_indexes]
    power_ratio = float(np.median(user_energy + 1e-4) / np.median(ref_energy + 1e-4))
    if np.std(user_energy) > 1e-6 and np.std(ref_energy) > 1e-6:
        power_correlation = float(np.corrcoef(user_energy, ref_energy)[0, 1])
    else:
        power_correlation = 0.0
    user_scale = max(float(np.percentile(user_energy, 90) - np.percentile(user_energy, 10)), 1e-5)
    ref_scale = max(float(np.percentile(ref_energy, 90) - np.percentile(ref_energy, 10)), 1e-5)
    user_energy_norm = (user_energy - np.median(user_energy)) / user_scale
    ref_energy_norm = (ref_energy - np.median(ref_energy)) / ref_scale
    power_difference = np.abs(user_energy_norm - ref_energy_norm)
    threshold = float(np.percentile(power_difference, 85))
    power_intervals = _weak_intervals(times, power_difference, threshold=max(threshold, 0.15))

    issues: list[dict] = []
    category_map = {
        "左臂": "胳膊",
        "右臂": "胳膊",
        "肩部与躯干": "框架",
        "髋部": "髋部",
        "左腿": "腿部",
        "右腿": "腿部",
    }
    for region in sorted(regions, key=lambda item: item.score):
        if region.score >= 72:
            continue
        level = "高" if region.score < 45 else "中"
        issues.append(
            {
                "category": category_map[region.name],
                "level": level,
                "title": f"{region.name}与参考差异较明显",
                "evidence": (
                    f"区域得分 {region.score}/100，平均角度偏差 "
                    f"{region.mean_angle_error:.1f}°"
                ),
                "intervals": region.weak_intervals,
                "suggestion": _recommendation(region),
            }
        )
    if rhythm_score < 70:
        direction = "晚" if timing_offset > 0 else "早"
        issues.append(
            {
                "category": "节奏",
                "level": "高" if rhythm_score < 45 else "中",
                "title": "动作重音与参考的节奏组织不同",
                "evidence": (
                    f"节奏得分 {rhythm_score}/100，重音相对参考{direction} "
                    f"{abs(timing_offset):.3f}s，速度比 {tempo_ratio:.3f}"
                ),
                "intervals": [],
                "suggestion": "先用参考视频的同一 8 拍做重拍定格，再补完整手脚路线。",
            }
        )
    if power_ratio < 0.75 or power_ratio > 1.35 or power_correlation < 0.45:
        direction = "偏强/偏大" if power_ratio > 1 else "偏弱/偏小"
        issues.append(
            {
                "category": "发力表现",
                "level": "中",
                "title": "动作能量分布与参考不一致",
                "evidence": (
                    f"能量中位比 {power_ratio:.2f}（{direction}），"
                    f"能量曲线相关 {power_correlation:.2f}"
                ),
                "intervals": power_intervals,
                "suggestion": "在标记时段对照参考的蓄力、爆发和回收，不只追动作终点。",
            }
        )
    if flow_score < 65:
        issues.append(
            {
                "category": "衔接",
                "level": "中",
                "title": "动作之间的速度变化与参考不同",
                "evidence": f"连接流畅得分 {flow_score}/100",
                "intervals": power_intervals,
                "suggestion": (
                    "把问题时段前后各扩展 2 拍慢练，重点练上一动作的回收"
                    "和下一动作的启动。"
                ),
            }
        )
    if confidence < 0.7:
        issues.insert(
            0,
            {
                "category": "录制质量",
                "level": "高",
                "title": "关键点可信度偏低",
                "evidence": f"综合姿态可信度约 {confidence * 100:.0f}%",
                "intervals": [],
                "suggestion": "确保双方都是单人、全身入镜、无遮挡且镜头固定，再看细粒度差异。",
            },
        )

    level_order = {"高": 0, "中": 1, "观察": 2}
    issues.sort(key=lambda item: level_order.get(item["level"], 9))
    segments: list[dict[str, float]] = []
    for start in np.arange(0, float(user.times[-1]) + 0.01, 4.0):
        end = min(float(start + 4.0), float(user.times[-1]))
        selected = (times >= start) & (times < end)
        if not np.any(selected):
            continue
        segments.append(
            {
                "start": round(float(start), 1),
                "end": round(end, 1),
                "user_energy": round(float(np.mean(user_energy[selected])), 3),
                "reference_energy": round(float(np.mean(ref_energy[selected])), 3),
                "energy_difference": round(float(np.mean(power_difference[selected])), 3),
            }
        )
    extra = {
        "power_energy_ratio": round(power_ratio, 3),
        "power_correlation": round(power_correlation, 3),
        "power_mismatch_intervals": power_intervals,
        "comparison_segments": segments,
        "force_metric_boundary": "动作能量仅是速度/加速度代理，不代表真实肌肉力量。",
    }
    return issues, extra


def build_result(
    user: PoseSequence,
    reference: PoseSequence,
    user_landmarks: np.ndarray,
    path: np.ndarray,
    user_audio: AudioFeatures,
    reference_audio: AudioFeatures,
    mirror_used: bool,
    metadata: dict,
) -> AnalysisResult:
    regions = compute_region_metrics(user, reference, user_landmarks, path)
    pose_score = round(float(np.mean([region.score for region in regions])), 1)
    rhythm_score, timing_offset, tempo_ratio, rhythm_issues = compute_rhythm(
        user, reference, user_landmarks, user_audio, reference_audio
    )
    flow_score = compute_flow(user, reference, user_landmarks, path)
    overall = round(0.55 * pose_score + 0.25 * rhythm_score + 0.20 * flow_score, 1)
    confidence = float(
        np.clip(1 - (user.low_confidence_ratio + reference.low_confidence_ratio) / 2, 0, 1)
    )
    detailed_issues, extra_metadata = _comparison_details(
        user,
        reference,
        user_landmarks,
        path,
        regions,
        rhythm_score,
        timing_offset,
        tempo_ratio,
        flow_score,
        confidence,
    )
    metadata.update(extra_metadata)
    weakest = sorted(regions, key=lambda item: item.score)[:3]
    differences = [
        f"{item.name}平均角度偏差 {item.mean_angle_error:.1f}°，"
        f"95 分位偏差 {item.max_angle_error:.1f}°"
        for item in weakest
    ]
    recommendations = []
    for issue in detailed_issues:
        suggestion = issue["suggestion"]
        if suggestion not in recommendations:
            recommendations.append(suggestion)
    if timing_offset > 0.08:
        recommendations.append("用 0.75× 速度练习：每次在重拍前完成预备动作，再逐级恢复原速。")
    elif timing_offset < -0.08:
        recommendations.append("先数满“1-e-&-a”再落重心，避免手脚提前冲到动作终点。")
    else:
        recommendations.append("保持原速做 8 拍循环，用录像检查每个重拍的定格轮廓。")
    if confidence < 0.7:
        recommendations.append("重新录制全身无遮挡、固定机位的视频，以提高关键点可信度。")
    recommendations.extend(
        [
            "训练顺序：先修最高优先级问题，再按 0.75× 慢练、原速定格、录像复查做 3 轮。",
            "只比较同一编舞和同一音乐段落；参考多人同框时先裁成单人画面。",
        ]
    )
    return AnalysisResult(
        overall_score=overall,
        pose_score=pose_score,
        rhythm_score=rhythm_score,
        flow_score=flow_score,
        timing_offset_seconds=round(timing_offset, 3),
        tempo_ratio=round(tempo_ratio, 3),
        mirror_used=mirror_used,
        confidence=round(confidence, 3),
        regions=regions,
        differences=differences,
        rhythm_issues=rhythm_issues,
        recommendations=recommendations,
        detailed_issues=detailed_issues,
        metadata=metadata,
    )


def _recommendation(region: RegionMetric) -> str:
    interval = (
        f"（优先复练 {region.weak_intervals[0][0]:.1f}–"
        f"{region.weak_intervals[0][1]:.1f}s）"
        if region.weak_intervals
        else ""
    )
    drills = {
        "左臂": "左臂做肩—肘—腕分解：先定肩线，再慢速伸肘，最后补手腕方向。",
        "右臂": "右臂做肩—肘—腕分解：对镜保持肩不耸，逐拍校准肘角。",
        "肩部与躯干": "做肩胯反向隔离和躯干波浪，每 4 拍只移动一个关节层级。",
        "髋部": "练习重心左右换胯：膝盖放松，髋先到位，再带动上身。",
        "左腿": "左腿做慢速屈伸与落地定格，检查膝盖、脚尖方向一致。",
        "右腿": "右腿做慢速屈伸与落地定格，避免膝盖内扣并明确发力终点。",
    }
    return f"{drills[region.name]}{interval}"
