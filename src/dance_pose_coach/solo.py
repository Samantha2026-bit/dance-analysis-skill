from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks

from .models import AudioFeatures, MediaInfo, PoseSequence, SoloAnalysisResult
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

KEY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def _intervals(
    times: np.ndarray,
    mask: np.ndarray,
    gap: float = 0.4,
    limit: int = 10,
) -> list[tuple[float, float]]:
    selected = times[mask]
    if not len(selected):
        return []
    result: list[tuple[float, float]] = []
    start = previous = float(selected[0])
    for item in selected[1:]:
        current = float(item)
        if current - previous > gap:
            result.append((round(start, 2), round(previous, 2)))
            start = current
        previous = current
    result.append((round(start, 2), round(previous, 2)))
    return sorted(result, key=lambda span: span[1] - span[0], reverse=True)[:limit]


def _nearest_signed(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if not len(values) or len(targets) < 2:
        return np.array([], dtype=float)
    indexes = np.clip(np.searchsorted(targets, values), 1, len(targets) - 1)
    left, right = targets[indexes - 1], targets[indexes]
    nearest = np.where(np.abs(values - left) <= np.abs(values - right), left, right)
    return values - nearest


def _range(values: np.ndarray) -> float:
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def _balance(left: float, right: float) -> float:
    high = max(abs(left), abs(right), 1e-6)
    return float(np.clip(min(abs(left), abs(right)) / high * 100, 0, 100))


def _issue(
    category: str,
    level: str,
    title: str,
    evidence: str,
    suggestion: str,
    spans: list[tuple[float, float]] | None = None,
    impact: str = "",
    drill_steps: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "category": category,
        "level": level,
        "title": title,
        "evidence": evidence,
        "intervals": spans or [],
        "suggestion": suggestion,
        "impact": impact,
        "drill_steps": drill_steps or [suggestion],
    }


def _merge_segment_spans(
    segments: list[dict[str, float]], key: str, threshold: float
) -> list[tuple[float, float]]:
    """Turn consecutive low-value four-second windows into readable practice spans."""
    low = [row for row in segments if row.get(key, float("inf")) < threshold]
    if not low:
        return []
    spans: list[tuple[float, float]] = []
    start = float(low[0]["start"])
    end = float(low[0]["end"])
    for row in low[1:]:
        row_start, row_end = float(row["start"]), float(row["end"])
        if row_start <= end + 0.05:
            end = row_end
        else:
            spans.append((round(start, 1), round(end, 1)))
            start, end = row_start, row_end
    spans.append((round(start, 1), round(end, 1)))
    return spans


def build_solo_result(
    pose: PoseSequence,
    audio: AudioFeatures,
    media: MediaInfo,
    metadata: dict,
) -> SoloAnalysisResult:
    landmarks = pose.landmarks
    normalized = normalize_pose(landmarks)
    fps = pose.analyzed_fps
    energy = motion_energy(landmarks, fps)
    prominence = max(float(np.std(energy) * 0.35), 1e-5)
    peak_indexes, properties = find_peaks(
        energy,
        distance=max(1, int(fps * 0.25)),
        prominence=prominence,
    )
    accent_times = pose.times[peak_indexes]
    offsets = _nearest_signed(accent_times, audio.beat_times)
    median_offset = float(np.median(offsets)) if len(offsets) else 0.0
    median_abs_offset = float(np.median(np.abs(offsets))) if len(offsets) else 0.0
    within_120 = float(np.mean(np.abs(offsets) <= 0.12)) if len(offsets) else 0.0
    within_200 = float(np.mean(np.abs(offsets) <= 0.20)) if len(offsets) else 0.0

    important = [LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]
    joint_velocity = (
        np.diff(normalized[:, important, :3], axis=0, prepend=normalized[:1, important, :3])
        * fps
    )
    joint_energy = np.linalg.norm(joint_velocity, axis=-1)
    acceleration = np.diff(joint_velocity, axis=0) * fps
    jerk = np.diff(acceleration, axis=0) * fps
    jerk_curve = np.mean(np.linalg.norm(jerk, axis=-1), axis=1) if len(jerk) else np.array([])
    jerk_times = pose.times[2 : 2 + len(jerk_curve)]

    frame_visibility = np.mean(landmarks[:, KEY_JOINTS, 3] >= 0.45, axis=1)
    low_visibility = _intervals(pose.times, frame_visibility < 0.65)
    high_energy = _intervals(pose.times, energy >= np.percentile(energy, 75))
    abrupt = (
        _intervals(jerk_times, jerk_curve >= np.percentile(jerk_curve, 90))
        if len(jerk_curve)
        else []
    )

    left_elbow = angle_degrees(
        normalized[:, LEFT_SHOULDER, :3],
        normalized[:, LEFT_ELBOW, :3],
        normalized[:, LEFT_WRIST, :3],
    )
    right_elbow = angle_degrees(
        normalized[:, RIGHT_SHOULDER, :3],
        normalized[:, RIGHT_ELBOW, :3],
        normalized[:, RIGHT_WRIST, :3],
    )
    left_knee = angle_degrees(
        normalized[:, LEFT_HIP, :3],
        normalized[:, LEFT_KNEE, :3],
        normalized[:, LEFT_ANKLE, :3],
    )
    right_knee = angle_degrees(
        normalized[:, RIGHT_HIP, :3],
        normalized[:, RIGHT_KNEE, :3],
        normalized[:, RIGHT_ANKLE, :3],
    )
    wrist_ranges: list[float] = []
    for joint in (LEFT_WRIST, RIGHT_WRIST):
        xy = normalized[:, joint, :2]
        extent = np.percentile(xy, 95, axis=0) - np.percentile(xy, 5, axis=0)
        wrist_ranges.append(float(np.linalg.norm(extent)))

    shoulder_center = np.mean(normalized[:, [LEFT_SHOULDER, RIGHT_SHOULDER], :2], axis=1)
    hip_center = np.mean(normalized[:, [LEFT_HIP, RIGHT_HIP], :2], axis=1)
    torso_vector = shoulder_center - hip_center
    torso_lean = np.degrees(np.arctan2(torso_vector[:, 0], -torso_vector[:, 1]))
    shoulder_vector = normalized[:, RIGHT_SHOULDER, :2] - normalized[:, LEFT_SHOULDER, :2]
    # Measure tilt against an unsigned horizontal line.  Using atan2 directly can
    # report a nearly horizontal shoulder line as 180 degrees when the detector's
    # left/right ordering is reversed by the camera.
    shoulder_tilt = np.degrees(
        np.arctan2(np.abs(shoulder_vector[:, 1]), np.abs(shoulder_vector[:, 0]))
    )
    wrist_span = np.linalg.norm(
        normalized[:, LEFT_WRIST, :2] - normalized[:, RIGHT_WRIST, :2], axis=1
    )
    elbow_span = np.linalg.norm(
        normalized[:, LEFT_ELBOW, :2] - normalized[:, RIGHT_ELBOW, :2], axis=1
    )

    raw_shoulder_center = np.mean(
        landmarks[:, [LEFT_SHOULDER, RIGHT_SHOULDER], :2], axis=1
    )
    raw_hip_center = np.mean(landmarks[:, [LEFT_HIP, RIGHT_HIP], :2], axis=1)
    raw_torso_length = np.linalg.norm(raw_shoulder_center - raw_hip_center, axis=1)
    torso_scale = max(float(np.median(raw_torso_length)), 1e-4)
    hip_vertical_torso = raw_hip_center[:, 1] / torso_scale
    chest_gap_torso = (
        raw_hip_center[:, 1] - raw_shoulder_center[:, 1]
    ) / torso_scale
    shoulder_hip_offset = (
        raw_shoulder_center[:, 0] - raw_hip_center[:, 0]
    ) / torso_scale
    hip_horizontal_range = _range(raw_hip_center[:, 0])
    hip_vertical_range = _range(raw_hip_center[:, 1])
    elbow_balance = _balance(_range(left_elbow), _range(right_elbow))
    wrist_balance = _balance(wrist_ranges[0], wrist_ranges[1])
    knee_balance = _balance(_range(left_knee), _range(right_knee))

    energy_p50 = float(np.percentile(energy, 50))
    energy_p90 = float(np.percentile(energy, 90))
    peak_prominence = properties.get("prominences", np.array([], dtype=float))
    accent_sharpness = (
        float(np.median(peak_prominence) / max(energy_p50, 1e-6))
        if len(peak_prominence)
        else 0.0
    )
    stillness_ratio = float(np.mean(energy < max(energy_p90 * 0.15, 1e-5)))
    arm_energy = float(np.mean(joint_energy[:, :2]))
    center_energy = float(np.mean(joint_energy[:, 2:4]))
    leg_energy = float(np.mean(joint_energy[:, 4:6]))
    energy_total = max(arm_energy + center_energy + leg_energy, 1e-6)

    segments: list[dict[str, float]] = []
    for start in np.arange(0, media.duration, 4.0):
        end = min(float(start + 4.0), media.duration)
        selected = (pose.times >= start) & (pose.times < end)
        if not np.any(selected):
            continue
        segment_accents = accent_times[(accent_times >= start) & (accent_times < end)]
        segment_offsets = _nearest_signed(segment_accents, audio.beat_times)
        segment_hip = hip_vertical_torso[selected]
        segment_chest = chest_gap_torso[selected]
        segment_span = wrist_span[selected]
        segments.append(
            {
                "start": round(float(start), 1),
                "end": round(end, 1),
                "motion_energy": round(float(np.mean(energy[selected])), 3),
                "visibility": round(float(np.mean(frame_visibility[selected])), 3),
                "accent_count": float(len(segment_accents)),
                "median_beat_error": round(
                    float(np.median(np.abs(segment_offsets))) if len(segment_offsets) else 0.0,
                    3,
                ),
                "vertical_bounce": round(
                    float(np.percentile(segment_hip, 90) - np.percentile(segment_hip, 10)),
                    3,
                ),
                "chest_pulse": round(
                    float(
                        np.percentile(segment_chest, 90)
                        - np.percentile(segment_chest, 10)
                    ),
                    3,
                ),
                "arm_frame": round(float(np.percentile(segment_span, 75)), 3),
            }
        )

    arm_metrics = {
        "left_wrist_range": round(wrist_ranges[0], 3),
        "right_wrist_range": round(wrist_ranges[1], 3),
        "wrist_range_balance": round(wrist_balance, 1),
        "left_elbow_angle_range": round(_range(left_elbow), 1),
        "right_elbow_angle_range": round(_range(right_elbow), 1),
        "elbow_range_balance": round(elbow_balance, 1),
    }
    hip_metrics = {
        "horizontal_range_frame_widths": round(hip_horizontal_range, 3),
        "vertical_range_frame_heights": round(hip_vertical_range, 3),
        "center_motion_share": round(center_energy / energy_total, 3),
    }
    frame_metrics = {
        "median_wrist_span_torso_units": round(float(np.median(wrist_span)), 3),
        "median_elbow_span_torso_units": round(float(np.median(elbow_span)), 3),
        "torso_lean_p90_degrees": round(float(np.percentile(np.abs(torso_lean), 90)), 1),
        "shoulder_tilt_p90_degrees": round(
            float(np.percentile(np.abs(shoulder_tilt), 90)), 1
        ),
    }
    lower_body_metrics = {
        "left_knee_angle_range": round(_range(left_knee), 1),
        "right_knee_angle_range": round(_range(right_knee), 1),
        "knee_range_balance": round(knee_balance, 1),
        "leg_motion_share": round(leg_energy / energy_total, 3),
    }
    power_metrics = {
        "mean_motion_energy": round(float(np.mean(energy)), 3),
        "p90_motion_energy": round(energy_p90, 3),
        "accent_sharpness_proxy": round(accent_sharpness, 3),
        "arm_motion_share": round(arm_energy / energy_total, 3),
        "center_motion_share": round(center_energy / energy_total, 3),
        "leg_motion_share": round(leg_energy / energy_total, 3),
    }
    rhythm_metrics = {
        "median_signed_offset_seconds": round(median_offset, 3),
        "median_absolute_offset_seconds": round(median_abs_offset, 3),
        "accents_within_120ms": round(within_120, 3),
        "accents_within_200ms": round(within_200, 3),
        "rhythm_consistency_index": round((0.4 * within_120 + 0.6 * within_200) * 100, 1),
    }
    flow_metrics = {
        "abrupt_transition_count": float(len(abrupt)),
        "stillness_ratio": round(stillness_ratio, 3),
        "jerk_rms": round(float(np.sqrt(np.mean(jerk**2))) if jerk.size else 0.0, 3),
    }

    beat_bounce_ranges: list[float] = []
    for beat_start, beat_end in zip(audio.beat_times[:-1], audio.beat_times[1:]):
        selected = (pose.times >= beat_start) & (pose.times < beat_end)
        if np.count_nonzero(selected) >= 2:
            values = hip_vertical_torso[selected]
            beat_bounce_ranges.append(float(np.max(values) - np.min(values)))
    if len(beat_bounce_ranges) >= 4:
        beat_mean = float(np.mean(beat_bounce_ranges))
        bounce_consistency = float(
            np.clip(100 * (1 - np.std(beat_bounce_ranges) / max(beat_mean, 1e-5)), 0, 100)
        )
    else:
        bounce_consistency = 0.0
    average_knee_bend = 180 - (left_knee + right_knee) / 2
    if np.std(hip_vertical_torso) > 1e-6 and np.std(average_knee_bend) > 1e-6:
        bounce_knee_coupling = float(
            np.corrcoef(hip_vertical_torso, average_knee_bend)[0, 1]
        )
    else:
        bounce_knee_coupling = 0.0
    if not np.isfinite(bounce_knee_coupling):
        bounce_knee_coupling = 0.0
    groove_metrics = {
        "vertical_bounce_range_torso_units": round(_range(hip_vertical_torso), 3),
        "chest_pulse_range_torso_units": round(_range(chest_gap_torso), 3),
        "shoulder_hip_opposition_range": round(_range(shoulder_hip_offset), 3),
        "bounce_consistency": round(bounce_consistency, 1),
        "bounce_knee_coupling": round(bounce_knee_coupling, 3),
        "arm_frame_p75_torso_units": round(float(np.percentile(wrist_span, 75)), 3),
    }

    low_bounce = _merge_segment_spans(segments, "vertical_bounce", 0.12)
    low_chest_pulse = _merge_segment_spans(segments, "chest_pulse", 0.12)
    compact_frame = _merge_segment_spans(segments, "arm_frame", 0.85)
    rhythm_loose_spans = [
        (float(row["start"]), float(row["end"]))
        for row in segments
        if row["median_beat_error"] > 0.15
    ]

    confidence = float(np.clip(1 - pose.low_confidence_ratio, 0, 1))
    issues: list[dict[str, Any]] = []
    if confidence < 0.7:
        issues.append(
            _issue(
                "录制质量",
                "高",
                "部分身体关键点识别不稳定",
                f"整体姿态可信度约 {confidence * 100:.0f}%",
                "固定机位并让头、手、脚完整入镜；先改善画面，再判断动作细节。",
                low_visibility[:5],
                "关键点跳动会让手脚角度、胸髋起伏和节奏判断一起失真。",
                [
                    "把手机固定在腰到胸口高度，不要手持跟拍。",
                    "退后到头顶、双手最高点和双脚都留有余量。",
                    "用正面补光并避开与衣服颜色接近的背景，再录 8 个八拍复查。",
                ],
            )
        )
    if low_bounce:
        issues.append(
            _issue(
                "律动",
                "中",
                "部分时段身体上下律动偏平",
                "髋部在这些 4 秒窗口里的上下变化不足约 0.12 个躯干长度",
                "先把膝盖弹性和重心下沉练出来，再叠加手臂路线。",
                low_bounce[:5],
                "画面上容易显得人在做动作，但身体没有跟着拍子弹起来，律动会偏平。",
                [
                    "双脚与肩同宽，膝盖保持放松，先不做手部动作。",
                    "跟音乐做 8 拍“下沉—回弹”，重拍下沉，下一拍自然回弹。",
                    "连续做 4 轮，要求头、胸、髋一起产生小幅上下波动，不要只点头。",
                    "最后把原动作手臂加回去，重点复练报告标出的时段。",
                ],
            )
        )
    if low_chest_pulse:
        issues.append(
            _issue(
                "胸部与躯干",
                "中",
                "胸腔起伏和收放不够明显",
                "肩线与髋线之间的纵向变化在这些窗口低于约 0.12 个躯干长度",
                "单独练胸腔的提起、回收和下沉，让躯干也参与音乐重音。",
                low_chest_pulse[:5],
                "动作可能主要由手脚完成，胸口没有呼吸感，身体层次就会少一层。",
                [
                    "骨盆先保持稳定，肩膀放松，不要耸肩。",
                    "用 4 拍做“胸口微提—回中—微收—回中”，幅度小但方向要清楚。",
                    "再改成 2 拍一组，并让呼气配合胸口回收。",
                    "把这组胸腔收放加回标记时段，录像确认腰部没有代偿后仰。",
                ],
            )
        )
    if compact_frame:
        issues.append(
            _issue(
                "胳膊与框架",
                "中",
                "部分时段手臂框架偏收",
                "双腕张开度在这些窗口的较高值仍低于约 0.85 个躯干长度",
                "练清肩、肘、腕的延伸顺序，让动作终点更大、更完整。",
                compact_frame[:5],
                "手已经移动，但肘部和肩线没有把空间撑开时，动作会显得短、挤、没完成。",
                [
                    "每个手臂动作先定住终点，检查肩膀是否下沉、肘部是否打开。",
                    "用 4 拍完成一次“肩带动—肘延伸—腕到位—保持”，左右各做 8 次。",
                    "原速练习时，在每个重拍终点多停半拍，记住最大轮廓。",
                    "最后去掉停顿，但保留同样的肘腕距离和手指方向。",
                ],
            )
        )
    if wrist_balance < 75 or elbow_balance < 75:
        issues.append(
            _issue(
                "胳膊",
                "中",
                "左右手臂活动范围不均",
                f"手腕幅度平衡 {wrist_balance:.0f}/100，肘角变化平衡 {elbow_balance:.0f}/100",
                "做左右各 8 拍的肩—肘—腕分解，保持相同终点高度和伸展长度。",
                impact="左右轮廓不一致会让动作看起来一边完整、一边没有做到位。",
                drill_steps=[
                    "面对镜子只做左侧 8 拍，再只做右侧 8 拍。",
                    "在相同动作终点比较手腕高度、肘角和离身体的距离。",
                    "最后双侧一起做 3 轮，并用定格画面复查。",
                ],
            )
        )
    if knee_balance < 75:
        issues.append(
            _issue(
                "腿部",
                "中",
                "左右膝屈伸幅度差异较大",
                f"左右膝角变化平衡 {knee_balance:.0f}/100",
                "慢速做左右换重心，检查两侧屈膝深度和落地时间是否一致。",
            )
        )
    if frame_metrics["torso_lean_p90_degrees"] > 22:
        issues.append(
            _issue(
                "框架",
                "中",
                "躯干轴线变化较大",
                f"躯干倾斜绝对值的 90 分位为 {frame_metrics['torso_lean_p90_degrees']:.1f}°",
                "先固定胸腔—骨盆轴线，再加入手臂；用正面录像检查身体是否被手臂带偏。",
            )
        )
    if within_120 < 0.55 or within_200 < 0.6:
        issues.append(
            _issue(
                "节奏",
                "高" if within_120 < 0.35 else "中",
                "卡点有些松，动作终点没有稳定落在拍上",
                f"每 10 个动作重音里约有 {within_120 * 10:.0f} 个紧贴节拍，其余会稍早或稍晚",
                "先把脚下重心落拍练准，再把手臂和胸腔动作逐层加回来。",
                rhythm_loose_spans[:5],
                "观感上会出现有几拍很准、有几拍像在追音乐，动作不够整齐利落。",
                [
                    "先听音乐拍手数 1–8，连续 4 轮都不抢拍或拖拍。",
                    "只做脚下换重心，每个重拍到位后定住半拍。",
                    "加入胸髋律动，再加入手臂；每次只增加一层。",
                    "原速录像，检查动作终点是否和鼓点同时发生，而不是动作开始时碰到鼓点。",
                ],
            )
        )
    if abs(median_offset) > 0.1:
        direction = "慢" if median_offset > 0 else "早"
        issues.append(
            _issue(
                "节奏",
                "中",
                f"动作重音整体略{direction}",
                f"多数动作重音整体比拍子稍{direction}，幅度约为十分之一秒",
                "用 0.75× 速度循环 8 拍，先让重心终点落拍，再补手部细节。",
                impact="连续的轻微提前或滞后会让人感觉动作与音乐没有咬在一起。",
                drill_steps=[
                    "选最清楚的 8 拍，把每个动作终点写成 1–8 的落点。",
                    "0.75× 速度只做重心和动作终点，不追求完整路线。",
                    "连续 3 轮落点稳定后恢复原速，再加入连接动作。",
                ],
            )
        )
    if accent_sharpness < 0.2:
        issues.append(
            _issue(
                "发力表现",
                "观察",
                "动作重音边界偏柔",
                f"动作能量峰值突出度代理为 {accent_sharpness:.2f}",
                "练习“蓄力—到点—释放”三段式，让重拍的终点更清楚。",
                high_energy[:5],
            )
        )
    elif accent_sharpness > 1.6:
        issues.append(
            _issue(
                "发力表现",
                "观察",
                "动作能量峰值较尖锐",
                f"动作能量峰值突出度代理为 {accent_sharpness:.2f}",
                "保留重拍力度，同时检查重拍之后是否有缓冲和连续衔接。",
                abrupt[:5],
            )
        )
    if len(abrupt) >= 6:
        issues.append(
            _issue(
                "衔接",
                "中",
                "快速变化时段较多",
                f"检测到 {len(abrupt)} 组速度/加速度快速变化区间",
                "把这些时段前后各扩展 2 拍慢练，减少动作终点之间的急停和补动作。",
                abrupt[:6],
                "动作到点后如果立刻断掉或临时补路线，观感会碎，发力也无法顺利传到下一拍。",
                [
                    "截取标记点前后各 2 拍，先确认前一个动作的终点和下一个动作的起点。",
                    "用 0.5× 速度连续画出中间路线，不在两个终点之间停住。",
                    "提高到 0.75×，保持前一拍释放的力量自然带入下一拍。",
                    "最后原速做 3 次；若仍有急停，减少动作幅度后再逐步放大。",
                ],
            )
        )
    if not issues:
        issues.append(
            _issue(
                "整体",
                "观察",
                "未发现达到当前阈值的明显问题",
                "主要指标处于当前工程阈值范围内",
                "继续结合参考视频检查动作还原度和编舞特定要求。",
            )
        )

    level_order = {"高": 0, "中": 1, "观察": 2, "低": 3}
    issues.sort(key=lambda item: level_order.get(item["level"], 9))
    recommendations = [item["suggestion"] for item in issues[:6]]
    recommendations.extend(
        [
            "每次只选一个问题练 3 轮：0.75× 慢练、原速定格、录像复查。",
            "如需判断动作是否正确还原，请提供同编舞、同段落、单人且视角相近的参考视频。",
        ]
    )

    top_titles = "、".join(item["title"] for item in issues[:3])
    coach_summary = (
        f"这段视频最值得先改的是：{top_titles}。"
        "建议一次只练一个问题，先把身体律动和动作终点练清楚，再追求完整速度。"
    )

    return SoloAnalysisResult(
        duration_seconds=round(media.duration, 2),
        source_fps=round(media.fps, 3),
        resolution=(media.width, media.height),
        analysis_fps=round(fps, 3),
        pose_backend=pose.backend,
        pose_confidence=round(confidence, 3),
        estimated_tempo_bpm=round(audio.tempo, 2),
        detected_beats=int(len(audio.beat_times)),
        motion_accents=int(len(accent_times)),
        timing_offset_seconds=round(median_offset, 3),
        median_absolute_offset_seconds=round(median_abs_offset, 3),
        accents_within_120ms=round(within_120, 3),
        accents_within_200ms=round(within_200, 3),
        left_wrist_range=round(wrist_ranges[0], 3),
        right_wrist_range=round(wrist_ranges[1], 3),
        hip_horizontal_range=round(hip_horizontal_range, 3),
        high_energy_intervals=high_energy,
        abrupt_transition_intervals=abrupt,
        low_visibility_intervals=low_visibility,
        segments=segments,
        arm_metrics=arm_metrics,
        hip_metrics=hip_metrics,
        frame_metrics=frame_metrics,
        lower_body_metrics=lower_body_metrics,
        power_metrics=power_metrics,
        rhythm_metrics=rhythm_metrics,
        flow_metrics=flow_metrics,
        groove_metrics=groove_metrics,
        coach_summary=coach_summary,
        issues=issues,
        recommendations=recommendations,
        metadata=metadata,
    )
