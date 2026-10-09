from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from . import __version__
from .alignment import choose_mirror
from .audio import analyze_audio
from .config import AnalysisConfig
from .media import extract_audio, inspect_video, validate_duration
from .metrics import build_result
from .models import AnalysisResult, RecognitionResult, SearchCandidate, SoloAnalysisResult
from .pose import extract_pose
from .recognition import recognize_track
from .report import write_reports
from .retrieval import download_reference, search_reference
from .solo import build_solo_result
from .solo_report import write_solo_reports

ProgressCallback = Callable[[float, str], None]


def _noop_progress(_: float, __: str) -> None:
    pass


def new_run_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root.expanduser().resolve() / f"{stamp}-{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def identify_and_search(
    user_video: str | Path,
    query_override: str = "",
    config: AnalysisConfig | None = None,
) -> tuple[RecognitionResult, list[SearchCandidate]]:
    config = config or AnalysisConfig()
    if not config.allow_online_services:
        raise RuntimeError("在线识曲/检索默认关闭。请明确启用在线服务，或直接上传本地参考视频。")
    run_dir = new_run_dir(config.output_root / "search")
    audio_path = extract_audio(
        user_video,
        run_dir / "sample.wav",
        min(config.max_duration_seconds, 45),
    )
    recognition = recognize_track(audio_path, user_video, allow_online=True)
    query = query_override.strip() or recognition.query
    candidates = search_reference(query)
    if not config.keep_intermediates:
        shutil.rmtree(run_dir, ignore_errors=True)
    return recognition, candidates


def analyze_solo(
    user_video: str | Path,
    config: AnalysisConfig | None = None,
    progress: ProgressCallback = _noop_progress,
) -> tuple[SoloAnalysisResult, dict[str, Path]]:
    """Analyze one local video without inventing a reference-based pose score."""
    config = config or AnalysisConfig()
    config.validate()
    run_dir = new_run_dir(config.output_root)
    progress(0.05, "检查视频")
    user_info = inspect_video(user_video)
    validate_duration(user_info, config.max_duration_seconds)
    audio_path = extract_audio(user_video, run_dir / "user.wav", config.max_duration_seconds)
    progress(0.20, "提取姿态")
    user_pose = extract_pose(user_video, config)
    progress(0.78, "分析节拍与动作重音")
    user_audio = analyze_audio(audio_path)
    progress(0.90, "整理单视频动作特征")
    result = build_solo_result(
        user_pose,
        user_audio,
        user_info,
        metadata={
            "version": __version__,
            "analysis_mode": "single-video",
            "analysis_fps": config.analysis_fps,
            "privacy": "local",
        },
    )
    report_paths = write_solo_reports(result, run_dir)
    if not config.keep_intermediates:
        audio_path.unlink(missing_ok=True)
    progress(1.0, "单视频报告已生成")
    return result, report_paths


def analyze(
    user_video: str | Path,
    reference_video: str | Path | None = None,
    reference_url: str = "",
    search_query: str = "",
    config: AnalysisConfig | None = None,
    progress: ProgressCallback = _noop_progress,
) -> tuple[AnalysisResult, dict[str, Path]]:
    config = config or AnalysisConfig()
    config.validate()
    if not reference_video and not config.allow_online_services:
        raise RuntimeError("未提供本地参考视频，且在线服务未启用。请上传参考视频或明确允许联网。")
    run_dir = new_run_dir(config.output_root)
    progress(0.03, "检查用户视频")
    user_info = inspect_video(user_video)
    validate_duration(user_info, config.max_duration_seconds)
    user_audio_path = extract_audio(user_video, run_dir / "user.wav", config.max_duration_seconds)
    recognition = recognize_track(
        user_audio_path,
        user_video,
        allow_online=config.allow_online_services,
    )

    source = "本地文件"
    resolved_reference: Path
    if reference_video:
        resolved_reference = Path(reference_video).expanduser().resolve()
    else:
        progress(0.12, "检索原版示范")
        selected_url = reference_url.strip()
        if not selected_url:
            candidates = search_reference(search_query.strip() or recognition.query)
            if not candidates:
                raise RuntimeError("未找到原版候选，请填写舞蹈名或直接提供参考视频。")
            selected_url = candidates[0].url
        source = selected_url
        resolved_reference = download_reference(selected_url, run_dir / "reference_download")

    progress(0.22, "检查参考视频")
    reference_info = inspect_video(resolved_reference)
    validate_duration(reference_info, config.max_duration_seconds)
    ref_audio_path = extract_audio(
        resolved_reference, run_dir / "reference.wav", config.max_duration_seconds
    )
    progress(0.30, "提取用户姿态")
    user_pose = extract_pose(user_video, config)
    progress(0.52, "提取参考姿态")
    reference_pose = extract_pose(resolved_reference, config)
    progress(0.73, "分析节拍与动作重音")
    user_audio = analyze_audio(user_audio_path)
    reference_audio = analyze_audio(ref_audio_path)
    progress(0.82, "对齐骨架序列")
    aligned_user, path, mirror_used = choose_mirror(user_pose, reference_pose, config.mirror_mode)
    progress(0.90, "计算分区指标与建议")
    result = build_result(
        user_pose,
        reference_pose,
        aligned_user,
        path,
        user_audio,
        reference_audio,
        mirror_used,
        metadata={
            "version": __version__,
            "recognized_title": recognition.title,
            "recognized_artist": recognition.artist,
            "recognition_method": recognition.method,
            "recognition_confidence": recognition.confidence,
            "reference_source": source,
            "user_duration": round(user_info.duration, 2),
            "reference_duration": round(reference_info.duration, 2),
            "analysis_fps": config.analysis_fps,
            "user_pose_backend": user_pose.backend,
            "reference_pose_backend": reference_pose.backend,
        },
    )
    report_paths = write_reports(result, run_dir)
    if not config.keep_intermediates:
        for audio in (user_audio_path, ref_audio_path):
            audio.unlink(missing_ok=True)
    progress(1.0, "报告已生成")
    return result, report_paths
