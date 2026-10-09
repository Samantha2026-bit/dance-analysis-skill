from __future__ import annotations

import subprocess
from pathlib import Path

import cv2

from .ffmpeg import find_ffmpeg
from .models import MediaInfo


class MediaError(RuntimeError):
    pass


def require_ffmpeg() -> str:
    executable = find_ffmpeg()
    if executable is None:
        raise MediaError("未找到随项目安装的 FFmpeg。请重新运行 `python -m pip install -e .`。")
    return executable


def inspect_video(path: str | Path) -> MediaInfo:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise MediaError(f"视频不存在：{path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise MediaError(f"无法读取视频：{path.name}")
    # iPhone MOV files often store portrait orientation as metadata rather than
    # physically rotating pixels. Ask OpenCV's FFmpeg backend to honor it.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frames <= 0:
        raise MediaError("视频缺少有效的帧率或时长信息。")
    return MediaInfo(path, frames / fps, fps, width, height)


def extract_audio(video: str | Path, output: str | Path, max_duration: float) -> Path:
    """Extract mono audio at librosa-friendly 22.05 kHz."""
    ffmpeg = require_ffmpeg()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(Path(video).resolve()),
        "-t",
        str(max_duration),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "22050",
        str(output.resolve()),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise MediaError(f"音频提取失败：{exc.stderr.strip()}") from exc
    return output


def validate_duration(info: MediaInfo, max_duration: float) -> None:
    if info.duration < 2:
        raise MediaError("视频过短，至少需要 2 秒。")
    if info.duration > max_duration:
        raise MediaError(
            f"当前版本最多分析 {max_duration:.0f} 秒，上传视频为 {info.duration:.1f} 秒。"
        )
