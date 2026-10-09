from __future__ import annotations

import shutil
from pathlib import Path


def find_ffmpeg() -> str | None:
    """Find system FFmpeg first, then the executable bundled by imageio-ffmpeg."""
    system_executable = shutil.which("ffmpeg")
    if system_executable:
        return system_executable
    try:
        import imageio_ffmpeg

        bundled_executable = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None
    path = Path(bundled_executable)
    return str(path) if path.is_file() else None
