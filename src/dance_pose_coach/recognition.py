from __future__ import annotations

import asyncio
import re
from pathlib import Path

from .models import RecognitionResult


def query_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    cleaned = re.sub(r"[_\-.]+", " ", stem)
    cleaned = re.sub(
        r"\b(cover|dance|practice|mirror|mirrored|教程|练习|翻跳)\b",
        " ",
        cleaned,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


async def _recognize_shazam(audio_path: Path) -> dict:
    from shazamio import Shazam  # optional dependency

    return await Shazam().recognize(str(audio_path))


def recognize_track(
    audio_path: str | Path,
    video_path: str | Path,
    *,
    allow_online: bool = False,
) -> RecognitionResult:
    """Use optional Shazam recognition, then degrade cleanly to the filename."""
    filename_query = query_from_filename(video_path)
    if allow_online:
        try:
            payload = asyncio.run(_recognize_shazam(Path(audio_path)))
            track = payload.get("track", {}) if isinstance(payload, dict) else {}
            title = str(track.get("title", "")).strip()
            artist = str(track.get("subtitle", "")).strip()
            if title:
                return RecognitionResult(
                    query=f"{artist} {title} official dance practice".strip(),
                    title=title,
                    artist=artist,
                    confidence="medium",
                    method="shazam",
                    raw={"track_key": track.get("key")},
                )
        except Exception:  # optional network integration must never block local analysis
            pass
    return RecognitionResult(
        query=f"{filename_query} official dance practice".strip(),
        title=filename_query,
        confidence="low",
        method="filename",
    )
