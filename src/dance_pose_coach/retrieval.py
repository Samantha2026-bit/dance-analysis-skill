from __future__ import annotations

from pathlib import Path

from .ffmpeg import find_ffmpeg
from .models import SearchCandidate


class RetrievalError(RuntimeError):
    pass


def search_reference(query: str, limit: int = 5) -> list[SearchCandidate]:
    if not query.strip():
        return []
    try:
        import yt_dlp

        options = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    except Exception as exc:  # network and extractor errors have many concrete types
        raise RetrievalError(f"原版视频搜索失败：{exc}") from exc
    candidates: list[SearchCandidate] = []
    for entry in info.get("entries", []) if info else []:
        if not entry:
            continue
        url = entry.get("webpage_url") or entry.get("url")
        if url and not str(url).startswith("http"):
            url = f"https://www.youtube.com/watch?v={url}"
        candidates.append(
            SearchCandidate(
                title=str(entry.get("title") or "未命名视频"),
                url=str(url),
                uploader=str(entry.get("uploader") or entry.get("channel") or ""),
                duration=entry.get("duration"),
            )
        )
    return candidates


def download_reference(url: str, output_dir: str | Path) -> Path:
    """Download one user-confirmed/search-selected reference; never playlists."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template = str(output_dir / "reference.%(ext)s")
    try:
        import yt_dlp

        options = {
            "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "outtmpl": template,
            "merge_output_format": "mp4",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            options["ffmpeg_location"] = ffmpeg
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            prepared = Path(ydl.prepare_filename(info))
    except Exception as exc:
        raise RetrievalError(f"原版视频下载失败：{exc}") from exc
    merged = output_dir / "reference.mp4"
    if merged.exists():
        return merged
    if prepared.exists():
        return prepared
    matches = sorted(output_dir.glob("reference.*"))
    if not matches:
        raise RetrievalError("下载器未生成可用视频文件。")
    return matches[0]
