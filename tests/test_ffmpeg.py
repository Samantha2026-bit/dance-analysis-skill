from __future__ import annotations

from dance_pose_coach.ffmpeg import find_ffmpeg


def test_system_ffmpeg_is_preferred(monkeypatch):
    monkeypatch.setattr("dance_pose_coach.ffmpeg.shutil.which", lambda _name: "/usr/bin/ffmpeg")
    assert find_ffmpeg() == "/usr/bin/ffmpeg"


def test_bundled_ffmpeg_is_used_when_system_copy_is_missing(monkeypatch, tmp_path):
    executable = tmp_path / "ffmpeg"
    executable.write_bytes(b"test")
    monkeypatch.setattr("dance_pose_coach.ffmpeg.shutil.which", lambda _name: None)
    monkeypatch.setattr("imageio_ffmpeg.get_ffmpeg_exe", lambda: str(executable))
    assert find_ffmpeg() == str(executable)
