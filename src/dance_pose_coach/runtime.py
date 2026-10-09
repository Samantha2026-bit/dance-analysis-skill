from __future__ import annotations

import platform
import socket
import sys
from dataclasses import dataclass

from .ffmpeg import find_ffmpeg


@dataclass(frozen=True)
class RuntimeInfo:
    system: str
    machine: str
    python: str
    ffmpeg: bool
    pose_cpu: bool
    supported: bool
    message: str


def inspect_runtime() -> RuntimeInfo:
    from .pose_opencv import probe_opencv_pose

    system = platform.system()
    machine = platform.machine().lower()
    python = platform.python_version()
    python_ok = (3, 9) <= sys.version_info[:2] < (3, 13)
    if system == "Darwin":
        platform_ok = machine in {"x86_64", "arm64"}
        platform_message = "支持 macOS Intel 与 Apple Silicon。"
    elif system == "Windows":
        platform_ok = machine in {"amd64", "x86_64"}
        platform_message = "支持 64 位 Windows；Windows on ARM 未验证。"
    elif system == "Linux":
        platform_ok = machine in {"x86_64", "amd64", "aarch64", "arm64"}
        platform_message = "Linux 为尽力支持的平台。"
    else:
        platform_ok = False
        platform_message = f"尚未支持 {system or '未知系统'}。"
    if not python_ok:
        platform_message += " 请使用 Python 3.9–3.12。"
    pose_cpu, pose_message = probe_opencv_pose()
    if not pose_cpu:
        platform_message += f" CPU 姿态模型不可用：{pose_message}"
    return RuntimeInfo(
        system=system,
        machine=machine,
        python=python,
        ffmpeg=find_ffmpeg() is not None,
        pose_cpu=pose_cpu,
        supported=platform_ok and python_ok and pose_cpu,
        message=platform_message,
    )


def lan_address(port: int) -> str:
    """Return a best-effort private-network URL without contacting the internet."""
    address = ""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 1))  # TEST-NET; connect only selects a local route.
        address = sock.getsockname()[0]
    except OSError:
        pass
    finally:
        sock.close()
    if not address or address.startswith("127."):
        try:
            address = socket.gethostbyname(socket.gethostname())
        except OSError:
            address = "<电脑的局域网 IP>"
    return f"http://{address}:{port}"


def doctor_main() -> None:
    info = inspect_runtime()
    print(f"系统：{info.system} {info.machine}")
    print(f"Python：{info.python}（{'兼容' if info.supported else '需要检查'}）")
    print(f"FFmpeg：{'已找到' if info.ffmpeg else '未找到'}")
    print(f"姿态识别：{'OpenCV CPU 模式可用' if info.pose_cpu else 'CPU 模式不可用'}")
    print(info.message)
    if not info.supported or not info.ffmpeg:
        raise SystemExit(1)
