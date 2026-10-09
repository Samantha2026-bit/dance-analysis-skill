from __future__ import annotations

import argparse
from pathlib import Path

from .config import AnalysisConfig
from .pipeline import analyze, analyze_solo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地舞蹈姿态与节奏对比")
    parser.add_argument("user_video", help="用户舞蹈视频")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--reference", help="本地参考视频")
    source.add_argument("--reference-url", default="", help="参考视频 URL")
    parser.add_argument("--query", default="", help="自动检索时使用的曲名/舞名")
    parser.add_argument("--mirror", choices=["auto", "never", "always"], default="auto")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-duration", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument(
        "--allow-online",
        action="store_true",
        help="允许 Shazam 识曲、在线视频搜索和下载；默认完全本地",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AnalysisConfig(
        analysis_fps=args.fps,
        max_duration_seconds=args.max_duration,
        mirror_mode=args.mirror,
        output_root=args.output,
        allow_online_services=args.allow_online,
    )

    def progress(fraction: float, label: str) -> None:
        print(f"[{fraction * 100:3.0f}%] {label}")

    single_mode = not args.reference and not args.reference_url and not args.allow_online
    if single_mode:
        result, paths = analyze_solo(args.user_video, config=config, progress=progress)
        print("单视频分析完成（没有参考视频，不生成动作还原总分）")
        print(f"姿态可信度：{result.pose_confidence * 100:.0f}%")
    else:
        result, paths = analyze(
            args.user_video,
            reference_video=args.reference,
            reference_url=args.reference_url,
            search_query=args.query,
            config=config,
            progress=progress,
        )
        print(f"总分：{result.overall_score}/100")
    print(f"HTML 报告：{paths['html']}")
    print(f"JSON 指标：{paths['json']}")


if __name__ == "__main__":
    main()
