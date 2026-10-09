from __future__ import annotations

import argparse
import html
import os
import secrets
from pathlib import Path

# Do not send Gradio usage analytics. This must be set before importing Gradio.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

from .config import AnalysisConfig
from .pipeline import analyze, analyze_solo, identify_and_search
from .report import render_html
from .runtime import inspect_runtime, lan_address
from .solo_report import render_solo_html

MIRROR_MAP = {"自动判断": "auto", "不镜像": "never", "强制镜像": "always"}


def _search(user_video: str, query: str, allow_online: bool) -> tuple[str, str]:
    if not user_video:
        raise gr.Error("请先上传用户视频。")
    if not allow_online:
        raise gr.Error("在线服务默认关闭。勾选“允许在线服务”后才会识曲和检索。")
    recognition, candidates = identify_and_search(
        user_video,
        query,
        AnalysisConfig(allow_online_services=True),
    )
    identified = (
        " - ".join(part for part in [recognition.artist, recognition.title] if part) or "未识别"
    )
    lines = [
        f"识别结果：**{html.escape(identified)}**"
        f"（{recognition.method} / {recognition.confidence}）",
        "",
        "候选原版（分析时 URL 留空会自动使用第 1 条，请先确认版本）：",
    ]
    for index, candidate in enumerate(candidates, 1):
        duration = f" · {candidate.duration:.0f}s" if candidate.duration else ""
        lines.append(
            f"{index}. [{html.escape(candidate.title)}]({candidate.url}) · "
            f"{html.escape(candidate.uploader)}{duration}"
        )
    if not candidates:
        lines.append("未找到候选。请补充曲名/舞名，或上传本地参考视频。")
    return "\n".join(lines), recognition.query


def _analyze_ui(
    user_video: str,
    reference_video: str | None,
    reference_url: str,
    query: str,
    mirror_label: str,
    analysis_fps: float,
    max_duration: float,
    allow_online: bool,
    progress=gr.Progress(track_tqdm=False),
) -> tuple[str, str, str, str]:
    if not user_video:
        raise gr.Error("请上传你的舞蹈视频。")
    config = AnalysisConfig(
        analysis_fps=float(analysis_fps),
        max_duration_seconds=float(max_duration),
        mirror_mode=MIRROR_MAP[mirror_label],
        output_root=Path("runs"),
        allow_online_services=bool(allow_online),
    )

    def update(fraction: float, label: str) -> None:
        progress(fraction, desc=label)

    try:
        single_mode = not reference_video and not reference_url.strip() and not allow_online
        if single_mode:
            result, paths = analyze_solo(user_video=user_video, config=config, progress=update)
        else:
            result, paths = analyze(
                user_video=user_video,
                reference_video=reference_video,
                reference_url=reference_url,
                search_query=query,
                config=config,
                progress=update,
            )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc
    if single_mode:
        status = (
            f"单视频分析完成：姿态可信度 {result.pose_confidence * 100:.0f}%；"
            f"报告保存在 {paths['html'].parent}。没有参考视频，因此不生成动作还原总分。"
        )
        report_html = render_solo_html(result)
    else:
        status = (
            f"对比分析完成：总分 {result.overall_score}/100；报告保存在 "
            f"{paths['html'].parent}。姿态与报告在运行本程序的电脑上计算。"
        )
        report_html = render_html(result)
    return report_html, str(paths["html"]), str(paths["json"]), status


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Dance Pose Coach", theme=gr.themes.Soft(primary_hue="violet")) as demo:
        gr.Markdown(
            "# Dance Pose Coach · 本地舞蹈分析\n"
            "只上传你的舞蹈即可做单视频分析；再上传参考视频则自动做动作对比。默认不调用任何在线服务。"
        )
        with gr.Row():
            user_video = gr.Video(label="① 你的舞蹈视频", sources=["upload"])
            reference_video = gr.Video(
                label="② 本地原版/示范（可选；不上传则做单视频分析）",
                sources=["upload"],
            )
        with gr.Accordion("自动识曲与原版检索", open=True):
            allow_online = gr.Checkbox(
                value=False,
                label="允许在线服务（会连接识曲/视频平台；完全本地使用请勿勾选）",
            )
            query = gr.Textbox(
                label="曲名 / 舞名（可选，填写后优先使用）",
                placeholder="例如：NewJeans Super Shy dance practice",
            )
            search_button = gr.Button("识别并查看候选", variant="secondary")
            candidates = gr.Markdown("尚未检索。")
            reference_url = gr.Textbox(
                label="确认后的参考视频 URL（可选）",
                placeholder="留空时，分析会自动使用检索结果第 1 条",
            )
            search_button.click(_search, [user_video, query, allow_online], [candidates, query])
        with gr.Accordion("分析设置", open=False):
            mirror = gr.Radio(list(MIRROR_MAP), value="自动判断", label="左右/镜像处理")
            fps = gr.Slider(5, 15, value=10, step=1, label="分析帧率（较旧电脑推荐 6–8）")
            duration = gr.Slider(15, 180, value=120, step=15, label="最大分析时长（秒）")
        gr.Markdown(
            "提示：单视频模式分析节奏、幅度和连续性，不判断动作是否正确。"
            "对比模式的参考版本必须与练习编舞一致。"
        )
        analyze_button = gr.Button("开始本地分析", variant="primary")
        status = gr.Textbox(label="状态", interactive=False)
        report = gr.HTML(label="分析报告")
        with gr.Row():
            html_file = gr.File(label="下载 HTML 报告")
            json_file = gr.File(label="下载 JSON 原始指标")
        analyze_button.click(
            _analyze_ui,
            [
                user_video,
                reference_video,
                reference_url,
                query,
                mirror,
                fps,
                duration,
                allow_online,
            ],
            [report, html_file, json_file, status],
        )
    return demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dance Pose Coach 本地网页界面")
    parser.add_argument(
        "--mode",
        choices=["local", "lan"],
        default="local",
        help="local 仅本机访问；lan 允许同一局域网中的手机访问",
    )
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--password", help="LAN 模式访问密码；省略时生成一次性随机密码")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开桌面浏览器")
    return parser


def launch_app(mode: str, port: int, password: str | None, inbrowser: bool) -> None:
    info = inspect_runtime()
    if not info.supported:
        raise SystemExit(
            f"当前运行环境不受支持：{info.system} {info.machine}, "
            f"Python {info.python}。{info.message}"
        )
    launch_options: dict = {
        "server_port": port,
        "share": False,
        "show_error": True,
        "inbrowser": inbrowser,
    }
    if mode == "lan":
        access_password = (
            password or os.environ.get("DANCE_COACH_PASSWORD") or secrets.token_urlsafe(8)
        )
        launch_options.update(
            server_name="0.0.0.0",
            auth=("dance", access_password),
            auth_message="仅供同一可信局域网中的本人设备使用。",
        )
        print("\n手机模式已开启（不创建公网分享链接）。")
        print(f"手机地址：{lan_address(port)}")
        print("用户名：dance")
        print(f"密码：{access_password}")
        print("请只在可信 Wi-Fi 使用；关闭此窗口即可停止访问。\n")
    else:
        launch_options["server_name"] = "127.0.0.1"
    build_app().queue(default_concurrency_limit=1).launch(**launch_options)


def main() -> None:
    args = build_parser().parse_args()
    launch_app(args.mode, args.port, args.password, not args.no_browser and args.mode == "local")


if __name__ == "__main__":
    main()
