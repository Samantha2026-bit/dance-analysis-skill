# Dance Pose Coach

一个**本地优先**的舞蹈分析工具。支持单视频动作分析和参考视频对比，面向 macOS Intel、macOS Apple Silicon 和 64 位 Windows；手机可通过同一可信局域网访问自己电脑上的分析界面。

只上传自己的跳舞视频，程序会用教练式报告分析上下律动、胸腔收放、胳膊延展、髋腿参与、身体框架、节奏、动作能量与衔接；同时上传参考视频，则进一步分析动作还原度、分区角度差异、发力表现差异和具体问题时段。默认模式不需要 GPU 或 OpenGL。

> 当前版本是轻量 MVP。评分用于辅助自练，不是专业赛事评分、医学或康复诊断。

## 隐私架构（先读这一节）

| 使用方式 | 视频在哪里计算 | 是否经过互联网 | 适合场景 |
| --- | --- | --- | --- |
| 桌面本机模式 | 当前电脑 | 默认不经过 | 隐私要求最高，推荐 |
| 手机局域网模式 | 当前电脑；手机把文件通过 Wi-Fi 传给电脑 | 不创建公网链接；默认不访问第三方 | 想用手机选择/拍摄视频，但家中有电脑 |
| 在线识曲/检索（可选） | 姿态仍在电脑；识曲/下载访问第三方 | 会经过互联网 | 没有本地参考视频 |

默认情况下，在线识曲、视频搜索和下载全部关闭，Gradio 使用统计也关闭。要联网必须在界面明确勾选“允许在线服务”，或在命令行添加 `--allow-online`。

手机模式**不是手机浏览器内计算**：Python、OpenCV CPU 姿态模型、FFmpeg 和报告生成都运行在你的电脑上。手机上传会经过电脑上的本地 Gradio 服务，但不会上传到本项目作者的服务器，也不会生成 Gradio 公网分享链接。局域网连接使用 HTTP，不是端到端加密；程序会生成访问密码，但仍只应在你信任的私人 Wi-Fi 上使用。公共 Wi-Fi 或高度敏感的视频请改用桌面本机模式。

纯浏览器本地分析目前不可行而没有实现：现有 Python 指标管线依赖 OpenCV、FFmpeg 和 librosa，将其完整移植到 WebAssembly/JavaScript 是独立项目，且移动设备性能与视频编解码支持差异很大。

## 功能

- Gradio 本地交互界面，支持本地用户视频和参考视频
- 不提供参考视频时自动生成单视频分析报告，不再报错
- 单视频报告先给关键问题和改善方法，BPM、延迟、比例等技术数字收在次要区域
- 分析上下律动、胸腔起伏与收放、肩髋错动、胳膊框架、髋腿、节奏、发力表现和衔接
- 每个问题提供可见影响、判断依据、重点时段、训练目标和 3–4 步具体练法
- 可选在线 Shazam 识曲及 `yt-dlp` 原版检索（必须明确启用）
- OpenCV DNN 纯 CPU 模式逐帧提取 33 个人体关键点
- CPU 模型随项目打包，首次分析不需要联网下载
- 可选保留旧版 MediaPipe 后端，但默认不会启动它
- 读取手机 MOV 方向元数据，正确处理常见竖屏录制
- 自动判断镜像方向，使用约束 DTW 对齐不同速度的视频
- 分析身体分区角度、BPM、动作重音、运动能量与流畅度
- 输出自包含 HTML、Markdown 和 JSON 报告
- 跨平台运行检查；FFmpeg 随 Python 依赖自动安装

## 支持范围与系统要求

已配置并面向以下组合：

- macOS 12 或更高：Intel x86_64、Apple Silicon arm64
- Windows 10/11：x86-64（64 位）；Windows on ARM 未验证
- Python 3.9–3.12（推荐 3.11）
- FFmpeg 由项目自动安装，不要求 Homebrew、winget 或手工配置 `PATH`
- 建议 8 GB 以上内存

0.3.0 起默认使用项目内置的 OpenCV Zoo MediaPipe Pose ONNX 模型和 OpenCV DNN CPU 后端，因此可在没有图形会话的终端、Codex 等隔离环境中运行，并避免 `kGpuService` / `NSOpenGLPixelFormat` 错误。旧版 `mp.solutions.pose` 仅作为可选兼容后端。

该项目不使用 GPU 或超大模型。默认最多分析 120 秒、10 FPS；较旧电脑建议降低到 6–8 FPS。处理时间取决于分辨率、人体遮挡和电脑性能。内置模型来自 OpenCV Zoo，模型目录采用 Apache 2.0 许可证。

## 安装

### macOS（Intel 与 Apple Silicon）

先安装 Python 3.11。使用 Homebrew 时：

```bash
brew install python@3.11
```

然后在项目目录运行（不要写死 `/usr/local`，Apple Silicon 的 Homebrew 通常位于 `/opt/homebrew`）：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
dance-pose-doctor
```

如果 macOS 阻止双击 `.command`，先在终端运行：

```bash
chmod +x start-*.command start-*.sh
```

### Windows 10/11（64 位）

1. 从 python.org 安装 64 位 Python 3.11，并勾选 “Add Python to PATH”。
2. 在项目目录的 PowerShell 运行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
dance-pose-doctor
```

如果 PowerShell 不允许激活脚本，可直接使用 `.\.venv\Scripts\python.exe -m pip install -e .`，运行时也可直接双击批处理文件。

### 可选在线识曲

```bash
python -m pip install -e ".[recognition]"
```

安装并不会自动联网；仍需在每次使用时明确允许在线服务。Shazam、视频平台与下载器可能受网络、地区、版权、服务条款或上游接口变化影响。

### 可选旧版 MediaPipe 后端

只有需要复现 0.2.x 行为时才安装：

```bash
python -m pip install -e ".[legacy-mediapipe]"
```

默认 OpenCV CPU 模式无需安装这个可选项。

## 桌面运行

macOS 可双击 `start-desktop.command`，Windows 可双击 `start-desktop.bat`。也可以在已激活的虚拟环境运行：

```bash
dance-pose-coach --mode local
```

浏览器会打开 `http://127.0.0.1:7860`。这个地址只接受本机访问。

使用步骤：

1. 上传自己的舞蹈视频。
2. 不上传参考视频时，直接点击开始分析即可生成单视频报告。
3. 如需动作还原度评分，再上传同编舞、同镜头方向的本地参考视频。
4. 选择镜像策略并开始分析。
5. 如确实需要自动识曲/检索，先勾选“允许在线服务”。
6. 下载 HTML 或 JSON 报告；结果保存在本机 `runs/`。

报告元数据会显示用户视频和参考视频实际使用的姿态后端；正常默认值为 `opencv-cpu`。

## 手机运行（电脑负责计算）

电脑和手机必须连接**同一个可信 Wi-Fi**，且电脑在整个分析过程中保持开机。

1. macOS 双击 `start-phone.command`；Windows 双击 `start-phone.bat`。
2. 窗口会显示手机地址、用户名 `dance` 和随机密码。
3. Windows/macOS 防火墙首次询问时，只允许“专用/私人网络”，不要允许公共网络。
4. 在手机浏览器打开显示的地址，输入用户名和密码。
5. 从手机选择用户视频和参考视频，等待电脑完成分析。
6. 用完关闭启动窗口，局域网入口随即停止。

命令行等价方式：

```bash
dance-pose-coach --mode lan
```

可自行指定端口和密码：

```bash
dance-pose-coach --mode lan --port 7860 --password "仅你知道的长密码"
```

排查手机无法连接：确认两台设备在同一 Wi-Fi、没有启用访客网络/客户端隔离、地址中的 IP 正确，并确认防火墙只为该私人网络放行 Python。程序不会使用 `share=True`，请不要把路由器端口映射到公网。

## 命令行分析

完全本地（推荐）：

```bash
dance-pose-analyze my-dance.mp4
```

上面会自动运行单视频模式。提供本地参考后自动切换到对比模式：

```bash
dance-pose-analyze my-dance.mp4 --reference original.mp4 --mirror auto
```

明确允许联网并自动检索：

```bash
dance-pose-analyze my-dance.mp4 --query "歌曲名 舞蹈名 official dance practice" --allow-online
```

指定在线参考 URL：

```bash
dance-pose-analyze my-dance.mp4 --reference-url "https://example.com/video" --allow-online
```

报告默认写到 `runs/日期-时间-随机ID/`。

## 如何录制，结果更可靠

- 固定机位，让头、手、脚始终完整入镜
- 保持光线充足，衣服和背景有明显对比
- 一次只保留一名主要舞者，减少遮挡和快速运镜
- 用户与参考尽量使用同一编舞版本、相近视角和音乐段落
- 如果开头不同，建议先剪出相同的 8 拍或 16 拍
- 自动检索结果可能是翻跳、镜像或教程，分析前务必确认

## 项目结构

```text
dance-pose-coach/
├── src/dance_pose_coach/           # UI、CLI、分析管线与运行环境检测
├── tests/                           # 指标、隐私开关、平台与启动测试
├── references/metrics.md            # 指标定义与解释边界
├── start-desktop.command/.bat/.sh   # 本机启动入口
├── start-phone.command/.bat/.sh     # 密码保护的局域网入口
├── pyproject.toml
└── LICENSE
```

## 开发与测试

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

仅检查 UI 能否构建：

```bash
python -c "from dance_pose_coach.app import build_app; build_app(); print('UI OK')"
```

## 数据与网络说明

- 本地抽帧、姿态计算、节拍分析和报告生成都在运行程序的电脑完成。
- `runs/` 已被 `.gitignore` 忽略；它仍是电脑上的普通文件夹，请自行妥善保管或删除。
- 手机模式会把选择的视频上传到电脑上的临时目录；操作系统或 Gradio 可能保留临时文件到后续清理。
- 在线开关启用后，可选 Shazam 会把用于识曲的数据发送给其服务，`yt-dlp` 会访问并可能下载来源平台内容。
- 请遵守平台服务条款、当地法律和版权许可，优先使用你有权分析的本地参考视频。

## 已知限制

- 未提供真正“只在手机浏览器中计算”的版本；手机必须连接一台运行本项目的电脑。
- 局域网模式是密码保护的 HTTP，不提供 TLS；仅适合可信私人网络。
- Windows on ARM 未验证；Windows 支持目标是 x86-64。
- 单人姿态：多人同框时只跟踪一个主要人体。
- 不是严格 3D 动捕；转身、遮挡和景深动作可能导致误差。
- 不同编舞、镜头角度、裁剪区间会显著影响分数。
- 自动识曲/搜索依赖第三方且不保证成功。
- 目前不生成叠加骨架视频。
- 指标阈值是工程启发式，仍需在更多舞种和体型数据上标定。
- “发力表现”来自关键点速度/加速度代理，不等于真实肌肉力量或关节负荷。

## 开源协议

项目代码以 [MIT License](LICENSE) 开源。OpenCV Zoo 模型、可选 MediaPipe、由 `imageio-ffmpeg` 提供的 FFmpeg、yt-dlp、librosa、Gradio 和可选识曲依赖各自遵循独立许可证与条款；MIT 协议不授予第三方视频、音乐或商标的使用权。
