#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "尚未安装。请先按照 README 的 macOS 安装步骤创建 .venv。"
  read -r -p "按回车键关闭..."
  exit 1
fi
exec .venv/bin/python -m dance_pose_coach.app --mode lan
