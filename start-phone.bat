@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 尚未安装。请先按照 README 的 Windows 安装步骤创建 .venv。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m dance_pose_coach.app --mode lan
pause
