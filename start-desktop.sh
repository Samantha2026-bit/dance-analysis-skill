#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x ".venv/bin/python" ]; then
  echo "Not installed. Create .venv using the README instructions first."
  exit 1
fi
exec .venv/bin/python -m dance_pose_coach.app --mode local
