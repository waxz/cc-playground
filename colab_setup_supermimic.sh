#!/usr/bin/env bash
# Set up SuperMimic on a Colab GPU VM and run a smoke test. Run via colab_job_start.
set -euxo pipefail
WS=/content/supermimic_ws
mkdir -p $WS && cd $WS
[ -d mjlab_research ] || git clone --depth 1 https://github.com/HybridRobotics/mjlab_research.git
[ -d supermimic ] || git clone --depth 1 https://github.com/waxz/supermimic.git
pip -q install uv
cd supermimic
nvidia-smi --query-gpu=name,driver_version --format=csv
uv sync --python 3.13 --extra cu126 2>&1 || uv sync --python 3.13
uv run python - <<'PY'
import torch, mjlab.tasks, supermimic.tasks
from mjlab.tasks.registry import list_tasks
print("cuda:", torch.cuda.is_available())
print("tasks:", [t for t in list_tasks() if "SuperMimic" in t])
PY
echo SETUP_OK
