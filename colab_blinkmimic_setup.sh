#!/usr/bin/env bash
# BlinkMimic setup on a fresh Colab GPU VM (no Drive needed). Run via colab_job_start.
set -euxo pipefail
pip -q install uv
cd /content
[ -d blinkmimic ] || git clone --depth 1 https://github.com/fuyichen0725/blinkmimic
cd blinkmimic
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
uv sync
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"
bash tools/download_data.sh
uv run python tools/test_env.py
echo BLINK_SETUP_OK
