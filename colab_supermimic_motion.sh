#!/usr/bin/env bash
# Build the default SuperMimic motion locally on Colab (no W&B registry needed).
# Run after colab_setup_supermimic.sh. Output: /content/motions_npz/lafan_run1_subject5.npz
set -euxo pipefail
CLIP=${1:-run1_subject5}
mkdir -p /content/motions /content/motions_npz
curl -sL -o /content/motions/$CLIP.csv \
  https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/$CLIP.csv
cd /content/supermimic_ws/supermimic
# csv_to_npz saves /tmp/motion.npz, then tries a W&B upload; offline mode makes that last step fail harmlessly.
WANDB_MODE=offline MUJOCO_GL=egl .venv/bin/python -m mjlab.scripts.csv_to_npz \
  --input-file /content/motions/$CLIP.csv --output-name lafan_$CLIP \
  --input-fps 30 --output-fps 50 --render False || true
test -s /tmp/motion.npz
cp /tmp/motion.npz /content/motions_npz/lafan_$CLIP.npz
ls -la /content/motions_npz
