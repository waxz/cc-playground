#!/usr/bin/env bash
# Train BlinkMimic on a Colab T4, resumable from Google Drive. Re-run this same script after a VM reset
# (after setup + Drive mount): it restores last.pt from Drive and passes --resume.
# Env: PI_BATCH (default 1024; paper recipe 2048 needs more than a T4), EXTRA (extra train.py flags).
set -u
cd /content/blinkmimic
source scripts/presets.sh
RUN=/content/runs/t4
DRIVE=/content/drive/MyDrive/Colab-MCP/blinkmimic/t4
mkdir -p "$RUN" "$DRIVE"
if [ -f "$DRIVE/last.pt" ] && [ ! -f "$RUN/last.pt" ]; then
  echo "[run] restoring $DRIVE/last.pt"; cp "$DRIVE/last.pt" "$RUN/last.pt.tmp" && mv "$RUN/last.pt.tmp" "$RUN/last.pt"
  cp "$DRIVE/stdout.log" "$RUN/stdout.log" 2>/dev/null || true
fi
RESUME=""; [ -f "$RUN/last.pt" ] && RESUME="--resume $RUN/last.pt" && echo "[run] resuming from $RUN/last.pt"
# released recipe, but: full (resumable) checkpoints instead of --ckpt_light, smaller policy batch, checkpoint every 50 its
FLAGS=$(echo "${PRESET[main]}" | sed "s/--ckpt_light//; s/--pi_batch 2048/--pi_batch ${PI_BATCH:-1024}/")
export OMP_NUM_THREADS=2
PYTHONPATH=src uv run python -u -m blinkmimic.train --log_dir "$RUN" $FLAGS --ckpt_every 50 $RESUME ${EXTRA:-} 2>&1 | tee -a "$RUN/stdout.log"
