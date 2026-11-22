#!/usr/bin/env bash
# Copy the resumable checkpoint to Drive every 60 s, atomically (tmp + rename), within Drive's free space.
RUN=/content/runs/t4; DRIVE=/content/drive/MyDrive/Colab-MCP/blinkmimic/t4
mkdir -p "$DRIVE"
while true; do
  sleep 60
  [ -f "$RUN/last.pt" ] || continue
  if [ ! -f "$DRIVE/last.pt" ] || [ "$RUN/last.pt" -nt "$DRIVE/last.pt" ]; then
    need=$(( $(stat -c %s "$RUN/last.pt") / 1048576 + 150 ))
    avail=$(df -m "$DRIVE" | tail -1 | awk '{print $4}')
    if [ "$avail" -lt "$need" ]; then echo "[sync] SKIP: Drive has ${avail} MB free, need ${need}"; continue; fi
    cp "$RUN/last.pt" "$DRIVE/last.pt.tmp" && mv "$DRIVE/last.pt.tmp" "$DRIVE/last.pt" && \
      echo "[sync] $(date +%T) last.pt -> Drive ($(du -m "$DRIVE/last.pt" | cut -f1) MB), free ${avail} MB"
  fi
  newest=$(ls -t "$RUN"/ckpt_*.pt 2>/dev/null | head -1)
  if [ -n "$newest" ] && [ ! -f "$DRIVE/$(basename "$newest")" ]; then
    rm -f "$DRIVE"/ckpt_*.pt; cp "$newest" "$DRIVE/" || true
  fi
  cp "$RUN/stdout.log" "$DRIVE/stdout.log" 2>/dev/null || true
done
