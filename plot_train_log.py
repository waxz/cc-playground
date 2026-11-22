"""Parse a SuperMimic train.py stdout log into curves. Usage: python plot_train_log.py train.log out.png"""
import re, sys, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log, out = sys.argv[1], sys.argv[2]
want = {"Mean reward": "Mean reward", "Mean episode length": "Mean episode length", "Policy/reward": "Policy/reward (imagined)",
        "Loss/state": "WM Loss/state", "Metrics/motion/error_body_pos": "error_body_pos", "Metrics/motion/error_joint_pos": "error_joint_pos"}
rows, cur = [], None
for line in open(log, errors="ignore"):
  m = re.search(r"Learning iteration (\d+)/", line)
  if m:
    cur = {"it": int(m.group(1))}; rows.append(cur); continue
  if cur is None: continue
  m = re.match(r"\s*([A-Za-z_/ ]+?):\s+(-?[\d.]+(?:e-?\d+)?)\s*$", line)
  if m and m.group(1).strip() in want: cur[m.group(1).strip()] = float(m.group(2))
fig, axs = plt.subplots(2, 3, figsize=(13, 6))
for ax, (k, title) in zip(axs.flat, want.items()):
  pts = [(r["it"], r[k]) for r in rows if k in r]
  if pts: ax.plot(*zip(*pts))
  ax.set_title(title); ax.set_xlabel("log step (x10 train steps)"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(out, dpi=110)
last = rows[-1] if rows else {}
print(json.dumps(last))
