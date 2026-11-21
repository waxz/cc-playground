"""Convert a SuperMimic train.py stdout log into TensorBoard event files.

  python log_to_tensorboard.py train.log tb_dir [--follow]
Scalars: every `Name: number` line inside each `Learning iteration N/M` block. x-axis = train step (N*log_interval,
here N*10). With --follow it keeps tailing the log so a running TensorBoard stays live.
"""
import re, sys, time
from torch.utils.tensorboard import SummaryWriter

log, out = sys.argv[1], sys.argv[2]
follow = "--follow" in sys.argv
w = SummaryWriter(out)
it_re = re.compile(r"Learning iteration (\d+)/")
kv_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_/ ]*?):\s+(-?\d+(?:\.\d+)?(?:e-?\d+)?)s?\s*$")
cur, pos = None, 0
while True:
  with open(log, errors="ignore") as f:
    f.seek(pos)
    for line in f:
      line = re.sub(r"\x1b\[[0-9;]*m", "", line)
      m = it_re.search(line)
      if m: cur = int(m.group(1)) * 10; continue
      m = kv_re.match(line)
      if cur is not None and m and m.group(1) not in ("Iteration time", "Total steps"):
        w.add_scalar(m.group(1).strip().replace(" ", "_"), float(m.group(2)), cur)
    pos = f.tell()
  w.flush()
  if not follow: break
  time.sleep(10)
