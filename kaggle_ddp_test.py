"""Kaggle 2x T4 test of data-parallel BlinkMimic training (patch: blinkmimic_ddp.patch, embedded below).

Push as a *script* kernel with GPU (NvidiaTeslaT4, 2 GPUs) + internet on. It runs, in order: setup of the patched repo, a short
2-GPU training with the released recipe, the repo's own evaluator on the final checkpoint, and a short --amp none speed comparison. As soon as the evaluation
finishes (or anything fails) it kills every child process, checks that no GPU process is left, writes the report and exits, so
the Kaggle session ends and no GPU time is wasted. A watchdog ends the run after MAX_MIN minutes whatever happens.

Outputs in /kaggle/working: ddp_report.txt / ddp_report.json, run logs (stdout.log, train_log.csv, args.json), evaluation json.
Nothing here needs a secret.
"""
import atexit, base64, glob, gzip, json, os, re, shutil, signal, statistics, subprocess, sys, threading, time

ITERS = int(os.environ.get("DDP_ITERS", "100"))
MAX_MIN = float(os.environ.get("DDP_MAX_MIN", "110"))           # hard wall-clock limit for the whole script
TRAIN_TIMEOUT = float(os.environ.get("DDP_TRAIN_MIN", "60")) * 60
EVAL_TIMEOUT = float(os.environ.get("DDP_EVAL_MIN", "20")) * 60
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp/ddp_out"
SCRATCH = "/tmp/blink_ddp"
REPO_URL, REPO_COMMIT = "https://github.com/fuyichen0725/blinkmimic", "d264658"
PATCH_B64 = "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"
os.makedirs(OUT, exist_ok=True); os.makedirs(SCRATCH, exist_ok=True)

T0 = time.time()
report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "iters": ITERS, "steps": {}}
lines, children = [], []


def log(msg=""):
    print(msg, flush=True); lines.append(str(msg))


def save():
    json.dump(report, open(os.path.join(OUT, "ddp_report.json"), "w"), indent=1, default=str)
    open(os.path.join(OUT, "ddp_report.txt"), "w").write("\n".join(lines) + "\n")


def kill_children():
    for p in children:
        if p.poll() is None:
            try: os.killpg(p.pid, signal.SIGKILL)
            except Exception: pass


def finish(code, why):
    """End the session now: kill children, confirm the GPUs are idle, write the report, exit."""
    kill_children()
    time.sleep(2)
    rc = subprocess.run("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader", shell=True, capture_output=True, text=True)
    report["gpu_processes_left"] = rc.stdout.strip() or "none"
    log(f"\n=== END: {why} (exit {code}, {(time.time() - T0) / 60:.1f} min) | GPU processes left: {report['gpu_processes_left']} ===")
    report["result"] = why; report["exit_code"] = code; report["minutes"] = round((time.time() - T0) / 60, 1)
    save()
    os._exit(code)


def watchdog():
    log(f"WATCHDOG: {MAX_MIN} min limit reached"); finish(2, "watchdog timeout")


threading.Timer(MAX_MIN * 60, watchdog).start()
atexit.register(kill_children)


def sh(cmd, timeout=900, cwd=None, logfile=None, env=None, stream=False):
    """Run a shell command in its own process group; returns (rc, last output lines, seconds).
    logfile: output goes to that file; stream=True also echoes every line to the console live (training / evaluation)."""
    t0 = time.time()
    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         bufsize=1, start_new_session=True, env=dict(os.environ, **(env or {})))
    children.append(p)
    buf = []

    def pump():
        f = open(logfile, "w") if logfile else None
        for line in p.stdout:
            buf.append(line)
            if f: f.write(line); f.flush()
            if stream: print(line.rstrip()[:300], flush=True)
        if f: f.close()

    th = threading.Thread(target=pump, daemon=True); th.start()
    stop = threading.Event()

    def heartbeat():  # once a minute: elapsed time, GPU utilisation / memory, last output line
        while not stop.wait(60):
            g = subprocess.run("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader", shell=True, capture_output=True, text=True).stdout.strip().replace("\n", " | ")
            last = (buf[-1].strip()[:120] if buf else "-")
            print(f"[hb {time.time() - t0:5.0f}s] gpu util,mem: {g} | last: {last}", flush=True)

    if stream: threading.Thread(target=heartbeat, daemon=True).start()
    try:
        p.wait(timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait(); rc = 124
    finally:
        stop.set()
    th.join(timeout=10)
    return rc, "".join(buf[-40:]).strip()[-3000:], time.time() - t0


def step(name, fn):
    log(f"\n=== {name} ===")
    t0 = time.time()
    try:
        res = fn()
        report["steps"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), "result": res}
    except Exception as e:
        report["steps"][name] = {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"}
        log(f"FAILED: {type(e).__name__}: {e}"); save()
        finish(1, f"step '{name}' failed")
    save()


REPO = os.path.join(SCRATCH, "blinkmimic")
RUN = os.path.join(SCRATCH, "run")


def hardware():
    rc, out, _ = sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader", timeout=60)
    log(out)
    n = len([l for l in out.splitlines() if l.strip()])
    if n < 2:
        raise RuntimeError(f"need 2 GPUs, found {n}")
    return {"gpus": out}


def setup():
    res = {}
    for name, cmd, to in [("pip_uv", "pip -q install uv", 300),
                          ("clone", f"git clone -q {REPO_URL} {REPO} && cd {REPO} && git checkout -q {REPO_COMMIT}", 300)]:
        rc, out, dt = sh(cmd, timeout=to); log(f"[{name}] rc={rc} {dt:.0f}s {out[-200:]}")
        if rc: raise RuntimeError(f"{name} failed: {out[-300:]}")
    open(os.path.join(SCRATCH, "ddp.patch"), "wb").write(gzip.decompress(base64.b64decode(PATCH_B64)))
    rc, out, _ = sh(f"git apply --whitespace=nowarn {SCRATCH}/ddp.patch && git status --short", cwd=REPO, timeout=60)
    log(f"[patch] rc={rc} {out}")
    if rc: raise RuntimeError("patch does not apply: " + out)
    for name, cmd, to in [("uv_sync", "uv sync", 900), ("download_data", "bash tools/download_data.sh", 600)]:
        rc, out, dt = sh(cmd, cwd=REPO, timeout=to); log(f"[{name}] rc={rc} {dt:.0f}s")
        res[name] = round(dt)
        if rc: raise RuntimeError(f"{name} failed: {out[-300:]}")
    return res


def train():
    # released recipe, then overrides (later flags win): short run, probes early enough to finish inside it
    over = f"--iterations {ITERS} --ckpt_every 50 --log_every 10 --eprobe_start 40 --eprobe_every 50 --max_hours 0.8 --log_dir {RUN}"
    cmd = (f"source scripts/presets.sh && export OMP_NUM_THREADS=1 && "
           f"uv run python tools/launch_ddp.py --nproc 2 --done_file {RUN}/train_done.txt -- ${{PRESET[main]}} {over}")
    rc, out, dt = sh(cmd, cwd=REPO, timeout=TRAIN_TIMEOUT, logfile=os.path.join(OUT, "stdout.log"), stream=True)
    log(out[-1500:])
    rows = re.findall(r"it (\d+) \| .*?\| ([\d.]+) it/s \| ddp x2 max_param_diff ([\d.eE+-]+)", open(os.path.join(OUT, "stdout.log")).read())
    res = {"rc": rc, "seconds": round(dt), "log_rows": len(rows)}
    if rows:
        ips = [float(r[1]) for r in rows]
        res.update(it_per_s_last=ips[-1], it_per_s_median=statistics.median(ips), max_param_diff=max(float(r[2]) for r in rows))
    log(f"train: {res}")
    for f in ("train_log.csv", "args.json", "policy_start.txt"):
        if os.path.exists(os.path.join(RUN, f)): shutil.copy(os.path.join(RUN, f), OUT)
    res["train_done_file"] = os.path.exists(os.path.join(RUN, "train_done.txt")); res["last_pt"] = os.path.exists(os.path.join(RUN, "last.pt"))
    if rc or not res["train_done_file"]: raise RuntimeError(f"2-GPU training failed (rc={rc}, done_file={res['train_done_file']})")
    if not rows: raise RuntimeError("no training log rows with ddp stats")
    if res["max_param_diff"] > 1e-3: raise RuntimeError(f"ranks drifted apart: max_param_diff={res['max_param_diff']}")
    if re.search(r"\bnan\b", open(os.path.join(OUT, "stdout.log")).read(), re.I): res["warning"] = "nan in log"
    return res


def evaluate():
    ck = sorted(glob.glob(os.path.join(RUN, "ckpt_*.pt")))
    if not ck: raise RuntimeError("no checkpoint to evaluate")
    ck = ck[-1]
    cmd = f"CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src uv run python -m blinkmimic.evaluate --run_dir {RUN} --ckpt {ck}"
    rc, out, dt = sh(cmd, cwd=REPO, timeout=EVAL_TIMEOUT, logfile=os.path.join(OUT, "eval.log"), stream=True)
    log(out[-1500:])
    for f in glob.glob(ck.replace(".pt", "_eval*.json")): shutil.copy(f, OUT)
    if rc: raise RuntimeError(f"evaluation failed (rc={rc})")
    m = re.search(r'"norm_return":\s*([\d.]+)', out)
    return {"checkpoint": os.path.basename(ck), "norm_return": float(m.group(1)) if m else None, "seconds": round(dt)}


step("hardware", hardware)
step("setup", setup)
step("train 2-GPU", train)
step("evaluate", evaluate)
try:
    step_name = "train 2-GPU amp none"
    log(f"\n=== {step_name} ===")
    report["steps"][step_name] = {"ok": True, "result": train_amp_none()}
except Exception as e:
    report["steps"]["train 2-GPU amp none"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}; log(f"amp-none comparison failed (optional): {e}")
save()
finish(0, "training + evaluation complete")
