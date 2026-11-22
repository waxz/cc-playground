"""Kaggle smoke test for the BlinkMimic workload. Push as a *script* kernel with GPU + internet on.

Stage A (about 1 min): hardware, internet, system torch, a small matmul/bandwidth benchmark per GPU.
Stage B (about 8-15 min, set SMOKE_FULL=0 to skip): the real stack - clone BlinkMimic, `uv sync` (torch 2.9 + CUDA 12.8,
warp 1.16), download the 40 LAFAN1 clips, run its env test, and repeat the matmul check with the venv's torch.

Everything is written to $OUT (default /kaggle/working) as smoke_report.json and smoke_report.txt after every step, so a
partial run still leaves something to download. Heavy files (venv, data) go to /tmp, outside the saved output.
Nothing here needs a secret.
"""
import json, os, shutil, subprocess, sys, time, platform

FULL = os.environ.get("SMOKE_FULL", "1") != "0"
OUT = os.environ.get("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp/smoke_out")
SCRATCH = os.environ.get("SCRATCH", "/tmp/blink_smoke")
REPO_URL = "https://github.com/fuyichen0725/blinkmimic"
os.makedirs(OUT, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)

report = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "full": FULL, "steps": {}}
lines = []


def log(msg=""):
    print(msg, flush=True)
    lines.append(str(msg))


def save():
    json.dump(report, open(os.path.join(OUT, "smoke_report.json"), "w"), indent=1, default=str)
    open(os.path.join(OUT, "smoke_report.txt"), "w").write("\n".join(lines) + "\n")


def sh(cmd, timeout=600, cwd=None, tail=40):
    """Run a shell command, return (exit_code, last `tail` lines of combined output)."""
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd, executable="/bin/bash")
        out = (p.stdout + p.stderr).strip().splitlines()
        return p.returncode, "\n".join(out[-tail:]), time.time() - t0
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout}s", time.time() - t0


def step(name, fn):
    log(f"\n=== {name} ===")
    t0 = time.time()
    try:
        res = fn()
        report["steps"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), "result": res}
    except Exception as e:  # keep going: a failed step is information
        report["steps"][name] = {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"}
        log(f"FAILED: {type(e).__name__}: {e}")
    save()


# ---------------------------------------------------------------- stage A
def hardware():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpu_count": os.cpu_count()}
    for key, cmd in {
        "gpus": "nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader",
        "nvidia_smi_head": "nvidia-smi | head -n 12",
        "ram": "free -g | sed -n 2p",
        "disk": "df -h /kaggle/working /tmp 2>/dev/null || df -h /tmp",
        "cuda_toolkit": "nvcc --version 2>/dev/null | tail -n 1 || echo 'no nvcc'",
        "kaggle_env": "env | grep -i -E '^KAGGLE_' | sed 's/=.*/=<set>/' | head",
    }.items():
        rc, out, _ = sh(cmd, timeout=60)
        info[key] = out
        log(f"[{key}] {out}")
    return info


def internet():
    res = {}
    for name, url in {"github": "https://github.com", "pypi": "https://pypi.org/simple/", "huggingface": "https://huggingface.co",
                      "pytorch_wheels": "https://download.pytorch.org/whl/cu128/", "kaggle": "https://www.kaggle.com"}.items():
        rc, out, dt = sh(f"curl -s -o /dev/null -m 20 -w '%{{http_code}}' {url}", timeout=30)
        code = out.strip() or f"rc={rc}"
        # 000 (or a curl error) = no connectivity; any real HTTP status, even 4xx on a bare root page, means the host is reachable
        res[name] = {"http": code, "reachable": code.isdigit() and code != "000"}
        log(f"{name:15s} http={code} reachable={res[name]['reachable']} ({dt:.1f}s)")
    res["internet_ok"] = all(v["reachable"] for k, v in res.items() if k != "internet_ok")
    log(f"internet_ok={res['internet_ok']}")
    return res


BENCH = r'''
import json, time, torch
res = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else []}
gpus = []
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        g = {"index": i, "name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i)),
             "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)}
        g["arch_supported"] = f"sm_{g['capability'][0]}{g['capability'][1]}" in res["arch_list"]
        try:
            with torch.cuda.device(i):
                def mm(dtype, n=4096, iters=20):
                    a = torch.randn(n, n, device="cuda", dtype=dtype); b = torch.randn(n, n, device="cuda", dtype=dtype)
                    for _ in range(3): a @ b
                    torch.cuda.synchronize(); t = time.time()
                    for _ in range(iters): a @ b
                    torch.cuda.synchronize(); return round(2 * n**3 * iters / (time.time() - t) / 1e12, 2)
                g["matmul_tflops_fp32"] = mm(torch.float32)
                g["matmul_tflops_fp16"] = mm(torch.float16)
                try: g["matmul_tflops_bf16"] = mm(torch.bfloat16)
                except Exception as e: g["matmul_tflops_bf16"] = f"unsupported: {type(e).__name__}"
                x = torch.empty(2**28, device="cuda"); torch.cuda.synchronize(); t = time.time()
                for _ in range(5): y = x.clone()
                torch.cuda.synchronize(); g["copy_bandwidth_GBps"] = round(5 * 2 * x.numel() * 4 / (time.time() - t) / 1e9, 0)
        except Exception as e:
            g["error"] = f"{type(e).__name__}: {e}"
        gpus.append(g)
res["gpus"] = gpus
print("BENCH_JSON " + json.dumps(res))
'''


def run_bench(python_cmd, cwd=None):
    path = os.path.join(SCRATCH, "bench.py")
    open(path, "w").write(BENCH)
    rc, out, dt = sh(f"{python_cmd} {path}", timeout=300, cwd=cwd, tail=60)
    for l in out.splitlines():
        if l.startswith("BENCH_JSON "):
            res = json.loads(l[len("BENCH_JSON "):])
            for g in res["gpus"]:
                log(f"GPU{g['index']} {g['name']} sm_{g['capability'][0]}{g['capability'][1]} {g['memory_gb']} GB arch_supported={g['arch_supported']} "
                    f"fp32={g.get('matmul_tflops_fp32')} fp16={g.get('matmul_tflops_fp16')} bf16={g.get('matmul_tflops_bf16')} "
                    f"bw={g.get('copy_bandwidth_GBps')} GB/s {g.get('error', '')}")
            log(f"torch {res['torch']} cuda_available={res['cuda_available']} arch_list={res['arch_list']}")
            return res
    log(out)
    raise RuntimeError("benchmark produced no result: " + out[-300:])


# ---------------------------------------------------------------- stage B
def blink_stack():
    res = {}
    repo = os.path.join(SCRATCH, "blinkmimic")
    for name, cmd, to in [
        ("pip_uv", "pip install -q uv", 300),
        ("clone", f"[ -d {repo} ] || git clone --depth 1 {REPO_URL} {repo}", 300),
        ("uv_sync", "uv sync", 1500),
        ("download_data", "bash tools/download_data.sh", 900),
        ("test_env", "uv run python tools/test_env.py", 900),
    ]:
        rc, out, dt = sh(cmd, timeout=to, cwd=repo if os.path.isdir(repo) else None, tail=25)
        res[name] = {"rc": rc, "seconds": round(dt, 1), "tail": out[-1500:]}
        log(f"[{name}] rc={rc} {dt:.0f}s\n{out[-600:]}")
        save()
        if rc != 0:
            res["stopped_at"] = name
            return res
    res["venv_bench"] = run_bench("uv run python", cwd=repo)
    return res


step("A1 hardware", hardware)
step("A2 internet", internet)
step("A3 system torch + GPU benchmark", lambda: run_bench(sys.executable))
if FULL:
    step("B blinkmimic stack", blink_stack)
else:
    log("\nSMOKE_FULL=0: skipping stage B")

report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
ok = {k: v["ok"] for k, v in report["steps"].items()}
log("\n=== SUMMARY ===")
log(json.dumps(ok))
save()
log("SMOKE_DONE")
shutil.rmtree(os.path.join(SCRATCH, "blinkmimic", ".venv"), ignore_errors=True) if os.environ.get("SMOKE_CLEAN") else None
