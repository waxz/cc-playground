# Kaggle smoke test for BlinkMimic

`kaggle_smoke_test.py` answers the open questions from the Kaggle evaluation before any real training is pushed:
which GPU a run gets (P100 vs 2x T4), whether internet works, CPU/RAM/disk, whether the pinned torch 2.9 + CUDA 12.8 build
supports the GPU's architecture (a P100 is `sm_60`; that build lists `sm_70` and up), a per-GPU matmul and bandwidth benchmark,
and (stage B) the real stack: clone BlinkMimic, `uv sync`, download the 40 LAFAN1 clips, run its `tools/test_env.py`.

It is untested on Kaggle. Locally I ran stage A with `SMOKE_FULL=0` (no GPU here, so the benchmark step correctly reports a failure).
Stage B mirrors `colab_blinkmimic_setup.sh`, which worked on a Colab T4.

## Run it with the Kaggle MCP server (new session, after the OAuth sign-in)
Tools are documented at https://www.kaggle.com/docs/mcp. Call `save_notebook` with:

| field | value |
|---|---|
| `slug` | `<your-kaggle-username>/blinkmimic-smoke` |
| `new_title` | `blinkmimic-smoke` |
| `text` | the contents of `kaggle_smoke_test.py` |
| `language` / `kernel_type` | `python` / `script` |
| `enable_internet` | `true` |
| `enable_gpu` | `true` (documented as deprecated: prefer `machine_shape`, whose valid values the docs page does not list) |
| `is_private` | `true` |

`save_notebook` saves and runs the script top to bottom as a batch run. Then:
1. `get_notebook_session_status` (`user_name`, `kernel_slug`) until it finishes. There is no live-log tool.
2. `list_notebook_session_output`, then `download_notebook_output` with `file_path` = `smoke_report.txt` (readable summary) or `smoke_report.json`.

To skip the heavy stage B for a 1-minute answer, put `import os; os.environ["SMOKE_FULL"] = "0"` at the top of the pushed text.

## What to look for
- `A1 hardware`: GPU names and memory, `cpu_count` (docs say 4), RAM (docs say ~29 GB).
- `A2 internet`: `internet_ok=True`; otherwise `enable_internet` was not honoured (a phone-verification requirement is possible; the docs I read do not say).
- `A3`: `arch_supported=True` for every GPU. If the GPU is a P100, expect `arch_supported=False` for the pinned torch 2.9/cu128, so BlinkMimic needs an older torch there.
- Compare `matmul_tflops_fp32` with a Colab T4 (BlinkMimic's learning updates reached about 2 TFLOPs effective there).
- `B blinkmimic stack`: `uv_sync` and `test_env` return code 0, plus the venv benchmark. Total time shows what the compile/setup overhead costs.

Outputs go to `/kaggle/working` (saved, up to 20 GB per the docs); the venv and data go to `/tmp`, outside the saved output.

## Alternative: Kaggle CLI (https://github.com/Kaggle/kaggle-cli)
Use this when the Kaggle MCP server is stuck on "needs authentication" (it was in the session that wrote this note: `/mcp` showed
`kaggle  needs authentication` and every tool returned `Unauthenticated`). `pip install kaggle` works in the cloud container (CLI 2.2.5 installed fine).

**Credentials (not present by default).** The container had no `KAGGLE_*` variables and no `~/.kaggle/`. Create a token at
https://www.kaggle.com/settings (API section) and add it as an environment secret, ideally before the session starts:
- `KAGGLE_API_TOKEN=<token>`, or the legacy pair `KAGGLE_USERNAME` + `KAGGLE_KEY`.
Do not paste the token into chat or commit it. Check with `kaggle kernels list --mine`. Outbound access to kaggle.com may also need
the environment's network policy to allow it (untested).

**Push the smoke test** (run from a scratch dir, not the repo root):
```bash
mkdir -p /tmp/kk && cp kaggle_smoke_test.py /tmp/kk/ && cd /tmp/kk
cat > kernel-metadata.json <<JSON
{
  "id": "<your-kaggle-username>/blinkmimic-smoke",
  "title": "blinkmimic-smoke",
  "code_file": "kaggle_smoke_test.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true
}
JSON
kaggle kernels push -p /tmp/kk
kaggle kernels status <your-kaggle-username>/blinkmimic-smoke     # repeat until complete
kaggle kernels output <your-kaggle-username>/blinkmimic-smoke -p /tmp/kk/out   # smoke_report.txt / .json
```
`kaggle kernels push` runs the script after saving. Add `--accelerator <name>` to pick a GPU type if the default is not what you want
(check `kaggle kernels push --help` for the valid names). Read the results as in "What to look for" above.
