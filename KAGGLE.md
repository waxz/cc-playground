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
