# Running code on Colab from Claude Code

Prereqs: `uv tool install google-colab-cli --with keyrings.alt` (keyrings.alt only needed where no OS keyring exists),
sign in once with `colab sessions`, and `.mcp.json` provides the `colab` MCP server.

## Short scripts (minutes): one shot
`colab_run(script="train_mnist.py", args=["--epochs","3"], gpu="T4")`
Allocates a VM, runs the local script, returns its output, releases the VM. Final line `RESULT {...}` is JSON.

## Long scripts or reruns: keep a session
1. `colab_new(session="dev", gpu="T4")` (once)
2. `colab_upload(local_path="train_mnist.py", remote_path="/content/train_mnist.py", session="dev")`
3. `colab_job_start(command="python -u train_mnist.py --epochs 20", session="dev")` -> job id
4. `colab_job_logs(job_id=..., tail=50)` / `colab_job_status(job_id=...)` to poll
5. `colab_download(remote_path="/content/out/metrics.json", local_path="results/metrics.json", session="dev")`
6. **`colab_stop(session="dev")` when done** - idle runtimes burn compute units.

Notebooks: `colab_notebook_to_script(path=..., out_path=...)` converts to a script first; avoid `colab_notebook_run_all`
(MCP calls time out after 60s even though the run continues).

## SuperMimic (waxz/supermimic)
`colab_setup_supermimic.sh` clones `supermimic` + sibling `mjlab_research` into `/content/supermimic_ws`, installs
Python 3.13 deps with uv (~2.5 min on T4) and checks that both tasks register. Upload to `/content/setup.sh` and run with
`colab_job_start(command="bash /content/setup.sh")`. Train with
`cd /content/supermimic_ws/supermimic && uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --runner.run_name test`
(needs a motion file via `--motion-file`, or `WANDB_API_KEY` + access to the default W&B registry).

### SuperMimic without the W&B registry
The default registry belongs to the authors' W&B org. `colab_supermimic_motion.sh [clip]` downloads the public Unitree
retargeted LAFAN1 G1 CSV (`lvhaidong/LAFAN1_Retargeting_Dataset`) and converts it with mjlab's `csv_to_npz` (30 -> 50 fps).
Train with `--motion-file /content/motions_npz/lafan_run1_subject5.npz` and `WANDB_MODE=offline`:
`.venv/bin/python -u src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --motion-file ... --num-envs 4096`
First start spends ~4 min in Warp kernel compile + torch.compile tracing before any step prints; that is normal.

### Results, curves and screenshots (SuperMimic)
- `plot_train_log.py train.log out.png` plots mean reward, episode length, imagined policy reward, world-model loss and tracking errors from the train stdout log.
- `supermimic_eval_render.py LOG_DIR --motion-file X.npz` (upload to the supermimic repo dir, run with the project venv, `MUJOCO_GL=egl`) rolls out
  `actor.pt` + `world_model.pt` and writes `rollout.mp4`, PNG snapshots and `eval_stats.json`. Do not run it while training: the T4 has 14.5 GB and 2 CPU cores.
- T4 limits: default policy batch (4096) OOMs; use `--runner.policy.batch-size 1024 --num-envs 256`. Each fresh process spends ~4-8 min compiling.
- Example outputs in `results/`.

### Web access through Cloudflare quick tunnels (SuperMimic on Colab)
Quick tunnels (`*.trycloudflare.com`) need no account but are **public and unauthenticated**: anyone with the URL can open them. They die with the job/VM.
1. `curl -sL -o /content/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /content/cloudflared`
2. TensorBoard: `python log_to_tensorboard.py train.log /content/tb [--follow]` (the runner only logs to W&B/stdout, so event files come from the log), then
   `.venv/bin/python -m tensorboard.main --logdir /content/tb --host 127.0.0.1 --port 6006` (venv needs `uv pip install 'setuptools<81'` for `pkg_resources`).
3. mjlab viser viewer: `play_local.py LOG_DIR --motion-file X.npz` (upload as `supermimic_play_local.py`). Colab's own proxy owns port 8080, so viser starts on **8081**; read the port from its log.
4. One tunnel per port, each as its own background job: `/content/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:<port>`; the URL is in the job's log.

### Paper-matched SuperMimic run on a T4 (what was launched)
Colab reclaims idle VMs, so rebuild with `colab_supermimic_bootstrap.sh` (needs `colab_setup_supermimic.sh` -> `/content/setup.sh`,
`colab_supermimic_motion.sh` -> `/content/motion.sh`; ~3 min). Then train with the paper's update-to-data ratio 0.4
(`replay_ratio = 65536 / num_envs`), horizon 16 (critic makes it match H=32) and a T4-sized policy batch:
`train.py SuperMimic-GenCoord-G1 --motion-file ... --num-envs 2048 --runner.replay-ratio 32 --runner.policy.batch-size 2048 --runner.policy.horizon 16`.
`supermimic_play_local.py` now re-reads `actor.pt`/`world_model.pt` every 60 s, so the viewer follows training.
