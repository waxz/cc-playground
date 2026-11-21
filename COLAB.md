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
