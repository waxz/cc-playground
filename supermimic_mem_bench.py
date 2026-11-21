"""GPU memory / time budget for SuperMimic learner settings on one GPU.

  python supermimic_mem_bench.py --motion-file X.npz --num-envs 256 --policy 1024:32,2048:32 --wm 2048:8
  python supermimic_mem_bench.py --motion-file X.npz --num-envs 2048          # env + replay buffer footprint only
--policy is batch:horizon pairs (unroll length = horizon), --wm is batch:horizon pairs. Prints `BENCH {json}` lines.
Peak numbers are torch allocator peaks (include the CUDA-graph pool); `gpu_used_mb` is nvidia-level (includes Warp/sim).
"""
import argparse, gc, json, time
import torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
import supermimic.tasks  # noqa: F401
from supermimic.envs.world_model_env import sample_start
from supermimic.learners.wm_learner import WorldModelLearner
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg

p = argparse.ArgumentParser()
p.add_argument("--motion-file", required=True)
p.add_argument("--task", default="SuperMimic-GenCoord-G1")
p.add_argument("--num-envs", type=int, default=256)
p.add_argument("--policy", default="")
p.add_argument("--wm", default="")
p.add_argument("--iters", type=int, default=6)
a = p.parse_args()
dev = "cuda:0"
MB = 1 << 20

def gpu_used_mb():
  free, total = torch.cuda.mem_get_info()
  return (total - free) / MB

def emit(**kw):
  print("BENCH", json.dumps(kw), flush=True)

base_gpu = gpu_used_mb()
cfg = load_env_cfg(a.task, play=False)
cfg.scene.num_envs = a.num_envs
cfg.commands["motion"].motion_file = a.motion_file
cfg.commands["motion"].sampling_mode = "start"
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev))
after_env = gpu_used_mb()
rcfg = SuperMimicRunnerCfg()
rcfg.sampling_mode = "start"
runner = SuperMimicRunner(env=env, cfg=rcfg, task_id=a.task, registry_name=None, device=dev)
env.reset(); runner._seed_sim_history()
after_runner = gpu_used_mb()
for _ in range(80):
  runner.step()
torch.cuda.synchronize()
emit(kind="envs", num_envs=a.num_envs, baseline_gpu_mb=round(base_gpu), env_mb=round(after_env - base_gpu),
     runner_nets_buffer_mb=round(after_runner - after_env), torch_alloc_mb=round(torch.cuda.memory_allocated() / MB),
     replay_buffer_transitions=runner.replay_buffer.num_transitions, gpu_used_mb=round(gpu_used_mb()))

def timed(fn):
  torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
  times = []
  for _ in range(a.iters):
    t = time.time(); fn(); torch.cuda.synchronize(); times.append(time.time() - t)
  return times

hist = rcfg.history_length
for spec in filter(None, a.wm.split(",")):
  bs, hz = map(int, spec.split(":"))
  rcfg.wm.batch_size, rcfg.wm.horizon = bs, hz
  runner.world_model_learner = WorldModelLearner(runner._dynamics, runner._representation, rcfg.wm, device=dev)
  def step_wm():
    data = (runner.replay_buffer.sample(hz, bs) for _ in range(rcfg.wm.num_batches))
    runner.world_model_learner.update(data)
  try:
    t = timed(step_wm)
    emit(kind="wm", batch=bs, horizon=hz, num_batches=rcfg.wm.num_batches, peak_alloc_mb=round(torch.cuda.max_memory_allocated() / MB),
         peak_reserved_mb=round(torch.cuda.max_memory_reserved() / MB), steady_update_s=round(min(t[-2:]), 3), gpu_used_mb=round(gpu_used_mb()))
  except torch.OutOfMemoryError:
    emit(kind="wm", batch=bs, horizon=hz, oom=True)
  gc.collect(); torch.cuda.empty_cache()

for spec in filter(None, a.policy.split(",")):
  bs, hz = map(int, spec.split(":"))
  rcfg.policy.batch_size, rcfg.policy.horizon = bs, hz
  runner.policy_learner = None; runner.wm_env = None; gc.collect(); torch.cuda.empty_cache()
  runner._init_policy()
  def step_policy():
    starts = (sample_start(runner.replay_buffer, hist, bs) for _ in range(rcfg.policy.num_batches))
    runner.policy_learner.update(starts)
  try:
    t = timed(step_policy)
    emit(kind="policy", batch=bs, horizon=hz, peak_alloc_mb=round(torch.cuda.max_memory_allocated() / MB),
         peak_reserved_mb=round(torch.cuda.max_memory_reserved() / MB), steady_update_s=round(min(t[-2:]), 3),
         first_update_s=round(t[0], 1), gpu_used_mb=round(gpu_used_mb()))
  except torch.OutOfMemoryError:
    emit(kind="policy", batch=bs, horizon=hz, oom=True)
    break  # CUDA-graph state is unreliable after an OOM; stop here
  gc.collect(); torch.cuda.empty_cache()
