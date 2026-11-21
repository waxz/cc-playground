"""Roll out a SuperMimic checkpoint headlessly; save mp4, PNG snapshots and reward stats.

Run inside the supermimic venv on the Colab VM:
  MUJOCO_GL=egl WANDB_MODE=disabled python supermimic_eval_render.py LOG_DIR --motion-file X.npz --steps 500 --out /content/out_eval
LOG_DIR holds actor.pt and world_model.pt (written by train.py).
"""
import argparse, json, os
import imageio.v2 as imageio
import numpy as np
import torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
import supermimic.tasks  # noqa: F401
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg

p = argparse.ArgumentParser()
p.add_argument("log_dir")
p.add_argument("--task", default="SuperMimic-GenCoord-G1")
p.add_argument("--motion-file", required=True)
p.add_argument("--steps", type=int, default=500)
p.add_argument("--num-envs", type=int, default=16)
p.add_argument("--snapshots", type=int, default=6)
p.add_argument("--out", default="/content/out_eval")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = "cuda:0"

cfg = load_env_cfg(a.task, play=True)
cfg.scene.num_envs = a.num_envs
cfg.commands["motion"].motion_file = a.motion_file
cfg.commands["motion"].sampling_mode = "start"  # always begin at clip start so frames line up
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev, render_mode="rgb_array"))

rcfg = SuperMimicRunnerCfg()
rcfg.sampling_mode = "start"  # must match the env cfg above (no return-sampler at eval)
if "MaxCoord" in a.task:
  rcfg.representation = "maxcoord"
runner = SuperMimicRunner(env=env, cfg=rcfg, task_id=a.task, registry_name=None, device=dev)
runner._wm.load_bundle(torch.load(os.path.join(a.log_dir, "world_model.pt"), map_location=dev))
runner.policy_learner.actor.load_state_dict(torch.load(os.path.join(a.log_dir, "actor.pt"), map_location=dev))
runner.policy_learner.actor.eval()

rec = {"rew": [], "term": []}
orig = runner.logger.process_env_step
def hook(rewards, terminated, extras):
  rec["rew"].append(rewards.float().mean().item())
  rec["term"].append(terminated.float().mean().item())
  return orig(rewards, terminated, extras)
runner.logger.process_env_step = hook

env.reset()
runner._seed_sim_history()
frames = []
for i in range(a.steps):
  runner.step()
  frames.append(env.unwrapped.render())

rew, term = np.array(rec["rew"]), np.array(rec["term"])
stats = {"steps": a.steps, "mean_reward": float(rew.mean()), "reward_first_100": float(rew[:100].mean()),
         "reward_last_100": float(rew[-100:].mean()), "termination_rate_per_step": float(term.mean()),
         "fraction_envs_terminated_total": float(min(1.0, term.sum()))}
json.dump(stats, open(os.path.join(a.out, "eval_stats.json"), "w"), indent=1)
np.save(os.path.join(a.out, "reward_per_step.npy"), rew)
imageio.mimsave(os.path.join(a.out, "rollout.mp4"), frames, fps=50, macro_block_size=1)
for k, i in enumerate(np.linspace(0, len(frames) - 1, a.snapshots).astype(int)):
  imageio.imwrite(os.path.join(a.out, f"frame_{k}_step{i}.png"), frames[i])
print("EVAL_STATS", json.dumps(stats))
