"""Headless rollout metrics for a SuperMimic actor checkpoint vs a freshly initialised (untrained) actor.
  python supermimic_eval_compare.py CKPT_DIR|none --motion-file X.npz [--steps 300] [--num-envs 32] [--seed 0]
CKPT_DIR=none keeps the random initial actor. Works on CPU. Prints one `EVAL {json}` line."""
import argparse, json
import numpy as np, torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
import supermimic.tasks  # noqa: F401
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg

p = argparse.ArgumentParser()
p.add_argument("ckpt"); p.add_argument("--motion-file", required=True)
p.add_argument("--task", default="SuperMimic-GenCoord-G1")
p.add_argument("--steps", type=int, default=300); p.add_argument("--num-envs", type=int, default=32)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed); np.random.seed(a.seed)

cfg = load_env_cfg(a.task, play=False)
cfg.scene.num_envs = a.num_envs
cfg.seed = a.seed
cfg.commands["motion"].motion_file = a.motion_file
cfg.commands["motion"].sampling_mode = "start"
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev))
rcfg = SuperMimicRunnerCfg(); rcfg.sampling_mode = "start"
runner = SuperMimicRunner(env=env, cfg=rcfg, task_id=a.task, registry_name=None, device=dev)
if a.ckpt != "none":
  runner._wm.load_bundle(torch.load(f"{a.ckpt}/world_model.pt", map_location=dev))
  runner.policy_learner.actor.load_state_dict(torch.load(f"{a.ckpt}/actor.pt", map_location=dev))
runner.policy_learner.actor.eval()

rec = {"rew": [], "term": []}
orig = runner.logger.process_env_step
def hook(rewards, terminated, extras):
  rec["rew"].append(rewards.float().mean().item()); rec["term"].append(terminated.float().cpu().numpy())
  return orig(rewards, terminated, extras)
runner.logger.process_env_step = hook
env.reset(); runner._seed_sim_history()
for _ in range(a.steps): runner.step()
rew = np.array(rec["rew"]); term = np.stack(rec["term"])          # (steps, envs)
first_fall = np.where(term.any(0), term.argmax(0), a.steps)      # first termination step per env
out = {"ckpt": a.ckpt, "steps": a.steps, "envs": a.num_envs,
       "mean_step_reward": round(float(rew.mean()), 4),
       "median_steps_to_first_fall": float(np.median(first_fall)),
       "mean_steps_to_first_fall": round(float(first_fall.mean()), 2),
       "envs_that_never_fell": int((first_fall >= a.steps).sum())}
print("EVAL", json.dumps(out))
