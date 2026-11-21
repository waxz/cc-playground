"""Viser web viewer for a local SuperMimic checkpoint (play.py without the W&B dependency).

  MUJOCO_GL=egl python supermimic_play_local.py LOG_DIR --motion-file X.npz
LOG_DIR holds actor.pt and world_model.pt. The viser URL/port is printed on startup (default 8080).
"""
import argparse, os
from pathlib import Path
import torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
import supermimic.tasks  # noqa: F401
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg, build_actor, build_world_model
from supermimic.scripts.viewers import SuperMimicViserViewer, create_wm_vis_model

p = argparse.ArgumentParser()
p.add_argument("log_dir")
p.add_argument("--task", default="SuperMimic-GenCoord-G1")
p.add_argument("--motion-file", required=True)
a = p.parse_args()
configure_torch_backends()
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

env_cfg = load_env_cfg(a.task, play=True)
env_cfg.scene.num_envs = 1
env_cfg.observations["actor"].enable_corruption = True
env_cfg.commands["motion"].motion_file = os.path.expanduser(a.motion_file)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=dev))

rcfg = SuperMimicRunnerCfg()
if "MaxCoord" in a.task:
  rcfg.representation = "maxcoord"
wm = build_world_model(rcfg, env, dev)
wm.load_bundle(torch.load(Path(a.log_dir) / "world_model.pt", map_location=dev))
wm.dynamics.eval()
actor = build_actor(rcfg, wm, dev)
actor.load_state_dict(torch.load(Path(a.log_dir) / "actor.pt", map_location=dev))
actor.eval()

SuperMimicViserViewer(
  env=env, actor=actor, dynamics=wm.dynamics, representation=wm.representation,
  wm_vis_model=create_wm_vis_model(env.unwrapped.sim.mj_model),
  num_steps_rollout=1000000, motion_names=[Path(a.motion_file).stem],
  sim_context=wm.sim_context, task_specs=actor.task_specs, initial_task="motion",
).run()
