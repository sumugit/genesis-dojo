"""Physics probe: does a base config's initial pose make sense? Runs zero actions (hold default joints) and prints
pitch (positive = head up), height and the feet heights every 0.5 s. No training, no reward is used.

    python scripts/probe_pose.py --base-config configs/base_handstand.yaml [--seconds 3]
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genesis as gs  # noqa: E402

from pai_loop.task_env import TaskEnv, nose_up_angle  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--base-config", default="configs/base.yaml")
parser.add_argument("--seconds", type=float, default=3.0)
args = parser.parse_args()

with open(args.base_config) as f:
    cfg = yaml.safe_load(f)
cfg["reward"] = {"module": "pai_loop.rewards.sit", "params": {}}  # the reward is never evaluated here
cfg["env"]["episode_length_s"] = 1000.0

gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=cfg["seed"])
# build the env without a real reward module: only the simulator part is needed
import pai_loop.task_env as te  # noqa: E402

te.importlib = type("I", (), {"import_module": staticmethod(lambda n: type("R", (), {"compute": staticmethod(lambda s, p: {}), "failed": staticmethod(lambda s, p: torch.zeros(s.base_pos.shape[0], dtype=torch.bool, device=s.base_pos.device))})())})
env = TaskEnv(cfg, num_envs=1)
zero = torch.zeros((1, env.num_actions), device=env.device)
for k in range(int(args.seconds / env.dt) + 1):
    if k % int(0.5 / env.dt) == 0:
        s = env.state
        pitch = float(torch.rad2deg(nose_up_angle(s.gravity_b))[0])
        z = [round(float(v), 3) for v in s.feet_pos[0, :, 2]]
        print(f"t={k * env.dt:.1f}s  nose_up={pitch:7.1f}deg  height={float(s.base_pos[0, 2]):.3f}  feet_z(FL,FR,RL,RR)={z}")
    env.step(zero)
