"""Train a policy for one experiment config.

Usage (from the repo root):
    python scripts/train.py configs/exp01_sit.yaml
Outputs go to runs/<exp_name>/ (checkpoints, tensorboard, config copy, summary.json).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genesis as gs  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from pai_loop.task_env import TaskEnv  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("config")
parser.add_argument("--max-iterations", type=int, help="override config (e.g. for a smoke test)")
parser.add_argument("--num-envs", type=int, help="override config")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)
if args.max_iterations is not None:
    cfg["max_iterations"] = args.max_iterations
if args.num_envs is not None:
    cfg["num_envs"] = args.num_envs

run_dir = os.path.join("runs", cfg["exp_name"])
if os.path.exists(run_dir):
    sys.exit(f"{run_dir} already exists; remove it or change exp_name")
os.makedirs(run_dir)
shutil.copy(args.config, os.path.join(run_dir, "config.yaml"))
shutil.copy(cfg["reward"]["module"].replace(".", "/") + ".py", os.path.join(run_dir, "reward.py"))

gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=cfg["seed"], performance_mode=True)
env = TaskEnv(cfg, num_envs=cfg["num_envs"])
train_cfg = dict(cfg["train"], run_name=cfg["exp_name"], seed=cfg["seed"])
runner = OnPolicyRunner(env, train_cfg, run_dir, device=gs.device)

t0 = time.perf_counter()
runner.learn(num_learning_iterations=cfg["max_iterations"], init_at_random_ep_len=True)
train_s = time.perf_counter() - t0

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
summary = {
    "exp_name": cfg["exp_name"],
    "seed": cfg["seed"],
    "iterations": cfg["max_iterations"],
    "num_envs": cfg["num_envs"],
    "train_seconds": round(train_s, 1),
    # Mean episode return over the last 100 finished episodes (same value rsl_rl prints as "Mean reward").
    "final_mean_reward": round(sum(rb) / len(rb), 3) if (rb := runner.logger.rewbuffer) else None,
    "last_checkpoint": max(int(n[6:-3]) for n in os.listdir(run_dir) if n.startswith("model_")),
    "commit": commit,
}
with open(os.path.join(run_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(json.dumps(summary, ensure_ascii=False))
