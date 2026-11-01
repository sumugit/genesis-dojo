"""Record a headless video of a trained policy and save task-independent metrics and key frames.

Usage (from the repo root):
    python scripts/record.py exp01_sit [--ckpt 300]
Writes videos/<exp_name>_<ckpt>.mp4, runs/<exp_name>/metrics_<ckpt>.json and runs/<exp_name>/frames_<ckpt>/t*.png.
The metrics are computed by pai_loop/metrics.py (posture, height, feet, horizontal travel, yaw/turning, symmetry,
joint speed); pai_loop/success.py judges them against a frozen success spec.
"""

import argparse
import json
import os
import sys
import time

import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genesis as gs  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from pai_loop.metrics import compute_metrics  # noqa: E402
from pai_loop.task_env import TaskEnv, nose_up_angle, roll_angle  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("exp_name")
parser.add_argument("--ckpt", type=int, help="default: last checkpoint in summary.json")
parser.add_argument("--camera-from", help="YAML whose `record:` camera keys (cam_offset, lookat_z, fov, res) replace the run's; for re-recording")
parser.add_argument("--tag", help="suffix for the outputs (video, frames, metrics), so that a re-recording does not overwrite the original")
args = parser.parse_args()
if bool(args.camera_from) != bool(args.tag):
    parser.error("--camera-from and --tag go together (a re-recording must not overwrite the original outputs)")

run_dir = os.path.join("runs", args.exp_name)
with open(os.path.join(run_dir, "config.yaml")) as f:
    cfg = yaml.safe_load(f)
ckpt = args.ckpt
if ckpt is None:
    with open(os.path.join(run_dir, "summary.json")) as f:
        ckpt = json.load(f)["last_checkpoint"]
rcfg = cfg["record"]
if args.camera_from:
    with open(args.camera_from) as f:
        rcfg = dict(rcfg, **{k: v for k, v in yaml.safe_load(f)["record"].items() if k in ("cam_offset", "lookat_z", "fov", "res")})
suffix = f"_{args.tag}" if args.tag else ""
lookat_z = rcfg.get("lookat_z", 0.25)
frame_times = rcfg.get("frame_times", [0.0, 0.5, 1.0, 2.0, 4.0, rcfg["seconds"] - 0.02])
video_path = os.path.join("videos", f"{args.exp_name}_{ckpt}{suffix}.mp4")
frames_dir = os.path.join(run_dir, f"frames_{ckpt}{suffix}")
os.makedirs(frames_dir, exist_ok=True)

gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=cfg["seed"])
camera = dict(res=rcfg["res"], pos=rcfg["cam_offset"], lookat=(0.0, 0.0, lookat_z), fov=rcfg["fov"])
env = TaskEnv(cfg, num_envs=1, camera=camera)
runner = OnPolicyRunner(env, dict(cfg["train"], run_name=args.exp_name), run_dir, device=gs.device)
runner.load(os.path.join(run_dir, f"model_{ckpt}.pt"))
policy = runner.get_inference_policy(device=gs.device)

n_steps = int(rcfg["seconds"] / env.dt)
frame_steps = {round(t / env.dt): t for t in frame_times}
log = {"nose_up_deg": [], "roll_deg": [], "base_pos": [], "base_quat": [], "feet_pos": [], "dof_vel_rms": [], "head_pos": [], "knee_pos": [], "dof_pos": []}
markers = []   # joint positions per step, for the overlay on the video (overlay_<ckpt>.json)
frames = []
n_failed = 0
obs = env.reset()
env.sim.start_recording(video_path, fps=int(1 / env.dt))
t0 = time.perf_counter()
with torch.no_grad():
    for i in range(n_steps):
        obs, _, dones, _ = env.step(policy(obs))
        s = env.state
        n_failed += int(bool(dones[0]) and not bool(env.extras["time_outs"][0]))
        p = s.base_pos[0].tolist()
        log["nose_up_deg"].append(float(torch.rad2deg(nose_up_angle(s.gravity_b))[0]))
        log["roll_deg"].append(float(torch.rad2deg(roll_angle(s.gravity_b))[0]))
        log["base_pos"].append(p)
        log["base_quat"].append(s.base_quat[0].tolist())
        log["feet_pos"].append(s.feet_pos[0].tolist())
        log["head_pos"].append(s.head_pos[0].tolist())
        log["knee_pos"].append(s.knee_pos[0].tolist())
        log["dof_pos"].append(s.dof_pos[0].tolist())
        markers.append([[round(v, 4) for v in q] for q in env.sim.marker_positions()[0].tolist()])
        log["dof_vel_rms"].append(float(torch.sqrt((s.dof_vel[0] ** 2).mean())))
        off = rcfg["cam_offset"]
        rgb = env.sim.render_frame((p[0] + off[0], p[1] + off[1], off[2]), (p[0], p[1], lookat_z))
        if i in frame_steps:
            path = os.path.join(frames_dir, f"t{frame_steps[i]:.2f}s.png")
            Image.fromarray(rgb).resize((rcfg["res"][0] // 2, rcfg["res"][1] // 2)).save(path)
            frames.append({"t": frame_steps[i], "path": path})
env.sim.stop_recording()
wall_s = time.perf_counter() - t0

metrics = {
    "exp_name": args.exp_name,
    "ckpt": ckpt,
    "video": video_path,
    "seconds": rcfg["seconds"],
    "metrics_after_s": rcfg["settle_s"],
    "joint_names": list(env.joint_names),
    **compute_metrics(log, env.dt, rcfg["settle_s"], n_failed),
    "record_wall_seconds": round(wall_s, 1),
    "frames": frames,
}
with open(os.path.join(run_dir, f"metrics_{ckpt}{suffix}.json"), "w") as f:
    json.dump(metrics, f, indent=1, ensure_ascii=False)
from pai_loop.sim.genesis_backend import MARKER_NAMES

with open(os.path.join(run_dir, f"overlay_{ckpt}{suffix}.json"), "w") as f:
    json.dump({"video": os.path.basename(video_path), "dt": env.dt, "res": rcfg["res"], "fov": rcfg["fov"],
               "cam_offset": rcfg["cam_offset"], "lookat_z": lookat_z, "names": MARKER_NAMES,
               "base_xy": [[round(m[0][0], 4), round(m[0][1], 4)] for m in markers], "pts": markers}, f, separators=(",", ":"))
print(json.dumps({k: v for k, v in metrics.items() if k not in ("series_0p1s", "frames")}, ensure_ascii=False))
