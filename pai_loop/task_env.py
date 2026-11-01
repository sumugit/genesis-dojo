"""Vectorized task environment (rsl_rl VecEnv-compatible) with a pluggable reward module.

The reward module is a plain Python module exposing:
    compute(s: TaskState, p: dict) -> dict[str, Tensor]   # weighted reward terms, each (N,)
    failed(s: TaskState, p: dict) -> Tensor                # (N,) bool, early termination
This is the part the LLM will write later; everything else stays fixed.
"""

import importlib
import math
from dataclasses import dataclass

import torch
from tensordict import TensorDict

from .sim import make_sim


def quat_rotate_inverse(q, v):
    """Rotate world vector(s) v into the frame of quaternion q (wxyz)."""
    w, u = q[..., :1], q[..., 1:]
    return v * (2.0 * w * w - 1.0) - 2.0 * w * torch.cross(u, v, dim=-1) + 2.0 * u * (u * v).sum(-1, keepdim=True)


HEAD_OFFSET_B = (0.30, 0.0, 0.0)  # nose position in the body frame (Head_upper joint is at x=0.285 in the URDF)


def quat_rotate(q, v):
    """Rotate body-frame vector(s) v into the world frame with quaternion q (wxyz)."""
    w, u = q[..., :1], q[..., 1:]
    return v * (2.0 * w * w - 1.0) + 2.0 * w * torch.cross(u, v, dim=-1) + 2.0 * u * (u * v).sum(-1, keepdim=True)


def nose_up_angle(gravity_b):
    """Pitch in rad, positive when the head is higher than the tail."""
    return torch.asin(torch.clamp(-gravity_b[..., 0], -1.0, 1.0))


def roll_angle(gravity_b):
    return torch.asin(torch.clamp(gravity_b[..., 1], -1.0, 1.0))


@dataclass
class TaskState:
    """Everything a reward function may read. Body-frame quantities are marked _b."""

    base_pos: torch.Tensor  # (N, 3) world
    base_quat: torch.Tensor  # (N, 4) wxyz
    lin_vel_b: torch.Tensor  # (N, 3)
    ang_vel_b: torch.Tensor  # (N, 3)
    gravity_b: torch.Tensor  # (N, 3) unit gravity in body frame; (0, 0, -1) when level
    dof_pos: torch.Tensor  # (N, 12)
    dof_vel: torch.Tensor  # (N, 12)
    default_dof_pos: torch.Tensor  # (12,)
    feet_pos: torch.Tensor  # (N, 4, 3) world, order FL, FR, RL, RR
    knee_pos: torch.Tensor  # (N, 4, 3) world, knee/elbow joints (calf-link origins), order FL, FR, RL, RR
    head_pos: torch.Tensor  # (N, 3) world, the nose (about 0.30 m ahead of the base origin along the body x axis)
    actions: torch.Tensor  # (N, 12)
    last_actions: torch.Tensor  # (N, 12)
    episode_time: torch.Tensor  # (N,) seconds since reset


class TaskEnv:
    def __init__(self, cfg: dict, num_envs: int, camera: dict | None = None):
        self.cfg = cfg
        ecfg = cfg["env"]
        self.joint_names = ecfg["joint_names"]
        self.num_envs = num_envs
        self.num_actions = len(self.joint_names)
        self.sim = make_sim(
            cfg.get("sim", "genesis"),
            num_envs=num_envs,
            joint_names=self.joint_names,
            kp=ecfg["kp"],
            kd=ecfg["kd"],
            init_pos=ecfg["base_init_pos"],
            camera=camera,
        )
        self.device = self.sim.device
        self.dt = self.sim.dt
        self.max_episode_length = math.ceil(ecfg["episode_length_s"] / self.dt)
        self.action_scale = ecfg["action_scale"]
        self.obs_scales = cfg["obs_scales"]

        self.reward_module = importlib.import_module(cfg["reward"]["module"])
        self.reward_params = cfg["reward"]["params"]

        f = dict(dtype=torch.float32, device=self.device)
        self.default_dof_pos = torch.tensor([ecfg["default_joint_angles"][n] for n in self.joint_names], **f)
        self.init_base_pos = torch.tensor(ecfg["base_init_pos"], **f).expand(num_envs, 3)
        self.init_base_quat = torch.tensor(ecfg["base_init_quat"], **f).expand(num_envs, 4)
        self.init_dof_pos = self.default_dof_pos.expand(num_envs, -1)

        self.actions = torch.zeros((num_envs, self.num_actions), **f)
        self.last_actions = torch.zeros_like(self.actions)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.episode_sums: dict[str, torch.Tensor] = {}
        self.extras: dict = {}
        self.reset()

    # --- rsl_rl VecEnv API ---
    def get_observations(self):
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    def reset(self):
        self.sim.reset(None, self.init_base_pos, self.init_base_quat, self.init_dof_pos)
        self.actions.zero_()
        self.last_actions.zero_()
        self.episode_length_buf.zero_()
        self.state = self._read_state()
        self._update_obs()
        return self.get_observations()

    def step(self, actions):
        self.actions = torch.clip(actions, -self.cfg["env"]["clip_actions"], self.cfg["env"]["clip_actions"])
        # One-step action latency, as on the real robot.
        self.sim.step(self.last_actions * self.action_scale + self.default_dof_pos)
        self.episode_length_buf += 1
        self.state = self._read_state()

        terms = self.reward_module.compute(self.state, self.reward_params)
        rew = torch.zeros(self.num_envs, device=self.device)
        for name, value in terms.items():
            value = value * self.dt
            rew += value
            self.episode_sums[name] = self.episode_sums.get(name, torch.zeros_like(rew)) + value

        time_outs = self.episode_length_buf > self.max_episode_length
        failed = self.reward_module.failed(self.state, self.reward_params) | self.sim.error_mask()
        dones = time_outs | failed
        self.extras = {"time_outs": time_outs.float(), "episode": {}}
        if dones.any():
            self._reset_envs(dones)

        self._update_obs()
        self.last_actions = self.actions.clone()
        return self.get_observations(), rew, dones, self.extras

    # --- internals ---
    def _reset_envs(self, mask):
        n = mask.sum()
        ep_len_s = self.cfg["env"]["episode_length_s"]
        for name, s in self.episode_sums.items():
            self.extras["episode"]["rew_" + name] = s[mask].sum() / n / ep_len_s
            s.masked_fill_(mask, 0.0)
        self.extras["episode"]["fail_rate"] = (mask & (self.episode_length_buf <= self.max_episode_length)).float().sum() / n
        self.sim.reset(mask, self.init_base_pos, self.init_base_quat, self.init_dof_pos)
        self.actions.masked_fill_(mask[:, None], 0.0)
        self.last_actions.masked_fill_(mask[:, None], 0.0)
        self.episode_length_buf.masked_fill_(mask, 0)
        # Observations after reset should reflect the reset pose.
        self.state = self._read_state()

    def _read_state(self) -> TaskState:
        s = self.sim.get_state()
        g = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand(self.num_envs, 3)
        return TaskState(
            base_pos=s.base_pos,
            base_quat=s.base_quat,
            lin_vel_b=quat_rotate_inverse(s.base_quat, s.base_lin_vel),
            ang_vel_b=quat_rotate_inverse(s.base_quat, s.base_ang_vel),
            gravity_b=quat_rotate_inverse(s.base_quat, g),
            dof_pos=s.dof_pos,
            dof_vel=s.dof_vel,
            default_dof_pos=self.default_dof_pos,
            feet_pos=s.feet_pos,
            knee_pos=s.knee_pos,
            head_pos=s.base_pos + quat_rotate(s.base_quat, torch.tensor(HEAD_OFFSET_B, device=self.device).expand_as(s.base_pos)),
            actions=self.actions,
            last_actions=self.last_actions,
            episode_time=self.episode_length_buf.float() * self.dt,
        )

    def _clock(self):
        """sin/cos of a fixed-frequency phase (env.clock_hz), so that a policy can learn a periodic gait. Empty if unset."""
        hz = self.cfg["env"].get("clock_hz")
        if not hz:
            return ()
        ph = 2.0 * math.pi * hz * self.state.episode_time
        return (torch.stack((torch.sin(ph), torch.cos(ph)), dim=-1),)

    def _update_obs(self):
        s = self.state
        self.obs_buf = torch.cat(
            (
                s.ang_vel_b * self.obs_scales["ang_vel"],
                s.gravity_b,
                (s.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                s.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
                *self._clock(),
            ),
            dim=-1,
        )
