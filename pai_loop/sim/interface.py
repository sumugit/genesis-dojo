"""Thin simulator interface so the task/reward code does not depend on Genesis.

Conventions:
- Quaternions are (w, x, y, z).
- Velocities in SimState are in the world frame; the task env converts them to the body frame.
- Joint order follows `joint_names` given to the backend.
"""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class SimState:
    base_pos: torch.Tensor  # (N, 3) world
    base_quat: torch.Tensor  # (N, 4) world, wxyz
    base_lin_vel: torch.Tensor  # (N, 3) world
    base_ang_vel: torch.Tensor  # (N, 3) world
    dof_pos: torch.Tensor  # (N, 12)
    dof_vel: torch.Tensor  # (N, 12)
    feet_pos: torch.Tensor  # (N, 4, 3) world, order FL, FR, RL, RR
    knee_pos: torch.Tensor  # (N, 4, 3) world, calf-link origins (the knee/elbow joints), order FL, FR, RL, RR


class QuadrupedSim(Protocol):
    num_envs: int
    dt: float
    device: torch.device

    def step(self, joint_targets: torch.Tensor) -> None:
        """Set PD position targets (N, 12) and advance one control step."""

    def get_state(self) -> SimState: ...

    def reset(self, envs_mask: torch.Tensor | None, base_pos, base_quat, dof_pos) -> None:
        """Teleport the selected envs (all if None) to the given pose with zero velocity."""

    def error_mask(self) -> torch.Tensor:
        """(N,) bool, True where the physics solver diverged."""

    def marker_positions(self) -> torch.Tensor:
        """(N, 17, 3) world positions of the trunk and, per leg (FR, FL, RR, RL), hip / thigh / calf joints and foot
        (only for the overlay on recorded videos)."""

    # Recording (only valid if the backend was built with a camera).
    def start_recording(self, path: str, fps: int) -> None: ...

    def render_frame(self, cam_pos, lookat):
        """Render one frame into the recording and return it as an (H, W, 3) uint8 array."""

    def stop_recording(self) -> None: ...
