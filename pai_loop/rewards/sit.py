"""exp01 (hand-written): "お座りして" — sit on the hind legs with the front legs straight.

Target posture: nose up by ~40 deg, trunk lowered to ~0.22 m, all four feet on the ground, no drifting.
"""

import torch


def nose_up_angle(s):
    """Pitch in rad, positive when the head is higher than the tail."""
    return torch.asin(torch.clamp(-s.gravity_b[:, 0], -1.0, 1.0))


def roll_angle(s):
    return torch.asin(torch.clamp(s.gravity_b[:, 1], -1.0, 1.0))


def compute(s, p):
    # Linear tolerance (not exp) so the gradient does not vanish while still standing (error ~0.7 rad).
    pitch_err = torch.abs(nose_up_angle(s) - p["pitch_target"])
    pitch = torch.clamp(1.0 - pitch_err / p["pitch_tolerance"], min=0.0)
    height = torch.exp(-torch.square((s.base_pos[:, 2] - p["height_target"]) / p["height_sigma"]))
    # Foot centres sit at ~foot radius when touching the ground; penalize lifting any foot above the margin.
    feet_lift = torch.clamp(s.feet_pos[:, :, 2] - p["foot_contact_z"], min=0.0).sum(dim=1)
    return {
        "pitch": p["w_pitch"] * pitch,
        "height": p["w_height"] * height,
        "feet_lift": -p["w_feet_lift"] * feet_lift,
        "roll": -p["w_roll"] * torch.square(roll_angle(s)),
        "drift": -p["w_drift"] * (torch.square(s.lin_vel_b[:, :2]).sum(dim=1) + torch.square(s.ang_vel_b[:, 2])),
        "action_rate": -p["w_action_rate"] * torch.square(s.actions - s.last_actions).sum(dim=1),
        "dof_vel": -p["w_dof_vel"] * torch.square(s.dof_vel).sum(dim=1),
    }


def failed(s, p):
    nose = torch.rad2deg(nose_up_angle(s))
    return (
        (nose > p["fail_nose_up_deg"])
        | (nose < -p["fail_nose_down_deg"])
        | (torch.abs(torch.rad2deg(roll_angle(s))) > p["fail_roll_deg"])
        | (s.base_pos[:, 2] < p["fail_min_height"])
    )
