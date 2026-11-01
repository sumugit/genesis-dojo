"""Synthetic rollouts for the metrics/success tests (no torch, no GPU)."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DT = 0.02


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def yaw_quat(deg):
    h = math.radians(deg) / 2
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def pitch_quat(nose_up_deg):
    """Rotation that tilts the body so the head is `nose_up_deg` above the tail (rotation about -y)."""
    h = math.radians(nose_up_deg) / 2
    return (math.cos(h), 0.0, -math.sin(h), 0.0)


def orientation(yaw_deg=0.0, nose_up_deg=0.0):
    return quat_mul(yaw_quat(yaw_deg), pitch_quat(nose_up_deg))


def standing_feet(z=0.023):
    """Feet in FL, FR, RL, RR order for an untilted, unrotated body at the origin (world coordinates)."""
    return [[0.2, 0.1, z], [0.2, -0.1, z], [-0.2, 0.1, z], [-0.2, -0.1, z]]


def make_log(n, state_fn):
    """state_fn(k) -> dict with optional keys: nose_up_deg, roll_deg, pos, quat, feet, dof_vel_rms."""
    log = {"nose_up_deg": [], "roll_deg": [], "base_pos": [], "base_quat": [], "feet_pos": [], "dof_vel_rms": []}
    for k in range(n):
        s = state_fn(k)
        log["nose_up_deg"].append(s.get("nose_up_deg", 0.0))
        log["roll_deg"].append(s.get("roll_deg", 0.0))
        log["base_pos"].append(s.get("pos", [0.0, 0.0, 0.29]))
        log["base_quat"].append(s.get("quat", (1.0, 0.0, 0.0, 0.0)))
        log["feet_pos"].append(s.get("feet", standing_feet()))
        log["dof_vel_rms"].append(s.get("dof_vel_rms", 0.1))
    return log
