"""Task-independent metrics computed from one recorded rollout.

Pure Python (no torch), so it can be tested without a GPU. `scripts/record.py` collects the per-step
log and calls `compute_metrics`; `pai_loop/success.py` judges the resulting series against a frozen spec.

Conventions:
- Quaternions are (w, x, y, z); positions are world coordinates; angles are in degrees.
- Yaw is the heading of the body's forward (+x) axis projected on the horizontal plane, unwrapped and measured
  from the first valid sample; counter-clockwise seen from above is positive. It is only meaningful while the
  body is not close to vertical: samples whose forward axis is within `HEADING_MAX_PITCH_DEG` of vertical are
  treated as undefined and the yaw is held at its last value.
- Series are sampled every `SAMPLE_PERIOD_S` seconds (sample k is at t = k * 0.1 s).
"""

from __future__ import annotations

import math

FEET = ["FL", "FR", "RL", "RR"]  # order of SimState.feet_pos
SAMPLE_PERIOD_S = 0.1
HEADING_MAX_PITCH_DEG = 70.0
_HEADING_MIN_HORIZONTAL = math.cos(math.radians(HEADING_MAX_PITCH_DEG))


def forward_axis(q):
    """World-frame direction of the body +x axis for quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    return (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y))


def heading_rad(q):
    """Heading of the forward axis on the horizontal plane, or None if the body is nearly vertical."""
    fx, fy, _ = forward_axis(q)
    if math.hypot(fx, fy) < _HEADING_MIN_HORIZONTAL:
        return None
    return math.atan2(fy, fx)


def unwrapped_yaw_deg(quats):
    """Per-step yaw (deg) relative to the first valid heading, unwrapped. Returns (yaw_list, valid_fraction)."""
    yaw = []
    prev = None
    acc = 0.0
    n_valid = 0
    for q in quats:
        h = heading_rad(q)
        if h is not None:
            n_valid += 1
            if prev is not None:
                d = (h - prev + math.pi) % (2.0 * math.pi) - math.pi
                acc += d
            prev = h
        yaw.append(math.degrees(acc))
    return yaw, (n_valid / len(yaw) if yaw else 0.0)


def _feet_forward_coords(base_pos, quat, feet_pos):
    """Forward (body +x) coordinate of each foot relative to the body centre."""
    ax = forward_axis(quat)
    out = []
    for p in feet_pos:
        out.append((p[0] - base_pos[0]) * ax[0] + (p[1] - base_pos[1]) * ax[1] + (p[2] - base_pos[2]) * ax[2])
    return out


def _r(x, nd=3):
    return round(float(x), nd)


def compute_metrics(log, dt, settle_s, n_failed):
    """Compute metrics from a per-step log.

    `log` (lists, one entry per control step): nose_up_deg, roll_deg, base_pos [x,y,z], base_quat [w,x,y,z],
    feet_pos [[x,y,z] x 4 in FEET order], dof_vel_rms.

    Returns a dict with window means (after `settle_s`), episode-level scalars and `series_0p1s`.
    """
    n = len(log["nose_up_deg"])
    if n == 0:
        raise ValueError("empty log")
    pos = log["base_pos"]
    height = [p[2] for p in pos]
    foot_z = {name: [f[i][2] for f in log["feet_pos"]] for i, name in enumerate(FEET)}

    x0, y0 = pos[0][0], pos[0][1]
    xy_dist = [math.hypot(p[0] - x0, p[1] - y0) for p in pos]
    xy_path = sum(math.hypot(pos[k][0] - pos[k - 1][0], pos[k][1] - pos[k - 1][1]) for k in range(1, n))
    xy_drift = math.hypot(pos[-1][0] - x0, pos[-1][1] - y0)

    yaw, heading_valid_fraction = unwrapped_yaw_deg(log["base_quat"])

    front_asym, rear_asym = [], []
    for k in range(n):
        xb = _feet_forward_coords(pos[k], log["base_quat"][k], log["feet_pos"][k])
        front_asym.append(abs(xb[0] - xb[1]))
        rear_asym.append(abs(xb[2] - xb[3]))

    settle_steps = min(int(round(settle_s / dt)), n - 1)

    def mean_after_settle(v):
        w = v[settle_steps:]
        return sum(w) / len(w)

    step = max(1, int(round(SAMPLE_PERIOD_S / dt)))

    def sample(v):
        return [_r(x) for x in v[::step]]

    series = {
        "nose_up_deg": sample(log["nose_up_deg"]),
        "roll_deg": sample(log["roll_deg"]),
        "height_m": sample(height),
        **{f"foot_z_{name}": sample(foot_z[name]) for name in FEET},
        "yaw_deg": sample(yaw),
        "xy_dist_m": sample(xy_dist),
        "front_feet_asym_m": sample(front_asym),
        "rear_feet_asym_m": sample(rear_asym),
        "dof_vel_rms": sample(log["dof_vel_rms"]),
        "x_m": sample([p[0] - x0 for p in pos]),
        "y_m": sample([p[1] - y0 for p in pos]),
    }
    if "head_pos" in log:
        series["head_z"] = sample([h[2] for h in log["head_pos"]])
        for i, name in enumerate(FEET):
            series[f"knee_z_{name}"] = sample([k[i][2] for k in log["knee_pos"]])
    if "dof_pos" in log:  # joint angles [rad], in the order of the config's joint_names (the dashboard's trace tab)
        for i in range(len(log["dof_pos"][0])):
            series[f"q_{i}"] = sample([q[i] for q in log["dof_pos"]])
    return {
        "nose_up_deg_mean": _r(mean_after_settle(log["nose_up_deg"])),
        "roll_deg_mean": _r(mean_after_settle(log["roll_deg"])),
        "height_m_mean": _r(mean_after_settle(height)),
        **{f"foot_z_{name}_m_mean": _r(mean_after_settle(foot_z[name])) for name in FEET},
        "xy_drift_m": _r(xy_drift),
        "xy_max_dist_m": _r(max(xy_dist)),
        "xy_path_length_m": _r(xy_path),
        "yaw_total_deg": _r(yaw[-1], 1),
        "heading_valid_fraction": _r(heading_valid_fraction),
        "early_terminations": int(n_failed),
        "series_0p1s": series,
    }
