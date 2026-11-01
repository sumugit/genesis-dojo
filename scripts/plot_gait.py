"""Plot front-foot heights (and head / front-knee heights) of one recorded run from its metrics json.

Usage: python scripts/plot_gait.py runs/<exp>_L1/metrics_599.json out.png [--seconds 8]
The series in `series_0p1s` are sampled every 0.1 s. The dashed line marks the 0.06 m lift threshold used by
`front_step_count_min` in the success specs.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("out")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    s = json.load(open(a.metrics))["series_0p1s"]
    t = [i * a.dt for i in range(len(s["foot_z_FL"]))]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax1.plot(t, s["foot_z_FL"], label="front-left foot", color="#1f77b4")
    ax1.plot(t, s["foot_z_FR"], label="front-right foot", color="#d62728")
    ax1.axhline(0.06, ls="--", color="gray", lw=1, label="lift threshold 0.06 m")
    ax1.set_ylabel("foot height [m]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title(a.title or "Front-foot heights (alternating steps)")
    ax2.plot(t, s["head_z"], label="head", color="#2ca02c")
    ax2.plot(t, [min(x, y) for x, y in zip(s["knee_z_FL"], s["knee_z_FR"])], label="front knee (lower of two)", color="#9467bd")
    ax2.axhline(0.2, ls=":", color="#2ca02c", lw=1)
    ax2.axhline(0.12, ls=":", color="#9467bd", lw=1)
    ax2.set_ylabel("height [m]")
    ax2.set_xlabel("time [s]")
    ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)


if __name__ == "__main__":
    main()
