"""Frozen numeric success criteria, judged mechanically from the recorded metrics.

A spec is a YAML file that a human approves *before* training (`approved: true`). The loop refuses to run
with an unapproved spec and never edits it. Criteria are declarative, over a fixed catalog of metrics:

    task: sit
    approved: true
    window_s: [3.0, 6.0]
    criteria:
      - {metric: nose_up_deg, between: [40, 65], min_fraction: 0.9}   # series: share of window samples that satisfy
      - {metric: xy_max_dist_m, max: 0.15}                            # episode scalar
      - {metric: yaw_rate_deg_s_mean, between: [60, 180]}             # window aggregate

Metric kinds:
- series: sampled every 0.1 s (see `SERIES_METRICS`); the criterion needs `min_fraction` (0 < f <= 1).
- window aggregate: `<series>_mean`, `<series>_max`, `<series>_min` over the window, and `yaw_rate_steady_fraction`.
- episode scalar: computed over the whole episode (see `EPISODE_METRICS`).
Exactly one of `between: [lo, hi]`, `min`, `max`, `equals` per criterion. An optional `comment` is ignored.
"""

from __future__ import annotations

import hashlib
import math

import yaml

SAMPLE_PERIOD_S = 0.1

SERIES_METRICS = {
    "nose_up_deg": "前後の傾き[deg]。正なら頭が尻より高い",
    "roll_deg_abs": "横の傾きの絶対値[deg]",
    "height_m": "胴体中心の高さ[m]",
    "foot_z_max_m": "4本の足のうち最も高い足先の高さ[m]（接地で約0.023）",
    "front_feet_z_max_m": "前の2本の足のうち高い方の足先の高さ[m]（両方接地なら約0.023。逆立ちで前足が支えるなら小さいまま）",
    "front_feet_z_min_m": "前の2本の足のうち低い方の足先の高さ[m]（少なくとも1本が接地して体を支えていれば小さい）",
    "foot_z_FL_m": "前左の足先の高さ[m]（接地で約0.023）。窓内の最大（_max）で「その足が地面から離れた」ことを見る",
    "foot_z_FR_m": "前右の足先の高さ[m]（接地で約0.023）。同上",
    "head_z_m": "鼻先の高さ[m]（頭を地面に着けると小さい。逆立ちで頭が地面に近づいていないかを見る）",
    "front_knee_z_min_m": "前の2本の膝（肘）の関節のうち低い方の高さ[m]（肘を地面に着けると小さい）",
    "front_one_foot_up": "前の足がちょうど1本だけ持ち上がっている（高さ0.05m超）なら1、それ以外は0。交互に踏み出す歩きなら、一定の割合で1になる",
    "rear_feet_z_min_m": "後ろの2本の足のうち低い方の足先の高さ[m]（両方が空中に上がっているほど大きい）",
    "foot_z_RL_m": "後左の足先の高さ[m]（接地で約0.023）。窓内の最大（_max）で「その足が地面から離れた」ことを見る",
    "foot_z_RR_m": "後右の足先の高さ[m]（接地で約0.023）。同上",
    "rear_feet_z_max_m": "後ろの2本の足のうち高い方の足先の高さ[m]（後ろ足で立って歩くとき、片方が持ち上がると大きくなる）",
    "rear_one_foot_up": "後ろの足がちょうど1本だけ持ち上がっている（高さ0.05m超）なら1、それ以外は0。後ろ足で交互に踏み出す歩きなら、一定の割合で1になる",
    "all_feet_up": "4本の足がすべて持ち上がっている（高さ0.05m超。跳ねている・浮いている）なら1、それ以外は0。歩きでは、ほぼ0であるべき",
    "yaw_rate_deg_s": "旋回速度[deg/s]。反時計回り（上から見て）が正。機体が垂直に近い間は定義できない",
    "yaw_rate_abs_deg_s": "旋回速度の絶対値[deg/s]（方向を問わない）",
    "xy_dist_m": "開始位置からの水平距離[m]",
    "front_feet_asym_m": "前の左右の足の前後位置の差[m]（左右対称なら0）",
    "rear_feet_asym_m": "後ろの左右の足の前後位置の差[m]（左右対称なら0）",
    "dof_vel_rms": "関節速度の二乗平均平方根[rad/s]（動きの激しさ）",
}
EPISODE_METRICS = {
    "xy_max_dist_m": "エピソード全体での、開始位置からの最大水平距離[m]（円を描いて戻っても大きくなる）",
    "xy_path_length_m": "エピソード全体の水平移動の総距離[m]",
    "yaw_total_deg": "エピソード終了時の累積回転角[deg]（反時計回りが正）",
    "heading_valid_fraction": "向きを定義できたステップの割合（0〜1）。機体が垂直に近いと下がる",
    "early_terminations": "途中打ち切りの回数",
}
WINDOW_SPECIAL = {
    "circle_radius_m": "窓内の水平な軌跡に当てはめた円の半径[m]（直線に近いと当てはめられず判定できない）",
    "circle_fit_rms_m": "窓内の軌跡の、当てはめた円からのずれの二乗平均平方根[m]（小さいほど円に沿っている）",
    "circle_arc_deg": "窓内の軌跡が当てはめた円の中心まわりに進んだ角度[deg]（円をどれだけ回ったか。一方向に増える角度の総量）",
    "front_step_count_min": "窓内で、前の左右の足がそれぞれ何回持ち上がった（高さ0.06m超へ上がった。滑りやジッターではなく本当に持ち上げた）かのうち、少ない方[回]。左右の足を使って歩いているか（2回以上の踏み出し）",
    "rear_step_count_min": "窓内で、後ろの左右の足がそれぞれ何回持ち上がった（高さ0.06m超へ上がった）かのうち、少ない方[回]。後ろ足で交互に踏み出して歩いているか",
    "xy_dist_range_m": "窓内で、開始位置からの水平距離 xy_dist_m が動いた幅（最大−最小）[m]。姿勢を保っている間に流れていないか（立ち上がりの移動は含まない）",
    "yaw_rate_steady_fraction": "窓内で旋回速度が平均の±max(15, 0.3×|平均|)[deg/s]に収まっていた割合（一定の向きに回り続けているか）",
}
_WINDOW_SUFFIXES = ("_mean", "_max", "_min")
_RULES = ("between", "min", "max", "equals")
_ALLOWED_KEYS = {"metric", "min_fraction", "comment", *_RULES}


class SpecError(ValueError):
    """The spec file is malformed or refers to an unknown metric."""


class NotApprovedError(SpecError):
    """The spec has not been approved by a human (`approved: true`)."""


def metric_kind(name):
    """Return 'series', 'window' or 'episode' for a catalog metric name, or None if unknown."""
    if name in SERIES_METRICS:
        return "series"
    if name in EPISODE_METRICS:
        return "episode"
    if name in WINDOW_SPECIAL:
        return "window"
    for suffix in _WINDOW_SUFFIXES:
        if name.endswith(suffix) and name[: -len(suffix)] in SERIES_METRICS:
            return "window"
    return None


def catalog_text():
    lines = ["# 使える指標"]
    lines += [f"- {k} (系列): {v}" for k, v in SERIES_METRICS.items()]
    lines += ["- <系列名>_mean / _max / _min (窓内の集計): 例 yaw_rate_deg_s_mean, dof_vel_rms_mean"]
    lines += [f"- {k} (窓内): {v}" for k, v in WINDOW_SPECIAL.items()]
    lines += [f"- {k} (エピソード): {v}" for k, v in EPISODE_METRICS.items()]
    return "\n".join(lines)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_spec(spec):
    """Raise SpecError on any structural problem."""
    if not isinstance(spec, dict):
        raise SpecError("spec must be a mapping")
    if not isinstance(spec.get("task"), str) or not spec["task"]:
        raise SpecError("`task` must be a non-empty string")
    if not isinstance(spec.get("approved"), bool):
        raise SpecError("`approved` must be true or false")
    w = spec.get("window_s")
    if not (isinstance(w, list) and len(w) == 2 and all(_is_number(v) for v in w) and 0 <= w[0] < w[1]):
        raise SpecError("`window_s` must be [start, end] in seconds with 0 <= start < end")
    crit = spec.get("criteria")
    if not isinstance(crit, list) or not crit:
        raise SpecError("`criteria` must be a non-empty list")
    for i, c in enumerate(crit):
        where = f"criteria[{i}]"
        if not isinstance(c, dict):
            raise SpecError(f"{where} must be a mapping")
        extra = set(c) - _ALLOWED_KEYS
        if extra:
            raise SpecError(f"{where} has unknown keys: {sorted(extra)}")
        name = c.get("metric")
        kind = metric_kind(name) if isinstance(name, str) else None
        if kind is None:
            raise SpecError(f"{where} unknown metric {name!r}\n{catalog_text()}")
        rules = [r for r in _RULES if r in c]
        if len(rules) != 1:
            raise SpecError(f"{where} needs exactly one of {list(_RULES)}, got {rules}")
        rule = rules[0]
        if rule == "between":
            v = c["between"]
            if not (isinstance(v, list) and len(v) == 2 and all(_is_number(x) for x in v) and v[0] < v[1]):
                raise SpecError(f"{where} `between` must be [lo, hi] with lo < hi")
        elif not _is_number(c[rule]):
            raise SpecError(f"{where} `{rule}` must be a finite number")
        if kind == "series":
            f = c.get("min_fraction")
            if not (_is_number(f) and 0 < f <= 1):
                raise SpecError(f"{where} series metric `{name}` needs `min_fraction` in (0, 1]")
        elif "min_fraction" in c:
            raise SpecError(f"{where} `min_fraction` is only for series metrics, not `{name}`")


def load_spec(path, require_approved=True):
    """Load and validate a spec. Returns (spec, raw_text, sha256_of_raw_text)."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    spec = yaml.safe_load(raw)
    validate_spec(spec)
    if require_approved and not spec["approved"]:
        raise NotApprovedError(
            f"{path} is not approved (approved: false). Review the criteria and set `approved: true` yourself; "
            "the loop must not run with an unapproved spec."
        )
    return spec, raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _series(metrics, name):
    """Return the 0.1 s series for `name` (derived from the stored series), or None if unavailable."""
    s = metrics.get("series_0p1s") or {}
    if name in ("nose_up_deg", "height_m", "xy_dist_m", "front_feet_asym_m", "rear_feet_asym_m", "dof_vel_rms"):
        return list(s[name]) if name in s else None
    if name == "roll_deg_abs":
        return [abs(v) for v in s["roll_deg"]] if "roll_deg" in s else None
    if name == "foot_z_max_m":
        keys = [f"foot_z_{n}" for n in ("FL", "FR", "RL", "RR")]
        return [max(vals) for vals in zip(*(s[k] for k in keys))] if all(k in s for k in keys) else None
    if name in ("foot_z_FL_m", "foot_z_FR_m"):
        key = name[:-2]
        return list(s[key]) if key in s else None
    if name == "front_feet_z_min_m":
        return [min(v) for v in zip(*(s[k] for k in ("foot_z_FL", "foot_z_FR")))] if all(k in s for k in ("foot_z_FL", "foot_z_FR")) else None
    if name == "front_one_foot_up":
        if not all(k in s for k in ("foot_z_FL", "foot_z_FR")):
            return None
        return [1.0 if (a > 0.05) != (b > 0.05) else 0.0 for a, b in zip(s["foot_z_FL"], s["foot_z_FR"])]
    if name in ("foot_z_RL_m", "foot_z_RR_m"):
        key = name[:-2]
        return list(s[key]) if key in s else None
    if name == "rear_one_foot_up":
        if not all(k in s for k in ("foot_z_RL", "foot_z_RR")):
            return None
        return [1.0 if (a > 0.05) != (b > 0.05) else 0.0 for a, b in zip(s["foot_z_RL"], s["foot_z_RR"])]
    if name == "all_feet_up":
        keys = [f"foot_z_{n}" for n in ("FL", "FR", "RL", "RR")]
        return [1.0 if min(vals) > 0.05 else 0.0 for vals in zip(*(s[k] for k in keys))] if all(k in s for k in keys) else None
    if name == "head_z_m":
        return list(s["head_z"]) if "head_z" in s else None
    if name == "front_knee_z_min_m":
        return [min(v) for v in zip(*(s[k] for k in ("knee_z_FL", "knee_z_FR")))] if all(k in s for k in ("knee_z_FL", "knee_z_FR")) else None
    if name in ("front_feet_z_max_m", "rear_feet_z_min_m"):
        keys, agg = (("foot_z_FL", "foot_z_FR"), max) if name == "front_feet_z_max_m" else (("foot_z_RL", "foot_z_RR"), min)
        return [agg(vals) for vals in zip(*(s[k] for k in keys))] if all(k in s for k in keys) else None
    if name in ("yaw_rate_deg_s", "yaw_rate_abs_deg_s"):
        if "yaw_deg" not in s:
            return None
        y = s["yaw_deg"]
        # rate[k] belongs to sample k (k >= 1); rate[0] is undefined
        rate = [None] + [(y[k] - y[k - 1]) / SAMPLE_PERIOD_S for k in range(1, len(y))]
        if name == "yaw_rate_abs_deg_s":
            rate = [None if r is None else abs(r) for r in rate]
        return rate
    return None


def _window(values, window_s):
    """Samples with t in [t0, t1]; None entries (undefined) are dropped."""
    t0, t1 = window_s
    k0 = int(math.ceil(t0 / SAMPLE_PERIOD_S - 1e-9))
    k1 = int(math.floor(t1 / SAMPLE_PERIOD_S + 1e-9))
    return [v for v in values[k0 : k1 + 1] if v is not None]


def _satisfies(v, c):
    if "between" in c:
        lo, hi = c["between"]
        return lo <= v <= hi
    if "min" in c:
        return v >= c["min"]
    if "max" in c:
        return v <= c["max"]
    return math.isclose(v, c["equals"], abs_tol=1e-9)


def _margin(v, c):
    """Distance to the nearest boundary in metric units; positive when satisfied."""
    if "between" in c:
        lo, hi = c["between"]
        return min(v - lo, hi - v)
    if "min" in c:
        return v - c["min"]
    if "max" in c:
        return c["max"] - v
    return -abs(v - c["equals"])


def describe_rule(c):
    if "between" in c:
        return f"{c['between'][0]:g}〜{c['between'][1]:g}"
    if "min" in c:
        return f"{c['min']:g} 以上"
    if "max" in c:
        return f"{c['max']:g} 以下"
    return f"= {c['equals']:g}"


def _fit_circle(xs, ys):
    """Least-squares circle (Kasa fit) through the points. Returns (cx, cy, radius, rms, swept_deg) or None."""
    n = len(xs)
    if n < 5:
        return None
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    syy = sum(y * y for y in ys)
    sxy = sum(x * y for x, y in zip(xs, ys))
    sxz = sum(x * (x * x + y * y) for x, y in zip(xs, ys))
    syz = sum(y * (x * x + y * y) for x, y in zip(xs, ys))
    sz = sum(x * x + y * y for x, y in zip(xs, ys))
    # normal equations of  [x y 1] . [a b c] = x^2 + y^2
    m = [[sxx, sxy, sx, sxz], [sxy, syy, sy, syz], [sx, sy, float(n), sz]]
    for i in range(3):  # Gaussian elimination with partial pivoting
        piv = max(range(i, 3), key=lambda r: abs(m[r][i]))
        if abs(m[piv][i]) < 1e-9:
            return None
        m[i], m[piv] = m[piv], m[i]
        for r in range(i + 1, 3):
            f = m[r][i] / m[i][i]
            m[r] = [a - f * b for a, b in zip(m[r], m[i])]
    c = m[2][3] / m[2][2]
    b = (m[1][3] - m[1][2] * c) / m[1][1]
    a = (m[0][3] - m[0][1] * b - m[0][2] * c) / m[0][0]
    cx, cy = a / 2.0, b / 2.0
    r2 = c + cx * cx + cy * cy
    if r2 <= 1e-9 or r2 > 1e6:
        return None
    r = math.sqrt(r2)
    rms = math.sqrt(sum((math.hypot(x - cx, y - cy) - r) ** 2 for x, y in zip(xs, ys)) / n)
    ang = [math.atan2(y - cy, x - cx) for x, y in zip(xs, ys)]
    swept = 0.0
    for k in range(1, n):
        d = ang[k] - ang[k - 1]
        d = (d + math.pi) % (2 * math.pi) - math.pi
        swept += d
    return cx, cy, r, rms, abs(math.degrees(swept))


def _window_value(metrics, name, window_s):
    """Scalar for a window-kind metric, or (None, note)."""
    if name == "yaw_rate_steady_fraction":
        rate = _series(metrics, "yaw_rate_deg_s")
        if rate is None:
            return None, "yaw_deg の系列が無い（新しい record.py で録画し直す）"
        w = _window(rate, window_s)
        if not w:
            return None, "窓内に有効なサンプルが無い（機体が垂直に近く向きが定義できない等）"
        m = sum(w) / len(w)
        tol = max(15.0, 0.3 * abs(m))
        return sum(1 for r in w if abs(r - m) <= tol) / len(w), ""
    if name in ("circle_radius_m", "circle_fit_rms_m", "circle_arc_deg"):
        s = metrics.get("series_0p1s") or {}
        if "x_m" not in s or "y_m" not in s:
            return None, "x_m / y_m の系列が無い（録画し直しが必要）"
        xs, ys = _window(s["x_m"], window_s), _window(s["y_m"], window_s)
        fit = _fit_circle(xs, ys)
        if fit is None:
            return None, "窓内の軌跡が円に当てはまらない（ほぼ直線か、動いていない）"
        cx, cy, r, rms, arc = fit
        return {"circle_radius_m": r, "circle_fit_rms_m": rms, "circle_arc_deg": arc}[name], ""
    if name in ("front_step_count_min", "rear_step_count_min"):
        s = metrics.get("series_0p1s") or {}
        pair = ("foot_z_FL", "foot_z_FR") if name.startswith("front") else ("foot_z_RL", "foot_z_RR")
        if not all(k in s for k in pair):
            return None, "足の高さの系列が無い（録画し直しが必要）"
        counts = []
        for k in pair:
            w = _window(s[k], window_s)
            counts.append(sum(1 for a, b in zip(w, w[1:]) if a <= 0.06 < b))
        return float(min(counts)), ""
    if name == "xy_dist_range_m":
        w = _window(_series(metrics, "xy_dist_m") or [], window_s)
        return (max(w) - min(w), "") if w else (None, "窓内に有効なサンプルが無い")
    for suffix in _WINDOW_SUFFIXES:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            vals = _series(metrics, base)
            if vals is None:
                return None, f"{base} の系列が無い（録画し直しが必要）"
            w = _window(vals, window_s)
            if not w:
                return None, "窓内に有効なサンプルが無い"
            agg = {"_mean": lambda x: sum(x) / len(x), "_max": max, "_min": min}[suffix]
            return agg(w), ""
    return None, "未対応の指標"


def evaluate(metrics, spec):
    """Judge `metrics` (a metrics_<ckpt>.json dict) against a validated spec."""
    validate_spec(spec)
    results = []
    for c in spec["criteria"]:
        name = c["metric"]
        kind = metric_kind(name)
        rec = {"metric": name, "kind": kind, "rule": describe_rule(c), "min_fraction": c.get("min_fraction"),
               "measured": None, "window_mean": None, "margin": None, "passed": False, "note": ""}
        if kind == "series":
            vals = _series(metrics, name)
            if vals is None:
                rec["note"] = f"{name} の系列がメトリクスに無い（録画し直しが必要）"
            else:
                w = _window(vals, spec["window_s"])
                if not w:
                    rec["note"] = "窓内に有効なサンプルが無い（機体が垂直に近く向きが定義できない等）"
                else:
                    frac = sum(1 for v in w if _satisfies(v, c)) / len(w)
                    rec["measured"] = round(frac, 3)
                    rec["window_mean"] = round(sum(w) / len(w), 3)
                    rec["passed"] = frac >= c["min_fraction"] - 1e-9
        else:
            if kind == "episode":
                v = metrics.get(name)
                note = "" if v is not None else f"{name} がメトリクスに無い（録画し直しが必要）"
            else:
                v, note = _window_value(metrics, name, spec["window_s"])
            if v is None:
                rec["note"] = note
            else:
                rec["measured"] = round(float(v), 3)
                rec["margin"] = round(_margin(v, c), 3)
                rec["passed"] = _satisfies(v, c)
        results.append(rec)
    return {"task": spec["task"], "approved": spec["approved"], "window_s": spec["window_s"],
            "passed": all(r["passed"] for r in results),
            "n_passed": sum(1 for r in results if r["passed"]), "n_criteria": len(results), "criteria": results}


def _line(r):
    mark = "合格" if r["passed"] else "不合格"
    if r["measured"] is None:
        return f"[{mark}] {r['metric']} 目標 {r['rule']}: 判定できない（{r['note']}）"
    if r["kind"] == "series":
        return (f"[{mark}] {r['metric']} 目標 {r['rule']}: 窓内の {r['measured'] * 100:.0f}% が範囲内"
                f"（必要 {r['min_fraction'] * 100:.0f}%、窓内平均 {r['window_mean']:g}）")
    return f"[{mark}] {r['metric']} 目標 {r['rule']}: 実測 {r['measured']:g}（余裕 {r['margin']:+g}）"


def format_result(result):
    head = f"数値の成功判定 [{result['task']}]: {'合格' if result['passed'] else '不合格'}（{result['n_passed']}/{result['n_criteria']}）"
    return "\n".join([head] + [_line(r) for r in result["criteria"]])


def failure_summary(result):
    """Failed criteria only, for the next reward request."""
    fails = [_line(r) for r in result["criteria"] if not r["passed"]]
    return "\n".join(fails) if fails else "（全基準を満たした）"
