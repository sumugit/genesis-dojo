"""Revise a success spec after a run: diagnose where the numeric check and the AI score disagreed (or why nothing passed),
propose a new spec version, and preview it on the loops that were already recorded (no training needed).

Nothing here writes a spec: the person reviews and approves the proposal (`dashboard/authoring.py` does the writing).
"""

from __future__ import annotations

import json
import os
import re

import yaml

from dashboard import authoring as au
from dashboard import state as st
from pai_loop import success

_METRICS = re.compile(r"^metrics_(\d+)\.json$")


def load_metrics(root, exp, loop):
    """metrics_<ckpt>.json of the loop's own recording (the latest checkpoint), or None."""
    if not st.EXP_RE.match(exp) or not isinstance(loop, int) or not 0 < loop < 1000:
        return None
    d = os.path.join(root, "runs", f"{exp}_L{loop}")
    files = sorted(((int(m.group(1)), f) for f in (os.listdir(d) if os.path.isdir(d) else []) if (m := _METRICS.match(f))))
    return st._read_json(os.path.join(d, files[-1][1])) if files else None


def load_spec_dict(root, task):
    try:
        with open(os.path.join(root, "configs", "tasks", f"{task}.success.yaml"), encoding="utf-8") as f:
            raw = f.read()
        return yaml.safe_load(raw), raw
    except (OSError, yaml.YAMLError):
        return None, ""


def next_task_id(root, task):
    base = re.sub(r"_v\d+$", "", task)
    n = 2
    while os.path.exists(os.path.join(root, "configs", "tasks", f"{base}_v{n}.success.yaml")):
        n += 1
    return f"{base}_v{n}"


def _scalars(metrics, limit=40):
    """The scalar summaries of a recording (the series are left out), for the prompt."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("ckpt", "seconds", "metrics_after_s", "record_wall_seconds"):
            out[k] = v
    return dict(list(out.items())[:limit])


def loop_summaries(root, state):
    out = []
    for lp in state["loops"]:
        n, e, d = lp.get("numeric") or {}, lp.get("evaluation") or {}, lp.get("decision") or {}
        m = load_metrics(root, state["exp"], lp["loop"])
        out.append({"loop": lp["loop"], "numeric": f"{n.get('n_passed')}/{n.get('n_criteria')}" if n else None,
                    "criteria": [{"metric": c["metric"], "rule": c["rule"], "measured": c["measured"], "passed": c["passed"]} for c in n.get("criteria", [])],
                    "ai_score": e.get("score"), "ai_success": e.get("success"), "ai_observations": e.get("observations"),
                    "ai_problems": e.get("problems"), "disagreement": d.get("disagreement"), "metrics": _scalars(m) if m else None})
    return out


def improve_prompt(state, spec_raw, loops):
    if state.get("stop_reason") == "success":
        intro = ("結果: 数値の成功条件とAI採点の両方を満たして、成功しました。ただし、AI採点には、まだ改善点の指摘（ai_problems、ai_observations）が残っています。\n"
                 "その改善点が、次の学習で直るように、成功条件を見直したいです（改善点を測る基準を足す、または既存の基準を厳しくする）。"
                 "改善点が出ている動きは、新しい基準で不合格になって構いません。ただし、基準を厳しくしすぎて、改善の余地のない動きまで落とさないこと。")
        task_2 = "2. 改善した成功条件: 上の改善点が直った動きだけが通るようにする（改善点を測る基準を足す、または既存の基準を厳しくする）。使えるのは上の指標だけ。窓（window_s）は録画の長さ以内。"
        task_1 = "1. 改善点の整理: AIの指摘のうち、数値で測れるものを挙げる（例: 左右の非対称、動きの滑らかさ、ばたつき）。測れないものは、そう書く。"
    else:
        intro = "結果が思わしくない（成功しなかった、または2つの判定が食い違った）ので、成功条件を見直したいです。"
        task_2 = "2. 改善した成功条件: 指示の意図を満たす動きだけが通り、意図から外れた動き（診断で見つけた抜け道）は通らないようにする。AIが成功とみなした動きは通ること。使えるのは上の指標だけ。窓（window_s）は録画の長さ以内。"
        task_1 = ("1. 診断: 数値判定が合格なのにAIが不合格にした周は、基準が緩い・抜け道がある（指示の意図を満たしていない動きが通る）。逆に、AIが合格にしたのに数値が不合格の周は、基準が厳しすぎる・測り方が合わない。"
                  "どの基準が原因か、具体的な測定値を挙げて説明する。基準ではなく、動きそのものが未熟（学習が足りない）という場合も、そう書く。")
    return f"""四脚ロボット（Unitree Go2、シミュレータ）に覚えさせる芸「{state['instruction']}」の学習を行い、数値の成功条件とAIの採点（動画を見て10点満点。8点以上で合格）の両方で判定しました。
{intro}

## 現在の成功条件（{state['task']}）
{spec_raw}

## 各周の結果（数値判定の各基準の測定値と合否、AIの採点、録画から計算した指標）
{json.dumps(loops, ensure_ascii=False, indent=1)}

{success.catalog_text()}

## やってほしいこと
{task_1}
{task_2}
   系列の指標には min_fraction が必須。基準は、必要なものだけにする（増やしすぎない）。
3. ヒント: 次の学習で報酬を書く LLM に渡す、設計上の短い助言（日本語、2〜4項目。例: 着地したら静止すること）。助言が不要なら空文字。

次のJSONだけを返してください:
{{"diagnosis": "診断（日本語、3〜6文）", "window_s": [3.0, 6.0],
  "criteria": [{{"metric": "...", "min": 0.0, "min_fraction": 0.9, "comment": "..."}}],
  "hint": "...", "rationale": "変更した点と理由（日本語、箇条書き風に2〜5文）"}}"""


def preview(root, state, draft):
    """How the (draft) spec would have judged each recorded loop: [{loop, n_passed, n_criteria, passed, old, ai_score, ai_pass, agrees}]."""
    spec = au.build_spec(draft, False)
    success.validate_spec(spec)
    rows = []
    for lp in state["loops"]:
        m = load_metrics(root, state["exp"], lp["loop"])
        if m is None:
            continue
        res = success.evaluate(m, spec)
        e, n = lp.get("evaluation") or {}, lp.get("numeric") or {}
        ai_pass = bool(e) and bool(e.get("success")) and float(e.get("score", 0)) >= 8
        rows.append({"loop": lp["loop"], "n_passed": res["n_passed"], "n_criteria": res["n_criteria"], "passed": res["passed"],
                     "old": f"{n.get('n_passed')}/{n.get('n_criteria')}" if n else None, "old_passed": n.get("passed"),
                     "ai_score": e.get("score"), "ai_pass": ai_pass if e else None, "agrees": (res["passed"] == ai_pass) if e else None})
    return rows
