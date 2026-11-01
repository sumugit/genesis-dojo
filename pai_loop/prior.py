"""Carry the findings of an earlier experiment into a new one (after the success spec was revised).

Only words and code are carried over: the best loop's reward code, its reward-term averages and the AI scorer's findings go
into the reward-writing prompt of the first loop. Trained weights are not reused. Pure stdlib (tested without torch).
"""

from __future__ import annotations

import json
import os


def load_entries(root, exp):
    """The loop records of an experiment (logs/loops/<exp>.jsonl), oldest first; [] when there are none."""
    path = os.path.join(root, "logs", "loops", f"{exp}.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def rank(entry):
    """A success first, then the mean of the numeric pass rate and the AI score (an error loop ranks last)."""
    if entry.get("error") or not entry.get("reward_code"):
        return -1.0
    if (entry.get("decision") or {}).get("success"):
        return 2.0
    chk, ev = entry.get("success_check") or {}, entry.get("evaluation") or {}
    try:
        num = chk["n_passed"] / chk["n_criteria"]
        return (num + float(ev["score"]) / 10) / 2
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return 0.0


def pick_prior(entries):
    """The best loop of an earlier experiment (later loops win ties), or None when no loop has usable reward code."""
    best = None
    for e in entries:
        if rank(e) >= 0 and (best is None or rank(e) >= rank(best)):
            best = e
    return best


def prior_section(entry, prev_exp):
    parts = [f"# 前の実験（{prev_exp}）の最良の周の結果",
             f"前の実験は、成功条件が不十分（緩すぎる、または測り方が合わない）だったため、今回の成功条件へ見直されました。"
             f"前の周の数値判定は、古い成功条件での結果です。今回の成功条件では、判定が変わります。"]
    parts.append(f"## 報酬コード（{entry.get('exp_name', prev_exp)}）\n```python\n{entry['reward_code']}\n```")
    if entry.get("train_terms"):
        parts.append("## 学習終盤の各報酬項の平均（1エピソード・1秒あたり）\n" + json.dumps(entry["train_terms"], ensure_ascii=False))
    if entry.get("evaluation"):
        parts.append("## 動画評価（審査員。報酬コードは見ていない）\n" + json.dumps(entry["evaluation"], ensure_ascii=False))
    parts.append("このコードを土台にして、動画評価の指摘と今回の成功条件に合うよう書き直してください。ゼロから書き直す必要はありません。")
    return "\n\n".join(parts)
