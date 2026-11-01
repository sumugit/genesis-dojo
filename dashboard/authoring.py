"""Create and approve a success spec from the dashboard (so that a person never has to edit YAML or commit).

Flow: a person types an instruction -> an LLM drafts criteria over the fixed metric catalog (`pai_loop/success.py`)
-> the page shows the draft for the person to edit -> only an explicit approval request (a click by the person) writes
`configs/tasks/<task>.success.yaml` with `approved: true`. The LLM output is never written as approved by itself, and
the written spec is the validated rebuild of the submitted fields (not text sent by the client).
"""

from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import re

import yaml

from pai_loop import success

TASK_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")
GENERATED = "catalog.generated.yaml"   # tasks created on the dashboard (configs/tasks/)
RULE_KEYS = ("between", "min", "max", "equals")


def list_bases(root):
    """Base configs the draft may choose from: [{path, note, record_s}]."""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "configs", "base*.yaml"))):
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read()
            cfg = yaml.safe_load(text) or {}
        except (OSError, yaml.YAMLError):
            continue
        first = next((ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("#")), "")
        out.append({"path": os.path.join("configs", os.path.basename(p)), "note": first[:160],
                    "record_s": float((cfg.get("record") or {}).get("seconds", 0) or 0)})
    return out


def metric_list():
    """Names a criterion may use: [{name, kind, text}]."""
    out = [{"name": k, "kind": "series", "text": v} for k, v in success.SERIES_METRICS.items()]
    out += [{"name": f"{k}{sfx}", "kind": "window", "text": f"{v}（窓内の{ {'_mean': '平均', '_max': '最大', '_min': '最小'}[sfx] }）"}
            for k, v in success.SERIES_METRICS.items() for sfx in success._WINDOW_SUFFIXES]
    out += [{"name": k, "kind": "window", "text": v} for k, v in success.WINDOW_SPECIAL.items()]
    out += [{"name": k, "kind": "episode", "text": v} for k, v in success.EPISODE_METRICS.items()]
    return out


def draft_prompt(root, instruction, existing, hint=None):
    extra = ""
    if hint:   # a gait the library knows: the criteria that keep out odd ways of walking, and the base made for it
        extra = (f"\n## 参考: 「{hint['name']}」の成功条件（人が用意した例。指示に合わせて調整してよい）\n{hint['spec_notes']}\n"
                 f"学習の土台は、通常 {hint['base']} を選ぶ。\n" if hint.get("spec_notes") else "")
    bases = "\n".join(f"- {b['path']}（録画 {b['record_s']:g} 秒）: {b['note']}" for b in list_bases(root))
    return f"""あなたは、四脚ロボット（Unitree Go2、シミュレータ）に覚えさせる「芸」の、数値の成功条件を設計します。
人が日本語で指示を書きました。その指示を、動画から機械的に測れる指標の条件に落とし込んでください。
この条件は、人が確認・修正してから承認し、承認後は学習中に変更されません。あとで合否を決める基準になります。

指示: 「{instruction}」

{success.catalog_text()}

条件の書き方:
- 判定は、録画の「窓」（window_s = [開始秒, 終了秒]）の間で行う。立ち上がりが落ち着く時間（約3秒）より後にする。窓の終了は録画の長さ以内。
- 各条件は、metric と、between: [下限, 上限] / min / max / equals のうち1つだけ、comment（日本語の短い説明）を持つ。
- 系列の指標（(系列)と書かれたもの）には、min_fraction（窓内でその条件を満たす時間の割合。0〜1。通常 0.9）が必須。窓内の集計・エピソードの指標には付けない。
- 立位の高さは約0.29m、足先の接地は約0.023m。条件は、見た目で成功といえる動きを通し、失敗や抜け道（例: 頭や肘で支える、滑る）は通さない値にする。
- 倒れないこと（early_terminations equals 0）は、通常入れる。
- 指標だけでは測れない指示（例: 物体をつかむ、特定の方向を向く）や、安全に学習できない指示のときは、criteria を空にして cannot_measure に理由を書く。

{extra}
学習の土台（base）は、次から1つ選ぶ。通常の立位から始める芸は configs/base.yaml:
{bases}

既に承認済みの task 名（重複しないこと）: {', '.join(existing) or 'なし'}

次のJSONだけを返してください（説明文は付けない）:
{{"task_id": "半角小文字と_だけの英語名（例: front_flip）", "name": "短い日本語名", "description": "この芸の説明（1〜2文）",
  "base": "上の一覧のパス", "window_s": [3.0, 6.0],
  "criteria": [{{"metric": "...", "min": 0.0, "min_fraction": 0.9, "comment": "..."}}],
  "rationale": "この基準にした理由（日本語、3〜5文。値の根拠と、見逃しうる点も）", "cannot_measure": null}}"""


def parse_draft(text):
    """The JSON object in an LLM reply as a dict, or None."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    return d if isinstance(d, dict) else None


def _clean_criterion(c):
    out = {"metric": c.get("metric")}
    for k in RULE_KEYS:
        if k in c:
            out[k] = c[k]
    if "min_fraction" in c and c["min_fraction"] is not None:
        out["min_fraction"] = c["min_fraction"]
    if c.get("comment"):
        out["comment"] = str(c["comment"])[:120]
    return out


def build_spec(draft, approved):
    """The spec dict for a (client or LLM) draft. Only known fields are taken."""
    return {"task": draft.get("task_id"), "approved": approved, "window_s": draft.get("window_s"),
            "criteria": [_clean_criterion(c) for c in draft.get("criteria") or [] if isinstance(c, dict)]}


def check_draft(root, draft, existing=()):
    """List of problems (Japanese, for the page); empty when the draft can be approved."""
    errs = []
    task = draft.get("task_id")
    if not isinstance(task, str) or not TASK_RE.match(task):
        errs.append("芸の名前（task_id）は、半角小文字と数字と _ の2〜41文字にしてください")
    elif task in existing:
        errs.append(f"芸の名前 {task} は、すでに使われています")
    base = next((b for b in list_bases(root) if b["path"] == draft.get("base")), None)
    if base is None:
        errs.append("学習の土台（base）が一覧にありません")
    try:
        success.validate_spec(build_spec(draft, False))
    except success.SpecError as e:
        errs.append(str(e).split("\n")[0])
    else:
        w = draft["window_s"]
        if base and base["record_s"] and w[1] > base["record_s"] + 1e-9:
            errs.append(f"判定の窓の終わり（{w[1]}秒）が、録画の長さ（{base['record_s']:g}秒）を超えています")
    return errs


def spec_text(draft, instruction, approved, when):
    spec = build_spec(draft, approved)
    head = (f"# 「{instruction}」の数値の成功条件。ダッシュボードで作成（{when}）。\n"
            f"# {'人がダッシュボード上で確認・修正して承認した（approved: true）。' if approved else '草案（未承認）。'}\n"
            f"# 学習の土台: {draft.get('base')}\n")
    body = yaml.safe_dump(spec, allow_unicode=True, sort_keys=False, default_flow_style=None, width=200)
    return head + body


def _load_generated(root):
    try:
        with open(os.path.join(root, "configs", "tasks", GENERATED), encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("tasks") or []
    except (OSError, yaml.YAMLError):
        return []


def approve(root, draft, instruction, existing, now=None):
    """Write the approved spec and register the task. Returns (task_id, sha256) or raises ValueError(list of problems)."""
    errs = check_draft(root, draft, existing)
    if errs:
        raise ValueError(errs)
    when = (now or datetime.datetime.now()).strftime("%Y-%m-%d %H:%M")
    text = spec_text(draft, instruction, True, when)
    task = draft["task_id"]
    path = os.path.join(root, "configs", "tasks", f"{task}.success.yaml")
    with open(path, "x", encoding="utf-8") as f:   # never overwrites an approved spec
        f.write(text)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    entries = _load_generated(root) + [{"task": task, "name": str(draft.get("name") or task)[:40],
                                       "description": str(draft.get("description") or "")[:200],
                                       "examples": [instruction], "base": draft["base"]}]
    with open(os.path.join(root, "configs", "tasks", GENERATED), "w", encoding="utf-8") as f:
        yaml.safe_dump({"tasks": entries}, f, allow_unicode=True, sort_keys=False)
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    with open(os.path.join(root, "logs", "approvals.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": when, "task": task, "sha256": sha, "instruction": instruction, "via": "dashboard"},
                           ensure_ascii=False) + "\n")
    return task, sha


def default_draft_ask(timeout_s=240):
    from pai_loop.llm import make_llm

    llm = make_llm("claude-cli", timeout_s=timeout_s)
    return lambda prompt: llm.ask(prompt).text
