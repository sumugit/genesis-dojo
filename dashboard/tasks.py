"""Pick the success spec for an instruction (so that a person only types the instruction).

The catalog (configs/tasks/catalog.yaml) lists the tasks that can be started; only those whose success spec is
approved are offered. An LLM maps the instruction to one of them, or to none. Nothing here creates or approves a spec.
"""

from __future__ import annotations

import json
import os
import re

import yaml

from dashboard import state as st


def load_catalog(root):
    """[{task, name, description, examples, criteria}] for the catalog entries with an approved success spec."""
    entries = []
    for name in ("catalog.yaml", "catalog.generated.yaml"):   # the second one is written when a person approves a spec in the UI
        try:
            with open(os.path.join(root, "configs", "tasks", name), encoding="utf-8") as f:
                entries += (yaml.safe_load(f) or {}).get("tasks") or []
        except (OSError, yaml.YAMLError):
            continue
    approved = {t["task"]: t for t in st.list_tasks(root) if t["approved"]}
    out = []
    for e in entries:
        if isinstance(e, dict) and e.get("task") in approved:
            out.append({"task": e["task"], "name": str(e.get("name", e["task"])), "description": str(e.get("description", "")),
                        "examples": [str(x) for x in e.get("examples") or []], "criteria": approved[e["task"]]["criteria"],
                        "base": e.get("base")})
    # a revised spec (…_v2, …_v3) replaces the older versions of the same task in the offers
    def fam(t):
        m = re.match(r"^(.*)_v(\d+)$", t)
        return (m.group(1), int(m.group(2))) if m else (t, 1)
    latest = {}
    for c in out:
        f, v = fam(c["task"])
        if f not in latest or v >= fam(latest[f]["task"])[1]:
            latest[f] = c
    return [c for c in out if latest[fam(c["task"])[0]] is c]


def choose_prompt(catalog, instruction):
    lines = "\n".join(f"- {c['task']}: {c['name']}。{c['description']}（例: {'、'.join(c['examples'])}）" for c in catalog)
    return ("四脚ロボットに覚えさせる芸の指示が、日本語で与えられます。次の候補のうち、指示に最も合うものを1つ選んでください。\n"
            "どの候補にも合わない（別の芸、意味が通らない、など）ときは null にしてください。\n\n"
            f"候補:\n{lines}\n\n指示: 「{instruction}」\n\n"
            'JSONだけを返してください。例: {"task": "sit"}  または  {"task": null}')


def parse_choice(text, catalog):
    """The task id in the reply, or None (also for ids that are not in the catalog)."""
    m = re.search(r"\{.*?\}", text or "", re.S)
    if not m:
        return None
    try:
        task = json.loads(m.group(0)).get("task")
    except (ValueError, AttributeError):
        return None
    return task if any(c["task"] == task for c in catalog) else None


def choose_task(root, instruction, ask):
    """(task, catalog). `ask(prompt) -> reply text`. task is None when no approved task fits."""
    catalog = load_catalog(root)
    if not catalog:
        return None, catalog
    return parse_choice(ask(choose_prompt(catalog, instruction)), catalog), catalog


def default_ask(model="haiku", timeout_s=90):
    from pai_loop.llm import make_llm

    llm = make_llm("claude-cli", model=model, timeout_s=timeout_s)
    return lambda prompt: llm.ask(prompt).text


# ---- gait design hints (configs/hints/index.yaml): human-written advice for the reward writer ----
def load_hints(root):
    """[{id, name, applies, file, base, spec_notes}] for the hints whose file exists."""
    try:
        with open(os.path.join(root, "configs", "hints", "index.yaml"), encoding="utf-8") as f:
            entries = (yaml.safe_load(f) or {}).get("hints") or []
    except (OSError, yaml.YAMLError):
        return []
    return [{"id": e["id"], "name": str(e.get("name", e["id"])), "applies": str(e.get("applies", "")), "file": e["file"],
             "base": e.get("base"), "spec_notes": str(e.get("spec_notes", "")).strip()}
            for e in entries if isinstance(e, dict) and e.get("id") and e.get("file")
            and re.match(r"^configs/[A-Za-z0-9_./-]+$", str(e["file"])) and ".." not in e["file"]
            and os.path.isfile(os.path.join(root, e["file"]))]


def hint_text(root, hint):
    with open(os.path.join(root, hint["file"]), encoding="utf-8") as f:
        return f.read().strip()


def choose_hint(root, instruction, ask):
    """The hint (dict) that fits the instruction, or None. `ask(prompt) -> reply text`."""
    hints = load_hints(root)
    if not hints:
        return None
    lines = "\n".join(f"- {h['id']}: {h['name']}。{h['applies']}" for h in hints)
    reply = ask("四足ロボットに覚えさせる芸の指示が、日本語で与えられます。歩き方の設計ヒントのうち、指示に当てはまるものを1つ選んでください。"
                "どれにも当てはまらないときは null にしてください。\n\n"
                f"候補:\n{lines}\n\n指示: 「{instruction}」\n\nJSONだけを返してください。例: " + '{"hint": "walk_quadruped"}  または  {"hint": null}')
    m = re.search(r"\{.*?\}", reply or "", re.S)
    try:
        hid = json.loads(m.group(0)).get("hint") if m else None
    except (ValueError, AttributeError):
        return None
    return next((h for h in hints if h["id"] == hid), None)
