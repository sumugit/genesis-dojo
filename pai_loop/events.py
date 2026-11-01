"""Stage events of an automatic loop, appended to logs/events/<exp>.jsonl (read by the dashboard).

One JSON object per line: {"t": epoch seconds, "exp", "loop" (1-based or None), "stage", "status", "data"}.
Stages follow the flow shown on the dashboard; "status" is "start", "end" or "error".
Pure stdlib so that it can be tested without torch.
"""

from __future__ import annotations

import json
import os
import time

EVENT_DIR = os.path.join("logs", "events")

# Order of the stages in the flow (the dashboard draws them in this order).
STAGES = ["instruction", "reward", "check", "train", "record", "numeric", "vlm", "decision"]
STAGE_LABELS = {
    "instruction": "指示",
    "reward": "報酬生成",
    "check": "検証",
    "train": "学習",
    "record": "録画",
    "numeric": "数値判定",
    "vlm": "VLM採点",
    "decision": "判定",
}


def event_path(exp, root="."):
    return os.path.join(root, EVENT_DIR, f"{exp}.jsonl")


def emit(exp, stage, status, loop=None, root=".", **data):
    """Append one event. Never raises: a logging problem must not stop a training loop."""
    rec = {"t": round(time.time(), 3), "exp": exp, "loop": loop, "stage": stage, "status": status, "data": data}
    try:
        path = event_path(exp, root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return rec


def read_events(exp, root="."):
    """All events of an experiment (oldest first). Malformed lines are skipped."""
    path = event_path(exp, root)
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out
