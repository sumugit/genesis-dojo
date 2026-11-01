"""Dashboard state of one experiment, built from stage events (logs/events) and loop records (logs/loops).

Pure stdlib (+ PyYAML for the task specs), no torch, so it is unit-tested without a GPU.

State shape (all times are epoch seconds; for reconstructed runs they start at 0):
    {"exp", "instruction", "task", "status": "running|done|stalled|unknown", "synthetic": bool,
     "t0", "t_end", "server_time", "criteria": [...], "window_s", "max_loops",
     "stop_reason", "current": {"loop", "stage"} | None,
     "loops": [{"loop", "name", "stages": {stage: {"start","end","status"}}, "reward_summary", "reward_code",
                "check_attempts": [...], "train": {"seconds","final_mean_reward","iterations","iteration","curve"},
                "video", "numeric", "numeric_label", "evaluation", "decision", "error"}]}
Nothing in the state contains absolute paths or account identifiers.
"""

from __future__ import annotations

import glob
import json
import os
import re
import time

from pai_loop import events as ev

EXP_RE = re.compile(r"^[A-Za-z0-9_]{1,60}$")
VIDEO_RE = re.compile(r"^[A-Za-z0-9_.-]+\.mp4$")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_ITER = re.compile(r"Learning iteration (\d+)/(\d+)")
_REWARD = re.compile(r"Mean reward:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
STALL_AFTER_S = 1200  # no event for this long and no "run end": the loop is probably dead


def parse_train_log(path):
    """Parse an rsl_rl training log. Returns {"iteration", "total", "curve": [[iteration, mean_reward], ...]}."""
    out = {"iteration": 0, "total": 0, "curve": []}
    if not path or not os.path.exists(path):
        return out
    cur = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = _ANSI.sub("", line)
            m = _ITER.search(line)
            if m:
                cur = int(m.group(1))
                out["iteration"], out["total"] = cur, int(m.group(2))
                continue
            m = _REWARD.search(line)
            if m and cur is not None:
                out["curve"].append([cur, float(m.group(1))])
    return out


def _read_records(root, exp):
    path = os.path.join(root, "logs", "loops", f"{exp}.jsonl")
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def list_experiments(root):
    """Experiments with an event log or a loop record, newest first."""
    found = {}
    for sub, kind in (("events", "events"), ("loops", "records")):
        for p in glob.glob(os.path.join(root, "logs", sub, "*.jsonl")):
            exp = os.path.basename(p)[: -len(".jsonl")]
            if not EXP_RE.match(exp):
                continue
            d = found.setdefault(exp, {"exp": exp, "mtime": 0.0, "sources": []})
            d["mtime"] = max(d["mtime"], os.path.getmtime(p))
            d["sources"].append(kind)
    out = []
    for d in found.values():
        instruction, loops = "", 0
        recs = _read_records(root, d["exp"])
        if recs:
            instruction, loops = recs[0].get("instruction", ""), len(recs)
        for e in ev.read_events(d["exp"], root):
            if e["stage"] == "instruction":
                instruction = e["data"].get("instruction", instruction)
                break
        out.append({"exp": d["exp"], "instruction": instruction, "loops": loops, "mtime": d["mtime"],
                    "live_events": "events" in d["sources"]})
    return sorted(out, key=lambda x: -x["mtime"])


def _decision_from_vlm(evaluation, threshold=8):
    """Runs from before the numeric success check: the VLM verdict was the only criterion."""
    try:
        score = float(evaluation.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    ok = bool(evaluation.get("success")) and score >= threshold
    return {"numeric_pass": None, "vlm_pass": ok, "success": ok, "disagreement": None, "legacy": True}


def _new_loop(i, exp):
    return {"loop": i, "name": f"{exp}_L{i}", "stages": {}, "reward_summary": None, "reward_code": None,
            "check_attempts": [], "train": None, "video": None, "numeric": None, "numeric_label": None,
            "evaluation": None, "decision": None, "error": None}


def _attach_train(root, loop):
    log = parse_train_log(os.path.join(root, "logs", f"{loop['name']}_train.log"))
    tr = loop["train"] or {}
    tr.update(iteration=log["iteration"], total=log["total"] or tr.get("iterations") or 0, curve=log["curve"])
    loop["train"] = tr


def _from_events(root, exp, evs, records):
    state = {"exp": exp, "synthetic": False, "criteria": [], "window_s": None, "instruction": "", "task": "",
             "max_loops": None, "stop_reason": None, "run_summary": None}
    loops = {}
    for e in evs:
        d = e.get("data") or {}
        stage, status, i = e.get("stage"), e.get("status"), e.get("loop")
        if stage == "instruction":
            state.update(instruction=d.get("instruction", ""), task=d.get("task", ""), max_loops=d.get("max_loops"),
                         criteria=d.get("criteria", []), window_s=d.get("window_s"))
            state["spec_sha256"] = (d.get("spec_sha256") or "")[:12]
            state["hint"], state["prev_exp"] = d.get("hint"), d.get("prev_exp")
            continue
        if stage == "run":
            state.update(stop_reason=d.get("stop_reason"), run_summary=d)
            state["t_run_end"] = e["t"]
            continue
        if i is None:
            continue
        lp = loops.setdefault(i, _new_loop(i, exp))
        if stage == "error":
            lp["error"] = d.get("message")
            continue
        st = lp["stages"].setdefault(stage, {"start": None, "end": None, "status": "pending"})
        if status == "start":
            st["start"], st["status"] = e["t"], "running"
            if stage == "check":
                st["end"] = None
        elif status == "end":
            st["end"], st["status"] = e["t"], "done"
            if st["start"] is None:
                st["start"] = e["t"]
        elif status == "error":
            st["end"], st["status"] = e["t"], "error"
        if stage == "reward" and status == "end":
            lp["reward_summary"], lp["reward_code"] = d.get("summary"), d.get("code")
        elif stage == "check":
            lp["check_attempts"].append({"attempt": d.get("attempt"), "ok": status == "end",
                                         "message": d.get("message")})
        elif stage == "train" and status == "end":
            lp["train"] = {"seconds": d.get("seconds"), "final_mean_reward": d.get("final_mean_reward"),
                           "iterations": state.get("train_iterations")}
        elif stage == "train" and status == "start":
            lp["train"] = {"iterations": d.get("iterations")}
        elif stage == "record" and status == "end":
            lp["video"] = d.get("video")
        elif stage == "numeric" and status == "end":
            lp["numeric"], lp["numeric_label"] = d.get("result"), "live"
        elif stage == "vlm" and status == "end":
            lp["evaluation"] = d.get("evaluation")
        elif stage == "decision" and status == "end":
            lp["decision"] = d.get("decision")
    state["loops"] = [loops[k] for k in sorted(loops)]
    for lp in state["loops"]:
        if lp["train"] is not None:
            _attach_train(root, lp)
    state["t0"] = evs[0]["t"] if evs else 0.0
    last = evs[-1]["t"] if evs else 0.0
    state["t_end"] = state.get("t_run_end") or last
    state["last_event"] = last
    return state


def _from_records(root, exp, records):
    """Reconstruct a timeline for runs that have no stage events (older runs), from the recorded durations."""
    state = {"exp": exp, "synthetic": True, "instruction": records[0].get("instruction", ""), "task": "",
             "criteria": [], "window_s": None, "max_loops": len(records), "stop_reason": None, "run_summary": None}
    t = 0.0
    loops = []
    for rec in records:
        i = rec.get("loop", len(loops) + 1)
        lp = _new_loop(i, exp)
        lp["name"] = rec.get("exp_name", lp["name"])
        calls = rec.get("llm_calls", [])
        reward_s = sum(c["seconds"] for c in calls if c.get("kind") == "reward")
        vlm_s = sum(c["seconds"] for c in calls if c.get("kind") == "evaluate")
        tr = rec.get("train") or {}
        train_s = float(tr.get("train_seconds") or 0.0)
        record_s = float((rec.get("metrics") or {}).get("record_wall_seconds") or 0.0)
        known = reward_s + vlm_s + train_s + record_s + 1.0
        overhead = max(0.0, float(rec.get("loop_seconds") or known) - known)  # start-up of the training/recording processes
        spans = [("reward", reward_s), ("check", 0.5), ("train", train_s + overhead / 2),
                 ("record", record_s + overhead / 2), ("numeric", 0.3), ("vlm", vlm_s), ("decision", 0.2)]
        if not rec.get("train"):
            spans = [s for s in spans if s[0] in ("reward", "check")]
        for stage, dur in spans:
            lp["stages"][stage] = {"start": round(t, 2), "end": round(t + dur, 2), "status": "done"}
            t += dur
        lp.update(reward_summary=rec.get("reward_summary"), reward_code=rec.get("reward_code"),
                  evaluation=rec.get("evaluation"), error=rec.get("error"))
        lp["train"] = {"seconds": tr.get("train_seconds"), "final_mean_reward": tr.get("final_mean_reward"),
                       "iterations": tr.get("iterations")} if tr else None
        if lp["train"] is not None:
            _attach_train(root, lp)
        ckpt = tr.get("last_checkpoint")
        if ckpt is not None:
            vid = f"{lp['name']}_{ckpt}.mp4"
            lp["video"] = vid if os.path.exists(os.path.join(root, "videos", vid)) else None
        check = rec.get("success_check")
        label = "live" if check else None
        if not check and ckpt is not None:
            check = _read_json(os.path.join(root, "runs", lp["name"], f"success_{ckpt}.json"))
            label = "retro" if check else None
        lp["numeric"], lp["numeric_label"] = check, label
        if rec.get("decision"):
            lp["decision"] = rec["decision"]
        elif rec.get("evaluation"):
            lp["decision"] = _decision_from_vlm(rec["evaluation"])
        if check and not state["task"]:
            state["task"] = check.get("task", "")
        loops.append(lp)
        if check and not state["criteria"]:
            state["criteria"] = [{"metric": c["metric"], "rule": c["rule"], "min_fraction": c.get("min_fraction"),
                                  "comment": ""} for c in check["criteria"]]
            state["window_s"] = check.get("window_s")
    state["loops"] = loops
    state["t0"], state["t_end"] = 0.0, round(t, 2)
    state["last_event"] = state["t_end"]
    last = records[-1]
    if (last.get("decision") or {}).get("success") or (not last.get("decision") and
                                                       _decision_from_vlm(last.get("evaluation") or {})["success"]):
        state["stop_reason"] = "success"
    return state


def _finish(state, now):
    loops = state["loops"]
    live = not state["synthetic"]
    if live:
        if state.get("stop_reason") == "crashed":
            state["status"] = "failed"
        elif state.get("stop_reason") == "stopped":   # the person stopped it from the dashboard
            state["status"] = "stopped"
        elif state.get("stop_reason"):
            state["status"] = "done"
        else:
            state["status"] = "running" if now - state["last_event"] < STALL_AFTER_S else "stalled"
    else:
        state["status"] = "done"
    if state["status"] in ("failed", "stopped"):   # stages that never ended did not complete
        for lp in loops:
            for st in lp["stages"].values():
                if st.get("end") is None and st.get("start") is not None:
                    st["end"], st["status"] = state.get("t_run_end") or now, "aborted"
    cur = None
    if state["status"] in ("running", "stalled"):
        for lp in loops:
            for stage, st in lp["stages"].items():
                if st["status"] == "running":
                    cur = {"loop": lp["loop"], "stage": stage}
    state["current"] = cur
    state["server_time"] = now
    return state


def build_state(root, exp, now=None):
    """State of `exp`, or None if there is no record of it."""
    if not EXP_RE.match(exp):
        return None
    now = time.time() if now is None else now
    evs = ev.read_events(exp, root)
    records = _read_records(root, exp)
    if evs:
        state = _from_events(root, exp, evs, records)
    elif records:
        state = _from_records(root, exp, records)
    else:
        return None
    return _finish(state, now)


def list_tasks(root):
    """Success specs under configs/tasks, with their approval state (for the 'run' form)."""
    from pai_loop import success

    out = []
    for p in sorted(glob.glob(os.path.join(root, "configs", "tasks", "*.success.yaml"))):
        try:
            spec, _, sha = success.load_spec(p, require_approved=False)
        except (success.SpecError, OSError):
            continue
        out.append({"task": spec["task"], "approved": spec["approved"], "sha256": sha[:12],
                    "criteria": len(spec["criteria"]), "file": os.path.basename(p)})
    return out


_METRICS_FILE = re.compile(r"^metrics_(\d+)(_[A-Za-z0-9]+)?\.json$")


def load_trace(root, exp, loop):
    """Time series of the recorded run of one loop (runs/<exp>_L<loop>/metrics_<ckpt>[_<tag>].json).

    {"dt", "joint_names" | None, "series": {name: [...]}}; None when the run has no metrics file.
    The latest checkpoint wins. A re-recording (`record.py --tag`) replays the same policy, so it is used when
    it carries joint angles and the original does not (runs recorded before joint angles were saved).
    """
    if not EXP_RE.match(exp) or not isinstance(loop, int) or not 0 < loop < 1000:
        return None
    run_dir = os.path.join(root, "runs", f"{exp}_L{loop}")
    files = sorted(((int(m.group(1)), m.group(2) is not None, f) for f in
                    (os.listdir(run_dir) if os.path.isdir(run_dir) else []) if (m := _METRICS_FILE.match(f))),
                   key=lambda x: (-x[0], x[1], x[2]))   # latest checkpoint first, the original before its re-recordings
    best = None
    for _, _, f in files:
        m = _read_json(os.path.join(run_dir, f)) or {}
        series = m.get("series_0p1s") or {}
        has_q = any(k.startswith("q_") for k in series)
        cand = {"dt": 0.1, "joint_names": m.get("joint_names") if has_q else None, "series": series}
        if has_q:
            return cand
        best = best or cand
    return best


_OVERLAY_FILE = re.compile(r"^overlay_(\d+)(_[A-Za-z0-9]+)?\.json$")


def load_overlay(root, exp, loop):
    """Joint positions per step of the recorded video of one loop (written by scripts/record.py), or None.

    The latest checkpoint wins, the untagged recording before a tagged re-recording. `video` names the video that the
    positions belong to: a re-recording has its own file.
    """
    if not EXP_RE.match(exp) or not isinstance(loop, int) or not 0 < loop < 1000:
        return None
    run_dir = os.path.join(root, "runs", f"{exp}_L{loop}")
    files = sorted(((int(m.group(1)), m.group(2) is not None, f) for f in
                    (os.listdir(run_dir) if os.path.isdir(run_dir) else []) if (m := _OVERLAY_FILE.match(f))),
                   key=lambda x: (-x[0], x[1], x[2]))
    return _read_json(os.path.join(run_dir, files[0][2])) if files else None
