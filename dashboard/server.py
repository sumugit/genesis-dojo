"""Local dashboard server (stdlib only): `python -m dashboard.server --root . --port 8765`.

Binds to 127.0.0.1 only. Reach it from another machine through an SSH tunnel.
Endpoints: /api/experiments, /api/state?exp=, /api/trace?exp=&loop=, /api/tasks, /api/queue, /api/config, POST /api/run, /video/<name>.mp4.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from dashboard import authoring as au
from dashboard import improve as im
from dashboard import state as st
from dashboard import tasks as tk
from pai_loop import events as ev

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MIME = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8", ".mp4": "video/mp4", ".json": "application/json"}
# Source files shown when a stage of the flow is clicked (a fixed list: nothing a client sends is used as a path)
STAGE_SOURCES = {
    "instruction": ["scripts/loop.py"],
    "reward": ["pai_loop/prompts.py", "pai_loop/llm.py"],
    "check": ["pai_loop/reward_check.py"],
    "train": ["scripts/train.py", "pai_loop/task_env.py"],
    "record": ["scripts/record.py", "pai_loop/metrics.py"],
    "numeric": ["{spec}", "pai_loop/success.py"],
    "vlm": ["pai_loop/prompts.py"],
    "decision": ["scripts/loop.py"],
}
MAX_INSTRUCTION = 200
MAX_BODY = 4096


def finite(o):
    """The same structure with NaN / Infinity replaced by None (they are not valid JSON; a browser cannot parse them)."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {k: finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [finite(v) for v in o]
    return o


class RunQueue:
    """Runs loop commands one after another (one GPU). `command_for` builds the argv from a request."""

    def __init__(self, root, command_for):
        self.root, self.command_for = root, command_for
        self.items, self.lock = [], threading.Lock()
        self.procs = {}
        self.wake = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    def add(self, exp, instruction, task, hint=None, prev_exp=None):
        with self.lock:
            if any(i["exp"] == exp for i in self.items):
                return None
            item = {"exp": exp, "instruction": instruction, "task": task, "status": "queued", "returncode": None,
                    "hint": hint, "prev_exp": prev_exp}
            self.items.append(item)
        self.wake.set()
        return item

    def can_stop(self, exp):
        with self.lock:
            return any(i["exp"] == exp and i["status"] in ("queued", "running") and not i.get("stopping") for i in self.items)

    def stop(self, exp):
        """Cancel a queued run or terminate a running one (the whole process group: loop.py and its children).
        Returns 'cancelled', 'stopping', or None when this server has no such live run."""
        with self.lock:
            item = next((i for i in self.items if i["exp"] == exp), None)
            if item is None:
                return None
            if item["status"] == "queued":
                item["status"] = "cancelled"
                return "cancelled"
            proc = self.procs.get(exp)
            if item["status"] != "running" or proc is None or item.get("stopping"):
                return None
            item["stopping"] = True
        self._signal(proc, signal.SIGTERM)
        threading.Timer(8.0, lambda: proc.poll() is None and self._signal(proc, signal.SIGKILL)).start()
        return "stopping"

    @staticmethod
    def _signal(proc, sig):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def snapshot(self):
        with self.lock:
            return [dict(i) for i in self.items]

    def _worker(self):
        while True:
            self.wake.wait()
            with self.lock:
                todo = next((i for i in self.items if i["status"] == "queued"), None)
                if todo is None:
                    self.wake.clear()
                    continue
                todo["status"] = "running"
            try:
                log = open(os.path.join(self.root, "logs", f"{todo['exp']}_dashboard.log"), "ab")
                proc = subprocess.Popen(self.command_for(todo), cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)   # its own process group, so that a stop reaches the children
                with self.lock:
                    self.procs[todo["exp"]] = proc
                rc = proc.wait()
            except OSError:
                rc = -1
            stopped = bool(todo.get("stopping"))
            if stopped:
                ev.emit(todo["exp"], "run", "end", root=self.root, stop_reason="stopped", returncode=rc)
            elif rc != 0:  # loop.py died without writing its closing event
                ev.emit(todo["exp"], "run", "end", root=self.root, stop_reason="crashed", returncode=rc)
            with self.lock:
                self.procs.pop(todo["exp"], None)
                todo["status"], todo["returncode"] = ("stopped" if stopped else "done" if rc == 0 else "failed"), rc


def pick_base(root, task):
    """Base config for a task: configs/base_<task>.yaml, else the same without a _vN suffix, else the first word
    (handstand_hold -> base_handstand.yaml), else configs/base.yaml."""
    entry = next((c for c in tk.load_catalog(root) if c["task"] == task and c.get("base")), None)
    if entry and os.path.exists(os.path.join(root, entry["base"])):   # a task created on the dashboard names its base
        return entry["base"]
    t = re.sub(r"_v\d+$", "", task)
    for name in (t, t.split("_")[0]):
        path = os.path.join("configs", f"base_{name}.yaml")
        if os.path.exists(os.path.join(root, path)):
            return path
    return os.path.join("configs", "base.yaml")


def default_command(root, python=sys.executable):
    def build(item):
        spec = os.path.join("configs", "tasks", f"{item['task']}.success.yaml")
        cmd = [python, "scripts/loop.py", "--exp", item["exp"], "--instruction", item["instruction"],
               "--success-spec", spec, "--base-config", pick_base(root, item["task"])]
        if item.get("hint"):
            cmd += ["--hint", item["hint"]]
        if item.get("prev_exp"):
            cmd += ["--prev-exp", item["prev_exp"]]
        return cmd
    return build


def make_handler(root, queue, chooser=None, drafter=None):
    root = os.path.abspath(root)

    class Handler(BaseHTTPRequestHandler):
        server_version = "pai-dashboard"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype="application/json", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(finite(body), ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/api/experiments":
                return self._send(200, st.list_experiments(root))
            if u.path == "/api/state":
                s = st.build_state(root, (q.get("exp") or [""])[0])
                if not s:
                    return self._send(404, {"error": "not found"})
                s["can_stop"] = bool(queue and queue.can_stop(s["exp"]))
                s["hint_info"] = next(({"name": h["name"]} for h in tk.load_hints(root) if s.get("hint") and s["hint"].endswith(h["file"])), {"name": "この芸専用のヒント"} if s.get("hint") else None)
                s["task_info"] = next((c for c in tk.load_catalog(root) if c["task"] == s.get("task")), None)
                return self._send(200, s)
            if u.path == "/api/source":
                return self._source((q.get("stage") or [""])[0], (q.get("exp") or [""])[0])
            if u.path == "/api/trace":
                try:
                    n = int((q.get("loop") or ["0"])[0])
                except ValueError:
                    n = 0
                t = st.load_trace(root, (q.get("exp") or [""])[0], n)
                return self._send(200, t) if t else self._send(404, {"error": "not found"})
            if u.path == "/api/bases":
                return self._send(200, au.list_bases(root))
            if u.path == "/api/metrics":
                return self._send(200, au.metric_list())
            if u.path == "/api/overlay":
                try:
                    n = int((q.get("loop") or ["0"])[0])
                except ValueError:
                    n = 0
                o = st.load_overlay(root, (q.get("exp") or [""])[0], n)
                return self._send(200, o) if o else self._send(404, {"error": "not found"})
            if u.path == "/api/catalog":
                return self._send(200, tk.load_catalog(root))
            if u.path == "/api/tasks":
                return self._send(200, st.list_tasks(root))
            if u.path == "/api/queue":
                return self._send(200, queue.snapshot() if queue else [])
            if u.path == "/api/config":
                return self._send(200, {"can_run": queue is not None, "stages": ev.STAGES, "labels": ev.STAGE_LABELS})
            if u.path.startswith("/video/"):
                return self._video(u.path[len("/video/"):])
            if u.path in ("/", "/index.html"):
                return self._static("index.html")
            if u.path in ("/app.js", "/style.css"):
                return self._static(u.path[1:])
            self._send(404, {"error": "not found"})

        def _source(self, stage, exp):
            files = STAGE_SOURCES.get(stage)
            if not files:
                return self._send(404, {"error": "not found"})
            s = st.build_state(root, exp) if exp else None
            out = []
            for rel in files:
                if rel == "{spec}":
                    if not s or not s.get("task"):
                        continue
                    rel = f"configs/tasks/{s['task']}.success.yaml"
                path = os.path.join(root, rel)
                if os.path.isfile(path) and re.match(r"^[A-Za-z0-9_./-]+$", rel) and ".." not in rel:
                    with open(path, encoding="utf-8") as f:
                        out.append({"path": rel, "text": f.read()})
            self._send(200, out)

        def _static(self, name):
            with open(os.path.join(STATIC_DIR, name), "rb") as f:
                self._send(200, f.read(), MIME[os.path.splitext(name)[1]])

        def _video(self, name):
            if not st.VIDEO_RE.match(name):
                return self._send(404, {"error": "not found"})
            path = os.path.join(root, "videos", name)
            if not os.path.isfile(path):
                return self._send(404, {"error": "not found"})
            size = os.path.getsize(path)
            start, end, code = 0, size - 1, 200
            m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end or start >= size:
                    return self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
                code = 206
            with open(path, "rb") as f:
                f.seek(start)
                body = f.read(end - start + 1)
            extra = {"Accept-Ranges": "bytes"}
            if code == 206:
                extra["Content-Range"] = f"bytes {start}-{end}/{size}"
            self._send(code, body, "video/mp4", extra)

        def _guard(self):
            """Same origin only, and only for a loopback host (an SSH tunnel may use another local port than the server)."""
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin", "")
            loopback = host.rsplit(":", 1)[0] in ("127.0.0.1", "localhost")
            if not (loopback and origin == f"http://{host}") or self.headers.get("X-Requested-With") != "dashboard":
                self._send(403, {"error": "forbidden"})
                return False
            if queue is None:
                self._send(409, {"error": "running is disabled on this server"})
                return False
            return True

        def _body(self):
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > MAX_BODY:
                    self._send(413, {"error": "too large"})
                    return None
                req = json.loads(self.rfile.read(n) or b"{}")
                if isinstance(req, dict):
                    return req
            except ValueError:
                pass
            self._send(400, {"error": "bad json"})
            return None

        def _instruction(self, req):
            ins = str(req.get("instruction", "")).strip()
            if not ins or len(ins) > MAX_INSTRUCTION:
                self._send(400, {"error": f"instruction must be 1-{MAX_INSTRUCTION} characters"})
                return None
            return ins

        def _library_hint(self, ins, task):
            """Path of the gait hint to pass to the run: none when the task has its own hint file; else the one that fits."""
            if os.path.exists(os.path.join(root, "configs", "tasks", f"{task}.hint.md")):
                return None
            try:
                h = tk.choose_hint(root, ins, chooser or tk.default_ask())
            except Exception:   # noqa: BLE001 - a missing hint must not stop a run
                return None
            return h["file"] if h else None

        def _prev_exp(self, req):
            p = req.get("prev_exp")
            if isinstance(p, str) and st.EXP_RE.match(p) and os.path.exists(os.path.join(root, "logs", "loops", f"{p}.jsonl")):
                return p
            return None

        def _exp_free(self, exp):
            if not isinstance(exp, str) or not st.EXP_RE.match(exp):
                self._send(400, {"error": "exp must match [A-Za-z0-9_]{1,60}"})
                return False
            if os.path.exists(ev.event_path(exp, root)) or os.path.exists(os.path.join(root, "logs", "loops", f"{exp}.jsonl")):
                self._send(409, {"error": "exp already exists"})
                return False
            return True

        def do_POST(self):
            u = urlparse(self.path)
            if u.path not in ("/api/run", "/api/draft", "/api/approve", "/api/stop", "/api/improve", "/api/preview"):
                return self._send(404, {"error": "not found"})
            if not self._guard():
                return
            req = self._body()
            if req is None:
                return
            if u.path == "/api/stop":
                exp = req.get("exp", "")
                result = queue.stop(exp) if isinstance(exp, str) and st.EXP_RE.match(exp) else None
                if result is None:
                    return self._send(404, {"error": "このダッシュボードから始めた、実行中の学習が見つかりません"})
                return self._send(200, {"result": result})
            if u.path in ("/api/improve", "/api/preview"):
                return self._improve(u.path, req)
            ins = self._instruction(req)
            if ins is None:
                return
            if u.path == "/api/draft":
                return self._draft(ins, bool(req.get("force_new")))
            if u.path == "/api/approve":
                return self._approve(ins, req)
            exp, task = req.get("exp", ""), req.get("task", "")
            approved = {t["task"] for t in st.list_tasks(root) if t["approved"]}
            if not self._exp_free(exp):
                return
            if not task:   # an LLM picks the approved success spec for the instruction
                try:
                    task, _ = tk.choose_task(root, ins, chooser or tk.default_ask())
                except Exception as e:   # noqa: BLE001 - the LLM backend can fail in many ways
                    return self._send(503, {"error": f"指示から芸を選べませんでした（{type(e).__name__}）"})
                if task is None:
                    return self._send(422, {"error": "この指示に合う、承認済みの成功条件がありません"})
            elif task not in approved:
                return self._send(400, {"error": "task has no approved success spec"})
            item = queue.add(exp, ins, task, hint=self._library_hint(ins, task), prev_exp=self._prev_exp(req))
            if item is None:
                return self._send(409, {"error": "already queued"})
            self._send(202, item)

        def _draft(self, ins, force_new):
            """Match the instruction to an approved task, else draft a new success spec (nothing is written)."""
            try:
                if not force_new:
                    task, cat = tk.choose_task(root, ins, chooser or tk.default_ask())
                    if task:
                        c = next(x for x in cat if x["task"] == task)
                        return self._send(200, {"reuse": {"task": task, "name": c["name"], "description": c["description"], "criteria": c["criteria"]}})
                existing = {t["task"] for t in st.list_tasks(root)}
                try:
                    hint = tk.choose_hint(root, ins, chooser or tk.default_ask())
                except Exception:   # noqa: BLE001 - a hint is optional
                    hint = None
                text = (drafter or au.default_draft_ask())(au.draft_prompt(root, ins, sorted(existing), hint))
            except Exception as e:   # noqa: BLE001
                return self._send(503, {"error": f"成功条件の草案を作れませんでした（{type(e).__name__}）"})
            d = au.parse_draft(text)
            if d is None:
                return self._send(502, {"error": "AIの返答を読み取れませんでした。もう一度お試しください"})
            if d.get("cannot_measure") or not d.get("criteria"):
                return self._send(200, {"cannot_measure": str(d.get("cannot_measure") or "測れる指標で表せませんでした")})
            tid = str(d.get("task_id") or "")
            base_id, n = tid, 2
            while tid in existing:   # keep the name unique
                tid, n = f"{base_id}_{n}", n + 1
            d["task_id"] = tid
            hint_info = {"id": hint["id"], "name": hint["name"], "applies": hint["applies"], "text": tk.hint_text(root, hint)} if hint else None
            self._send(200, {"draft": {k: d.get(k) for k in ("task_id", "name", "description", "base", "window_s", "criteria", "rationale")},
                             "errors": au.check_draft(root, d, existing), "hint": hint_info})

        def _improve(self, path, req):
            """Propose a revised success spec for a finished experiment (nothing is written), or preview a draft on its loops."""
            exp = req.get("exp", "")
            s = st.build_state(root, exp) if isinstance(exp, str) else None
            if not s or not s.get("loops") or s["status"] in ("running", "stalled"):
                return self._send(404, {"error": "終わった実験が見つかりません"})
            old, raw = im.load_spec_dict(root, s["task"])
            if old is None:
                return self._send(404, {"error": "この実験の成功条件のファイルが見つかりません"})
            if path == "/api/preview":
                d = req.get("draft")
                if not isinstance(d, dict):
                    return self._send(400, {"error": "draft missing"})
                try:
                    return self._send(200, {"preview": im.preview(root, s, d)})
                except Exception as e:   # noqa: BLE001 - an invalid draft
                    return self._send(422, {"errors": [str(e).split("\n")[0]]})
            try:
                text = (drafter or au.default_draft_ask())(im.improve_prompt(s, raw, im.loop_summaries(root, s)))
            except Exception as e:   # noqa: BLE001
                return self._send(503, {"error": f"見直し案を作れませんでした（{type(e).__name__}）"})
            d = au.parse_draft(text)
            if d is None or not d.get("criteria"):
                return self._send(502, {"error": "AIの返答を読み取れませんでした。もう一度お試しください"})
            existing = {t["task"] for t in st.list_tasks(root)}
            info = next((c for c in tk.load_catalog(root) if c["task"] == s["task"]), None) or {}
            draft = {"task_id": im.next_task_id(root, s["task"]), "name": (info.get("name") or s["task"]) + "（見直し）",
                     "description": info.get("description", ""), "base": pick_base(root, s["task"]), "window_s": d.get("window_s"),
                     "criteria": d.get("criteria"), "rationale": d.get("rationale"), "hint": d.get("hint") or ""}
            errors = au.check_draft(root, draft, existing)
            try:
                rows = im.preview(root, s, draft) if not errors else []
            except Exception:   # noqa: BLE001
                rows = []
            self._send(200, {"draft": draft, "diagnosis": d.get("diagnosis"), "errors": errors, "preview": rows,
                             "old": old.get("criteria"), "old_window_s": old.get("window_s")})

        def _approve(self, ins, req):
            """The person's approval: write the approved spec (rebuilt from the fields) and, unless told not to, queue the run."""
            exp, d, start = req.get("exp", ""), req.get("draft"), req.get("start", True) is not False
            if not isinstance(d, dict):
                return self._send(400, {"error": "draft missing"})
            if start and not self._exp_free(exp):
                return
            existing = {t["task"] for t in st.list_tasks(root)}
            try:
                task, sha = au.approve(root, d, ins, existing)
            except ValueError as e:
                return self._send(422, {"errors": e.args[0]})
            except FileExistsError:
                return self._send(409, {"error": "task already exists"})
            typed = str(req.get("hint") or "").strip()[:4000]
            lib = next((h for h in tk.load_hints(root) if h["id"] == req.get("hint_id")), None)
            if lib is None and req.get("hint_id") is None:   # not told: pick the one that fits the instruction
                try:
                    lib = tk.choose_hint(root, ins, chooser or tk.default_ask())
                except Exception:   # noqa: BLE001
                    lib = None
            parts = ([f"（設計ヒント「{lib['name']}」。人が事前に書いた助言）\n" + tk.hint_text(root, lib)] if lib else []) + \
                    ([f"（この実験の見直しで加えた助言）\n" + typed] if typed else [])
            if parts:   # advice for the reward writer in the next run (the AI scorer never sees it); loop.py picks it up by the task name
                with open(os.path.join(root, "configs", "tasks", f"{task}.hint.md"), "w", encoding="utf-8") as f:
                    f.write("\n\n".join(parts) + "\n")
            if not start:
                return self._send(200, {"task": task, "sha256": sha[:12]})
            item = queue.add(exp, ins, task, prev_exp=self._prev_exp(req))
            if item is None:
                return self._send(409, {"error": "already queued"})
            self._send(202, {**item, "sha256": sha[:12]})

    return Handler


def serve(root, port, queue=None, host="127.0.0.1", chooser=None, drafter=None):
    httpd = ThreadingHTTPServer((host, port), None)
    httpd.RequestHandlerClass = make_handler(root, queue, chooser, drafter)
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-run", action="store_true", help="read-only: do not accept new runs")
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    queue = None if a.no_run else RunQueue(root, default_command(root))
    httpd = serve(root, a.port, queue)
    print(f"dashboard: http://127.0.0.1:{httpd.server_address[1]}  (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
