"""Export recorded experiments as a static site: the public demo, which can only replay what was already learned.

    python scripts/export_static.py --root . --out demo --exps <exp1>,<exp2>

The site is the same screen as the Web UI (dashboard/static) over exported files (no server, no LLM, no GPU), so that
it can be hosted for free (GitHub Pages, Vercel). Nothing can be started from it. Stdlib only (ffmpeg is used when present).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard import server as srv  # noqa: E402
from dashboard import state as st  # noqa: E402
from dashboard import tasks as tk  # noqa: E402


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(srv.finite(obj), f, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def copy_video(root, out, name):
    src = os.path.join(root, "videos", name)
    if not (st.VIDEO_RE.match(name) and os.path.isfile(src)):
        return False
    dst = os.path.join(out, "data", "video", name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if shutil.which("ffmpeg"):   # 720p, H.264, no audio: a few hundred KB per loop
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", "scale=-2:720", "-c:v", "libx264", "-crf", "28",
               "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst]
        if subprocess.call(cmd) == 0:
            return True
    shutil.copy(src, dst)
    return True


def label_of(s):
    """What the card says: the instruction, the number of loops and how the run ended (a revision and a re-run differ in the 2nd part)."""
    n = len(s["loops"])
    result = "成功" if s.get("stop_reason") == "success" else "成功せず"
    note = "、成功条件を見直した版" if (s.get("task") or "").rsplit("_v", 1)[-1].isdigit() else ""
    return f"{s.get('instruction', '')}（{n}周、{result}{note}）"


def export(root, out, exps):
    catalog = tk.load_catalog(root)
    listing = {x["exp"]: x for x in st.list_experiments(root)}
    done, videos = [], 0
    for exp in exps:
        s = st.build_state(root, exp)
        if not s or exp not in listing:
            print(f"skip {exp}: no record")
            continue
        s["can_stop"] = False
        s["task_info"] = next((c for c in catalog if c["task"] == s.get("task")), None)
        s["hint_info"] = ({"name": "この芸専用のヒント"} if s.get("hint") else None)
        spec = f"configs/tasks/{s['task']}.success.yaml" if s.get("task") else None
        for stage, files in srv.STAGE_SOURCES.items():
            rows = []
            for rel in files:
                rel = spec if rel == "{spec}" else rel
                if rel and os.path.isfile(os.path.join(root, rel)):
                    with open(os.path.join(root, rel), encoding="utf-8") as f:
                        rows.append({"path": rel, "text": f.read()})
            write_json(os.path.join(out, "data", "source", f"{stage}__{exp}.json"), rows)
        for lp in s["loops"]:
            n = lp["loop"]
            trace = st.load_trace(root, exp, n)
            if trace is not None:
                write_json(os.path.join(out, "data", "trace", f"{exp}_{n}.json"), trace)
            overlay = st.load_overlay(root, exp, n)
            if overlay is not None:
                write_json(os.path.join(out, "data", "overlay", f"{exp}_{n}.json"), overlay)
                if overlay.get("video"):   # the video that the overlay's positions belong to (a re-recording has its own)
                    lp["video"] = overlay["video"]
            if lp.get("video"):
                videos += copy_video(root, out, lp["video"])
        write_json(os.path.join(out, "data", "state", f"{exp}.json"), s)
        item = {k: listing[exp][k] for k in ("exp", "instruction", "loops")}
        item["label"] = label_of(s)
        done.append(item)
    write_json(os.path.join(out, "data", "experiments.json"), done)
    write_json(os.path.join(out, "data", "catalog.json"), catalog)
    # the page: the Web UI's own files, with relative paths and the static-site flag
    static = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "static")
    for name in ("app.js", "style.css"):
        shutil.copy(os.path.join(static, name), os.path.join(out, name))
    with open(os.path.join(static, "index.html"), encoding="utf-8") as f:
        html = f.read()
    html = (html.replace('href="/style.css"', 'href="style.css"')
            .replace('<script src="/app.js"></script>', '<script>window.STATIC_SITE = true;</script>\n<script src="app.js"></script>'))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    return done, videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="demo")
    ap.add_argument("--exps", required=True, help="comma separated experiment names, in the order of the list")
    a = ap.parse_args()
    done, videos = export(os.path.abspath(a.root), a.out, [e for e in a.exps.split(",") if e])
    print(f"exported {len(done)} experiments, {videos} videos -> {a.out}")


if __name__ == "__main__":
    main()
