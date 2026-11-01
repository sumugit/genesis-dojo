import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from dashboard import server as srv
from dashboard import state as st
from pai_loop import events as ev

SPEC = """task: spin
approved: true
window_s: [3.0, 6.0]
criteria:
  - {metric: height_m, between: [0.25, 0.35], min_fraction: 0.9}
"""


CATALOG = """tasks:
  - {task: spin, name: その場で回る, description: 水平に回転する, examples: [回って]}
  - {task: draft, name: 下書き, description: 未承認, examples: [x]}
  - {task: ghost, name: 仕様なし, description: 成功条件のファイルが無い, examples: [y]}
"""


HINTS = """hints:
  - {id: walk_quadruped, name: 四足歩行, applies: 四足で歩く, file: configs/hints/walk_quadruped.hint.md, base: configs/base.yaml, spec_notes: "前進距離と足の交互の持ち上げを入れる"}
  - {id: gone, name: 消えたヒント, applies: x, file: configs/hints/gone.hint.md}
  - {id: evil, name: 外へ出るヒント, applies: x, file: configs/../secret.txt}
"""


def write(root, rel, text, mode="w"):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode) as f:
        f.write(text)


class EventsTest(unittest.TestCase):
    def test_roundtrip_and_malformed(self):
        with tempfile.TemporaryDirectory() as d:
            ev.emit("e1", "reward", "start", loop=1, root=d)
            ev.emit("e1", "reward", "end", loop=1, root=d, summary="s")
            with open(ev.event_path("e1", d), "a") as f:
                f.write("{broken\n")
            got = ev.read_events("e1", d)
            self.assertEqual([e["status"] for e in got], ["start", "end"])
            self.assertEqual(got[1]["data"]["summary"], "s")

    def test_emit_never_raises(self):
        ev.emit("e1", "train", "start", loop=1, root="/proc/nonexistent_dir")


class StateTest(unittest.TestCase):
    def test_parse_train_log_strips_ansi(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.log")
            write(d, "t.log", "\x1b[1m Learning iteration 0/300 \x1b[0m\n Mean reward: -1.50\n"
                  " Learning iteration 1/300\n Mean reward: 2.25\n")
            r = st.parse_train_log(p)
            self.assertEqual(r["iteration"], 1)
            self.assertEqual(r["total"], 300)
            self.assertEqual(r["curve"], [[0, -1.5], [1, 2.25]])
        self.assertEqual(st.parse_train_log("/nonexistent")["curve"], [])

    def test_state_from_events(self):
        with tempfile.TemporaryDirectory() as d:
            ev.emit("x1", "instruction", "end", root=d, instruction="回って", task="spin", max_loops=3,
                    criteria=[{"metric": "height_m", "rule": "0.25..0.35", "min_fraction": 0.9, "comment": ""}])
            ev.emit("x1", "reward", "start", loop=1, root=d)
            ev.emit("x1", "reward", "end", loop=1, root=d, summary="要約", code="PARAMS={}")
            ev.emit("x1", "train", "start", loop=1, root=d, iterations=300)
            write(d, "logs/x1_L1_train.log", "Learning iteration 10/300\nMean reward: 3.0\n")
            s = st.build_state(d, "x1")
            self.assertEqual(s["status"], "running")
            self.assertEqual(s["current"], {"loop": 1, "stage": "train"})
            self.assertFalse(s["synthetic"])
            self.assertEqual(s["loops"][0]["reward_summary"], "要約")
            self.assertEqual(s["loops"][0]["train"]["iteration"], 10)
            ev.emit("x1", "decision", "end", loop=1, root=d, decision={"success": True})
            ev.emit("x1", "run", "end", root=d, stop_reason="success")
            self.assertEqual(st.build_state(d, "x1")["status"], "done")

    def test_crashed_run_is_failed(self):
        with tempfile.TemporaryDirectory() as d:
            ev.emit("x3", "instruction", "end", root=d, instruction="a")
            ev.emit("x3", "run", "end", root=d, stop_reason="crashed")
            self.assertEqual(st.build_state(d, "x3")["status"], "failed")

    def test_stalled_run(self):
        with tempfile.TemporaryDirectory() as d:
            ev.emit("x2", "instruction", "end", root=d, instruction="a")
            ev.emit("x2", "reward", "start", loop=1, root=d)
            s = st.build_state(d, "x2", now=__import__("time").time() + st.STALL_AFTER_S + 5)
            self.assertEqual(s["status"], "stalled")

    def test_legacy_records_are_synthetic(self):
        with tempfile.TemporaryDirectory() as d:
            rec = {"exp_name": "old_L1", "loop": 1, "instruction": "お座り", "loop_seconds": 100,
                   "llm_calls": [{"kind": "reward", "seconds": 10}, {"kind": "evaluate", "seconds": 5}],
                   "train": {"train_seconds": 50, "iterations": 300, "last_checkpoint": 299, "final_mean_reward": 1.0},
                   "metrics": {"record_wall_seconds": 20}, "evaluation": {"score": 9, "success": True}}
            write(d, "logs/loops/old.jsonl", json.dumps(rec) + "\n")
            s = st.build_state(d, "old")
            self.assertTrue(s["synthetic"])
            self.assertEqual(s["status"], "done")
            self.assertTrue(s["loops"][0]["decision"]["legacy"])
            self.assertAlmostEqual(s["t_end"], 100, delta=15)
            stages = s["loops"][0]["stages"]
            self.assertLessEqual(stages["train"]["end"], stages["record"]["start"] + 1e-6)

    def test_bad_names(self):
        self.assertIsNone(st.build_state(".", "../x"))
        self.assertIsNone(st.build_state(".", "nope_not_here"))

    def test_list_tasks(self):
        with tempfile.TemporaryDirectory() as d:
            write(d, "configs/tasks/spin.success.yaml", SPEC)
            write(d, "configs/tasks/bad.success.yaml", "not: a spec")
            t = st.list_tasks(d)
            self.assertEqual([x["task"] for x in t], ["spin"])
            self.assertTrue(t[0]["approved"])


class PickBaseTest(unittest.TestCase):
    def test_pick_base(self):
        with tempfile.TemporaryDirectory() as d:
            for n in ("base", "base_handstand", "base_handstand_walk"):
                write(d, f"configs/{n}.yaml", "x: 1")
            B = lambda t: os.path.basename(srv.pick_base(d, t))
            self.assertEqual(B("sit"), "base.yaml")
            self.assertEqual(B("handstand_hold_v2"), "base_handstand.yaml")
            self.assertEqual(B("handstand_walk"), "base_handstand_walk.yaml")
            self.assertEqual(B("handstand_walk_v2"), "base_handstand_walk.yaml")
            self.assertEqual(B("handstand_circle"), "base_handstand.yaml")  # no base_handstand_circle in this tree


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        write(self.d, "configs/tasks/spin.success.yaml", SPEC)
        write(self.d, "configs/tasks/draft.success.yaml", SPEC.replace("spin", "draft").replace("true", "false"))
        write(self.d, "videos/a_L1_299.mp4", b"0123456789", "wb")
        ev.emit("e1", "instruction", "end", root=self.d, instruction="x")
        self.ran = []
        self.queue = srv.RunQueue(self.d, lambda item: ["true"])
        write(self.d, "configs/tasks/catalog.yaml", CATALOG)
        write(self.d, "scripts/loop.py", "print('loop')\n")
        self.asked = []
        write(self.d, "configs/base.yaml", "# 立位から始める\nrecord:\n  seconds: 6.0\n")
        write(self.d, "configs/hints/index.yaml", HINTS)
        write(self.d, "configs/hints/walk_quadruped.hint.md", "対角の足を交互に踏み出す。進んだ距離だけを報酬にしない。\n")
        self.draft_reply = None
        self.draft_prompts = []
        self.httpd = srv.serve(self.d, 0, self.queue, chooser=self.fake_llm, drafter=self.fake_drafter)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def fake_llm(self, prompt):
        self.asked.append(prompt)
        if "歩き方の設計ヒント" in prompt:
            return '{"hint": "walk_quadruped"}' if "歩" in prompt.split("指示:")[-1] else '{"hint": null}'
        return '{"task": "spin"}' if "回" in prompt.split("指示:")[-1] else '{"task": null}'

    def fake_drafter(self, prompt):
        self.draft_prompts.append(prompt)
        return self.draft_reply

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def get(self, path, headers=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", headers=headers or {})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)

    def post_to(self, path, body):
        h = {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{self.port}", "X-Requested-With": "dashboard"}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(), h)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def post(self, body, origin=None, xr="dashboard"):
        h = {"Content-Type": "application/json"}
        if origin is not False:
            h["Origin"] = origin or f"http://127.0.0.1:{self.port}"
        if xr:
            h["X-Requested-With"] = xr
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/run", json.dumps(body).encode(), h)
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_static_and_api(self):
        self.assertEqual(self.get("/")[0], 200)
        c, b, _ = self.get("/api/experiments")
        self.assertEqual(json.loads(b)[0]["exp"], "e1")
        self.assertEqual(self.get("/api/state?exp=zzz")[0], 404)

    def test_video_range_and_traversal(self):
        c, b, h = self.get("/video/a_L1_299.mp4", {"Range": "bytes=2-4"})
        self.assertEqual((c, b, h["Content-Range"]), (206, b"234", "bytes 2-4/10"))
        c, b, h = self.get("/video/a_L1_299.mp4", {"Range": "bytes=-3"})
        self.assertEqual((c, b), (206, b"789"))
        self.assertEqual(self.get("/video/a_L1_299.mp4", {"Range": "bytes=50-"})[0], 416)
        self.assertEqual(self.get("/video/..%2Fconfigs%2Ftasks%2Fspin.success.yaml")[0], 404)
        self.assertEqual(self.get("/video/nope.mp4")[0], 404)

    def test_trace(self):
        m = {"series_0p1s": {"foot_z_FL": [0.0, 0.1], "q_0": [0.1, 0.2]}, "joint_names": ["FR_hip_joint"]}
        write(self.d, "runs/e1_L1/metrics_299.json", json.dumps(m))
        write(self.d, "runs/e1_L1/metrics_99.json", json.dumps({"series_0p1s": {"foot_z_FL": [9]}}))
        c, b, _ = self.get("/api/trace?exp=e1&loop=1")
        t = json.loads(b)
        self.assertEqual((c, t["joint_names"], t["series"]["q_0"]), (200, ["FR_hip_joint"], [0.1, 0.2]))
        # a run from before joint angles were saved: no joint names, unless a tagged re-recording has them
        write(self.d, "runs/e1_L3/metrics_299.json", json.dumps({"series_0p1s": {"foot_z_FL": [1.0]}}))
        write(self.d, "runs/e1_L3/metrics_299_cam.json", m and json.dumps(m))
        self.assertEqual(json.loads(self.get("/api/trace?exp=e1&loop=3")[1])["joint_names"], ["FR_hip_joint"])
        write(self.d, "runs/e1_L2/metrics_299.json", json.dumps({"series_0p1s": {"foot_z_FL": [0.0]}, "joint_names": ["x"]}))
        self.assertIsNone(json.loads(self.get("/api/trace?exp=e1&loop=2")[1])["joint_names"])
        for bad in ("/api/trace?exp=e1&loop=4", "/api/trace?exp=..&loop=1", "/api/trace?exp=e1&loop=x",
                    "/api/trace?exp=e1&loop=-1", "/api/trace?exp=e1"):
            self.assertEqual(self.get(bad)[0], 404, bad)

    def test_task_is_chosen_from_the_instruction(self):
        c, body = self.post({"exp": "auto1", "instruction": "くるっと回って"})   # no task given
        self.assertEqual((c, body["task"]), (202, "spin"))
        self.assertIn("spin", self.asked[0])
        self.assertNotIn("draft", self.asked[0])   # unapproved and missing specs are not offered
        self.assertNotIn("ghost", self.asked[0])
        c, body = self.post({"exp": "auto2", "instruction": "空を飛んで"})
        self.assertEqual(c, 422)
        self.assertEqual(json.loads(self.get("/api/catalog")[1])[0]["task"], "spin")
        self.assertEqual(json.loads(self.get("/api/state?exp=e1")[1])["task_info"], None)

    DRAFT = {"task_id": "jump", "name": "ジャンプ", "description": "跳ぶ", "base": "configs/base.yaml", "window_s": [3.0, 6.0],
             "criteria": [{"metric": "height_m", "min": 0.4, "min_fraction": 0.2, "comment": "跳び上がる"},
                          {"metric": "early_terminations", "equals": 0}], "rationale": "理由", "cannot_measure": None}

    def test_draft_then_approve_writes_an_approved_spec(self):
        self.draft_reply = "結果: " + json.dumps(self.DRAFT, ensure_ascii=False)
        c, b = self.post_to("/api/draft", {"instruction": "ジャンプして", "force_new": True})
        self.assertEqual(c, 200)
        self.assertEqual((b["errors"], b["draft"]["task_id"]), ([], "jump"))
        self.assertFalse(os.path.exists(os.path.join(self.d, "configs/tasks/jump.success.yaml")))   # a draft writes nothing
        c, b = self.post_to("/api/approve", {"instruction": "ジャンプして", "exp": "j1", "draft": b["draft"]})
        self.assertEqual(c, 202)
        path = os.path.join(self.d, "configs/tasks/jump.success.yaml")
        self.assertTrue(os.path.exists(path))
        from pai_loop import success
        spec, _, sha = success.load_spec(path)   # approved: true and valid
        self.assertEqual(spec["task"], "jump")
        self.assertTrue(sha.startswith(b["sha256"]))
        self.assertEqual(self.post_to("/api/approve", {"instruction": "ジャンプして", "exp": "j2", "draft": {**self.DRAFT}})[0], 422)   # name taken
        cat = {c["task"]: c for c in json.loads(self.get("/api/catalog")[1])}
        self.assertEqual(cat["jump"]["base"], "configs/base.yaml")
        self.assertEqual(srv.pick_base(self.d, "jump"), "configs/base.yaml")
        with open(os.path.join(self.d, "logs/approvals.jsonl")) as f:
            self.assertEqual(json.loads(f.readline())["via"], "dashboard")

    def test_approve_rejects_bad_drafts_and_never_trusts_approved_field(self):
        bad = {**self.DRAFT, "task_id": "Bad Name"}
        self.assertEqual(self.post_to("/api/approve", {"instruction": "x", "exp": "b1", "draft": bad})[0], 422)
        bad = {**self.DRAFT, "task_id": "late", "window_s": [3.0, 9.0]}   # window beyond the 6 s recording
        c, b = self.post_to("/api/approve", {"instruction": "x", "exp": "b2", "draft": bad})
        self.assertEqual(c, 422)
        self.assertIn("録画", " ".join(b["errors"]))
        bad = {**self.DRAFT, "task_id": "nometric", "criteria": [{"metric": "nonsense", "min": 1}]}
        self.assertEqual(self.post_to("/api/approve", {"instruction": "x", "exp": "b3", "draft": bad})[0], 422)
        sneaky = {**self.DRAFT, "task_id": "sneaky", "approved": False, "extra": "x"}
        self.assertEqual(self.post_to("/api/approve", {"instruction": "x", "exp": "b4", "draft": sneaky})[0], 202)
        with open(os.path.join(self.d, "configs/tasks/sneaky.success.yaml")) as f:
            self.assertIn("approved: true", f.read())   # approval is the act of this request, not a field of the draft
        # no origin / header: no approval
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/approve", json.dumps({}).encode(), {"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 403)

    def test_draft_reuse_and_cannot_measure(self):
        c, b = self.post_to("/api/draft", {"instruction": "くるっと回って"})
        self.assertEqual(b["reuse"]["task"], "spin")
        self.draft_reply = json.dumps({**self.DRAFT, "criteria": [], "cannot_measure": "物をつかむ動きは測れません"}, ensure_ascii=False)
        c, b = self.post_to("/api/draft", {"instruction": "空を飛んで"})
        self.assertEqual((c, b["cannot_measure"]), (200, "物をつかむ動きは測れません"))
        self.draft_reply = "not json"
        self.assertEqual(self.post_to("/api/draft", {"instruction": "空を飛んで"})[0], 502)

    def _finished_run(self):
        """An experiment with one recorded loop whose numeric check passed while the AI said no."""
        write(self.d, "runs/jm_L1/metrics_299.json", json.dumps({"xy_max_dist_m": 5.0, "early_terminations": 0,
              "series_0p1s": {"height_m": [0.3] * 60, "roll_deg": [0.0] * 60}}))
        ev.emit("jm", "instruction", "end", root=self.d, instruction="ジャンプして", task="spin", max_loops=1,
                criteria=[], window_s=[3.0, 6.0])
        ev.emit("jm", "numeric", "end", loop=1, root=self.d, result={"passed": True, "n_passed": 1, "n_criteria": 1, "criteria": [
            {"metric": "height_m", "rule": "x", "measured": 1.0, "passed": True}]})
        ev.emit("jm", "vlm", "end", loop=1, root=self.d, evaluation={"score": 2, "success": False, "observations": "走り続けている"})
        ev.emit("jm", "run", "end", root=self.d, stop_reason="max_loops")

    IMPROVED = {"diagnosis": "移動距離の基準が無く、走り続ける動きが通った", "window_s": [3.0, 6.0], "hint": "着地したら止まる",
                "criteria": [{"metric": "height_m", "between": [0.25, 0.35], "min_fraction": 0.9, "comment": "高さ"},
                             {"metric": "xy_max_dist_m", "max": 0.5, "comment": "その場"}], "rationale": "移動を制限"}

    def test_improve_proposes_a_new_version_and_previews_it_on_recorded_loops(self):
        self._finished_run()
        self.draft_reply = json.dumps(self.IMPROVED, ensure_ascii=False)
        c, b = self.post_to("/api/improve", {"exp": "jm"})
        self.assertEqual(c, 200)
        self.assertEqual(b["draft"]["task_id"], "spin_v2")           # a new version, the old file is untouched
        self.assertEqual(b["draft"]["base"], "configs/base.yaml")
        row = b["preview"][0]
        self.assertEqual((row["passed"], row["old_passed"], row["ai_pass"], row["agrees"]), (False, True, False, True))   # now fails like the AI
        self.assertFalse(os.path.exists(os.path.join(self.d, "configs/tasks/spin_v2.success.yaml")))   # nothing written yet
        c, b2 = self.post_to("/api/preview", {"exp": "jm", "draft": {**b["draft"], "criteria": [b["draft"]["criteria"][0]]}})
        self.assertEqual((c, b2["preview"][0]["passed"]), (200, True))   # editing a criterion away changes the verdict
        self.assertEqual(self.post_to("/api/preview", {"exp": "jm", "draft": {**b["draft"], "criteria": [{"metric": "zzz", "min": 1}]}})[0], 422)
        self.assertEqual(self.post_to("/api/improve", {"exp": "nope"})[0], 404)

    def test_improve_prompt_after_a_success_asks_for_stricter_criteria(self):
        from dashboard import improve as im
        st_ = {"instruction": "歩いて", "task": "walk", "stop_reason": "success", "exp": "x"}
        text = im.improve_prompt(st_, "task: walk", [])
        self.assertIn("成功しました", text)
        self.assertIn("改善点の整理", text)
        self.assertIn("改善点が直った動きだけが通る", text)
        self.assertNotIn("AIが成功とみなした動きは通ること", text)   # a success may now fail the stricter criteria
        text = im.improve_prompt({**st_, "stop_reason": "max_loops"}, "task: walk", [])
        self.assertIn("結果が思わしくない", text)
        self.assertIn("AIが成功とみなした動きは通ること", text)

    def test_approve_without_starting_writes_spec_and_hint(self):
        self._finished_run()
        c, b = self.post_to("/api/approve", {"instruction": "ジャンプして", "start": False, "hint": "着地したら止まる",
                                             "draft": {"task_id": "spin_v2", "name": "回る2", "base": "configs/base.yaml", "window_s": [3.0, 6.0],
                                                       "criteria": self.IMPROVED["criteria"]}})
        self.assertEqual(c, 200)
        self.assertTrue(os.path.exists(os.path.join(self.d, "configs/tasks/spin_v2.success.yaml")))
        with open(os.path.join(self.d, "configs/tasks/spin_v2.hint.md")) as f:
            self.assertIn("着地", f.read())
        self.assertEqual(self.queue.snapshot(), [])   # not queued
        cat = [c["task"] for c in json.loads(self.get("/api/catalog")[1])]
        self.assertIn("spin_v2", cat)
        self.assertNotIn("spin", cat)   # the new version replaces the old one in the offers

    def test_non_finite_numbers_are_sent_as_null(self):
        # a run whose simulation diverged leaves NaN in its series; the browser must still be able to parse the reply
        write(self.d, "runs/e1_L1/metrics_299.json", '{"series_0p1s": {"foot_z_FL": [0.0, NaN, Infinity, 0.1]}, "joint_names": null}')
        write(self.d, "runs/e1_L1/overlay_299.json", '{"video": "v.mp4", "pts": [[[0, NaN, 1]]], "base_xy": [[0, 0]]}')
        for path in ("/api/trace?exp=e1&loop=1", "/api/overlay?exp=e1&loop=1"):
            c, b, _ = self.get(path)
            self.assertEqual(c, 200, path)
            json.loads(b, parse_constant=lambda x: self.fail(f"{x} in the reply of {path}"))   # strict JSON
        self.assertEqual(json.loads(self.get("/api/trace?exp=e1&loop=1")[1])["series"]["foot_z_FL"], [0.0, None, None, 0.1])

    def test_overlay_endpoint(self):
        write(self.d, "runs/e1_L1/overlay_299.json", json.dumps({"video": "e1_L1_299.mp4", "pts": [[[0, 0, 0]]]}))
        write(self.d, "runs/e1_L1/overlay_299_ov.json", json.dumps({"video": "e1_L1_299_ov.mp4"}))
        self.assertEqual(json.loads(self.get("/api/overlay?exp=e1&loop=1")[1])["video"], "e1_L1_299.mp4")   # untagged first
        self.assertEqual(self.get("/api/overlay?exp=e1&loop=2")[0], 404)
        self.assertEqual(self.get("/api/overlay?exp=..&loop=1")[0], 404)

    def test_gait_hint_is_offered_to_the_draft_written_with_the_spec_and_passed_to_runs(self):
        from dashboard import tasks as tk
        self.assertEqual([h["id"] for h in tk.load_hints(self.d)], ["walk_quadruped"])   # a missing file and a path outside configs are ignored
        self.draft_reply = json.dumps({**self.DRAFT, "task_id": "walk_x"}, ensure_ascii=False)
        c, b = self.post_to("/api/draft", {"instruction": "四足で歩いて", "force_new": True})
        self.assertEqual((c, b["hint"]["id"]), (200, "walk_quadruped"))
        self.assertIn("対角の足", b["hint"]["text"])
        self.assertTrue(any("前進距離と足の交互" in p for p in self.draft_prompts))   # the spec notes reached the draft prompt
        c, b2 = self.post_to("/api/approve", {"instruction": "四足で歩いて", "start": False, "hint_id": "walk_quadruped",
                                              "hint": "着地したら止まる", "draft": {**b["draft"], "base": "configs/base.yaml"}})
        self.assertEqual(c, 200)
        with open(os.path.join(self.d, "configs/tasks/walk_x.hint.md")) as f:
            text = f.read()
        self.assertIn("対角の足", text)     # the library hint
        self.assertIn("着地したら止まる", text)  # and the typed advice, both in the file the loop reads
        c, b3 = self.post_to("/api/draft", {"instruction": "空を飛んで", "force_new": True})
        self.assertIsNone(b3["hint"])       # not a gait: no hint

    def test_run_command_carries_hint_and_prev_exp(self):
        cmd = srv.default_command(self.d, python="py")({"exp": "e", "instruction": "i", "task": "spin", "hint": "configs/hints/walk_quadruped.hint.md", "prev_exp": "old"})
        self.assertEqual(cmd[cmd.index("--hint") + 1], "configs/hints/walk_quadruped.hint.md")
        self.assertEqual(cmd[cmd.index("--prev-exp") + 1], "old")
        self.assertNotIn("--hint", srv.default_command(self.d, python="py")({"exp": "e", "instruction": "i", "task": "spin"}))
        # a run of an existing task: the fitting hint is attached unless the task has its own hint file
        c, b = self.post({"exp": "gw1", "instruction": "四足で歩いて", "task": "spin"})
        self.assertEqual((c, b["hint"]), (202, "configs/hints/walk_quadruped.hint.md"))
        write(self.d, "configs/tasks/spin.hint.md", "own hint")
        c, b = self.post({"exp": "gw2", "instruction": "四足で歩いて", "task": "spin"})
        self.assertIsNone(b["hint"])   # loop.py reads configs/tasks/spin.hint.md by itself
        # prev_exp is taken only when that experiment has loop records
        write(self.d, "logs/loops/old.jsonl", "{}\n")
        c, b = self.post({"exp": "gw3", "instruction": "回って", "task": "spin", "prev_exp": "old"})
        self.assertEqual(b["prev_exp"], "old")
        c, b = self.post({"exp": "gw4", "instruction": "回って", "task": "spin", "prev_exp": "../x"})
        self.assertIsNone(b["prev_exp"])

    def test_parse_choice(self):
        cat = [{"task": "spin"}]
        from dashboard import tasks as tk
        self.assertEqual(tk.parse_choice('説明 {"task": "spin"} 以上', cat), "spin")
        self.assertIsNone(tk.parse_choice('{"task": "other"}', cat))   # not in the catalog
        self.assertIsNone(tk.parse_choice("spin", cat))
        self.assertIsNone(tk.parse_choice('{"task": null}', cat))

    def test_source_endpoint_serves_only_listed_files(self):
        c, b, _ = self.get("/api/source?stage=instruction")
        self.assertEqual((c, json.loads(b)[0]["path"]), (200, "scripts/loop.py"))
        c, b, _ = self.get("/api/source?stage=numeric&exp=e1")
        self.assertEqual(c, 200)
        self.assertEqual(self.get("/api/source?stage=../../etc/passwd")[0], 404)
        self.assertEqual(self.get("/api/source?stage=nope")[0], 404)

    def test_stop_terminates_the_run_and_marks_it_stopped(self):
        import subprocess as sp
        import time as _t
        q = srv.RunQueue(self.d, lambda item: ["sleep", "60"])
        httpd = srv.serve(self.d, 0, q)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        port = httpd.server_address[1]
        try:
            ev.emit("stp1", "instruction", "end", root=self.d, instruction="x")
            ev.emit("stp1", "train", "start", loop=1, root=self.d)
            self.assertIsNotNone(q.add("stp1", "x", "spin"))
            for _ in range(50):
                if q.snapshot()[0]["status"] == "running" and q.procs:
                    break
                _t.sleep(0.05)
            pid = q.procs["stp1"].pid
            self.assertTrue(q.can_stop("stp1"))

            def post(path, body, **h):
                hdr = {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}", "X-Requested-With": "dashboard", **h}
                req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), hdr)
                try:
                    with urllib.request.urlopen(req) as r:
                        return r.status, json.loads(r.read())
                except urllib.error.HTTPError as e:
                    return e.code, json.loads(e.read())
            self.assertEqual(post("/api/stop", {"exp": "nope"})[0], 404)            # not a run of this server
            self.assertEqual(post("/api/stop", {"exp": "stp1"}, Origin="http://evil.example")[0], 403)
            self.assertEqual(post("/api/stop", {"exp": "stp1"}), (200, {"result": "stopping"}))
            for _ in range(100):
                if q.snapshot()[0]["status"] == "stopped":
                    break
                _t.sleep(0.05)
            self.assertEqual(q.snapshot()[0]["status"], "stopped")
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)   # really gone
            s = st.build_state(self.d, "stp1")
            self.assertEqual((s["status"], s["stop_reason"]), ("stopped", "stopped"))
            self.assertEqual(s["loops"][0]["stages"]["train"]["status"], "aborted")   # the open stage is closed, not "running"
            self.assertEqual(post("/api/stop", {"exp": "stp1"})[0], 404)           # nothing left to stop
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_stop_cancels_a_queued_run(self):
        import time as _t
        q = srv.RunQueue(self.d, lambda item: ["sleep", "60"])
        q.add("qa", "x", "spin")
        for _ in range(50):
            if q.procs:
                break
            _t.sleep(0.05)
        q.add("qb", "x", "spin")   # waits behind qa
        self.assertEqual(q.stop("qb"), "cancelled")
        self.assertEqual(q.stop("qa"), "stopping")
        for _ in range(100):
            if q.snapshot()[0]["status"] == "stopped":
                break
            _t.sleep(0.05)
        self.assertEqual([i["status"] for i in q.snapshot()], ["stopped", "cancelled"])   # qb never started

    def test_run_guards(self):
        ok = {"exp": "new1", "instruction": "回って", "task": "spin"}
        self.assertEqual(self.post(ok, origin="http://evil.example")[0], 403)
        self.assertEqual(self.post(ok, origin="http://127.0.0.1:1")[0], 403)  # not the page's own origin
        self.assertEqual(self.post(ok, origin=False)[0], 403)
        self.assertEqual(self.post(ok, xr="")[0], 403)
        self.assertEqual(self.post({**ok, "exp": "../x"})[0], 400)
        self.assertEqual(self.post({**ok, "instruction": "あ" * 201})[0], 400)
        self.assertEqual(self.post({**ok, "task": "draft"})[0], 400)  # unapproved spec
        self.assertEqual(self.post({**ok, "exp": "e1"})[0], 409)       # exists
        self.assertEqual(self.post(ok)[0], 202)
        self.assertEqual(self.post(ok)[0], 409)                        # already queued


if __name__ == "__main__":
    unittest.main()
