import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import export_static as ex  # noqa: E402
from pai_loop import events as ev  # noqa: E402

SPEC = """task: spin
approved: true
window_s: [3.0, 6.0]
criteria:
  - {metric: height_m, between: [0.25, 0.35], min_fraction: 0.9}
"""


def write(root, rel, text, mode="w"):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, mode) as f:
        f.write(text)


class ExportTest(unittest.TestCase):
    def test_site_has_everything_the_screen_asks_for_and_is_valid_json(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as out:
            write(root, "configs/tasks/spin.success.yaml", SPEC)
            write(root, "scripts/loop.py", "print('loop')\n")
            ev.emit("e1", "instruction", "end", root=root, instruction="回って", task="spin", max_loops=1, criteria=[])
            ev.emit("e1", "record", "end", loop=1, root=root, video="e1_L1_9.mp4")
            ev.emit("e1", "run", "end", root=root, stop_reason="max_loops")
            write(root, "runs/e1_L1/metrics_9.json", '{"series_0p1s": {"foot_z_FL": [0.0, NaN]}}')   # NaN, as a diverged run leaves it
            write(root, "runs/e1_L1/overlay_9.json", json.dumps({"video": "e1_L1_9.mp4", "pts": [], "base_xy": []}))
            write(root, "videos/e1_L1_9.mp4", b"0123", "wb")
            done, videos = ex.export(root, out, ["e1", "missing"])
            self.assertEqual([d["exp"] for d in done], ["e1"])   # an experiment without a record is skipped
            self.assertEqual(videos >= 1, True)
            for rel in ("data/experiments.json", "data/catalog.json", "data/state/e1.json", "data/trace/e1_1.json",
                        "data/overlay/e1_1.json", "data/source/instruction__e1.json", "data/video/e1_L1_9.mp4", "app.js", "style.css"):
                self.assertTrue(os.path.exists(os.path.join(out, rel)), rel)
            for rel in ("data/state/e1.json", "data/trace/e1_1.json", "data/source/numeric__e1.json"):
                with open(os.path.join(out, rel), encoding="utf-8") as f:
                    json.loads(f.read(), parse_constant=lambda x: self.fail(f"{x} in {rel}"))   # strict JSON
            numeric = json.load(open(os.path.join(out, "data/source/numeric__e1.json"), encoding="utf-8"))
            self.assertIn("configs/tasks/spin.success.yaml", [r["path"] for r in numeric])
            html = open(os.path.join(out, "index.html"), encoding="utf-8").read()
            self.assertIn("window.STATIC_SITE = true", html)
            self.assertIn('href="style.css"', html)   # relative paths: the site can live under a sub-path
            self.assertNotIn('src="/app.js"', html)
            state = json.load(open(os.path.join(out, "data/state/e1.json"), encoding="utf-8"))
            self.assertFalse(state["can_stop"])


if __name__ == "__main__":
    unittest.main()
