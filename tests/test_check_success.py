import json
import os
import subprocess
import sys
import tempfile
import unittest

from helpers import DT, make_log

from pai_loop import metrics as M

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "check_success.py")

SPEC = """
task: sit
approved: {approved}
window_s: [3.0, 6.0]
criteria:
  - {{metric: nose_up_deg, between: [40, 65], min_fraction: 0.9}}
  - {{metric: early_terminations, equals: 0}}
"""


def sit_metrics(nose_up):
    log = make_log(301, lambda k: {"nose_up_deg": nose_up, "pos": [0.0, 0.0, 0.22]})
    return M.compute_metrics(log, DT, settle_s=3.0, n_failed=0)


class TestCheckSuccessCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        os.makedirs(os.path.join(self.tmp.name, "runs", "e1"))

    def write(self, rel, text):
        path = os.path.join(self.tmp.name, rel)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def run_cli(self, *extra, approved=True, nose_up=50.0, with_metrics=True, with_summary=True):
        spec = self.write("spec.yaml", SPEC.format(approved=str(approved).lower()))
        if with_summary:
            self.write("runs/e1/summary.json", json.dumps({"last_checkpoint": 7}))
        if with_metrics:
            self.write("runs/e1/metrics_7.json", json.dumps(sit_metrics(nose_up)))
        return subprocess.run([sys.executable, SCRIPT, "e1", "--spec", spec, *extra], cwd=self.tmp.name,
                              capture_output=True, text=True)

    def test_pass_exits_0_and_writes_the_result(self):
        r = self.run_cli()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        with open(os.path.join(self.tmp.name, "runs", "e1", "success_7.json"), encoding="utf-8") as f:
            saved = json.load(f)
        self.assertTrue(saved["passed"])
        self.assertEqual(len(saved["spec_sha256"]), 64)

    def test_fail_exits_1(self):
        r = self.run_cli(nose_up=30.0)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("不合格", r.stdout)

    def test_unapproved_spec_exits_2_unless_verification_flag(self):
        r = self.run_cli(approved=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("not approved", r.stderr)
        r = self.run_cli("--allow-unapproved", approved=False)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("未承認", r.stdout)

    def test_missing_inputs_exit_2(self):
        self.assertEqual(self.run_cli(with_metrics=False).returncode, 2)
        self.assertEqual(self.run_cli(with_summary=False, with_metrics=False).returncode, 2)


if __name__ == "__main__":
    unittest.main()
