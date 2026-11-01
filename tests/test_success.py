import math
import os
import tempfile
import unittest

import yaml
from helpers import DT, make_log, yaw_quat

from pai_loop import metrics as M
from pai_loop import success as S

SIT_SPEC = """
task: sit
approved: true
window_s: [3.0, 6.0]
criteria:
  - {metric: nose_up_deg, between: [40, 65], min_fraction: 0.9}
  - {metric: height_m, between: [0.17, 0.26], min_fraction: 0.9}
  - {metric: foot_z_max_m, max: 0.04, min_fraction: 0.9}
  - {metric: roll_deg_abs, max: 10, min_fraction: 0.9}
  - {metric: xy_max_dist_m, max: 0.15}
  - {metric: early_terminations, equals: 0}
"""

SPIN_SPEC = """
task: spin
approved: true
window_s: [3.0, 6.0]
criteria:
  - {metric: yaw_rate_abs_deg_s_mean, between: [60, 180]}
  - {metric: yaw_rate_steady_fraction, min: 0.8}
  - {metric: xy_max_dist_m, max: 0.5}
  - {metric: early_terminations, equals: 0}
"""


def spec_of(text, **override):
    spec = yaml.safe_load(text)
    spec.update(override)
    return spec


def sit_metrics(nose_up=50.0, roll=3.0, height=0.22, foot_z=0.023, n_failed=0):
    feet = [[0.2, 0.1, foot_z], [0.2, -0.1, foot_z], [-0.2, 0.1, foot_z], [-0.2, -0.1, foot_z]]
    log = make_log(301, lambda k: {"nose_up_deg": nose_up, "roll_deg": roll, "pos": [0.0, 0.0, height], "feet": feet})
    return M.compute_metrics(log, DT, settle_s=3.0, n_failed=n_failed)


def spin_metrics(rate_deg_s=90.0, radius=0.0):
    omega = 2 * math.pi / 3

    def state(k):
        t = k * DT
        return {"pos": [radius * math.sin(omega * t), radius * (1 - math.cos(omega * t)), 0.29],
                "quat": yaw_quat(rate_deg_s * t)}

    return M.compute_metrics(make_log(301, state), DT, settle_s=3.0, n_failed=0)


class TestValidation(unittest.TestCase):
    def check_error(self, spec, needle):
        with self.assertRaises(S.SpecError) as cm:
            S.validate_spec(spec)
        self.assertIn(needle, str(cm.exception))

    def test_valid_specs(self):
        S.validate_spec(spec_of(SIT_SPEC))
        S.validate_spec(spec_of(SPIN_SPEC))

    def test_unknown_metric_lists_the_catalog(self):
        spec = spec_of(SIT_SPEC)
        spec["criteria"][0]["metric"] = "nose_down"
        self.check_error(spec, "unknown metric")
        self.check_error(spec, "yaw_rate_deg_s")

    def test_series_metric_needs_min_fraction(self):
        spec = spec_of(SIT_SPEC)
        del spec["criteria"][0]["min_fraction"]
        self.check_error(spec, "min_fraction")

    def test_scalar_metric_rejects_min_fraction(self):
        spec = spec_of(SIT_SPEC)
        spec["criteria"][4]["min_fraction"] = 0.9
        self.check_error(spec, "only for series")

    def test_exactly_one_rule(self):
        spec = spec_of(SIT_SPEC)
        spec["criteria"][4]["min"] = 0.0
        self.check_error(spec, "exactly one")

    def test_between_must_be_ordered(self):
        spec = spec_of(SIT_SPEC)
        spec["criteria"][0]["between"] = [65, 40]
        self.check_error(spec, "lo < hi")

    def test_approved_window_and_criteria_shapes(self):
        spec = spec_of(SIT_SPEC)
        spec["approved"] = "yes"
        self.check_error(spec, "approved")
        self.check_error(spec_of(SIT_SPEC, window_s=[6.0, 3.0]), "window_s")
        self.check_error(spec_of(SIT_SPEC, criteria=[]), "criteria")

    def test_non_finite_numbers_are_rejected(self):
        spec = spec_of(SIT_SPEC)
        spec["criteria"][4]["max"] = float("nan")
        self.check_error(spec, "finite")

    def test_metric_kinds(self):
        self.assertEqual(S.metric_kind("nose_up_deg"), "series")
        self.assertEqual(S.metric_kind("yaw_rate_deg_s_mean"), "window")
        self.assertEqual(S.metric_kind("foot_z_max_m_max"), "window")
        self.assertEqual(S.metric_kind("xy_max_dist_m"), "episode")
        self.assertIsNone(S.metric_kind("bogus_mean"))


class TestLoadSpec(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_unapproved_spec_is_refused(self):
        path = self.write(SIT_SPEC.replace("approved: true", "approved: false"))
        with self.assertRaises(S.NotApprovedError):
            S.load_spec(path)
        spec, _, _ = S.load_spec(path, require_approved=False)
        self.assertFalse(spec["approved"])

    def test_hash_changes_when_the_file_changes(self):
        a = S.load_spec(self.write(SIT_SPEC))[2]
        b = S.load_spec(self.write(SIT_SPEC.replace("0.17", "0.10")))[2]
        self.assertEqual(len(a), 64)
        self.assertNotEqual(a, b)


class TestShippedSpecs(unittest.TestCase):
    """Every spec under configs/tasks must stay valid (approval is a human decision and is not asserted here)."""

    def test_all_task_specs_validate(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tasks_dir = os.path.join(root, "configs", "tasks")
        names = sorted(f for f in os.listdir(tasks_dir) if f.endswith(".success.yaml"))
        self.assertTrue(names)
        for name in names:
            spec, _, sha = S.load_spec(os.path.join(tasks_dir, name), require_approved=False)
            self.assertEqual(len(sha), 64, name)
            self.assertEqual(spec["task"], name[: -len(".success.yaml")], name)


class TestEvaluateSit(unittest.TestCase):
    def test_good_sit_passes(self):
        r = S.evaluate(sit_metrics(), spec_of(SIT_SPEC))
        self.assertTrue(r["passed"], S.format_result(r))
        self.assertEqual(r["n_passed"], r["n_criteria"])

    def test_pitch_too_small_fails_only_that_criterion(self):
        r = S.evaluate(sit_metrics(nose_up=36.0), spec_of(SIT_SPEC))
        self.assertFalse(r["passed"])
        failed = [c["metric"] for c in r["criteria"] if not c["passed"]]
        self.assertEqual(failed, ["nose_up_deg"])
        self.assertEqual(r["criteria"][0]["measured"], 0.0)  # 0% of the window in range
        self.assertAlmostEqual(r["criteria"][0]["window_mean"], 36.0, places=3)
        self.assertIn("nose_up_deg", S.failure_summary(r))

    def test_a_lifted_foot_fails(self):
        r = S.evaluate(sit_metrics(foot_z=0.065), spec_of(SIT_SPEC))
        self.assertIn("foot_z_max_m", [c["metric"] for c in r["criteria"] if not c["passed"]])

    def test_early_termination_fails(self):
        r = S.evaluate(sit_metrics(n_failed=1), spec_of(SIT_SPEC))
        self.assertIn("early_terminations", [c["metric"] for c in r["criteria"] if not c["passed"]])

    def test_min_fraction_semantics(self):
        series = [50.0] * 61
        for k in (35, 40, 45, 50):  # 4 of the 31 window samples (k=30..60) out of range -> 27/31 = 87%
            series[k] = 20.0
        metrics = {"series_0p1s": {"nose_up_deg": series}}
        spec = {"task": "t", "approved": True, "window_s": [3.0, 6.0],
                "criteria": [{"metric": "nose_up_deg", "between": [40, 65], "min_fraction": 0.9}]}
        self.assertFalse(S.evaluate(metrics, spec)["passed"])
        spec["criteria"][0]["min_fraction"] = 0.8
        self.assertTrue(S.evaluate(metrics, spec)["passed"])

    def test_missing_metric_fails_with_a_clear_note(self):
        legacy = {"series_0p1s": {"nose_up_deg": [50.0] * 61}, "early_terminations": 0}  # no height, feet, xy ...
        r = S.evaluate(legacy, spec_of(SIT_SPEC))
        self.assertFalse(r["passed"])
        by_name = {c["metric"]: c for c in r["criteria"]}
        self.assertTrue(by_name["nose_up_deg"]["passed"])
        self.assertFalse(by_name["xy_max_dist_m"]["passed"])
        self.assertIn("録画し直し", by_name["xy_max_dist_m"]["note"])
        self.assertIn("判定できない", S.format_result(r))


class TestEvaluateSpin(unittest.TestCase):
    def test_handstand_derived_series(self):
        n = 60
        low, high = [0.023] * n, [0.40] * n
        m = {"series_0p1s": {"foot_z_FL": low, "foot_z_FR": [0.03] * n, "foot_z_RL": high, "foot_z_RR": [0.30] * n}}
        self.assertEqual(S._series(m, "front_feet_z_max_m"), [0.03] * n)
        self.assertEqual(S._series(m, "rear_feet_z_min_m"), [0.30] * n)
        self.assertEqual(S.metric_kind("rear_feet_z_min_m_min"), "window")
        self.assertIsNone(S._series({"series_0p1s": {}}, "rear_feet_z_min_m"))

    def test_xy_dist_range_ignores_the_move_before_the_window(self):
        d = [0.0, 0.2, 0.5, 0.55, 0.56, 0.55, 0.56]  # moves early, then holds (0.1 s samples; window 0.3-0.6 s)
        v, note = S._window_value({"series_0p1s": {"xy_dist_m": d}}, "xy_dist_range_m", [0.3, 0.6])
        self.assertAlmostEqual(v, 0.01, places=6)
        self.assertEqual(S.metric_kind("xy_dist_range_m"), "window")

    def test_front_foot_series_for_gait(self):
        fl, fr = [0.02, 0.10, 0.02, 0.02], [0.02, 0.02, 0.09, 0.02]
        m = {"series_0p1s": {"foot_z_FL": fl, "foot_z_FR": fr}}
        self.assertEqual(S._series(m, "front_feet_z_min_m"), [0.02] * 4)
        v, _ = S._window_value(m, "foot_z_FL_m_max", [0.0, 0.3])
        self.assertEqual(v, 0.10)
        self.assertEqual(S.metric_kind("foot_z_FR_m_max"), "window")

    def _circle_metrics(self, arc_deg, radius=1.0, n=60):
        pts = [(radius * math.cos(t) - radius, radius * math.sin(t)) for t in [i * math.radians(arc_deg) / (n - 1) for i in range(n)]]
        return {"series_0p1s": {"x_m": [p[0] for p in pts], "y_m": [p[1] for p in pts]}}

    def test_circle_metrics(self):
        m = self._circle_metrics(270)
        w = [0.0, 5.9]
        self.assertAlmostEqual(S._window_value(m, "circle_radius_m", w)[0], 1.0, places=3)
        self.assertAlmostEqual(S._window_value(m, "circle_arc_deg", w)[0], 270.0, places=1)
        self.assertLess(S._window_value(m, "circle_fit_rms_m", w)[0], 1e-6)
        self.assertEqual(S.metric_kind("circle_arc_deg"), "window")

    def test_straight_or_missing_path_has_no_circle(self):
        line = {"series_0p1s": {"x_m": [i * 0.1 for i in range(40)], "y_m": [0.0] * 40}}
        v, note = S._window_value(line, "circle_radius_m", [0.0, 3.9])
        self.assertIsNone(v)
        self.assertIn("円", note)
        self.assertIsNone(S._window_value({"series_0p1s": {}}, "circle_arc_deg", [0, 1])[0])

    def test_front_step_count_counts_separate_lifts_of_each_foot(self):
        fl = [0.02, 0.09, 0.02, 0.09, 0.02, 0.09]  # 3 lifts
        fr = [0.02, 0.02, 0.09, 0.09, 0.02, 0.02]  # 1 lift
        m = {"series_0p1s": {"foot_z_FL": fl, "foot_z_FR": fr}}
        self.assertEqual(S._window_value(m, "front_step_count_min", [0.0, 0.5])[0], 1.0)
        m["series_0p1s"]["foot_z_FR"] = [0.02, 0.09, 0.02, 0.09, 0.02, 0.09]
        self.assertEqual(S._window_value(m, "front_step_count_min", [0.0, 0.5])[0], 3.0)
        self.assertEqual(S.metric_kind("front_step_count_min"), "window")

    def test_rear_and_flight_metrics(self):
        z = [0.02, 0.09, 0.02, 0.09, 0.02, 0.09]
        m = {"series_0p1s": {"foot_z_FL": [0.02] * 6, "foot_z_FR": [0.02] * 6, "foot_z_RL": z, "foot_z_RR": [0.02, 0.02, 0.09, 0.09, 0.02, 0.02]}}
        self.assertEqual(S._window_value(m, "rear_step_count_min", [0.0, 0.5])[0], 1.0)
        self.assertEqual(S.metric_kind("rear_step_count_min"), "window")
        self.assertEqual(S._series(m, "rear_one_foot_up"), [0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        self.assertEqual(S._series(m, "foot_z_RL_m"), z)
        self.assertEqual(S.metric_kind("rear_feet_z_max_m"), "series")
        self.assertEqual(set(S._series(m, "all_feet_up")), {0.0})   # the front feet stay down: never airborne
        for k in ("foot_z_FL", "foot_z_FR"):
            m["series_0p1s"][k] = [0.2] * 6
        m["series_0p1s"]["foot_z_RR"] = [0.09] * 6
        self.assertEqual(S._series(m, "all_feet_up"), [0.0, 1.0, 0.0, 1.0, 0.0, 1.0])   # the rear-left foot is down at samples 0, 2, 4
        # each of the new names is accepted by the validator
        for name in ("foot_z_RR_m", "rear_feet_z_max_m", "rear_one_foot_up", "all_feet_up"):
            S.validate_spec({"task": "t", "approved": False, "window_s": [0, 1], "criteria": [{"metric": name, "max": 1, "min_fraction": 0.5}]})

    def test_front_one_foot_up(self):
        m = {"series_0p1s": {"foot_z_FL": [0.02, 0.10, 0.10, 0.02], "foot_z_FR": [0.02, 0.02, 0.10, 0.10]}}
        self.assertEqual(S._series(m, "front_one_foot_up"), [0.0, 1.0, 0.0, 1.0])

    def test_head_and_knee_series(self):
        m = {"series_0p1s": {"head_z": [0.3, 0.1], "knee_z_FL": [0.2, 0.05], "knee_z_FR": [0.1, 0.4]}}
        self.assertEqual(S._series(m, "head_z_m"), [0.3, 0.1])
        self.assertEqual(S._series(m, "front_knee_z_min_m"), [0.1, 0.05])
        self.assertIsNone(S._series({"series_0p1s": {}}, "front_knee_z_min_m"))

    def test_in_place_spin_passes(self):
        r = S.evaluate(spin_metrics(90.0, radius=0.0), spec_of(SPIN_SPEC))
        self.assertTrue(r["passed"], S.format_result(r))

    def test_circling_is_rejected_even_though_it_returns_to_the_start(self):
        m = spin_metrics(90.0, radius=0.5)
        self.assertLess(m["xy_drift_m"], 0.01)
        r = S.evaluate(m, spec_of(SPIN_SPEC))
        self.assertFalse(r["passed"])
        self.assertEqual([c["metric"] for c in r["criteria"] if not c["passed"]], ["xy_max_dist_m"])

    def test_too_slow_fails_and_direction_is_ignored_by_the_abs_metric(self):
        self.assertFalse(S.evaluate(spin_metrics(30.0), spec_of(SPIN_SPEC))["passed"])
        self.assertTrue(S.evaluate(spin_metrics(-90.0), spec_of(SPIN_SPEC))["passed"])

    def test_signed_rate_distinguishes_direction(self):
        spec = spec_of(SPIN_SPEC)
        spec["criteria"][0] = {"metric": "yaw_rate_deg_s_mean", "between": [60, 180]}
        self.assertTrue(S.evaluate(spin_metrics(90.0), spec)["passed"])
        self.assertFalse(S.evaluate(spin_metrics(-90.0), spec)["passed"])

    def test_reversing_spin_is_not_steady(self):
        omega = 2 * math.pi / 3

        def state(k):
            t = k * DT
            yaw = 90.0 * t if t < 4.5 else 90.0 * 4.5 - 90.0 * (t - 4.5)  # reverses halfway through the window
            return {"pos": [0.0, 0.0, 0.29], "quat": yaw_quat(yaw)}

        m = M.compute_metrics(make_log(301, state), DT, settle_s=3.0, n_failed=0)
        spec = spec_of(SPIN_SPEC)
        spec["criteria"] = [{"metric": "yaw_rate_steady_fraction", "min": 0.8}]
        r = S.evaluate(m, spec)
        self.assertFalse(r["passed"])
        self.assertLess(r["criteria"][0]["measured"], 0.8)


if __name__ == "__main__":
    unittest.main()
