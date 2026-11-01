import math
import unittest

from helpers import DT, make_log, orientation, pitch_quat, standing_feet, yaw_quat

from pai_loop import metrics as M


class TestHeading(unittest.TestCase):
    def test_forward_axis_identity(self):
        self.assertEqual(M.forward_axis((1.0, 0.0, 0.0, 0.0)), (1.0, 0.0, 0.0))

    def test_forward_axis_vertical_component_equals_nose_up(self):
        for phi in (0, 30, 45, 80):
            fz = M.forward_axis(pitch_quat(phi))[2]
            self.assertAlmostEqual(math.degrees(math.asin(fz)), phi, places=6)

    def test_heading_is_undefined_when_nearly_vertical(self):
        self.assertIsNotNone(M.heading_rad(pitch_quat(60)))
        self.assertIsNone(M.heading_rad(pitch_quat(85)))

    def test_unwrap_crosses_plus_minus_180(self):
        quats = [yaw_quat(d) for d in (170, 175, 185, 190)]  # 185 and 190 wrap to -175 and -170
        yaw, valid = M.unwrapped_yaw_deg(quats)
        for got, want in zip(yaw, (0, 5, 15, 20)):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual(valid, 1.0)

    def test_yaw_is_held_while_heading_is_undefined(self):
        quats = [yaw_quat(0), yaw_quat(10), orientation(yaw_deg=50, nose_up_deg=85)]
        yaw, valid = M.unwrapped_yaw_deg(quats)
        self.assertAlmostEqual(yaw[1], 10.0, places=6)
        self.assertAlmostEqual(yaw[2], 10.0, places=6)
        self.assertAlmostEqual(valid, 2 / 3, places=6)


class TestComputeMetrics(unittest.TestCase):
    def test_spin_in_circle_is_caught_by_max_dist_but_not_by_net_drift(self):
        # 6 s: yaw 90 deg/s, walking a circle of radius 0.5 m twice around (returns to the start point).
        n, radius, omega = 301, 0.5, 2 * math.pi / 3

        def state(k):
            t = k * DT
            return {"pos": [radius * math.sin(omega * t), radius * (1 - math.cos(omega * t)), 0.29],
                    "quat": yaw_quat(90.0 * t)}

        m = M.compute_metrics(make_log(n, state), DT, settle_s=3.0, n_failed=0)
        self.assertLess(m["xy_drift_m"], 0.01)  # net start->end distance is ~0 although it ran in circles
        self.assertGreater(m["xy_max_dist_m"], 0.99)  # diameter
        self.assertAlmostEqual(m["xy_path_length_m"], radius * omega * 6.0, delta=0.02)
        self.assertAlmostEqual(m["yaw_total_deg"], 540.0, delta=0.5)
        self.assertEqual(m["heading_valid_fraction"], 1.0)
        self.assertEqual(len(m["series_0p1s"]["yaw_deg"]), 61)

    def test_standing_still(self):
        m = M.compute_metrics(make_log(301, lambda k: {}), DT, settle_s=3.0, n_failed=2)
        self.assertEqual(m["xy_max_dist_m"], 0.0)
        self.assertEqual(m["yaw_total_deg"], 0.0)
        self.assertEqual(m["early_terminations"], 2)
        self.assertEqual(m["height_m_mean"], 0.29)
        self.assertEqual(m["foot_z_FL_m_mean"], 0.023)

    def test_feet_asymmetry_uses_forward_coordinate(self):
        feet = standing_feet()
        feet[0][0] = 0.30  # FL is 0.1 m further forward than FR (0.2)
        feet[3][0] = -0.35  # RR is 0.15 m further back than RL (-0.2)
        m = M.compute_metrics(make_log(60, lambda k: {"feet": feet}), DT, settle_s=1.0, n_failed=0)
        self.assertAlmostEqual(m["series_0p1s"]["front_feet_asym_m"][-1], 0.1, places=6)
        self.assertAlmostEqual(m["series_0p1s"]["rear_feet_asym_m"][-1], 0.15, places=6)

    def test_legacy_keys_are_kept(self):
        m = M.compute_metrics(make_log(60, lambda k: {"nose_up_deg": 40.0}), DT, settle_s=1.0, n_failed=0)
        for key in ("nose_up_deg_mean", "roll_deg_mean", "height_m_mean", "foot_z_FL_m_mean", "foot_z_RR_m_mean",
                    "xy_drift_m", "early_terminations", "series_0p1s"):
            self.assertIn(key, m)
        for key in ("nose_up_deg", "roll_deg", "height_m", "foot_z_FL", "foot_z_FR", "foot_z_RL", "foot_z_RR"):
            self.assertIn(key, m["series_0p1s"])
        self.assertEqual(m["nose_up_deg_mean"], 40.0)

    def test_joint_angles_become_q_series(self):
        log = make_log(60, lambda k: {})
        log["dof_pos"] = [[0.1 * j + 0.01 * k for j in range(12)] for k in range(60)]
        m = M.compute_metrics(log, DT, settle_s=1.0, n_failed=0)
        series = m["series_0p1s"]
        self.assertEqual([k for k in series if k.startswith("q_")], [f"q_{i}" for i in range(12)])
        self.assertAlmostEqual(series["q_3"][0], 0.3, places=6)
        # logs without joint angles (older runs) simply have no q_* series
        self.assertFalse([k for k in M.compute_metrics(make_log(60, lambda k: {}), DT, 1.0, 0)["series_0p1s"] if k.startswith("q_")])

    def test_empty_log_is_rejected(self):
        with self.assertRaises(ValueError):
            M.compute_metrics(make_log(0, lambda k: {}), DT, settle_s=1.0, n_failed=0)


if __name__ == "__main__":
    unittest.main()
