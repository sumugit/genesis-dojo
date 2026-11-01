import json
import os
import tempfile
import unittest

from pai_loop import prior


def entry(name, code="x = 1", n=(5, 8), score=3, success=False, error=None):
    return {"exp_name": name, "reward_code": code, "success_check": {"n_passed": n[0], "n_criteria": n[1]},
            "evaluation": {"score": score, "success": success, "observations": "走り続けている"},
            "decision": {"success": success}, "error": error}


class PriorTest(unittest.TestCase):
    def test_best_loop_wins_and_later_loops_break_ties(self):
        es = [entry("a_L1", n=(5, 8), score=1), entry("a_L2", n=(8, 8), score=2), entry("a_L3", n=(8, 8), score=2)]
        self.assertEqual(prior.pick_prior(es)["exp_name"], "a_L3")
        es.append(entry("a_L4", n=(1, 8), score=0))
        self.assertEqual(prior.pick_prior(es)["exp_name"], "a_L3")
        self.assertEqual(prior.pick_prior([entry("b_L1", success=True, n=(8, 8), score=9), entry("b_L2", n=(8, 8), score=8)])["exp_name"], "b_L1")

    def test_loops_without_code_or_with_errors_are_never_picked(self):
        self.assertIsNone(prior.pick_prior([entry("c_L1", code=None), entry("c_L2", error="training failed")]))
        self.assertIsNone(prior.pick_prior([]))

    def test_load_entries_skips_bad_lines(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "logs", "loops"))
            with open(os.path.join(d, "logs", "loops", "e.jsonl"), "w") as f:
                f.write(json.dumps(entry("e_L1")) + "\n{broken\n" + json.dumps(entry("e_L2")) + "\n")
            self.assertEqual([e["exp_name"] for e in prior.load_entries(d, "e")], ["e_L1", "e_L2"])
            self.assertEqual(prior.load_entries(d, "nope"), [])

    def test_section_carries_code_and_findings_but_marks_the_old_spec(self):
        e = entry("jm_L3", code="def compute(s, p):\n    return {}")
        e["train_terms"] = {"alive": 0.5}
        text = prior.prior_section(e, "jm")
        self.assertIn("def compute(s, p)", text)
        self.assertIn("走り続けている", text)
        self.assertIn("古い成功条件", text)
        self.assertIn('"alive": 0.5', text)


if __name__ == "__main__":
    unittest.main()
