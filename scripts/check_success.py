"""Judge a recorded run against a frozen success spec, without a GPU.

Usage (from the repo root):
    python scripts/check_success.py demo_sit_L2 --spec configs/tasks/sit.success.yaml [--ckpt 299]
Reads runs/<exp_name>/metrics_<ckpt>.json, prints the result and writes runs/<exp_name>/success_<ckpt>.json.
The spec must be approved (`approved: true`). `--allow-unapproved` is only for verifying a draft spec on past runs.
Exit code: 0 = all criteria passed, 1 = at least one failed, 2 = spec/input problem.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pai_loop import success  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("exp_name")
parser.add_argument("--spec", required=True)
parser.add_argument("--ckpt", type=int, help="default: last checkpoint in summary.json")
parser.add_argument("--allow-unapproved", action="store_true", help="verification only; never use this in the loop")
args = parser.parse_args()


def fail(message):
    """Cannot judge (bad spec or missing input): exit code 2, distinct from 'judged and failed' (1)."""
    print(message, file=sys.stderr)
    sys.exit(2)


try:
    spec, _, sha = success.load_spec(args.spec, require_approved=not args.allow_unapproved)
except (success.SpecError, OSError) as e:
    fail(f"spec error: {e}")
if args.allow_unapproved and not spec["approved"]:
    print("注意: 未承認の条件で評価しています（検証用）。この結果は実験の成否には使えません。")

run_dir = os.path.join("runs", args.exp_name)
ckpt = args.ckpt
if ckpt is None:
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(summary_path):
        fail(f"{summary_path} not found; pass --ckpt or train {args.exp_name} first")
    with open(summary_path) as f:
        ckpt = json.load(f)["last_checkpoint"]
metrics_path = os.path.join(run_dir, f"metrics_{ckpt}.json")
if not os.path.exists(metrics_path):
    fail(f"{metrics_path} not found; run scripts/record.py {args.exp_name} --ckpt {ckpt} first")
with open(metrics_path) as f:
    metrics = json.load(f)

result = success.evaluate(metrics, spec)
result.update(exp_name=args.exp_name, ckpt=ckpt, spec_path=args.spec, spec_sha256=sha)
out = os.path.join(run_dir, f"success_{ckpt}.json")
with open(out, "w") as f:
    json.dump(result, f, indent=1, ensure_ascii=False)
print(success.format_result(result))
print(f"-> {out}")
sys.exit(0 if result["passed"] else 1)
