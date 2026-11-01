"""Parse LLM-written reward modules and check them on a fake batch before spending GPU time."""

import importlib.util
import math
import re

import torch

from .task_env import TaskState


def parse_reply(text: str) -> tuple[str, str]:
    """Return (summary, code) from a reply formatted as '要点: ...' + a python code block."""
    m = re.search(r"```python\n(.*?)```", text, re.S)
    if not m:
        raise ValueError("python code block not found in the reply")
    s = re.search(r"要点[:：]\s*(.+)", text)
    return (s.group(1).strip() if s else ""), m.group(1).strip() + "\n"


def fake_state(n: int = 8) -> TaskState:
    q = torch.randn(n, 4)
    q = q / q.norm(dim=1, keepdim=True)
    g = torch.randn(n, 3)
    return TaskState(
        base_pos=torch.rand(n, 3) * 0.5,
        base_quat=q,
        lin_vel_b=torch.randn(n, 3),
        ang_vel_b=torch.randn(n, 3),
        gravity_b=g / g.norm(dim=1, keepdim=True),
        dof_pos=torch.randn(n, 12),
        dof_vel=torch.randn(n, 12),
        default_dof_pos=torch.zeros(12),
        feet_pos=torch.rand(n, 4, 3) * 0.3,
        knee_pos=torch.rand(n, 4, 3) * 0.5,
        head_pos=torch.rand(n, 3) * 0.5,
        actions=torch.randn(n, 12),
        last_actions=torch.randn(n, 12),
        episode_time=torch.rand(n) * 10,
    )


def load_and_check(path: str):
    """Import the module at `path` and run compute/failed on a fake batch. Raises with a readable message."""
    spec = importlib.util.spec_from_file_location("candidate_reward", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("PARAMS", "compute", "failed"):
        if not hasattr(mod, name):
            raise ValueError(f"module must define {name}")
    n = 8
    s = fake_state(n)
    terms = mod.compute(s, mod.PARAMS)
    if not isinstance(terms, dict) or not terms:
        raise ValueError("compute must return a non-empty dict of tensors")
    for k, v in terms.items():
        if not isinstance(v, torch.Tensor) or v.shape != (n,):
            raise ValueError(f"term '{k}' must be a tensor of shape (N,), got {getattr(v, 'shape', type(v))}")
        if not torch.isfinite(v).all():
            raise ValueError(f"term '{k}' has NaN/inf on random states")
    f = mod.failed(s, mod.PARAMS)
    if not isinstance(f, torch.Tensor) or f.shape != (n,) or f.dtype != torch.bool:
        raise ValueError(f"failed must return a bool tensor of shape (N,), got {getattr(f, 'dtype', type(f))}")
    for k, v in mod.PARAMS.items():
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"PARAMS['{k}'] is not finite")
    return mod
