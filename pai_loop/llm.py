"""LLM/VLM client behind a small interface so the backend can be swapped (claude -p now, API later)."""

import base64
import json
import mimetypes
import subprocess
import time
from dataclasses import dataclass


class LLMError(RuntimeError):
    pass


class LLMQuotaError(LLMError):
    """Usage limit reached; the loop should stop rather than retry."""


@dataclass
class LLMReply:
    text: str
    seconds: float
    cost_usd: float | None  # as reported by the backend (claude -p reports an API-equivalent cost)


class ClaudeCLI:
    """Calls `claude -p` with stream-json input so images can be passed as content blocks.

    --safe-mode skips CLAUDE.md/skills/plugins/MCP, and --tools "" disables tools, so the
    model only sees our system prompt, text and images.
    """

    QUOTA_MARKERS = ("usage limit", "rate limit", "limit reached", "credit balance")

    def __init__(self, model: str | None = None, timeout_s: int = 600):
        self.model = model
        self.timeout_s = timeout_s

    def ask(self, text: str, images: list[str] = (), system: str | None = None) -> LLMReply:
        content = []
        for path in images:
            media_type = mimetypes.guess_type(path)[0] or "image/png"
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        content.append({"type": "text", "text": text})
        msg = json.dumps({"type": "user", "message": {"role": "user", "content": content}}, ensure_ascii=False)

        cmd = ["claude", "-p", "--safe-mode", "--tools", "", "--no-session-persistence",
               "--input-format", "stream-json", "--output-format", "stream-json", "--verbose"]
        if system:
            cmd += ["--system-prompt", system]
        if self.model:
            cmd += ["--model", self.model]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, input=msg, capture_output=True, text=True, timeout=self.timeout_s)
        except subprocess.TimeoutExpired as e:
            raise LLMError(f"claude -p timed out after {self.timeout_s}s") from e
        seconds = time.perf_counter() - t0

        result = None
        for line in proc.stdout.splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "result":
                result = d
        if result is None or result.get("is_error"):
            detail = (result or {}).get("result") or proc.stderr[-500:] or proc.stdout[-500:]
            if any(m in str(detail).lower() for m in self.QUOTA_MARKERS):
                raise LLMQuotaError(detail)
            raise LLMError(f"claude -p failed (exit {proc.returncode}): {detail}")
        return LLMReply(text=result["result"], seconds=round(seconds, 1), cost_usd=result.get("total_cost_usd"))


def make_llm(name: str = "claude-cli", **kwargs):
    if name == "claude-cli":
        return ClaudeCLI(**kwargs)
    # "anthropic-api" goes here when the claude -p quota becomes a bottleneck.
    raise ValueError(f"unknown LLM backend: {name}")
