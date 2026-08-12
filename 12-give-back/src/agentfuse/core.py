"""Core types every fuse shares: the call being judged, and the verdict it gets.

A "fuse" here is the electrical kind. It sits in the path, it is dumber than the thing
it protects, and it blows BEFORE the expensive part burns. That is the whole doctrine
this library carries over from projects 1-11 of the series: the model reasons, plain
deterministic code enforces.

Two rules hold for everything in this package:

1. **A verdict is a pure function of recorded facts.** Same history in, same verdict out.
   A guard you cannot replay is a guard you cannot debug at 3am.
2. **Deny/blow is the default on ambiguity.** An unknown tool, a missing price, an
   unparseable argument — every one of those resolves to "stop", never to "probably fine".

There are no third-party imports anywhere in `agentfuse`. A safety layer that drags in a
dependency tree is a safety layer nobody installs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

# Verdict decisions.
ALLOW = "allow"
BLOCK = "block"

__all__ = ["ALLOW", "BLOCK", "ToolCall", "Verdict", "FuseError", "canonical_signature"]


class FuseError(RuntimeError):
    """Raised by adapters when a fuse blows and the caller asked for exceptions.

    Carries the verdict so a handler can log the reason rather than a bare string.
    """

    def __init__(self, verdict: "Verdict") -> None:
        super().__init__(verdict.reason)
        self.verdict = verdict


def canonical_signature(name: str, args: dict | None) -> str:
    """Stable identity for "this exact call".

    `json.dumps(..., sort_keys=True)` so `{"a":1,"b":2}` and `{"b":2,"a":1}` are the same
    call — an agent that re-asks the same question with the keys in a different order is
    still stuck. `default=str` keeps unserialisable args from crashing the guard: a fuse
    that raises on weird input is worse than no fuse.
    """
    try:
        blob = json.dumps(args or {}, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:  # noqa: BLE001 — never let signing break the protected call path
        blob = repr(args)
    digest = hashlib.sha256(f"{name}\x00{blob}".encode("utf-8")).hexdigest()[:16]
    return f"{name}#{digest}"


@dataclass(frozen=True)
class ToolCall:
    """One requested tool invocation, before it has run.

    `id` is whatever the provider gave it (OpenAI `tool_call_id`, LangGraph call id); it is
    carried for correlation only and is deliberately NOT part of the signature — two
    identical calls with different ids are the same call.
    """

    name: str
    args: dict = field(default_factory=dict)
    id: str = ""

    @property
    def signature(self) -> str:
        return canonical_signature(self.name, self.args)


@dataclass(frozen=True)
class Verdict:
    """The answer a fuse gives. `allowed` is the only field a caller must respect."""

    fuse: str
    decision: str
    reason: str = ""
    evidence: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision == ALLOW

    @property
    def blocked(self) -> bool:
        return self.decision == BLOCK

    def line(self) -> str:
        mark = "ALLOW" if self.allowed else "BLOCK"
        tail = f" | {self.evidence}" if self.evidence else ""
        return f"{mark:<5} [{self.fuse}] {self.reason}{tail}"

    @classmethod
    def ok(cls, fuse: str, reason: str = "", evidence: str = "") -> "Verdict":
        return cls(fuse=fuse, decision=ALLOW, reason=reason, evidence=evidence)

    @classmethod
    def stop(cls, fuse: str, reason: str, evidence: str = "") -> "Verdict":
        return cls(fuse=fuse, decision=BLOCK, reason=reason, evidence=evidence)
