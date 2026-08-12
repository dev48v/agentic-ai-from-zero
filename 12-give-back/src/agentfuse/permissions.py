"""PermissionFuse — deny-by-default tool scoping.

Lifted from Project 4 (multi-tool orchestrator), where a tool declared a single permission
scope and the orchestrator refused to run it unless the run had been granted that scope.
In that project the restricted grant `{network, read}` refused `ledger_write`, and the
model was then TOLD it was refused so it could answer honestly instead of pretending.

Two things are generalised here:

  * a tool may declare SEVERAL scopes (`{"write", "spend_money"}`) and needs all of them;
  * a tool the registry has never heard of is DENIED, not run. Prompt injection does not
    have to invent a clever argument if it can simply name a tool you forgot to list.

That second rule is the whole point of "deny-by-default", and it is what a plain
`{name: fn}` dispatch table cannot give you: a dict lookup either finds a function and
runs it or raises KeyError somewhere deep in the executor. Neither is a policy decision.

Scopes are free-form strings. This library ships no opinion about your taxonomy; the
series used `read` / `write` / `network` / `spend_money` / `external_write` and those are
offered as constants for convenience, not as a closed set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .core import ToolCall, Verdict

# Suggested (not enforced) scope vocabulary, carried over from projects 4 and 6.
READ = "read"
WRITE = "write"
NETWORK = "network"
SPEND_MONEY = "spend_money"
EXTERNAL_WRITE = "external_write"

__all__ = ["ToolSpec", "PermissionFuse", "READ", "WRITE", "NETWORK", "SPEND_MONEY",
           "EXTERNAL_WRITE"]


@dataclass(frozen=True)
class ToolSpec:
    """What a tool is allowed to need. `scopes` is the set the run must hold ALL of."""

    name: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    description: str = ""

    @classmethod
    def of(cls, name: str, *scopes: str, description: str = "") -> "ToolSpec":
        return cls(name=name, scopes=frozenset(scopes), description=description)


class PermissionFuse:
    """`check(call)` returns ALLOW only when the tool is known and every scope is granted."""

    name = "permission"

    def __init__(
        self,
        granted: Iterable[str],
        specs: Iterable[ToolSpec] | Mapping[str, ToolSpec],
        allow_unknown_tools: bool = False,
    ) -> None:
        self.granted = frozenset(granted)
        if isinstance(specs, Mapping):
            self.specs = dict(specs)
        else:
            self.specs = {s.name: s for s in specs}
        # Kept as an explicit opt-out rather than a silent default, so switching it on is
        # a decision somebody has to write down in code review.
        self.allow_unknown_tools = allow_unknown_tools
        self.denials: list[Verdict] = []

    def check(self, call: ToolCall) -> Verdict:
        spec = self.specs.get(call.name)
        if spec is None:
            if not self.allow_unknown_tools:
                return self._deny(
                    f"tool '{call.name}' is not in the registry — deny-by-default",
                    evidence=f"known tools: {sorted(self.specs) or '(none)'}",
                )
            return Verdict.ok(self.name, f"unknown tool '{call.name}' allowed by config")

        missing = sorted(spec.scopes - self.granted)
        if missing:
            return self._deny(
                f"tool '{call.name}' needs scope(s) {missing} not granted to this run",
                evidence=f"granted: {sorted(self.granted) or '(none)'}",
            )
        return Verdict.ok(
            self.name,
            f"tool '{call.name}' within granted scope",
            evidence=f"needs {sorted(spec.scopes) or '(none)'}",
        )

    def _deny(self, reason: str, evidence: str = "") -> Verdict:
        verdict = Verdict.stop(self.name, reason, evidence)
        self.denials.append(verdict)
        return verdict

    def refusal_note(self) -> str:
        """A short block to paste back into the model's context.

        Project 6 established this: a refused agent that is not told it was refused will
        report success it never had. Telling it produces an honest "I could not do that".
        """
        if not self.denials:
            return ""
        lines = [f"- {v.reason}" for v in self.denials]
        return ("The following actions were REFUSED by the permission layer and did NOT "
                "happen:\n" + "\n".join(lines) +
                "\nSay plainly that they could not be performed, and why. Never claim a "
                "refused action succeeded.")
