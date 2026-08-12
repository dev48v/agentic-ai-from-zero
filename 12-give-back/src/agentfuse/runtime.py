"""FuseBox — the fuses wired together into one thing an agent loop can ask.

An agent loop should not have to know how many guards exist. It asks two questions:

    box.check_tool_call(call)   -> may I run this tool?
    box.preflight_llm(text, n)  -> may I make this model call?

and reports two facts back:

    box.record_tool_call(call)
    box.record_llm(response)

Order of evaluation is deliberate and fixed: **permission, then loop, then budget.**

  * Permission first because it is the only judgement that does not depend on history. A
    tool the run was never allowed to touch should be refused on attempt 1, not on
    attempt 3 once a counter agrees.
  * Loop before budget because "the agent is stuck" is a more useful thing to tell an
    operator than "the agent ran out of money", even though both are true by then. The
    budget ceiling is the backstop for the failure you did not anticipate.

Every verdict, allow or block, goes into `box.log`. The whole argument of this series is
that an agent's failures are invisible unless you write them down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .budget import BudgetFuse
from .core import ToolCall, Verdict
from .loops import LoopFuse
from .permissions import PermissionFuse

__all__ = ["FuseBox"]


@dataclass
class FuseBox:
    loop: LoopFuse | None = None
    permission: PermissionFuse | None = None
    budget: BudgetFuse | None = None

    log: list[Verdict] = field(default_factory=list)

    def _remember(self, verdict: Verdict) -> Verdict:
        self.log.append(verdict)
        return verdict

    # -- tool side ---------------------------------------------------------- #
    def check_tool_call(self, call: ToolCall) -> Verdict:
        for fuse in (self.permission, self.loop):
            if fuse is None:
                continue
            verdict = fuse.check(call)
            if verdict.blocked:
                return self._remember(verdict)
        return self._remember(Verdict.ok("fusebox", f"tool '{call.name}' cleared"))

    def record_tool_call(self, call: ToolCall) -> None:
        if self.loop is not None:
            self.loop.record(call)

    # -- model side --------------------------------------------------------- #
    def preflight_llm(self, prompt_text: str = "", max_completion_tokens: int = 0,
                      prompt_tokens: int | None = None) -> Verdict:
        if self.budget is None:
            return self._remember(Verdict.ok("fusebox", "no budget fuse configured"))
        return self._remember(
            self.budget.preflight(prompt_text, max_completion_tokens, prompt_tokens))

    def record_llm(self, response) -> float:
        return self.budget.record_response(response) if self.budget else 0.0

    # -- housekeeping ------------------------------------------------------- #
    def reset(self) -> None:
        """Between requests. Loop state and budget are PER REQUEST; permissions are not."""
        if self.loop is not None:
            self.loop.reset()
        if self.budget is not None:
            self.budget.reset()
        self.log.clear()

    @property
    def blocks(self) -> list[Verdict]:
        return [v for v in self.log if v.blocked]

    def report(self) -> str:
        lines = [v.line() for v in self.log]
        if self.budget is not None:
            s = self.budget.summary()
            lines.append(f"spend: ${s['spent_usd']:.6f} over {s['calls']} model call(s), "
                         f"{s['total_tokens']} tokens")
        return "\n".join(lines)
