"""LoopFuse — stop an agent that is going round in circles, BEFORE it pays for another lap.

Comes from Project 11 (production observability), where a `LoopRule` counted repeated tool
signatures across exported spans and raised an alert. That version was honest but late: by
the time the rule fired, every one of those calls had already been billed. This is the same
idea moved onto the hot path, plus the failure mode P11 could not see.

Two detectors, both deterministic:

  REPEAT  the same (tool, args) signature seen `repeat_threshold` times in one run.
          The classic stuck agent: it re-asks the identical question forever because the
          answer it got was not the answer it wanted.

  CYCLE   the last 2N signatures are two identical N-length blocks — A,B,A,B or
          A,B,C,A,B,C. This is the one P11 missed. No single signature repeats often
          enough to trip a counter, so a pure count-based rule stays silent while the
          agent ping-pongs between two tools until the step limit kills it. Cycles are
          checked longest-first so A,B,A,B is reported as a period-2 cycle, not as a
          degenerate period-1 one.

Why not just cap the steps? Because a step cap answers "has this gone on too long" — it
does not answer "is this making progress". A ten-step run that never repeats itself is
healthy; a four-step run that repeats twice is already dead and should stop at step 3, not
at step 25. Frameworks ship the step cap (LangGraph's `recursion_limit`, CrewAI's
`max_iter`, smolagents' `max_steps`) and none of them ship the progress check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core import ToolCall, Verdict

DEFAULT_REPEAT_THRESHOLD = 3
DEFAULT_MAX_CYCLE_LEN = 4

__all__ = ["LoopFuse", "LoopState"]


@dataclass
class LoopState:
    """Everything the fuse knows. Exposed so a caller can log or assert on it."""

    history: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    def record(self, signature: str) -> int:
        self.history.append(signature)
        self.counts[signature] = self.counts.get(signature, 0) + 1
        return self.counts[signature]


class LoopFuse:
    """Ask `check(call)` before running a tool; call `record(call)` after you run it.

    `check` is side-effect free, so a caller may ask twice and get the same answer. That
    matters for adapters that want to log a verdict and then act on it.
    """

    name = "loop"

    def __init__(
        self,
        repeat_threshold: int = DEFAULT_REPEAT_THRESHOLD,
        max_cycle_len: int = DEFAULT_MAX_CYCLE_LEN,
        detect_cycles: bool = True,
    ) -> None:
        if repeat_threshold < 2:
            raise ValueError("repeat_threshold must be >= 2 — a first call is never a loop")
        self.repeat_threshold = repeat_threshold
        self.max_cycle_len = max_cycle_len
        self.detect_cycles = detect_cycles
        self.state = LoopState()

    # -- detectors ---------------------------------------------------------- #
    def _repeat_verdict(self, signature: str) -> Verdict | None:
        # +1 because we are judging the call that has NOT been recorded yet.
        would_be = self.state.counts.get(signature, 0) + 1
        if would_be < self.repeat_threshold:
            return None
        return Verdict.stop(
            self.name,
            f"identical tool call would run {would_be}x in one request "
            f"(threshold {self.repeat_threshold}) — the agent is not making progress",
            evidence=f"signature `{signature}`",
        )

    def _cycle_verdict(self, signature: str) -> Verdict | None:
        if not self.detect_cycles:
            return None
        seq = self.state.history + [signature]
        # Longest period first: A,B,A,B is a period-2 cycle, and reporting it as period-1
        # would be wrong (no single signature repeats 2x in a row there).
        for period in range(min(self.max_cycle_len, len(seq) // 2), 1, -1):
            tail = seq[-2 * period:]
            if tail[:period] == tail[period:]:
                cycle = " -> ".join(s.split("#")[0] for s in tail[:period])
                return Verdict.stop(
                    self.name,
                    f"agent is cycling through {period} tool calls with no new state "
                    f"({cycle} -> repeat)",
                    evidence="last %d calls: %s" % (
                        2 * period, ", ".join(s.split("#")[0] for s in tail)),
                )
        return None

    # -- public API --------------------------------------------------------- #
    def check(self, call: ToolCall) -> Verdict:
        sig = call.signature
        # Repeat first: it is the cheaper and more specific explanation of the two.
        for verdict in (self._repeat_verdict(sig), self._cycle_verdict(sig)):
            if verdict is not None:
                return verdict
        return Verdict.ok(self.name, f"call {len(self.state.history) + 1} is new state")

    def record(self, call: ToolCall) -> None:
        self.state.record(call.signature)

    def reset(self) -> None:
        """New request, new fuse. Loop state must never leak between requests."""
        self.state = LoopState()

    @classmethod
    def from_history(cls, signatures, **kwargs) -> "LoopFuse":
        """Rebuild a fuse from signatures that have already run.

        For frameworks where the guard is a stateless per-call hook rather than an object
        that lives for the request — LangGraph's `wrap_tool_call` is one — the honest
        thing is to derive the history from the transcript every time. A node instance is
        shared across requests and across threads; a counter stored on it would leak one
        user's loop into another user's run, which is a worse bug than the one being
        fixed.
        """
        fuse = cls(**kwargs)
        for sig in signatures:
            fuse.state.record(sig)
        return fuse
