"""JudgeGate — co-gate an LLM's self-assessment with deterministic hard checks.

Project 10 built an LLM-as-judge that graded the agent's own output against a rubric, and
caught it red-handed: an 8B judge scored a draft 1.00, rating a criterion 5/5 on text that
did not mention the thing the criterion asked for. A plain-Python hard check ("does the
answer contain the 30% figure?") said no, held the gate at FAIL, and the refine step then
actually fixed the answer.

So the rule this library ships is: **an LLM score can only lower the gate, never raise it.**

    PASS  requires  llm_score >= threshold  AND  every required hard check green.
    FAIL  otherwise, with the failing checks named.

That asymmetry is the entire contribution. A self-evaluating agent with no deterministic
anchor converges on flattering itself, and the failure is silent — the run looks like a
success, the metric looks green, and the output is wrong. One cheap `in`-check on the
answer string costs nothing and cannot be talked round.

Hard checks are plain predicates over the answer text. Keep them boring: a substring, a
regex, a JSON parse, a compile(), a unit-test invocation. A hard check that itself needs a
model is not a hard check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

__all__ = ["HardCheck", "CheckResult", "GateVerdict", "JudgeGate"]


@dataclass(frozen=True)
class HardCheck:
    """`check(answer) -> bool`. Required checks gate; advisory ones only report."""

    id: str
    description: str
    check: Callable[[str], bool]
    required: bool = True

    def run(self, answer: str) -> "CheckResult":
        # A check that explodes counts as FAILED, never as passed. A guard that fails open
        # is not a guard.
        try:
            passed = bool(self.check(answer))
            error = ""
        except Exception as exc:  # noqa: BLE001
            passed, error = False, f"{type(exc).__name__}: {exc}"
        return CheckResult(id=self.id, description=self.description, passed=passed,
                           required=self.required, error=error)


@dataclass(frozen=True)
class CheckResult:
    id: str
    description: str
    passed: bool
    required: bool = True
    error: str = ""


@dataclass
class GateVerdict:
    passed: bool
    llm_score: float
    threshold: float
    results: list[CheckResult] = field(default_factory=list)
    reason: str = ""

    @property
    def failed_required(self) -> list[CheckResult]:
        return [r for r in self.results if r.required and not r.passed]

    @property
    def hard_score(self) -> float:
        req = [r for r in self.results if r.required]
        return round(sum(r.passed for r in req) / len(req), 4) if req else 1.0

    def line(self) -> str:
        req = [r for r in self.results if r.required]
        return (f"{'PASS' if self.passed else 'FAIL'} "
                f"llm={self.llm_score:.2f}/{self.threshold:.2f} "
                f"hard={sum(r.passed for r in req)}/{len(req)} — {self.reason}")

    def critique_note(self) -> str:
        """Feed the failures back to the refiner as concrete, non-negotiable instructions."""
        if self.passed:
            return ""
        bits = [f"- MISSING: {r.description}" + (f" (check error: {r.error})" if r.error else "")
                for r in self.failed_required]
        if self.llm_score < self.threshold:
            bits.append(f"- The rubric score {self.llm_score:.2f} is below the required "
                        f"{self.threshold:.2f}.")
        return "The answer did not pass the gate. Fix exactly these:\n" + "\n".join(bits)


class JudgeGate:
    """Combine a model's self-score with hard checks. The hard checks always win."""

    name = "judge-gate"

    def __init__(self, hard_checks: Iterable[HardCheck] = (), threshold: float = 0.80) -> None:
        self.hard_checks = list(hard_checks)
        self.threshold = threshold

    def evaluate(self, answer: str, llm_score: float) -> GateVerdict:
        results = [hc.run(answer) for hc in self.hard_checks]
        failed = [r for r in results if r.required and not r.passed]
        score_ok = llm_score >= self.threshold

        if failed:
            reason = ("deterministic hard check(s) failed: "
                      + ", ".join(r.id for r in failed)
                      + ("; the model scored this a PASS anyway"
                         if score_ok else ""))
            return GateVerdict(False, llm_score, self.threshold, results, reason)
        if not score_ok:
            return GateVerdict(False, llm_score, self.threshold, results,
                               f"self-score {llm_score:.2f} below threshold "
                               f"{self.threshold:.2f}")
        return GateVerdict(True, llm_score, self.threshold, results,
                           "self-score and every hard check agree")
