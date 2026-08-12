"""CanaryFuse — deterministic traffic split plus a gated promote/rollback decision.

Straight out of Project 11, generalised so it works for any agent config change, not just a
model swap: a new system prompt, a new tool set, a bigger context window, a different
temperature. All of those are releases, and none of them are covered by your web tier's
deploy pipeline because nothing about the binary changed.

Two halves:

  ROUTING    `route(request_id)` hashes the id and compares it to the canary percentage.
             Hash, not `random()`, for two reasons: the same request always lands on the
             same version so a user gets a consistent experience across a retry, and a run
             replays identically, so "the canary looked fine on my machine" becomes a
             checkable claim rather than an anecdote.

  GATES      `evaluate(baseline, candidate)` compares MEASURED stats through named
             thresholds — added error rate, p95 latency ratio, cost-per-request ratio —
             and returns rollback if any gate fails. In P11's recorded run it was the cost
             gate that fired: the candidate was 2.20x the baseline's cost per request and
             was pulled automatically.

Cost is the gate agent releases actually need and web releases mostly do not. A prompt
change that adds one extra reasoning step will not move your error rate and will barely
move p95, and it can still double the bill. That is why cost is a first-class gate here
rather than something you notice on an invoice at the end of the month.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

__all__ = ["VersionStats", "Gate", "CanaryVerdict", "CanaryFuse", "percentile"]


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile: the smallest observed value at or above pct% of the data.

    Nearest-rank rather than interpolation, and `math.ceil` rather than `round`, for the
    same reason Project 11 chose it: the answer is always a latency somebody really
    experienced, and it never drifts with float rounding. Same numbers in, same p95 out.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100.0 * len(ordered)))
    return round(ordered[min(rank, len(ordered)) - 1], 3)


@dataclass
class VersionStats:
    """Measured behaviour of one version. Feed it, do not compute it from estimates."""

    version: str
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0
    cost_usd: float = 0.0

    def record(self, latency_ms: float, cost_usd: float = 0.0, error: bool = False) -> None:
        self.latencies_ms.append(float(latency_ms))
        self.cost_usd = round(self.cost_usd + float(cost_usd), 8)
        self.errors += int(bool(error))

    @property
    def count(self) -> int:
        return len(self.latencies_ms)

    @property
    def error_rate(self) -> float:
        return round(self.errors / self.count, 4) if self.count else 0.0

    @property
    def p50(self) -> float:
        return percentile(self.latencies_ms, 50)

    @property
    def p95(self) -> float:
        return percentile(self.latencies_ms, 95)

    @property
    def cost_per_request(self) -> float:
        return round(self.cost_usd / self.count, 8) if self.count else 0.0


@dataclass
class Gate:
    name: str
    ok: bool
    detail: str


@dataclass
class CanaryVerdict:
    decision: str                        # "promote" | "rollback"
    gates: list[Gate] = field(default_factory=list)
    active_version_after: str = ""

    @property
    def failed(self) -> list[Gate]:
        return [g for g in self.gates if not g.ok]

    @property
    def rolled_back(self) -> bool:
        return self.decision == "rollback"


class CanaryFuse:
    name = "canary"

    def __init__(
        self,
        stable: str,
        candidate: str,
        candidate_pct: int,
        max_error_rate_delta: float = 0.05,
        max_p95_ratio: float = 1.50,
        max_cost_ratio: float = 1.30,
        min_candidate_requests: int = 3,
    ) -> None:
        if not 0 <= candidate_pct <= 100:
            raise ValueError("candidate_pct must be between 0 and 100")
        self.stable = stable
        self.candidate = candidate
        self.candidate_pct = candidate_pct
        self.max_error_rate_delta = max_error_rate_delta
        self.max_p95_ratio = max_p95_ratio
        self.max_cost_ratio = max_cost_ratio
        self.min_candidate_requests = min_candidate_requests
        self.active = stable
        self.events: list[str] = [
            f"release: {candidate} to {candidate_pct}% of traffic, {stable} serving the rest"]

    def route(self, request_id: str) -> str:
        bucket = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16) % 100
        return self.candidate if bucket < self.candidate_pct else self.stable

    def evaluate(self, baseline: VersionStats, candidate: VersionStats) -> CanaryVerdict:
        gates: list[Gate] = []

        # Not enough evidence is a rollback, not a promote. "We did not see a problem"
        # from four requests is not the same statement as "there is no problem".
        if candidate.count < self.min_candidate_requests:
            gates.append(Gate("sample-size", False,
                              f"candidate served {candidate.count} request(s), "
                              f"need >= {self.min_candidate_requests} to judge"))
            return self._finish(CanaryVerdict("rollback", gates))
        if baseline.count == 0:
            gates.append(Gate("baseline", False, "no stable baseline to compare against"))
            return self._finish(CanaryVerdict("rollback", gates))

        d_err = round(candidate.error_rate - baseline.error_rate, 4)
        gates.append(Gate("error-rate", d_err <= self.max_error_rate_delta,
                          f"candidate {candidate.error_rate:.1%} vs stable "
                          f"{baseline.error_rate:.1%} (delta {d_err:+.1%}, "
                          f"max +{self.max_error_rate_delta:.0%})"))

        p95_ratio = round(candidate.p95 / baseline.p95, 3) if baseline.p95 else float("inf")
        gates.append(Gate("latency-p95", p95_ratio <= self.max_p95_ratio,
                          f"candidate p95 {candidate.p95:.0f}ms vs stable "
                          f"{baseline.p95:.0f}ms ({p95_ratio:.2f}x, "
                          f"max {self.max_p95_ratio:.2f}x)"))

        cost_ratio = (round(candidate.cost_per_request / baseline.cost_per_request, 3)
                      if baseline.cost_per_request else float("inf"))
        gates.append(Gate("cost-per-request", cost_ratio <= self.max_cost_ratio,
                          f"candidate ${candidate.cost_per_request:.6f} vs stable "
                          f"${baseline.cost_per_request:.6f} ({cost_ratio:.2f}x, "
                          f"max {self.max_cost_ratio:.2f}x)"))

        decision = "promote" if all(g.ok for g in gates) else "rollback"
        return self._finish(CanaryVerdict(decision, gates))

    def _finish(self, verdict: CanaryVerdict) -> CanaryVerdict:
        if verdict.rolled_back:
            self.active = self.stable
            why = ", ".join(g.name for g in verdict.failed) or "gate failure"
            self.events.append(f"ROLLBACK: {self.candidate} pulled ({why}); "
                               f"100% back on {self.stable}")
        else:
            self.active = self.candidate
            self.events.append(f"PROMOTE: {self.candidate} passed every gate; 100% moved")
        verdict.active_version_after = self.active
        return verdict
