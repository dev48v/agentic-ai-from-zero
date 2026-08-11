"""ALERTING and the CANARY / ROLLBACK controller — the two things that turn traces
into an operational safety net. Deterministic Python, no model calls.

Tracing tells you what happened. This file decides what to *do* about it:

  3. ALERT ON LOOPS + FAILURES -> `AlertEngine` runs a fixed set of rules over the spans:
       LoopRule         a trace repeats the SAME tool with the SAME arguments N+ times.
                        This is the failure mode that quietly bankrupts an agent: it does
                        not crash, it does not return junk, it just goes round and round
                        burning tokens. A signature count over a threshold catches it
                        without any heuristics about "meaning".
       FailureRateRule  error spans / total spans over a threshold in this window.
       LatencySLORule   p95 of the request span over the SLO.
       CostBudgetRule   total spend over the window budget.

  4. CANARY + ROLLBACK -> `CanaryController`: deterministically routes a slice of traffic
       to a new version (hash the request id — same id always lands on the same version,
       so a run is reproducible and a user has a stable experience), then compares the
       canary's measured stats against the stable baseline through explicit GATES. Any
       failed gate, or any critical alert attributed to the canary, and the verdict is
       ROLLBACK — the active version flips back and the reason is recorded.

Every threshold is a named constant and every verdict is a pure function of the spans.
An on-call decision you cannot re-derive from the data is not an on-call decision.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from telemetry import ERROR, Span, Stats, percentile

# --------------------------------------------------------------------------- #
# Thresholds — every one of them named, in one place, tunable without touching logic.
# --------------------------------------------------------------------------- #
LOOP_REPEAT_THRESHOLD = 3        # same tool + same args this many times in one trace
FAILURE_RATE_THRESHOLD = 0.10    # 10% of a version's requests in error
LATENCY_SLO_P95_MS = 8000.0      # p95 of a whole request; set just above the measured
                                 # healthy baseline, which is the only honest way to pick one
COST_BUDGET_USD = 0.01           # spend budget for the window

CRITICAL = "CRITICAL"
WARNING = "WARNING"


@dataclass
class Alert:
    rule: str
    severity: str
    message: str
    evidence: str = ""
    version: str = ""            # which deployed version this is attributed to, if any

    def line(self) -> str:
        who = f" [{self.version}]" if self.version else ""
        return f"{self.severity:<8} {self.rule:<16}{who} {self.message}"


# --------------------------------------------------------------------------- #
# Rules. Each one: spans in, alerts out. Pure.
# --------------------------------------------------------------------------- #
def _version_of(spans: list[Span], trace_id: str) -> str:
    for sp in spans:
        if sp.trace_id == trace_id and sp.parent_span_id is None:
            return str(sp.attributes.get("deploy.version", ""))
    return ""


class LoopRule:
    """The one that matters most. A tool called with identical arguments again and again
    inside a single request means the agent is stuck: it is not making progress, it is
    re-asking. Signature = tool name + serialised args, counted per trace."""

    name = "loop-detected"

    def __init__(self, threshold: int = LOOP_REPEAT_THRESHOLD) -> None:
        self.threshold = threshold

    def evaluate(self, spans: list[Span]) -> list[Alert]:
        counts: dict[tuple[str, str], int] = {}
        for sp in spans:
            if sp.kind != "tool":
                continue
            sig = str(sp.attributes.get("tool.signature", sp.name))
            counts[(sp.trace_id, sig)] = counts.get((sp.trace_id, sig), 0) + 1
        alerts = []
        for (trace_id, sig), n in sorted(counts.items()):
            if n >= self.threshold:
                alerts.append(Alert(
                    rule=self.name, severity=CRITICAL,
                    message=f"agent repeated an identical tool call {n}x in one request "
                            f"(threshold {self.threshold}) — not making progress",
                    evidence=f"trace {trace_id[:12]}… signature `{sig}`",
                    version=_version_of(spans, trace_id)))
        return alerts


class FailureRateRule:
    """Measured over REQUESTS (root spans), PER VERSION.

    Two deliberate choices. Requests, not spans: the number an SLO is written against is
    "what fraction of customers got a broken answer", and a request with five healthy
    sub-spans and one dead tool call is still a broken answer — averaging it across all
    spans dilutes it into silence. Per version, not globally: pooling the versions hides a
    canary that is failing behind a healthy majority, and it also blames the canary for the
    baseline's failures. Alerting on a blended number is how a bad release survives.
    """

    name = "failure-rate"

    def __init__(self, threshold: float = FAILURE_RATE_THRESHOLD) -> None:
        self.threshold = threshold

    def evaluate(self, spans: list[Span]) -> list[Alert]:
        roots = [s for s in spans if s.parent_span_id is None]
        by_version: dict[str, list[Span]] = {}
        for sp in roots:
            by_version.setdefault(str(sp.attributes.get("deploy.version", "unknown")),
                                  []).append(sp)
        alerts = []
        for version, reqs in sorted(by_version.items()):
            errs = [s for s in reqs if s.status == ERROR]
            rate = len(errs) / len(reqs)
            if rate <= self.threshold:
                continue
            causes = sorted({s.error_message for s in errs if s.error_message})
            alerts.append(Alert(
                rule=self.name, severity=WARNING,
                message=f"request failure rate {rate:.1%} over threshold "
                        f"{self.threshold:.0%} ({len(errs)}/{len(reqs)} requests failed)",
                evidence="; ".join(causes)[:160] or "no error message recorded",
                version=version))
        return alerts


class LatencySLORule:
    name = "latency-slo"

    def __init__(self, p95_ms: float = LATENCY_SLO_P95_MS) -> None:
        self.p95_ms = p95_ms

    def evaluate(self, spans: list[Span]) -> list[Alert]:
        roots = [s for s in spans if s.parent_span_id is None]
        alerts = []
        by_version: dict[str, list[float]] = {}
        for sp in roots:
            by_version.setdefault(str(sp.attributes.get("deploy.version", "unknown")),
                                  []).append(sp.duration_ms)
        for version, lat in sorted(by_version.items()):
            p95 = percentile(lat, 95)
            if p95 > self.p95_ms:
                alerts.append(Alert(
                    rule=self.name, severity=CRITICAL,
                    message=f"request p95 {p95:.0f}ms over SLO {self.p95_ms:.0f}ms",
                    evidence=f"{len(lat)} requests, p50 {percentile(lat, 50):.0f}ms",
                    version=version))
        return alerts


class CostBudgetRule:
    name = "cost-budget"

    def __init__(self, budget_usd: float = COST_BUDGET_USD) -> None:
        self.budget_usd = budget_usd

    def evaluate(self, spans: list[Span]) -> list[Alert]:
        spend = round(sum(float(s.attributes.get("cost.usd", 0.0)) for s in spans), 8)
        if spend <= self.budget_usd:
            return []
        return [Alert(rule=self.name, severity=WARNING,
                      message=f"window spend ${spend:.6f} over budget ${self.budget_usd:.6f}",
                      evidence=f"{sum(int(s.attributes.get('gen_ai.usage.total_tokens', 0)) for s in spans)} tokens")]


class AlertEngine:
    """Runs every rule over the same span list. Same spans in -> same alerts out, in the
    same order: an alert you cannot reproduce is noise."""

    def __init__(self, rules=None) -> None:
        self.rules = rules or [LoopRule(), FailureRateRule(), LatencySLORule(),
                               CostBudgetRule()]

    def evaluate(self, spans: list[Span]) -> list[Alert]:
        out: list[Alert] = []
        for rule in self.rules:
            out.extend(rule.evaluate(spans))
        return out

    @staticmethod
    def fingerprint(alerts: list[Alert]) -> str:
        """A stable hash of an alert set, so 'deterministic' is testable, not asserted."""
        blob = "|".join(f"{a.rule}:{a.severity}:{a.version}:{a.message}" for a in alerts)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# 4. CANARY + ROLLBACK
# --------------------------------------------------------------------------- #
@dataclass
class Gate:
    """One release criterion, evaluated against measured numbers."""

    name: str
    ok: bool
    detail: str


@dataclass
class CanaryVerdict:
    decision: str                       # "promote" | "rollback"
    gates: list[Gate] = field(default_factory=list)
    blocking_alerts: list[Alert] = field(default_factory=list)
    active_version_after: str = ""

    @property
    def failed(self) -> list[Gate]:
        return [g for g in self.gates if not g.ok]


class CanaryController:
    """Deterministic traffic split + a gated promote/rollback decision.

    Routing hashes the request id, so request `req-003` always goes to the same version.
    Nothing here is random: a canary you cannot replay is a canary you cannot debug.
    """

    # release gates — the canary must be no worse than the baseline by more than this
    MAX_ERROR_RATE_DELTA = 0.05     # canary may not add more than 5pp of errors
    MAX_P95_RATIO = 1.50            # canary p95 may not be >1.5x the baseline
    MAX_COST_RATIO = 1.30           # canary cost/request may not be >1.3x the baseline

    def __init__(self, stable: str, canary: str, canary_pct: int) -> None:
        self.stable = stable
        self.canary = canary
        self.canary_pct = canary_pct
        self.active = stable            # what 100% of traffic gets after a rollback
        self.events: list[str] = [f"deploy: {canary} released to {canary_pct}% of traffic; "
                                  f"{stable} serving the rest"]

    def route(self, request_id: str) -> str:
        bucket = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16) % 100
        return self.canary if bucket < self.canary_pct else self.stable

    def evaluate(self, by_version: dict[str, Stats], alerts: list[Alert]) -> CanaryVerdict:
        base = by_version.get(self.stable)
        cand = by_version.get(self.canary)
        gates: list[Gate] = []

        if cand is None or cand.count == 0:
            gates.append(Gate("traffic", False, "canary received no traffic — cannot judge"))
            return self._finish(CanaryVerdict("rollback", gates, []))
        if base is None or base.count == 0:
            gates.append(Gate("baseline", False, "no stable baseline to compare against"))
            return self._finish(CanaryVerdict("rollback", gates, []))

        d_err = round(cand.error_rate - base.error_rate, 4)
        gates.append(Gate(
            "error-rate", d_err <= self.MAX_ERROR_RATE_DELTA,
            f"canary {cand.error_rate:.1%} vs stable {base.error_rate:.1%} "
            f"(Δ {d_err:+.1%}, max +{self.MAX_ERROR_RATE_DELTA:.0%})"))

        ratio_p95 = round(cand.p95 / base.p95, 3) if base.p95 else float("inf")
        gates.append(Gate(
            "latency-p95", ratio_p95 <= self.MAX_P95_RATIO,
            f"canary p95 {cand.p95:.0f}ms vs stable {base.p95:.0f}ms "
            f"({ratio_p95:.2f}x, max {self.MAX_P95_RATIO:.2f}x)"))

        ratio_cost = round(cand.cost_per_call / base.cost_per_call, 3) if base.cost_per_call else float("inf")
        gates.append(Gate(
            "cost-per-request", ratio_cost <= self.MAX_COST_RATIO,
            f"canary ${cand.cost_per_call:.6f} vs stable ${base.cost_per_call:.6f} "
            f"({ratio_cost:.2f}x, max {self.MAX_COST_RATIO:.2f}x)"))

        blocking = [a for a in alerts if a.severity == CRITICAL and a.version == self.canary]
        gates.append(Gate(
            "no-critical-alerts", not blocking,
            "clean" if not blocking else
            f"{len(blocking)} critical alert(s) on the canary: "
            + "; ".join(sorted({a.rule for a in blocking}))))

        decision = "promote" if all(g.ok for g in gates) else "rollback"
        return self._finish(CanaryVerdict(decision, gates, blocking))

    def _finish(self, verdict: CanaryVerdict) -> CanaryVerdict:
        if verdict.decision == "rollback":
            self.active = self.stable
            why = ", ".join(g.name for g in verdict.failed) or "gate failure"
            self.events.append(f"ROLLBACK: {self.canary} pulled from traffic ({why}); "
                               f"100% back on {self.stable}")
        else:
            self.active = self.canary
            self.events.append(f"PROMOTE: {self.canary} passed every gate; 100% of traffic moved")
        verdict.active_version_after = self.active
        return verdict
