"""The observability core — TRACING, METRICS and PRICING. Zero model calls, zero
vendor SDKs, fully deterministic.

This is the part of a production agent that nobody demos but everybody needs. It has
three jobs:

  1. TRACING  -> `Tracer` / `Span`. A span is one unit of work (the whole agent run, one
                 LLM call, one tool call) with a `trace_id`, its own `span_id`, a
                 `parent_span_id`, timings, attributes and a status. Nesting spans gives
                 you the tree that shows *where the time and the money actually went*.
                 The span shape here is deliberately the OpenTelemetry one, and LLM
                 spans carry the OTel **GenAI semantic conventions** (`gen_ai.*`), so an
                 exporter that speaks OTLP can ship these straight to LangSmith / Arize
                 Phoenix / Jaeger without the agent code changing a line. We hand-roll the
                 emitter (a ~120-line dataclass + a JSONL writer) instead of pulling a
                 vendor SDK: same data, no account, no key, no lock-in.

  2. METRICS  -> `MetricsStore`. Spans in, aggregates out: count, error rate, p50/p95/p99
                 latency, tokens and dollars, sliced by operation and by deployed version.
                 Percentiles use the nearest-rank definition so the same spans always
                 produce the same number.

  3. PRICING  -> `price_usd`. Token counts come from the provider's REAL `usage` block on
                 every response; the $ figure multiplies them by a published reference
                 rate. Being straight about it: the NVIDIA NIM free tier bills $0, so the
                 dollars below are what this traffic WOULD cost at a typical hosted-8B
                 rate. The tokens are real; the rate is a constant you can change in one
                 place.

Nothing in this file knows what an agent is. That separation is the point: telemetry you
can unit-test without a network, and an agent you can trace without rewriting it.
"""

from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# 3. PRICING — published reference rate ($ per 1k tokens). One constant, one place.
# --------------------------------------------------------------------------- #
PRICE_PER_1K_INPUT = 0.00015
PRICE_PER_1K_OUTPUT = 0.00060


def price_usd(input_tokens: int, output_tokens: int) -> float:
    """Dollars for a call, from REAL provider token counts and the reference rate."""
    dollars = (input_tokens / 1000.0) * PRICE_PER_1K_INPUT + \
              (output_tokens / 1000.0) * PRICE_PER_1K_OUTPUT
    return round(dollars, 8)


# --------------------------------------------------------------------------- #
# 1. TRACING — the span.
# --------------------------------------------------------------------------- #
OK = "OK"
ERROR = "ERROR"
UNSET = "UNSET"


@dataclass
class Span:
    """One unit of work. OpenTelemetry's shape: ids + a parent link + timings +
    attributes + a status. The parent link is what makes a flat list of spans a tree."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str = "internal"            # "agent" | "llm" | "tool" | "internal"
    start_unix_ns: int = 0            # wall clock, for export
    _start_perf_ns: int = 0           # monotonic, for duration (Windows clock is coarse)
    end_perf_ns: int = 0
    attributes: dict = field(default_factory=dict)
    status: str = UNSET
    error_message: str = ""

    @property
    def ended(self) -> bool:
        return self.end_perf_ns > 0

    @property
    def duration_ms(self) -> float:
        if not self.ended:
            return 0.0
        return round((self.end_perf_ns - self._start_perf_ns) / 1e6, 3)

    def set(self, **attrs) -> "Span":
        """Attach attributes. Returns self so it chains inside a `with` block."""
        self.attributes.update(attrs)
        return self

    def record_error(self, message: str) -> "Span":
        self.status = ERROR
        self.error_message = str(message)
        return self

    # -- the wire format ---------------------------------------------------- #
    def to_otlp(self) -> dict:
        """An OTLP-shaped span dict. Field names are OpenTelemetry's, so this is what a
        collector (or Phoenix / LangSmith) expects; only the transport differs."""
        end_unix_ns = self.start_unix_ns + (self.end_perf_ns - self._start_perf_ns)
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id or "",
            "name": self.name,
            "kind": self.kind,
            "startTimeUnixNano": self.start_unix_ns,
            "endTimeUnixNano": end_unix_ns if self.ended else 0,
            "durationMs": self.duration_ms,
            "attributes": dict(self.attributes),
            "status": {"code": self.status, "message": self.error_message},
        }


# --------------------------------------------------------------------------- #
# Exporters — where finished spans go. Swap this seam for an OTLP/HTTP exporter and
# the very same spans land in LangSmith or Arize Phoenix.
# --------------------------------------------------------------------------- #
class InMemoryExporter:
    """Keeps spans in a list — what the metrics, alerts and self-checks read."""

    def __init__(self) -> None:
        self.spans: list[Span] = []

    def export(self, span: Span) -> None:
        self.spans.append(span)


class JsonlExporter:
    """Appends one OTLP-shaped JSON object per finished span. Poor-man's collector:
    `tail -f` it, or replay the file into a real backend later."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.count = 0
        open(self.path, "w", encoding="utf-8").close()   # truncate per run

    def export(self, span: Span) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(span.to_otlp(), ensure_ascii=False) + "\n")
        self.count += 1


class FanOutExporter:
    def __init__(self, *exporters) -> None:
        self.exporters = list(exporters)

    def export(self, span: Span) -> None:
        for e in self.exporters:
            e.export(span)


# --------------------------------------------------------------------------- #
# The tracer — creates spans, links parents, exports on close.
# --------------------------------------------------------------------------- #
_ACTIVE: contextvars.ContextVar[Span | None] = contextvars.ContextVar("active_span",
                                                                     default=None)


class Tracer:
    """Hand-rolled, ~zero-dependency tracer with the OpenTelemetry span model.

    `with tracer.span("llm.plan", kind="llm") as sp:` opens a child of whatever span is
    currently active (a contextvar, so nesting is automatic), times it, marks it ERROR if
    the block raises, and exports it on close — always, even on the exception path.
    """

    def __init__(self, service_name: str, exporter, on_span=None) -> None:
        self.service_name = service_name
        self.exporter = exporter
        self._on_span = on_span            # optional live hook, used by run.py to print

    @staticmethod
    def _new_id(n: int) -> str:
        return uuid.uuid4().hex[:n]

    @contextmanager
    def span(self, name: str, kind: str = "internal", **attrs):
        parent = _ACTIVE.get()
        sp = Span(
            trace_id=parent.trace_id if parent else self._new_id(32),
            span_id=self._new_id(16),
            parent_span_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            start_unix_ns=time.time_ns(),
            _start_perf_ns=time.perf_counter_ns(),
            attributes={"service.name": self.service_name, **attrs},
        )
        token = _ACTIVE.set(sp)
        try:
            yield sp
        except Exception as exc:                     # a failed span is still a span
            sp.record_error(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            _ACTIVE.reset(token)
            sp.end_perf_ns = time.perf_counter_ns()
            if sp.status == UNSET:
                sp.status = OK
            self.exporter.export(sp)
            if self._on_span:
                self._on_span(sp)


def record_llm_usage(span: Span, model: str, usage, operation: str) -> tuple[int, int, float]:
    """Stamp an LLM span with the OTel GenAI semantic conventions using the provider's
    REAL usage block, plus the derived cost. Returns (in, out, $)."""
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    cost = price_usd(in_tok, out_tok)
    span.set(**{
        "gen_ai.system": "nvidia_nim",
        "gen_ai.operation.name": operation,
        "gen_ai.request.model": model,
        "gen_ai.usage.input_tokens": in_tok,
        "gen_ai.usage.output_tokens": out_tok,
        "gen_ai.usage.total_tokens": in_tok + out_tok,
        "cost.usd": cost,
    })
    return in_tok, out_tok, cost


# --------------------------------------------------------------------------- #
# 2. METRICS — spans in, aggregates out. Deterministic.
# --------------------------------------------------------------------------- #
def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value at or above p% of the sorted data.
    Chosen over interpolation because it always returns a value that was really observed
    and never drifts with float rounding — the same spans give the same p95, always."""
    if not values:
        return 0.0
    ordered = sorted(values)
    import math
    rank = max(1, math.ceil((p / 100.0) * len(ordered)))
    return round(ordered[min(rank, len(ordered)) - 1], 3)


@dataclass
class Stats:
    """Aggregates for one slice (an operation, or a deployed version)."""

    label: str
    count: int = 0
    errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

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
    def p99(self) -> float:
        return percentile(self.latencies_ms, 99)

    @property
    def total_ms(self) -> float:
        return round(sum(self.latencies_ms), 3)

    @property
    def cost_per_call(self) -> float:
        return round(self.cost_usd / self.count, 8) if self.count else 0.0


class MetricsStore:
    """Rolls a flat span list up into the numbers a dashboard shows.

    Deliberately computed FROM THE SPANS rather than counted alongside them: if it isn't
    in the trace it isn't on the dashboard, so the two can never disagree.
    """

    def __init__(self, spans: list[Span]) -> None:
        self.spans = spans

    def _rollup(self, key, keep=lambda s: True) -> dict[str, Stats]:
        out: dict[str, Stats] = {}
        for sp in self.spans:
            if not keep(sp):
                continue
            k = key(sp)
            if k is None:
                continue
            st = out.setdefault(k, Stats(label=k))
            st.count += 1
            if sp.status == ERROR:
                st.errors += 1
            st.latencies_ms.append(sp.duration_ms)
            st.input_tokens += int(sp.attributes.get("gen_ai.usage.input_tokens", 0))
            st.output_tokens += int(sp.attributes.get("gen_ai.usage.output_tokens", 0))
            st.cost_usd = round(st.cost_usd + float(sp.attributes.get("cost.usd", 0.0)), 8)
        return out

    def by_operation(self) -> dict[str, Stats]:
        return self._rollup(lambda s: s.name)

    def by_version(self) -> dict[str, Stats]:
        """Only root (agent) spans — one row per REQUEST, which is what an SLO is about.
        Token/cost totals are re-attributed from the whole trace below."""
        roots = {s.span_id: s for s in self.spans if s.parent_span_id is None}
        out: dict[str, Stats] = {}
        for sp in roots.values():
            v = str(sp.attributes.get("deploy.version", "unknown"))
            st = out.setdefault(v, Stats(label=v))
            st.count += 1
            if sp.status == ERROR:
                st.errors += 1
            st.latencies_ms.append(sp.duration_ms)
        # money + tokens belong to the request that caused them, not to the LLM span
        root_version = {s.trace_id: str(s.attributes.get("deploy.version", "unknown"))
                        for s in roots.values()}
        for sp in self.spans:
            v = root_version.get(sp.trace_id)
            if v is None or v not in out:
                continue
            out[v].input_tokens += int(sp.attributes.get("gen_ai.usage.input_tokens", 0))
            out[v].output_tokens += int(sp.attributes.get("gen_ai.usage.output_tokens", 0))
            out[v].cost_usd = round(out[v].cost_usd + float(sp.attributes.get("cost.usd", 0.0)), 8)
        return out

    def traces(self) -> dict[str, list[Span]]:
        out: dict[str, list[Span]] = {}
        for sp in self.spans:
            out.setdefault(sp.trace_id, []).append(sp)
        return out

    def total_cost(self) -> float:
        return round(sum(float(s.attributes.get("cost.usd", 0.0)) for s in self.spans), 8)

    def total_tokens(self) -> int:
        return sum(int(s.attributes.get("gen_ai.usage.total_tokens", 0)) for s in self.spans)


def default_log_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces.jsonl")
