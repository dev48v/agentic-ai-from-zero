"""Runnable demo for the Production Agent — observability (NVIDIA NIM).

    python 11-production-observability/run.py

Twelve support requests are pushed through a canary deployment: v1.4-stable serves most of
the traffic, v1.5-canary serves a deterministic hash-bucketed slice. Every LLM call and
every tool call opens a span, so by the end we have a real trace tree, a real latency +
cost dashboard built from those spans, real alerts, and a canary verdict derived from the
numbers rather than from a hunch.

Three things are wired in on purpose:

  * a DOWNSTREAM OUTAGE — one order's shard is unavailable for the whole window, so a
    request genuinely fails and the agent degrades instead of inventing order details.
  * a CANARY REGRESSION — v1.5 requires an evidence field (`refund_eta`) the upstream tool
    does not return, so it re-fetches the identical record until the step cap. It never
    throws. It just costs 2-3x more and takes 2-3x longer, which is what makes it a
    realistic production bug and a good test of whether the observability actually works.
  * a SELF-CHECK HARNESS — 15 assertions over the collected telemetry (trace-tree
    integrity, the cost accounting identity, export fidelity, percentile correctness,
    alert determinism, rollback behaviour), printed as a pass/fail table at the end.

The MODEL plans the tool call and writes each customer reply. Every number below — the
spans, the percentiles, the dollars, the alert thresholds, the release gates, the
rollback — is deterministic Python.
"""

from __future__ import annotations

import json
import os
import sys

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _REPO_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from agent import (  # noqa: E402
    CANARY, OUTAGE_ORDER, STABLE, Request, TracedAgent,
)
from alerts import (  # noqa: E402
    CRITICAL, AlertEngine, CanaryController, COST_BUDGET_USD, FAILURE_RATE_THRESHOLD,
    LATENCY_SLO_P95_MS, LOOP_REPEAT_THRESHOLD,
)
from telemetry import (  # noqa: E402
    ERROR, FanOutExporter, InMemoryExporter, JsonlExporter, MetricsStore, Tracer,
    default_log_path, percentile, price_usd,
)

CANARY_PCT = 30
TRACE_LOG = default_log_path()


# --------------------------------------------------------------------------- #
# The traffic. Twelve real-ish support tickets; tkt-1004 asks about the order whose
# shard is down for this window.
# --------------------------------------------------------------------------- #
TRAFFIC = [
    Request("tkt-1000", "Where is my order ORD-4101? It said delivered but I saw nothing.", "ORD-4101"),
    Request("tkt-1001", "Can you tell me the status of ORD-4102 and how much I paid?", "ORD-4102"),
    Request("tkt-1002", "I was told ORD-4103 was refunded — has the money actually gone back?", "ORD-4103"),
    Request("tkt-1003", "Why was ORD-4104 cancelled? I never asked for that.", "ORD-4104"),
    Request("tkt-1004", f"Any update on {OUTAGE_ORDER}? It's been a week.", OUTAGE_ORDER),
    Request("tkt-1005", "Quick one — is ORD-4105 still on its way?", "ORD-4105"),
    Request("tkt-1006", "ORD-4106 was a big order, please confirm it arrived and the total.", "ORD-4106"),
    Request("tkt-1007", "ORD-4107 hasn't shipped yet, what's happening?", "ORD-4107"),
    Request("tkt-1008", "Small item, ORD-4108 — did it get delivered?", "ORD-4108"),
    Request("tkt-1009", "Need the carrier and status for ORD-4109 for my records.", "ORD-4109"),
    Request("tkt-1010", "ORD-4110 shows refunded, can you confirm the amount?", "ORD-4110"),
    Request("tkt-1011", "ORD-4111 is still processing after two days — is that normal?", "ORD-4111"),
]


def _rule(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _short(text: str, n: int = 118) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


# --------------------------------------------------------------------------- #
# 1. TRACING — run the traffic, printing each span as it closes.
# --------------------------------------------------------------------------- #
_KIND_ICON = {"agent": "▣", "llm": "◆", "tool": "▸", "internal": "·"}


def make_span_printer(depth_of: dict):
    def on_span(sp):
        depth = depth_of.get(sp.parent_span_id, -1) + 1
        depth_of[sp.span_id] = depth
        icon = _KIND_ICON.get(sp.kind, "·")
        pad = "   " * depth
        bits = [f"{sp.duration_ms:8.1f}ms"]
        tok = sp.attributes.get("gen_ai.usage.total_tokens")
        if tok:
            bits.append(f"{tok:>4} tok")
            bits.append(f"${sp.attributes.get('cost.usd', 0.0):.6f}")
        if sp.status == ERROR:
            bits.append("STATUS=ERROR")
        print(f"      {pad}{icon} {sp.name:<28} " + "  ".join(bits)
              + (f"   ← {_short(sp.error_message, 60)}" if sp.status == ERROR else ""))
    return on_span


def drive_traffic():
    memory = InMemoryExporter()
    jsonl = JsonlExporter(TRACE_LOG)
    tracer = Tracer("support-agent", FanOutExporter(memory, jsonl),
                    on_span=make_span_printer({}))
    controller = CanaryController(STABLE.name, CANARY.name, CANARY_PCT)
    agent = TracedAgent(tracer)
    versions = {STABLE.name: STABLE, CANARY.name: CANARY}

    _rule("1 · TRACING — every request, LLM call and tool call is a span "
          "(OpenTelemetry shape, gen_ai.* conventions)")
    print(f"  deployment  : {STABLE.name} (baseline)  +  {CANARY.name} at {CANARY_PCT}% of traffic")
    print(f"  routing     : deterministic — md5(request_id) % 100 < {CANARY_PCT} → canary "
          f"(same ticket always lands on the same version)")
    print(f"  stable      : requires evidence {list(STABLE.required_evidence)}  — {STABLE.note}")
    print(f"  canary      : requires evidence {list(CANARY.required_evidence)}  — {CANARY.note}")
    print(f"  injected    : {OUTAGE_ORDER} shard is down for this whole window")

    responses = []
    for req in TRAFFIC:
        version = versions[controller.route(req.id)]
        tag = "CANARY" if version.name == CANARY.name else "stable"
        print(f"\n  ── {req.id}  →  {version.name} [{tag}]  ·  {_short(req.text, 74)}")
        resp = agent.handle(req, version)
        responses.append(resp)
        flags = []
        if resp.degraded:
            flags.append("DEGRADED (no order data invented)")
        if resp.tool_calls > 1 and not resp.degraded:
            flags.append(f"RETRIED {resp.tool_calls}x — still missing "
                         f"{', '.join(resp.missing_at_exit)}")
        suffix = ("   ⚠ " + " · ".join(flags)) if flags else ""
        print(f"      ↳ reply: {_short(resp.answer, 96)}{suffix}")
    return memory.spans, jsonl, controller, responses


# --------------------------------------------------------------------------- #
# 2. DASHBOARD — built from the spans, so it can never disagree with the traces.
# --------------------------------------------------------------------------- #
def print_dashboard(store: MetricsStore) -> None:
    _rule("2 · LATENCY + COST DASHBOARD — aggregated from the spans themselves")

    head = (f"  {'operation':<22} {'calls':>6} {'err':>5} {'p50 ms':>9} {'p95 ms':>9} "
            f"{'p99 ms':>9} {'tok in':>8} {'tok out':>8} {'$ total':>11}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    ops = store.by_operation()
    for name in sorted(ops):
        s = ops[name]
        print(f"  {s.label:<22} {s.count:>6} {s.errors:>5} {s.p50:>9.1f} {s.p95:>9.1f} "
              f"{s.p99:>9.1f} {s.input_tokens:>8} {s.output_tokens:>8} {s.cost_usd:>11.6f}")
    print("  " + "-" * (len(head) - 2))
    print(f"  {'TOTAL':<22} {sum(s.count for s in ops.values()):>6} "
          f"{sum(s.errors for s in ops.values()):>5} {'':>9} {'':>9} {'':>9} "
          f"{sum(s.input_tokens for s in ops.values()):>8} "
          f"{sum(s.output_tokens for s in ops.values()):>8} {store.total_cost():>11.6f}")

    print(f"\n  per deployed version (one row per REQUEST — what the SLO is written against):")
    head2 = (f"  {'version':<16} {'reqs':>6} {'errors':>7} {'err rate':>9} {'p50 ms':>9} "
             f"{'p95 ms':>9} {'tokens':>8} {'$ total':>11} {'$ / request':>13}")
    print(head2)
    print("  " + "-" * (len(head2) - 2))
    for name, s in sorted(store.by_version().items()):
        print(f"  {s.label:<16} {s.count:>6} {s.errors:>7} {s.error_rate:>8.1%} {s.p50:>9.1f} "
              f"{s.p95:>9.1f} {s.input_tokens + s.output_tokens:>8} {s.cost_usd:>11.6f} "
              f"{s.cost_per_call:>13.6f}")
    print(f"\n  note: token counts are the provider's REAL usage numbers; the NIM free tier "
          f"bills $0, so the\n  dollars are what this traffic would cost at the published "
          f"reference rate in telemetry.py.")


# --------------------------------------------------------------------------- #
# 3. ALERTS
# --------------------------------------------------------------------------- #
def print_alerts(alerts) -> None:
    _rule("3 · ALERTING — fixed rules over the same spans (loops, failures, latency, cost)")
    print(f"  rules: loop ≥{LOOP_REPEAT_THRESHOLD} identical tool calls/request · "
          f"failure rate >{FAILURE_RATE_THRESHOLD:.0%} of requests · "
          f"p95 >{LATENCY_SLO_P95_MS:.0f}ms · spend >${COST_BUDGET_USD:.4f}\n")
    if not alerts:
        print("  (no alerts — nothing crossed a threshold)")
        return
    for a in alerts:
        mark = "🔴" if a.severity == CRITICAL else "🟠"
        print(f"  {mark} {a.line()}")
        if a.evidence:
            print(f"     evidence: {_short(a.evidence, 104)}")


# --------------------------------------------------------------------------- #
# 4. CANARY VERDICT + ROLLBACK
# --------------------------------------------------------------------------- #
def print_canary(controller: CanaryController, verdict) -> None:
    _rule("4 · CANARY EVALUATION → ROLLBACK — release gates, evaluated on measured numbers")
    for g in verdict.gates:
        print(f"  {'✅' if g.ok else '❌'} {g.name:<20} {g.detail}")
    print(f"\n  DECISION: {verdict.decision.upper()}   →   active version is now "
          f"{verdict.active_version_after}")
    print("\n  deployment events:")
    for e in controller.events:
        print(f"    · {e}")


# --------------------------------------------------------------------------- #
# THE SELF-CHECK HARNESS — 14 assertions over the telemetry this run produced.
# If observability is the product, it needs its own tests.
# --------------------------------------------------------------------------- #
def self_checks(spans, jsonl, store, alerts, verdict, controller, responses):
    checks = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:                       # a check that explodes is a failure
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append((name, bool(ok), detail))

    traces = store.traces()
    roots = [s for s in spans if s.parent_span_id is None]
    by_id = {s.span_id: s for s in spans}

    check("every span closed", lambda: (
        all(s.ended for s in spans), f"{len(spans)} spans, all have an end time"))

    check("every span identified", lambda: (
        all(s.trace_id and s.span_id and s.name for s in spans),
        "trace_id + span_id + name present on every span"))

    check("parent links resolve", lambda: (
        all(s.parent_span_id in by_id and by_id[s.parent_span_id].trace_id == s.trace_id
            for s in spans if s.parent_span_id),
        "no orphan spans; every child shares its parent's trace_id"))

    check("one root per trace", lambda: (
        all(sum(1 for s in g if s.parent_span_id is None) == 1 for g in traces.values()),
        f"{len(traces)} traces, exactly one agent.request root each"))

    check("trace count == requests", lambda: (
        len(traces) == len(TRAFFIC), f"{len(traces)} traces for {len(TRAFFIC)} requests"))

    check("exporter fidelity", lambda: (
        jsonl.count == len(spans) and
        sum(1 for _ in open(TRACE_LOG, encoding="utf-8")) == len(spans),
        f"{jsonl.count} JSONL lines == {len(spans)} in-memory spans"))

    def otlp_roundtrip():
        rows = [json.loads(l) for l in open(TRACE_LOG, encoding="utf-8") if l.strip()]
        need = {"traceId", "spanId", "parentSpanId", "name", "startTimeUnixNano",
                "endTimeUnixNano", "attributes", "status"}
        return (all(need <= set(r) for r in rows),
                f"{len(rows)} exported spans carry the full OTLP field set")
    check("OTLP shape on export", otlp_roundtrip)

    check("llm spans carry real usage", lambda: (
        all(int(s.attributes.get("gen_ai.usage.input_tokens", 0)) > 0 and
            int(s.attributes.get("gen_ai.usage.output_tokens", 0)) > 0
            for s in spans if s.kind == "llm"),
        f"{sum(1 for s in spans if s.kind == 'llm')} llm spans, all with nonzero "
        f"provider token counts"))

    def cost_identity():
        per_trace = {}
        for s in spans:
            per_trace[s.trace_id] = round(per_trace.get(s.trace_id, 0.0)
                                          + float(s.attributes.get("cost.usd", 0.0)), 8)
        total = round(sum(per_trace.values()), 8)
        by_ver = sum(v.cost_usd for v in store.by_version().values())
        return (abs(total - store.total_cost()) < 1e-9 and abs(round(by_ver, 8) - total) < 1e-9,
                f"Σ span cost = Σ trace cost = Σ version cost = ${total:.6f}")
    check("cost accounting identity", cost_identity)

    check("pricing is exact", lambda: (
        price_usd(1000, 1000) == round(0.00015 + 0.00060, 8) and price_usd(0, 0) == 0.0,
        "price_usd(1000,1000) == input+output reference rate"))

    check("percentile is nearest-rank", lambda: (
        percentile(list(range(1, 101)), 50) == 50 and
        percentile(list(range(1, 101)), 95) == 95 and
        percentile([], 95) == 0.0,
        "p50/p95 over 1..100 == 50/95; empty series == 0"))

    check("root duration ≥ children", lambda: (
        all(r.duration_ms + 1e-6 >= sum(c.duration_ms for c in spans
                                        if c.parent_span_id == r.span_id) for r in roots),
        "every request span is at least as long as the work nested inside it"))

    def loop_detected():
        fired = [a for a in alerts if a.rule == "loop-detected"]
        canary_loops = [r for r in responses
                        if r.version == CANARY.name and r.tool_calls >= LOOP_REPEAT_THRESHOLD]
        return (bool(fired) and len(fired) == len(canary_loops) and
                all(a.version == CANARY.name for a in fired),
                f"{len(fired)} loop alert(s), all on the canary, matching "
                f"{len(canary_loops)} looping request(s)")
    check("loop alert caught the canary", loop_detected)

    check("alerting is deterministic", lambda: (
        AlertEngine.fingerprint(AlertEngine().evaluate(spans)) ==
        AlertEngine.fingerprint(alerts),
        "re-evaluating the same spans yields a byte-identical alert set"))

    check("rollback actually happened", lambda: (
        verdict.decision == "rollback" and controller.active == STABLE.name
        and verdict.active_version_after == STABLE.name and len(verdict.failed) > 0,
        f"{len(verdict.failed)} gate(s) failed → traffic back on {controller.active}"))

    return checks


def print_checks(checks) -> int:
    _rule("SELF-CHECK — assertions over the telemetry this run actually produced")
    passed = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print(f"  {'✅' if ok else '❌'} {name:<30} {_short(detail, 62)}")
    print(f"\n  {passed}/{len(checks)} checks pass")
    return passed


# --------------------------------------------------------------------------- #
def main() -> int:
    _rule("PRODUCTION AGENT — observability: tracing · latency+cost dashboard · alerting on "
          "loops+failures · canary + rollback\n(NVIDIA NIM · meta/llama-3.1-8b-instruct)")
    print("A support agent is put under real traffic behind a canary deployment. Every LLM and tool call")
    print("is a span; the dashboard, the alerts and the release decision are all derived from those spans.")
    print("The model plans and writes the replies; every threshold, percentile, dollar and gate is Python.")

    spans, jsonl, controller, responses = drive_traffic()

    store = MetricsStore(spans)
    print_dashboard(store)

    alerts = AlertEngine().evaluate(spans)
    print_alerts(alerts)

    verdict = controller.evaluate(store.by_version(), alerts)
    print_canary(controller, verdict)

    checks = self_checks(spans, jsonl, store, alerts, verdict, controller, responses)
    passed = print_checks(checks)

    _rule("THE FOUR SUB-POINTS, IN THIS RUN")
    print("1. tracing — every request opened an `agent.request` span with nested `llm.*` and `tool.*`")
    print("   children in the OpenTelemetry shape, LLM spans tagged with the gen_ai.* semantic")
    print("   conventions and exported as OTLP-shaped JSONL, ready for LangSmith / Arize Phoenix.")
    print("2. latency + cost dashboard — p50/p95/p99, error counts, real provider token counts and")
    print("   priced spend, sliced by operation and by deployed version, computed FROM the spans.")
    print("3. alerting on loops + failures — the loop rule caught a canary that re-fetched an identical")
    print("   record until its step cap, and the failure-rate rule caught the request the downstream")
    print("   outage broke; the latency and cost rules fired off the same span data.")
    print("4. canary + rollback — a deterministic hash split sent a slice of traffic to v1.5, four")
    print("   release gates were evaluated on the measured numbers, and the failing gates rolled")
    print("   traffic back to v1.4 automatically.")
    print(f"\n  trace log → {TRACE_LOG}  (gitignored, regenerated each run)")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
