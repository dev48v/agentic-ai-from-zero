# 11 — Production Agent (observability)

**Goal:** take an agent that works and make it **operable** — so that when it goes wrong in
production you find out from the data, not from a customer. Three of the previous ten
projects built agents that *reason*; this one builds the layer that tells you **what the
agent actually did, what it cost, and whether the version you just shipped made things
worse**.

The failure this project is designed around is the one that hurts most in real life: **the
agent that does not crash.** It throws no exception, returns a plausible answer, passes
review — and quietly re-fetches the same record three times per request, tripling your
latency and your bill. No `try/except` catches that. A trace does.

The model plans the tool call and writes the customer replies. **Every number here — the
spans, the percentiles, the dollars, the alert thresholds, the release gates, the
rollback — is deterministic Python.** Same spans in → same dashboard, same alerts, same
verdict out.

## The four ideas (hand-rolled, no vendor SDK)

| # | Sub-point | Where it lives |
|---|-----------|----------------|
| 1 | **tracing** | [`telemetry.py`](telemetry.py) `Tracer` / `Span` — the OpenTelemetry span model (`trace_id`, `span_id`, `parent_span_id`, timings, attributes, status), nested via a `contextvar` so `with tracer.span(...)` builds the tree automatically. LLM spans carry the OTel **GenAI semantic conventions** (`gen_ai.usage.input_tokens`, `gen_ai.request.model`, …) and export as **OTLP-shaped JSONL**. |
| 2 | **latency + cost dashboard** | [`telemetry.py`](telemetry.py) `MetricsStore` — count, error rate, **p50/p95/p99** (nearest-rank), real provider token counts and priced spend, rolled up **by operation** and **by deployed version**, computed *from the spans*. |
| 3 | **alerting on loops + failures** | [`alerts.py`](alerts.py) `AlertEngine` with four pure rules: `LoopRule` (same tool + same args ≥3× in one request), `FailureRateRule` (per version, over requests), `LatencySLORule` (p95 over SLO), `CostBudgetRule` (window spend over budget). |
| 4 | **canary + rollback** | [`alerts.py`](alerts.py) `CanaryController` — deterministic hash-bucketed traffic split, then four **release gates** (error-rate delta, p95 ratio, cost-per-request ratio, no critical alerts on the canary). Any gate fails → **ROLLBACK**, traffic flips back, reason recorded. |

## Why hand-roll it instead of importing LangSmith / Arize Phoenix

The roadmap called for LangSmith or Arize Phoenix tracing. What those tools consume is
**OTLP spans with the GenAI semantic conventions** — so that is exactly what this emits.
The `Tracer` is ~120 lines and the exporter is a seam: `JsonlExporter` writes the same
objects an OTLP/HTTP exporter would POST. Point that seam at a collector and these traces
land in Phoenix or LangSmith unchanged, with **no edit to the agent**.

Keeping it hand-rolled buys three things that matter for this series: no account, no key
and no vendor lock-in (the repo stays 100% free and runnable by anyone who clones it); the
span model stays visible instead of hiding behind a decorator; and the whole telemetry
layer is testable offline, which is what the self-check harness does. The honest trade-off:
you get no hosted UI, no retention, no cross-run comparison. Those are the reasons to adopt
a real backend — not the instrumentation itself, which is the cheap part.

## The two versions, and the bug between them

```python
STABLE = AgentVersion(required_evidence=("status", "total_cents"),               max_steps=3)
CANARY = AgentVersion(required_evidence=("status", "total_cents", "refund_eta"), max_steps=3)
```

One tuple entry. `lookup_order` does not return `refund_eta` yet, so the canary's
deterministic "do I have enough evidence?" test never passes: it re-fetches the identical
record, re-drafts the answer, checks again, and only stops when it hits the step cap.

- it never raises
- it returns a **correct, sensible reply** every time
- it costs **~2.2× more** and takes **~2× longer** per request

That is the shape of a real regression. It is caught here by the loop rule and the
cost-per-request gate, not by a status code.

The exit condition is plain Python (`_missing_evidence`), deliberately **not** a model
judgement — so the failure reproduces identically on every run rather than depending on how
the 8B model feels about its own confidence that day.

## What the demo shows ([`run.py`](run.py))

Twelve support tickets are pushed through a canary deployment (`v1.5-canary` at 30% of
traffic, routed by `md5(request_id) % 100` so the same ticket always lands on the same
version and a run is replayable). Two things are injected on purpose: the **canary
regression** above, and a **downstream outage** — one order's shard is unavailable for the
whole window, so a request genuinely fails and the agent has to degrade without inventing
order details.

```bash
# from the repo root, with .venv active and NVIDIA_API_KEY set in .env
python 11-production-observability/run.py
```

It ends with a **self-check harness: 15 assertions over the telemetry the run just
produced** — trace-tree integrity (every span closed, parent links resolve, one root per
trace), export fidelity (JSONL line count matches, full OTLP field set), the **cost
accounting identity** (Σ span cost = Σ trace cost = Σ version cost), percentile
correctness, alert determinism (re-evaluating the same spans gives a byte-identical alert
set), and that the rollback actually happened. If observability is the product, it needs
its own tests. `run.py` exits non-zero if any check fails.

See [`recorded-run.md`](recorded-run.md) for the real captured transcript against NVIDIA
NIM. Headline numbers from that run: **64 spans across 12 traces, 32 live LLM calls,
15/15 checks pass**, canary at **$0.000194/request vs stable $0.000088 (2.20×)**, four
loop alerts (all on the canary), one failure-rate alert (the outage), and an automatic
**ROLLBACK** to `v1.4-stable`.

## What observability does *not* buy you — from this run

In the recorded run the agent told a customer their order total was **£207.08** when the
record said `21050` pence — **£210.50**. Every span was green. Latency was fine, cost was
fine, no loop, no error. **Tracing tells you about loops, latency, spend and failures; it
tells you nothing about whether the answer was right.** Correctness needs a different
mechanism — a rubric and an LLM-as-judge co-gated by deterministic hard checks, which is
exactly what [Project 10](../10-self-reflective/) built. The two layers are complementary
and neither substitutes for the other.

Two more honest notes from the same run:

- **The p95 gate passed** (1.46×, just under the 1.50× limit) even though the canary was
  genuinely broken. With eight baseline samples, p95 *is* the slowest baseline request, and
  one slow cold call inflated it. **Percentiles over small samples are noisy** — the
  rollback was driven by the cost gate and the loop alert, which is why you gate on several
  independent signals rather than one.
- **The latency SLO fired on the baseline too.** The free-tier endpoint was slow that
  minute. A real SLO is set from a long baseline, not from twelve requests; the threshold
  here is a named constant precisely so it is obviously a knob and not a law.

## Files

- `telemetry.py` — `Span` / `Tracer` / exporters (`InMemory`, `Jsonl`, `FanOut`), the
  `gen_ai.*` usage stamper, `price_usd`, `percentile`, `Stats` and `MetricsStore`.
  **No model calls, no network.**
- `alerts.py` — the four alert rules + `AlertEngine` (with a stable fingerprint so
  "deterministic" is testable), and `CanaryController` with the release gates and the
  rollback. **Pure functions of the spans.**
- `agent.py` — the instrumented support agent, the two deployed versions, the deterministic
  evidence check, and the scripted downstream outage. **The only place a model call happens.**
- `run.py` — the runnable demo: live trace tree, dashboard, alerts, canary verdict, and the
  15-assertion self-check harness.
- `recorded-run.md` — the real captured transcript against NVIDIA NIM.
- `traces.jsonl` — OTLP-shaped span export written by `run.py` (gitignored, regenerated each run).

## Note on the model

The model does two jobs: pick the tool + argument, and write the reply. It does **not**
open spans, compute a percentile, decide what an alert is, or vote on the release. That
split is the whole point of this project — the parts of production you must be able to
trust at 3am should be code you can read, re-run and diff, not a number a model asserted
about itself. A malformed model reply is repaired once and then degrades to a validated
default; it never breaks the trace, and a broken trace would fail the self-check rather
than pass silently.
