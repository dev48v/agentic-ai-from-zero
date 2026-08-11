# Recorded run — Production Agent (observability)

A **real** transcript captured by putting twelve support tickets through a canary
deployment:

```bash
python 11-production-observability/run.py
```

- **Provider / endpoint:** NVIDIA NIM — `POST https://integrate.api.nvidia.com/v1/chat/completions`.
- **Model:** `meta/llama-3.1-8b-instruct` (warm on the free tier).
- **Date:** 2026-08-11.
- **Traffic:** 12 tickets · **12 traces · 64 spans · 32 live LLM calls · 20 tool calls**.
- **Deployment:** `v1.4-stable` (baseline) + `v1.5-canary` at 30% of traffic, routed by
  `md5(request_id) % 100` — deterministic, so the same ticket always lands on the same
  version. Actual split: **8 stable / 4 canary**.
- **Injected on purpose:** (1) the canary requires an evidence field `refund_eta` that
  `lookup_order` does not return → a **silent retry loop**; (2) `ORD-4199`'s shard is
  **down for the whole window** → a genuinely failed request the agent must degrade on.
- **Outcome:** four loop alerts (all on the canary), one failure-rate alert (the outage),
  two SLO alerts, canary at **2.20× the cost per request** → automatic **ROLLBACK**, and
  **15/15 self-checks pass** (`run.py` exits 0).

The model plans the tool call and writes the replies. Every span, percentile, dollar,
threshold, gate and the rollback are deterministic Python. Everything below is verbatim
from the run.

---

## 1 · Tracing — a healthy request

```
  ── tkt-1001  →  v1.4-stable [stable]  ·  Can you tell me the status of ORD-4102 and how much I paid?
      ◆ llm.plan                       1999.9ms   173 tok  $0.000040
      ▸ tool.lookup_order                 0.0ms
      ◆ llm.answer                     1262.8ms   225 tok  $0.000047
      ▣ agent.request                  3266.5ms
      ↳ reply: Your order ORD-4102 is in transit and you paid £154.00.
```

One request = one trace: an `agent.request` root with `llm.*` and `tool.*` children. The
exported span is OTLP-shaped, with the OTel GenAI semantic conventions on the LLM call:

```json
{
  "traceId": "2815a6d7642340809e571c841a526e1b",
  "spanId": "513047e370b74bfb",
  "parentSpanId": "f1541cb42e7c4059",
  "name": "llm.plan",
  "kind": "llm",
  "startTimeUnixNano": 1786447102318475400,
  "endTimeUnixNano": 1786447106191844700,
  "durationMs": 3873.369,
  "attributes": {
    "service.name": "support-agent",
    "gen_ai.system": "nvidia_nim",
    "gen_ai.operation.name": "plan",
    "gen_ai.request.model": "meta/llama-3.1-8b-instruct",
    "gen_ai.usage.input_tokens": 142,
    "gen_ai.usage.output_tokens": 33,
    "gen_ai.usage.total_tokens": 175,
    "cost.usd": 4.11e-05,
    "plan.tool": "lookup_order",
    "plan.order_id": "ORD-4101",
    "plan.reason": "Customer is disputing delivery status of their order."
  },
  "status": { "code": "OK", "message": "" }
}
```

### The canary, looping — the bug that throws nothing

```
  ── tkt-1007  →  v1.5-canary [CANARY]  ·  ORD-4107 hasn't shipped yet, what's happening?
      ◆ llm.plan                       4890.2ms   166 tok  $0.000037
      ▸ tool.lookup_order                 0.0ms
      ◆ llm.answer                     4331.4ms   227 tok  $0.000050
      ▸ tool.lookup_order                 0.0ms
      ◆ llm.answer                     3068.6ms   220 tok  $0.000049
      ▸ tool.lookup_order                 0.0ms
      ◆ llm.answer                     1607.0ms   220 tok  $0.000049
      ▣ agent.request                 13909.3ms
      ↳ reply: The order ORD-4107 is currently in the processing status and the total amount is £67.25.   ⚠ RETRIED 3x — still missing refund_eta
```

Three identical `lookup_order(order_id=ORD-4107)` calls, three drafts, **a correct final
answer**, no exception, `status: OK` on every span — and 13.9 seconds instead of 3.3. This
is what a regression looks like when nothing crashes.

### The downstream outage — a real failure, degraded honestly

```
  ── tkt-1004  →  v1.4-stable [stable]  ·  Any update on ORD-4199? It's been a week.
      ◆ llm.plan                       2167.2ms   170 tok  $0.000039
      ▸ tool.lookup_order                 0.1ms  STATUS=ERROR   ← order-service shard unavailable for ORD-4199 (503)
      ◆ llm.answer                     2521.1ms   178 tok  $0.000043
      ▣ agent.request                  4696.9ms  STATUS=ERROR   ← order-service shard unavailable for ORD-4199 (503)
      ↳ reply: Our system is currently experiencing technical difficulties and is temporarily unavailable. I w…   ⚠ DEGRADED (no order data invented)
```

The tool span carries the error, the root span is marked failed (the customer did not get
what they asked for — that is what the SLO counts), and the agent invents nothing.

---

## 2 · Latency + cost dashboard — built from those same spans

```
  operation               calls   err    p50 ms    p95 ms    p99 ms   tok in  tok out     $ total
  -----------------------------------------------------------------------------------------------
  agent.request              12     1    5398.6   13909.3   13909.3        0        0    0.000000
  llm.answer                 20     0    1674.0    3319.7    4331.4     3750      740    0.001007
  llm.plan                   12     0    1515.4    8415.0    8415.0     1677      376    0.000477
  tool.lookup_order          20     1       0.0       0.0       0.1        0        0    0.000000
  -----------------------------------------------------------------------------------------------
  TOTAL                      64     2                                   5427     1116    0.001484

  per deployed version (one row per REQUEST — what the SLO is written against):
  version            reqs  errors  err rate    p50 ms    p95 ms   tokens     $ total   $ / request
  ------------------------------------------------------------------------------------------------
  v1.4-stable           8       1    12.5%    4689.9    9529.2     3136    0.000706      0.000088
  v1.5-canary           4       0     0.0%    6826.9   13909.3     3407    0.000778      0.000194
```

Read the version table and the bug is obvious even without the alerts: the canary took
**4 requests** and spent **more in total** than 8 stable requests — **$0.000194 vs
$0.000088 per request, 2.20×**. Token counts are the provider's real `usage` numbers; the
NIM free tier bills $0, so the dollars are what this traffic would cost at the published
reference rate in `telemetry.py`.

Note `tool.lookup_order` at **0.0ms** — the tool is an in-process dict. That is deliberate:
it makes it unmistakable that **all** the latency and **all** the cost live in the LLM
calls, and that the canary's damage is entirely "it made more of them".

---

## 3 · Alerting

```
  rules: loop ≥3 identical tool calls/request · failure rate >10% of requests · p95 >8000ms · spend >$0.0100

  🔴 CRITICAL loop-detected    [v1.5-canary] agent repeated an identical tool call 3x in one request (threshold 3) — not making progress
     evidence: trace da51ab12abbd… signature `lookup_order(order_id=ORD-4104)`
  … (4 loop alerts in total — one per canary request, none on stable)
  🟠 WARNING  failure-rate     [v1.4-stable] request failure rate 12.5% over threshold 10% (1/8 requests failed)
     evidence: order-service shard unavailable for ORD-4199 (503)
  🔴 CRITICAL latency-slo      [v1.4-stable] request p95 9529ms over SLO 8000ms
     evidence: 8 requests, p50 4690ms
  🔴 CRITICAL latency-slo      [v1.5-canary] request p95 13909ms over SLO 8000ms
     evidence: 4 requests, p50 6827ms
```

**Every loop alert landed on the canary and none on stable** — the rule is a per-trace
count of identical tool signatures, so it separates the broken version from the healthy one
without knowing anything about what the agent was trying to do.

The failure-rate rule is measured **per version**, not pooled: pooled, the single outage
would have been 1/12 = 8.3% and stayed silent under a 10% threshold. Split by version it is
1/8 = 12.5% on stable and fires — and it is correctly *not* blamed on the canary.

---

## 4 · Canary evaluation → rollback

```
  ✅ error-rate           canary 0.0% vs stable 12.5% (Δ -12.5%, max +5%)
  ✅ latency-p95          canary p95 13909ms vs stable 9529ms (1.46x, max 1.50x)
  ❌ cost-per-request     canary $0.000194 vs stable $0.000088 (2.20x, max 1.30x)
  ❌ no-critical-alerts   5 critical alert(s) on the canary: latency-slo; loop-detected

  DECISION: ROLLBACK   →   active version is now v1.4-stable

  deployment events:
    · deploy: v1.5-canary released to 30% of traffic; v1.4-stable serving the rest
    · ROLLBACK: v1.5-canary pulled from traffic (cost-per-request, no-critical-alerts); 100% back on v1.4-stable
```

**Two gates that did *not* catch it, and why that matters.** The error-rate gate passed —
the canary never errored, it was the *healthy* version by that measure. The p95 gate passed
too, at 1.46× against a 1.50× limit: with eight baseline samples, p95 *is* the slowest
baseline request, and one slow cold call inflated the denominator. A release gated on
error rate alone would have promoted a broken build. **The rollback came from the cost
gate and the loop alert** — which is the argument for gating on several independent signals
instead of the one everybody watches.

---

## Self-check — 15 assertions over the telemetry this run produced

```
  ✅ every span closed              64 spans, all have an end time
  ✅ every span identified          trace_id + span_id + name present on every span
  ✅ parent links resolve           no orphan spans; every child shares its parent's trace_id
  ✅ one root per trace             12 traces, exactly one agent.request root each
  ✅ trace count == requests        12 traces for 12 requests
  ✅ exporter fidelity              64 JSONL lines == 64 in-memory spans
  ✅ OTLP shape on export           64 exported spans carry the full OTLP field set
  ✅ llm spans carry real usage     32 llm spans, all with nonzero provider token counts
  ✅ cost accounting identity       Σ span cost = Σ trace cost = Σ version cost = $0.001484
  ✅ pricing is exact               price_usd(1000,1000) == input+output reference rate
  ✅ percentile is nearest-rank     p50/p95 over 1..100 == 50/95; empty series == 0
  ✅ root duration ≥ children       every request span is at least as long as the work nested ins…
  ✅ loop alert caught the canary   4 loop alert(s), all on the canary, matching 4 looping reques…
  ✅ alerting is deterministic      re-evaluating the same spans yields a byte-identical alert set
  ✅ rollback actually happened     2 gate(s) failed → traffic back on v1.4-stable

  15/15 checks pass
```

---

## The honest miss — what tracing cannot see

On `tkt-1009` the agent replied:

```
      ↳ reply: Your order ORD-4109 is with carrier DPD and is in transit. The total amount is £207.08.
```

The record says `total_cents: 21050` — **£210.50**. The model fumbled the pence→pounds
conversion and shipped a wrong number to a customer.

Every span on that request was green. Latency normal, cost normal, no loop, no error, no
alert. **Observability tells you about loops, latency, spend and failures; it says nothing
about whether the answer was correct.** That is a different mechanism — a rubric plus an
LLM-as-judge co-gated by deterministic hard checks, which is what
[Project 10](../10-self-reflective/) built. A production agent needs both layers, and
neither one covers for the other.

---

## The four sub-points, together

| Sub-point | Evidence in this run |
|-----------|----------------------|
| **tracing** | 64 spans across 12 traces, nested `agent.request → llm.* / tool.*`, OTLP-shaped with `gen_ai.*` semantic conventions and real provider token counts; exported to `traces.jsonl` and verified line-for-line by the self-check. |
| **latency + cost dashboard** | p50/p95/p99, error counts, tokens and priced spend, by operation *and* by deployed version — computed from the spans, so the dashboard and the traces cannot disagree. |
| **alerting on loops + failures** | 4 loop alerts (identical tool signature 3× in one request, all on the canary, none on stable), 1 per-version failure-rate alert from the injected outage, 2 p95 SLO alerts; the alert set is fingerprint-stable across re-evaluation. |
| **canary + rollback** | Deterministic 30% hash split (8 stable / 4 canary), four release gates on measured numbers, 2 failed → automatic ROLLBACK to `v1.4-stable`, with the deployment events recorded. |

> Note: the model is small (8B) and the free endpoint's latency varies minute to minute, so
> a re-run will word the replies differently and the millisecond figures will move. What is
> **stable by construction**: the traffic split (hashed, not random), the canary's retry
> loop (a deterministic evidence check, not a model judgement), the span tree, the
> percentile and pricing arithmetic, the alert rules and the release gates. Given the same
> spans, the dashboard, the alerts and the rollback decision are identical every time —
> which is the only reason any of it is worth paging someone about.

## The durable state (JSONL — gitignored / regenerated)

`traces.jsonl` holds one OTLP-shaped span per line — 64 lines for this run. It is what an
OTLP exporter would POST to a collector, so replaying it into LangSmith / Arize Phoenix
needs no change to the agent.
