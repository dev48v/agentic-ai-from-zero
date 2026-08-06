# Recorded run — Event-Triggered Automation Agent

A **real** transcript captured by running the mixed event stream:

```bash
python 08-event-triggered/run.py
```

- **Provider / endpoint:** NVIDIA NIM — `POST https://integrate.api.nvidia.com/v1/chat/completions` (every handler call below returned `HTTP/1.1 200 OK` in the live `httpx` log).
- **Model:** `meta/llama-3.1-8b-instruct` (warm on the free tier).
- **Date:** 2026-08-06.
- **The webhook is real:** event #1 was delivered by an actual `POST` to a stdlib `http.server` bound to `127.0.0.1:8808` — `GET /health` returned `200` and the `POST /events` returned `HTTP 202 {"status":"enqueued",...}` before the worker ran. (If a localhost port were unavailable in some sandbox, `run.py` falls back to enqueuing #1 directly and says so — here it bound cleanly.)
- **What the run proves:** a mixed stream from **two sources** (1 webhook + 6 queued) is drained by **one** deterministic worker that **dispatches** each event to a model-backed handler, **dedups** a re-delivered event, **retries** a transient failure with backoff and **succeeds**, **dead-letters** a permanently-failing one, and **rejects** a poison event **without** retrying it.

The model only **reasons inside a handler** (classify / draft / summarize / route). The queue, the FIFO claim, the idempotency dedup, the retry/backoff, the dead-letter decision, and the poison-vs-transient distinction are deterministic Python. Everything below is verbatim from the run.

---

## The stream (1 webhook POST + 6 queued; six distinct ids)

```
==========================================================================================
PRODUCER — emitting a mixed event stream (1 webhook POST + 6 queued)
==========================================================================================
  🌐 webhook receiver LIVE at http://127.0.0.1:8808  ·  GET /health → {'status': 'ok', 'queue_depth': 0}
  📨 POST /events ord-1001 (new_order) → HTTP 202 {'status': 'enqueued', 'id': 'ord-1001', 'type': 'new_order'}
  🌐 webhook receiver stopped (its event is already durable on the queue).
  📥 7 events on the queue (...\08-event-triggered\queue.jsonl); worker will drain FIFO.
     1. ord-1001  new_order      — normal · high-value + express → expect EXPEDITE · via WEBHOOK
     2. eml-2001  support_email  — normal · expect billing / negative → drafts a reply
     3. file-3001 file_uploaded  — normal · expect finance route + one-line summary
     4. ord-1001  new_order      — RE-DELIVERY of ord-1001 → must be DEDUPED (idempotency)
     5. eml-2002  support_email  — downstream flaky (fails 2×) → RETRY with backoff → HANDLED on try 3
     6. ord-1002  new_order      — downstream DOWN (fails always) → exhausts 3 retries → DEAD-LETTER
     7. ord-1003  new_order      — invalid (total $0) → BusinessReject → NOT retried
```

The first event arrived over HTTP (`source=webhook`); the rest were appended to the file/JSONL queue (`source=queue`). The worker cannot tell the difference — both are just events to claim.

---

## The worker, event by event (real NIM calls, `HTTP/1.1 200 OK`)

### Events 1–3 — normal triggers, each **HANDLED** by the model on the first try

```
── event 1: ord-1001  ·  type=new_order  ·  source=webhook ─────────────────────
  ▶️  attempt 1/3 — dispatch → `new_order` handler (real NIM call)…
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
  ✅ HANDLED — order_triaged: order ord-1001 → EXPEDITE → fulfilment-queue
      model decision: {"priority": "expedite", "reason": "Express shipping and high-value order", "customer_message": "Your order ord-1001 has been expedited and will be s…

── event 2: eml-2001  ·  type=support_email  ·  source=queue ─────────────────────
  ▶️  attempt 1/3 — dispatch → `support_email` handler (real NIM call)…
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
  ✅ HANDLED — email_triaged: email from dana@example.com → billing/high → ticketing-system
      model decision: {"category": "billing", "sentiment": "negative", "priority": "high", "draft_reply": "Sorry to hear that you were double charged. I'll look into this …

── event 3: file-3001  ·  type=file_uploaded  ·  source=queue ─────────────────────
  ▶️  attempt 1/3 — dispatch → `file_uploaded` handler (real NIM call)…
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
  ✅ HANDLED — file_routed: Q3-financials.csv → financials → team:finance
      model decision: {"doc_type": "financials", "summary": "Revenue was up 12% quarter-over-quarter with steady margins.", "route_to": "finance"}
```

The dispatcher routed each event **type** to its handler; the model did the judgement — triaged the high-value express order to **EXPEDITE**, read the double-charge email as **billing / negative / high** and drafted an apology, and summarized the CSV and routed it to **finance**. Three types, three handlers, three real model calls.

### Event 4 — re-delivery of `ord-1001` → **DEDUP** (idempotent execution)

```
── event 4: ord-1001  ·  type=new_order  ·  source=queue ─────────────────────
  ♻️  DEDUP — ord-1001 already in the ledger as `handled` (try 1); skipping. The handler + its model call do NOT run again.
```

`ord-1001` was already delivered once (over the webhook) and handled. The worker checks the processed ledger **before** doing any work, so the second delivery is a **no-op** — no dispatch, no model call, no duplicate fulfilment. At-least-once delivery in, exactly-once effect out.

### Event 5 — `eml-2002`, downstream fails 2× → **RETRY with backoff → HANDLED on try 3**

```
── event 5: eml-2002  ·  type=support_email  ·  source=queue ─────────────────────
  ▶️  attempt 1/3 — dispatch → `support_email` handler (real NIM call)…
INFO  httpx  HTTP Request: POST .../chat/completions "HTTP/1.1 200 OK"
  ⏳ TRANSIENT — downstream 'ticketing-system' unavailable (attempt 1 of a simulated 2-attempt outage) — HTTP 503
      retry 2/3 after backoff 0.10s…
  ▶️  attempt 2/3 — dispatch → `support_email` handler (real NIM call)…
INFO  httpx  HTTP Request: POST .../chat/completions "HTTP/1.1 200 OK"
  ⏳ TRANSIENT — downstream 'ticketing-system' unavailable (attempt 2 of a simulated 2-attempt outage) — HTTP 503
      retry 3/3 after backoff 0.20s…
  ▶️  attempt 3/3 — dispatch → `support_email` handler (real NIM call)…
INFO  httpx  HTTP Request: POST .../chat/completions "HTTP/1.1 200 OK"
  ✅ HANDLED — email_triaged: email from sam@example.com → technical/normal → ticketing-system
      model decision: {"category": "technical", "sentiment": "neutral", "priority": "normal", "draft_reply": "Thank you for reaching out. I'd be happy to help you reset yo…
```

The downstream `ticketing-system` returned `503` on the first two attempts. The worker **retried with bounded exponential backoff** (`0.10s`, then `0.20s`) and the third attempt succeeded → the event was **handled on try 3**, not dead-lettered. The model call is genuine on every attempt (three `200 OK`s); only the downstream's recovery is scripted.

### Event 6 — `ord-1002`, downstream stays down → **DEAD-LETTER**

```
── event 6: ord-1002  ·  type=new_order  ·  source=queue ─────────────────────
  ▶️  attempt 1/3 — dispatch → `new_order` handler (real NIM call)…
INFO  httpx  HTTP Request: POST .../chat/completions "HTTP/1.1 200 OK"
  ⏳ TRANSIENT — downstream 'fulfilment-queue' unavailable (attempt 1 of a simulated 99-attempt outage) — HTTP 503
      retry 2/3 after backoff 0.10s…
  ▶️  attempt 2/3 — dispatch → `new_order` handler (real NIM call)…
INFO  httpx  HTTP Request: POST .../chat/completions "HTTP/1.1 200 OK"
  ⏳ TRANSIENT — downstream 'fulfilment-queue' unavailable (attempt 2 of a simulated 99-attempt outage) — HTTP 503
      retry 3/3 after backoff 0.20s…
  ▶️  attempt 3/3 — dispatch → `new_order` handler (real NIM call)…
INFO  httpx  HTTP Request: POST .../chat/completions "HTTP/1.1 200 OK"
  ☠️  DEAD-LETTER — transient failure persisted through 3 attempts → DLQ. last error: downstream 'fulfilment-queue' unavailable (attempt 3 of a simulated 99-attempt outage) — HTTP 503
```

Here the downstream never recovered. The worker retried the **bounded** number of times (`MAX_ATTEMPTS = 3`) and then **dead-lettered** the event — it landed in the DLQ with its payload, attempt count, and last error for a human to inspect and replay. One stuck event did **not** block the rest of the stream; the worker moved straight on to event 7.

### Event 7 — `ord-1003`, `total = $0` → **REJECTED** (business), not retried

```
── event 7: ord-1003  ·  type=new_order  ·  source=queue ─────────────────────
  ▶️  attempt 1/3 — dispatch → `new_order` handler (real NIM call)…
  🚫 REJECTED (business) — order total must be a positive number, got 0.0
      not retried, not dead-lettered — a semantic reject is terminal.
```

The order failed **validation** (a `$0` total is a poison order), so the handler raised `BusinessReject` **before** ever calling the model. The worker recorded it as terminally `rejected` and did **not** retry it and did **not** dead-letter it — retrying a semantically-invalid event can never help. Note there is **no** `httpx 200 OK` line for this event: a poison message doesn't get to spend a model call.

---

## Processed ledger (idempotency registry) — one row per terminally-handled id

```
  event_id    type            outcome        try  detail
  ------------------------------------------------------
  ord-1001    new_order       handled          1  order ord-1001 → EXPEDITE → fulfilment-queue
  eml-2001    support_email   handled          1  email from dana@example.com → billing/high → ticketing-syst…
  file-3001   file_uploaded   handled          1  Q3-financials.csv → financials → team:finance
  eml-2002    support_email   handled          3  email from sam@example.com → technical/normal → ticketing-s…
  ord-1002    new_order       dead_lettered    3  downstream 'fulfilment-queue' unavailable (attempt 3 of a s…
  ord-1003    new_order       rejected         1  order total must be a positive number, got 0.0

  outcomes: {'handled': 4, 'dead_lettered': 1, 'rejected': 1}
```

Six distinct ids, six terminal rows — even though **seven** events were delivered. The seventh (the re-delivered `ord-1001`) produced **no** new row: it was deduped. `eml-2002` shows `try 3` — the two transient failures then the success. `ord-1002` is `dead_lettered` after 3 tries; `ord-1003` is `rejected` after 1 (no retry).

---

## Dead-letter queue — quarantined for inspection / replay

```
  ☠️  ord-1002 (new_order) after 3 attempts
      error: downstream 'fulfilment-queue' unavailable (attempt 3 of a simulated 99-attempt outage) — HTTP 503
      payload: {"order_id": "ord-1002", "customer": "globex@example.com", "shipping": "standard", "items": [{"sku": "GADGET", "qty": 1…
```

The dead-lettered event keeps its full payload, so it can be re-driven once the downstream recovers — a failed event is **quarantined, never dropped**.

---

## The four sub-points, together

| Sub-point | Evidence in this run |
|-----------|----------------------|
| **Listen to webhooks + queues** | Event `ord-1001` arrived via a real `POST` to `http://127.0.0.1:8808/events` (`HTTP 202`); the other six arrived on the append-only JSONL queue. Both sources converged on one worker — it reacts to events, it is never "called". |
| **Execute workflows on triggers** | The dispatcher mapped `new_order` / `support_email` / `file_uploaded` to three handlers; the model triaged the order to **EXPEDITE**, classified the email as **billing/negative/high** with a drafted reply, and summarized + routed the CSV to **finance**. |
| **Idempotent execution** | `ord-1001` was delivered **twice** (once by webhook, once re-queued). The second delivery hit the processed ledger and became a **no-op** — handled exactly once, no duplicate model call, no duplicate fulfilment. |
| **Dead-letter + retry** | `eml-2002` failed transiently **2×** then **succeeded on the backed-off 3rd try**; `ord-1002`'s downstream stayed down and **dead-lettered** after `MAX_ATTEMPTS`; `ord-1003` was a **BusinessReject** (poison) and was **not** retried — the transient-vs-business distinction, on the record. |

> Note: the model is small (8B) and not perfectly deterministic even at low temperature, so a re-run may phrase the drafted replies, summaries, or `reason` fields slightly differently, and the exact wall-clock timestamps will differ. What is **stable** by construction: the FIFO claim order, the dedup of a re-delivered id, the retry count + backoff schedule, the dead-letter after `MAX_ATTEMPTS`, and the immediate terminal reject of a poison event — those are pure Python, so the *outcome* of each event (handled / deduped / handled-after-retry / dead-lettered / rejected) is the same every run.

## The durable state (JSONL — gitignored / regenerated)

**`processed-ledger.jsonl`** (idempotency registry):

```json
{"event_id": "ord-1001", "type": "new_order", "outcome": "handled", "attempts": 1, "detail": "order ord-1001 → EXPEDITE → fulfilment-queue", "ts": "2026-08-06T14:01:35"}
{"event_id": "eml-2001", "type": "support_email", "outcome": "handled", "attempts": 1, "detail": "email from dana@example.com → billing/high → ticketing-system", "ts": "2026-08-06T14:01:37"}
{"event_id": "file-3001", "type": "file_uploaded", "outcome": "handled", "attempts": 1, "detail": "Q3-financials.csv → financials → team:finance", "ts": "2026-08-06T14:01:37"}
{"event_id": "eml-2002", "type": "support_email", "outcome": "handled", "attempts": 3, "detail": "email from sam@example.com → technical/normal → ticketing-system", "ts": "2026-08-06T14:01:41"}
{"event_id": "ord-1002", "type": "new_order", "outcome": "dead_lettered", "attempts": 3, "detail": "downstream 'fulfilment-queue' unavailable (attempt 3 of a simulated 99-attempt outage) — HTTP 503", "ts": "2026-08-06T14:01:44"}
{"event_id": "ord-1003", "type": "new_order", "outcome": "rejected", "attempts": 1, "detail": "order total must be a positive number, got 0.0", "ts": "2026-08-06T14:01:44"}
```

**`dead-letter.jsonl`** (the quarantine):

```json
{"event_id": "ord-1002", "type": "new_order", "attempts": 3, "error": "downstream 'fulfilment-queue' unavailable (attempt 3 of a simulated 99-attempt outage) — HTTP 503", "payload": {"order_id": "ord-1002", "customer": "globex@example.com", "shipping": "standard", "items": [{"sku": "GADGET", "qty": 1}], "total": 45.0, "_inject": {"transient_fails": 99}}, "ts": "2026-08-06T14:01:44"}
```
