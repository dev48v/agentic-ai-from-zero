# 08 — Event-Triggered Automation Agent

**Goal:** an agent that **fires on events, not chat**. No one types at it. A new order is
placed, a support email lands, a file is uploaded — each is an **event** that triggers the
agent to **classify, draft, summarize, or route** it. The interesting part isn't "call an
LLM"; it's everything around the call that makes an event pipeline *trustworthy*:
**idempotent** (a re-delivered event is handled once), **retried** (a transient blip is
retried with backoff), and **dead-lettered** (a permanently-failing event is quarantined,
never silently dropped and never left to block the stream).

The model does exactly one thing: **reason about one event inside a handler** (triage the
order, draft the reply, route the document). **Everything that makes it a reliable pipeline
is deterministic Python** — the queue, the dedup ledger, the retry/backoff, the DLQ, and the
poison-vs-transient distinction. Same stream in → same handled / deduped / retried /
dead-lettered / rejected outcome out, no matter what the model says.

## The four ideas (hand-rolled, no framework)

| # | Sub-point | Where it lives |
|---|-----------|----------------|
| 1 | **listen to webhooks + queues** | [`queue.py`](queue.py) `EventQueue` — an append-only JSONL "queue" (a pulled-broker source). [`webhook.py`](webhook.py) `WebhookReceiver` — a stdlib `http.server` on localhost that turns every `POST /events` into an `Event` and `append()`s it to the **same** queue (a pushed source). Both converge on one worker. |
| 2 | **execute workflows on triggers** | [`agent.py`](agent.py) `Dispatcher` maps an event **type** → a handler; each handler VALIDATEs, then calls the model to **classify + draft / summarize + route**, then DELIVERs to a downstream. Unknown types are a `BusinessReject` — nothing can handle them. |
| 3 | **idempotent execution** | [`queue.py`](queue.py) `ProcessedLedger` — one line per terminally-handled event **id**. The worker checks `seen(id)` **before any work**, so a re-delivered event is a no-op: the handler and its model call never run twice for the same id. |
| 4 | **dead-letter + retry** | worker loop in [`run.py`](run.py) — a `TransientError` is retried with bounded exponential `backoff_delay`; after `MAX_ATTEMPTS` the event goes to the [`queue.py`](queue.py) `DeadLetterQueue` with its last error. A `BusinessReject` is **terminal** — recorded as `rejected`, **never retried**. |

## The pipeline — the one thing that is never the model's

```
  event sources                         the worker (deterministic)
  ─────────────                         ──────────────────────────
  webhook POST ─┐                        ┌─ seen(id)? ──yes──►  DEDUP (no-op)          ← idempotency
                ├──►  EventQueue  ──claim──┤            no
  file / queue ─┘   (append-only JSONL)   │
                                          ▼
                                     dispatch(type) ──► handler:  validate → MODEL → deliver
                                          │
             ┌────────────────────────────┼─────────────────────────────┐
             ▼                             ▼                             ▼
        Outcome ✓                    TransientError                 BusinessReject
        record `handled`             retry w/ backoff               record `rejected`
                                     └─ exhausted N? ─► DLQ          (NEVER retried)      ← poison ≠ transient
                                        record `dead_lettered`
```

- **Idempotency is checked first**, before the handler runs — so at-least-once delivery in becomes exactly-once effect out.
- **Transient vs business is the key distinction.** A downstream 503 is *retriable* (the work was fine, the dependency blinked). An order with a `$0` total is *poison* (retrying it a thousand times still fails). They demand opposite responses, so they are separate exception types.
- **The DLQ is a quarantine, not a bin.** A dead-lettered event keeps its payload, attempt count, and last error, so a human (or a replay job) can inspect and re-drive it — one poison message never blocks the rest of the stream.

## Why transient ≠ business (the bug this pattern avoids)

Conflating the two failure kinds is the classic event-agent bug, in both directions:

- **Retry a poison message forever** → the queue wedges on one bad event, burning API calls on something that can never succeed.
- **Dead-letter a transient blip** → a single retry would have fixed it, but the event is quarantined and a human gets paged for nothing.

So `agent.py` raises **`TransientError`** only from the *downstream delivery* step (a
dependency hiccup) and **`BusinessReject`** from *validation* (the event is semantically
invalid). The worker retries the first and immediately terminates the second.

## The simulated downstream (honest note)

Handlers end by "delivering" their result to a downstream system (a fulfilment queue, a
ticket API, a team). To make transient failures **reproducible**, the event payload may carry
a deterministic fault switch, `_inject.transient_fails = N`: the delivery raises a
`TransientError` on attempts `1..N` and succeeds after. `N = 2` demonstrates
**retry-then-succeed**; `N` larger than `MAX_ATTEMPTS` models a dependency that stays down
long enough to **dead-letter**. This is a stand-in for a genuinely flaky dependency — the
**model calls are real** on every attempt; only the downstream's success/failure is scripted,
so the demo reproduces exactly.

## What the demo shows ([`run.py`](run.py)) — one mixed stream

`run.py` is the **producer** (emits the stream — event #1 over a **real localhost webhook
POST**, the rest onto the file queue) and the **worker** (drains it). Seven deliveries, six
distinct ids, all four sub-points:

1. `ord-1001` **new_order** (via **webhook**) — high-value + express → model triages **EXPEDITE** → **HANDLED**.
2. `eml-2001` **support_email** — model classifies **billing / negative / high** + drafts a reply → **HANDLED**.
3. `file-3001` **file_uploaded** — model summarizes + routes to **finance** → **HANDLED**.
4. `ord-1001` **new_order** (**re-delivered**) — already in the ledger → **DEDUP**, no-op. ← *idempotency*
5. `eml-2002` **support_email** (downstream fails 2×) — `TransientError` ×2 → backoff → **HANDLED on try 3**. ← *retry*
6. `ord-1002` **new_order** (downstream always down) — `TransientError` ×3 → **DEAD-LETTER**. ← *dead-letter*
7. `ord-1003` **new_order** (`total = $0`) — `BusinessReject` → **REJECTED**, not retried. ← *poison ≠ transient*

```bash
# from the repo root, with .venv active and NVIDIA_API_KEY set in .env
python 08-event-triggered/run.py

# or run the webhook receiver standalone and POST to it yourself:
python 08-event-triggered/webhook.py --port 8808 --queue 08-event-triggered/queue.jsonl
curl -X POST localhost:8808/events -H 'Content-Type: application/json' \
     -d '{"type":"support_email","payload":{"from":"you@x.com","subject":"hi","body":"help please"}}'
```

See [`recorded-run.md`](recorded-run.md) for the **real** captured transcript against NVIDIA
NIM — the live webhook `HTTP 202`, every handler call a `HTTP/1.1 200 OK` to
`integrate.api.nvidia.com`, the dedup firing, the retry-then-succeed, the dead-letter, and
the full processed-ledger + DLQ contents.

## Files

- `queue.py` — the deterministic infra: `Event`, the append-only `EventQueue` (+ FIFO claim
  cursor), the `ProcessedLedger` (idempotency), the `DeadLetterQueue`, and `backoff_delay`.
  **No model calls.**
- `webhook.py` — a dependency-free `http.server` `WebhookReceiver` that enqueues inbound
  `POST /events` onto the same `EventQueue` (the pushed event source). Enqueues only — never
  runs a handler, so a webhook can't block on the model.
- `agent.py` — the `Dispatcher` (type → handler) and the three model-backed handlers, plus
  the `TransientError` / `BusinessReject` failure taxonomy. The only place a model call happens.
- `run.py` — the runnable producer (webhook + queue) and the deterministic worker loop
  (claim → dedup → dispatch → retry/backoff → DLQ) + the reports.
- `recorded-run.md` — a real transcript hitting NVIDIA NIM (incl. the ledger + DLQ JSONL).
- `queue.jsonl` / `processed-ledger.jsonl` / `dead-letter.jsonl` — runtime state written by
  `run.py` (gitignored; regenerated each run).

## Note on the model

Per event the model is asked only to **reason about that event** — classify, draft,
summarize, route — and return strict JSON. The **queue, the FIFO claim, the idempotency
dedup, the retry/backoff schedule, the dead-letter decision, and the transient-vs-business
distinction are all deterministic Python**. That is deliberate: the value of an
event-triggered agent is that reliability — *handled once, retried sanely, never dropped* —
is a property of the pipeline you can read, test, and reproduce, not a behavior an 8B model
has to remember to perform.
