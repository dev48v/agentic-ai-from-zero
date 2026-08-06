"""Runnable demo for the Event-Triggered Automation Agent (NVIDIA NIM).

    python 08-event-triggered/run.py

The agent fires on EVENTS, not chat. This script is the PRODUCER (it emits a mixed stream
of events — one over a real localhost WEBHOOK, the rest onto a file/JSONL QUEUE) and the
WORKER (a deterministic loop that claims events, dedups, dispatches to a model-backed
handler, retries transient failures with backoff, and dead-letters the ones that never
recover). It exercises all four sub-points in one pass:

  1. new_order   ord-1001  (via WEBHOOK)   → model triages → HANDLED
  2. support_email eml-2001 (via queue)    → model classifies + drafts reply → HANDLED
  3. file_uploaded file-3001 (via queue)   → model summarizes + routes → HANDLED
  4. new_order   ord-1001  (RE-DELIVERED)  → already in the ledger → DEDUP, no-op   ← idempotency
  5. support_email eml-2002 (fails 2×)     → TransientError ×2 → backoff → HANDLED on try 3  ← retry
  6. new_order   ord-1002  (fails always)  → TransientError ×3 → exhausts retries → DLQ   ← dead-letter
  7. new_order   ord-1003  (total = $0)    → BusinessReject → REJECTED, NOT retried   ← poison ≠ transient

The MODEL reasons inside each handler (classify / draft / summarize / route); the queue,
the dedup, the retry/backoff, and the DLQ are deterministic Python — so the same stream
always produces the same handled / deduped / retried / dead-lettered / rejected outcome.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.request

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _REPO_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from queue import (  # noqa: E402
    DeadLetterQueue, Event, EventQueue, ProcessedLedger, backoff_delay,
)
from agent import BusinessReject, Dispatcher, TransientError  # noqa: E402
from webhook import WebhookReceiver  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    stream=sys.stdout,
)

QUEUE_PATH = os.path.join(_PROJECT_DIR, "queue.jsonl")
LEDGER_PATH = os.path.join(_PROJECT_DIR, "processed-ledger.jsonl")
DLQ_PATH = os.path.join(_PROJECT_DIR, "dead-letter.jsonl")

MAX_ATTEMPTS = 3        # bounded retry: after this many tries a transient failure dead-letters


def _rule(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _short(text: str, n: int = 96) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


# --------------------------------------------------------------------------- #
# The mixed event stream (id, type, payload, note). ord-1001 appears twice on
# purpose — the second delivery must be deduped.
# --------------------------------------------------------------------------- #
def build_stream():
    return [
        ("ord-1001", "new_order", {
            "order_id": "ord-1001", "customer": "acme@example.com", "shipping": "express",
            "items": [{"sku": "WIDGET-PRO", "qty": 2}], "total": 640.00},
         "normal · high-value + express → expect EXPEDITE · via WEBHOOK"),
        ("eml-2001", "support_email", {
            "from": "dana@example.com", "subject": "Double charged on my invoice",
            "body": "Hi, I was billed twice for order 5581 this month. Can you refund the duplicate? Frustrated."},
         "normal · expect billing / negative → drafts a reply"),
        ("file-3001", "file_uploaded", {
            "filename": "Q3-financials.csv",
            "text": "Quarter,Revenue,COGS,NetIncome\nQ3-2026,1840000,910000,420000\nNotes: revenue up 12% QoQ, margin steady."},
         "normal · expect finance route + one-line summary"),
        ("ord-1001", "new_order", {
            "order_id": "ord-1001", "customer": "acme@example.com", "shipping": "express",
            "items": [{"sku": "WIDGET-PRO", "qty": 2}], "total": 640.00},
         "RE-DELIVERY of ord-1001 → must be DEDUPED (idempotency)"),
        ("eml-2002", "support_email", {
            "from": "sam@example.com", "subject": "How do I reset my API token?",
            "body": "I can't find where to rotate my API key in the dashboard. Please advise.",
            "_inject": {"transient_fails": 2}},
         "downstream flaky (fails 2×) → RETRY with backoff → HANDLED on try 3"),
        ("ord-1002", "new_order", {
            "order_id": "ord-1002", "customer": "globex@example.com", "shipping": "standard",
            "items": [{"sku": "GADGET", "qty": 1}], "total": 45.00,
            "_inject": {"transient_fails": 99}},
         "downstream DOWN (fails always) → exhausts 3 retries → DEAD-LETTER"),
        ("ord-1003", "new_order", {
            "order_id": "ord-1003", "customer": "wayne@example.com", "shipping": "standard",
            "items": [{"sku": "DECOY", "qty": 1}], "total": 0.0},
         "invalid (total $0) → BusinessReject → NOT retried"),
    ]


# --------------------------------------------------------------------------- #
# PRODUCER — emit the stream: event #1 over a real webhook, the rest onto the queue.
# --------------------------------------------------------------------------- #
def produce(queue: EventQueue, stream) -> dict:
    _rule("PRODUCER — emitting a mixed event stream (1 webhook POST + 6 queued)")
    meta = {"webhook_ok": False, "webhook_url": None}

    first_id, first_type, first_payload, first_note = stream[0]
    receiver = None
    for port in (8808, 0):     # try a friendly fixed port, fall back to an OS-assigned one
        try:
            receiver = WebhookReceiver(queue, host="127.0.0.1", port=port).start()
            break
        except OSError as e:
            print(f"  (port {port} unavailable: {e})")
    if receiver is not None:
        try:
            meta["webhook_url"] = receiver.url
            # health check — proves the receiver is live and bound
            with urllib.request.urlopen(receiver.url + "/health", timeout=5) as r:
                health = json.loads(r.read().decode())
            print(f"  🌐 webhook receiver LIVE at {receiver.url}  ·  GET /health → {health}")
            # POST the first event exactly as an external system (Stripe/GitHub/a form) would
            body = json.dumps({"id": first_id, "type": first_type, "payload": first_payload}).encode()
            req = urllib.request.Request(receiver.url + "/events", data=body,
                                         headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=5) as r:
                resp = json.loads(r.read().decode())
            print(f"  📨 POST /events {first_id} ({first_type}) → HTTP {r.status} {resp}")
            meta["webhook_ok"] = True
        except Exception as e:                     # pragma: no cover - env-dependent
            print(f"  ⚠️  webhook POST failed in this env ({e}); enqueuing #1 directly instead.")
            queue.append(Event.new(first_type, first_payload, id=first_id, source="queue"))
        finally:
            receiver.stop()
            print("  🌐 webhook receiver stopped (its event is already durable on the queue).")
    else:
        print("  ⚠️  could not bind a localhost port in this env; enqueuing #1 directly instead.")
        queue.append(Event.new(first_type, first_payload, id=first_id, source="queue"))

    # the remaining events arrive on the file/JSONL queue (a pulled broker source)
    for eid, etype, payload, note in stream[1:]:
        queue.append(Event.new(etype, payload, id=eid, source="queue"))
    print(f"  📥 {len(stream)} events on the queue ({QUEUE_PATH}); worker will drain FIFO.")
    for i, (eid, etype, _p, note) in enumerate(stream, 1):
        print(f"     {i}. {eid:<9} {etype:<14} — {note}")
    return meta


# --------------------------------------------------------------------------- #
# WORKER — claim → dedup → dispatch → retry/backoff → DLQ. Deterministic.
# --------------------------------------------------------------------------- #
def work(queue: EventQueue, ledger: ProcessedLedger, dlq: DeadLetterQueue,
         dispatcher: Dispatcher) -> None:
    _rule("WORKER — draining the queue (idempotent dispatch · retry with backoff · dead-letter)")
    n = 0
    while True:
        event = queue.claim()
        if event is None:
            break
        n += 1
        print(f"\n── event {n}: {event.id}  ·  type={event.type}  ·  source={event.source} "
              f"─────────────────────")

        # 1) IDEMPOTENCY — a terminally-handled id is a no-op, no matter the source.
        if ledger.seen(event.id):
            prior = ledger.get(event.id)
            print(f"  ♻️  DEDUP — {event.id} already in the ledger as `{prior.outcome}` "
                  f"(try {prior.attempts}); skipping. The handler + its model call do NOT run again.")
            continue

        # 2) DISPATCH with bounded retry + backoff.
        attempt = 0
        while True:
            attempt += 1
            try:
                print(f"  ▶️  attempt {attempt}/{MAX_ATTEMPTS} — dispatch → `{event.type}` handler "
                      f"(real NIM call)…")
                outcome = dispatcher.dispatch(event, attempt)
                ledger.record(event.id, event.type, "handled", attempt, outcome.summary)
                print(f"  ✅ HANDLED — {outcome.action}: {outcome.summary}")
                if outcome.decision:
                    print(f"      model decision: {_short(json.dumps(outcome.decision, ensure_ascii=False), 150)}")
                break

            except BusinessReject as e:
                # poison message — retrying can NEVER help, so record terminal + move on.
                ledger.record(event.id, event.type, "rejected", attempt, str(e))
                print(f"  🚫 REJECTED (business) — {e}")
                print("      not retried, not dead-lettered — a semantic reject is terminal.")
                break

            except TransientError as e:
                if attempt >= MAX_ATTEMPTS:
                    dlq.add(event, error=str(e), attempts=attempt)
                    ledger.record(event.id, event.type, "dead_lettered", attempt, str(e))
                    print(f"  ☠️  DEAD-LETTER — transient failure persisted through "
                          f"{attempt} attempts → DLQ. last error: {e}")
                    break
                delay = backoff_delay(attempt)
                print(f"  ⏳ TRANSIENT — {e}")
                print(f"      retry {attempt + 1}/{MAX_ATTEMPTS} after backoff {delay:.2f}s…")
                time.sleep(delay)


# --------------------------------------------------------------------------- #
# Reports — the processed ledger table + the DLQ contents + the sub-point recap.
# --------------------------------------------------------------------------- #
def report(ledger: ProcessedLedger, dlq: DeadLetterQueue, meta: dict) -> None:
    _rule("PROCESSED LEDGER (idempotency registry) — one row per terminally-handled event id")
    head = f"  {'event_id':<10}  {'type':<14}  {'outcome':<13}  {'try':>3}  detail"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in ledger.rows():
        print(f"  {r.event_id:<10}  {r.type:<14}  {r.outcome:<13}  {r.attempts:>3}  {_short(r.detail, 60)}")

    counts: dict[str, int] = {}
    for r in ledger.rows():
        counts[r.outcome] = counts.get(r.outcome, 0) + 1
    print(f"\n  outcomes: {counts}")

    _rule("DEAD-LETTER QUEUE — events quarantined after exhausting retries (inspect / replay)")
    if not dlq.rows():
        print("  (empty)")
    for d in dlq.rows():
        print(f"  ☠️  {d.event_id} ({d.type}) after {d.attempts} attempts")
        print(f"      error: {d.error}")
        print(f"      payload: {_short(json.dumps(d.payload, ensure_ascii=False), 120)}")

    _rule("THE FOUR SUB-POINTS, IN THIS RUN")
    wh = ("a real localhost webhook POST" if meta.get("webhook_ok")
          else "the file queue (webhook bind unavailable in this env — see README)")
    print(f"1. listen to webhooks + queues — event #1 arrived via {wh}; the other six arrived on the")
    print("   append-only JSONL queue. Both sources converge on ONE worker that reacts to events, not chat.")
    print("2. execute on triggers — a dispatcher mapped each event TYPE to a handler; the model classified,")
    print("   drafted, summarized, and routed. The reasoning is the model's; the routing is deterministic.")
    print("3. idempotent execution — ord-1001 was delivered twice; the second delivery hit the processed")
    print("   ledger and became a no-op → handled exactly once despite at-least-once delivery.")
    print("4. dead-letter + retry — eml-2002 failed transiently twice then SUCCEEDED on the backed-off third")
    print("   try; ord-1002's downstream stayed down and DEAD-LETTERED after 3; ord-1003 was a BusinessReject")
    print("   (invalid order) and was NOT retried — poison ≠ transient.")
    print(f"\n  ledger → {LEDGER_PATH}\n  dlq    → {DLQ_PATH}")


def main() -> int:
    _rule("EVENT-TRIGGERED AUTOMATION AGENT — a mixed event stream on NVIDIA NIM (meta/llama-3.1-8b-instruct)")
    print("Fires on EVENTS, not chat. The model reasons inside each handler; the queue, the dedup,")
    print("the retry/backoff, and the dead-letter queue are deterministic Python. One stream exercises")
    print("all four sub-points: webhook+queue sources, trigger dispatch, idempotent dedup, and retry→DLQ.")

    queue = EventQueue(QUEUE_PATH, reset=True)
    ledger = ProcessedLedger(LEDGER_PATH, reset=True)
    dlq = DeadLetterQueue(DLQ_PATH, reset=True)
    dispatcher = Dispatcher()

    stream = build_stream()
    meta = produce(queue, stream)
    work(queue, ledger, dlq, dispatcher)
    report(ledger, dlq, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
