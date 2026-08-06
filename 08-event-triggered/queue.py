"""The deterministic infrastructure of the Event-Triggered Automation Agent — the
event model, the append-only queue, the idempotency ledger, the dead-letter queue,
and the retry backoff. NO model calls live here on purpose.

An event-driven agent needs plumbing that is boring and correct, so that the only
interesting (and non-deterministic) thing in the system is the model's reasoning inside
a handler. Everything in this file is plain Python over JSONL files — the same event
always lands, is deduped, retried, and dead-lettered the same way:

  Event            — an immutable envelope (id, type, payload, source, ts). The `id` is
                     the idempotency key; two deliveries of the same id are the SAME event.
  EventQueue       — an append-only JSONL "queue" file. Producers (a script, or the HTTP
                     webhook receiver) `append()`; the worker `claim()`s in FIFO order.
                     Append-only + a claim cursor is exactly how a real broker (SQS/Kafka)
                     looks from the consumer's side, minus the network.
  ProcessedLedger  — one line per TERMINALLY-handled event id. `seen()` makes a re-delivered
                     event a no-op — this is idempotent execution. At-least-once delivery is
                     the norm in real systems, so dedup is not optional.
  DeadLetterQueue  — where an event lands after it exhausts its retries: the event, the last
                     error, and the attempt count, so a human can inspect and replay it.
  backoff_delay    — bounded exponential backoff between retry attempts.

The split is the whole point of the series: the MODEL decides/acts inside a handler; the
QUEUE, the DEDUP, the RETRY, and the DLQ are deterministic Python, so the pipeline is
reproducible and auditable no matter what the model says.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


# --------------------------------------------------------------------------- #
# The event envelope — `id` is the idempotency key.
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """One event on the bus. `id` identifies the event across deliveries — two POSTs of
    the same order are ONE event with the same id, which is what makes dedup possible.
    `payload` is the domain data a handler reasons over; `source` records where it came
    from (`queue` for the file producer, `webhook` for an HTTP POST)."""
    id: str
    type: str
    payload: dict = field(default_factory=dict)
    source: str = "queue"
    ts: str = field(default_factory=_now_iso)

    @staticmethod
    def new(type: str, payload: dict, id: str | None = None, source: str = "queue") -> "Event":
        return Event(id=id or f"evt_{uuid.uuid4().hex[:8]}", type=type, payload=payload, source=source)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Event":
        return Event(id=d["id"], type=d["type"], payload=d.get("payload", {}),
                     source=d.get("source", "queue"), ts=d.get("ts", _now_iso()))


# --------------------------------------------------------------------------- #
# The queue — append-only JSONL + a FIFO claim cursor.
# --------------------------------------------------------------------------- #
class EventQueue:
    """An append-only JSONL file that behaves like a broker's inbound topic.

    Producers `append()` a line; the worker `claim()`s the next un-claimed line in FIFO
    order. Because `claim()` re-reads the file, an event appended MID-RUN (e.g. by the HTTP
    webhook receiver on another thread) is picked up by the very next claim — the worker
    reacts to events, it does not need to be handed them.
    """

    def __init__(self, path: str, reset: bool = True) -> None:
        self.path = path
        self._cursor = 0                     # how many events have been claimed
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if reset and os.path.exists(path):
            os.remove(path)

    def append(self, event: Event) -> Event:
        """Enqueue one event (used by both the file producer and the webhook receiver)."""
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return event

    def _all(self) -> list[Event]:
        if not os.path.exists(self.path):
            return []
        out: list[Event] = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(Event.from_dict(json.loads(line)))
        return out

    def claim(self) -> Event | None:
        """Return the next un-claimed event in FIFO order, or None if the queue is drained.
        Re-reads the file each call so late-arriving (e.g. webhook) events are seen."""
        events = self._all()
        if self._cursor >= len(events):
            return None
        ev = events[self._cursor]
        self._cursor += 1
        return ev

    def depth(self) -> int:
        """How many un-claimed events remain right now."""
        return max(0, len(self._all()) - self._cursor)


# --------------------------------------------------------------------------- #
# Idempotent execution — the processed ledger.
# --------------------------------------------------------------------------- #
@dataclass
class LedgerRow:
    event_id: str
    type: str
    outcome: str                 # "handled" | "rejected" | "dead_lettered"
    attempts: int
    detail: str
    ts: str = field(default_factory=_now_iso)


class ProcessedLedger:
    """Append-only idempotency ledger — one row per event id that reached a TERMINAL state
    (handled, business-rejected, or dead-lettered). `seen()` is checked BEFORE any work, so
    a re-delivered event is a no-op — the handler (and its model call + side effects) never
    runs twice for the same id. This is the whole of "idempotent execution": at-least-once
    delivery in, exactly-once effect out."""

    def __init__(self, path: str, reset: bool = True) -> None:
        self.path = path
        self._rows: dict[str, LedgerRow] = {}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if reset and os.path.exists(path):
            os.remove(path)
        elif os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        self._rows[d["event_id"]] = LedgerRow(**d)

    def seen(self, event_id: str) -> bool:
        return event_id in self._rows

    def get(self, event_id: str) -> LedgerRow | None:
        return self._rows.get(event_id)

    def record(self, event_id: str, type: str, outcome: str, attempts: int, detail: str) -> LedgerRow:
        row = LedgerRow(event_id=event_id, type=type, outcome=outcome, attempts=attempts, detail=detail)
        self._rows[event_id] = row
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
        return row

    def rows(self) -> list[LedgerRow]:
        return list(self._rows.values())


# --------------------------------------------------------------------------- #
# Dead-letter queue — where an event lands after exhausting retries.
# --------------------------------------------------------------------------- #
@dataclass
class DeadLetter:
    event_id: str
    type: str
    attempts: int
    error: str
    payload: dict
    ts: str = field(default_factory=_now_iso)


class DeadLetterQueue:
    """A durable JSONL sink for events that failed permanently. Keeping the event, the
    attempt count, and the LAST error means a human (or a replay job) can inspect and
    re-drive it later — a failed event is quarantined, never silently dropped and never
    left to block the rest of the stream."""

    def __init__(self, path: str, reset: bool = True) -> None:
        self.path = path
        self._rows: list[DeadLetter] = []
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if reset and os.path.exists(path):
            os.remove(path)

    def add(self, event: Event, error: str, attempts: int) -> DeadLetter:
        dl = DeadLetter(event_id=event.id, type=event.type, attempts=attempts,
                        error=error, payload=event.payload)
        self._rows.append(dl)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(dl), ensure_ascii=False) + "\n")
        return dl

    def rows(self) -> list[DeadLetter]:
        return list(self._rows)


# --------------------------------------------------------------------------- #
# Bounded exponential backoff between retry attempts.
# --------------------------------------------------------------------------- #
def backoff_delay(attempt: int, base: float = 0.1, factor: float = 2.0, cap: float = 2.0) -> float:
    """Delay (seconds) BEFORE retry #`attempt` (1-based): base * factor**(attempt-1), capped.
    Bounded so a run cannot hang; the cap models the ceiling you'd set in production."""
    return min(cap, base * (factor ** max(0, attempt - 1)))
