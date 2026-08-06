"""The Event-Triggered Automation Agent — the DISPATCHER and its HANDLERS.

This is the only place a model call happens. A `Dispatcher` maps an event TYPE to a
handler; each handler uses `meta/llama-3.1-8b-instruct` to REASON about one event —
classify + draft a reply, summarize + route, triage an order — and returns an `Outcome`.
The model does the judgement; the routing (type → handler), the retry, the dedup, and the
dead-lettering are deterministic Python in `queue.py` and the worker in `run.py`.

Two failure kinds are modelled the way real automation systems distinguish them, because
they demand OPPOSITE responses:

  TransientError — a downstream dependency hiccupped (a payment gateway timed out, the
                   ticket API 503'd). The work was fine; try again. → the worker RETRIES
                   with backoff, and only DEAD-LETTERS after N attempts.
  BusinessReject — the event is semantically invalid (an order with a non-positive total,
                   an email with no body). Retrying will NEVER help. → the worker records
                   it as terminally `rejected` and does NOT retry, does NOT dead-letter.

Conflating the two is the classic event-agent bug: you either retry a poison message
forever, or you dead-letter a blip that a single retry would have fixed. They are kept
distinct here on purpose.

A handler's shape is deliberately uniform:
  1. VALIDATE (deterministic) — reject semantically-bad events BEFORE paying the model.
  2. REASON (the model) — the real NVIDIA NIM call that classifies / drafts / summarizes.
  3. DELIVER (side effect) — push the result to a simulated downstream that can hiccup.

Only step 3 can raise a TransientError, and it does so via a clearly-named, deterministic
fault switch in the event payload (`_inject.transient_fails`) so the demo reproduces — a
stand-in for a genuinely flaky dependency, documented as such.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from common.client import DEFAULT_MODEL, get_client

logger = logging.getLogger("event-agent")


# --------------------------------------------------------------------------- #
# Failure taxonomy — the two errors the worker treats in OPPOSITE ways.
# --------------------------------------------------------------------------- #
class TransientError(Exception):
    """A retriable failure — a downstream dependency hiccupped. Try again with backoff."""


class BusinessReject(Exception):
    """A permanent, semantic rejection — the event is invalid. Retrying cannot help."""


# --------------------------------------------------------------------------- #
# What a handler returns on success.
# --------------------------------------------------------------------------- #
@dataclass
class Outcome:
    action: str                       # what the handler did, one word for the ledger
    summary: str                      # human-readable one-liner
    decision: dict = field(default_factory=dict)   # the model's structured decision
    delivered_to: str = ""            # the downstream the result was pushed to


# --------------------------------------------------------------------------- #
# The simulated downstream — the ONLY place a TransientError comes from.
# --------------------------------------------------------------------------- #
def _deliver(event, attempt: int, target: str) -> str:
    """Push a handler's result to a downstream system (a ticket API, a fulfilment queue…).

    Real downstreams fail transiently. To make that reproducible, the event payload may
    carry a fault switch `_inject.transient_fails = N`: this delivery raises a TransientError
    on attempts 1..N and succeeds afterwards. `N` larger than the worker's max attempts
    models a dependency that stays down long enough to exhaust retries → dead-letter.
    """
    fails = int(event.payload.get("_inject", {}).get("transient_fails", 0) or 0)
    if attempt <= fails:
        raise TransientError(
            f"downstream '{target}' unavailable (attempt {attempt} of a simulated "
            f"{fails}-attempt outage) — HTTP 503")
    return target


# --------------------------------------------------------------------------- #
# The dispatcher — TYPE → handler, plus the shared model call.
# --------------------------------------------------------------------------- #
class Dispatcher:
    """Routes an event to the handler registered for its type, and owns the one model call
    every handler shares. Unknown event types are a BusinessReject (nothing can handle
    them — retrying will not conjure a handler)."""

    def __init__(self, model: str = DEFAULT_MODEL, on_event=None) -> None:
        self.model = model
        self.client = get_client()
        self.on_event = on_event
        self._handlers = {
            "new_order": self._handle_new_order,
            "support_email": self._handle_support_email,
            "file_uploaded": self._handle_file_uploaded,
        }

    def handles(self, event_type: str) -> bool:
        return event_type in self._handlers

    def dispatch(self, event, attempt: int) -> Outcome:
        """Run the handler for `event.type` once (one attempt). May raise TransientError
        (retriable) or BusinessReject (terminal). `attempt` is the 1-based try number,
        threaded through so the simulated downstream knows which try this is."""
        handler = self._handlers.get(event.type)
        if handler is None:
            raise BusinessReject(f"no handler registered for event type '{event.type}'")
        return handler(event, attempt)

    # ---- the shared model call: strict-JSON classify/draft, tolerant parse + one retry --- #
    def _reason(self, system: str, user: str) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        raw = ""
        for _ in (1, 2):
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.2)
            raw = (resp.choices[0].message.content or "").strip()
            parsed = _parse_json(raw)
            if parsed is not None:
                return parsed
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": "That was not valid JSON. Reply ONLY the JSON object."})
        # Never crash the worker on a malformed model reply — degrade to a minimal decision.
        return {"_unparsed": raw}

    def _emit(self, kind: str, **data) -> None:
        if self.on_event:
            self.on_event(kind, data)

    # ------------------------------------------------------------------ #
    # Handler: new_order — validate → triage (model) → hand to fulfilment.
    # ------------------------------------------------------------------ #
    def _handle_new_order(self, event, attempt: int) -> Outcome:
        p = event.payload
        total = p.get("total")
        # 1. VALIDATE (deterministic) — a non-positive / missing total is a poison order.
        if not isinstance(total, (int, float)) or total <= 0:
            raise BusinessReject(f"order total must be a positive number, got {total!r}")
        if not p.get("items"):
            raise BusinessReject("order has no line items")

        # 2. REASON (model) — triage priority + draft a customer confirmation line.
        system = (
            "You are an order-triage agent. Given an order, decide fulfilment priority and "
            "draft one short customer-facing confirmation sentence. Return STRICT JSON only: "
            '{"priority": "expedite"|"standard", "reason": <short>, "customer_message": <one sentence>}. '
            "Choose `expedite` for high-value (> $500) or express-shipping orders, else `standard`. "
            "No markdown, no extra keys.")
        user = json.dumps({"order_id": p.get("order_id"), "customer": p.get("customer"),
                           "items": p.get("items"), "total": total,
                           "shipping": p.get("shipping", "standard")})
        d = self._reason(system, user)
        priority = str(d.get("priority", "standard")).lower()
        priority = priority if priority in ("expedite", "standard") else "standard"

        # 3. DELIVER — enqueue to the fulfilment system (can hiccup transiently).
        target = _deliver(event, attempt, "fulfilment-queue")
        return Outcome(
            action="order_triaged",
            summary=f"order {p.get('order_id')} → {priority.upper()} → {target}",
            decision={"priority": priority, "reason": d.get("reason", ""),
                      "customer_message": d.get("customer_message", "")},
            delivered_to=target)

    # ------------------------------------------------------------------ #
    # Handler: support_email — validate → classify + draft reply (model) → ticket API.
    # ------------------------------------------------------------------ #
    def _handle_support_email(self, event, attempt: int) -> Outcome:
        p = event.payload
        body = (p.get("body") or "").strip()
        if not body:
            raise BusinessReject("support email has an empty body")

        system = (
            "You are a support-desk triage agent. Classify the email and draft a brief, "
            "empathetic first reply. Return STRICT JSON only: "
            '{"category": "billing"|"technical"|"complaint"|"general", '
            '"sentiment": "positive"|"neutral"|"negative", "priority": "high"|"normal", '
            '"draft_reply": <2 sentences max>}. No markdown, no extra keys.')
        user = json.dumps({"from": p.get("from"), "subject": p.get("subject"), "body": body})
        d = self._reason(system, user)
        category = str(d.get("category", "general")).lower()

        target = _deliver(event, attempt, "ticketing-system")
        return Outcome(
            action="email_triaged",
            summary=f"email from {p.get('from')} → {category}/{d.get('priority', 'normal')} → {target}",
            decision={"category": category, "sentiment": d.get("sentiment", "neutral"),
                      "priority": d.get("priority", "normal"), "draft_reply": d.get("draft_reply", "")},
            delivered_to=target)

    # ------------------------------------------------------------------ #
    # Handler: file_uploaded — validate → summarize + route (model) → destination team.
    # ------------------------------------------------------------------ #
    def _handle_file_uploaded(self, event, attempt: int) -> Outcome:
        p = event.payload
        text = (p.get("text") or "").strip()
        if not p.get("filename"):
            raise BusinessReject("file event has no filename")
        if not text:
            raise BusinessReject("file has no extractable text")

        system = (
            "You are a document-routing agent. Given a file's name and extracted text, "
            "identify the document type, summarize it in one sentence, and route it to the "
            "right team. Return STRICT JSON only: "
            '{"doc_type": <short label>, "summary": <one sentence>, '
            '"route_to": "finance"|"legal"|"engineering"|"hr"|"general"}. No markdown.')
        user = json.dumps({"filename": p.get("filename"), "text": text[:1200]})
        d = self._reason(system, user)
        route_to = str(d.get("route_to", "general")).lower()

        target = _deliver(event, attempt, f"team:{route_to}")
        return Outcome(
            action="file_routed",
            summary=f"{p.get('filename')} → {d.get('doc_type', 'document')} → {target}",
            decision={"doc_type": d.get("doc_type", "document"), "summary": d.get("summary", ""),
                      "route_to": route_to},
            delivered_to=target)


# --------------------------------------------------------------------------- #
# Tolerant JSON extraction (same approach the rest of the series uses).
# --------------------------------------------------------------------------- #
def _parse_json(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.split("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
