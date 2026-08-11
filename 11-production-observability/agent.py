"""The agent under observation — a small support-triage agent, fully instrumented.

This is the only file that talks to a model. It exists to be *watched*: every LLM call and
every tool call opens a span, so the trace tree, the dashboard, the alerts and the canary
verdict in the other two files are all derived from what actually happened here.

TWO DEPLOYED VERSIONS, and the difference between them is the whole demo:

  v1.4-stable   answer once the record satisfies its evidence checklist
                (`status`, `total_cents`) — the tool returns both, so one fetch is enough.

  v1.5-canary   same code path, one line changed: the checklist now also requires
                `refund_eta`. The upstream `lookup_order` tool does not return that field
                yet. Nothing crashes. Nothing returns junk. The agent simply decides it is
                "missing evidence", re-fetches the *identical* record, re-drafts, decides
                it is still missing evidence... until the step cap stops it.

That is the realistic production regression this project is about: a change that passes
review, throws no exception, and quietly triples the cost and latency of every request it
touches. You do not catch it with a try/except — you catch it because the trace shows the
same tool signature three times in one request, and because the canary's p95 and cost per
request jump against the baseline.

The loop-exit condition is DETERMINISTIC Python (`_missing_evidence`), not the model's
mood, so the failure reproduces exactly on every run. The model still does the real work:
it plans which tool to call, and it writes the customer-facing answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from common.client import DEFAULT_MODEL, get_client
from telemetry import ERROR, Tracer, record_llm_usage


# --------------------------------------------------------------------------- #
# The downstream world: one deterministic tool + a scripted outage.
# --------------------------------------------------------------------------- #
class ToolUnavailable(RuntimeError):
    """The downstream service is down for this record — a real failure, not a refusal."""


# Note what is NOT in these records: `refund_eta`. That absence is the canary's bug.
_ORDERS = {
    "ORD-4101": {"status": "delivered",   "total_cents": 8990,  "currency": "GBP",
                 "placed_on": "2026-07-28", "carrier": "Royal Mail"},
    "ORD-4102": {"status": "in_transit",  "total_cents": 15400, "currency": "GBP",
                 "placed_on": "2026-08-02", "carrier": "DPD"},
    "ORD-4103": {"status": "refunded",    "total_cents": 4599,  "currency": "GBP",
                 "placed_on": "2026-07-19", "carrier": "Evri"},
    "ORD-4104": {"status": "cancelled",   "total_cents": 12000, "currency": "GBP",
                 "placed_on": "2026-08-05", "carrier": "—"},
    "ORD-4105": {"status": "in_transit",  "total_cents": 2350,  "currency": "GBP",
                 "placed_on": "2026-08-07", "carrier": "DPD"},
    "ORD-4106": {"status": "delivered",   "total_cents": 33750, "currency": "GBP",
                 "placed_on": "2026-07-31", "carrier": "DHL"},
    "ORD-4107": {"status": "processing",  "total_cents": 6725,  "currency": "GBP",
                 "placed_on": "2026-08-09", "carrier": "pending"},
    "ORD-4108": {"status": "delivered",   "total_cents": 1899,  "currency": "GBP",
                 "placed_on": "2026-08-01", "carrier": "Evri"},
    "ORD-4109": {"status": "in_transit",  "total_cents": 21050, "currency": "GBP",
                 "placed_on": "2026-08-06", "carrier": "DPD"},
    "ORD-4110": {"status": "refunded",    "total_cents": 7800,  "currency": "GBP",
                 "placed_on": "2026-07-25", "carrier": "Royal Mail"},
    "ORD-4111": {"status": "processing",  "total_cents": 9990,  "currency": "GBP",
                 "placed_on": "2026-08-09", "carrier": "pending"},
}

# One record whose shard is "down" for this whole window — every fetch fails, so the
# failure is a sustained outage, not a flake, and the agent has to degrade.
OUTAGE_ORDER = "ORD-4199"


def lookup_order(order_id: str) -> dict:
    """The only tool. Deterministic; raises on the record whose shard is down."""
    if order_id == OUTAGE_ORDER:
        raise ToolUnavailable(f"order-service shard unavailable for {order_id} (503)")
    rec = _ORDERS.get(order_id)
    if rec is None:
        return {"order_id": order_id, "found": False}
    return {"order_id": order_id, "found": True, **rec}


TOOL_CATALOGUE = {
    "lookup_order": "look up one order by its id; returns status, total_cents, currency, "
                    "placed_on, carrier",
}


# --------------------------------------------------------------------------- #
# The two deployed versions.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AgentVersion:
    name: str
    required_evidence: tuple[str, ...]   # fields the record MUST have before answering
    max_steps: int
    note: str


STABLE = AgentVersion(
    name="v1.4-stable",
    required_evidence=("status", "total_cents"),
    max_steps=3,
    note="answers as soon as it has status + total — one fetch is enough")

CANARY = AgentVersion(
    name="v1.5-canary",
    required_evidence=("status", "total_cents", "refund_eta"),
    max_steps=3,
    note="also requires refund_eta, which the tool does not return yet -> retry loop")


def _missing_evidence(record: dict, version: AgentVersion) -> list[str]:
    """DETERMINISTIC loop-exit condition. No model involved — which is exactly why the
    canary's regression reproduces identically on every run."""
    return [f for f in version.required_evidence if record.get(f) in (None, "")]


# --------------------------------------------------------------------------- #
# A request and what came back.
# --------------------------------------------------------------------------- #
@dataclass
class Request:
    id: str
    text: str
    order_id: str


@dataclass
class Response:
    request_id: str
    version: str
    trace_id: str
    answer: str
    tool_calls: int = 0
    llm_calls: int = 0
    degraded: bool = False
    missing_at_exit: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The traced agent.
# --------------------------------------------------------------------------- #
class TracedAgent:
    def __init__(self, tracer: Tracer, model: str | None = None) -> None:
        self.tracer = tracer
        self.model = model or DEFAULT_MODEL
        self.client = get_client()

    # -- model plumbing: strict JSON, one repair retry, never crashes the request -- #
    def _chat_json(self, span, system: str, user: str, operation: str,
                   temperature: float) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        raw = ""
        for _ in (1, 2):
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature)
            raw = (resp.choices[0].message.content or "").strip()
            record_llm_usage(span, self.model, resp.usage, operation)
            parsed = _parse_json(raw)
            if parsed is not None:
                return parsed
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": "That was not valid JSON. Reply ONLY the JSON object."})
        return {"_unparsed": raw}

    # -- the request path ---------------------------------------------------- #
    def handle(self, req: Request, version: AgentVersion) -> Response:
        t = self.tracer
        with t.span("agent.request", kind="agent",
                    **{"deploy.version": version.name,
                       "request.id": req.id,
                       "request.text": req.text}) as root:
            resp = Response(request_id=req.id, version=version.name,
                            trace_id=root.trace_id, answer="")

            # ---- 1. PLAN — the model picks the tool + args. ---- #
            with t.span("llm.plan", kind="llm") as sp:
                plan = self._plan(sp, req)
                resp.llm_calls += 1
            tool_name = plan.get("tool", "lookup_order")
            order_id = plan.get("order_id") or req.order_id
            root.set(**{"agent.planned_tool": tool_name, "agent.order_id": order_id})

            record: dict = {}
            answer, confidence = "", 0.0
            missing: list[str] = list(version.required_evidence)

            # ---- 2. ACT + ANSWER, bounded. The exit test is deterministic. ---- #
            for step in range(1, version.max_steps + 1):
                signature = f"{tool_name}(order_id={order_id})"
                with t.span(f"tool.{tool_name}", kind="tool",
                            **{"tool.name": tool_name,
                               "tool.signature": signature,
                               "agent.step": step}) as sp:
                    try:
                        record = lookup_order(order_id)
                        sp.set(**{"tool.found": bool(record.get("found"))})
                    except ToolUnavailable as exc:
                        sp.record_error(str(exc))
                        record = {"order_id": order_id, "found": False,
                                  "error": str(exc)}
                    resp.tool_calls += 1

                with t.span("llm.answer", kind="llm", **{"agent.step": step}) as sp:
                    answer, confidence = self._answer(sp, req, record, version, missing)
                    resp.llm_calls += 1

                if "error" in record:                 # downstream is down: degrade, do not spin
                    resp.degraded = True
                    root.record_error(record["error"])
                    root.set(**{"agent.degraded": True})
                    break

                missing = _missing_evidence(record, version)
                root.set(**{"agent.missing_evidence": ",".join(missing) or "none"})
                if not missing:                       # <- the loop-exit test
                    break

            resp.answer = answer
            resp.missing_at_exit = missing if not resp.degraded else []
            root.set(**{"agent.tool_calls": resp.tool_calls,
                        "agent.llm_calls": resp.llm_calls,
                        "agent.answer_confidence": confidence,
                        "agent.looped": resp.tool_calls > 1 and not resp.degraded})
            return resp

    # -- the two model roles ------------------------------------------------- #
    def _plan(self, span, req: Request) -> dict:
        catalogue = "\n".join(f"  - {n}: {d}" for n, d in TOOL_CATALOGUE.items())
        system = ("You are the planner of a customer-support agent. Choose the ONE tool to "
                  "call and the order id to call it with. Return STRICT JSON only, no "
                  'markdown: {"tool": <tool name>, "order_id": <the order id>, '
                  '"reason": <one short sentence>}')
        user = f"TOOLS:\n{catalogue}\n\nCUSTOMER MESSAGE:\n{req.text}"
        d = self._chat_json(span, system, user, "plan", temperature=0.0)
        # Deterministic validation of a model choice — never trust a name it invented.
        tool = str(d.get("tool", "")).strip()
        if tool not in TOOL_CATALOGUE:
            tool = "lookup_order"
        order_id = str(d.get("order_id", "")).strip().upper()
        if not re.fullmatch(r"ORD-\d{4}", order_id):
            order_id = req.order_id
        span.set(**{"plan.tool": tool, "plan.order_id": order_id,
                    "plan.reason": str(d.get("reason", ""))[:160]})
        return {"tool": tool, "order_id": order_id}

    def _answer(self, span, req: Request, record: dict, version: AgentVersion,
                missing: list[str]) -> tuple[str, float]:
        if record.get("error"):
            system = ("You are a customer-support agent. The order lookup FAILED, so you have "
                      "NO order data. Apologise briefly, say the system is temporarily "
                      "unavailable, and tell the customer you will follow up — do NOT invent "
                      "any order details. Return STRICT JSON only: "
                      '{"answer": <2 sentences max>, "confidence": <0.0-1.0>}')
            user = f"CUSTOMER MESSAGE:\n{req.text}\n\nLOOKUP ERROR:\n{record['error']}"
        else:
            gap = (f"\n\nSTILL MISSING (policy requires it): {', '.join(missing)}"
                   if missing else "")
            system = ("You are a customer-support agent. Answer the customer using ONLY the "
                      "order record given. Be specific: state the status and the amount "
                      "(total_cents is in pence). Two sentences maximum. Return STRICT JSON "
                      'only, no markdown: {"answer": <the reply>, "confidence": <0.0-1.0>}')
            user = (f"CUSTOMER MESSAGE:\n{req.text}\n\nORDER RECORD:\n"
                    f"{json.dumps(record, ensure_ascii=False)}{gap}")
        d = self._chat_json(span, system, user, "answer", temperature=0.2)
        answer = str(d.get("answer", "")).strip() or "(no answer returned)"
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 0.0))))
        except (TypeError, ValueError):
            conf = 0.0
        span.set(**{"answer.confidence": conf, "answer.chars": len(answer)})
        return answer, conf


# --------------------------------------------------------------------------- #
# Tolerant JSON extraction (the same helper the rest of the series uses).
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
