"""A few tools with DECLARED risk — the raw material the approval gate reasons over.

Every tool carries two things the gate reads deterministically:

  risk          — "safe" (auto-runnable) or "high" (never runs without a human OK).
  capabilities  — a set of capability tags; some are on a SENSITIVE list
                  (send_email / spend_money / delete / external_write) so that even
                  a tool someone forgot to mark high-risk still trips the gate.

The line-up (name — risk — capabilities):

  lookup_order  — safe — {read}                     read-only; auto-approved.
  send_email    — high — {send_email, external_write} sends a real-looking email.
  issue_refund  — high — {spend_money}               moves money.

The two high-risk tools are exactly the actions you never want an 8B model firing on
its own — emailing a customer and refunding money. `lookup_order` is the safe read the
agent can just do. The gate (approval.py) decides which is which; the tools themselves
only DECLARE their risk and DO the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# --------------------------------------------------------------------------- #
# Uniform tool return (mirrors the rest of the series).
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    data: dict = field(default_factory=dict)
    error: str = ""

    def as_line(self) -> str:
        return self.output if self.ok else f"ERROR: {self.error}"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    risk: str                       # "safe" | "high"
    capabilities: frozenset[str]
    args_schema: dict
    func: Callable[[dict], ToolResult]

    def run(self, args: dict) -> ToolResult:
        """Invoke the tool; enforce required args; never crash the agent."""
        required = self.args_schema.get("required", [])
        missing = [k for k in required if k not in (args or {}) or args.get(k) in (None, "")]
        if missing:
            return ToolResult(ok=False, error=f"missing required arg(s) {missing} for '{self.name}'")
        try:
            return self.func(args or {})
        except Exception as exc:  # noqa: BLE001 — a tool must never crash the loop
            return ToolResult(ok=False, error=f"{type(exc).__name__}: {exc}")


KNOWN_RISKS = frozenset({"safe", "high"})
REGISTRY: dict[str, Tool] = {}


def tool(*, name: str, description: str, risk: str,
         capabilities: set[str], args_schema: dict) -> Callable:
    """Self-registering decorator — a tool DECLARES its risk + capabilities here."""
    if risk not in KNOWN_RISKS:
        raise ValueError(f"tool '{name}' declares unknown risk '{risk}'")

    def deco(fn: Callable[[dict], ToolResult]) -> Callable[[dict], ToolResult]:
        REGISTRY[name] = Tool(name=name, description=description, risk=risk,
                              capabilities=frozenset(capabilities),
                              args_schema=args_schema, func=fn)
        return fn
    return deco


# --------------------------------------------------------------------------- #
# Side-effect sinks (so a "sent" email / "issued" refund is inspectable, not real).
# --------------------------------------------------------------------------- #
OUTBOX: list[dict] = []      # emails send_email "sent"
REFUNDS: list[dict] = []     # refunds issue_refund "paid"

# A tiny order book lookup_order reads from.
_ORDERS = {
    "A-1042": {
        "order_id": "A-1042",
        "customer": "Priya Menon",
        "email": "priya.menon@example.com",
        "item": "Ceramic pour-over coffee set",
        "total": 84.00,
        "shipping_fee": 60.00,
        "status": "delivered — reported damaged",
    },
}


# --------------------------------------------------------------------------- #
# lookup_order — SAFE, read-only. The agent may run this without asking.
# --------------------------------------------------------------------------- #
@tool(
    name="lookup_order",
    description="Look up an order by id (customer, item, total, shipping fee, status). Read-only.",
    risk="safe",
    capabilities={"read"},
    args_schema={"properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
)
def lookup_order(args: dict) -> ToolResult:
    oid = (args.get("order_id", "") or "").strip().upper()
    order = _ORDERS.get(oid)
    if not order:
        return ToolResult(ok=False, error=f"no order on file for '{oid}'")
    return ToolResult(
        ok=True,
        output=(f"{order['order_id']}: {order['customer']} <{order['email']}> — "
                f"{order['item']}, total ${order['total']:.2f}, shipping "
                f"${order['shipping_fee']:.2f}, status: {order['status']}"),
        data=dict(order),
    )


# --------------------------------------------------------------------------- #
# send_email — HIGH RISK + sensitive capability. Never auto-runs.
# --------------------------------------------------------------------------- #
@tool(
    name="send_email",
    description="Send an email to a customer. Args: to, subject, body. IRREVERSIBLE once sent.",
    risk="high",
    capabilities={"send_email", "external_write"},
    args_schema={
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    },
)
def send_email(args: dict) -> ToolResult:
    msg = {"to": args["to"].strip(), "subject": args["subject"].strip(),
           "body": args["body"].strip()}
    OUTBOX.append(msg)
    return ToolResult(
        ok=True,
        output=f"email #{len(OUTBOX)} sent to {msg['to']} — subject: “{msg['subject']}”",
        data={"outbox_index": len(OUTBOX), **msg},
    )


# --------------------------------------------------------------------------- #
# issue_refund — HIGH RISK + spends money. Never auto-runs.
# --------------------------------------------------------------------------- #
@tool(
    name="issue_refund",
    description="Refund money to a customer's order. Args: order_id, amount (number). MOVES MONEY.",
    risk="high",
    capabilities={"spend_money"},
    args_schema={
        "properties": {
            "order_id": {"type": "string"},
            "amount": {"type": "number"},
        },
        "required": ["order_id", "amount"],
    },
)
def issue_refund(args: dict) -> ToolResult:
    oid = (args.get("order_id", "") or "").strip().upper()
    amount = round(float(args.get("amount", 0)), 2)
    if amount <= 0:
        return ToolResult(ok=False, error=f"refund amount must be positive, got {amount}")
    entry = {"order_id": oid, "amount": amount}
    REFUNDS.append(entry)
    return ToolResult(
        ok=True,
        output=f"refund #{len(REFUNDS)}: ${amount:.2f} issued against order {oid}",
        data={"refund_index": len(REFUNDS), **entry},
    )


def all_tools() -> dict[str, Tool]:
    return dict(REGISTRY)


def catalog_for_prompt() -> str:
    """The tool menu shown to the model when it plans — name, risk, args, purpose."""
    lines = []
    for t in REGISTRY.values():
        props = ", ".join(t.args_schema.get("properties", {}).keys()) or "—"
        lines.append(f"  - {t.name}  (risk={t.risk}; args: {props})\n      {t.description}")
    return "\n".join(lines)
