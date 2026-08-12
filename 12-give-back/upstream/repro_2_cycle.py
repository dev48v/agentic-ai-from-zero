"""Reproducer 2 — the A,B,A,B cycle that a repeat COUNTER cannot see.

Reproducer 1 shows the obvious loop: the same call over and over. This one is the version
that survives a naive fix, and it is the reason `LoopFuse` ships a second detector.

The agent alternates between two tools — check the order, then check the customer, then
check the order again — with identical arguments each time. No single signature ever
reaches a repeat threshold of 3 within the window a counting rule looks at, so:

  * LangGraph's `recursion_limit` does not care (it only counts supersteps);
  * a per-signature COUNT rule — which is exactly what Project 11 of this series shipped —
    stays silent for a long time;
  * the cycle detector stops it on the first completed repeat of the block.

Deterministic, no API key, no cost:

    python upstream/repro_2_cycle.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.graph import START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from agentfuse import FuseBox, LoopFuse, ToolCall  # noqa: E402
from agentfuse.adapters.langgraph_guard import guard_tool_node  # noqa: E402

COUNTS = {"llm": 0, "tool": 0}
CYCLE = [("check_order", {"order_id": "42"}), ("check_customer", {"customer_id": "c-7"})]


@tool
def check_order(order_id: str) -> str:
    """Look up an order by id."""
    COUNTS["tool"] += 1
    return "order 42: status PENDING, customer c-7"


@tool
def check_customer(customer_id: str) -> str:
    """Look up a customer by id."""
    COUNTS["tool"] += 1
    return "customer c-7: 1 open order, id 42"


class State(TypedDict):
    messages: Annotated[list, add_messages]


def ping_pong_model(state: State) -> dict:
    """Each tool's answer points at the other tool. The agent never gets new information
    and never notices, because every individual step looks reasonable."""
    name, args = CYCLE[COUNTS["llm"] % len(CYCLE)]
    COUNTS["llm"] += 1
    return {"messages": [AIMessage(
        content="",
        tool_calls=[{"name": name, "args": dict(args), "id": f"call_{COUNTS['llm']}"}],
    )]}


def build(tools_node, self_routing: bool = False):
    builder = StateGraph(State)
    builder.add_node("model", ping_pong_model)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "model")
    builder.add_edge("model", "tools")
    if not self_routing:
        builder.add_edge("tools", "model")
    return builder.compile()


def reset() -> None:
    COUNTS["llm"] = COUNTS["tool"] = 0


def count_only_rule(limit: int = 40) -> dict:
    """Project 11's own rule, replayed offline: count each signature, alert at 3.

    Answers the question honestly — how long WOULD the shipped version have taken to
    notice? — instead of asserting that the new detector is better.
    """
    counts: dict[str, int] = {}
    for step in range(1, limit + 1):
        name, args = CYCLE[(step - 1) % len(CYCLE)]
        sig = ToolCall(name, args).signature
        counts[sig] = counts.get(sig, 0) + 1
        if counts[sig] >= 3:
            return {"fired_at_call": step}
    return {"fired_at_call": None}


def run(guarded: bool, recursion_limit: int) -> dict:
    reset()
    box = FuseBox(loop=LoopFuse(repeat_threshold=3, max_cycle_len=4))
    node = (guard_tool_node(ToolNode([check_order, check_customer]), box,
                            resume_node="model")
            if guarded else ToolNode([check_order, check_customer]))
    graph = build(node, self_routing=guarded)
    started = time.time()
    outcome, detail = "completed", ""
    try:
        state = graph.invoke({"messages": [HumanMessage("is order 42 shipped?")]},
                             {"recursion_limit": recursion_limit})
        detail = str(state["messages"][-1].content)
    except GraphRecursionError as exc:
        outcome, detail = "GraphRecursionError", str(exc).splitlines()[0]
    return {"outcome": outcome, "detail": detail, "seconds": round(time.time() - started, 2),
            "llm_calls": COUNTS["llm"], "tool_calls": COUNTS["tool"],
            "blocks": [v.line() for v in box.blocks]}


def _print(title: str, result: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"  outcome     : {result['outcome']}")
    print(f"  model calls : {result['llm_calls']}")
    print(f"  tool calls  : {result['tool_calls']}")
    print(f"  wall time   : {result['seconds']}s")
    if result.get("detail"):
        print(f"  detail      : {result['detail'][:170]}")
    for line in result.get("blocks", []):
        print(f"  fuse        : {line}")


if __name__ == "__main__":
    from importlib.metadata import version

    limit = int(os.getenv("REPRO_LIMIT", "40"))
    print("=" * 78)
    print("REPRO 2 — the A,B,A,B cycle a repeat counter cannot see")
    print("=" * 78)
    print(f"langgraph       : {version('langgraph')}")
    print(f"recursion_limit : {limit}  (kept small — the point is WHEN each guard fires)")

    counted = count_only_rule(limit)
    print(f"\ncount-only rule (what Project 11 shipped): "
          f"would first fire at tool call {counted['fired_at_call']}")

    _print("PART A  stock LangGraph", run(guarded=False, recursion_limit=limit))
    _print("PART B  + agentfuse LoopFuse (cycle detection on)",
           run(guarded=True, recursion_limit=limit))
