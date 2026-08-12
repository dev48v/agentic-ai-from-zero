"""Reproducer 1 — LangGraph has a STEP cap, not a PROGRESS check.

Deterministic: the "model" here is a plain function that always emits the same tool call,
so this reproduces byte-for-byte on any machine, with no API key and at zero cost. That is
deliberate — a bug report nobody can run is a bug report nobody fixes.

    python upstream/repro_1_no_progress.py

What it shows:

  PART A  a stock LangGraph graph whose agent asks the identical question forever. Nothing
          stops it until `recursion_limit` is exhausted, and the run then ends in a
          GraphRecursionError — an exception, after all the work is already paid for.
  PART B  the identical graph with `agentfuse.adapters.langgraph_guard.guard_tool_node`
          wrapped around the tool node. It stops on the 3rd identical call, with a reason,
          and returns a normal result instead of raising.

Every number printed is measured in this process.
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

from agentfuse import FuseBox, LoopFuse  # noqa: E402
from agentfuse.adapters.langgraph_guard import guard_tool_node  # noqa: E402

COUNTS = {"llm": 0, "tool": 0}


@tool
def search_orders(query: str) -> str:
    """Search the order database for a customer order."""
    COUNTS["tool"] += 1
    return "no results found"


class State(TypedDict):
    messages: Annotated[list, add_messages]


def stuck_model(state: State) -> dict:
    """A model that never gives up on the same idea.

    Not a strawman: this is what a small model does when the tool keeps answering
    "no results" and the prompt keeps insisting the answer exists. Projects 3 and 10 of
    this series both hit it live before the self-critic and the hard checks went in.
    """
    COUNTS["llm"] += 1
    return {"messages": [AIMessage(
        content="",
        tool_calls=[{"name": "search_orders",
                     "args": {"query": "order 42"},
                     "id": f"call_{COUNTS['llm']}"}],
    )]}


def build(tools_node, self_routing: bool = False):
    """The classic two-node agent loop: model -> tools -> model -> ...

    `self_routing=True` drops the static `tools -> model` edge because the guarded node
    returns a `Command` and routes itself. LangGraph takes a static edge IN ADDITION to a
    Command's goto, so leaving the edge in would quietly cancel the stop.
    """
    builder = StateGraph(State)
    builder.add_node("model", stuck_model)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "model")
    builder.add_edge("model", "tools")
    if not self_routing:
        builder.add_edge("tools", "model")
    return builder.compile()


def reset() -> None:
    COUNTS["llm"] = COUNTS["tool"] = 0


def part_a(recursion_limit: int | None) -> dict:
    """Stock LangGraph. `recursion_limit=None` means 'whatever the library defaults to'."""
    reset()
    graph = build(ToolNode([search_orders]))
    config = {} if recursion_limit is None else {"recursion_limit": recursion_limit}
    started = time.time()
    outcome, detail = "completed", ""
    try:
        graph.invoke({"messages": [HumanMessage("where is order 42?")]}, config)
    except GraphRecursionError as exc:
        outcome, detail = "GraphRecursionError", str(exc).splitlines()[0]
    return {"outcome": outcome, "detail": detail, "seconds": round(time.time() - started, 2),
            "llm_calls": COUNTS["llm"], "tool_calls": COUNTS["tool"]}


def part_b(recursion_limit: int | None) -> dict:
    """Same graph, tool node wrapped in the fuse."""
    reset()
    box = FuseBox(loop=LoopFuse(repeat_threshold=3))
    graph = build(guard_tool_node(ToolNode([search_orders]), box, resume_node="model"),
                  self_routing=True)
    config = {} if recursion_limit is None else {"recursion_limit": recursion_limit}
    started = time.time()
    outcome, detail = "completed", ""
    try:
        state = graph.invoke({"messages": [HumanMessage("where is order 42?")]}, config)
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
    print(f"  tool calls  : {result['tool_calls']}   (all identical: "
          f"search_orders(query='order 42'))")
    print(f"  wall time   : {result['seconds']}s")
    if result.get("detail"):
        print(f"  detail      : {result['detail'][:160]}")
    for line in result.get("blocks", []):
        print(f"  fuse        : {line}")


if __name__ == "__main__":
    import langgraph
    from langgraph._internal._config import DEFAULT_RECURSION_LIMIT
    from importlib.metadata import version

    print("=" * 78)
    print("REPRO 1 — a step cap is not a progress check")
    print("=" * 78)
    print(f"langgraph               : {version('langgraph')}")
    print(f"DEFAULT_RECURSION_LIMIT : {DEFAULT_RECURSION_LIMIT}")

    limit = int(os.getenv("REPRO_LIMIT", "0")) or None
    _print("PART A  stock LangGraph", part_a(limit))
    _print("PART B  same graph + agentfuse LoopFuse(repeat_threshold=3)", part_b(limit))
    print("\nRun again with REPRO_LIMIT=25 to see the historical default's behaviour.")
