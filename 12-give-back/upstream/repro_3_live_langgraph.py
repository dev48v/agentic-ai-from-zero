"""Reproducer 3 — the same gap, with a REAL model, through real LangGraph.

Reproducers 1 and 2 use a scripted model so anyone can run them for free. A reviewer's
fair objection to that is "a real model would not do this". This one answers it: a live
`meta/llama-3.1-8b-instruct` on NVIDIA NIM, called through `langchain_openai.ChatOpenAI`
inside a real LangGraph agent loop, given a tool that can never satisfy it.

    python upstream/repro_3_live_langgraph.py

Needs `NVIDIA_API_KEY` in the repo-root `.env` (gitignored). The key is never printed.

`recursion_limit` is pinned low on purpose. The point is not how much money can be burned;
it is that NOTHING between the model and the bill notices the repetition, and that the
same graph with one wrapped node stops on the third identical call. Real token counts come
off `usage_metadata`, never from an estimate.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Annotated, TypedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from agentfuse import FuseBox, LoopFuse  # noqa: E402
from agentfuse.adapters.langgraph_guard import guard_tool_node  # noqa: E402

load_dotenv(os.path.join(HERE, "..", "..", ".env"))

BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")
RECURSION_LIMIT = int(os.getenv("REPRO_LIMIT", "12"))

STATS = {"llm": 0, "tool": 0, "in_tok": 0, "out_tok": 0}

SYSTEM = ("You are an order-tracking assistant. You have one tool, search_orders. "
          "The user's order definitely exists in the system. Do not give up: if the "
          "search comes back empty, search again until you find it.")


@tool
def search_orders(order_id: str) -> str:
    """Search the order database for an order id and return its tracking number."""
    STATS["tool"] += 1
    return "no results found"


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _api_key() -> str:
    for name in ("NVIDIA_API_KEY", "NIM_API_KEY", "OPENAI_API_KEY"):
        if os.getenv(name):
            return os.environ[name]
    raise SystemExit("No NVIDIA_API_KEY in .env — see the repo README.")


def build(guarded: bool):
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=_api_key(),
                     temperature=0.0, max_tokens=200, timeout=60)
    bound = llm.bind_tools([search_orders])

    def model_node(state: State) -> dict:
        reply = bound.invoke(state["messages"])
        STATS["llm"] += 1
        usage = getattr(reply, "usage_metadata", None) or {}
        STATS["in_tok"] += int(usage.get("input_tokens", 0) or 0)
        STATS["out_tok"] += int(usage.get("output_tokens", 0) or 0)
        return {"messages": [reply]}

    def route(state: State) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    box = FuseBox(loop=LoopFuse(repeat_threshold=3))
    tools_node = (guard_tool_node(ToolNode([search_orders]), box, resume_node="model")
                  if guarded else ToolNode([search_orders]))

    builder = StateGraph(State)
    builder.add_node("model", model_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route, {"tools": "tools", END: END})
    if not guarded:
        builder.add_edge("tools", "model")
    return builder.compile(), box


def run(guarded: bool) -> dict:
    for k in STATS:
        STATS[k] = 0
    graph, box = build(guarded)
    started = time.time()
    outcome, answer = "completed", ""
    try:
        state = graph.invoke(
            {"messages": [SystemMessage(SYSTEM),
                          HumanMessage("What is the tracking number for order 42?")]},
            {"recursion_limit": RECURSION_LIMIT})
        answer = str(state["messages"][-1].content)
    except GraphRecursionError as exc:
        outcome, answer = "GraphRecursionError", str(exc).splitlines()[0]
    return {"outcome": outcome, "answer": answer,
            "seconds": round(time.time() - started, 2),
            "llm_calls": STATS["llm"], "tool_calls": STATS["tool"],
            "tokens": STATS["in_tok"] + STATS["out_tok"],
            "in_tok": STATS["in_tok"], "out_tok": STATS["out_tok"],
            "blocks": [v.line() for v in box.blocks]}


def _print(title: str, r: dict) -> None:
    print(f"\n--- {title} ---")
    print(f"  outcome        : {r['outcome']}")
    print(f"  live LLM calls : {r['llm_calls']}")
    print(f"  tool calls     : {r['tool_calls']}")
    print(f"  REAL tokens    : {r['tokens']}  (in {r['in_tok']} / out {r['out_tok']})")
    print(f"  wall time      : {r['seconds']}s")
    print(f"  final text     : {r['answer'][:200]}")
    for line in r["blocks"]:
        print(f"  fuse           : {line}")


if __name__ == "__main__":
    from importlib.metadata import version

    print("=" * 78)
    print("REPRO 3 — live model, real LangGraph")
    print("=" * 78)
    print(f"langgraph {version('langgraph')} | langchain-openai {version('langchain-openai')}")
    print(f"model {MODEL} @ NVIDIA NIM | recursion_limit {RECURSION_LIMIT}")

    stock = run(guarded=False)
    _print("PART A  stock LangGraph", stock)
    guarded = run(guarded=True)
    _print("PART B  + agentfuse LoopFuse(repeat_threshold=3)", guarded)

    if stock["tokens"] and guarded["tokens"]:
        saved = 1 - guarded["tokens"] / stock["tokens"]
        print(f"\ntokens on the wire: {stock['tokens']} -> {guarded['tokens']} "
              f"({saved:.1%} fewer)")
