"""Verify `langgraph-no-progress-limit.patch` against an installed LangGraph.

The patch adds one optional argument, `ToolNode(..., no_progress_limit=N)`. This script
proves it does what the PR description claims, and it is the thing to run before anyone
submits that PR anywhere.

    # in a throwaway environment
    pip install langgraph langchain-openai
    cd "$(python -c 'import langgraph,os;print(os.path.dirname(os.path.dirname(langgraph.__file__)))')"
    patch -p3 < .../upstream/langgraph-no-progress-limit.patch
    python upstream/verify_patch.py

Three parts:

  A  the argument exists, validates its input, and is off by default (no behaviour change
     for anyone who does not ask for it — the bar any upstream patch has to clear).
  B  with a scripted model, the TOOL runs exactly `no_progress_limit - 1` times however
     long the graph keeps spinning. This is the guarantee: the expensive, side-effecting
     half stops even though the graph does not.
  C  with a LIVE model on NVIDIA NIM, the refusal is enough for the agent to stop by
     itself and answer honestly. Needs NVIDIA_API_KEY in the repo-root `.env`; skipped,
     loudly, without one. Whatever the model does is printed as it happened.
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, TypedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

COUNTS = {"tool": 0, "llm": 0}
RESULTS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((label, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))


@tool
def search_orders(order_id: str) -> str:
    """Search the order database for an order id."""
    COUNTS["tool"] += 1
    return "no results found"


class State(TypedDict):
    messages: Annotated[list, add_messages]


def stuck_model(state: State) -> dict:
    COUNTS["llm"] += 1
    return {"messages": [AIMessage(content="", tool_calls=[
        {"name": "search_orders", "args": {"order_id": "42"}, "id": f"c{COUNTS['llm']}"}])]}


def graph_with(tools_node, model=stuck_model, conditional=False):
    b = StateGraph(State)
    b.add_node("model", model)
    b.add_node("tools", tools_node)
    b.add_edge(START, "model")
    if conditional:
        b.add_conditional_edges(
            "model",
            lambda s: "tools" if getattr(s["messages"][-1], "tool_calls", None) else END,
            {"tools": "tools", END: END})
    else:
        b.add_edge("model", "tools")
    b.add_edge("tools", "model")
    return b.compile()


def part_a() -> None:
    print("\n--- A: the argument, and no behaviour change when it is off ---")
    import inspect
    sig = inspect.signature(ToolNode.__init__)
    check("ToolNode accepts no_progress_limit", "no_progress_limit" in sig.parameters,
          f"default {sig.parameters['no_progress_limit'].default!r}"
          if "no_progress_limit" in sig.parameters else "argument missing")

    rejected = False
    try:
        ToolNode([search_orders], no_progress_limit=1)
    except ValueError:
        rejected = True
    check("no_progress_limit=1 is rejected", rejected, "a first call is never a repeat")

    COUNTS["tool"] = COUNTS["llm"] = 0
    try:
        graph_with(ToolNode([search_orders])).invoke(
            {"messages": [HumanMessage("where is order 42?")]}, {"recursion_limit": 20})
    except GraphRecursionError:
        pass
    check("off by default: the stock behaviour is untouched", COUNTS["tool"] >= 9,
          f"{COUNTS['tool']} identical tool calls, then GraphRecursionError")


def part_b() -> None:
    print("\n--- B: the tool stops even though the graph does not ---")
    for limit in (2, 3, 5):
        COUNTS["tool"] = COUNTS["llm"] = 0
        node = ToolNode([search_orders], no_progress_limit=limit)
        try:
            graph_with(node).invoke({"messages": [HumanMessage("where is order 42?")]},
                                    {"recursion_limit": 30})
        except GraphRecursionError:
            pass
        check(f"no_progress_limit={limit} runs the tool exactly {limit - 1}x",
              COUNTS["tool"] == limit - 1,
              f"tool ran {COUNTS['tool']}x across {COUNTS['llm']} model turns")

    # A genuinely progressing agent must be untouched.
    seen = {"n": 0}

    def progressing(state):
        seen["n"] += 1
        COUNTS["llm"] += 1
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "search_orders", "args": {"order_id": str(seen["n"])},
             "id": f"c{seen['n']}"}])]}

    COUNTS["tool"] = COUNTS["llm"] = seen["n"] = 0
    try:
        graph_with(ToolNode([search_orders], no_progress_limit=3),
                   model=progressing).invoke(
            {"messages": [HumanMessage("search everything")]}, {"recursion_limit": 20})
    except GraphRecursionError:
        pass
    check("an agent that keeps changing its arguments is never refused",
          COUNTS["tool"] >= 8, f"{COUNTS['tool']} distinct tool calls, none refused")


def part_c() -> None:
    print("\n--- C: live model, does the refusal actually end the run? ---")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(HERE, "..", "..", ".env"))
    key = next((os.environ[n] for n in ("NVIDIA_API_KEY", "NIM_API_KEY", "OPENAI_API_KEY")
                if os.getenv(n)), None)
    if not key:
        print("  SKIPPED — no NVIDIA_API_KEY in .env")
        return

    from langchain_openai import ChatOpenAI
    model_id = os.getenv("NIM_MODEL", "meta/llama-3.1-8b-instruct")
    llm = ChatOpenAI(model=model_id,
                     base_url=os.getenv("NIM_BASE_URL",
                                        "https://integrate.api.nvidia.com/v1"),
                     api_key=key, temperature=0.0, max_tokens=200, timeout=60)
    bound = llm.bind_tools([search_orders])
    tokens = {"n": 0}

    def live_model(state: State) -> dict:
        reply = bound.invoke(state["messages"])
        COUNTS["llm"] += 1
        usage = getattr(reply, "usage_metadata", None) or {}
        tokens["n"] += int(usage.get("input_tokens", 0) or 0) + \
            int(usage.get("output_tokens", 0) or 0)
        return {"messages": [reply]}

    prompt = [SystemMessage("You are an order-tracking assistant with one tool, "
                            "search_orders. The user's order definitely exists. Do not "
                            "give up: if the search is empty, search again."),
              HumanMessage("What is the tracking number for order 42?")]

    for label, node in (("stock", ToolNode([search_orders])),
                        ("patched (no_progress_limit=3)",
                         ToolNode([search_orders], no_progress_limit=3))):
        COUNTS["tool"] = COUNTS["llm"] = tokens["n"] = 0
        outcome, answer = "completed", ""
        try:
            state = graph_with(node, model=live_model, conditional=True).invoke(
                {"messages": list(prompt)}, {"recursion_limit": 12})
            answer = str(state["messages"][-1].content)
        except GraphRecursionError:
            outcome = "GraphRecursionError"
        print(f"    {label:<32} {outcome:<22} "
              f"llm={COUNTS['llm']} tool={COUNTS['tool']} tokens={tokens['n']}")
        print(f"      final: {answer[:150]!r}")
        if label.startswith("patched"):
            check("live: the patched node stops the agent without an exception",
                  outcome == "completed" and COUNTS["tool"] == 2,
                  f"{COUNTS['tool']} live tool calls, {tokens['n']} real tokens")


if __name__ == "__main__":
    from importlib.metadata import version
    print("=" * 78)
    print(f"VERIFY PATCH — langgraph {version('langgraph')}")
    print("=" * 78)
    part_a()
    part_b()
    part_c()
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    sys.exit(0 if passed == len(RESULTS) else 1)
