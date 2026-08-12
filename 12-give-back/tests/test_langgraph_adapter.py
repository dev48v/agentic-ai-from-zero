"""Adapter tests. Skipped wholesale when LangGraph is not installed — `agentfuse` itself
has no dependencies, and its test suite must pass in an environment that has never heard
of LangGraph."""

import pytest

pytest.importorskip("langgraph", reason="LangGraph is an optional extra")

from typing import Annotated, TypedDict  # noqa: E402

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from langgraph.graph import START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from agentfuse import FuseBox, FuseError, LoopFuse, PermissionFuse, ToolSpec  # noqa: E402
from agentfuse.adapters.langgraph_guard import (extract_tool_calls,  # noqa: E402
                                                guard_tool_node)

CALLS = {"tool": 0, "llm": 0}


@tool
def search_orders(query: str) -> str:
    """Search orders."""
    CALLS["tool"] += 1
    return "no results found"


@tool
def issue_refund(amount: float) -> str:
    """Refund a customer."""
    CALLS["tool"] += 1
    return f"refunded {amount}"


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _stuck_model(name="search_orders", args=None):
    def node(state):
        CALLS["llm"] += 1
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": name, "args": dict(args or {"query": "order 42"}),
             "id": f"c{CALLS['llm']}"}])]}
    return node


def _graph(tools_node, model=None, self_routing=False):
    b = StateGraph(State)
    b.add_node("model", model or _stuck_model())
    b.add_node("tools", tools_node)
    b.add_edge(START, "model")
    b.add_edge("model", "tools")
    if not self_routing:
        b.add_edge("tools", "model")
    return b.compile()


@pytest.fixture(autouse=True)
def _reset():
    CALLS["tool"] = CALLS["llm"] = 0


def test_extract_tool_calls_reads_the_last_ai_message():
    state = {"messages": [HumanMessage("hi"),
                          AIMessage(content="", tool_calls=[
                              {"name": "search_orders", "args": {"query": "x"},
                               "id": "c1"}])]}
    calls = extract_tool_calls(state)
    assert len(calls) == 1 and calls[0].name == "search_orders" and calls[0].id == "c1"


def test_extract_tool_calls_is_empty_when_there_is_nothing_pending():
    assert extract_tool_calls({"messages": [HumanMessage("hi")]}) == []
    assert extract_tool_calls({"messages": []}) == []


def test_stock_graph_runs_until_the_recursion_limit():
    """The gap this adapter exists to close, asserted rather than described."""
    graph = _graph(ToolNode([search_orders]))
    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage("where is order 42?")]},
                     {"recursion_limit": 20})
    assert CALLS["tool"] >= 9


def test_guarded_graph_stops_on_the_third_identical_call():
    box = FuseBox(loop=LoopFuse(repeat_threshold=3))
    graph = _graph(guard_tool_node(ToolNode([search_orders]), box, resume_node="model"),
                   self_routing=True)
    state = graph.invoke({"messages": [HumanMessage("where is order 42?")]},
                         {"recursion_limit": 20})
    assert CALLS["tool"] == 2
    assert len(box.blocks) == 1
    assert "REFUSED" in state["messages"][-1].content


def test_permission_denial_never_reaches_the_tool():
    box = FuseBox(permission=PermissionFuse(granted={"read"},
                                            specs=[ToolSpec.of("issue_refund", "write")]))
    graph = _graph(guard_tool_node(ToolNode([issue_refund]), box, resume_node="model"),
                   model=_stuck_model("issue_refund", {"amount": 40.0}),
                   self_routing=True)
    graph.invoke({"messages": [HumanMessage("refund me")]}, {"recursion_limit": 20})
    assert CALLS["tool"] == 0
    assert box.blocks and box.blocks[0].fuse == "permission"


def test_on_block_raise_surfaces_a_fuse_error():
    box = FuseBox(loop=LoopFuse(repeat_threshold=2))
    graph = _graph(guard_tool_node(ToolNode([search_orders]), box, on_block="raise"))
    with pytest.raises(FuseError) as excinfo:
        graph.invoke({"messages": [HumanMessage("go")]}, {"recursion_limit": 20})
    assert excinfo.value.verdict.fuse == "loop"


def test_on_block_end_without_resume_node_fails_at_wiring_time():
    with pytest.raises(ValueError) as excinfo:
        guard_tool_node(ToolNode([search_orders]), FuseBox(), on_block="end")
    assert "resume_node" in str(excinfo.value)


def test_unknown_on_block_is_rejected():
    with pytest.raises(ValueError):
        guard_tool_node(ToolNode([search_orders]), FuseBox(), on_block="shrug")


def test_a_healthy_graph_is_untouched_by_the_guard():
    """The guard must be invisible when nothing is wrong."""
    seen = {"n": 0}

    def progressing_model(state):
        seen["n"] += 1
        CALLS["llm"] += 1
        if seen["n"] > 3:
            return {"messages": [AIMessage(content="found it: TRK-9")]}
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "search_orders", "args": {"query": f"order {seen['n']}"},
             "id": f"c{seen['n']}"}])]}

    from langgraph.graph import END

    box = FuseBox(loop=LoopFuse(repeat_threshold=3))
    b = StateGraph(State)
    b.add_node("model", progressing_model)
    b.add_node("tools", guard_tool_node(ToolNode([search_orders]), box,
                                        resume_node="model"))
    b.add_edge(START, "model")
    b.add_conditional_edges(
        "model", lambda s: "tools" if getattr(s["messages"][-1], "tool_calls", None) else END,
        {"tools": "tools", END: END})
    state = b.compile().invoke({"messages": [HumanMessage("find it")]},
                               {"recursion_limit": 20})
    assert state["messages"][-1].content == "found it: TRK-9"
    assert CALLS["tool"] == 3 and box.blocks == []


# --------------------------------------------------------------------------- #
# fuse_wrap_tool_call — LangGraph's own interceptor hook
# --------------------------------------------------------------------------- #
from agentfuse.adapters.langgraph_guard import (executed_signatures,  # noqa: E402
                                                fuse_wrap_tool_call)


def test_wrap_tool_call_refuses_the_third_identical_call():
    blocks = []
    node = ToolNode([search_orders],
                    wrap_tool_call=fuse_wrap_tool_call(repeat_threshold=3, blocks=blocks))
    graph = _graph(node)
    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage("go")]}, {"recursion_limit": 20})
    # The tool itself runs exactly twice however long the graph flails afterwards.
    assert CALLS["tool"] == 2
    assert blocks and blocks[0].fuse == "loop"


def test_wrap_tool_call_enforces_permissions():
    blocks = []
    node = ToolNode([issue_refund],
                    wrap_tool_call=fuse_wrap_tool_call(
                        permission=PermissionFuse({"read"},
                                                  [ToolSpec.of("issue_refund", "write")]),
                        blocks=blocks))
    graph = _graph(node, model=_stuck_model("issue_refund", {"amount": 40.0}))
    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage("refund me")]}, {"recursion_limit": 8})
    assert CALLS["tool"] == 0
    assert blocks[0].fuse == "permission"


def test_wrap_tool_call_keeps_no_state_between_requests():
    """The reason the history is derived from the transcript: a ToolNode instance is
    shared by every request the process serves."""
    node = ToolNode([search_orders], wrap_tool_call=fuse_wrap_tool_call(repeat_threshold=3))
    graph = _graph(node)
    for _ in range(2):
        CALLS["tool"] = CALLS["llm"] = 0
        with pytest.raises(GraphRecursionError):
            graph.invoke({"messages": [HumanMessage("go")]}, {"recursion_limit": 12})
        assert CALLS["tool"] == 2       # the second request gets its own two attempts


def test_executed_signatures_ignores_calls_that_were_refused():
    from langchain_core.messages import ToolMessage
    state = {"messages": [
        AIMessage(content="", tool_calls=[{"name": "t", "args": {"a": 1}, "id": "c1"}]),
        ToolMessage(content="ok", tool_call_id="c1", name="t"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {"a": 1}, "id": "c2"}]),
        ToolMessage(content="REFUSED", tool_call_id="c2", name="t", status="error"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {"a": 1}, "id": "c3"}]),
    ]}
    assert len(executed_signatures(state)) == 1


def test_wrap_tool_call_lets_a_progressing_agent_through():
    seen = {"n": 0}

    def progressing(state):
        seen["n"] += 1
        CALLS["llm"] += 1
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": "search_orders", "args": {"query": f"order {seen['n']}"},
             "id": f"c{seen['n']}"}])]}

    blocks = []
    node = ToolNode([search_orders],
                    wrap_tool_call=fuse_wrap_tool_call(repeat_threshold=3, blocks=blocks))
    graph = _graph(node, model=progressing)
    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage("go")]}, {"recursion_limit": 12})
    assert blocks == [] and CALLS["tool"] >= 5
