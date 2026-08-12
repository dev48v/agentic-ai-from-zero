"""Adapter: put the fuses in front of a LangGraph `ToolNode`.

LangGraph is imported LAZILY, inside the functions that need it. `agentfuse` itself has no
dependencies and must stay importable in a project that has never heard of LangGraph.

WHY THIS EXISTS
---------------
LangGraph's only defence against an agent that stops making progress is `recursion_limit`
— a cap on total supersteps that, when reached, raises `GraphRecursionError`. That is a
step cap, not a progress check, and the two are not the same thing:

  * it cannot tell "ten useful steps" from "the same step ten times";
  * it fires at the END of the wasted work, so every one of those calls is already paid for;
  * on the installed version the default is large enough that "it will stop eventually" is
    not a comfort (see `12-give-back/upstream/` for the measured number).

TWO WAYS IN, and the first one is the one to reach for
------------------------------------------------------
1. `fuse_wrap_tool_call` — an interceptor for LangGraph's OWN documented extension point,
   `ToolNode(tools, wrap_tool_call=...)`. No change to the graph's wiring, and because a
   `ToolNode` is accepted wherever tools are, it works with `create_react_agent` too. It
   is stateless: the repeat history is derived from the transcript on every call, because
   a `ToolNode` instance is shared across every request the process serves.

       from langgraph.prebuilt import ToolNode, create_react_agent
       from agentfuse.adapters.langgraph_guard import fuse_wrap_tool_call

       node = ToolNode(tools, wrap_tool_call=fuse_wrap_tool_call(repeat_threshold=3))
       agent = create_react_agent(model, node)

2. `guard_tool_node` — wraps the whole node instead of each call. More invasive (it takes
   over routing out of the node) and it can therefore do the thing an interceptor cannot:
   end the graph outright when the agent is stuck, rather than letting it spend one more
   model call discovering that it has been refused.

       box = FuseBox(loop=LoopFuse(repeat_threshold=3))
       builder.add_node("tools", guard_tool_node(ToolNode(tools), box, resume_node="model"))
       builder.add_edge("model", "tools")
       # NOTE: no `add_edge("tools", "model")` — the guarded node routes itself so that it
       # can route to END when a fuse blows. See `guard_tool_node` for why.
"""

from __future__ import annotations

from typing import Any, Callable

from ..core import FuseError, ToolCall, canonical_signature
from ..loops import LoopFuse
from ..permissions import PermissionFuse
from ..runtime import FuseBox

__all__ = ["guard_tool_node", "fuse_wrap_tool_call", "extract_tool_calls",
           "executed_signatures"]

ON_BLOCK_END = "end"        # stop the graph, leaving an explanation in the state
ON_BLOCK_MESSAGE = "message"  # hand the refusal back to the model and let it respond
ON_BLOCK_RAISE = "raise"    # raise FuseError — for tests and for fail-closed deployments


def extract_tool_calls(state: Any) -> list[ToolCall]:
    """Pull the pending tool calls off the last AI message of a LangGraph state.

    Accepts the dict-shaped state `create_react_agent` uses and anything else exposing
    `.messages`, so it also works with a custom `TypedDict` schema.
    """
    messages = state.get("messages") if isinstance(state, dict) else getattr(state, "messages", None)
    if not messages:
        return []
    last = messages[-1]
    raw = getattr(last, "tool_calls", None) or []
    out = []
    for tc in raw:
        if isinstance(tc, dict):
            out.append(ToolCall(name=tc.get("name", ""), args=tc.get("args") or {},
                                id=tc.get("id") or ""))
        else:  # pragma: no cover — provider-specific object shape
            out.append(ToolCall(name=getattr(tc, "name", ""),
                                args=getattr(tc, "args", {}) or {},
                                id=getattr(tc, "id", "") or ""))
    return out


def guard_tool_node(
    tool_node: Callable,
    fusebox: FuseBox,
    on_block: str = ON_BLOCK_END,
    resume_node: str | None = None,
) -> Callable:
    """Return a drop-in replacement node that checks the fuses before delegating.

    `on_block` controls what a blown fuse does:

        "end"      stop the graph. The wrapper then OWNS the routing out of this node and
                   returns a `Command` on both paths, so you must pass `resume_node` (the
                   node the tool results normally flow back to, usually your model node)
                   and you must NOT add a static edge out of the guarded node. LangGraph
                   static edges are unconditional and are taken IN ADDITION to a Command's
                   `goto`, so leaving one in place silently defeats the stop — which is
                   why the missing `resume_node` is a wiring-time ValueError rather than a
                   surprise at 3am.
        "message"  hand the refusal back to the model as a tool result and let the graph
                   carry on. Static edges are fine. Use it when the model can reasonably
                   recover; know that a genuinely stuck model will simply be blocked again.
        "raise"    raise `FuseError`. Fail-closed, and the easiest thing to assert on.

    Checking is all-or-nothing per turn: if ANY pending call is blocked, none of that
    turn's calls run and none are recorded as progress. Half-executing a batch leaves the
    loop counter describing work that never happened.

    Every pending tool call gets a `ToolMessage` back either way, so the history stays
    well-formed — an assistant tool_call with no matching tool result is a protocol error
    at most providers, and "we stopped it" is not an excuse the next provider will accept.
    """
    if on_block not in (ON_BLOCK_END, ON_BLOCK_MESSAGE, ON_BLOCK_RAISE):
        raise ValueError(f"on_block must be one of end/message/raise, got {on_block!r}")
    if on_block == ON_BLOCK_END and not resume_node:
        raise ValueError(
            "on_block='end' needs resume_node=<the node tool results flow back to>, and "
            "the guarded node must have no static outgoing edge — otherwise LangGraph "
            "takes that edge as well as the Command and the graph never stops.")

    def _node(state: Any, *args, **kwargs):
        from langchain_core.messages import ToolMessage  # lazy: see module docstring

        calls = extract_tool_calls(state)
        verdicts = [(call, fusebox.check_tool_call(call)) for call in calls]
        blocked = [(c, v) for c, v in verdicts if v.blocked]

        if not blocked:
            for call, _ in verdicts:
                fusebox.record_tool_call(call)
            result = _delegate(tool_node, state, *args, **kwargs)
            if on_block == ON_BLOCK_END:
                from langgraph.types import Command
                return Command(goto=resume_node, update=result)
            return result

        if on_block == ON_BLOCK_RAISE:
            raise FuseError(blocked[0][1])

        messages = []
        for call, verdict in verdicts:
            if verdict.blocked:
                body = (f"REFUSED by the {verdict.fuse} fuse: {verdict.reason}. "
                        f"This action did NOT happen.")
            else:
                body = ("Not executed: the request was stopped because another call in "
                        "this turn was refused.")
            messages.append(ToolMessage(content=body, tool_call_id=call.id or call.name,
                                        name=call.name, status="error"))

        if on_block == ON_BLOCK_MESSAGE:
            return {"messages": messages}

        from langgraph.graph import END
        from langgraph.types import Command
        # goto=END, not "back to the model": the model is the thing that is stuck, and
        # asking it again is how you turn one wasted call into twenty.
        return Command(goto=END, update={"messages": messages})

    _node.__name__ = "guarded_tool_node"
    return _node


def _delegate(tool_node: Any, state: Any, *args, **kwargs):
    """Call the wrapped node whatever shape it is.

    `ToolNode` is a `RunnableCallable`, which is invoked via `.invoke(state, config)` and
    is NOT plain-callable; a hand-written node function is plain-callable and has no
    `.invoke`. Supporting both is three lines here and saves every user a stack trace.
    """
    invoke = getattr(tool_node, "invoke", None)
    if callable(invoke):
        return invoke(state, *args, **kwargs)
    return tool_node(state, *args, **kwargs)


# --------------------------------------------------------------------------- #
# The idiomatic route: LangGraph's own `wrap_tool_call` interceptor.
# --------------------------------------------------------------------------- #
def executed_signatures(state: Any, messages_key: str = "messages") -> list[str]:
    """Signatures of the tool calls in this transcript that ACTUALLY RAN, in order.

    A call counts as having run when its `ToolMessage` exists and is not an error — so a
    call this guard already refused is not counted a second time, and a call that blew up
    inside the tool is not mistaken for progress either.

    Reading the transcript rather than keeping a counter is what makes the interceptor
    safe: a `ToolNode` instance is shared across every request the process serves, so any
    state stored on it belongs to whoever ran last.
    """
    messages = (state.get(messages_key) if isinstance(state, dict)
                else getattr(state, messages_key, None)) or []
    failed_ids = {getattr(m, "tool_call_id", None) for m in messages
                  if getattr(m, "type", "") == "tool" and getattr(m, "status", "") == "error"}
    answered_ids = {getattr(m, "tool_call_id", None) for m in messages
                    if getattr(m, "type", "") == "tool"}
    out: list[str] = []
    for msg in messages:
        for tc in (getattr(msg, "tool_calls", None) or []):
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tc_id in answered_ids and tc_id not in failed_ids:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                args = (tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})) or {}
                out.append(canonical_signature(name, args))
    return out


def fuse_wrap_tool_call(
    repeat_threshold: int = 3,
    max_cycle_len: int = 4,
    detect_cycles: bool = True,
    permission: "PermissionFuse | None" = None,
    blocks: list | None = None,
    messages_key: str = "messages",
) -> Callable:
    """Build a `wrap_tool_call` interceptor that refuses a call instead of running it.

    This is the version to prefer: `wrap_tool_call` is LangGraph's own documented
    extension point, it needs no change to the graph's wiring, and because
    `ToolNode` is accepted wherever tools are, it reaches `create_react_agent` too.

        from langgraph.prebuilt import ToolNode, create_react_agent
        from agentfuse.adapters.langgraph_guard import fuse_wrap_tool_call

        node = ToolNode(tools, wrap_tool_call=fuse_wrap_tool_call(repeat_threshold=3))
        agent = create_react_agent(model, node)

    A refused call comes back as a `ToolMessage(status="error")` explaining itself, which
    the agent loop already knows how to carry. It does NOT end the graph — an interceptor
    cannot, and should not, take that decision on the graph's behalf. Pair it with a small
    `recursion_limit` if you want a hard stop as well: the point of the fuse is that the
    expensive tool call never happens, and that the transcript says why.

    Pass `blocks` a list to collect the verdicts for logging or assertions.
    """
    loop_kwargs = dict(repeat_threshold=repeat_threshold, max_cycle_len=max_cycle_len,
                       detect_cycles=detect_cycles)

    def _wrapper(request: Any, execute: Callable):
        from langchain_core.messages import ToolMessage  # lazy: see module docstring

        raw = request.tool_call
        call = ToolCall(name=raw.get("name", ""), args=raw.get("args") or {},
                        id=raw.get("id") or "")

        verdicts = []
        if permission is not None:
            verdicts.append(permission.check(call))
        fuse = LoopFuse.from_history(
            executed_signatures(request.state, messages_key), **loop_kwargs)
        verdicts.append(fuse.check(call))

        for verdict in verdicts:
            if verdict.blocked:
                if blocks is not None:
                    blocks.append(verdict)
                return ToolMessage(
                    content=(f"REFUSED by the {verdict.fuse} fuse: {verdict.reason}. "
                             f"This action did NOT happen."),
                    tool_call_id=call.id or call.name, name=call.name, status="error")
        return execute(request)

    return _wrapper
