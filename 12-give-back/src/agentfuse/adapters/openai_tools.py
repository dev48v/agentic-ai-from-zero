"""Adapter: a guarded tool-calling loop for any OpenAI-compatible endpoint.

Works unchanged against NVIDIA NIM, Groq, OpenAI, OpenRouter, vLLM — anything that speaks
`client.chat.completions.create(...)` with `tools=[...]`. The client object is passed in,
so this module imports nothing beyond the standard library.

The loop is the plain one from projects 3 and 4, with three things added at the exact
points where money and damage happen:

    before every model call   -> budget preflight
    before every tool call    -> permission + loop check
    after every model call    -> real usage booked

A blocked tool call is not an exception. It becomes a `tool` role message telling the model
what was refused and why, exactly as Project 6 did with human denials — the model then has
to answer honestly instead of inventing a result it never got. A blocked MODEL call ends
the run, because there is nothing left to say it with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Mapping

from ..core import ToolCall
from ..runtime import FuseBox

__all__ = ["GuardedRun", "run_guarded_tool_loop"]


@dataclass
class GuardedRun:
    """Everything that happened, so a caller can assert on it instead of reading logs."""

    answer: str = ""
    stop_reason: str = ""          # "final-answer" | "blocked" | "max-turns"
    turns: int = 0
    llm_calls: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    blocked: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    cost_usd: float = 0.0
    dropped_tool_calls: int = 0    # trimmed by max_parallel_tool_calls, not by a fuse


def run_guarded_tool_loop(
    client,
    model: str,
    messages: list[dict],
    tools: list[dict],
    executors: Mapping[str, Callable[[dict], str]],
    fusebox: FuseBox,
    max_turns: int = 8,
    max_completion_tokens: int = 400,
    temperature: float = 0.0,
    max_parallel_tool_calls: int | None = None,
    terminal_fuses: frozenset[str] = frozenset({"loop"}),
) -> GuardedRun:
    """Run the loop until the model answers, a fuse blows, or `max_turns` is reached.

    `max_parallel_tool_calls` caps how many of a turn's requested calls are honoured. Not
    a guard — a compatibility valve, and one this project needed the moment it ran for
    real: `meta/llama-3.1-8b-instruct` on NVIDIA NIM answers a two-tool turn with

        500 — Failed to apply prompt template: invalid operation:
              This model only supports single tool-calls at once!

    and it fails on the NEXT request, when the two-call assistant turn is replayed as
    history, so the run is already several calls deep before anything breaks. With the cap
    set, only the honoured calls are echoed into the transcript; the rest are simply not
    claimed, and the model is free to ask for them again next turn. Nothing is fabricated
    and the history stays valid.

    `terminal_fuses` names the fuses whose refusal ENDS the run instead of being handed
    back to the model. It defaults to the loop fuse, and the reason is a measurement, not
    a preference: the first live run of this project fed every refusal back, and the
    guarded agent then spent MORE tokens than the unguarded one (2,230 vs 1,924) because
    it kept paying for turns after the fuse had already declared it stuck. A permission
    refusal is different — the model can still write an honest "I could not do that", and
    on that same run it did — so permission is deliberately not terminal.
    """
    run = GuardedRun(messages=list(messages))

    for _ in range(max_turns):
        run.turns += 1

        # --- model call, gated on budget --------------------------------- #
        prompt_text = "".join(str(m.get("content") or "") for m in run.messages)
        pre = fusebox.preflight_llm(prompt_text, max_completion_tokens)
        if pre.blocked:
            run.stop_reason = "blocked"
            run.blocked.append(pre)
            run.answer = (run.answer or
                          f"Stopped before the next model call: {pre.reason}")
            return run

        response = client.chat.completions.create(
            model=model,
            messages=run.messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )
        run.llm_calls += 1
        run.cost_usd = round(run.cost_usd + fusebox.record_llm(response), 8)

        choice = response.choices[0].message
        requested = list(getattr(choice, "tool_calls", None) or [])
        if max_parallel_tool_calls is not None:
            dropped = len(requested) - max_parallel_tool_calls
            requested = requested[:max_parallel_tool_calls]
            if dropped > 0:
                run.dropped_tool_calls += dropped

        # Echo the assistant turn back, carrying exactly the calls we are honouring:
        # a tool result whose matching assistant tool_call is missing is rejected by most
        # providers, and an assistant tool_call with no result is rejected by the rest.
        run.messages.append({
            "role": "assistant",
            "content": choice.content or "",
            **({"tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in requested]} if requested else {}),
        })

        if not requested:
            run.answer = choice.content or ""
            run.stop_reason = "final-answer"
            return run

        # --- tool calls, gated on permission + loop ----------------------- #
        terminal: list = []
        for tc in requested:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                # Unparseable arguments are their own signature; a model emitting broken
                # JSON over and over is still a loop, and must not crash the guard.
                args = {"__raw__": tc.function.arguments}

            call = ToolCall(name=tc.function.name, args=args, id=tc.id)
            verdict = fusebox.check_tool_call(call)

            if verdict.blocked:
                run.blocked.append(verdict)
                run.messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": f"REFUSED by the {verdict.fuse} fuse: {verdict.reason}. "
                               f"This action did NOT happen. Do not retry it; tell the "
                               f"user plainly what could not be done and why.",
                })
                if verdict.fuse in terminal_fuses:
                    terminal.append(verdict)
                continue

            fusebox.record_tool_call(call)
            run.tool_calls.append(call)
            executor = executors.get(call.name)
            if executor is None:
                result = f"ERROR: no executor registered for tool '{call.name}'"
            else:
                try:
                    result = str(executor(args))
                except Exception as exc:  # noqa: BLE001 — a tool must not kill the loop
                    result = f"ERROR: {type(exc).__name__}: {exc}"
            run.messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if terminal:
            # Do not spend another turn asking a model the guard has just declared stuck.
            run.stop_reason = "blocked"
            run.answer = ("Stopped by the %s fuse: %s"
                          % (terminal[0].fuse, terminal[0].reason))
            return run

    run.stop_reason = "max-turns"
    return run
