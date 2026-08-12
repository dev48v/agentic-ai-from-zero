"""Tests for the OpenAI-compatible adapter, driven by a scripted client.

No network: the client is a stub that replays a fixed list of turns. The point is the
control flow around the model — what runs, what is refused, when the loop stops — and that
must be testable without an API key or a bill.
"""

from types import SimpleNamespace

import pytest

from agentfuse import BudgetFuse, FuseBox, LoopFuse, PermissionFuse, Price, ToolSpec
from agentfuse.adapters.openai_tools import run_guarded_tool_loop


def _tool_call(name, arguments, id_):
    return SimpleNamespace(id=id_, type="function",
                           function=SimpleNamespace(name=name, arguments=arguments))


class ScriptedClient:
    """Replays `turns`; the last turn repeats forever, like a model that will not stop."""

    def __init__(self, turns, usage=(100, 20)):
        self.turns = turns
        self.usage = usage
        self.calls = 0
        self.last_messages = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_messages = kwargs["messages"]
        turn = self.turns[min(self.calls, len(self.turns) - 1)]
        self.calls += 1
        content, tool_calls = turn
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=content, tool_calls=tool_calls))],
            usage=SimpleNamespace(prompt_tokens=self.usage[0],
                                  completion_tokens=self.usage[1]))


SEARCH = [{"type": "function", "function": {"name": "search", "description": "search",
                                            "parameters": {"type": "object",
                                                           "properties": {}}}}]
BASE = [{"role": "user", "content": "find order 42"}]


def _repeat_turn(i):
    return ("", [_tool_call("search", '{"q": "order 42"}', f"c{i}")])


def test_a_final_answer_ends_the_loop():
    client = ScriptedClient([("here it is: TRK-9", None)])
    run = run_guarded_tool_loop(client, "m", list(BASE), SEARCH, {}, FuseBox())
    assert run.stop_reason == "final-answer" and run.answer == "here it is: TRK-9"
    assert run.llm_calls == 1


def test_a_repeating_model_is_stopped_by_the_loop_fuse():
    client = ScriptedClient([_repeat_turn(i) for i in range(10)])
    ran = []
    box = FuseBox(loop=LoopFuse(repeat_threshold=3))
    run = run_guarded_tool_loop(client, "m", list(BASE), SEARCH,
                                {"search": lambda a: ran.append(a) or "no results"},
                                box, max_turns=10)
    assert len(ran) == 2                      # executed twice, refused on the third
    assert run.stop_reason == "blocked"
    assert run.llm_calls == 3                 # and NOT a fourth
    assert box.blocks[0].fuse == "loop"


def test_a_loop_block_does_not_buy_another_model_turn():
    """The bug the first live run exposed: feeding a loop refusal back to a stuck model
    made the guarded run cost MORE than the unguarded one."""
    client = ScriptedClient([_repeat_turn(i) for i in range(10)])
    box = FuseBox(loop=LoopFuse(repeat_threshold=3),
                  budget=BudgetFuse(price=Price(1.0, 1.0)))
    guarded = run_guarded_tool_loop(client, "m", list(BASE), SEARCH,
                                    {"search": lambda a: "no results"}, box, max_turns=10)

    client2 = ScriptedClient([_repeat_turn(i) for i in range(10)])
    open_box = FuseBox(budget=BudgetFuse(price=Price(1.0, 1.0)))
    unguarded = run_guarded_tool_loop(client2, "m", list(BASE), SEARCH,
                                      {"search": lambda a: "no results"}, open_box,
                                      max_turns=10)
    assert guarded.llm_calls < unguarded.llm_calls
    assert box.budget.spent_usd < open_box.budget.spent_usd


def test_a_permission_refusal_is_handed_back_so_the_model_can_answer_honestly():
    """Permission is deliberately NOT terminal: the model can still write a truthful
    'I could not do that', and it needs a turn to do it in."""
    client = ScriptedClient([
        ("", [_tool_call("refund", '{"amount": 40}', "c1")]),
        ("I could not issue the refund — I do not have permission.", None),
    ])
    ran = []
    box = FuseBox(permission=PermissionFuse({"read"},
                                            [ToolSpec.of("refund", "spend_money")]))
    run = run_guarded_tool_loop(client, "m", list(BASE), SEARCH,
                                {"refund": lambda a: ran.append(a) or "done"}, box)
    assert ran == []
    assert run.stop_reason == "final-answer"
    assert "could not" in run.answer
    assert box.blocks[0].fuse == "permission"


def test_terminal_fuses_is_configurable():
    client = ScriptedClient([("", [_tool_call("refund", '{"amount": 40}', "c1")]),
                             ("anything", None)])
    box = FuseBox(permission=PermissionFuse({"read"},
                                            [ToolSpec.of("refund", "spend_money")]))
    run = run_guarded_tool_loop(client, "m", list(BASE), SEARCH, {}, box,
                                terminal_fuses=frozenset({"permission"}))
    assert run.stop_reason == "blocked" and run.llm_calls == 1


def test_budget_stops_the_run_before_the_model_call():
    client = ScriptedClient([_repeat_turn(i) for i in range(10)])
    box = FuseBox(budget=BudgetFuse(max_usd=0.001, price=Price(1.0, 1.0)))
    run = run_guarded_tool_loop(client, "m", list(BASE), SEARCH,
                                {"search": lambda a: "x"}, box, max_turns=5)
    assert run.llm_calls == 0 and run.stop_reason == "blocked"
    assert client.calls == 0            # the guard fired before any HTTP would have


def test_real_usage_is_booked_per_call():
    client = ScriptedClient([("done", None)], usage=(300, 100))
    box = FuseBox(budget=BudgetFuse(price=Price(0.10, 0.30)))
    run_guarded_tool_loop(client, "m", list(BASE), SEARCH, {}, box)
    assert box.budget.prompt_tokens == 300 and box.budget.completion_tokens == 100
    assert box.budget.spent_usd == pytest.approx(0.06)


def test_max_parallel_tool_calls_trims_the_echoed_assistant_turn():
    """NIM's llama-3.1-8b rejects a replayed two-call assistant turn with a 500. Only the
    honoured calls may appear in the history."""
    client = ScriptedClient([
        ("", [_tool_call("search", '{"q": "a"}', "c1"), _tool_call("search", '{"q": "b"}', "c2")]),
        ("done", None)])
    ran = []
    run = run_guarded_tool_loop(client, "m", list(BASE), SEARCH,
                                {"search": lambda a: ran.append(a) or "ok"}, FuseBox(),
                                max_parallel_tool_calls=1)
    assistant = [m for m in run.messages if m["role"] == "assistant" and m.get("tool_calls")]
    assert len(assistant[0]["tool_calls"]) == 1
    assert len(ran) == 1 and run.dropped_tool_calls == 1
    tool_ids = {m["tool_call_id"] for m in run.messages if m["role"] == "tool"}
    assert tool_ids == {"c1"}


def test_unparseable_arguments_do_not_crash_the_loop():
    client = ScriptedClient([("", [_tool_call("search", "{not json", "c1")]), ("done", None)])
    run = run_guarded_tool_loop(client, "m", list(BASE), SEARCH,
                                {"search": lambda a: "ok"}, FuseBox())
    assert run.stop_reason == "final-answer"
    assert run.tool_calls[0].args["__raw__"] == "{not json"


def test_a_tool_that_raises_becomes_an_error_string_not_a_crash():
    client = ScriptedClient([("", [_tool_call("search", "{}", "c1")]), ("done", None)])
    run = run_guarded_tool_loop(client, "m", list(BASE), SEARCH,
                                {"search": lambda a: 1 / 0}, FuseBox())
    err = [m for m in run.messages if m["role"] == "tool"][0]["content"]
    assert err.startswith("ERROR: ZeroDivisionError")
    assert run.stop_reason == "final-answer"


def test_a_missing_executor_is_reported_not_raised():
    client = ScriptedClient([("", [_tool_call("search", "{}", "c1")]), ("done", None)])
    run = run_guarded_tool_loop(client, "m", list(BASE), SEARCH, {}, FuseBox())
    err = [m for m in run.messages if m["role"] == "tool"][0]["content"]
    assert "no executor registered" in err


def test_max_turns_is_the_backstop():
    client = ScriptedClient([_repeat_turn(i) for i in range(20)])
    run = run_guarded_tool_loop(client, "m", list(BASE), SEARCH,
                                {"search": lambda a: "x"}, FuseBox(), max_turns=4)
    assert run.stop_reason == "max-turns" and run.turns == 4
