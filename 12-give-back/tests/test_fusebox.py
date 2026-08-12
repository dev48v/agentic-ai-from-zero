from agentfuse import (BudgetFuse, FuseBox, LoopFuse, PermissionFuse, Price, ToolCall,
                       ToolSpec)


def _box(**kw):
    return FuseBox(
        loop=kw.get("loop", LoopFuse(repeat_threshold=3)),
        permission=kw.get("permission", PermissionFuse(
            granted={"read"},
            specs=[ToolSpec.of("search", "read"), ToolSpec.of("refund", "spend_money")])),
        budget=kw.get("budget", BudgetFuse(max_usd=0.01, price=Price(0.10, 0.30))),
    )


def test_permission_is_evaluated_before_the_loop_counter():
    """A forbidden tool must be refused on attempt 1, not once a counter agrees."""
    box = _box()
    v = box.check_tool_call(ToolCall("refund", {"amount": 10}))
    assert v.blocked and v.fuse == "permission"


def test_loop_blocks_an_allowed_tool_that_repeats():
    box = _box()
    call = ToolCall("search", {"q": "x"})
    for _ in range(2):
        assert box.check_tool_call(call).allowed
        box.record_tool_call(call)
    v = box.check_tool_call(call)
    assert v.blocked and v.fuse == "loop"


def test_a_blocked_call_is_never_recorded_as_progress():
    box = _box()
    box.check_tool_call(ToolCall("refund", {}))
    assert box.loop.state.history == []


def test_budget_preflight_flows_through_the_box():
    box = _box(budget=BudgetFuse(max_usd=0.001, price=Price(0.10, 0.30)))
    assert box.preflight_llm("hello", max_completion_tokens=1000).blocked


def test_every_verdict_lands_in_the_log():
    box = _box()
    box.check_tool_call(ToolCall("search", {"q": "a"}))
    box.check_tool_call(ToolCall("refund", {}))
    box.preflight_llm("hi", 10)
    assert len(box.log) == 3
    assert len(box.blocks) == 1


def test_reset_clears_loop_budget_and_log_but_not_permissions():
    box = _box()
    call = ToolCall("search", {"q": "x"})
    box.check_tool_call(call)
    box.record_tool_call(call)
    box.budget.record(1000, 0)
    box.reset()
    assert box.loop.state.history == [] and box.budget.spent_usd == 0.0 and box.log == []
    assert box.check_tool_call(ToolCall("refund", {})).blocked


def test_a_box_with_no_fuses_allows_everything():
    box = FuseBox()
    assert box.check_tool_call(ToolCall("anything", {})).allowed
    assert box.preflight_llm("x", 10).allowed


def test_report_includes_the_spend_line():
    box = _box()
    box.budget.record(1000, 1000)
    assert "spend: $" in box.report()
