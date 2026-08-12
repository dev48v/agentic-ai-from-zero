from agentfuse import LoopFuse, ToolCall


def _drive(fuse: LoopFuse, calls: list[ToolCall]):
    """Run calls through the fuse the way an agent loop would: check, then record."""
    verdicts = []
    for call in calls:
        v = fuse.check(call)
        verdicts.append(v)
        if v.allowed:
            fuse.record(call)
    return verdicts


def test_identical_call_blows_on_the_threshold():
    fuse = LoopFuse(repeat_threshold=3)
    call = ToolCall("search", {"q": "order 42"})
    verdicts = _drive(fuse, [call, call, call])
    assert [v.allowed for v in verdicts] == [True, True, False]
    assert "not making progress" in verdicts[-1].reason


def test_argument_key_order_does_not_change_identity():
    fuse = LoopFuse(repeat_threshold=2)
    a = ToolCall("t", {"x": 1, "y": 2})
    b = ToolCall("t", {"y": 2, "x": 1})
    assert a.signature == b.signature
    assert _drive(fuse, [a, b])[-1].blocked


def test_call_id_is_not_part_of_identity():
    a = ToolCall("t", {"x": 1}, id="call_1")
    b = ToolCall("t", {"x": 1}, id="call_2")
    assert a.signature == b.signature


def test_different_arguments_are_progress():
    fuse = LoopFuse(repeat_threshold=2)
    calls = [ToolCall("search", {"q": f"order {i}"}) for i in range(6)]
    assert all(v.allowed for v in _drive(fuse, calls))


def test_two_step_cycle_is_caught_when_no_counter_would_fire():
    """A,B,A,B — no signature reaches a repeat threshold of 3, so a count-only rule
    (what Project 11 shipped) stays silent. The cycle detector must not."""
    fuse = LoopFuse(repeat_threshold=3)
    a, b = ToolCall("alpha", {"n": 1}), ToolCall("beta", {"n": 2})
    verdicts = _drive(fuse, [a, b, a, b])
    assert [v.allowed for v in verdicts] == [True, True, True, False]
    assert "cycling" in verdicts[-1].reason
    assert max(fuse.state.counts.values()) < 3


def test_three_step_cycle_is_reported_with_its_real_period():
    fuse = LoopFuse(repeat_threshold=9)
    a, b, c = (ToolCall("a", {}), ToolCall("b", {}), ToolCall("c", {}))
    verdicts = _drive(fuse, [a, b, c, a, b, c])
    assert verdicts[-1].blocked
    assert "3 tool calls" in verdicts[-1].reason


def test_cycle_detection_can_be_switched_off():
    fuse = LoopFuse(repeat_threshold=5, detect_cycles=False)
    a, b = ToolCall("a", {}), ToolCall("b", {})
    assert all(v.allowed for v in _drive(fuse, [a, b] * 3))


def test_check_is_side_effect_free():
    fuse = LoopFuse(repeat_threshold=2)
    call = ToolCall("t", {})
    assert fuse.check(call).allowed
    assert fuse.check(call).allowed
    assert fuse.state.history == []


def test_reset_clears_state_between_requests():
    fuse = LoopFuse(repeat_threshold=2)
    call = ToolCall("t", {})
    _drive(fuse, [call, call])
    fuse.reset()
    assert fuse.check(call).allowed


def test_threshold_below_two_is_rejected():
    try:
        LoopFuse(repeat_threshold=1)
    except ValueError:
        return
    raise AssertionError("repeat_threshold=1 should be rejected")


def test_unserialisable_arguments_do_not_crash_the_guard():
    fuse = LoopFuse(repeat_threshold=2)
    weird = ToolCall("t", {"obj": object()})
    assert fuse.check(weird).allowed
    fuse.record(weird)
    assert fuse.check(ToolCall("t", {"obj": object()})) is not None
