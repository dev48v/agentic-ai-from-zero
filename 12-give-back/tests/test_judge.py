from agentfuse import HardCheck, JudgeGate

MENTIONS_30 = HardCheck("mentions-30pct", "states the 30% discount figure",
                        lambda a: "30%" in a)
MENTIONS_RAISES = HardCheck("documents-raises", "documents what it raises",
                            lambda a: "raises" in a.lower())
STYLE = HardCheck("british-spelling", "uses British spelling",
                  lambda a: "colour" in a.lower(), required=False)


def test_a_perfect_self_score_cannot_override_a_failed_hard_check():
    """The exact Project 10 failure: the 8B judge scored its own draft 1.00 on a criterion
    the text did not satisfy. The hard check has to win."""
    gate = JudgeGate([MENTIONS_30], threshold=0.80)
    v = gate.evaluate("The price is reduced considerably.", llm_score=1.00)
    assert not v.passed
    assert "scored this a PASS anyway" in v.reason
    assert [r.id for r in v.failed_required] == ["mentions-30pct"]


def test_pass_needs_both_halves():
    gate = JudgeGate([MENTIONS_30], threshold=0.80)
    assert gate.evaluate("A 30% discount applies.", 0.85).passed


def test_a_low_self_score_still_fails_even_with_green_hard_checks():
    """The gate is an AND, not a vote — the model may lower it, never raise it."""
    gate = JudgeGate([MENTIONS_30], threshold=0.80)
    v = gate.evaluate("A 30% discount applies.", 0.40)
    assert not v.passed and "below threshold" in v.reason


def test_advisory_checks_report_but_do_not_gate():
    gate = JudgeGate([MENTIONS_30, STYLE], threshold=0.80)
    v = gate.evaluate("A 30% discount applies to the color range.", 0.90)
    assert v.passed
    assert any(r.id == "british-spelling" and not r.passed for r in v.results)


def test_a_check_that_raises_counts_as_failed():
    boom = HardCheck("explodes", "cannot possibly pass",
                     lambda a: 1 / 0)  # noqa: ARG005
    v = JudgeGate([boom]).evaluate("anything", 1.0)
    assert not v.passed
    assert v.results[0].error.startswith("ZeroDivisionError")


def test_hard_score_is_the_fraction_of_required_checks_passed():
    gate = JudgeGate([MENTIONS_30, MENTIONS_RAISES], threshold=0.80)
    v = gate.evaluate("A 30% discount applies.", 1.0)
    assert v.hard_score == 0.5


def test_critique_note_names_only_the_real_gaps():
    gate = JudgeGate([MENTIONS_30, MENTIONS_RAISES], threshold=0.80)
    note = gate.evaluate("A 30% discount applies.", 1.0).critique_note()
    assert "documents what it raises" in note
    assert "30% discount figure" not in note


def test_a_passing_verdict_has_no_critique():
    gate = JudgeGate([MENTIONS_30], threshold=0.80)
    assert gate.evaluate("A 30% discount applies.", 0.99).critique_note() == ""


def test_gate_with_no_hard_checks_is_score_only():
    gate = JudgeGate([], threshold=0.80)
    assert gate.evaluate("anything", 0.81).passed
    assert not gate.evaluate("anything", 0.79).passed
    assert gate.evaluate("anything", 0.81).hard_score == 1.0
