from agentfuse import BudgetFuse, Price, estimate_prompt_tokens


class _FakeUsage:
    def __init__(self, p, c):
        self.prompt_tokens, self.completion_tokens = p, c


class _FakeResponse:
    def __init__(self, p, c):
        self.usage = _FakeUsage(p, c)


PRICE = Price(prompt_per_1k=0.10, completion_per_1k=0.30)


def test_estimator_is_chars_over_four_rounded_up():
    assert estimate_prompt_tokens("a" * 400) == 100
    assert estimate_prompt_tokens("a" * 401) == 101  # rounded up, never down
    assert estimate_prompt_tokens("") == 1


def test_preflight_allows_inside_the_ceiling():
    fuse = BudgetFuse(max_usd=1.0, price=PRICE)
    assert fuse.preflight("hello " * 100, max_completion_tokens=200).allowed


def test_preflight_blocks_a_call_that_would_breach_the_ceiling():
    fuse = BudgetFuse(max_usd=0.01, price=PRICE)
    v = fuse.preflight("x" * 4000, max_completion_tokens=1000)
    assert v.blocked and "ceiling" in v.reason


def test_preflight_charges_the_full_completion_cap():
    """Pessimistic on purpose: a guard that under-estimates lets through the exact call
    it exists to stop."""
    fuse = BudgetFuse(max_usd=0.05, price=PRICE)
    assert fuse.preflight("", max_completion_tokens=100).allowed      # ~$0.03
    assert fuse.preflight("", max_completion_tokens=1000).blocked     # ~$0.30


def test_token_ceiling_binds_independently_of_dollars():
    fuse = BudgetFuse(max_tokens=500, price=Price())   # free, but capped in tokens
    assert fuse.preflight("x" * 400, max_completion_tokens=100).allowed
    assert fuse.preflight("x" * 400, max_completion_tokens=1000).blocked


def test_real_usage_is_what_moves_the_meter():
    fuse = BudgetFuse(max_usd=1.0, price=PRICE)
    cost = fuse.record(prompt_tokens=1000, completion_tokens=1000)
    assert cost == 0.40
    assert fuse.spent_usd == 0.40 and fuse.spent_tokens == 2000 and fuse.calls == 1


def test_spending_accumulates_until_preflight_refuses():
    fuse = BudgetFuse(max_usd=0.5, price=PRICE)
    fuse.record(1000, 1000)                       # $0.40 spent
    assert fuse.preflight("", max_completion_tokens=100).allowed     # +$0.03 -> 0.43
    assert fuse.preflight("", max_completion_tokens=1000).blocked    # +$0.30 -> 0.70


def test_missing_usage_books_zero_rather_than_guessing():
    fuse = BudgetFuse(max_usd=1.0, price=PRICE)

    class NoUsage:
        usage = None

    assert fuse.record_response(NoUsage()) == 0.0
    assert fuse.spent_usd == 0.0 and fuse.calls == 1


def test_record_response_reads_openai_shaped_usage():
    fuse = BudgetFuse(max_usd=1.0, price=PRICE)
    assert fuse.record_response(_FakeResponse(500, 500)) == 0.20


def test_strict_pricing_refuses_to_spend_blind():
    fuse = BudgetFuse(max_usd=1.0, price=Price(), strict_pricing=True)
    v = fuse.preflight("hello", max_completion_tokens=10)
    assert v.blocked and "blind" in v.reason


def test_reset_clears_the_meter_but_keeps_the_ceiling():
    fuse = BudgetFuse(max_usd=0.5, price=PRICE)
    fuse.record(1000, 1000)
    fuse.reset()
    assert fuse.spent_usd == 0.0 and fuse.max_usd == 0.5
