"""BudgetFuse — a hard spend ceiling for one request, enforced before the call.

From Project 7 (cost-aware router), where `Budget.affords()` was a pre-flight check that
could refuse or degrade a query, and from Project 11, where a `CostBudgetRule` alerted on
window spend after the fact.

The pre-flight/actual split is the part people get wrong, so it is explicit here:

  * `preflight(prompt_text, max_completion_tokens)` uses a deliberately crude estimator
    (chars/4 for the prompt, plus the completion cap you already have to declare) to decide
    whether the NEXT call may run. Estimates are for gating, never for reporting.
  * `record(prompt_tokens, completion_tokens)` takes the REAL numbers off the provider's
    `usage` object and is the only thing that moves `spent_usd`. Project 7 reported a
    78.5% saving using real usage exactly because the estimator was never allowed near the
    final figure.

The estimator is intentionally pessimistic in one direction: it charges the FULL completion
cap even though most completions come in short. A budget guard that under-estimates lets
the very call it was meant to stop go through.

Prices are per 1,000 tokens and default to 0.0. A zero price is not "free" as far as the
fuse is concerned — with no price the $ ceiling cannot bind, so set `max_tokens` too, and
`strict_pricing=True` will refuse to run at all when a price is missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core import Verdict

__all__ = ["Price", "BudgetFuse", "estimate_prompt_tokens"]

# The universal crude tokenizer: ~4 characters per token for English. Wrong in the third
# decimal, right enough to stop a runaway, and it costs nothing to compute.
CHARS_PER_TOKEN = 4


def estimate_prompt_tokens(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


@dataclass(frozen=True)
class Price:
    """$ per 1,000 tokens, split because completions usually cost more than prompts."""

    prompt_per_1k: float = 0.0
    completion_per_1k: float = 0.0

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(
            prompt_tokens / 1000.0 * self.prompt_per_1k
            + completion_tokens / 1000.0 * self.completion_per_1k,
            8,
        )


@dataclass
class BudgetFuse:
    """One request's ceiling. Create a fresh one per request, or call `reset()`."""

    max_usd: float = float("inf")
    max_tokens: int = 2 ** 62
    price: Price = field(default_factory=Price)
    strict_pricing: bool = False

    spent_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    name: str = "budget"

    @property
    def spent_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def remaining_usd(self) -> float:
        return round(self.max_usd - self.spent_usd, 8)

    def preflight(self, prompt_text: str = "", max_completion_tokens: int = 0,
                  prompt_tokens: int | None = None) -> Verdict:
        """May the next model call run? Pass `prompt_tokens` if you already counted them."""
        if self.strict_pricing and self.price.prompt_per_1k <= 0 \
                and self.price.completion_per_1k <= 0:
            return Verdict.stop(
                self.name,
                "no price configured and strict_pricing is on — refusing to spend blind")

        est_prompt = prompt_tokens if prompt_tokens is not None \
            else estimate_prompt_tokens(prompt_text)
        est_total = est_prompt + max(0, max_completion_tokens)
        est_cost = self.price.cost(est_prompt, max(0, max_completion_tokens))

        # 1e-9 slack so a budget of exactly the projected spend is not lost to float noise.
        if self.spent_usd + est_cost > self.max_usd + 1e-9:
            return Verdict.stop(
                self.name,
                f"projected spend ${self.spent_usd + est_cost:.6f} would breach the "
                f"${self.max_usd:.6f} ceiling",
                evidence=f"spent ${self.spent_usd:.6f}, this call ~${est_cost:.6f} "
                         f"(~{est_total} tok)",
            )
        if self.spent_tokens + est_total > self.max_tokens:
            return Verdict.stop(
                self.name,
                f"projected {self.spent_tokens + est_total} tokens would breach the "
                f"{self.max_tokens}-token ceiling",
                evidence=f"spent {self.spent_tokens} tok, this call ~{est_total} tok",
            )
        return Verdict.ok(
            self.name,
            f"call affordable (~${est_cost:.6f}, ${self.remaining_usd:.6f} left)")

    def record(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Book the REAL usage from the provider response. Returns this call's cost."""
        cost = self.price.cost(prompt_tokens, completion_tokens)
        self.prompt_tokens += int(prompt_tokens)
        self.completion_tokens += int(completion_tokens)
        self.spent_usd = round(self.spent_usd + cost, 8)
        self.calls += 1
        return cost

    def record_response(self, response) -> float:
        """Convenience for OpenAI-compatible responses: reads `response.usage`.

        Missing usage books ZERO rather than guessing — an invented number in a spend
        ledger is worse than a gap, because it looks authoritative.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            self.calls += 1
            return 0.0
        return self.record(int(getattr(usage, "prompt_tokens", 0) or 0),
                           int(getattr(usage, "completion_tokens", 0) or 0))

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.spent_tokens,
            "spent_usd": round(self.spent_usd, 8),
            "max_usd": self.max_usd,
            "remaining_usd": self.remaining_usd,
        }

    def reset(self) -> None:
        self.spent_usd = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
