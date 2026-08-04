"""The deterministic half of the Cost-Aware Router — tiers, the price table, the
complexity classifier, the token/cost budget, and the spend ledger. NO model calls
live here on purpose.

Four ideas, one per sub-point:

  1. token budgeting per task    -> `Budget`: each task carries a hard max-cost ($) and
                                    max-tokens ceiling. `Budget.affords()` is a pre-flight
                                    check the router runs BEFORE it pays for a tier; if the
                                    projected spend would breach the ceiling it refuses /
                                    degrades instead of overspending.
  2. route by complexity + cost  -> `classify_complexity` (a cheap heuristic — no API call)
                                    scores each query trivial / medium / hard, and
                                    `entry_tier` maps that to a TIER: `direct` (one cheap
                                    call), `cot` (a reasoning step), `escalate` (best-of-N).
                                    Each tier has a published $/1k rate in `PRICE_TABLE`.
  3. early exit on confidence    -> `EARLY_EXIT_CONFIDENCE`: after a cheap tier answers and
                                    self-rates, if its confidence clears the bar the router
                                    STOPS — it never pays for the expensive tier it could
                                    have used. (The routing decision is Python; only the
                                    answer + the self-rating come from the model.)
  4. cost-per-decision analytics -> `CostLedger`: a running, append-only record of tokens +
                                    $ per query, the tier taken, and whether it early-exited
                                    / degraded / hit cache — plus a summary (total spend, avg
                                    cost/query, and $ SAVED vs an always-escalate baseline).

The split is the whole point: the MODEL answers and rates its own confidence; every
NUMBER — the complexity score, the tier choice, the budget verdict, the price, the
$-saved — is deterministic Python, so the same inputs always earn the same routing
decision and the ledger is reproducible and auditable.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field

# --------------------------------------------------------------------------- #
# 2. Route by complexity + cost — the TIERS and their published price table.
# --------------------------------------------------------------------------- #
# We route by EFFORT TIER, not by swapping model names (the free NIM tier serves one
# warm model, meta/llama-3.1-8b-instruct — the 70B/Nemotron models cold-start for >100s).
# The price DIFFERENCE is modelled two ways, exactly as real cheap→frontier routing bills:
#   • each tier spends a different number of REAL tokens (one terse call vs a reasoning
#     call vs best-of-N), AND
#   • each tier carries a different blended $/1k RATE, as if `direct` ran on a small/cheap
#     model and `escalate` ran on a frontier one.
# cost = (prompt_tokens + completion_tokens) / 1000 * rate_per_1k.


@dataclass(frozen=True)
class Tier:
    name: str
    rate_per_1k: float      # blended $ per 1,000 tokens (models cheap→frontier pricing)
    samples: int            # how many model calls this tier makes (best-of-N for escalate)
    blurb: str

    def price(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round((prompt_tokens + completion_tokens) / 1000.0 * self.rate_per_1k, 6)


# The ladder, cheapest → dearest. `escalate` is best-of-3 AND priced like a frontier model,
# so it is ~60× the $/1k of `direct` — which is what makes cheap routing worth it.
PRICE_TABLE: dict[str, Tier] = {
    "direct":   Tier("direct",   0.0005, 1, "one terse call — a cheap/small model answers directly"),
    "cot":      Tier("cot",      0.0030, 1, "one call that reasons step-by-step before answering"),
    "escalate": Tier("escalate", 0.0300, 3, "best-of-3 reasoned samples, priced like a frontier model"),
}
LADDER: list[str] = ["direct", "cot", "escalate"]   # ascending cost/effort

# 3. Early exit — the confidence bar. A tier whose self-rated confidence clears this is
# trusted; the router stops and does NOT pay to escalate. Below it → escalate (budget
# permitting). 0.75 is deliberately strict: "sure enough to not spend more".
EARLY_EXIT_CONFIDENCE = 0.75


def entry_tier(label: str) -> str:
    """Map a complexity label to the tier the router STARTS at (before any budget check)."""
    return {"trivial": "direct", "medium": "cot", "hard": "escalate"}[label]


def next_tier(tier: str) -> str | None:
    """The next dearer tier on the ladder, or None if already at the top."""
    i = LADDER.index(tier)
    return LADDER[i + 1] if i + 1 < len(LADDER) else None


def cheaper_or_equal(tier: str) -> list[str]:
    """Tiers at or below `tier` on the ladder, dearest-first (for budget step-down)."""
    return list(reversed(LADDER[: LADDER.index(tier) + 1]))


# --------------------------------------------------------------------------- #
# 2b. The complexity classifier — a cheap, deterministic heuristic (NO API call).
# --------------------------------------------------------------------------- #
# The roadmap allows "heuristic + optional model self-rating"; we default to the heuristic
# so classification itself costs $0 and is reproducible. A hard keyword ("prove", "design",
# "justify"…) dominates → hard; an explanation/arithmetic signal → medium; a short factual
# lookup → trivial. Signals are surfaced (like a risk gate) so the choice is legible.
_HARD_KW = re.compile(
    r"\b(prove|proof|rigorous|rigorously|derive|design|architect(?:ure)?|compare|justif\w*|"
    r"optim\w*|algorithm|trade[- ]?offs?|analy[sz]e|theorem|complexity|distributed|"
    r"fault[- ]tolerant)\b", re.I)
_MED_KW = re.compile(
    r"\b(explain|why|how|describe|summar\w+|reason|calculate|compute|solve|difference|"
    r"how many|what if|steps?)\b", re.I)
_TRIVIAL_START = re.compile(
    r"^\s*(what\s+is|who\s+is|where\s+is|when\s+(?:did|was|is)|capital\s+of|define|name\s+the)\b",
    re.I)
_ARITHMETIC = re.compile(r"\d.*[+\-*/x×÷]|\b\d+\s*(?:plus|minus|times|divided)\b|\d.*\d", re.I)


@dataclass
class Complexity:
    label: str              # "trivial" | "medium" | "hard"
    score: int              # a small integer for display / ordering
    signals: list[str] = field(default_factory=list)


def classify_complexity(query: str) -> Complexity:
    """Score a query trivial / medium / hard from cheap surface features — no model call."""
    q = query.strip()
    words = len(q.split())
    signals: list[str] = []

    hard = bool(_HARD_KW.search(q))
    med = bool(_MED_KW.search(q))
    arith = bool(_ARITHMETIC.search(q))
    trivial_start = bool(_TRIVIAL_START.search(q))
    longish = words > 24

    if hard:
        signals.append(f"hard-reasoning keyword ({_HARD_KW.search(q).group(0).lower()})")
    if med:
        signals.append(f"explanation/derivation keyword ({_MED_KW.search(q).group(0).lower()})")
    if arith:
        signals.append("contains arithmetic / multiple numbers")
    if longish:
        signals.append(f"long prompt ({words} words)")
    if trivial_start and not (hard or med or arith):
        signals.append("short factual lookup pattern")

    # Categorical precedence: a hard keyword (or a long multi-part prompt) dominates.
    if hard or longish:
        label, score = "hard", 3
    elif med or arith:
        label, score = "medium", 2
    elif trivial_start:
        label, score = "trivial", 0
    else:
        # unknown shape: short → trivial, otherwise medium (spend a reasoning step to be safe)
        label, score = ("trivial", 0) if words <= 12 else ("medium", 2)
        signals.append("no strong signal — length-based default")

    return Complexity(label=label, score=score, signals=signals)


# --------------------------------------------------------------------------- #
# 1. Token budgeting per task — the hard ceiling + a pre-flight estimator.
# --------------------------------------------------------------------------- #
# You cannot know a call's real token count until AFTER you pay for it, so the budget
# gate uses a deterministic pre-flight ESTIMATE (chars/4 + a per-tier completion + system
# overhead, × the tier's sample count). The LEDGER, by contrast, records the REAL usage
# the API returns. Estimate to decide whether to spend; reconcile with truth after.
_SYSTEM_OVERHEAD_TOKENS = 90            # our fixed system-prompt footprint, roughly
_COMPLETION_EST = {"direct": 40, "cot": 230, "escalate": 260}   # typical answer size / call


def estimate_tokens(query: str, tier: str) -> int:
    """Deterministic pre-call token estimate for a tier (chars/4 prompt + a tier completion)."""
    prompt = _SYSTEM_OVERHEAD_TOKENS + max(1, len(query) // 4)
    per_call = prompt + _COMPLETION_EST[tier]
    return per_call * PRICE_TABLE[tier].samples


def estimate_cost(query: str, tier: str) -> float:
    """The projected $ the budget gate checks BEFORE running a tier."""
    return round(estimate_tokens(query, tier) / 1000.0 * PRICE_TABLE[tier].rate_per_1k, 6)


@dataclass
class Budget:
    """A per-task ceiling. `max_cost` is the hard $ cap; `max_tokens` an optional token cap.
    `label` is a human tag ('generous' / 'tight') for the transcript."""
    max_cost: float
    max_tokens: int = 1_000_000
    label: str = ""

    def affords(self, spent_cost: float, spent_tokens: int,
                projected_cost: float, projected_tokens: int) -> bool:
        """Would running the next tier keep BOTH the $ and token ceilings intact?"""
        return (spent_cost + projected_cost <= self.max_cost + 1e-9
                and spent_tokens + projected_tokens <= self.max_tokens)

    def best_affordable_at_or_below(self, entry: str, query: str) -> str | None:
        """Dearest tier ≤ `entry` whose SOLO projected cost fits the budget from a cold
        start — used to pick a starting tier when the intended one is too dear (degrade)."""
        for tier in cheaper_or_equal(entry):
            if self.affords(0.0, 0, estimate_cost(query, tier), estimate_tokens(query, tier)):
                return tier
        return None


# --------------------------------------------------------------------------- #
# 4. Cost-per-decision analytics — the running ledger + the summary.
# --------------------------------------------------------------------------- #
@dataclass
class LedgerRow:
    idx: int
    query: str
    complexity: str
    budget: float
    entry_tier: str                 # where complexity said to start
    final_tier: str                 # where we actually ended (may be degraded down)
    tiers_run: list[str]            # every tier we actually paid for, in order
    prompt_tokens: int
    completion_tokens: int
    cost: float                     # REAL spend for this query (from API usage)
    confidence: float
    early_exit: bool                # stopped at a cheap tier because it was confident enough
    degraded: bool                  # wanted more effort but the budget blocked it
    cached: bool                    # served from cache for $0
    refused: bool                   # nothing fit the budget at all
    answer: str
    always_escalate_cost: float = 0.0   # measured baseline: this query on the escalate tier
    saved: float = 0.0                  # always_escalate_cost - cost

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class CostLedger:
    """Append-only spend ledger. Each routed query is one row; the ledger also writes a
    durable JSONL file (mirrors Project 6's audit trail) and computes the summary — total
    spend, avg cost/query, and $ saved vs a measured always-escalate baseline."""

    def __init__(self, path: str | None = None, reset: bool = True) -> None:
        self.rows: list[LedgerRow] = []
        self.path = path
        if path and reset and os.path.exists(path):
            os.remove(path)
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def add(self, row: LedgerRow) -> LedgerRow:
        self.rows.append(row)
        return row

    def write_jsonl(self, path: str | None = None) -> None:
        """Persist the ledger as one JSON line per query (call AFTER the baseline is
        measured, so each row carries its always-escalate cost + $ saved)."""
        path = path or self.path
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for r in self.rows:
                fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    # -- the summary: what the whole batch cost, and what routing SAVED ---------- #
    def summary(self) -> dict:
        actual = round(sum(r.cost for r in self.rows), 6)
        baseline = round(sum(r.always_escalate_cost for r in self.rows), 6)
        saved = round(baseline - actual, 6)
        n = len(self.rows) or 1
        return {
            "queries": len(self.rows),
            "total_tokens": sum(r.total_tokens for r in self.rows),
            "total_cost": actual,
            "avg_cost_per_query": round(actual / n, 6),
            "always_escalate_baseline": baseline,
            "saved_vs_always_escalate": saved,
            "saved_pct": round(100.0 * saved / baseline, 1) if baseline else 0.0,
            "early_exits": sum(1 for r in self.rows if r.early_exit),
            "degraded": sum(1 for r in self.rows if r.degraded),
            "cache_hits": sum(1 for r in self.rows if r.cached),
            "refused": sum(1 for r in self.rows if r.refused),
        }

    # -- a compact, aligned analytics table for the console + the recorded run --- #
    def render_table(self) -> str:
        head = (f"{'#':>2}  {'query':<34}  {'cmplx':<7}  {'route':<16}  "
                f"{'tokens':>7}  {'cost $':>9}  {'conf':>5}  flags")
        lines = [head, "-" * len(head)]
        for r in self.rows:
            route = "→".join(r.tiers_run) if r.tiers_run else ("cache" if r.cached else "—")
            flags = []
            if r.cached:
                flags.append("CACHE $0")
            if r.early_exit:
                flags.append("early-exit")
            if r.degraded:
                flags.append("DEGRADED")
            if r.refused:
                flags.append("REFUSED")
            q = (r.query[:31] + "…") if len(r.query) > 32 else r.query
            lines.append(
                f"{r.idx:>2}  {q:<34}  {r.complexity:<7}  {route:<16}  "
                f"{r.total_tokens:>7}  {r.cost:>9.6f}  {r.confidence:>5.2f}  {', '.join(flags)}")
        s = self.summary()
        lines.append("-" * len(head))
        lines.append(
            f"{'':>2}  {'TOTAL':<34}  {'':<7}  {'':<16}  "
            f"{s['total_tokens']:>7}  {s['total_cost']:>9.6f}")
        return "\n".join(lines)
