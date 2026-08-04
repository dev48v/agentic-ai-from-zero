"""Cost-Aware Agent Router — spends the least money that still solves the task.

The MODEL does exactly three things, and never touches a number that matters:
  (1) ANSWER a query at a given effort tier,
  (2) optionally REASON step-by-step first (the `cot` / `escalate` tiers), and
  (3) SELF-RATE its confidence in its own answer (0..1).

Everything that DECIDES or COUNTS is deterministic Python in `router.py`: the complexity
classifier, the tier choice, the budget ceiling, the early-exit threshold, the best-of-N
pick, the price, and the ledger. The model proposes an answer + a confidence; Python
decides how much to spend getting it.

The route per query:

  CACHE   — exact-match cache-before-call: if we have answered this query, return it for $0.
  CLASSIFY— a cheap heuristic scores the query trivial / medium / hard (no API call).
  START   — map complexity → an entry tier, then STEP DOWN if the budget can't afford it
            (the first way the budget can degrade a query).
  RUN     — call the model at the current tier (best-of-N for `escalate`, picked in Python),
            record the REAL token usage → real $.
  EARLY EXIT — if the tier's self-rated confidence clears the bar, STOP: don't pay to escalate.
  ESCALATE or DEGRADE — otherwise try the next dearer tier; but FIRST the budget gate
            projects its cost — if that would breach the ceiling, DEGRADE (keep the best cheap
            answer we have, flagged) instead of overspending.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from common.client import DEFAULT_MODEL, get_client
from router import (
    Budget, Complexity, CostLedger, LedgerRow, PRICE_TABLE,
    EARLY_EXIT_CONFIDENCE, classify_complexity, entry_tier, estimate_cost,
    estimate_tokens, next_tier,
)

logger = logging.getLogger("cost-router")


# --------------------------------------------------------------------------- #
# One model call's real result — the answer, the self-rated confidence, and the
# REAL token usage the API reported (this is what the ledger prices, not an estimate).
# --------------------------------------------------------------------------- #
@dataclass
class Attempt:
    tier: str
    answer: str
    confidence: float
    reasoning: str
    prompt_tokens: int
    completion_tokens: int
    samples: int = 1                 # calls made (>1 for best-of-N escalate)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> float:
        return PRICE_TABLE[self.tier].price(self.prompt_tokens, self.completion_tokens)


@dataclass
class RouteResult:
    row: LedgerRow
    attempts: list[Attempt]


# The three prompts differ only in HOW MUCH THINKING they ask for — which is exactly the
# thing that makes them cost different amounts. All demand STRICT JSON so Python can read
# the answer + the self-rating without trusting prose.
_DIRECT_SYSTEM = (
    "You answer in as FEW tokens as possible. Return STRICT JSON only: "
    '{"answer": <short answer>, "confidence": <0..1>}. '
    "No explanation, no working, no markdown. `confidence` is how sure you are the answer "
    "is correct (1.0 = certain). If you are guessing, say so with a low confidence."
)
_COT_SYSTEM = (
    "Think step by step, THEN answer. Return STRICT JSON only: "
    '{"reasoning": <your concise step-by-step>, "answer": <final answer>, "confidence": <0..1>}. '
    "No markdown fences. `confidence` is how sure you are the final answer is correct "
    "(1.0 = certain); lower it honestly if a step was shaky."
)
# The escalate tier reuses the CoT prompt but is sampled N times at higher temperature for
# diversity; Python then keeps the most-confident sample (a deterministic best-of-N pick).
_ESCALATE_SYSTEM = _COT_SYSTEM


class CostAwareRouter:
    def __init__(self, model: str = DEFAULT_MODEL, ledger: CostLedger | None = None,
                 on_event=None) -> None:
        self.model = model
        self.client = get_client()
        self.ledger = ledger or CostLedger()
        self.cache: dict[str, Attempt] = {}     # cache-before-call: query -> best Attempt
        self.on_event = on_event                # optional UI callback(str, dict)

    # ------------------------------------------------------------------ #
    # One real model call at a tier — returns the answer + self-rating + REAL usage.
    # ------------------------------------------------------------------ #
    def _call(self, query: str, tier: str, temperature: float) -> Attempt:
        system = {"direct": _DIRECT_SYSTEM, "cot": _COT_SYSTEM, "escalate": _ESCALATE_SYSTEM}[tier]
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": query}]
        raw, pt, ct = "", 0, 0
        for attempt in (1, 2):
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature)
            raw = (resp.choices[0].message.content or "").strip()
            u = resp.usage                       # REAL token usage from NVIDIA NIM
            pt += int(getattr(u, "prompt_tokens", 0) or 0)
            ct += int(getattr(u, "completion_tokens", 0) or 0)
            parsed = _parse_json(raw)
            if parsed is not None:
                answer = str(parsed.get("answer", "")).strip()
                conf = _clamp(parsed.get("confidence", 0.5))
                reasoning = str(parsed.get("reasoning", "")).strip()
                if answer:
                    return Attempt(tier, answer, conf, reasoning, pt, ct, samples=1)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": "That was not valid JSON. Reply ONLY the JSON object."})
        # Both attempts unparseable: keep the raw text as the answer, low confidence, still
        # count the (real) tokens we paid for — never silently drop spend.
        return Attempt(tier, raw or "(no answer)", 0.3, "", pt, ct, samples=1)

    def _run_tier(self, query: str, tier: str) -> Attempt:
        """Run a tier once (direct/cot) or best-of-N (escalate), summing REAL tokens."""
        spec = PRICE_TABLE[tier]
        if spec.samples == 1:
            return self._call(query, tier, temperature=0.0)
        # escalate: N diverse samples, then a DETERMINISTIC pick — the highest self-confidence
        # (ties broken by the longer reasoning). The choice is Python's, not the model's.
        cands = [self._call(query, tier, temperature=0.7) for _ in range(spec.samples)]
        pt = sum(c.prompt_tokens for c in cands)
        ct = sum(c.completion_tokens for c in cands)
        best = max(cands, key=lambda c: (c.confidence, len(c.reasoning)))
        return Attempt(tier, best.answer, best.confidence, best.reasoning, pt, ct,
                       samples=spec.samples)

    def _emit(self, kind: str, **data) -> None:
        if self.on_event:
            self.on_event(kind, data)

    # ------------------------------------------------------------------ #
    # The route — classify, pick a budget-affordable tier, run, early-exit or degrade.
    # ------------------------------------------------------------------ #
    def route(self, idx: int, query: str, budget: Budget) -> RouteResult:
        # 0) cache-before-call — the cheapest possible answer is one you already have.
        if query in self.cache:
            hit = self.cache[query]
            self._emit("cache", idx=idx, query=query)
            row = LedgerRow(idx, query, "—", budget.max_cost, "cache", "cache", [],
                            0, 0, 0.0, hit.confidence, early_exit=False, degraded=False,
                            cached=True, refused=False, answer=hit.answer)
            return RouteResult(self.ledger.add(row), [])

        # 1) classify complexity (deterministic, no API call) and pick the entry tier.
        comp: Complexity = classify_complexity(query)
        want = entry_tier(comp.label)
        self._emit("classify", idx=idx, query=query, complexity=comp.label,
                   score=comp.score, signals=comp.signals, want=want)

        # 2) budget step-down: start at the dearest tier ≤ `want` that the budget affords.
        tier = budget.best_affordable_at_or_below(want, query)
        degraded = False
        if tier is None:
            # not even the cheapest tier fits — refuse rather than overspend.
            self._emit("refused", idx=idx, query=query, budget=budget.max_cost)
            row = LedgerRow(idx, query, comp.label, budget.max_cost, want, "refused", [],
                            0, 0, 0.0, 0.0, early_exit=False, degraded=False,
                            cached=False, refused=True,
                            answer=f"(refused — cannot answer within ${budget.max_cost:.4f})")
            return RouteResult(self.ledger.add(row), [])
        if tier != want:
            degraded = True     # forced to start cheaper than complexity wanted
            self._emit("degrade_start", idx=idx, want=want, tier=tier,
                       projected=estimate_cost(query, want), budget=budget.max_cost)

        # 3) run the ladder from `tier` upward, honouring confidence AND budget.
        attempts: list[Attempt] = []
        tiers_run: list[str] = []
        spent_cost, spent_tokens = 0.0, 0
        early_exit = False

        while True:
            att = self._run_tier(query, tier)
            attempts.append(att)
            tiers_run.append(tier)
            spent_cost += att.cost
            spent_tokens += att.total_tokens
            self._emit("ran", idx=idx, tier=tier, samples=att.samples,
                       tokens=att.total_tokens, cost=att.cost, confidence=att.confidence,
                       answer=att.answer)

            # early exit — confident enough that paying to escalate would be waste.
            if att.confidence >= EARLY_EXIT_CONFIDENCE:
                early_exit = tier != "escalate" and not degraded
                if early_exit:
                    self._emit("early_exit", idx=idx, tier=tier, confidence=att.confidence)
                break

            nxt = next_tier(tier)
            if nxt is None:
                break                          # already at the top of the ladder

            # budget gate BEFORE escalating — project the next tier's cost.
            proj_c, proj_t = estimate_cost(query, nxt), estimate_tokens(query, nxt)
            if not budget.affords(spent_cost, spent_tokens, proj_c, proj_t):
                degraded = True                # wanted more effort, budget said no
                self._emit("degrade_escalate", idx=idx, tier=tier, blocked=nxt,
                           projected=proj_c, spent=round(spent_cost, 6),
                           budget=budget.max_cost)
                break
            self._emit("escalate", idx=idx, frm=tier, to=nxt, confidence=att.confidence)
            tier = nxt

        best = attempts[-1]
        pt = sum(a.prompt_tokens for a in attempts)
        ct = sum(a.completion_tokens for a in attempts)
        cost = round(sum(a.cost for a in attempts), 6)
        self.cache[query] = best              # remember for a future cache hit

        row = LedgerRow(
            idx=idx, query=query, complexity=comp.label, budget=budget.max_cost,
            entry_tier=want, final_tier=tiers_run[-1], tiers_run=tiers_run,
            prompt_tokens=pt, completion_tokens=ct, cost=cost, confidence=best.confidence,
            early_exit=early_exit, degraded=degraded, cached=False, refused=False,
            answer=best.answer)
        return RouteResult(self.ledger.add(row), attempts)

    # ------------------------------------------------------------------ #
    # Baseline: the REAL cost this query WOULD have incurred on always-escalate.
    # Measured, not modelled — we actually run the escalate tier (reusing the router's
    # own escalate result when it already paid for one).
    # ------------------------------------------------------------------ #
    def measure_always_escalate(self, query: str, reuse: Attempt | None = None) -> float:
        if reuse is not None and reuse.tier == "escalate":
            return reuse.cost
        return self._run_tier(query, "escalate").cost


def _clamp(v, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return 0.5


def _parse_json(raw: str) -> dict | None:
    """Tolerant JSON extraction — strips ```fences``` and slices the outermost object."""
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.split("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        # last resort: pull "answer" / "confidence" out of a nearly-JSON blob
        m_a = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
        m_c = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        if m_a:
            return {"answer": m_a.group(1), "confidence": float(m_c.group(1)) if m_c else 0.5}
        return None
