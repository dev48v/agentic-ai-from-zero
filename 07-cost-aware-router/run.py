"""Runnable demo for the Cost-Aware Agent Router (NVIDIA NIM).

    python 07-cost-aware-router/run.py

A MIXED batch of six queries with per-task budgets exercises all four sub-points:

  1. "capital of Japan?"       trivial · generous → DIRECT, confident → EARLY-EXIT (cheap)
  2. "why is the sky blue?"    medium  · generous → CoT, confident → EARLY-EXIT (skips escalate)
  3. "17 sheep, all but 9…"    medium  · generous → CoT reasoning step
  4. "prove √2 is irrational"  hard    · generous → ESCALATE (best-of-3, priced like frontier)
  5. "design a rate limiter"   hard    · TIGHT $  → wants ESCALATE, budget blocks it → DEGRADE
  6. "capital of Japan?"       (repeat)          → CACHE HIT, $0

The model only answers + rates its own confidence. The complexity score, the tier choice,
the budget ceiling, the early-exit, the degrade, the best-of-N pick, and the ledger are
deterministic Python. After routing, we MEASURE the true always-escalate baseline (by
actually running the escalate tier on every query) and report the $ SAVED by routing.
"""

from __future__ import annotations

import logging
import os
import sys

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _REPO_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from agent import CostAwareRouter  # noqa: E402
from router import (  # noqa: E402
    Budget, CostLedger, EARLY_EXIT_CONFIDENCE, PRICE_TABLE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    stream=sys.stdout,
)

LEDGER_PATH = os.path.join(_PROJECT_DIR, "ledger.jsonl")

# (query, budget, one-line intent) — a mix of cheap early-exits, a real escalate, a
# budget-forced degrade, and a repeat that hits the cache.
BATCH = [
    ("What is the capital of Japan?",
     Budget(max_cost=0.05, label="generous"), "trivial lookup — expect DIRECT + early-exit"),
    ("Explain in two sentences why the sky is blue.",
     Budget(max_cost=0.05, label="generous"), "medium — expect CoT + early-exit (skips escalate)"),
    ("A farmer has 17 sheep and all but 9 run away. How many sheep are left?",
     Budget(max_cost=0.05, label="generous"), "medium reasoning — a classic trip-up"),
    ("Rigorously prove that the square root of 2 is irrational.",
     Budget(max_cost=0.10, label="generous"), "hard — expect ESCALATE (best-of-3)"),
    ("Design a fault-tolerant distributed rate limiter and justify token-bucket vs sliding-window.",
     Budget(max_cost=0.004, label="tight"), "hard but TIGHT budget — escalate blocked → DEGRADE"),
    ("What is the capital of Japan?",
     Budget(max_cost=0.05, label="generous"), "repeat of #1 — expect CACHE hit, $0"),
]


def _rule(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _on_event(kind: str, d: dict) -> None:
    """Readable, per-query narration of the routing decisions as they happen."""
    if kind == "classify":
        print(f"  ⚖️  complexity = {d['complexity'].upper()} (score {d['score']}) "
              f"→ entry tier `{d['want']}`   signals: {'; '.join(d['signals']) or 'none'}")
    elif kind == "degrade_start":
        print(f"  💸 BUDGET: `{d['want']}` would cost ~${d['projected']:.5f} but cap is "
              f"${d['budget']:.4f} → DEGRADE, start at `{d['tier']}` instead")
    elif kind == "ran":
        note = f" (best-of-{d['samples']})" if d['samples'] > 1 else ""
        print(f"  ▶️  ran `{d['tier']}`{note} → {d['tokens']} tokens · ${d['cost']:.6f} · "
              f"self-confidence {d['confidence']:.2f}")
        print(f"      answer: {_short(d['answer'])}")
    elif kind == "early_exit":
        print(f"  ✅ EARLY-EXIT at `{d['tier']}` — confidence {d['confidence']:.2f} ≥ "
              f"{EARLY_EXIT_CONFIDENCE:.2f}; NOT paying to escalate.")
    elif kind == "escalate":
        print(f"  ↗️  confidence {d['confidence']:.2f} < {EARLY_EXIT_CONFIDENCE:.2f} → "
              f"escalate `{d['frm']}` → `{d['to']}`")
    elif kind == "degrade_escalate":
        print(f"  💸 BUDGET: escalate to `{d['blocked']}` (~${d['projected']:.5f}) would "
              f"breach ${d['budget']:.4f} (spent ${d['spent']:.6f}) → DEGRADE, keep `{d['tier']}` answer")
    elif kind == "cache":
        print("  ♻️  CACHE HIT — identical query already answered; returning it for $0.")
    elif kind == "refused":
        print(f"  ⛔ REFUSED — even the cheapest tier exceeds the ${d['budget']:.4f} budget.")


def _short(text: str, n: int = 100) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _price_table() -> str:
    lines = ["  tier      $/1k tokens   calls   what it does"]
    for name in ("direct", "cot", "escalate"):
        t = PRICE_TABLE[name]
        lines.append(f"  {t.name:<9} {t.rate_per_1k:>10.4f}   {t.samples:>5}   {t.blurb}")
    return "\n".join(lines)


def main() -> int:
    _rule("COST-AWARE AGENT ROUTER — a mixed batch on NVIDIA NIM (meta/llama-3.1-8b-instruct)")
    print("The model answers + self-rates; Python decides the TIER, the BUDGET, the early-exit,")
    print("and the ledger. Route by complexity → cheap tier first → early-exit if confident →")
    print(f"escalate only if unsure AND the budget allows. Early-exit bar = confidence "
          f"≥ {EARLY_EXIT_CONFIDENCE:.2f}.\n")
    print("Price table (blended $/1k tokens — models a cheap→frontier ladder on one warm model):")
    print(_price_table())

    ledger = CostLedger(LEDGER_PATH, reset=True)
    router = CostAwareRouter(ledger=ledger, on_event=_on_event)

    results = []
    for i, (query, budget, intent) in enumerate(BATCH, start=1):
        _rule(f"QUERY {i}/{len(BATCH)}  ·  budget ${budget.max_cost:.4f} ({budget.label})  ·  {intent}")
        print(f"  Q: {query}")
        results.append(router.route(i, query, budget))

    # ---- measure the REAL always-escalate baseline, then fill $ saved per row ----
    _rule("BASELINE — measuring the true always-escalate cost (running escalate on every query)")
    print("For an honest '$ saved', we actually run the escalate tier on each query and price")
    print("its REAL tokens — reusing the router's own escalate call where it already paid for one.\n")
    for res in results:
        row = res.row
        reuse = res.attempts[-1] if res.attempts else None
        base = router.measure_always_escalate(row.query, reuse=reuse)
        row.always_escalate_cost = base
        row.saved = round(base - row.cost, 6)
        print(f"  #{row.idx} {_short(row.query, 44):<45} always-escalate ${base:.6f}  "
              f"vs actual ${row.cost:.6f}  → saved ${row.saved:.6f}")

    ledger.write_jsonl(LEDGER_PATH)

    # ---- the cost-per-decision analytics table + summary ----
    _rule("COST-PER-DECISION LEDGER (sub-point 4) — tokens + $ per query, tier, flags")
    print(ledger.render_table())

    s = ledger.summary()
    _rule("SPEND SUMMARY — what routing SAVED vs always sending everything to the frontier")
    print(f"  queries              : {s['queries']}")
    print(f"  total tokens (real)  : {s['total_tokens']}")
    print(f"  total spend          : ${s['total_cost']:.6f}")
    print(f"  avg cost / query     : ${s['avg_cost_per_query']:.6f}")
    print(f"  always-escalate base : ${s['always_escalate_baseline']:.6f}  (measured, not modelled)")
    print(f"  SAVED by routing     : ${s['saved_vs_always_escalate']:.6f}  "
          f"({s['saved_pct']:.1f}% cheaper)")
    print(f"  early-exits          : {s['early_exits']}   "
          f"degraded: {s['degraded']}   cache hits: {s['cache_hits']}   refused: {s['refused']}")

    _rule("THE FOUR SUB-POINTS, IN THIS RUN")
    print("1. token budgeting per task — every query carried a max-$ ceiling; query 5's tight "
          f"${BATCH[4][1].max_cost:.4f} budget BLOCKED the escalate it wanted → it degraded.")
    print("2. route by complexity + cost — a heuristic scored each query trivial/medium/hard and "
          "mapped it to direct / cot / escalate, each with a published $/1k rate.")
    print("3. early exit on confidence — the cheap tier's self-rated confidence cleared "
          f"{EARLY_EXIT_CONFIDENCE:.2f} on the easy queries, so the router STOPPED and never paid to escalate.")
    print("4. cost-per-decision analytics — the ledger above prices every query from REAL API "
          "token usage and reports the $ saved vs a measured always-escalate baseline.")
    print(f"\n{s['queries']} rows written to {LEDGER_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
