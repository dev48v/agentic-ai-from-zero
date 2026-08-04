# 07 — Cost-Aware Agent Router

**Goal:** an agent that **spends the least money that still solves the task**. Trivial
questions take a single cheap call; hard ones escalate to a best-of-N frontier tier — and
only when they have to. Every query carries a **hard budget**, the router **exits early**
the moment a cheap answer is confident enough, **degrades** instead of overspending, and
writes a **cost-per-decision ledger** that proves how much routing saved.

The model does exactly three things: **answer** a query at a given effort tier, optionally
**reason** first, and **self-rate its own confidence**. **Every number that matters is
deterministic Python** — the complexity score, the tier choice, the budget ceiling, the
early-exit threshold, the best-of-N pick, the price, and the $-saved. The model proposes an
answer + a confidence; Python decides how much to spend getting it.

## The four ideas (hand-rolled, no framework)

| # | Sub-point | Where it lives |
|---|-----------|----------------|
| 1 | **token budgeting per task** | [`router.py`](router.py) `Budget` — each task carries a hard `max_cost` ($) and `max_tokens` ceiling. `Budget.affords()` is a **pre-flight** check the router runs *before* it pays for a tier; a pre-flight token **estimate** projects the spend, and if it would breach the ceiling the router **refuses or degrades** instead of overspending. |
| 2 | **route by complexity + cost** | [`router.py`](router.py) `classify_complexity` (a cheap heuristic — **no API call**) scores each query trivial / medium / hard, and `entry_tier` maps that to a **TIER**: `direct` (one terse call), `cot` (a reasoning step), `escalate` (best-of-3). Each tier has a published `$/1k` rate in `PRICE_TABLE`. |
| 3 | **early exit on confidence** | [`agent.py`](agent.py) `CostAwareRouter.route` — after a cheap tier answers and self-rates, if its confidence clears `EARLY_EXIT_CONFIDENCE` (0.75) the router **STOPS**; it never pays for the dearer tier it could have used. |
| 4 | **cost-per-decision analytics** | [`router.py`](router.py) `CostLedger` — a running record of **real** tokens + $ per query, the tier taken, and the early-exit / degrade / cache flags, plus a summary: total spend, avg cost/query, and **$ saved vs a measured always-escalate baseline**. |
| — | **cache-before-call** | [`agent.py`](agent.py) an exact-match cache returns a repeat query for **$0** — the cheapest answer is one you already have. |

## The route — the one decision that is never the model's

```
  query
    │
    ▼
  cache?  ──hit──►  return for $0
    │ miss
    ▼
  classify_complexity  (heuristic, no API call)  →  trivial | medium | hard
    │                                                 │        │       │
    ▼                                              direct     cot   escalate   (entry tier)
  budget step-down: start at the dearest tier ≤ entry the budget can afford
    │                       (can't afford even `direct`? → REFUSE)
    ▼
  run tier  (real NIM call; escalate = best-of-3, Python picks the most-confident)
    │
    ├── confidence ≥ 0.75 ?  ──yes──►  EARLY-EXIT — stop, don't pay to escalate
    │                          no
    ▼
  next tier costs too much for the budget ?  ──yes──►  DEGRADE — keep the cheap answer
    │                                          no
    └──────────────────►  escalate one step, repeat
    │
    ▼
  price the REAL tokens → append a row to the cost ledger
```

- **Trivial** = a factual lookup (`what is the capital of…`) → `direct`, and it early-exits at full confidence.
- **Hard** = `prove` / `design` / `justify` → `escalate` (best-of-3, priced like a frontier model) — *unless* the budget can't afford it, in which case it **degrades**.
- **The budget is a hard ceiling**, checked *before* the spend — a hard query on a tight budget does **not** get a blank cheque.

## Why the model never decides the spend

The actor is `meta/llama-3.1-8b-instruct` — fast and free on the NIM tier, but small. It
gets three jobs: **answer**, **reason**, and **rate its own confidence**. It does **not**
pick the tier, set the budget, decide the early-exit, or compute a price. That split is the
whole point: the confidence number is the model's *opinion*; the routing policy is
deterministic Python that turns that opinion into a spend decision — so the same query at
the same budget always earns the same route, and the ledger is reproducible.

> **Honest limitation from the recorded run:** on the "17 sheep, all but 9 run away" riddle
> the model answered **8** (correct is **9**) and self-rated it **1.00** — so the router
> early-exited on a *confidently wrong* cheap answer. Confidence-based early-exit trades a
> small quality risk for a large cost saving, and it is only as good as the model's
> self-calibration. It's left in, not hidden, because it's exactly the trade-off this
> pattern makes visible.

## Pricing — one warm model, a modelled cheap→frontier ladder

We route by **effort tier**, not by swapping model names: the free NIM tier serves one warm
model (the 70B/Nemotron models cold-start for >100s). The price gap is modelled the way real
cheap→frontier routing bills — each tier spends a different number of **real tokens** *and*
carries a different blended `$/1k` rate:

| tier | $/1k tokens | calls | what it does |
|------|-------------|-------|--------------|
| `direct` | 0.0005 | 1 | one terse call — a cheap/small model answers directly |
| `cot` | 0.0030 | 1 | one call that reasons step-by-step before answering |
| `escalate` | 0.0300 | 3 | best-of-3 reasoned samples, priced like a frontier model |

`escalate` is ~**60×** the `$/1k` of `direct` — which is what makes cheap routing worth the
risk. **Cost is priced from the real `usage.prompt_tokens` + `usage.completion_tokens`**
NVIDIA NIM returns; the deterministic chars/4 estimator is used *only* for the pre-flight
budget projection (you can't know a call's real token count until after you pay for it).

## What the demo shows ([`run.py`](run.py)) — one mixed batch

Six queries with per-task budgets exercise all four sub-points:

1. `capital of Japan?` — trivial · generous → **DIRECT**, confident → **early-exit** (cheap).
2. `why is the sky blue?` — medium · generous → **CoT**, confident → **early-exit** (skips escalate).
3. `17 sheep, all but 9…` — medium · generous → **CoT** reasoning step (and a real, confidently-wrong miss).
4. `prove √2 is irrational` — hard · generous → **ESCALATE** (best-of-3).
5. `design a rate limiter` — hard · **tight $** → wants escalate, budget blocks it → **DEGRADE** to CoT.
6. `capital of Japan?` (repeat) → **CACHE HIT**, $0.

```bash
# from the repo root, with .venv active and NVIDIA_API_KEY set in .env
python 07-cost-aware-router/run.py
```

After routing, the demo **measures the true always-escalate baseline** (it actually runs the
escalate tier on every query and prices the real tokens) and reports the **$ saved**. In the
recorded run: **`$0.039971` instead of `$0.185790` — 78.5% cheaper**, a measured comparison.

See [`recorded-run.md`](recorded-run.md) for the **real** captured transcript against NVIDIA
NIM — every call a live `HTTP/1.1 200 OK` to `integrate.api.nvidia.com`, all four sub-points
firing, and the full ledger.

## Files

- `router.py` — the deterministic half: the `Tier`/`PRICE_TABLE` ladder, `classify_complexity`
  (the heuristic), `Budget` (the ceiling + pre-flight estimator), and `CostLedger` (the JSONL
  ledger + summary + $-saved). **No model calls.**
- `agent.py` — the `CostAwareRouter`: the model-backed answer/reason/self-rate calls, the
  best-of-N escalate with a deterministic pick, the cache, and the classify → budget → run →
  early-exit / degrade route loop.
- `run.py` — the runnable mixed batch + the measured always-escalate baseline + the analytics.
- `recorded-run.md` — a real transcript hitting NVIDIA NIM (incl. the ledger JSONL).
- `ledger.jsonl` — runtime spend ledger written by `run.py` (gitignored; regenerated).

## Note on the model

Per query the model is asked only to **answer + reason + self-rate** (one call for
`direct`/`cot`, three for `escalate`). The **complexity score, the tier choice, the budget
verdict, the early-exit, the degrade, the best-of-N pick, and every price** are deterministic
Python. That is deliberate: the value of a cost-aware router is that *how much to spend* is a
policy you can read, test, and reproduce — not a number an 8B model talked itself into.
