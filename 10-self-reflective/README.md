# 10 — Self-Reflective Agent (auto-eval)

**Goal:** an agent that **grades and improves its own work** — instead of returning a first
draft, it produces an answer, **judges it against an explicit rubric** (LLM-as-judge on its
*own* output), and if it misses a **quality gate** it **regenerates under the critique** and
re-scores, looping — bounded — until it passes or hits an iteration cap. The interesting part
isn't "ask the model to try again"; it's the structure around it: an **explicit rubric**
(named criteria scored 1-5, not vibes), a **critic that returns actionable reasoning** (what's
wrong, what's missing, how to fix), **regeneration under constraints** (keep the good, fix the
named gaps), and a **gate that is deterministic Python** — including **hard checks** the model
cannot flatter its way past.

The model does exactly three things: **draft, judge, refine**. **Everything that turns those
judgements into a verdict is deterministic Python** — the score aggregation, the hard checks,
the pass/fail gate, the loop control, the "what changed" diff, and the improvement metrics.
Same per-criterion scores in → same gate + same metrics out, no matter how the 8B model
phrases its prose.

## The four ideas (hand-rolled, no framework)

| # | Sub-point | Where it lives |
|---|-----------|----------------|
| 1 | **execute + evaluate via LLM-as-judge** | [`agent.py`](agent.py) `Drafter` produces a fast draft; [`Judge`](agent.py) scores it against the **explicit rubric** in [`rubric.py`](rubric.py) — each criterion an integer **1-5**, with required **evidence** and a **number per criterion**. This is LLM-as-judge on the agent's *own* output. |
| 2 | **critic reasoning** | [`agent.py`](agent.py) `Judge` returns, per criterion, **specific actionable critique** — what is wrong, what is missing, and *how to fix it* — not just a score. That critique is exactly what the refinement round consumes. |
| 3 | **regenerate with constraints** | [`agent.py`](agent.py) `Refiner.refine` regenerates the answer **keeping what scored well and fixing the named gaps** (plus any failed hard check), **bounded** by `max_iters`. The loop stops on the deterministic quality gate. |
| 4 | **log improvement metrics** | [`agent.py`](agent.py) `ReflectionLoop` records the **per-iteration score**, **which criteria moved** (measured from the scores, not the model's word), the **hard-check trail**, the final **delta**, and the **stop reason** (`passed` / `max-iters`). |

## The loop — the one thing that is never the model's

```
                      the model does these 3 (soft, judgement-heavy)
                      ───────────────────────────────────────────────
   task ─►  DRAFT ──►  JUDGE (score vs rubric 1-5 + evidence + critique)
              ▲          │
              │          ▼
              │     ┌───────────────────────────────────────────┐
              │     │  DETERMINISTIC PYTHON (rubric.py)           │
              │     │   aggregate 1-5 → overall 0-1               │
              │     │   run hard checks (non-LLM, objective)      │
              │     │   gate: overall ≥ threshold AND all hard ok │
              │     └───────────────────────────────────────────┘
              │          │
              │     passed? ──yes──►  STOP (reason: passed)
              │          │no
        REFINE ◄─────────┘   (keep the good, fix the named gaps; bounded by max_iters)
              │
              └─ hit max_iters without passing ──►  STOP (reason: max-iters)
```

- **The gate is a property of quality, not of the model's mood.** A judge that could pass its
  own work would say "great" every time. Here the pass decision is `overall ≥ threshold AND
  every hard check passes` — pure Python, so a genuinely incomplete answer *cannot* clear the
  bar just because the model liked it.
- **Hard checks anchor the soft score.** Alongside the 1-5 rubric, each task ships a few
  **deterministic, non-LLM checks** on the answer text (does the docstring actually name
  `Raises`? does the answer actually contain `28`?). They co-gate the pass decision, so an
  over-generous judge can't wave through an answer that is objectively missing something.
- **Improvement is measured, not claimed.** `_score_diff` compares each criterion's score
  before vs after a refinement, so "the edge-cases criterion improved" is read off the scores —
  the model's own "here's what I changed" note is kept only as color.

## Mitigating self-judge bias (stated plainly)

The judge is the **same model family grading its own output**, so it can share the drafter's
blind spots and drift optimistic. This is a real limitation, not a solved problem. Four things
reduce it (they do **not** eliminate it):

1. **A fixed, explicit rubric** — the judge scores named criteria with per-score definitions,
   not a vague "is this good?".
2. **Required evidence per score** — each score must cite a quote/observation from the answer,
   which makes a lazy rubber-stamp harder (and, when the judge *does* over-credit, visible).
3. **A fresh, stateless judge call** framed as an independent grader with no memory of writing
   the draft.
4. **Deterministic hard checks co-gate the decision** — the objective floor the model cannot
   talk past. This is the backstop, and in the recorded run it is what actually catches an
   over-generous judge (see below).

## What the demo shows ([`run.py`](run.py)) — two tasks, both with a mediocre first draft

`run.py` runs the reflection loop on **two** tasks chosen so the fast first draft has real room
to improve:

1. **`docstring`** — write a precise docstring (with edge cases) for a remainder-splitting
   function. The quick draft documents the params and return but **omits `Raises` and the
   zero/negative-people edge case**. Expectation: reflection **adds them and the score climbs**.
2. **`discount`** — answer a tricky question completely: two successive discounts (20% then
   10%) off the original. The additive **"30%" trap** vs the correct **28%**. Expectation:
   reflection completes the reasoning.

```bash
# from the repo root, with .venv active and NVIDIA_API_KEY set in .env
python 10-self-reflective/run.py
```

See [`recorded-run.md`](recorded-run.md) for the **real** captured transcript against NVIDIA
NIM. Two honest outcomes:

- **`docstring`** — a genuine climb: **0.56 → 1.00** (Δ +0.44), with `edge_cases` 1→5 and
  `raises` 1→5, and the `documents_raises` hard check flipping ❌ → ✅.
- **`discount`** — an **honest non-improvement of the LLM score**: the draft was already
  numerically correct (28%), so the over-generous 8B judge scored it a flat **1.00 → 1.00**
  (it even rated *"addresses the 30% trap"* 5/5 on a draft that never mentions 30%). The
  **deterministic hard check caught the missing explanation** and held the gate at FAIL; the
  refinement added *"this is not 30% because discounts compound…"* and the hard-check trail rose
  **2/3 → 3/3**, flipping the gate to PASS. **Reflection isn't magic, and the LLM score isn't
  enough — the hard checks are why the gate is trustworthy.**

## Files

- `rubric.py` — the deterministic scoring core: `Criterion` / `Rubric` / `HardCheck`, the
  `aggregate` (1-5 → 0-1 weighted mean), `run_hard_checks`, and the `gate`, plus the two
  concrete `Task`s and their rubrics. **No model calls.**
- `agent.py` — the model-backed roles (`Drafter`, `Judge`, `Refiner`) + the shared strict-JSON
  call, and the `ReflectionLoop` that runs generate → (reflect → refine)* with the loop
  control, the score diff, and the improvement metrics. **The only place a model call happens.**
- `run.py` — the runnable demo: two tasks, a live phase-by-phase transcript, an improvement-
  metrics table, and a compact JSONL log of every event.
- `recorded-run.md` — a real transcript hitting NVIDIA NIM (both tasks, incl. the score trail,
  the hard-check trail, and the stop reasons).
- `reflection-log.jsonl` — runtime transcript written by `run.py` (gitignored; regenerated each run).

## Note on the model

Per role the model does exactly one thing — **draft** a fast answer, **judge** an answer
against the rubric, or **refine** it under the critique — and returns strict JSON. It does
**not** aggregate the scores, run the hard checks, decide the pass/fail, count the iterations,
or compute the deltas. That split is the point: the trustworthy part of self-improvement —
*is this actually good enough yet?* — is deterministic Python you can read and test, not a
number an 8B model asserts about its own work. The 8B judge is measurably **over-generous**
(it saturates the rubric readily); the hard checks + the gate are precisely the guardrail that
keeps a self-grading agent honest. A malformed model reply degrades to a minimal parsed object
(a refiner that returns nothing keeps the previous answer); it never crashes the loop or
corrupts the metrics.
