# 09 — Multi-Agent Debate System

**Goal:** get a **better answer out of several agents arguing** than any one agent would give
alone. Three **proposers** with different personas answer the same question independently, a
**critic** pressure-tests every answer, each proposer then **rebuts** — revising or defending
after reading the critique — and a **judge** synthesizes the winning answer. The interesting
part isn't "ask an LLM three times"; it's the structure around it: **perspective diversity**
(distinct personas so answers genuinely diverge), **structured critique** (a round that can
change a mind), **deterministic consensus** (each agent's final stance is a ballot Python
tallies), and a **confidence that is derived from agreement**, not asserted by the judge.

The model does exactly four things: **propose, critique, rebut, synthesize**. **Everything
that turns arguments into a verdict is deterministic Python** — the round order, the ballot
tally, the consensus ratio, the mind-change detection, and the confidence derivation. Same
final stances in → same tally + confidence out, no matter how the 8B model phrases its prose.

## The four ideas (hand-rolled, no framework)

| # | Sub-point | Where it lives |
|---|-----------|----------------|
| 1 | **agents propose** | [`agents.py`](agents.py) `Proposer` + `PERSONAS` — three proposers (cautious 🛡️ / creative 💡 / literal 📏) each get a **distinct system prompt** and answer the same question **cold**, at higher temperature, so the proposals genuinely diverge instead of echoing one another. |
| 2 | **critic evaluates** | [`agents.py`](agents.py) `Critic` reads **all** proposals at once and names each one's main flaw / hidden assumption / unsupported leap. Then a **rebuttal** round ([`Proposer.rebut`](agents.py)) lets each proposer revise or defend after reading the critique + the others — this is where a good critique can change a mind. |
| 3 | **voting / consensus** | [`debate.py`](debate.py) — each agent's **final stance is its ballot**. `normalize_stance` canonicalizes it (so `"$0.05"`, `"0.05"`, `".05"` tally together), `tally` counts positions, `consensus_ratio` = winner ÷ total. **Pure Python, reproducible from the stances alone.** |
| 4 | **aggregator synthesizes with confidence** | [`agents.py`](agents.py) `Judge` combines the best points into one answer; [`debate.py`](debate.py) `derive_confidence` sets the confidence from the **agreement math** — unanimous → **HIGH**, majority → **MEDIUM**, split/tie → **LOW (flagged)**. The judge does **not** get to invent the confidence. |

## The pipeline — the one thing that is never the model's

```
  three personas                          the debate (round order = deterministic Python)
  ──────────────                          ──────────────────────────────────────────────
  🛡️ cautious ─┐
  💡 creative ─┼─►  1. PROPOSE (cold, independent) ──►  three diverging proposals
  📏 literal  ─┘                                              │
                                                              ▼
                        2. CRITIQUE  ── one critic names each proposal's flaw
                                                              │
                                                              ▼
                        3. REBUT     ── each persona revises OR defends  ◄── can change a mind
                                                              │
                     ┌────────────────────────────────────────┴───────────────┐
                     ▼                                                          ▼
         4. TALLY final stances (Python)                         5. JUDGE synthesizes
            normalize → count → consensus_ratio                     the best points → ONE answer
                     │                                                          │
                     ▼                                                          │
         derive_confidence(ratio):  ◄─────────── attached to ──────────────────┘
            all agree      → HIGH
            majority       → MEDIUM
            split / tie    → LOW (flagged)
```

- **Confidence is a property of agreement, not a vibe.** A judge that could set its own
  confidence would just say "high" every time. Here it falls out of how many agents actually
  landed on the same position — so a genuine split is *visibly* low-confidence.
- **The stance is the ballot.** Each agent's final stance is normalized to a canonical key and
  counted. Agents that mean the same thing tally together; genuinely different positions stay
  distinct — so "consensus" measures agreement on the **substance**, not on wording.
- **A mind-change is measured, not claimed.** `detect_mind_changes` compares each agent's
  canonical stance before vs after the rebuttal, so "the critique changed a mind" is read off
  the stances, not trusted from the model's self-report.

## Why a debate beats a single call (the bug this pattern avoids)

A single LLM call gives you one framing, one blind spot, and a confident tone whether or not
the answer is actually contested. That's two failure modes at once:

- **False confidence on a split question** → one call happily commits to "bootstrap" (or
  "microservices", or "yes") and *sounds* just as sure as it does on `2 + 2`. A debate surfaces
  that reasonable agents disagree and **flags it low-confidence**.
- **A blind spot no one challenges** → one call never gets its assumption named. The critic +
  rebuttal round gives a wrong or shaky proposal a chance to be corrected *before* the judge
  ever sees it.

So the panel is deliberately diverse (three personas), deliberately adversarial (a critic),
and deliberately honest about disagreement (confidence from consensus). The model argues; the
Python keeps score.

## What the demo shows ([`run.py`](run.py)) — two questions, both ends of the behaviour

`run.py` runs the debate on **two** questions chosen to show convergence *and* genuine
disagreement:

1. **`clear`** — *"A bat and a ball cost \$1.10; the bat costs \$1.00 more than the ball. How
   much is the ball?"* A question with a **definite answer** (\$0.05) and a famous intuitive
   trap (\$0.10). Expectation: the panel **converges → HIGH confidence**.
2. **`debatable`** — *"Should a founder bootstrap, or raise venture capital?"* A genuine
   values/strategy trade-off with **no single right answer**. Expectation: the panel **splits →
   LOW confidence, flagged**, and the critique moves at least one mind.

```bash
# from the repo root, with .venv active and NVIDIA_API_KEY set in .env
python 09-multi-agent-debate/run.py
```

See [`recorded-run.md`](recorded-run.md) for the **real** captured transcript against NVIDIA
NIM — every proposal, the critique, the rebuttals (with two genuine mind-changes on the
debatable question), the deterministic tally, and the judge's synthesis with its
agreement-derived confidence (`HIGH 100%` on the clear question, `LOW 33%` on the debatable one).

## Files

- `agents.py` — the model-backed roles: the three `PERSONAS`, the `Proposer` (propose + rebut),
  the `Critic`, and the `Judge`, plus the shared strict-JSON model call. **The only place a
  model call happens.**
- `debate.py` — the deterministic core: `normalize_stance`, `tally`, `consensus_ratio`,
  `derive_confidence`, `detect_mind_changes`, and the `Debate` orchestrator that runs the five
  phases in order. **No model calls.**
- `run.py` — the runnable demo: two questions, a live phase-by-phase transcript, a side-by-side
  comparison, and a compact JSONL log of every event.
- `recorded-run.md` — a real transcript hitting NVIDIA NIM (both debates, incl. the tally +
  confidence).
- `debate-log.jsonl` — runtime transcript written by `run.py` (gitignored; regenerated each run).

## Note on the model

Per role the model does exactly one thing — **propose** an answer from one persona, **critique**
the proposals, **rebut** after reading the critique, or **synthesize** the final answer — and
returns strict JSON. It does **not** decide the winner, count the ballots, compute the consensus
ratio, or set the confidence. That split is the point: the trustworthy part of a debate — *how
much do the agents actually agree, and how sure should we therefore be?* — is deterministic
Python you can read and test, not a number an 8B model asserts. A malformed model reply degrades
to a minimal parsed object; it never crashes the debate or corrupts the tally.
