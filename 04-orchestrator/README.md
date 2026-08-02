# 04 — Multi-Tool Orchestrator

**Goal:** route a task across *many* tools **safely** and **in parallel** — pick
tools by what they can *do* (not their names), refuse tools a run isn't permitted
to use, run the independent ones concurrently, and resolve conflicting results
with a documented policy.

## The five ideas (hand-rolled, no framework)

| # | Sub-point | Where it lives |
|---|-----------|----------------|
| 1 | **dynamic tool registry** | [`registry.py`](registry.py) — the `@tool(...)` decorator. A tool self-registers (name, description, **capabilities** set, **permission**, JSON arg schema) the moment its module is imported. The router reads the registry only through *capability queries*, so a new tool becomes routable **without touching the router**. |
| 2 | **capability-based routing** | `Orchestrator.route` in [`agent.py`](agent.py) — the model is shown the registry's *advertised capabilities* (`price_quote`, `weather`, `persist`, …), **not** tool names, and picks the capabilities a task needs; the orchestrator resolves those to concrete tools (`registry.tools_for_capabilities`). |
| 3 | **permission scoping** | `Orchestrator._enforce` — **deny-by-default**. Each tool declares one scope (`read` / `write` / `network`); a tool runs only if its scope is in the run's *granted* set, else it is **refused and logged**. |
| 4 | **parallel execution** | `Orchestrator.invoke_batch(mode="parallel")` — independent tools run **concurrently** in a `ThreadPoolExecutor`. `execute(compare=True)` times **serial vs parallel** over the same tool set to show the wall-clock saving. |
| 5 | **conflict resolution** | `Orchestrator.resolve_price_conflict` — when the price feeds disagree, a **documented deterministic policy** (`majority → trust-priority → freshness`) picks a winner and **records the conflict** (all feeds, who disagreed, which policy step decided). |

## The six tools ([`tools.py`](tools.py))

| tool | capabilities | permission | note |
|------|--------------|-----------|------|
| `price_alpha` | `price_quote` | `network` | price feed; AAPL = **$150.25** (trust 2) |
| `price_beta` | `price_quote` | `network` | price feed; AAPL = **$172.40** (trust 1, **stalest**) — the outlier |
| `price_gamma` | `price_quote` | `network` | price feed; AAPL = **$150.25** (trust 3) |
| `weather_lookup` | `weather` | `read` | read-only city lookup |
| `fx_convert` | `currency`, `math` | `read` | read-only currency conversion |
| `ledger_write` | `persist` | **`write`** | the one write tool — a restricted run is **denied** it |

Three feeds (not two) so **majority** voting is meaningful; each also carries a
`trust_priority` and an `as_of` timestamp so the tie-break steps are fully
specified. On AAPL, `alpha` and `gamma` agree and `beta` disagrees — a forced
conflict the resolver must settle.

## Conflict policy (deterministic, documented)

1. **Majority** — the modal price wins.
2. Tie on vote count → **trust-priority** — lowest `trust_priority` number (most trusted) wins.
3. Still tied → **freshness** — latest `as_of` wins.

> Note the deliberate design: `beta` is the *most-trusted* feed (trust 1) but is
> outvoted 2-to-1, so **majority-first** picks **$150.25** and records that beta
> disagreed at $172.40. Swapping the policy order would change the winner — which
> is exactly why the policy is written down rather than implicit.

## What the demo shows ([`run.py`](run.py))

- **DEMO 1 — dynamic registry:** dumps the six self-registered tools + the
  capability menu the router uses.
- **DEMO 2 — routing + parallel + conflict (full grant):** "price of AAPL?" →
  the LLM routes to capability `price_quote` → three feeds run **concurrently**
  (**1.80s serial → 0.61s parallel, ~3× faster**) → `beta` disagrees → the
  `majority` policy resolves it and records the conflict → the model synthesises
  the answer (**$150.25**).
- **DEMO 3 — permission scoping:** a **restricted** run (granted `read`+`network`,
  **not** `write`) is asked to write to the ledger → routing lands on
  `ledger_write` → **DENIED** and logged. The *same* task under a full grant then
  **succeeds** — proving the scope actually gates.

## Run it

```bash
# from the repo root, with .venv active and NVIDIA_API_KEY set in .env
python 04-orchestrator/run.py
```

See [`recorded-run.md`](recorded-run.md) for a real captured transcript against
NVIDIA NIM (`meta/llama-3.1-8b-instruct`) — every routing/synthesis call is a live
`HTTP/1.1 200 OK` to `integrate.api.nvidia.com`.

## Files

- `registry.py` — the dynamic registry: `@tool` decorator, `Tool`/`ToolResult`, and the capability-query views the router uses.
- `tools.py` — the six pure-Python tools (self-registering) with varied capabilities + permissions.
- `agent.py` — the `Orchestrator`: capability routing (LLM), permission enforcement, parallel batch execution, conflict resolution, answer synthesis.
- `run.py` — a runnable demo exercising all five sub-points.
- `recorded-run.md` — a real transcript hitting NVIDIA NIM.

## Note on the model

`meta/llama-3.1-8b-instruct` does the **routing** (capability selection) and the
final-answer **synthesis** — both real NIM calls. Routing is validated against the
registry (hallucinated capabilities are dropped) with a **keyword fallback** so a
small model can never wedge the pipeline; each run states whether it routed
`via=llm` or `via=fallback`. **Permission enforcement and conflict resolution are
pure Python** — the safety-critical decisions are deterministic, not left to the
model.
