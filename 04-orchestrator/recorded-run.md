# Recorded run — Multi-Tool Orchestrator

A **real** transcript captured by running:

```bash
python 04-orchestrator/run.py
```

- **Provider / endpoint:** NVIDIA NIM — `POST https://integrate.api.nvidia.com/v1/chat/completions` (every routing + synthesis call returned `HTTP/1.1 200 OK` in the live log below).
- **Model:** `meta/llama-3.1-8b-instruct` (warm on the free tier).
- **Date:** 2026-08-03.

The single run exercises **all five** sub-points: a dynamic tool registry,
capability-based routing, a permission **denial**, **parallel** execution with a
measured speed-up, and a **conflict** resolved by a documented policy. Everything
below is verbatim from the run.

---

## DEMO 1 — dynamic tool registry

The six tools self-registered via the `@tool(...)` decorator at import time; this
is the registry the router queries by capability.

```
name            | permission | capabilities            | description
--------------------------------------------------------------------------------------------
fx_convert      | read       | currency,math           | Convert an amount between currencies (USD/EUR/GBP/INR) at fixed table rates. Read-only. Args: amount (number), from (code), to (code).
ledger_write    | write      | persist                 | Append a one-line note to the audit ledger. Requires WRITE scope.
price_alpha     | network    | price_quote             | Price feed ALPHA. Returns the latest quote for a stock symbol.
price_beta      | network    | price_quote             | Price feed BETA. Returns the latest quote for a stock symbol.
price_gamma     | network    | price_quote             | Price feed GAMMA. Returns the latest quote for a stock symbol.
weather_lookup  | read       | weather                 | Look up today's weather for a city from a local table. Read-only.

advertised capabilities (the routing menu):
  - currency     -> fx_convert
  - math         -> fx_convert
  - persist      -> ledger_write
  - price_quote  -> price_alpha, price_beta, price_gamma
  - weather      -> weather_lookup
```

**Shows:** tools register themselves at runtime; the router is offered
*capabilities* (`price_quote`, `persist`, …), never tool names.

---

## DEMO 2 — capability routing + PARALLEL execution + CONFLICT resolution

> **Task:** What is the current reference price of AAPL? Consult the price feeds.
> **Granted:** `['network', 'read', 'write']`

Live log lines (real NIM calls + the orchestrator's own logging):

```
INFO    httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO    orchestrator  route via=llm caps=['price_quote'] -> tools=['price_alpha', 'price_beta', 'price_gamma']
WARNING orchestrator  CONFLICT on AAPL resolved by majority -> $150.25 (3 feeds queried; prices {alpha=$150.25, beta=$172.40, gamma=$150.25}. Policy 'majority' selected $150.25 (sources: alpha, gamma); disagreeing: beta.)
INFO    httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
```

Result:

```
Task     : What is the current reference price of AAPL? Consult the price feeds.
Granted  : ['network', 'read', 'write']
Route    : via=llm  capabilities=['price_quote']
           reason: The task requires a price quote, which is the smallest set of capabilities to satisfy the task.
           selected tools: ['price_alpha', 'price_beta', 'price_gamma']
Ran      :
           [ok ] price_alpha     0.60s  alpha: AAPL = $150.25 (trust=2, as_of=2026-08-03T09:30:00Z)
           [ok ] price_beta      0.60s  beta: AAPL = $172.40 (trust=1, as_of=2026-08-03T09:12:00Z)
           [ok ] price_gamma     0.60s  gamma: AAPL = $150.25 (trust=3, as_of=2026-08-03T09:28:00Z)
Timing   :
           serial   wall-clock : 1.80s
           parallel wall-clock : 0.61s
           saved 1.19s  (2.97x faster running concurrently)
Conflict :
           subject      : AAPL
           feeds        : alpha=$150.25(trust=2), beta=$172.40(trust=1), gamma=$150.25(trust=3)
           policy       : majority
           winner       : $150.25  from ['alpha', 'gamma']
           disagreeing  : ['beta']
           recorded     : 3 feeds queried; prices {alpha=$150.25, beta=$172.40, gamma=$150.25}. Policy 'majority' selected $150.25 (sources: alpha, gamma); disagreeing: beta.
ANSWER   : The current reference price of AAPL is $150.25, as determined by the majority vote of price feeds alpha and gamma, with beta disagreeing.
```

**Shows three sub-points at once:**

- **capability routing** — `route via=llm caps=['price_quote']`: the model picked
  the *capability*, and the orchestrator resolved it to the three feeds.
- **parallel execution** — each feed cost `0.60s`; run **serially** that is
  `1.80s`, run **concurrently** it is `0.61s` → **2.97× faster** (a real measured
  A/B over the same tool set).
- **conflict resolution** — `beta` disagreed ($172.40 vs $150.25); the documented
  `majority` policy selected **$150.25** and the conflict was **recorded** (note
  that beta is the *most-trusted* feed at trust=1 yet is correctly outvoted,
  because the policy is majority-first).

---

## DEMO 3 — permission scoping (deny-by-default)

Same task run twice: once **without** the `write` scope (denied), once **with**
it (succeeds) — proving the scope actually gates.

### 3a — RESTRICTED run → DENIED

> **Task:** Record an audit note in the ledger that AAPL was reviewed today.
> **Granted:** `['network', 'read']`  ← no `write`

```
INFO    httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO    orchestrator  route via=llm caps=['persist'] -> tools=['ledger_write']
WARNING orchestrator  DENIED ledger_write — permission 'write' not in granted scope ['network', 'read']
INFO    httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
```

```
Task     : Record an audit note in the ledger that AAPL was reviewed today.
Granted  : ['network', 'read']
Route    : via=llm  capabilities=['persist']
           reason: Only persist capability is required to record an audit note in the ledger.
           selected tools: ['ledger_write']
Denials  :
           DENIED ledger_write (needs 'write') — permission 'write' not in granted scope ['network', 'read']
Timing   : parallel wall-clock 0.00s
ANSWER   : The action could not be performed and the reason is that the permission 'write' is not in the granted scope.
```

### 3b — same task WITH `write` granted → SUCCEEDS

> **Granted:** `['network', 'read', 'write']`

```
INFO    httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO    orchestrator  route via=llm caps=['persist'] -> tools=['ledger_write']
INFO    httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
```

```
Task     : Record an audit note in the ledger that AAPL was reviewed today.
Granted  : ['network', 'read', 'write']
Route    : via=llm  capabilities=['persist']
           selected tools: ['ledger_write']
Ran      :
           [ok ] ledger_write    0.00s  wrote ledger entry #1: 2026-08-03T00:26:07 | AAPL reviewed 2026-08-03; reference price $150.25 (majority of 3 feeds).
ANSWER   : The audit note was recorded in the ledger: AAPL was reviewed today with a reference price of $150.25.

ledger contents now: ['2026-08-03T00:26:07 | AAPL reviewed 2026-08-03; reference price $150.25 (majority of 3 feeds).']
```

**Shows:** **deny-by-default permission scoping**. The routing was identical both
times (`persist → ledger_write`); the *only* difference was the granted scope. The
restricted run refused the write tool and logged the denial; the granted run wrote
the entry. The safety decision is pure Python — it never depends on the model.

---

## The five sub-points, together

| Sub-point | Evidence in this run |
|-----------|----------------------|
| **Dynamic tool registry** | DEMO 1 — six tools self-registered via `@tool`; capability menu printed. |
| **Capability-based routing** | `route via=llm caps=['price_quote']` / `caps=['persist']` — the model picked capabilities, the orchestrator resolved tools. |
| **Permission scoping** | DEMO 3a — `DENIED ledger_write — permission 'write' not in granted scope ['network','read']`; DEMO 3b writes once `write` is granted. |
| **Parallel execution** | DEMO 2 — serial `1.80s` vs parallel `0.61s`, **2.97× faster** over the same three feeds. |
| **Conflict resolution** | DEMO 2 — `beta` disagreed; `majority` policy selected `$150.25` and recorded the conflict. |

> Note: the model is small (8B) and not perfectly deterministic even at
> `temperature=0`, so a re-run may phrase the routing `reason`/final answer
> differently. The **capabilities** it selects, the **denial**, the **timing
> shape**, and the **conflict winner** are stable — routing is validated against
> the registry (with a keyword fallback logged as `via=fallback`), and permission
> enforcement + conflict resolution are deterministic Python, not model output.
