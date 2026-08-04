# Recorded run — Cost-Aware Agent Router

A **real** transcript captured by running the mixed batch:

```bash
python 07-cost-aware-router/run.py
```

- **Provider / endpoint:** NVIDIA NIM — `POST https://integrate.api.nvidia.com/v1/chat/completions` (every call below returned `HTTP/1.1 200 OK` in the live `httpx` log).
- **Model:** `meta/llama-3.1-8b-instruct` (warm on the free tier).
- **Date:** 2026-08-05.
- **Costs are REAL:** every `$` figure is priced from the **actual `usage.prompt_tokens` + `usage.completion_tokens`** NVIDIA NIM returned per call — not an estimator. (The deterministic chars/4 estimator is used only for the *pre-flight budget projection*, since you cannot know a call's real token count until after you pay for it.)
- **What the run proves:** a mixed batch routes **cheap queries to cheap tiers with early-exit**, a **hard** query to **best-of-3 escalate**, a hard query on a **tight budget degrades** instead of overspending, and a **repeat hits the cache for $0** — then the router's spend is compared against a **measured always-escalate baseline**.

The model only **answers + self-rates its confidence**. The complexity score, the tier choice, the budget ceiling, the early-exit, the degrade, the best-of-N pick, and the ledger are deterministic Python. Everything below is verbatim from the run.

---

## Price table (blended $/1k tokens — a cheap→frontier ladder on one warm model)

```
  tier      $/1k tokens   calls   what it does
  direct        0.0005       1   one terse call — a cheap/small model answers directly
  cot           0.0030       1   one call that reasons step-by-step before answering
  escalate      0.0300       3   best-of-3 reasoned samples, priced like a frontier model
```

We route by **effort tier**, not by swapping model names (the free NIM tier serves one warm model; the 70B/Nemotron models cold-start for >100s). The price gap is modelled the way real cheap→frontier routing bills: each tier spends a different number of **real tokens** *and* carries a different **$/1k rate**, so `escalate` is ~60× the $/1k of `direct`. Early-exit bar = **confidence ≥ 0.75**.

---

## The batch, query by query (real NIM calls, `HTTP/1.1 200 OK`)

### Query 1 — `What is the capital of Japan?` — trivial → **DIRECT → early-exit**

```
  ⚖️  complexity = TRIVIAL (score 0) → entry tier `direct`   signals: short factual lookup pattern
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
  ▶️  ran `direct` → 129 tokens · $0.000065 · self-confidence 1.00
      answer: Tokyo
  ✅ EARLY-EXIT at `direct` — confidence 1.00 ≥ 0.75; NOT paying to escalate.
```

One terse call, the cheapest tier, answered at full confidence → the router stopped. It never paid for `cot` or `escalate`.

### Query 2 — `Explain in two sentences why the sky is blue.` — medium → **CoT → early-exit**

```
  ⚖️  complexity = MEDIUM (score 2) → entry tier `cot`   signals: explanation/derivation keyword (explain)
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
  ▶️  ran `cot` → 217 tokens · $0.000651 · self-confidence 0.95
      answer: The sky appears blue due to Rayleigh scattering of sunlight by atmospheric gases.
  ✅ EARLY-EXIT at `cot` — confidence 0.95 ≥ 0.75; NOT paying to escalate.
```

The heuristic sent an *explain* query to the reasoning tier, which was confident enough to **skip the escalate tier entirely** — the single biggest saving in the batch comes from *not* escalating a query that didn't need it.

### Query 3 — `A farmer has 17 sheep and all but 9 run away…` — medium → **CoT → early-exit** (a real miss)

```
  ⚖️  complexity = MEDIUM (score 2) → entry tier `cot`   signals: explanation/derivation keyword (how); contains arithmetic / multiple numbers
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
  ▶️  ran `cot` → 228 tokens · $0.000684 · self-confidence 1.00
      answer: 8
  ✅ EARLY-EXIT at `cot` — confidence 1.00 ≥ 0.75; NOT paying to escalate.
```

> **Left in, honestly.** "all but 9 run away" means **9 remain** — the model answered **8** (`17 − 9`) and self-rated it **1.00**. This is the real failure mode of confidence-based early-exit: an 8B model that is *confidently wrong* clears the bar and the router trusts it. Early-exit trades a small quality risk for a large cost saving; it is only as good as the model's self-calibration, and here that calibration failed. The mechanism is honest — it did exactly what it was told — which is why the miss is worth showing rather than hiding.

### Query 4 — `Rigorously prove that √2 is irrational.` — hard → **ESCALATE (best-of-3)**

```
  ⚖️  complexity = HARD (score 3) → entry tier `escalate`   signals: hard-reasoning keyword (rigorously)
INFO  httpx  HTTP Request: POST .../chat/completions "HTTP/1.1 200 OK"
INFO  httpx  HTTP Request: POST .../chat/completions "HTTP/1.1 200 OK"
INFO  httpx  HTTP Request: POST .../chat/completions "HTTP/1.1 200 OK"
  ▶️  ran `escalate` (best-of-3) → 1196 tokens · $0.035880 · self-confidence 1.00
      answer: The square root of 2 is irrational.
```

`rigorously` / `prove` tripped the hard classifier, so the query went **straight to escalate**: three diverse reasoned samples (temperature 0.7), then a **deterministic Python pick** of the most-confident one. 1,196 real tokens at the frontier rate — this is the expensive tier, spent only where the query earns it.

### Query 5 — `Design a fault-tolerant distributed rate limiter…` — hard, **TIGHT budget → DEGRADE**

```
  ⚖️  complexity = HARD (score 3) → entry tier `escalate`   signals: hard-reasoning keyword (design)
  💸 BUDGET: `escalate` would cost ~$0.03357 but cap is $0.0040 → DEGRADE, start at `cot` instead
INFO  httpx  HTTP Request: POST .../chat/completions "HTTP/1.1 200 OK"
  ▶️  ran `cot` → 897 tokens · $0.002691 · self-confidence 0.90
      answer: Token Bucket
```

The classifier wanted **escalate**, but the **pre-flight budget projection** (~$0.03357) exceeded the query's tight **$0.0040** ceiling. So the router **refused to overspend** and **degraded** to the dearest tier it *could* afford (`cot`, ~$0.0027 real) — the answer is flagged `DEGRADED` in the ledger. This is the token-budget ceiling doing its job: a hard query does **not** get a blank cheque.

### Query 6 — `What is the capital of Japan?` (repeat) — **CACHE HIT, $0**

```
  ♻️  CACHE HIT — identical query already answered; returning it for $0.
```

Cache-before-call: the exact query from #1 was already answered, so it came back for **zero tokens and zero dollars** — the literal cheapest way to solve a task is to not call the model at all.

---

## Baseline — the true always-escalate cost (measured, not modelled)

For an honest "$ saved vs always-escalate", the run **actually escalates every query** and prices its **real** tokens (reusing the router's own escalate call for query 4, which already paid for one):

```
  #1 What is the capital of Japan?                 always-escalate $0.014070  vs actual $0.000065  → saved $0.014005
  #2 Explain in two sentences why the sky is blu…  always-escalate $0.019440  vs actual $0.000651  → saved $0.018789
  #3 A farmer has 17 sheep and all but 9 run awa…  always-escalate $0.016470  vs actual $0.000684  → saved $0.015786
  #4 Rigorously prove that the square root of 2 …  always-escalate $0.035880  vs actual $0.035880  → saved $0.000000
  #5 Design a fault-tolerant distributed rate li…  always-escalate $0.085560  vs actual $0.002691  → saved $0.082869
  #6 What is the capital of Japan?                 always-escalate $0.014370  vs actual $0.000000  → saved $0.014370
```

Query 4 saves **nothing** — a genuinely hard query costs the same whether you route to it or always escalate; the savings come entirely from the cheap-routed, early-exited, degraded, and cached queries.

---

## Cost-per-decision ledger (sub-point 4)

```
 #  query                               cmplx    route              tokens     cost $   conf  flags
---------------------------------------------------------------------------------------------------
 1  What is the capital of Japan?       trivial  direct                129   0.000065   1.00  early-exit
 2  Explain in two sentences why th…    medium   cot                   217   0.000651   0.95  early-exit
 3  A farmer has 17 sheep and all b…    medium   cot                   228   0.000684   1.00  early-exit
 4  Rigorously prove that the squar…    hard     escalate             1196   0.035880   1.00
 5  Design a fault-tolerant distrib…    hard     cot                   897   0.002691   0.90  DEGRADED
 6  What is the capital of Japan?       —        cache                   0   0.000000   1.00  CACHE $0
---------------------------------------------------------------------------------------------------
    TOTAL                                                             2667   0.039971
```

## Spend summary — what routing SAVED

```
  queries              : 6
  total tokens (real)  : 2667
  total spend          : $0.039971
  avg cost / query     : $0.006662
  always-escalate base : $0.185790  (measured, not modelled)
  SAVED by routing     : $0.145819  (78.5% cheaper)
  early-exits          : 3   degraded: 1   cache hits: 1   refused: 0
```

**Routing solved the same six tasks for `$0.039971` instead of `$0.185790` — 78.5% cheaper — a measured comparison, both numbers priced from real NIM token usage.**

---

## The four sub-points, together

| Sub-point | Evidence in this run |
|-----------|----------------------|
| **Token budgeting per task** | Every query carried a hard `$` ceiling. Query 5's tight **$0.0040** budget made the pre-flight projection reject `escalate` (~$0.0336) and **degrade** to `cot` — the ceiling is enforced *before* the spend, so it can never be breached. |
| **Route by complexity + cost** | A no-API heuristic scored each query trivial / medium / hard from surface signals and mapped it to `direct` / `cot` / `escalate`, each with a published `$/1k` rate — trivial lookups went cheap, `prove`/`design` went to best-of-3. |
| **Early exit on confidence** | 3 of 6 queries **stopped at the cheap tier** because the model's self-rated confidence cleared 0.75 — the router never paid to escalate #1, #2, or #3 (and #6 skipped the model entirely via cache). |
| **Cost-per-decision analytics** | The ledger prices every query from **real `usage` tokens**, tags the tier / early-exit / degrade / cache, and reports **$0.145819 (78.5%) saved** vs a **measured** always-escalate baseline. |

> Note: the model is small (8B) and not perfectly deterministic even at temperature 0, so a re-run may phrase answers, confidences, or exact token counts slightly differently. What is **stable**: the classifier, the tier ladder, the budget gate, the early-exit threshold, and the best-of-N pick are pure Python — the same query at the same budget always earns the same *routing decision*, and a tight budget *always* degrades rather than overspends.

## The full ledger (durable JSONL — `ledger.jsonl`, gitignored / regenerated)

```json
{"idx": 1, "query": "What is the capital of Japan?", "complexity": "trivial", "budget": 0.05, "entry_tier": "direct", "final_tier": "direct", "tiers_run": ["direct"], "prompt_tokens": 113, "completion_tokens": 16, "cost": 6.5e-05, "confidence": 1.0, "early_exit": true, "degraded": false, "cached": false, "refused": false, "answer": "Tokyo", "always_escalate_cost": 0.01407, "saved": 0.014005}
{"idx": 2, "query": "Explain in two sentences why the sky is blue.", "complexity": "medium", "budget": 0.05, "entry_tier": "cot", "final_tier": "cot", "tiers_run": ["cot"], "prompt_tokens": 119, "completion_tokens": 98, "cost": 0.000651, "confidence": 0.95, "early_exit": true, "degraded": false, "cached": false, "refused": false, "answer": "The sky appears blue due to Rayleigh scattering of sunlight by atmospheric gases.", "always_escalate_cost": 0.01944, "saved": 0.018789}
{"idx": 3, "query": "A farmer has 17 sheep and all but 9 run away. How many sheep are left?", "complexity": "medium", "budget": 0.05, "entry_tier": "cot", "final_tier": "cot", "tiers_run": ["cot"], "prompt_tokens": 128, "completion_tokens": 100, "cost": 0.000684, "confidence": 1.0, "early_exit": true, "degraded": false, "cached": false, "refused": false, "answer": "8", "always_escalate_cost": 0.01647, "saved": 0.015786}
{"idx": 4, "query": "Rigorously prove that the square root of 2 is irrational.", "complexity": "hard", "budget": 0.1, "entry_tier": "escalate", "final_tier": "escalate", "tiers_run": ["escalate"], "prompt_tokens": 366, "completion_tokens": 830, "cost": 0.03588, "confidence": 1.0, "early_exit": false, "degraded": false, "cached": false, "refused": false, "answer": "The square root of 2 is irrational.", "always_escalate_cost": 0.03588, "saved": 0.0}
{"idx": 5, "query": "Design a fault-tolerant distributed rate limiter and justify token-bucket vs sliding-window.", "complexity": "hard", "budget": 0.004, "entry_tier": "escalate", "final_tier": "cot", "tiers_run": ["cot"], "prompt_tokens": 127, "completion_tokens": 770, "cost": 0.002691, "confidence": 0.9, "early_exit": false, "degraded": true, "cached": false, "refused": false, "answer": "Token Bucket", "always_escalate_cost": 0.08556, "saved": 0.082869}
{"idx": 6, "query": "What is the capital of Japan?", "complexity": "—", "budget": 0.05, "entry_tier": "cache", "final_tier": "cache", "tiers_run": [], "prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "confidence": 1.0, "early_exit": false, "degraded": false, "cached": true, "refused": false, "answer": "Tokyo", "always_escalate_cost": 0.01437, "saved": 0.01437}
```
