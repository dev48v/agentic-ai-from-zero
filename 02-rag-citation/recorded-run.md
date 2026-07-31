# Recorded run — RAG Agent (citation grounding)

A **real** captured run against **NVIDIA NIM** (no mock, no replay). It shows all
three behaviours: a grounded answer with inline `[n]` citations, a low-confidence
flag that refuses to hallucinate, and a fallback to (simulated) web search when
the corpus has nothing relevant.

| | |
|---|---|
| **Date** | 2026-08-01 03:34 IST |
| **Provider** | NVIDIA NIM — `https://integrate.api.nvidia.com/v1` (OpenAI-compatible) |
| **Model** | `meta/llama-3.1-8b-instruct` |
| **Temperature** | `0` (deterministic) |
| **Retriever** | local TF-IDF + cosine, offline/deterministic (25 passages from `corpus/`) |
| **Thresholds** | fallback `< 0.15`, confident `>= 0.30` (top-1 cosine) |
| **LLM calls** | 3 (one per case) — retrieval itself is local, no network |

**Evidence it hit NIM** — each of the three cases made exactly one live call that
returned `HTTP/1.1 200 OK` (httpx log). Case 3 short-circuits to the fallback
*before* the grounded call, so its single call is the fallback answerer:

```
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"   # case 1 grounded
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"   # case 2 grounded (model: unsupported)
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"   # case 3 fallback
```

Run header:

```
model             : meta/llama-3.1-8b-instruct
corpus            : 25 passages from .../02-rag-citation/corpus
fallback threshold: 0.15  (top<this => web search)
confident thresh. : 0.3   (top>=this + grounded => GROUNDED)
```

---

## CASE 1 — well-supported → GROUNDED with citations

**Q:** *What uptime SLA does the Nimbus Enterprise plan guarantee and what is its
monthly price per seat?*

Retrieved passages (cosine similarity, highest first):

```
[1] 0.6180  sla.md       :: The signed uptime SLA is available only on the Enterprise plan. …
[2] 0.3857  pricing.md   :: The Enterprise plan costs 99 US dollars per seat each month, has…
[3] 0.2057  sla.md       :: If monthly uptime falls below 99.95 percent, Enterprise customer…
```

Decision: `top_score = 0.6180 >= 0.30` **and** the model reported `supported=true`
→ **`GROUNDED ✓`**.

Answer (note the inline `[1]`/`[2]` markers — real model output):

> The Nimbus Enterprise plan guarantees 99.95 percent uptime **[1]** and costs 99
> dollars per seat each month **[2]**.

Sources (each marker mapped back to its passage + score):

```
[1] sla.md — Nimbus Cloud — Uptime SLA and Support  (score 0.618)
    The signed uptime SLA is available only on the Enterprise plan. Nimbus
    guarantees 99.95 percent monthly uptime for Enterprise customers.
[2] pricing.md — Nimbus Cloud — Plans and Pricing  (score 0.3857)
    The Enterprise plan costs 99 US dollars per seat each month, has no seat cap,
    includes 5 TB of object storage, and adds a dedicated account manager, audit
    logging, and a signed uptime SLA.
```

Both cited facts (`99.95%`, `$99/seat/month`) trace exactly to the cited passages.
`elapsed 1.59s`.

---

## CASE 2 — weakly supported → LOW-CONFIDENCE (no hallucination)

**Q:** *Can I pay for my Nimbus subscription using cryptocurrency such as Bitcoin
or Ethereum?*

Retrieved passages:

```
[1] 0.2185  security.md  :: Single sign-on using SAML 2.0 is included on the Team and Enterp…
[2] 0.1028  pricing.md   :: Nimbus Cloud is billed per seat, per month, with no setup fee. T…
[3] 0.0879  sla.md       :: The signed uptime SLA is available only on the Enterprise plan. …
```

The top hit (0.2185) clears the fallback floor (so we *do* try to ground), but
it is only a lexical near-miss — none of the passages mention payment methods.
The model correctly reported `supported=false`.

Decision: model says it cannot support an answer (and `0.2185 < 0.30`) →
**`LOW-CONFIDENCE ⚠`**.

```
top_score      : 0.2185
CONFIDENCE     : LOW-CONFIDENCE ⚠
reason         : model reported it could not fully support an answer from the retrieved passages (top score 0.2185)
model grounded : False
```

Answer — the agent flags it instead of inventing a payment policy:

> There is not enough information to answer this question.

This is the whole point of sub-point 3: a plausible-but-unsupported question gets
**flagged**, not fabricated. `elapsed 1.21s`.

---

## CASE 3 — outside the corpus → FALLBACK to search

**Q:** *What are the common symptoms of a vitamin D deficiency in adults?*

Retrieved passages — nothing relevant; every score is `0.0000`:

```
[1] 0.0000  limits.md    :: The Nimbus REST API is rate limited per API key. The Starter pla…
[2] 0.0000  limits.md    :: When a key exceeds its limit the API returns HTTP 429 with a Ret…
[3] 0.0000  limits.md    :: A single API request body may be at most 10 MB. Individual objec…
```

Decision: `top_score = 0.0000 < 0.15` → retrieval too weak → **`FALLBACK ↪`**.
The agent short-circuits *before* the grounded call and routes to the
clearly-labeled web-search fallback:

```
INFO  rag_citation_agent  FALLBACK: top retrieval similarity 0.0000 < fallback threshold 0.15 -> corpus has no relevant passage
```

Answer (real model output — note the mandatory disclaimer banner, and that the
**sources list is empty** because nothing in the trusted corpus supports it):

> **[FALLBACK - general web knowledge, NOT from the Nimbus knowledge base; verify independently.]**
>
> Common symptoms of a vitamin D deficiency in adults may include:
> 1. Fatigue and weakness
> 2. Muscle pain or cramps
> 3. Bone pain or tenderness
> 4. Osteoporosis or osteopenia
> 5. Weakened immune system
> 6. Mood changes, such as depression or seasonal affective disorder
> 7. Hair loss
> 8. Skin problems, such as acne or eczema
> 9. Increased risk of falls and fractures
> 10. Difficulty healing from injuries or wounds
>
> It's essential to note that some people may not exhibit any noticeable symptoms…

```
SOURCES:
  (none — no trusted corpus passage supports this answer)
```

`elapsed 2.92s`.

---

## What this run demonstrates (the 4 sub-points)

1. **Retrieve context** — a local TF-IDF + cosine retriever surfaced the top-3
   passages with real similarity scores for every query.
2. **Generate answers with sources** — Case 1's answer carries inline `[1]`/`[2]`
   markers that map back to the exact passages the facts came from.
3. **Flag low-confidence responses** — Case 2 combined a sub-threshold similarity
   (0.2185) with the model's own `supported=false` and returned a **flag**, not a
   made-up crypto-payment policy.
4. **Fallback to search** — Case 3 had zero retrieval signal, so the agent took a
   distinct, loudly-labeled web-search fallback path and attached no fake sources.
