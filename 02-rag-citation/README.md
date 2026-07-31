# 02 — RAG Agent (citation grounding)

**Goal:** answers backed by retrieved sources, with honest confidence — never a
confident-sounding hallucination.

## The four ideas (hand-rolled, no framework)

| # | Sub-point | Where it lives |
|---|-----------|----------------|
| 1 | **Retrieve context** | `TfidfRetriever` in [`retriever.py`](retriever.py) — TF-IDF bag-of-words + cosine top-k over [`corpus/`](corpus) |
| 2 | **Generate answers with sources (citations)** | `_generate_grounded` in [`agent.py`](agent.py) — the model answers ONLY from the numbered passages and cites them inline with `[1]`, `[2]` |
| 3 | **Flag low-confidence responses** | two signals: top-1 similarity band **and** the model's own `supported` flag → `LOW_CONFIDENCE` instead of a guess |
| 4 | **Fallback to search** | `_fallback_search` — when retrieval is too weak, a clearly-labeled (simulated) web-search path |

## How confidence is decided

The top-1 cosine similarity and the model's self-reported grounding are combined:

```
top_score < FALLBACK_THRESHOLD (0.15)              -> FALLBACK        (retrieval too weak; don't even ground)
FALLBACK <= top_score < CONFIDENT (0.30)           -> LOW_CONFIDENCE  (thin support)
top_score >= CONFIDENT  AND model says grounded    -> GROUNDED        (answer + inline citations)
model reports it cannot support the answer          -> LOW_CONFIDENCE  (flag, never hallucinate)
```

The thresholds are tuned against this specific corpus — the measured scores are in
[`recorded-run.md`](recorded-run.md).

## Why a local TF-IDF retriever (not an embedding API)

Retrieval is a deterministic, offline TF-IDF + cosine step, so the recorded run is
reproducible byte-for-byte and needs no network for the *retrieval* half. Only the
*generation* step calls the LLM. NVIDIA NIM does expose an embedding endpoint;
`retriever.embed()` is written so you could swap it for a NIM `embeddings.create`
call without touching the agent.

## The corpus

Six short markdown docs about a **fictional** product, *Nimbus Cloud* (pricing,
SLA, regions, security, retention, limits). Fictional on purpose: the model can't
answer from parametric memory, so a correct answer *must* come from retrieval —
which is exactly what makes the citations load-bearing.

## The three cases the demo shows

1. **Well-supported** — "What uptime SLA does Enterprise guarantee and its price?"
   → top score **0.6180** → `GROUNDED` with inline `[1]`/`[2]` citations.
2. **Weakly supported** — "Can I pay with cryptocurrency?" → top score **0.2185**,
   the model reports `supported=false` → `LOW_CONFIDENCE`, no hallucination.
3. **Outside the corpus** — "Symptoms of vitamin D deficiency?" → top score
   **0.0000** → `FALLBACK` to a clearly-labeled web-search answer, no fake sources.

## Run it

```bash
# from the repo root, with .venv active and NVIDIA_API_KEY set in .env
python 02-rag-citation/run.py
```

See [`recorded-run.md`](recorded-run.md) for a real captured transcript against
NVIDIA NIM.

## Files

- `corpus/` — six fictional *Nimbus Cloud* fact docs (the knowledge base)
- `retriever.py` — TF-IDF fit + cosine top-k retrieval (offline, deterministic)
- `agent.py` — retrieve → grounded generation with citations → confidence → fallback
- `run.py` — a runnable demo exercising all three cases
- `recorded-run.md` — a real transcript hitting NVIDIA NIM
