# 05 — Memory-Enabled Conversational Agent

**Goal:** an agent that remembers **across turns and across sessions** — keep the recent
conversation verbatim, fold older turns into compact notes when the window overflows,
store everything in a hand-rolled vector memory, and on every user turn recall only the
*relevant* memories — so a brand-new process can still answer questions about facts you
gave it in a previous run.

## The four ideas (hand-rolled, no framework, no paid embedding API)

| # | Sub-point | Where it lives |
|---|-----------|----------------|
| 1 | **short-term buffer** | [`memory.py`](memory.py) `MemoryStore.buffer` — the most recent `max_turns` exchanges kept **verbatim** in a turn-bounded window (also token-estimated). Recent context stays exact. |
| 2 | **long-term vector recall** | [`memory.py`](memory.py) `HashingEmbedder` + `MemoryStore.recall` — a **hand-rolled hashing bag-of-words embedding** + **cosine** similarity over stored memories. Same spirit as Project 2's TF-IDF retriever, but **incremental** (a growing store needs no refit) and using a **stable `hashlib` hash** so a memory embeds identically in a fresh process. No network, no paid embeddings. |
| 3 | **context compression** | [`memory.py`](memory.py) `MemoryStore.append` overflow path — when the buffer overflows, the oldest `compress_chunk` exchanges are **summarised by the model** into one compact note, stored long-term, and the raw turns are dropped. The window stays bounded. |
| 4 | **relevance scoring** | [`memory.py`](memory.py) `MemoryStore.recall` + [`agent.py`](agent.py) — on each user turn, the top-`k` memories by cosine are found and **only those above `min_score`** are injected into the prompt (each `[INJECT]`/`skip` decision is logged with its score). |
| — | **cross-session sync** | [`memory.py`](memory.py) `MemoryStore.save`/`load` — the long-term store (and the residual buffer) persist to `memory_store.json`, so a **new process** recalls facts from a **prior** run. This is what the two-phase demo proves. |

## The two tiers

```
                     ┌───────────────────────────── prompt to the model ─────────────────────────────┐
  user turn ──▶ RECALL (cosine, top-k) ──▶ [ injected long-term memories ] + [ verbatim buffer ] + user
                     │                                                              │
                     ▼                                                              ▼
        long-term vector store  ◀── COMPRESS (summarise) ◀────────── short-term buffer overflows
        (past turns + summaries)                                     (max_turns kept verbatim)
                     │
                     ▼  save() / load()
              memory_store.json   ──────────────▶  a FRESH process loads it (cross-session recall)
```

- **Short-term** = exact recent turns (cheap, high-fidelity, but bounded).
- **Long-term** = a vector store of **archived past turns** *and* **compression summaries**; searched by relevance, unbounded but lossy. Storing the raw user turn keeps the user's own words as the best recall keys; the summary keeps the buffer small and gives a coarse, higher-level memory.

## The hand-rolled embedding (why it's honest)

`HashingEmbedder` tokenises (lowercase, stop-word filtered), hashes each token into a
fixed-dim vector with **signed** buckets, and L2-normalises — so **cosine == dot
product**. It uses `hashlib.md5`, **not** Python's built-in `hash()`, on purpose:
`hash("str")` is salted per process (`PYTHONHASHSEED`), which would give a memory a
*different* vector in session 2 and silently break cross-session recall. A stable hash
makes the whole thing reproducible across processes and machines. (NIM does expose an
embeddings endpoint; `embed()` is written so you could swap in a real embedding call
without touching the agent — TF-IDF/hashing is used here to stay free and reproducible.)

## What the demo shows ([`run.py`](run.py)) — two phases, two processes

- **`session1`** — a fresh store. The user says their **name**, a **peanut allergy**, and
  a **project** (Safar, Flutter + Supabase), then chats until the buffer overflows.
  **Compression fires twice**, folding old turns into long-term notes; the store is
  persisted and the process exits.
- **`session2`** — a **brand-new process**. It loads `memory_store.json` and asks three
  questions that need session-1 facts. Each shows the **relevance scores**, which memory
  was **injected**, and the correct answer: *"peanuts"*, *"Safar / Flutter + Supabase"*,
  *"Devanshu"* — recalled from a store that out-lived the first process.

## Run it

```bash
# from the repo root, with .venv active and NVIDIA_API_KEY set in .env
python 05-memory/run.py session1     # teach facts; buffer overflows -> compression; exit
python 05-memory/run.py session2     # a FRESH process; recall session-1 facts
```

See [`recorded-run.md`](recorded-run.md) for a **real** captured transcript against
NVIDIA NIM (`meta/llama-3.1-8b-instruct`) — every answer and every compression call is a
live `HTTP/1.1 200 OK` to `integrate.api.nvidia.com`, and session 2 is genuinely a
separate process.

## Files

- `memory.py` — the two-tier store: `HashingEmbedder` (embed + cosine), the short-term buffer, long-term vector recall, compression-on-overflow, and JSON save/load.
- `agent.py` — the `MemoryAgent`: recall + relevance scoring, prompt assembly (injected memories + buffer), the answer call, and the model-backed summariser handed to the store.
- `run.py` — the two-phase (`session1` / `session2`) runnable demo.
- `recorded-run.md` — a real two-session transcript hitting NVIDIA NIM.
- `memory_store.json` — runtime state written by `run.py` (gitignored; regenerated).

## Note on the model

The model does exactly two jobs — **compress** old turns into a note, and write the
final **answer** from the injected memories. Everything else is deterministic Python:
the **embedding**, the **cosine recall**, the **relevance threshold**, and the **buffer
window**. That separation is deliberate — the memory mechanics are auditable and
reproducible, and the model is only trusted to summarise and phrase, never to decide what
gets remembered or retrieved.
