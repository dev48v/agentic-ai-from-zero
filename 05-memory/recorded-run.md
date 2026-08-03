# Recorded run — Memory-Enabled Conversational Agent

A **real** two-phase transcript captured by running two SEPARATE processes that share
one on-disk store:

```bash
python 05-memory/run.py session1   # teach facts; buffer overflows -> COMPRESSION; exit
python 05-memory/run.py session2   # a FRESH process; recall session-1 facts
```

- **Provider / endpoint:** NVIDIA NIM — `POST https://integrate.api.nvidia.com/v1/chat/completions` (every answer + every compression call returned `HTTP/1.1 200 OK` in the live `httpx` log below).
- **Model:** `meta/llama-3.1-8b-instruct` (warm on the free tier).
- **Date:** 2026-08-03.
- **Why two processes:** cross-session memory can only be *proven* if session 2 is a brand-new OS process that shares nothing with session 1 except the JSON file on disk. It is.

Everything below is verbatim from the run.

---

## SESSION 1 — teach facts, overflow the buffer, COMPRESS

The user states three durable facts (name, a peanut allergy, a project), then chats
until the short-term buffer (`max_turns=4`) overflows. Each overflow **summarises** the
two oldest exchanges into a compact long-term note (`compress_chunk=2`) and drops the
raw turns from the window. Every user turn is *also* archived to the long-term vector
store, so both **past turns** and **compression summaries** are recallable.

### The two compression events (real NIM summarisation calls)

```
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO  memory-agent  COMPRESSED 2 old exchange(s) -> long-term memory #6 (freed ~80 tok): Devanshu is allergic to peanuts.
   🗜  COMPRESSION FIRED — buffer overflowed; summarised 2 oldest exchange(s) into long-term memory #6 (freed ~80 tokens):
        “Devanshu is allergic to peanuts.”
   📦 buffer: 3 exchanges (~166 tok) · long-term: 6 memories
```

```
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO  memory-agent  COMPRESSED 2 old exchange(s) -> long-term memory #9 (freed ~94 tok): She's building Safar, a travel-planner app using Flutter and Supabase, and considered the tagline "Explore Safar, Explore the World."
   🗜  COMPRESSION FIRED — buffer overflowed; summarised 2 oldest exchange(s) into long-term memory #9 (freed ~94 tokens):
        “She's building Safar, a travel-planner app using Flutter and Supabase, and considered the tagline "Explore Safar, Explore the World."”
   📦 buffer: 3 exchanges (~290 tok) · long-term: 9 memories
```

**Shows context compression:** when the verbatim window overflowed, older turns were
folded into a one-line note by the model and moved to long-term — the buffer stayed
bounded (`~166 → capped`) instead of growing forever, and ~80 and ~94 estimated tokens
were freed each time.

### Relevance scoring is already live *within* session 1

Even before a fresh session, each user turn scores every long-term memory and injects
only the ones above threshold (`min_score=0.05`, `top_k=3`):

```
👤 USER : Give me a catchy one-line tagline for a travel app.
   🔎 relevance scoring over long-term memory (top-k cosine):
        [INJECT] score=+0.272  #3 (turn, session-1): I'm building a travel-planner app called Safar, in Flutter with a Supabase backend.
        [ skip ] score=+0.000  #1 (turn, session-1): Hi! My name is Devanshu.
        [ skip ] score=+0.000  #2 (turn, session-1): Important: I'm allergic to peanuts — please remember that.
🤖 AGENT: I remember you're building a travel-planner app called Safar. Here's a possible tagline: "Explore Safar, Explore the World."
```

### End state of session 1 (persisted to `memory_store.json`)

```
short-term buffer now holds 4 recent exchanges (verbatim):
   • Suggest a calm colour for the app's main theme.
   • List three trip-planning features worth adding first.
   • What's a good font pairing for a travel brand?
   • Recommend one app-store screenshot idea that sells it.

long-term vector store now holds 10 memories (8 archived turns + 2 compression summaries):
   #1 [turn] Hi! My name is Devanshu.
   #2 [turn] Important: I'm allergic to peanuts — please remember that.
   #3 [turn] I'm building a travel-planner app called Safar, in Flutter with a Supabase backend.
   #4 [turn] Give me a catchy one-line tagline for a travel app.
   #5 [turn] Suggest a calm colour for the app's main theme.
   #6 [summary] Devanshu is allergic to peanuts.
   #7 [turn] List three trip-planning features worth adding first.
   #8 [turn] What's a good font pairing for a travel brand?
   #9 [summary] She's building Safar, a travel-planner app using Flutter and Supabase, and considered the tagline "Explore Safar, Explore the World."
   #10 [turn] Recommend one app-store screenshot idea that sells it.

💾 saved. The next command starts a FRESH process that only sees this file.
```

Note the **two tiers**: the *recent* four exchanges are kept **verbatim** (short-term
buffer); the older ones survive only as **compressed summaries** (#6, #9) plus the
archived raw turns — the durable facts (name, allergy, project) are all in long-term
now, no longer in the verbatim window.

---

## SESSION 2 — a FRESH process recalls session-1 facts

`python 05-memory/run.py session2` is a **new OS process**. It shares nothing with
session 1 except `memory_store.json` on disk, which it loads:

```
loaded 10 long-term memories + 4 residual buffer exchanges from a PRIOR process:
   #1 [turn, session-1] Hi! My name is Devanshu.
   #2 [turn, session-1] Important: I'm allergic to peanuts — please remember that.
   #3 [turn, session-1] I'm building a travel-planner app called Safar, in Flutter with a Supabase backend.
   ...
   #6 [summary, session-1] Devanshu is allergic to peanuts.
   #9 [summary, session-1] She's building Safar, a travel-planner app using Flutter and Supabase, and considered the tagline "Explore Safar, Explore the World."
```

### Recall 1 — the allergy (summary + past turn both retrieved)

```
INFO  memory-agent  recall q='Remind me — what am I allergic to?' -> [(6, 0.4082, 'inject'), (2, 0.3536, 'inject'), (1, 0.0, 'skip')]
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

👤 USER : Remind me — what am I allergic to?
   🔎 relevance scoring over long-term memory (top-k cosine):
        [INJECT] score=+0.408  #6 (summary, session-1): Devanshu is allergic to peanuts.
        [INJECT] score=+0.354  #2 (turn, session-1): Important: I'm allergic to peanuts — please remember that.
        [ skip ] score=+0.000  #1 (turn, session-1): Hi! My name is Devanshu.
🤖 AGENT: You're allergic to peanuts.
```

### Recall 2 — the project + its tech stack

```
INFO  memory-agent  recall q="What's my travel app called, and what's it built with?" -> [(3, 0.5, 'inject'), (4, 0.4082, 'inject'), (12, 0.2942, 'inject')]
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

👤 USER : What's my travel app called, and what's it built with?
   🔎 relevance scoring over long-term memory (top-k cosine):
        [INJECT] score=+0.500  #3 (turn, session-1): I'm building a travel-planner app called Safar, in Flutter with a Supabase backend.
        [INJECT] score=+0.408  #4 (turn, session-1): Give me a catchy one-line tagline for a travel app.
        [INJECT] score=+0.294  #12 (summary, session-2): ...Three initial features for the Safar app are Destination Search, Itinerary Builder, and Travel Recommendations.
🤖 AGENT: Your travel app is called Safar, and it's built with Flutter and has a Supabase backend.
```

### Recall 3 — the name (highest-scoring memory wins)

```
INFO  memory-agent  recall q="And what's my name again?" -> [(1, 0.5, 'inject'), (2, 0.0, 'skip'), (3, 0.0, 'skip')]
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

👤 USER : And what's my name again?
   🔎 relevance scoring over long-term memory (top-k cosine):
        [INJECT] score=+0.500  #1 (turn, session-1): Hi! My name is Devanshu.
        [ skip ] score=+0.000  #2 (turn, session-1): Important: I'm allergic to peanuts — please remember that.
        [ skip ] score=+0.000  #3 (turn, session-1): I'm building a travel-planner app called Safar, in Flutter with a Supabase backend.
🤖 AGENT: Your name is Devanshu.
```

Each answer is **correct** and each was produced **only** because the right memory —
scored by hand-rolled cosine and injected into the prompt — survived on disk from a
**different process**. Compression also fired again in session 2 (memories #12, #15),
proving the buffer stays bounded no matter which session you are in.

---

## The four sub-points, together

| Sub-point | Evidence in this run |
|-----------|----------------------|
| **Short-term buffer** | Session 1 kept only the most recent 4 exchanges verbatim (`📦 buffer: 3–4 exchanges`); older ones left the window. |
| **Long-term vector recall** | Session 2 retrieved memories by hand-rolled cosine — `#3` scored `0.500` for the app query, `#6`/`#2` scored `0.408`/`0.354` for the allergy. |
| **Context compression** | Two model-summarised notes in session 1 (`#6` "Devanshu is allergic to peanuts.", `#9` the Safar summary), each freeing ~80–94 tokens from the buffer. |
| **Relevance scoring** | Every turn shows `[INJECT]`/`[ skip ]` decisions at `min_score=0.05`, `top_k=3` — only above-threshold memories entered the prompt. |
| **Cross-session sync** | Session 2 is a **fresh process** that loaded `memory_store.json` and answered "peanuts", "Safar / Flutter + Supabase", and "Devanshu" — all facts from session 1. |

> Note: the model is small (8B) and not perfectly deterministic even at low temperature,
> so a re-run may phrase the summaries or answers differently. What is stable: the
> **embedding is deterministic** (a stable `hashlib` hash, not Python's per-process
> salted `hash()`), so a memory earns the **same cosine score** in session 2 as it would
> in session 1 — which is the whole reason cross-session recall works. Recall, relevance
> scoring, and the buffer window are pure Python; only compression and the final answer
> use the model.
