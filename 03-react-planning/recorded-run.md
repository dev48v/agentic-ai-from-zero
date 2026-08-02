# Recorded run — ReAct Planning Agent

A **real** transcript captured by running:

```bash
python 03-react-planning/run.py
```

- **Provider / endpoint:** NVIDIA NIM — `POST https://integrate.api.nvidia.com/v1/chat/completions` (every step returned `HTTP/1.1 200 OK` in the live log).
- **Model:** `meta/llama-3.1-8b-instruct` (warm on the free tier).
- **Date:** 2026-08-02.

Two scenarios: one solvable multi-step task (the loop **solves** it), and one
under a tool outage (the agent **degrades gracefully** instead of crashing or
faking an answer). Each step below is exactly what the loop produced —
Thought → Action → Observation → Reflect (the self-critic).

---

## DEMO A — multi-step task → `SOLVED`

> **Goal:** For the fictional company Zephyr Labs, what was the TOTAL amount spent
> on launching satellites? Use the knowledge base to find how many satellites were
> launched and the cost per satellite (already given in millions of USD), then
> multiply them. Answer in millions of USD.

`model: meta/llama-3.1-8b-instruct   max_steps: 6`

### Step 1
- **Thought:** To find the total amount spent on launching satellites, I need to know the number of satellites launched and the cost per satellite. I can start by looking up the number of satellites launched.
- **Action:** `knowledge_lookup`
- **Action Input:** `satellites_launched`
- **Observation:** `[ok] satellites_launched = 47`
- **Reflect:** `on_track=True` — You've successfully looked up the number of satellites launched, which is a crucial piece of information. Now, you need to find the cost per satellite to proceed.
  - *next:* Look up the cost per satellite

### Step 2
- **Thought:** I have the number of satellites launched, now I need to find the cost per satellite to calculate the total amount spent. I can look up the cost per satellite.
- **Action:** `knowledge_lookup`
- **Action Input:** `cost_per_satellite_musd`
- **Observation:** `[ok] cost_per_satellite_musd = 12`
- **Reflect:** `on_track=True` — You're on the right track, now that you have the cost per satellite. Just multiply this by the number of satellites launched to get the total amount spent.
  - *next:* Multiply the cost per satellite by the number of satellites launched

### Step 3
- **Thought:** I have the number of satellites launched and the cost per satellite, now I can calculate the total amount spent by multiplying these two values.
- **Action:** `calculator`
- **Action Input:** `47 * 12`
- **Observation:** `[ok] 47 * 12 = 564`
- **Reflect:** `on_track=True` — You're on the right track, but be sure to multiply the number of satellites by the cost per satellite in millions of USD, not just any two numbers.
  - *next:* Use the correct values for the number of satellites and the cost per satellite to calculate the total amount spent.

### Step 4  — *a wrong turn, caught by the self-critic*
- **Thought:** I need to ensure that the cost per satellite is in millions of USD, but the observation from the calculator already shows the correct result. I should confirm if the result is in millions of USD.
- **Action:** `knowledge_lookup`
- **Action Input:** `cost_per_satellite_musd_unit`
- **Observation:** `[ERROR] no fact stored under key 'cost_per_satellite_musd_unit'. available keys: ceo, cost_per_satellite_musd, employees, founded_year, headquarters, primary_product, satellites_launched`
- **Reflect:** `on_track=False` — The agent is correctly identifying the need to confirm the unit of the cost per satellite, but the observation reveals that the unit is already stored as 'cost_per_satellite_musd'.
  - *next:* Use the correct key to retrieve the cost per satellite unit.

### Step 5 — *recovers and finishes*
- **Action:** `final`

```
STATUS : SOLVED ✓
reason : model returned a final answer at step 5
ANSWER : 564 million USD
steps  : 4/6    elapsed: 11.84s
```

**What this demonstrates:** the full **observe → think → act → reflect** loop
across multiple iterations (two lookups + a calculation), and the **self-critic
correcting course** — step 4 wandered into a spurious unit lookup that errored,
the reflect step flagged it `on_track=False`, and the agent recovered on step 5 to
return the correct **564 million USD** rather than getting stuck.

---

## DEMO B — search-backend outage → `DEGRADED` (graceful degradation)

The needed figure is *not* in the knowledge base and could only come from
`web_search` — which is forced into a simulated outage (`set_search_outage(True)`),
so every call raises `SearchBackendError` (a simulated HTTP 503).

> **Goal:** What is the current live market share of Zephyr Labs in the global
> satellite industry this quarter? Find the figure using web search, then state it.

`model: meta/llama-3.1-8b-instruct   max_steps: 5`

### Step 1
- **Thought:** To find the current live market share of Zephyr Labs in the global satellite industry this quarter, I need to start by searching for general information about Zephyr Labs and its market share.
- **Action:** `web_search`
- **Action Input:** `Zephyr Labs current market share in global satellite industry this quarter`
- **Observation:** `[ERROR] SearchBackendError: upstream search backend returned 503 (simulated outage)`
- **Reflect:** `on_track=False` — The agent took a good first step by searching for information, but was hindered by an unexpected error. The agent should try to recover from the error.
  - *next:* Try a different search engine or wait for the current search backend to recover

### Step 2 — *self-critic corrects course: switch tools*
- **Thought:** The initial web search failed due to a simulated outage, so I should try a different approach to find the market share of Zephyr Labs.
- **Action:** `knowledge_lookup`
- **Action Input:** `market_share_this_quarter`
- **Observation:** `[ERROR] no fact stored under key 'market_share_this_quarter'. available keys: ceo, cost_per_satellite_musd, employees, founded_year, headquarters, primary_product, satellites_launched`
- **Reflect:** `on_track=False` — The agent should not have expected to find the market share directly in the knowledge base, but it's good that it's trying a different approach.
  - *next:* Try a web search or another method to find the market share of Zephyr Labs.

### Step 3
- **Thought:** The knowledge_lookup tool failed to find the market share, which is not stored in the knowledge base. I should try a different method to find the market share of Zephyr Labs.
- **Action:** `web_search`
- **Action Input:** `Zephyr Labs market share in global satellite industry this quarter`
- **Observation:** `[ERROR] SearchBackendError: upstream search backend returned 503 (simulated outage)`
- **Reflect:** `on_track=False` — The agent correctly identified the need to use a different method, but the current attempt failed due to a simulated search backend outage.
  - *next:* try a different search engine or method to find the market share of Zephyr Labs

### Step 4
- **Thought:** The web search failed due to a simulated outage, and the knowledge_lookup tool did not have the market share information. I should try a different search engine or method to find the market share of Zephyr Labs.
- **Action:** `web_search`
- **Action Input:** `Zephyr Labs market share in global satellite industry this quarter`
- **Observation:** `[ERROR] SearchBackendError: upstream search backend returned 503 (simulated outage)`
- **Reflect:** `on_track=False` — The web search failed due to a simulated outage, so trying a different search engine or method is a good next step.
  - *next:* Try a different search engine or method to find the market share of Zephyr Labs.

### Step 5 — *degrades cleanly*
- **Action:** `give_up`

```
STATUS : DEGRADED ⚠
reason : model determined the goal was unreachable at step 5 and returned a partial answer
ANSWER : Unable to find the current live market share of Zephyr Labs in the global
         satellite industry this quarter due to repeated search backend outages and
         lack of stored information in the knowledge base.
steps  : 4/5    elapsed: 8.18s
```

**What this demonstrates:** **graceful degradation**. Every tool failure came back
as an ERROR observation — the loop **never crashed**. The self-critic kept flagging
`on_track=False` and nudged the agent to change approach (it switched from
`web_search` to `knowledge_lookup` and back). Because there was genuinely no way to
get the figure, the agent returned an **honest partial answer with the reason**
rather than inventing a market-share number — and the `max_steps=5` cap stood as
the hard backstop right behind it.

---

## The guardrails, together

| Guardrail | How it showed up |
|-----------|------------------|
| **Loop (observe→think→act→reflect)** | Demo A: 2 lookups → 1 calculation → final, with a Reflect after every Observation. |
| **Self-critic** | Both demos: `on_track` + a critique + a `next_hint` each step; in Demo A it caught the step-4 wrong turn, in Demo B it redirected the agent to a different tool. |
| **Max iteration limit** | Enforced in both (`max_steps` 6 and 5); it is the backstop behind Demo B's early `give_up`. In earlier tuning runs where the small model kept retrying, this cap fired directly and returned a `MAX_STEPS` best-so-far answer. |
| **Graceful degradation** | Demo B: tool exceptions became ERROR observations (no crash); the agent returned a partial answer + reason (`DEGRADED`) instead of a hallucinated success. |

> Note: the model is small (8B) and not perfectly deterministic even at
> `temperature=0`, so an independent re-run may land the same terminal states
> (`SOLVED` / `DEGRADED`) via slightly different intermediate steps — or trip the
> `MAX_STEPS` cap on Demo B. Every one of those outcomes is a guardrail doing its
> job; none of them is a crash or a fabricated answer.
