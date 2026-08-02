# 03 — ReAct Planning Agent

**Goal:** an agent that *reasons and acts in a loop* — with guardrails so it can
never loop forever, critiques its own progress, and degrades cleanly when stuck
instead of crashing or faking success.

## The four ideas (hand-rolled, no framework)

| # | Sub-point | Where it lives |
|---|-----------|----------------|
| 1 | **observe → think → act → reflect loop** | `ReActAgent.run` in [`agent.py`](agent.py) — each iteration the model emits a **Thought + Action** (`_think_act`), the tool runs and returns an **Observation**, then a separate self-critic **Reflects** (`_reflect`); the reflection is written into the transcript the next Thought reads. |
| 2 | **max iteration limits** | `max_steps` (default 6). The `for n in range(1, max_steps+1)` loop can never run forever; on hitting the cap it returns status `MAX_STEPS` with a best-so-far answer. |
| 3 | **self-critic** | `_reflect` — a distinct LLM call that grades the last step (`on_track?`, a critique, a `next_hint`). A wrong turn (an ERROR, an irrelevant result) gets flagged and corrected on the following Thought. |
| 4 | **graceful degradation** | tools never crash the loop (`tools.Tool.run` turns any exception into an ERROR observation); the model can emit `give_up` to return a **partial** answer + reason (status `DEGRADED`), and the step cap is the backstop. Never a hallucinated success. |

## The loop, concretely

```
        ┌───────────────────────────────────────────────┐
        │  THINK + ACT  (LLM → JSON: thought, action,    │
        │                action_input)                   │
        └───────────────────────────────────────────────┘
                 │ action = tool            │ action = final  → SOLVED
                 ▼                          │ action = give_up → DEGRADED
        ┌──────────────────┐               │
        │  ACT: run tool   │  errors become an ERROR observation (no crash)
        └──────────────────┘
                 │ Observation
                 ▼
        ┌──────────────────┐   self-critic: on_track? what next?
        │  REFLECT (LLM)   │   → next_hint fed into the next Thought
        └──────────────────┘
                 │  (loop, up to max_steps)
                 ▼
        cap reached → MAX_STEPS → synthesise best-so-far partial answer
```

Each step is a strict JSON object (`{"thought","action","action_input"}`), parsed
fence-tolerantly; `temperature=0` for reproducibility. The model chooses one of
five actions each step: the three tools, `final`, or `give_up`.

## The three tools ([`tools.py`](tools.py))

- **`calculator`** — a *safe* arithmetic evaluator that walks the Python AST
  (`+ - * / // % **` on numbers only). No `eval()`, so a hostile expression can't
  execute code — it just returns an error observation.
- **`knowledge_lookup`** — a tiny key/value knowledge base about a **fictional**
  company, *Zephyr Labs*. Fictional on purpose: the model can't answer from
  memory, so a correct answer must flow through the tool.
- **`web_search`** — a **mock** web search over a canned index, with a switchable
  *outage* (`set_search_outage(True)`) that raises a `SearchBackendError` (a
  simulated HTTP 503) — this is what makes the degradation path demonstrable.

## What the demo shows ([`run.py`](run.py))

- **DEMO A — a genuine multi-step task** → `SOLVED`.
  "What did Zephyr Labs spend launching satellites?" needs *two* knowledge
  lookups (`satellites_launched` = 47, `cost_per_satellite_musd` = 12) *then* a
  `calculator` step (`47 * 12 = 564`). The recorded run also takes a wrong turn
  mid-solve (a spurious unit lookup that errors), the self-critic flags it
  `on_track=false`, and the next step recovers and finalises **564 million USD**.
- **DEMO B — a search-backend outage** → `DEGRADED`.
  The answer needs a live figure only `web_search` could supply, but the backend
  is forced into an outage so every call raises. The agent catches each ERROR,
  the self-critic corrects course (it switches to `knowledge_lookup`, which also
  has no such fact), and rather than invent a number it degrades to a **partial
  answer + the reason**. The `max_steps` cap stands as the backstop.

## Run it

```bash
# from the repo root, with .venv active and NVIDIA_API_KEY set in .env
python 03-react-planning/run.py
```

See [`recorded-run.md`](recorded-run.md) for a real captured transcript against
NVIDIA NIM (`meta/llama-3.1-8b-instruct`).

## Files

- `tools.py` — the three pure-Python tools + a uniform `ToolResult`/`Tool` wrapper
- `agent.py` — the ReAct loop: think+act → observe → reflect, with the cap,
  self-critic, and degradation paths
- `run.py` — a runnable demo: one solvable multi-step task + one outage/degradation
- `recorded-run.md` — a real transcript hitting NVIDIA NIM

## Note on the model

`meta/llama-3.1-8b-instruct` is fast/warm on the NIM free tier but a small model —
it can second-guess a correct tool result. That is *exactly* why the guardrails
matter: the self-critic, the step cap, and the degradation paths keep a shaky
reasoner bounded and honest. The prompts explicitly tell both the actor and the
critic to treat tool observations as ground truth and to stop once the answer is
in hand.
