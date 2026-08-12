# Agentic AI from Zero

**12 agent patterns, built from zero. All 12 are done.** Each project teaches one core agentic idea, hand-rolled with the raw OpenAI-compatible SDK — no heavy framework until it is genuinely needed. Progressive: every project builds on the ideas of the last.

See [`ROADMAP.md`](ROADMAP.md) for the full 12-project plan.

## Stack

- **Language:** Python 3.10+
- **LLM:** [NVIDIA NIM](https://build.nvidia.com) free API — an OpenAI-compatible endpoint (`https://integrate.api.nvidia.com/v1`) that hosts Llama-3.3, Nemotron, DeepSeek, Qwen and more, with tool-calling + JSON support. Free key, no card, for personal use.
- **Client:** one swappable OpenAI-compatible client in [`common/client.py`](common/client.py) — point it at any free provider (Groq, Gemini, OpenRouter) by changing two env vars.
- **No local compute** — everything is a hosted free API.

## Setup

```bash
# 1. clone / cd into the repo
cd agentic-ai-from-zero

# 2. create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# 4. add your NVIDIA NIM key
cp .env.example .env
# then edit .env and paste your key:
#   NVIDIA_API_KEY=nvapi-xxxxxxxx
# get one free at https://build.nvidia.com
```

## Run a project

Each project is a self-contained folder with its own `run.py` and `README.md`:

```bash
python 01-structured-output/run.py
```

Project 12 also ships an installable library and a test suite:

```bash
pip install -e 12-give-back
pytest 12-give-back/tests -q
```

## Projects

| # | Project | Teaches |
|---|---------|---------|
| 01 | [Structured Output Agent](01-structured-output/) | schema-first agents, the parse→retry loop, typed I/O |
| 02 | [RAG Agent — citation grounding](02-rag-citation/) | retrieval → grounded generation, inline citations, confidence |
| 03 | [ReAct Planning Agent](03-react-planning/) | observe→think→act→reflect, bounded iteration, self-critique |
| 04 | [Multi-Tool Orchestrator](04-orchestrator/) | dynamic tool registry, capability routing, permissions, parallel execution |
| 05 | [Memory-Enabled Conversational Agent](05-memory/) | short-term buffer + long-term recall, compression, cross-session sync |
| 06 | [Human-in-the-Loop Approval Agent](06-human-in-the-loop/) | uncertainty detection, pause/resume, audit trail |
| 07 | [Cost-Aware Agent Router](07-cost-aware-router/) | token budgeting, complexity routing, early exit, cost analytics |
| 08 | [Event-Triggered Automation Agent](08-event-triggered/) | webhooks + queues, idempotent execution, retry + dead-letter |
| 09 | [Multi-Agent Debate](09-multi-agent-debate/) | proposers + critic, voting/consensus, confidence-weighted synthesis |
| 10 | [Self-Reflective Agent (auto-eval)](10-self-reflective/) | LLM-as-judge on its own output, rubric gate, constrained refinement |
| 11 | [Production Agent (observability)](11-production-observability/) | tracing, latency + cost dashboard, alerting on loops, canary + rollback |
| 12 | [Give Back — `agentfuse` + a LangGraph gap report](12-give-back/) | packaging the series' lessons as a real library, and contributing them upstream |

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `NVIDIA_API_KEY` | *(required)* | your NVIDIA NIM key |
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible base URL |
| `NIM_MODEL` | `meta/llama-3.1-8b-instruct` | default model id (warm/fast on the free tier) |

Secrets live in `.env` (gitignored). Never commit your key.

## Where it ended up

Twelve projects, one sentence: **the model reasons, plain code enforces.**

Every project that worked put the judgement in the model and the enforcement in ordinary,
replayable Python — permissions in P4, human approval in P6, the vote in P9, the rubric's
hard checks in P10, the rollback guard in P11. Project 12 packages five of those guards as
[`agentfuse`](12-give-back/), a dependency-free library, and takes the loop-detection
lesson back to an open-source agent framework as a reproduced gap report and a verified
patch.
