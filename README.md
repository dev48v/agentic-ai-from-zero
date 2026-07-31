# Agentic AI from Zero

**12 agent patterns, built from zero.** Each project teaches one core agentic idea, hand-rolled with the raw OpenAI-compatible SDK — no heavy framework until it is genuinely needed. Progressive: every project builds on the ideas of the last.

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

## Projects

| # | Project | Teaches |
|---|---------|---------|
| 01 | [Structured Output Agent](01-structured-output/) | schema-first agents, the parse→retry loop, typed I/O |
| 02 | RAG Agent — citation grounding | retrieval → grounded generation, confidence |
| 03 | ReAct Planning Agent | observe→think→act→reflect, bounded iteration |
| 04 | Multi-Tool Orchestrator | dynamic tool registry, parallel execution |
| … | *(see ROADMAP)* | |

## Environment variables

| Var | Default | Purpose |
|-----|---------|---------|
| `NVIDIA_API_KEY` | *(required)* | your NVIDIA NIM key |
| `NIM_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible base URL |
| `NIM_MODEL` | `meta/llama-3.1-8b-instruct` | default model id (warm/fast on the free tier) |

Secrets live in `.env` (gitignored). Never commit your key.
