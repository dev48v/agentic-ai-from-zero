# Agentic AI from Zero — Roadmap

**What:** 12 agent patterns built FROM ZERO, progressive. Each teaches one core agentic idea, culminating in production hardening + an open-source contribution.

## Decisions (locked)
- **Stack:** Python, **100% FREE — no paid API.** Default LLM = **NVIDIA NIM free API** (build.nvidia.com / API Catalog): OpenAI-compatible endpoint `https://integrate.api.nvidia.com/v1`, free credits + free key (no card, personal use), hosts Llama-3.3-70B / Nemotron / DeepSeek / Qwen with tool-calling + JSON. Accessed via ONE swappable OpenAI-compatible client in `common/` (so any free provider — Groq/Gemini/OpenRouter-free — drops in). **Fallback = Groq free** (also fully hosted, fast, OpenAI-compatible) — NO local compute anywhere (Ollama ruled out: too slow locally). Hand-rolled patterns, NO heavy framework until it's genuinely needed (#11 tracing, #12 give-back).
- **Repo:** monorepo `dev48v/agentic-ai-from-zero` (PUBLIC) — one folder per project, progressive, shared `common/` utils. Secrets via `.env` (gitignored). #12 = PR to an *external* framework repo.
- **Live pages (dev48v):** **recorded real runs** — repo code is genuinely runnable; each dev48v page shows a REAL captured transcript/trace (no in-browser key). LOOK / UNDERSTAND / BUILD, same as OrderHub.
- **Cadence:** 2nd daily tech slice, **parallel with OrderHub**, starting Bot 10 **Day-49**. 1 agent project per day.

## Per-project format
Each project = **Goal · Sub-points (features/steps) · Teaches · Recorded-run artifact.**

Legend: sub-points marked ✅ = user-specified; 🟡 = my draft (edit freely).

---

### 1. Structured Output Agent
- **Goal:** an agent whose every output is a validated, typed object — never free text you have to parse.
- **Sub-points:** ✅ enforce a Pydantic JSON schema · ✅ validate tool responses · ✅ retry on parse errors · ✅ log validation failures
- **Teaches:** schema-first agents, tool-arg validation, the parse→retry loop, why typed I/O is the foundation for everything after.
- **Artifact:** transcript of a run that hits a parse error, retries, and lands a valid object + the failure log.

### 2. RAG Agent — citation grounding
- **Goal:** answers backed by retrieved sources, with honest confidence.
- **Sub-points:** ✅ retrieve context · ✅ generate answers with sources (citations) · ✅ flag low-confidence responses · ✅ fallback to search
- **Teaches:** retrieval → grounded generation, inline citations, confidence signaling, graceful fallback when the corpus is thin.
- **Artifact:** a grounded answer with citations + a low-confidence case that triggers the search fallback.

### 3. ReAct Planning Agent
- **Goal:** an agent that reasons and acts in a loop, with guardrails.
- **Sub-points:** ✅ observe→think→act→reflect loop · ✅ max iteration limits · ✅ self-critic · ✅ graceful degradation
- **Teaches:** the ReAct loop, bounded iteration (no infinite loops), self-critique between steps, degrading safely when stuck.
- **Artifact:** a multi-step ReAct trace showing the loop, a self-critique correcting course, and the iteration cap firing.

### 4. Multi-Tool Orchestrator Agent
- **Goal:** route across many tools safely and in parallel.
- **Sub-points:** ✅ dynamic tool registry · ✅ capability-based routing · ✅ permission scoping · ✅ parallel execution · ✅ conflict resolution
- **Teaches:** registering tools at runtime, routing by capability, per-tool permission scopes, running independent tools concurrently, resolving conflicting results.
- **Artifact:** a run that registers tools, routes a request, executes 2+ in parallel, and resolves a conflict.

---

### 5. Memory-Enabled Conversational Agent
- **Goal:** an agent that remembers across turns and sessions.
- **Sub-points:** ✅ short-term buffer + long-term vector recall · ✅ context compression · ✅ relevance scoring · ✅ cross-session sync
- **Teaches:** the two-tier memory model, when to summarize vs store, retrieving the *right* memory, session continuity.
- **Artifact:** a two-session transcript where the agent recalls a fact from session 1 in session 2.

### 6. Human-in-the-Loop Approval Agent
- **Goal:** pause for human approval on risky actions, then resume.
- **Sub-points:** ✅ uncertainty detection · ✅ pause for human input · ✅ resume with validated context · ✅ full audit trail
- **Teaches:** durable pause/resume, the approve-reject-edit surface, auditability, escalation when no one responds.
- **Artifact:** a run that pauses on a risky action, gets an edited approval, checkpoints, and resumes.

### 7. Cost-Aware Agent Router
- **Goal:** spend the least money that still solves the task.
- **Sub-points:** ✅ token budgeting per task · ✅ model routing by complexity and cost · ✅ early exit on confidence · ✅ cost-per-decision analytics
- **Teaches:** model tiering, complexity estimation, hard budget ceilings, cache-before-call. (Mirrors your own "+budget" orchestration instinct.)
- **Artifact:** a batch where easy tasks route to a cheap model, hard ones to frontier, with a spend report vs a naive all-frontier baseline.

### 8. Event-Triggered Automation Agent
- **Goal:** an agent that fires on events, not chat.
- **Sub-points:** ✅ listen to webhooks + queues · ✅ execute workflows on triggers · ✅ idempotent execution · ✅ dead-letter handling + retry logic
- **Teaches:** event-driven agents, idempotency (ties to OrderHub Day-32), retry/backoff, dead-lettering (OrderHub Day-31).
- **Artifact:** a webhook that triggers the agent, a duplicate event that's deduped, and a failing event landing in the DLQ.

### 9. Multi-Agent Debate System
- **Goal:** multiple agents argue to a better answer than one.
- **Sub-points:** ✅ multiple agents propose solutions · ✅ critic evaluates · ✅ voting / consensus logic · ✅ aggregator synthesizes with confidence
- **Teaches:** perspective diversity, structured critique, judge panels, knowing when to stop. (This is the workflow "judge panel" pattern, hand-built.)
- **Artifact:** a debate transcript where two agents disagree and the judge synthesizes the winning answer.

### 10. Self-Reflective Agent — auto-eval
- **Goal:** an agent that grades and improves its own work.
- **Sub-points:** ✅ execute + evaluate via LLM-as-judge · ✅ critic reasoning · ✅ regenerate with constraints · ✅ log improvement metrics
- **Teaches:** self-improvement loops, building an eval set, LLM-as-judge + hard checks, gating on quality not vibes.
- **Artifact:** a before/after showing the score climbing across revisions until it passes the gate.

### 11. Production Agent — observability
- **Goal:** take an earlier agent and make it production-grade.
- **Sub-points:** ✅ LangSmith / Arize (Phoenix) tracing · ✅ latency + cost dashboard · ✅ alerting on loops + failures · ✅ canary testing + rollback
- **Teaches:** wrapping an agent for real traffic — the exact observability arc you just did on OrderHub (Day 37-38), applied to an agent.
- **Artifact:** a traced run + a metrics/dashboard snapshot of the agent under load.

### 12. Open-Source Agent Framework Contribution
- **Goal:** give back — a real merged PR to an agent framework.
- **Sub-points:** ✅ extend LangGraph / CrewAI / AutoGen · ✅ write docs + demo · ✅ publish benchmarks · ✅ submit PR + tutorial
- **Teaches:** navigating a real OSS codebase, matching its conventions, seeing a change through review to merge.
- **Artifact:** the merged (or open) PR link + the writeup.

---

## Repo layout (planned, monorepo)
```
agentic-ai-from-zero/
  README.md
  ROADMAP.md            <- this file
  common/               <- shared: Anthropic client, tracing, .env loader
  01-structured-output/
  02-rag-citation/
  03-react-planning/
  ...
  11-production-observability/
  12-oss-contribution/   <- links + writeup (the PR lives in the external repo)
  .env.example
  .gitignore
```

## Status
- Roadmap LOCKED — all 12 projects' sub-points user-specified (✅). Teaches/Artifact lines are my supporting notes.
- Repo NOT yet created/pushed (awaiting explicit go).
- First build (Project 1 — Structured Output Agent) = Bot 10 Day-49, as the 2nd daily tech slice alongside OrderHub Day 39.

## Note (#11 vs #12 external tools)
- #11 uses **LangSmith / Arize-Phoenix** for tracing (hosted/OSS observability) — first place the series leans on an external framework. Fine per your spec; keep the raw-SDK agent underneath.
- #12 targets **LangGraph / CrewAI / AutoGen** — the give-back. PR lives in the external repo, not our monorepo.
