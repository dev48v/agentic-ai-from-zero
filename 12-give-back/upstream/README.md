# Upstream gap report — LangGraph has a step cap, not a progress check

**Status: NOT SUBMITTED.** Everything here is prepared locally and verified locally. No
issue has been opened, no pull request exists, nothing has been pushed to anyone else's
repository. A human has to read it, agree with it, and decide whether to send it.

Measured against **langgraph 1.2.11 / langchain-core 1.5.4 / langchain-openai 1.4.3**,
Python 3.11.9, on 2026-08-12. Re-run everything before quoting a number: two of these
figures are version constants and they have moved before.

---

## The claim

LangGraph's only defence against an agent that has stopped making progress is
`recursion_limit`, a cap on total supersteps that raises `GraphRecursionError` when
exhausted. A step cap answers *"has this gone on too long?"*. It does not answer
*"is this getting anywhere?"* — and those are different questions:

* a ten-step run that never repeats itself is healthy;
* a four-step run that has asked the identical question three times is already dead, and
  every further lap costs tokens, latency and — if the tool has side effects — real damage.

The whole of this 12-project series has been about that distinction. Project 11 called it
*the agent bug that never throws*: an agent can loop, overspend and degrade while every
single HTTP call returns 200.

## What the code actually does — measured, not assumed

Run `python upstream/repro_1_no_progress.py`. A two-node graph, a scripted model that asks
`search_orders(order_id="42")` forever, and a tool that always answers `no results found`.
No API key, no cost, byte-identical on any machine.

| `recursion_limit` | model calls | identical tool calls | outcome | wall time |
|---|---|---|---|---|
| 25 | 13 | 12 | `GraphRecursionError` | 0.12 s |
| **default (10007)** | **5,004** | **5,003** | `GraphRecursionError` | 122.8 s |
| default, with a `LoopFuse(repeat_threshold=3)` in front of the tool node | 3 | 2 | completed normally | 0.01 s |

`DEFAULT_RECURSION_LIMIT` is `10007` in this version
(`langgraph/_internal/_config.py`, overridable with `LANGGRAPH_DEFAULT_RECURSION_LIMIT`).
On the standard model→tools→model loop that is **5,003 identical tool calls** before
anything intervenes. This constant has changed across releases, so check yours rather than
trusting this table.

### It is worse for cycles than for repeats

Run `python upstream/repro_2_cycle.py`. The agent alternates `check_order` and
`check_customer` with fixed arguments — each tool's answer points at the other one. No
single signature repeats often enough for a counting rule to trip.

| guard | first fires at tool call |
|---|---|
| LangGraph `recursion_limit=40` | 20 (the limit) |
| a per-signature **count** rule — which is exactly what Project 11 of this series shipped | 5 |
| `LoopFuse` with cycle detection | **4** |

The middle row is the interesting one. The count rule is the obvious fix, this series
shipped it, and it is still the slowest of the two guards. `agentfuse` ships both
detectors for that reason.

### A real model does this, it is not a strawman

Run `python upstream/repro_3_live_langgraph.py` — live `meta/llama-3.1-8b-instruct` on
NVIDIA NIM through `langchain_openai.ChatOpenAI`, inside a real LangGraph loop, given a
tool that cannot satisfy it and a prompt that tells it not to give up.

| | outcome | live LLM calls | tool calls | REAL tokens |
|---|---|---|---|---|
| stock, `recursion_limit=12` | `GraphRecursionError` | 6 | 6 (all identical) | 2,029 |
| + `LoopFuse(repeat_threshold=3)` | completed | 3 | 2 | 862 |

**57.5% fewer tokens on the wire**, and the run ends with an explanation instead of a
stack trace. Token counts are read off `usage_metadata`, not estimated.

---

## What LangGraph already gives you, and why it is not enough

Credit where it is due: `ToolNode` has a documented interceptor,
`wrap_tool_call(request, execute)`, and `ToolCallRequest` carries `state`. So a progress
check **is** implementable today without touching the library, and
`agentfuse.adapters.langgraph_guard.fuse_wrap_tool_call` is exactly that — about forty
lines.

Three things still argue for having it upstream:

1. **Nobody writes it.** It is the guard you need on the day you did not think to write it.
2. **The obvious implementation is wrong.** A `ToolNode` instance is shared across every
   request the process serves, so a counter kept on the node leaks one user's loop into
   another user's run. The correct version derives the history from the transcript on
   every call, and skips calls whose `ToolMessage` is an error because those never ran.
   That is not obvious, and getting it wrong is worse than not having it.
3. **The default makes it urgent.** 10,007 supersteps is not a safety net anybody is
   relying on deliberately.

## The proposed change

`langgraph-no-progress-limit.patch` — one new optional argument on `ToolNode`:

```python
ToolNode(tools, no_progress_limit=3)
```

Off by default. When set, a tool call whose `(name, args)` signature has already run
`no_progress_limit - 1` times in this request is refused instead of executed, and comes
back as a `ToolMessage(status="error")` explaining itself. Stateless, derived from the
transcript, all-or-nothing per turn. Roughly 100 added lines in
`libs/langgraph/langgraph/prebuilt/tool_node.py`, no new dependency, no behaviour change
for anyone who does not pass the argument.

### Verified, on the installed library

```bash
cd <site-packages>
patch -p3 < upstream/langgraph-no-progress-limit.patch
python upstream/verify_patch.py     # restore the file afterwards
```

**8/8 checks passed** against langgraph 1.2.11:

* the argument exists, defaults to `None`, and rejects `no_progress_limit=1`;
* off by default, stock behaviour is bit-for-bit unchanged (10 identical calls, then
  `GraphRecursionError`);
* with `no_progress_limit` at 2 / 3 / 5 the tool runs exactly 1 / 2 / 4 times — however
  many turns the graph then spins;
* an agent that keeps changing its arguments is never refused;
* **live**: the same NIM agent that stock LangGraph could only stop with a
  `GraphRecursionError` after 6 tool calls and 1,939 tokens finished normally after
  **2 tool calls and 1,676 tokens** with the patch.

### The honest limitation

This bounds **tool execution**, not graph execution. The graph can keep looping between
the model and a node that now refuses instantly — the model is told, and it is up to the
model and your routing to act on it. In the live run above it did stop; a more stubborn
model might not. Pair it with `recursion_limit` as the outer backstop. Stopping the
side-effecting, billable half is the part worth having, and it is the part `recursion_limit`
cannot do.

An alternative worth discussing with maintainers is putting the check in the Pregel loop
so it can end the graph outright, the way `recursion_limit` does. That is a much larger
change and it is not what this patch attempts.

## Files here

| file | what it is |
|---|---|
| `repro_1_no_progress.py` | identical-call loop, scripted model, free to run |
| `repro_2_cycle.py` | A,B,A,B cycle; also replays the count-only rule for comparison |
| `repro_3_live_langgraph.py` | the same gap with a live NIM model (needs a key) |
| `langgraph-no-progress-limit.patch` | the proposed change, `patch -p3` against site-packages |
| `verify_patch.py` | 8 checks that the patch does what the PR says |
| `PR_DESCRIPTION.md` | the pull request text, ready for a human to review and send |
