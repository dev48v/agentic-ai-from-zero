# Project 12 — Give Back

**The last project of the twelve.** The other eleven took things from open source. This one
puts something back: the guards that eleven projects of failing in public turned out to
need, packaged so somebody else can install them, plus a documented gap in a real agent
framework with a patch that has been verified against the installed library.

```bash
python 12-give-back/run.py          # 25 self-checks, live model included
```

## What is here

```
12-give-back/
  src/agentfuse/          the library — five fuses, zero dependencies, pip-installable
  tests/                  85 tests
  upstream/               the LangGraph gap report, reproducers, patch, PR draft
  run.py                  the self-checks
  recorded-run.md         an actual run, output and all
```

## Part 1 — `agentfuse`

A runtime safety layer for LLM agents. "Fuse" as in the electrical kind: it sits in the
path, it is dumber than the thing it protects, and it blows **before** the expensive part
burns.

```python
from agentfuse import FuseBox, LoopFuse, PermissionFuse, ToolSpec, BudgetFuse, Price, ToolCall

box = FuseBox(
    permission=PermissionFuse(granted={"read"},
                              specs=[ToolSpec.of("lookup_order", "read"),
                                     ToolSpec.of("issue_refund", "write", "spend_money")]),
    loop=LoopFuse(repeat_threshold=3),
    budget=BudgetFuse(max_usd=0.05, price=Price(0.05, 0.15)),
)

verdict = box.check_tool_call(ToolCall("issue_refund", {"amount": 40}))
assert verdict.blocked   # "needs scope(s) ['spend_money', 'write'] not granted to this run"
```

Five fuses, and every one of them exists because something went wrong in an earlier
project of this series — not because it seemed like a good idea:

| fuse | what it stops | where it came from |
|---|---|---|
| `PermissionFuse` | a tool the run was never allowed to touch, or one nobody registered | **P4** — a restricted grant refused `ledger_write` while the model insisted |
| `LoopFuse` | the same call again, and the A,B,A,B cycle a counter cannot see | **P11** — repeated-signature alerting, moved onto the hot path |
| `BudgetFuse` | the next model call, when it would breach the $ or token ceiling | **P7** — pre-flight estimate to gate, real `usage` to report |
| `JudgeGate` | an LLM marking its own homework | **P10** — an 8B judge scored its own draft 1.00 on text that did not satisfy the criterion |
| `CanaryFuse` | a config release that costs more than the one it replaces | **P11** — a candidate at 2.20× baseline cost, rolled back automatically |

Three rules hold throughout:

1. **No dependencies.** A safety layer that drags in a dependency tree is a safety layer
   nobody installs, and one more supply chain sitting in the path of your tool calls.
2. **Every verdict is a pure function of recorded facts.** A guard you cannot replay is a
   guard you cannot debug at 3am.
3. **Ambiguity resolves to stop.** An unknown tool, a missing price, unparseable arguments
   — all of them mean "no", never "probably fine".

### What is new, versus what was extracted

Two things here did not exist in P1–P11:

* **cycle detection.** P11 counted signatures. That silently misses `A,B,A,B`, where no
  single signature repeats often enough to trip a threshold. Measured on the reproducer:
  the cycle detector fires at tool call 4, the count rule at 5, LangGraph at 20.
* **`terminal_fuses`.** This one is a bug the first live run of this project found. The
  loop adapter originally handed every refusal back to the model, and the *guarded* agent
  then spent **more** tokens than the unguarded one — 2,230 against 1,924 — because it
  kept paying for turns after the fuse had already declared it stuck. A loop refusal now
  ends the run. A permission refusal still does not: the model can write an honest "I
  could not do that", and on that same run it did.

### Adapters

* `agentfuse.adapters.openai_tools` — a guarded tool-calling loop for any
  OpenAI-compatible endpoint (NIM, Groq, OpenAI, OpenRouter, vLLM).
* `agentfuse.adapters.langgraph_guard` — `fuse_wrap_tool_call` for LangGraph's own
  `ToolNode(wrap_tool_call=...)` hook (works with `create_react_agent`), and
  `guard_tool_node` for when you want the graph to stop outright.

Both import their framework lazily. Installing `agentfuse` pulls in nothing.

## Part 2 — the upstream contribution

**[`upstream/`](upstream/)** — a documented, reproduced gap in **langgraph 1.2.11**:
`recursion_limit` is a step cap, not a progress check. With the library's default of
**10,007** supersteps, a stuck agent makes **5,003 identical tool calls** before anything
intervenes. A live NIM agent burns 2,029 real tokens on 6 identical calls and then dies
with a `GraphRecursionError`.

The folder contains three runnable reproducers (two need no API key and cost nothing), a
patch adding an opt-in `ToolNode(no_progress_limit=...)`, a verification script that scored
**8/8 against the patched library** including a live model, and a ready-to-send PR
description.

**Nothing was submitted.** No issue, no pull request, nothing pushed anywhere. It is
prepared for a human to review and decide on — which is also the only honest way to do
this, since the maintainers deserve a patch someone has actually re-verified against
current `main`.

## Install and test

```bash
pip install -e 12-give-back            # or: pip install 12-give-back/
pytest 12-give-back/tests -q           # 85 passed
```

The LangGraph tests and the upstream reproducers need the optional extra:

```bash
pip install "agentfuse[langgraph]"     # or just: pip install langgraph langchain-openai
```

Without it, those tests skip and the rest of the suite still passes — which is the point of
having no dependencies.

## The self-checks

`run.py` prints 25 checks in four sections: the package (dependency-free, pip-installable,
tests green), the lessons (each fuse replaying the project that produced it), **live**
(a real guarded agent against NVIDIA NIM, with real token counts), and **upstream** (the
LangGraph gap, reproduced against the installed library). See
[`recorded-run.md`](recorded-run.md) for the output of an actual run.

## The one idea, twelve projects later

**The model reasons. Plain code enforces.**

P4 said it about permissions, P6 about human approval, P9 about votes, P10 about rubrics,
P11 about rollbacks. `agentfuse` is that sentence with a `pyproject.toml` attached.
