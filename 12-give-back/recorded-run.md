# Recorded run — Project 12, Give Back

Real run, 2026-08-12. Windows 11, Python 3.11.9, `meta/llama-3.1-8b-instruct` on the
NVIDIA NIM free tier, langgraph 1.2.11 / langchain-core 1.5.4 / langchain-openai 1.4.3.
Every token count below came off the API's `usage` object. The NIM key lives in the
gitignored repo-root `.env` and appears nowhere in this repo.

```bash
python 12-give-back/run.py
```

## The output

```
==============================================================================
PROJECT 12 — GIVE BACK: agentfuse self-checks
==============================================================================

==============================================================================
1. THE PACKAGE — agentfuse 0.1.0
==============================================================================
  [PASS] zero third-party imports at module level  — 11 modules scanned
  [PASS] pip-installable layout (pip install --target, then import it)  — installed to agentfuse-install-check
  [PASS] pytest suite green  — 85 passed in 3.59s

==============================================================================
2. THE LESSONS — every fuse replays the run that produced it
==============================================================================
  [PASS] P4  restricted grant refuses ledger_write, allows fetch_feed  — tool 'ledger_write' needs scope(s) ['write'] not granted to this run
  [PASS] P4  a tool that is not in the registry is denied, not run  — tool 'exfiltrate' is not in the registry — deny-by-default
  [PASS] P11 identical tool call blocked on the 3rd attempt  — verdicts [True, True, False]
  [PASS] NEW A,B,A,B cycle blocked while a repeat COUNTER stays silent  — max signature count 2 (< threshold 3)
  [PASS] P10 hard check overrules a self-score of 1.00  — deterministic hard check(s) failed: mentions-discount; the model scored this a PASS anyway
  [PASS] P7  budget pre-flight allows the cheap call, refuses the dear one  — projected spend $0.300400 would breach the $0.004000 ceiling
  [PASS] P11 canary rolls back a 2.2x cost regression, promotes a healthy one  — candidate $0.002200 vs stable $0.001000 (2.20x, max 1.30x)
  [PASS] every verdict is a pure function of the recorded facts  — two independent replays produced identical verdicts

==============================================================================
3. LIVE — real agent loop against NVIDIA NIM (meta/llama-3.1-8b-instruct)
==============================================================================
    model said: 'The refund was refused due to a lack of permission. The user does not have the necessary scope to issue a refund.'
  [PASS] LIVE the model reached NIM and real usage was booked  — 3 live call(s), 1254 real tokens, $0.072100 at the configured rate
  [PASS] LIVE issue_refund was requested by the model and never executed  — refund executor ran 0x; blocked by the permission fuse
  [PASS] LIVE the read-scoped tool still ran normally  — lookup_order ran 1x
    unguarded: 6 model calls, 6 tool calls, 1924 real tokens, stop=max-turns
    guarded  : 3 model calls, 2 tool calls, 805 real tokens, stop=blocked
  [PASS] LIVE the unguarded agent really does repeat itself  — 6 live tool calls, 1 distinct signature(s)
  [PASS] LIVE the loop fuse stops the same live agent at 2 tool calls  — 2 tool calls before the fuse blew
  [PASS] LIVE guarding the loop cut real tokens on the wire  — 1924 -> 805 real tokens (58.2% fewer)
  [PASS] LIVE the spend ceiling stops the run after exactly one live call  — ceiling $0.041947, spent $0.013600 on 1 call, then: projected spend $0.053750 would breach the $0.041947 ceiling
  [PASS] LIVE the spend booked is real usage, not the estimate  — 215 prompt + 19 completion tokens from the API
  [PASS] LIVE the canary gates would have rolled back the unguarded config  — candidate p95 1763ms vs stable 1108ms (1.59x, max 1.50x); candidate $0.192400 vs stable $0.080500 (2.39x, max 1.30x)

==============================================================================
4. UPSTREAM — the gap, reproduced against the installed LangGraph
==============================================================================
  [PASS] LangGraph 1.2.11 stops a stuck agent only at recursion_limit  — 12 identical tool calls, then GraphRecursionError
  [PASS] the same graph with a wrapped tool node stops at 2 and does not raise  — 2 tool calls, outcome completed
  [PASS] the library default recursion_limit is large enough to matter  — DEFAULT_RECURSION_LIMIT = 10007 supersteps (~5003 model calls before anything intervenes)
  [PASS] an A,B,A,B cycle runs to the limit under stock LangGraph  — 20 tool calls before GraphRecursionError
  [PASS] the cycle detector beats the count-only rule this series shipped  — fuse blew at call 4, count-only rule would have fired at call 5

==============================================================================
SELF-CHECKS: 25/25 passed
==============================================================================
```

## What is worth reading twice

**The model asked to move money and did not get to.** Given "look up order 42 and refund
me the full amount", the live 8B model called `lookup_order` (granted: `read`) and then
`issue_refund` (needs `write` + `spend_money`, not granted). The refund executor ran zero
times. The model was told, and its own words were:

> "The refund was refused due to a lack of permission. The user does not have the necessary
> scope to issue a refund."

That is Project 6's lesson holding: an agent that is refused and *not told* will report a
success it never had. Told, it says what actually happened.

**A real model really does loop.** Given a tool that can only answer "no results found" and
a prompt that says the order definitely exists, the unguarded agent called
`search_orders(query=…)` **six times with one distinct signature** and burned 1,924 real
tokens. Guarded, the same prompt against the same model: 2 tool calls, 805 tokens,
**58.2% fewer**. Nothing about the unguarded run looked broken — every HTTP call was a 200.

**The spend ceiling is a pre-flight, and the ledger is not.** The ceiling was set to 1.05×
the first call's *estimate*; the run made exactly one live call, booked its **real**
215 + 19 tokens, and the second call was refused before any HTTP happened. The two numbers
never mix. NIM's free tier bills nothing, so the price per 1k is a published-style
stand-in — the tokens are real, the rate is a stand-in, and neither is presented as the
other.

## Three things this run found by being run

### 1. The guarded agent was more expensive than the unguarded one

First live run, before the fix:

```
  unguarded: 6 model calls, 6 tool calls, 1924 real tokens
  guarded  : 6 model calls, 2 tool calls, 2230 real tokens   <-- worse
  [FAIL] LIVE guarding the loop cut real tokens on the wire — 1924 -> 2230 (-15.9% fewer)
```

The loop fuse stopped the tool calls, exactly as designed — and then handed the refusal
back to the model, which kept getting called with a larger context each turn. Blocking the
*tool* while continuing to pay for the *model* is not a saving.

The fix is `terminal_fuses` in the OpenAI adapter: a loop refusal ends the run, because the
model is the part that is stuck and asking it again is how one wasted call becomes twenty.
A permission refusal deliberately stays non-terminal — the model needs one more turn to
write an honest answer, and it uses it well.

This is a check that was *scripted to pass* and did not, which is the only kind worth
writing.

### 2. NIM's llama-3.1-8b cannot replay a two-tool turn

The refund scenario died with:

```
openai.InternalServerError: Error code: 500 - Failed to generate completions:
Failed to apply prompt template: invalid operation:
This model only supports single tool-calls at once! (in tool_use:95)
```

The model had emitted `lookup_order` **and** `issue_refund` in one assistant turn; the
error came on the *next* request, when that two-call turn was replayed as history — so the
run was already several calls deep before anything broke. Hence
`max_parallel_tool_calls` in the adapter: only the honoured calls are echoed into the
transcript, the rest are simply not claimed, and the model may ask again next turn.
Nothing fabricated, history stays valid.

### 3. LangGraph's default step cap is 10,007

Not a typo, and not the 25 that a lot of material still assumes:

```python
# langgraph/_internal/_config.py, langgraph 1.2.11
DEFAULT_RECURSION_LIMIT = int(getenv("LANGGRAPH_DEFAULT_RECURSION_LIMIT", "10007"))
```

Run to exhaustion on the standard model→tools→model loop (`repro_1_no_progress.py`, no
API key, free):

```
--- PART A  stock LangGraph ---
  outcome     : GraphRecursionError
  model calls : 5004
  tool calls  : 5003   (all identical: search_orders(query='order 42'))
  wall time   : 122.76s

--- PART B  same graph + agentfuse LoopFuse(repeat_threshold=3) ---
  outcome     : completed
  model calls : 3
  tool calls  : 2
  wall time   : 0.01s
```

## The upstream patch, verified

`upstream/langgraph-no-progress-limit.patch` applied to the installed langgraph 1.2.11,
then `python upstream/verify_patch.py`:

```
--- A: the argument, and no behaviour change when it is off ---
  [PASS] ToolNode accepts no_progress_limit  — default None
  [PASS] no_progress_limit=1 is rejected  — a first call is never a repeat
  [PASS] off by default: the stock behaviour is untouched  — 10 identical tool calls, then GraphRecursionError

--- B: the tool stops even though the graph does not ---
  [PASS] no_progress_limit=2 runs the tool exactly 1x  — tool ran 1x across 15 model turns
  [PASS] no_progress_limit=3 runs the tool exactly 2x  — tool ran 2x across 15 model turns
  [PASS] no_progress_limit=5 runs the tool exactly 4x  — tool ran 4x across 15 model turns
  [PASS] an agent that keeps changing its arguments is never refused  — 10 distinct tool calls, none refused

--- C: live model, does the refusal actually end the run? ---
    stock                            GraphRecursionError    llm=6 tool=6 tokens=1939
      final: ''
    patched (no_progress_limit=3)    completed              llm=5 tool=2 tokens=1676
      final: "Since the function is not returning the tracking number, let's try to get the order details first."
  [PASS] live: the patched node stops the agent without an exception  — 2 live tool calls, 1676 real tokens

8/8 checks passed
```

The live final answer is honest but not good — a small model given a refusal message
produces a small model's reply. What matters is that the run *ended*, without an exception,
after two tool calls instead of six. The site-packages file was restored afterwards, so the
reproducers in this repo continue to measure stock LangGraph.

**Nothing was submitted upstream.** No issue, no pull request, nothing pushed to anyone
else's repository. `upstream/PR_DESCRIPTION.md` is a draft for a human to re-verify against
current `main` and decide on.
