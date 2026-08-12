# prebuilt: add optional `no_progress_limit` to `ToolNode`

> **This has not been submitted.** It is a draft for a human to review, re-verify against
> the current `main`, and decide whether to open. Numbers below were measured against
> langgraph 1.2.11 on 2026-08-12 and should be re-measured before posting.

## Summary

Adds one optional argument to `ToolNode`:

```python
ToolNode(tools, no_progress_limit=3)
```

When set, a tool call whose `(name, args)` signature has already executed
`no_progress_limit - 1` times **in this request** is refused instead of run, and returns a
`ToolMessage(status="error")` saying so. Default `None` — off, no behaviour change.

## Motivation

`recursion_limit` caps how many steps a graph may take. It cannot tell ten useful steps
from the same step ten times, and it only fires once the work is already paid for.

A two-node graph whose model keeps asking `search_orders(order_id="42")` while the tool
keeps answering "no results" (`repro_1_no_progress.py`):

| `recursion_limit` | model calls | identical tool calls | outcome |
|---|---|---|---|
| 25 | 13 | 12 | `GraphRecursionError` |
| default (`10007`) | 5,004 | 5,003 | `GraphRecursionError` |

With a live `meta/llama-3.1-8b-instruct` and `recursion_limit=12`, the same shape costs
6 identical tool calls and 2,029 real tokens before the exception. It is not a strawman
failure: a small model told "the order definitely exists, do not give up" will do exactly
this, and every call in the sequence returns HTTP 200.

`wrap_tool_call` already makes this implementable in user code, and that is a good hook.
But the obvious user implementation is wrong in a way that is hard to notice: a `ToolNode`
instance is shared across every request the process serves, so a repeat counter stored on
it leaks one request's loop into another's. Deriving the history from the transcript — and
excluding calls whose `ToolMessage` is an error, because those never ran — is the correct
version, and it seems worth shipping once rather than having everyone rediscover it.

## What the patch does

* new keyword-only argument `no_progress_limit: int | None = None`;
* `ValueError` for values below 2 — a first call is never a repeat;
* `ToolNode._executed_signatures(input)` reads the message history and returns the
  signatures of tool calls that actually ran (matching `ToolMessage`, not `status="error"`);
* `ToolNode._call_signature(name, args)` is `json.dumps(args, sort_keys=True, default=str)`,
  so argument reordering cannot disguise a repeat and an unserialisable argument degrades
  instead of raising;
* `_func` and `_afunc` consult `_no_progress_refusals(...)` before building the executor;
  if any pending call is over the limit, the whole turn is refused, all-or-nothing —
  half-running a batch would leave the repeat history describing work that never happened;
* no new dependency; `json` is already imported in the module.

## Scope, stated plainly

This bounds **tool execution**, not graph execution. The graph may keep cycling between
the model and a node that now refuses instantly; the model is told why and can stop, and
`recursion_limit` remains the outer backstop. Stopping the side-effecting, billable half
is the part `recursion_limit` cannot do.

If maintainers would rather this lived in the Pregel loop so it can end the graph the way
`recursion_limit` does, that is a bigger change and I am happy to take it there instead —
this patch is deliberately the small, opt-in version.

## Testing

`verify_patch.py` — 8/8 against a patched langgraph 1.2.11:

* argument present, default `None`, `no_progress_limit=1` rejected;
* **off by default the stock behaviour is unchanged** (10 identical calls, then
  `GraphRecursionError`);
* `no_progress_limit` of 2 / 3 / 5 runs the tool exactly 1 / 2 / 4 times, regardless of how
  many turns the graph then takes;
* an agent that keeps changing its arguments is never refused;
* live NIM agent: stock ends in `GraphRecursionError` after 6 tool calls / 1,939 tokens;
  patched finishes normally after 2 tool calls / 1,676 tokens.

Happy to add these to `libs/langgraph/tests/test_prebuilt.py` in the project's own style if
the direction is welcome.

## Open questions for maintainers

1. `ToolNode` or `create_agent`/middleware — where do you want this to live?
2. Should the refusal also be surfaced as a typed exception (`NoProgressError`) for callers
   who would rather fail closed than hand the model an error message?
3. Is a default other than `None` ever acceptable, given `DEFAULT_RECURSION_LIMIT` is 10007?
