# 01 — Structured Output Agent

**Goal:** an agent whose every output is a validated, typed object — never free text you have to parse.

## The four ideas (hand-rolled, no framework)

| # | Sub-point | Where it lives |
|---|-----------|----------------|
| 1 | **Enforce a Pydantic JSON schema** | `Invoice` in [`schemas.py`](schemas.py); parsed via `Invoice.model_validate_json(...)` in [`agent.py`](agent.py) |
| 2 | **Validate tool responses** | `get_tax_rate` tool output validated against `TaxRateResult` before use (`resolve_tax`) |
| 3 | **Retry on parse errors** | the parse→re-prompt loop in `extract_invoice` (bounded, `max_retries=3`) |
| 4 | **Log validation failures** | structured `logging` on every rejected attempt (`Attempt` records + logs) |

## Why it retries

The `Invoice` schema is deliberately strict:

- `invoice_id` must match `^INV-\d{6}$` — **zero-padded to 6 digits**
- `tax_rate` is a **fraction** in `[0, 1]` (`0.08`, never `8`)
- `subtotal` must equal `sum(quantity * unit_price)`
- `total` must equal `subtotal * (1 + tax_rate)`

The first prompt lists the field *names* but withholds these strict rules — they
live only in the schema. So the model's early attempts violate them: in the
[recorded run](recorded-run.md) attempt 1 returns `invoice_id` as the JSON number
`4521` (fails the string/pattern rule), and attempt 2 fixes the id to
`"INV-004521"` but gets the `total` arithmetic wrong (caught by the cross-field
`model_validator`). Each time, the agent appends the exact Pydantic error to the
conversation and re-prompts; the model self-corrects and lands a valid object on
attempt 3. That is the whole lesson: **the schema is the contract, and validation
errors are the teaching signal.**

## Run it

```bash
# from the repo root, with .venv active and NVIDIA_API_KEY set in .env
python 01-structured-output/run.py
```

You'll see: the tool response being validated, attempt #1 rejected with its raw
output and error, the retry, and the final typed `Invoice`.

See [`recorded-run.md`](recorded-run.md) for a real captured transcript against
NVIDIA NIM.

## Files

- `schemas.py` — the Pydantic models (`Invoice`, `LineItem`, `TaxRateResult`, `Currency`)
- `agent.py` — the agent: tool round-trip + the parse→retry loop
- `run.py` — a runnable demo engineered to trigger a retry
- `recorded-run.md` — a real transcript hitting NVIDIA NIM
