# Recorded run — Structured Output Agent

A **real** captured run against **NVIDIA NIM** (no mock, no replay). It shows the
tool response being validated, the parse→retry loop firing **twice** on two
distinct failure modes, and the final typed `Invoice` landing valid.

| | |
|---|---|
| **Date** | 2026-07-31 11:38 IST |
| **Provider** | NVIDIA NIM — `https://integrate.api.nvidia.com/v1` (OpenAI-compatible) |
| **Model** | `meta/llama-3.1-8b-instruct` |
| **Temperature** | `0` (deterministic) |
| **max_retries** | `3` |
| **Total elapsed** | ~2.6 s (4 HTTP calls: 1 tool + 3 extraction attempts) |

**Evidence it hit NIM** — every call returned `HTTP/1.1 200 OK` from the live
endpoint (httpx log):

```
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"   # tool call
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"   # attempt 1
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"   # attempt 2
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"   # attempt 3
```

---

## The prompt (source text)

```
Invoice 4521 for customer billing@acme-corp.com (California, US-CA).
They ordered 3 units of the Widget Pro at $12.50 each, billed in US dollars.
Apply the standard California sales tax.
```

Region passed to the tool: `US-CA`.

---

## Step 1 — resolve tax via tool (response validated against `TaxRateResult`)

The model issued a real `get_tax_rate` tool call; the tool's raw dict was then
validated against the `TaxRateResult` schema **before** the agent trusted it:

```
INFO  structured_output_agent  tool get_tax_rate({'region': 'US-CA'}) -> {'region': 'US-CA', 'rate': 0.0825, 'source': 'static-tax-table-v1'} (via model tool-call)
validated tool response: {'region': 'US-CA', 'rate': 0.0825, 'source': 'static-tax-table-v1'}
```

`rate = 0.0825` passes the `0 <= rate <= 1` bound (it is a fraction, not `8.25`).

---

## Step 2 — extract `Invoice` (enforce schema, retry on parse errors)

### Attempt 1 — REJECTED (type error)

Raw model output:

```json
{
  "invoice_id": 4521,
  "customer_email": "billing@acme-corp.com",
  "currency": "USD",
  "line_items": [
    { "description": "Widget Pro", "quantity": 3, "unit_price": 12.5 }
  ],
  "subtotal": 37.5,
  "tax_rate": 0.0825,
  "total": 42.1875
}
```

Pydantic `ValidationError` (logged):

```
attempt 1/3 failed: invoice_id: Input should be a valid string
```

`invoice_id` came back as the JSON **number** `4521`, not a `^INV-\d{6}$` string.
The agent appended this exact error to the conversation and re-prompted.

### Attempt 2 — REJECTED (cross-field math)

Raw model output:

```json
{
  "invoice_id": "INV-004521",
  "customer_email": "billing@acme-corp.com",
  "currency": "USD",
  "line_items": [
    { "description": "Widget Pro", "quantity": 3, "unit_price": 12.5 }
  ],
  "subtotal": 37.5,
  "tax_rate": 0.0825,
  "total": 42.1875
}
```

Pydantic `ValidationError` (logged):

```
attempt 2/3 failed: : Value error, total 42.1875 must equal subtotal*(1+tax_rate) = 40.59
```

It fixed `invoice_id` (now correctly zero-padded to `INV-004521`, every digit of
4521 preserved), but `total` was still wrong — the `model_validator` cross-field
arithmetic check caught it. Re-prompted again with the computed expected value.

### Attempt 3 — VALID

```
INFO  structured_output_agent  valid invoice on attempt 3
```

---

## Final — validated, typed `Invoice`

```json
{
  "invoice_id": "INV-004521",
  "customer_email": "billing@acme-corp.com",
  "currency": "USD",
  "line_items": [
    {
      "description": "Widget Pro",
      "quantity": 3,
      "unit_price": 12.5
    }
  ],
  "subtotal": 37.5,
  "tax_rate": 0.0825,
  "total": 40.5925
}
```

```
attempts used : 3
elapsed       : 2.61s
type          : Invoice (Pydantic-validated)
```

`37.5 × (1 + 0.0825) = 40.59375`, which rounds to `40.59`; the model's `40.5925`
is within the one-cent tolerance the validator allows, so it passes.

---

## What this run demonstrates (the 4 sub-points)

1. **Enforce a Pydantic JSON schema** — the final object is a real `Invoice`
   instance, not text; every field satisfied its type/pattern/range constraint.
2. **Validate tool responses** — `get_tax_rate`'s output was validated against
   `TaxRateResult` before use.
3. **Retry on parse errors** — two `ValidationError`s triggered bounded
   re-prompts (type error, then a cross-field math error) that the model
   self-corrected, landing valid on attempt 3 of 3.
4. **Log validation failures** — each rejected attempt was structured-logged with
   its attempt number, the exact error, and the raw model output.
