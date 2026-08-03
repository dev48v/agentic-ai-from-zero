# Recorded run — Human-in-the-Loop Approval Agent

A **real** transcript captured by running the scripted demo:

```bash
python 06-human-in-the-loop/run.py scripted
```

- **Provider / endpoint:** NVIDIA NIM — `POST https://integrate.api.nvidia.com/v1/chat/completions` (both the plan call and the final-reply call returned `HTTP/1.1 200 OK` in the live `httpx` log below).
- **Model:** `meta/llama-3.1-8b-instruct` (warm on the free tier).
- **Date:** 2026-08-04.
- **Gate policy:** `high-risk OR sensitive-capability OR model-confidence < 0.60 → PAUSE for a human`.
- **What the run proves:** one **safe auto-approved** action, one **high-risk action approved with a human edit**, one **high-risk action denied** — the pauses fire, the agent resumes on each decision, and every event is written to a durable JSONL audit trail.

The model does exactly two things — **plan the task + rate its own confidence**, and
**write the final note**. The risk gate, the pause, the approve/deny/edit, and the audit
trail are deterministic Python. Everything below is verbatim from the run.

---

## Task

> Customer Priya emailed about order A-1042 — it arrived damaged and she's upset. Look up
> the order, email her a short apology, and refund her the $60 shipping fee.

The model planned **3 actions** (one NIM call, `HTTP/1.1 200 OK`), self-rating each:

```
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
INFO  hitl-agent  planned 3 action(s): [('lookup_order', 1.0), ('send_email', 0.9), ('issue_refund', 0.8)]
```

---

## Step 1 — `lookup_order` — SAFE → auto-approved (no pause)

```
✅ step 1 `lookup_order` — SAFE, auto-approved (confidence 1.00) → A-1042: Priya Menon
   <priya.menon@example.com> — Ceramic pour-over coffee set, total $84.00, shipping
   $60.00, status: delivered — reported damaged
```

The gate saw `risk=safe`, no sensitive capability, and confidence `1.00 ≥ 0.60`, so it
**auto-approved** — the agent just ran it. This is the "don't bother a human for a
read-only lookup" path.

## Step 2 — `send_email` — HIGH → PAUSE → APPROVED with a human edit → resume

```
⏸  PAUSE — step 2: agent wants to call `send_email` and is asking a human.
   args    : {'to': 'Priya', 'subject': 'Apology for Damaged Order A-1042', 'body': 'Dear
             Priya, we apologize for the damage to your order A-1042. We will process a
             refund for the shipping fee as soon as possible. Thank you for your patience.'}
   why ask : tool 'send_email' declares risk=high; touches sensitive capability
             ['external_write', 'send_email']
   model reason: Notify customer of apology and refund
▶️  step 2 `send_email` — APPROVED (with edit) by Devanshu → RESUMED → email #1 sent to
   Priya — subject: “Our sincere apology for your damaged order A-1042”
```

The agent **suspended** and asked. The scripted human answered **edit** (an approve that
carries a patch): they kept the body but rewrote the subject. On resume the agent
**merged the human's edit** (`subject`) over the model's args and executed — the sent
email carries the human's subject line, not the model's.

> Note the model addressed the email `to: "Priya"` — a name, not the
> `priya.menon@example.com` address it had *just* looked up. A small, real slip, left in
> to be honest: it's precisely what a human reviewing the pause is there to catch.

## Step 3 — `issue_refund` — HIGH → PAUSE → DENIED → not run, task continues

```
⏸  PAUSE — step 3: agent wants to call `issue_refund` and is asking a human.
   args    : {'order_id': 'A-1042', 'amount': 60}
   why ask : tool 'issue_refund' declares risk=high; touches sensitive capability
             ['spend_money']
   model reason: Refund shipping fee
🚫 step 3 `issue_refund` — DENIED by Devanshu → not run (“refunds over $50 need manager
   sign-off — withholding pending review”)
```

The agent suspended again. The scripted human answered **deny** with a reason. The agent
**recorded the refusal and carried on gracefully** — `issue_refund` was never called, so
**no money moved**.

---

## The agent's final reply (a second NIM call, `HTTP/1.1 200 OK`)

The final note is generated from the *outcomes* — including the human decisions — so the
denial is carried forward honestly:

```
INFO  httpx  HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"

Summary: The damaged order A-1042 was looked up and an apology email was sent to Priya.
However, the refund for the $60 shipping fee was withheld due to the need for manager
sign-off for refunds over $50.

Next step: I recommend escalating the refund request to the manager for review and
approval to complete the refund process.
```

It never claims the refund happened — the refusal survived from the gate all the way
into what the agent says.

## Side effects — what actually touched the outside world

```
📧 outbox: 1 email(s) sent
   → to Priya — “Our sincere apology for your damaged order A-1042”
💸 refunds ledger: 0 refund(s) issued  (the $60 refund was DENIED — money was NOT moved)
```

One email sent (with the human's edited subject); **zero** refunds — the denied action
left no trace on the money ledger.

---

## The full audit trail (durable JSONL — `audit-log.jsonl`)

Every proposal, risk verdict, pause, human decision, and outcome, each with a monotonic
`seq` counter (no wall clock, so it's reproducible). This is the exact file written by
the run:

```json
{"seq": 1, "step": 1, "event": "proposed", "tool": "lookup_order", "args": {"order_id": "A-1042"}, "reason": "Verify order details", "confidence": 1.0}
{"seq": 2, "step": 1, "event": "risk_verdict", "tool": "lookup_order", "requires_approval": false, "risk": "safe", "confidence": 1.0, "signals": []}
{"seq": 3, "step": 1, "event": "auto_approved", "tool": "lookup_order"}
{"seq": 4, "step": 1, "event": "executed", "tool": "lookup_order", "args": {"order_id": "A-1042"}, "ok": true, "output": "A-1042: Priya Menon <priya.menon@example.com> — Ceramic pour-over coffee set, total $84.00, shipping $60.00, status: delivered — reported damaged"}
{"seq": 5, "step": 2, "event": "proposed", "tool": "send_email", "args": {"to": "Priya", "subject": "Apology for Damaged Order A-1042", "body": "Dear Priya, we apologize for the damage to your order A-1042. We will process a refund for the shipping fee as soon as possible. Thank you for your patience."}, "reason": "Notify customer of apology and refund", "confidence": 0.9}
{"seq": 6, "step": 2, "event": "risk_verdict", "tool": "send_email", "requires_approval": true, "risk": "high", "confidence": 0.9, "signals": ["tool 'send_email' declares risk=high", "touches sensitive capability ['external_write', 'send_email']"]}
{"seq": 7, "step": 2, "event": "paused", "tool": "send_email", "why": ["tool 'send_email' declares risk=high", "touches sensitive capability ['external_write', 'send_email']"]}
{"seq": 8, "step": 2, "event": "human_decision", "tool": "send_email", "decision": "edit", "approver": "Devanshu", "reason": "apology is fine — tightening the subject line before it goes out", "edited_args": {"subject": "Our sincere apology for your damaged order A-1042"}, "escalated": false}
{"seq": 9, "step": 2, "event": "executed", "tool": "send_email", "args": {"to": "Priya", "subject": "Our sincere apology for your damaged order A-1042", "body": "Dear Priya, we apologize for the damage to your order A-1042. We will process a refund for the shipping fee as soon as possible. Thank you for your patience."}, "ok": true, "output": "email #1 sent to Priya — subject: “Our sincere apology for your damaged order A-1042”"}
{"seq": 10, "step": 3, "event": "proposed", "tool": "issue_refund", "args": {"order_id": "A-1042", "amount": 60}, "reason": "Refund shipping fee", "confidence": 0.8}
{"seq": 11, "step": 3, "event": "risk_verdict", "tool": "issue_refund", "requires_approval": true, "risk": "high", "confidence": 0.8, "signals": ["tool 'issue_refund' declares risk=high", "touches sensitive capability ['spend_money']"]}
{"seq": 12, "step": 3, "event": "paused", "tool": "issue_refund", "why": ["tool 'issue_refund' declares risk=high", "touches sensitive capability ['spend_money']"]}
{"seq": 13, "step": 3, "event": "human_decision", "tool": "issue_refund", "decision": "deny", "approver": "Devanshu", "reason": "refunds over $50 need manager sign-off — withholding pending review", "edited_args": null, "escalated": false}
{"seq": 14, "step": 3, "event": "refused", "tool": "issue_refund", "reason": "refunds over $50 need manager sign-off — withholding pending review"}
{"seq": 15, "step": 0, "event": "final_response", "text": "Summary: The damaged order A-1042 was looked up and an apology email was sent to Priya. However, the refund for the $60 shipping fee was withheld due to the need for manager sign-off for refunds over $50. \n\nNext step: I recommend escalating the refund request to the manager for review and approval to complete the refund process."}
```

Readable render (also printed by `run.py`):

```
#1  [proposed]      step 1: lookup_order args={"order_id": "A-1042"} (model confidence 1.00)
#2  [risk_verdict]  step 1: lookup_order → auto-approve (risk=safe); signals: none
#3  [auto_approved] step 1: lookup_order — safe, ran without asking
#4  [executed]      step 1: lookup_order → OK: A-1042: Priya Menon <priya.menon@example.com> ...
#5  [proposed]      step 2: send_email args={"to": "Priya", ...} (model confidence 0.90)
#6  [risk_verdict]  step 2: send_email → NEEDS APPROVAL (risk=high); signals: declares risk=high; sensitive ['external_write', 'send_email']
#7  [paused]        step 2: send_email — SUSPENDED, awaiting a human
#8  [human_decision]step 2: send_email — EDIT by Devanshu — "tightening the subject line" edited_args={"subject": "Our sincere apology for your damaged order A-1042"}
#9  [executed]      step 2: send_email → OK: email #1 sent to Priya — subject: "Our sincere apology..."
#10 [proposed]      step 3: issue_refund args={"order_id": "A-1042", "amount": 60} (model confidence 0.80)
#11 [risk_verdict]  step 3: issue_refund → NEEDS APPROVAL (risk=high); signals: declares risk=high; sensitive ['spend_money']
#12 [paused]        step 3: issue_refund — SUSPENDED, awaiting a human
#13 [human_decision]step 3: issue_refund — DENY by Devanshu — "refunds over $50 need manager sign-off"
#14 [refused]       step 3: issue_refund — NOT run: refunds over $50 need manager sign-off — withholding pending review
#15 [final_response]agent's final reply recorded
```

---

## The four sub-points, together

| Sub-point | Evidence in this run |
|-----------|----------------------|
| **Uncertainty detection** | Each step earned a `risk_verdict` (seq #2, #6, #11) from declared risk + sensitive-capability + model confidence. `lookup_order` → auto (no signals); `send_email` and `issue_refund` → NEEDS APPROVAL (`risk=high` + a sensitive capability). |
| **Pause for human input** | `send_email` and `issue_refund` both **SUSPENDED** (seq #7, #12) and asked a human — scripted here (`APPROVE-with-edit`, then `DENY`), a live CLI prompt in interactive mode. |
| **Resume with validated context** | The email **resumed** with the human's edited `subject` **merged** over the model's args (seq #8 → #9); the denial (seq #13/#14) was **carried forward** into the final reply, which never claims the refund happened. |
| **Full audit trail** | 15 events in `audit-log.jsonl`, each with a monotonic `seq` and the approver on every human decision — proposal → verdict → pause → decision → outcome, end to end. |

> Note: the model is small (8B) and not perfectly deterministic even at temperature 0,
> so a re-run may phrase the email body, the confidences, or the final note slightly
> differently. What is **stable**: the gate is pure Python, so the same proposal always
> earns the same verdict — a high-risk tool *always* pauses and can *never* run without a
> human — and the scripted decisions replay in the same order, so the shape of the run
> (auto → approve-with-edit → deny) is reproducible.
