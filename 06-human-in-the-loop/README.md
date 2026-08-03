# 06 — Human-in-the-Loop Approval Agent

**Goal:** an agent that **knows when to stop and ask a human**. Safe, confident actions
run on their own; risky or uncertain ones **pause**, surface an approve / deny / edit
request, and only **resume** once a human answers — and every proposal, verdict,
decision, and outcome lands in a durable **audit trail**.

The model proposes the plan and rates its own confidence, and writes the final note.
**Everything that decides or records is deterministic Python** — the risk gate, the
pause/resume, and the audit log. The model is never trusted to wave its own risky
action through.

## The four ideas (hand-rolled, no framework)

| # | Sub-point | Where it lives |
|---|-----------|----------------|
| 1 | **uncertainty detection** | [`approval.py`](approval.py) `ApprovalGate.assess` — one verdict from **three independent signals**: the tool's **declared risk** (`risk="high"`), a **sensitive-capability** check (`send_email` / `spend_money` / `delete` / `external_write`), and the **model's self-rated confidence** vs a threshold. Any one trips it. |
| 2 | **pause for human input** | [`approval.py`](approval.py) `ApprovalRequest` + `DecisionSource` — on a "needs approval" verdict the agent **suspends**, emits the request (action + args + why), and asks a source for a `Decision`. Two sources: `InteractiveDecisionSource` (a real CLI prompt) and `ScriptedDecisionSource` (a preset list, so the recorded run reproduces). |
| 3 | **resume with validated context** | [`agent.py`](agent.py) `HITLAgent.run` — on **approve/edit** it executes, **merging any human-edited args** first; on **deny** it records the refusal and **carries on gracefully**. The decision (who + why + edits) is threaded into the running context handed to the final-reply call. |
| 4 | **full audit trail** | [`approval.py`](approval.py) `AuditLog` — every proposal, risk verdict, human decision (who / what / with a **monotonic seq counter**, no wall clock), and outcome is appended to a **JSONL** file and printed as a readable trail. |
| — | **escalation when no one responds** | [`approval.py`](approval.py) `ScriptedDecisionSource.decide` returns `None` when nobody is left to answer; [`agent.py`](agent.py) turns that into a **safe-default deny** and audits it as escalated. |

## The gate — the one decision that is never the model's

```
  proposed action  (tool, args, model-confidence)
        │
        ▼
  ApprovalGate.assess ─── risk="high" ?            ┐
                     ├─── touches a sensitive cap? ├─ ANY true → NEEDS APPROVAL
                     └─── confidence < min?        ┘        else → auto-approve
        │
   ┌────┴─────────────────────────┐
   ▼                              ▼
 auto-approve                 NEEDS APPROVAL
   │                              │  SUSPEND → emit ApprovalRequest
   ▼                              ▼  ask DecisionSource (CLI or scripted)
 execute now            ┌──── approve / edit ────┐         deny ────┐
   │                    ▼                        │                 ▼
   │            merge human edits → execute       │        record refusal, carry on
   └──────────────┬──────────────────────────────┴─────────────────┘
                  ▼
       append every event to the JSONL audit trail (monotonic seq)
```

- **Safe** = a read like `lookup_order` — auto-runs.
- **High** = `send_email` (sends to a customer) or `issue_refund` (moves money) — never
  runs without a human, even if the model is fully confident.
- **Uncertain** = a *safe* tool the model rated **below** `min_confidence` still pauses —
  low confidence is treated as risk.

## Why the model never approves itself

The actor is `meta/llama-3.1-8b-instruct` — fast and free on the NIM tier, but small.
It gets exactly two jobs: **plan** the task into proposed actions + **rate its own
confidence** (the uncertainty signal), and **write the final note** once outcomes are
known. It does **not** decide approvals, and it **cannot** run a high-risk tool on its
own — the gate is pure Python keyed off the tool's declared risk, so the same proposal
always earns the same verdict. That separation is the whole safety property: an 8B model
guessing an email body or a refund amount can propose, but a human holds the trigger on
anything that emails a customer or moves money.

> Live example from the recorded run: the model addressed the email to `"Priya"` (a
> name) instead of the `priya.menon@example.com` address it had *just* looked up — a
> small but real slip. That is *exactly* the kind of thing the pause is for; a human
> reviewing the request can catch and edit it before it goes out.

## What the demo shows ([`run.py`](run.py)) — one scenario, two ways to answer

**Task:** *"Customer Priya emailed about order A-1042 — it arrived damaged. Look up the
order, email her a short apology, and refund her the $60 shipping fee."*

The model plans three steps; the gate routes each:

1. `lookup_order` — **safe** → **auto-approved**, runs immediately.
2. `send_email` — **high** → **PAUSES** → (scripted) **APPROVED with a human edit** to the
   subject → resumes and sends.
3. `issue_refund` — **high** → **PAUSES** → (scripted) **DENIED** → not run; the refusal is
   carried into the final reply and **no money moves**.

```bash
# from the repo root, with .venv active and NVIDIA_API_KEY set in .env
python 06-human-in-the-loop/run.py               # SCRIPTED (default) — reproducible
python 06-human-in-the-loop/run.py interactive   # you answer each pause at the CLI
```

- **scripted** — the two pauses are answered from a preset list (`APPROVE-with-edit`,
  then `DENY`), so the run reproduces byte-for-byte apart from the model's own wording.
- **interactive** — the same run, but each pause is a real `[a]pprove / [d]eny / [e]dit`
  prompt you answer, with your name recorded as the approver.

See [`recorded-run.md`](recorded-run.md) for a **real** captured transcript against
NVIDIA NIM — the plan call and the final-reply call are both live `HTTP/1.1 200 OK` to
`integrate.api.nvidia.com`, the two pauses fire, one is approved and one denied, and the
full JSONL audit trail is included.

## Files

- `tools.py` — three tools that **declare** their risk + capabilities: `lookup_order`
  (safe), `send_email` (high), `issue_refund` (high), plus the inspectable side-effect
  sinks (outbox / refunds ledger).
- `approval.py` — the deterministic half: the `ApprovalGate` (uncertainty detection),
  `ApprovalRequest` + `Decision` + the two `DecisionSource`s (pause), and the `AuditLog`
  (JSONL + readable render).
- `agent.py` — the `HITLAgent`: the model-backed planner (plan + confidence), the
  per-step assess → run/pause → resume loop, and the model-backed final reply.
- `run.py` — the runnable scenario in `scripted` / `interactive` modes.
- `recorded-run.md` — a real transcript hitting NVIDIA NIM (incl. the audit JSONL).
- `audit-log.jsonl` — runtime audit trail written by `run.py` (gitignored; regenerated).

## Note on the model

Two model calls per run — **plan + confidence** and **final reply** — and nothing else.
The **risk verdict**, the **pause**, the **approve/deny/edit surface**, the **arg merge on
resume**, and the **audit trail** are all deterministic Python. That is deliberate: the
whole value of a human-in-the-loop agent is that the risky decision is *taken away* from
the model and given to a human, with an auditable record of who decided what.
