"""Runnable demo for the Human-in-the-Loop Approval Agent (NVIDIA NIM).

One scenario, two ways to answer the pauses:

  python 06-human-in-the-loop/run.py            # SCRIPTED (default) — reproducible
  python 06-human-in-the-loop/run.py scripted   # same as above
  python 06-human-in-the-loop/run.py interactive # you answer each pause at the CLI

Scenario: a customer reports a damaged order and asks for a refund. The agent plans:
  1. lookup_order   — SAFE  → auto-approved, runs immediately.
  2. send_email     — HIGH  → PAUSES → (scripted) APPROVED with a human edit → sent.
  3. issue_refund   — HIGH  → PAUSES → (scripted) DENIED → not run, task continues.

The model plans the steps + rates its confidence, and writes the final note. The gate,
the pauses, the approve/deny/edit, and the audit trail are deterministic Python — so
the scripted run reproduces: one safe auto action, one approval, one denial, and a full
JSONL audit trail printed at the end.
"""

from __future__ import annotations

import logging
import os
import sys

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _REPO_ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from agent import HITLAgent, RunResult, StepOutcome  # noqa: E402
from approval import (  # noqa: E402
    ApprovalGate, ApprovalRequest, AuditLog, Decision,
    InteractiveDecisionSource, ScriptedDecisionSource,
)
import tools  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    stream=sys.stdout,
)

AUDIT_PATH = os.path.join(_PROJECT_DIR, "audit-log.jsonl")

TASK = (
    "Customer Priya emailed about order A-1042 — it arrived damaged and she's upset. "
    "Look up the order, email her a short apology, and refund her the $60 shipping fee."
)

# The pre-baked human decisions the SCRIPTED source hands back, in the order the gate
# pauses. First pause (the email) is APPROVED with a small human edit to the subject;
# second pause (the refund) is DENIED pending manager sign-off.
SCRIPTED_DECISIONS = [
    Decision(
        verdict="edit", approver="Devanshu",
        reason="apology is fine — tightening the subject line before it goes out",
        edited_args={"subject": "Our sincere apology for your damaged order A-1042"},
    ),
    Decision(
        verdict="deny", approver="Devanshu",
        reason="refunds over $50 need manager sign-off — withholding pending review",
    ),
]


def _rule(title: str) -> None:
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


def _on_pause(req: ApprovalRequest) -> None:
    print("\n" + "⏸ " * 20)
    print(f"⏸  PAUSE — step {req.step}: agent wants to call `{req.tool}` and is asking a human.")
    print(f"   args    : {req.args}")
    print(f"   why ask : {'; '.join(req.verdict.signals)}")
    print(f"   model reason: {req.reason}")


def _on_step(o: StepOutcome) -> None:
    if not o.requires_approval:
        print(f"\n✅ step {o.step} `{o.action.tool}` — SAFE, auto-approved "
              f"(confidence {o.action.confidence:.2f}) → {o.result.as_line()}")
    elif o.executed and o.result:
        tag = "APPROVED (with edit)" if o.decision.verdict == "edit" else "APPROVED"
        print(f"▶️  step {o.step} `{o.action.tool}` — {tag} by {o.decision.approver} → RESUMED → {o.result.as_line()}")
    else:
        esc = " [escalated — no responder]" if o.escalated else ""
        print(f"🚫 step {o.step} `{o.action.tool}` — DENIED by {o.decision.approver}{esc} → not run "
              f"(“{o.decision.reason}”)")


def _report(res: RunResult, audit: AuditLog) -> None:
    _rule("AGENT'S FINAL REPLY (model call #2 — carries the human decisions forward)")
    print(res.final_reply)

    _rule("SIDE EFFECTS — what actually touched the outside world")
    print(f"📧 outbox: {len(tools.OUTBOX)} email(s) sent")
    for m in tools.OUTBOX:
        print(f"   → to {m['to']} — “{m['subject']}”")
    print(f"💸 refunds ledger: {len(tools.REFUNDS)} refund(s) issued  "
          f"(the $60 refund was DENIED, so this is empty — money was NOT moved)")
    for r in tools.REFUNDS:
        print(f"   → ${r['amount']:.2f} on {r['order_id']}")

    _rule(f"FULL AUDIT TRAIL (durable JSONL → {os.path.basename(AUDIT_PATH)})")
    print(audit.render())
    print(f"\n{len(audit.entries)} events written to {AUDIT_PATH}")


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "scripted"
    if mode not in ("scripted", "interactive"):
        print("usage: python 06-human-in-the-loop/run.py [scripted|interactive]")
        return 2

    _rule(f"HUMAN-IN-THE-LOOP APPROVAL AGENT — {mode.upper()} mode")
    print("Task:", TASK)
    print("\nGate policy: high-risk OR sensitive-capability OR model-confidence < 0.60 → PAUSE for a human.")

    gate = ApprovalGate(min_confidence=0.60)
    audit = AuditLog(AUDIT_PATH, reset=True)
    source = (InteractiveDecisionSource() if mode == "interactive"
              else ScriptedDecisionSource(SCRIPTED_DECISIONS))
    if mode == "scripted":
        print("Scripted decisions: [1] email → APPROVE (edit subject) · [2] refund → DENY")

    agent = HITLAgent(tools.all_tools(), gate, audit, source,
                      on_pause=_on_pause, on_step=_on_step)
    res = agent.run(TASK)
    _report(res, audit)

    _rule("THE FOUR SUB-POINTS, IN THIS RUN")
    print("1. uncertainty detection — each step got a risk verdict from declared risk + "
          "sensitive-capability + model confidence (see the [risk_verdict] audit lines).")
    print("2. pause for human input — send_email and issue_refund SUSPENDED and asked a human "
          "(scripted here, live at the CLI in interactive mode).")
    print("3. resume with validated context — the email RESUMED with the human's edited "
          "subject merged in; the denial was carried forward into the final reply.")
    print("4. full audit trail — every proposal, verdict, decision, and outcome is in "
          f"{os.path.basename(AUDIT_PATH)} with a monotonic seq counter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
