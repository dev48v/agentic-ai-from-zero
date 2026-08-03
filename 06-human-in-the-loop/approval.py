"""The deterministic half of the agent — risk verdict, the approval gate, the
decision surface, and a durable audit trail. NO model calls live here on purpose.

Four ideas, one per sub-point:

  1. uncertainty detection  -> `ApprovalGate.assess`: combine a tool's DECLARED risk,
                               a SENSITIVE-capability check, and the model's self-rated
                               confidence into one verdict. ANY of the three trips it.
  2. pause for human input   -> `ApprovalRequest` + `DecisionSource`: when the verdict
                               requires approval the agent SUSPENDS and asks a source
                               for an approve / deny / edit `Decision`. Two sources:
                               interactive (CLI) and scripted (reproducible).
  3. resume with context     -> a `Decision` can carry `edited_args` (a human patch),
                               which the agent merges before executing on resume.
  4. full audit trail        -> `AuditLog`: every proposal, verdict, decision, and
                               outcome is appended to a JSONL file with a MONOTONIC
                               sequence counter (no wall clock) and printed readably.

The split is the whole point: the model proposes and rates; this file DECIDES and
RECORDS. The gate is pure Python, so the same proposal always earns the same verdict —
auditable and reproducible.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol

# Capabilities that ALWAYS require a human, even if a tool forgot to mark itself
# high-risk. Defence in depth: risk is declared per-tool AND checked per-capability.
SENSITIVE_CAPABILITIES = frozenset({"send_email", "spend_money", "delete", "external_write"})


# --------------------------------------------------------------------------- #
# 1. Uncertainty detection — the risk verdict.
# --------------------------------------------------------------------------- #
@dataclass
class RiskVerdict:
    requires_approval: bool
    risk: str                     # the tool's declared risk ("safe" | "high")
    confidence: float             # the model's self-rated confidence for THIS action
    signals: list[str] = field(default_factory=list)  # which rules fired, human-readable

    @property
    def headline(self) -> str:
        return "NEEDS APPROVAL" if self.requires_approval else "auto-approve"


class ApprovalGate:
    """Turns (tool, args, model-confidence) into a RiskVerdict — deterministically.

    Three independent signals can each demand a human:
      • the tool DECLARES risk="high";
      • the tool touches a SENSITIVE capability (send_email / spend_money / …);
      • the model's own confidence is BELOW `min_confidence` (it is uncertain).
    A safe, high-confidence, non-sensitive action is the only thing that auto-runs.
    """

    def __init__(self, min_confidence: float = 0.6) -> None:
        self.min_confidence = min_confidence

    def assess(self, tool, args: dict, confidence: float) -> RiskVerdict:  # noqa: ARG002
        signals: list[str] = []
        if tool.risk == "high":
            signals.append(f"tool '{tool.name}' declares risk=high")
        sensitive = tool.capabilities & SENSITIVE_CAPABILITIES
        if sensitive:
            signals.append(f"touches sensitive capability {sorted(sensitive)}")
        if confidence < self.min_confidence:
            signals.append(
                f"model confidence {confidence:.2f} < {self.min_confidence:.2f} — uncertain"
            )
        return RiskVerdict(
            requires_approval=bool(signals),
            risk=tool.risk,
            confidence=confidence,
            signals=signals,
        )


# --------------------------------------------------------------------------- #
# 2. Pause for human input — the request + the decision + the sources.
# --------------------------------------------------------------------------- #
@dataclass
class ApprovalRequest:
    """Emitted when the agent SUSPENDS: what it wants to do, and why it must ask."""
    step: int
    tool: str
    args: dict
    reason: str                   # the model's stated why for this action
    verdict: RiskVerdict


@dataclass
class Decision:
    """A human (or an escalation) answering an ApprovalRequest.

    verdict = "approve" | "edit" | "deny".  "edit" is an approve carrying a PATCH of
    argument overrides in `edited_args`, merged over the proposed args on resume.
    """
    verdict: str
    approver: str = "human"
    reason: str = ""
    edited_args: Optional[dict] = None

    @property
    def is_go(self) -> bool:
        return self.verdict in ("approve", "edit")


class DecisionSource(Protocol):
    def decide(self, request: ApprovalRequest) -> Optional[Decision]:
        """Return a Decision, or None to signal NO responder (→ escalation)."""
        ...


class ScriptedDecisionSource:
    """Pre-baked decisions consumed in order — makes the recorded run reproducible.

    When the queue is empty it returns None: 'no one is left to respond', which the
    agent turns into a safe-default escalation (deny). That's the 'escalation when no
    one responds' behaviour, exercised without a real timeout.
    """

    def __init__(self, decisions: list[Decision]) -> None:
        self._queue = list(decisions)

    def decide(self, request: ApprovalRequest) -> Optional[Decision]:  # noqa: ARG002
        if not self._queue:
            return None
        return self._queue.pop(0)


class InteractiveDecisionSource:
    """Reads a real approve / deny / edit decision from the CLI."""

    def decide(self, request: ApprovalRequest) -> Optional[Decision]:
        print("\n" + "!" * 74)
        print(f"⏸  APPROVAL REQUIRED — step {request.step}: {request.tool}")
        print(f"   why: {', '.join(request.verdict.signals)}")
        print(f"   args: {json.dumps(request.args, ensure_ascii=False)}")
        print("!" * 74)
        who = input("   your name (approver): ").strip() or "human"
        choice = input("   [a]pprove / [d]eny / [e]dit ? ").strip().lower()
        if choice.startswith("e"):
            raw = input("   JSON patch of args to override: ").strip()
            try:
                patch = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                print("   (unparseable patch — treating as plain approve)")
                patch = {}
            reason = input("   reason: ").strip()
            return Decision(verdict="edit", approver=who, reason=reason, edited_args=patch)
        if choice.startswith("d"):
            reason = input("   reason for denial: ").strip()
            return Decision(verdict="deny", approver=who, reason=reason)
        reason = input("   reason: ").strip()
        return Decision(verdict="approve", approver=who, reason=reason)


# --------------------------------------------------------------------------- #
# 4. Full audit trail — durable JSONL with a monotonic counter.
# --------------------------------------------------------------------------- #
class AuditLog:
    """Append-only audit trail. Every event gets a monotonic `seq` (no wall clock —
    a step counter, so the trail is reproducible), is written as ONE JSON line to a
    durable `.jsonl` file, and is also kept in memory for a readable printout.
    """

    def __init__(self, path: str, reset: bool = True) -> None:
        self.path = path
        self.entries: list[dict] = []
        self._seq = 0
        if reset and os.path.exists(path):
            os.remove(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def record(self, event: str, step: int, **fields) -> dict:
        self._seq += 1
        entry = {"seq": self._seq, "step": step, "event": event, **fields}
        self.entries.append(entry)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    # convenience recorders (each maps to one thing that happened) --------- #
    def proposed(self, step, tool, args, reason, confidence):
        return self.record("proposed", step, tool=tool, args=args,
                            reason=reason, confidence=confidence)

    def risk_verdict(self, step, tool, verdict: RiskVerdict):
        return self.record("risk_verdict", step, tool=tool,
                           requires_approval=verdict.requires_approval,
                           risk=verdict.risk, confidence=verdict.confidence,
                           signals=verdict.signals)

    def auto_approved(self, step, tool):
        return self.record("auto_approved", step, tool=tool)

    def paused(self, step, tool, signals):
        return self.record("paused", step, tool=tool, why=signals)

    def human_decision(self, step, tool, decision: Decision, escalated=False):
        return self.record("human_decision", step, tool=tool,
                           decision=decision.verdict, approver=decision.approver,
                           reason=decision.reason, edited_args=decision.edited_args,
                           escalated=escalated)

    def executed(self, step, tool, args, result):
        return self.record("executed", step, tool=tool, args=args,
                           ok=result.ok, output=result.as_line())

    def refused(self, step, tool, reason):
        return self.record("refused", step, tool=tool, reason=reason)

    def final_response(self, text):
        return self.record("final_response", 0, text=text)

    # ------------------------------------------------------------------- #
    def render(self) -> str:
        """A human-readable rendering of the whole trail (icons per event)."""
        icon = {
            "proposed": "📝", "risk_verdict": "⚖️ ", "auto_approved": "✅",
            "paused": "⏸ ", "human_decision": "🧑", "executed": "▶️ ",
            "refused": "🚫", "final_response": "💬",
        }
        out = []
        for e in self.entries:
            head = f"#{e['seq']:<2} [{icon.get(e['event'], '·')}{e['event']}]"
            if e["event"] == "proposed":
                out.append(f"{head} step {e['step']}: {e['tool']} "
                           f"args={json.dumps(e['args'], ensure_ascii=False)} "
                           f"(model confidence {e['confidence']:.2f})")
            elif e["event"] == "risk_verdict":
                tag = "NEEDS APPROVAL" if e["requires_approval"] else "auto-approve"
                out.append(f"{head} step {e['step']}: {e['tool']} → {tag} "
                           f"(risk={e['risk']}); signals: {'; '.join(e['signals']) or 'none'}")
            elif e["event"] == "auto_approved":
                out.append(f"{head} step {e['step']}: {e['tool']} — safe, ran without asking")
            elif e["event"] == "paused":
                out.append(f"{head} step {e['step']}: {e['tool']} — SUSPENDED, awaiting a human")
            elif e["event"] == "human_decision":
                extra = f" edited_args={json.dumps(e['edited_args'], ensure_ascii=False)}" if e.get("edited_args") else ""
                esc = " (ESCALATED — no responder)" if e.get("escalated") else ""
                out.append(f"{head} step {e['step']}: {e['tool']} — {e['decision'].upper()} "
                           f"by {e['approver']}{esc} — “{e['reason']}”{extra}")
            elif e["event"] == "executed":
                out.append(f"{head} step {e['step']}: {e['tool']} → {'OK' if e['ok'] else 'FAIL'}: {e['output']}")
            elif e["event"] == "refused":
                out.append(f"{head} step {e['step']}: {e['tool']} — NOT run: {e['reason']}")
            elif e["event"] == "final_response":
                out.append(f"{head} agent's final reply recorded")
        return "\n".join(out)
