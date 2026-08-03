"""Human-in-the-Loop Approval Agent — pauses on risky actions, resumes on a human OK.

The MODEL does exactly two things: (1) PLAN the task into proposed tool actions and
SELF-RATE its confidence in each (the uncertainty signal), and (2) write the FINAL
reply once the outcomes — including what a human approved or denied — are known.

Everything that DECIDES or RECORDS is deterministic Python in `approval.py`:
the risk gate, the approve/deny/edit surface, and the JSONL audit trail.

The loop per proposed action:

  ASSESS   — the gate turns (declared risk + sensitive-capability + model confidence)
             into a verdict: auto-approve, or NEEDS APPROVAL.
  RUN or PAUSE
             — auto  → execute immediately.
             — needs approval → SUSPEND: emit an ApprovalRequest, ask the DecisionSource,
               and RESUME on the answer: approve/edit → execute (merging any human arg
               edits), deny → refuse and carry on. No responder → escalate (safe-default
               deny). Every branch is written to the audit trail.

  CARRY    — the outcome (executed / refused, plus who decided and why) is appended to a
             running context that is handed to the final-reply model call — so the human
             decision is carried forward into what the agent ultimately says.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from common.client import DEFAULT_MODEL, get_client
from approval import ApprovalGate, ApprovalRequest, AuditLog, Decision, DecisionSource
from tools import Tool, ToolResult, catalog_for_prompt

logger = logging.getLogger("hitl-agent")


@dataclass
class ProposedAction:
    tool: str
    args: dict
    reason: str
    confidence: float


@dataclass
class StepOutcome:
    step: int
    action: ProposedAction
    requires_approval: bool
    executed: bool
    result: ToolResult | None
    decision: Decision | None       # None for the auto path
    escalated: bool = False


@dataclass
class RunResult:
    task: str
    plan: list[ProposedAction]
    outcomes: list[StepOutcome] = field(default_factory=list)
    final_reply: str = ""


_PLAN_SYSTEM = (
    "You are an operations agent that plans a task as a short sequence of TOOL calls, "
    "then a human-in-the-loop gate decides which calls need approval. Return STRICT JSON "
    "and nothing else: {\"plan\": [ {\"tool\": <name>, \"args\": {..}, \"reason\": <str>, "
    "\"confidence\": <0..1> } , ... ]}.\n"
    "Rules:\n"
    "• Use ONLY the tools listed; use their exact arg names.\n"
    "• Order matters — look things up BEFORE you act on them.\n"
    "• `confidence` is how sure you are this exact action + arguments are correct and "
    "SAFE to run without a human checking first (1.0 = certain; lower it when you are "
    "guessing a value like an amount or an address).\n"
    "• Do NOT decide approvals yourself and do NOT skip risky steps — propose the action "
    "you believe the task needs; the gate will pause the risky ones for a human.\n"
    "• No prose, no markdown fences — JSON only."
)

_REPLY_SYSTEM = (
    "You are the same operations agent, now writing the final note to your human operator "
    "after a human-in-the-loop gate ran your plan. You are given, per step, what was "
    "EXECUTED versus what a human REFUSED (with their reason). Write a short, honest "
    "summary: what got done, what was withheld and why, and the single next step you "
    "recommend. Never claim a refused action happened. Be concise — 3-5 sentences."
)


class HITLAgent:
    def __init__(self, tools: dict[str, Tool], gate: ApprovalGate, audit: AuditLog,
                 decisions: DecisionSource, model: str = DEFAULT_MODEL,
                 on_pause=None, on_step=None) -> None:
        self.tools = tools
        self.gate = gate
        self.audit = audit
        self.decisions = decisions
        self.model = model
        self.client = get_client()
        self.on_pause = on_pause          # optional UI callback(ApprovalRequest)
        self.on_step = on_step            # optional UI callback(StepOutcome)

    # ------------------------------------------------------------------ #
    # The model call #1 — PLAN + self-rated confidence (parse, retry once).
    # ------------------------------------------------------------------ #
    def plan(self, task: str) -> list[ProposedAction]:
        user = (f"Tools available:\n{catalog_for_prompt()}\n\n"
                f"Task:\n{task}\n\nReturn the JSON plan.")
        messages = [{"role": "system", "content": _PLAN_SYSTEM},
                    {"role": "user", "content": user}]
        for attempt in (1, 2):
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0)
            raw = (resp.choices[0].message.content or "").strip()
            try:
                actions = self._parse_plan(raw)
                if actions:
                    return actions
                err = "empty plan"
            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                err = f"{type(exc).__name__}: {exc}"
            logger.info("plan parse failed (attempt %d): %s", attempt, err)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user",
                             "content": f"That was not valid ({err}). Return ONLY the JSON object."})
        raise RuntimeError("model did not return a usable plan after 2 attempts")

    def _parse_plan(self, raw: str) -> list[ProposedAction]:
        text = raw.strip()
        if text.startswith("```"):                      # tolerate ```json fences
            text = text.split("```")[1] if "```" in text[3:] else text[3:]
            text = text.split("\n", 1)[1] if text.lower().startswith("json") else text
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]
        data = json.loads(text)
        out: list[ProposedAction] = []
        for item in data.get("plan", []):
            name = item["tool"]
            if name not in self.tools:                  # ignore hallucinated tools
                logger.info("dropping unknown tool in plan: %r", name)
                continue
            out.append(ProposedAction(
                tool=name,
                args=dict(item.get("args", {})),
                reason=str(item.get("reason", "")).strip(),
                confidence=max(0.0, min(1.0, float(item.get("confidence", 0.5)))),
            ))
        return out

    # ------------------------------------------------------------------ #
    # The model call #2 — final reply, given the executed/refused outcomes.
    # ------------------------------------------------------------------ #
    def _final_reply(self, task: str, outcomes: list[StepOutcome]) -> str:
        lines = []
        for o in outcomes:
            if o.executed and o.result:
                who = f" (approved by {o.decision.approver})" if o.decision else " (auto)"
                lines.append(f"EXECUTED{who}: {o.action.tool} → {o.result.as_line()}")
            else:
                why = o.decision.reason if o.decision else "escalated — no responder"
                who = o.decision.approver if o.decision else "auto-escalation"
                lines.append(f"REFUSED by {who}: {o.action.tool}({json.dumps(o.action.args, ensure_ascii=False)}) — {why}")
        context = "Task:\n" + task + "\n\nWhat actually happened, per step:\n" + "\n".join(lines)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": _REPLY_SYSTEM},
                      {"role": "user", "content": context}],
            temperature=0.2)
        return (resp.choices[0].message.content or "").strip()

    # ------------------------------------------------------------------ #
    # The main loop.
    # ------------------------------------------------------------------ #
    def run(self, task: str) -> RunResult:
        plan = self.plan(task)
        logger.info("planned %d action(s): %s", len(plan),
                    [(a.tool, round(a.confidence, 2)) for a in plan])
        result = RunResult(task=task, plan=plan)

        for i, action in enumerate(plan, start=1):
            tool = self.tools[action.tool]
            self.audit.proposed(i, action.tool, action.args, action.reason, action.confidence)
            verdict = self.gate.assess(tool, action.args, action.confidence)
            self.audit.risk_verdict(i, action.tool, verdict)

            if not verdict.requires_approval:
                # ---- AUTO path: safe, confident, non-sensitive -> just run.
                self.audit.auto_approved(i, action.tool)
                res = tool.run(action.args)
                self.audit.executed(i, action.tool, action.args, res)
                outcome = StepOutcome(i, action, False, res.ok, res, None)
            else:
                # ---- PAUSE path: SUSPEND and ask a human.
                request = ApprovalRequest(step=i, tool=action.tool, args=action.args,
                                          reason=action.reason, verdict=verdict)
                self.audit.paused(i, action.tool, verdict.signals)
                if self.on_pause:
                    self.on_pause(request)
                decision = self.decisions.decide(request)

                escalated = decision is None
                if escalated:                             # nobody responded -> safe default
                    decision = Decision(verdict="deny", approver="auto-escalation",
                                        reason="no responder — safe default is to withhold")
                self.audit.human_decision(i, action.tool, decision, escalated=escalated)

                if decision.is_go:
                    # ---- RESUME with validated (optionally human-edited) context.
                    run_args = dict(action.args)
                    if decision.verdict == "edit" and decision.edited_args:
                        run_args.update(decision.edited_args)   # merge the human's patch
                    res = tool.run(run_args)
                    self.audit.executed(i, action.tool, run_args, res)
                    outcome = StepOutcome(i, action, True, res.ok, res, decision, escalated)
                else:
                    # ---- DENY: record the refusal, carry on gracefully.
                    self.audit.refused(i, action.tool, decision.reason)
                    outcome = StepOutcome(i, action, True, False, None, decision, escalated)

            result.outcomes.append(outcome)
            if self.on_step:
                self.on_step(outcome)

        result.final_reply = self._final_reply(task, result.outcomes)
        self.audit.final_response(result.final_reply)
        return result
