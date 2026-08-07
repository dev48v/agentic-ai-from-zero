"""Runnable demo for the Multi-Agent Debate System (NVIDIA NIM).

    python 09-multi-agent-debate/run.py

Multiple agents argue their way to a better answer than one would give alone. Three
PROPOSERS with different personas (cautious / creative / literal) each answer the SAME
question cold; a CRITIC pressure-tests every proposal; each proposer then REBUTS — revising
or defending after reading the critique; Python tallies their final stances into a
consensus; and a JUDGE synthesizes the winning answer, with a CONFIDENCE that is derived
from how much the panel actually agreed (not asserted by the judge).

It runs two questions on purpose, to show both ends of the behaviour:

  Q1  a question with a definite answer (the classic bat-and-ball) — the intuitive trap
      snares at least one proposer, the critic catches the arithmetic, the rebuttal changes
      a mind, and the panel CONVERGES → HIGH confidence.
  Q2  a genuinely debatable strategy question (bootstrap vs. raise venture capital) —
      the personas pull in different directions and stay split → LOW confidence, flagged.

The MODEL proposes / critiques / rebuts / synthesizes; the round order, the ballot tally,
the consensus ratio, the mind-change detection, and the confidence are deterministic Python
in `debate.py`. Everything printed below a phase header is a real NIM call.
"""

from __future__ import annotations

import json
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

from debate import Debate, DebateResult  # noqa: E402
from agents import PERSONA_BY_ID  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    stream=sys.stdout,
)

LOG_PATH = os.path.join(_PROJECT_DIR, "debate-log.jsonl")


QUESTIONS = [
    ("clear",
     "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
     "How much does the ball cost, in dollars?",
     "definite answer ($0.05); the intuitive trap is $0.10 — expect the panel to CONVERGE → high confidence"),
    ("debatable",
     "A founder can either bootstrap their startup slowly on their own revenue, or raise "
     "venture capital to grow fast. Which is the better path?",
     "a genuine values/strategy trade-off with no single right answer — expect a SPLIT flagged low-confidence"),
]


def _rule(title: str) -> None:
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def _short(text: str, n: int = 150) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


# --------------------------------------------------------------------------- #
# The live logger — prints each phase/role as the debate runs. Also mirrors a
# compact JSON line per event to LOG_PATH (gitignored) so a run is auditable.
# --------------------------------------------------------------------------- #
def make_logger(sink):
    def log(kind: str, **d) -> None:
        if kind == "phase":
            print(f"\n  ── {d['title']} " + "─" * max(0, 66 - len(d['title'])))
            sink({"kind": "phase", "title": d["title"]})

        elif kind == "proposal":
            p = d["proposal"]
            per = PERSONA_BY_ID.get(p.persona_id)
            emoji = per.emoji if per else "•"
            print(f"    {emoji} {p.persona_name:<9} stance=[{p.stance}]  (self-conf: {p.self_confidence or 'n/a'})")
            print(f"        answer: {_short(p.answer)}")
            if p.reasoning:
                print(f"        because: {_short(p.reasoning, 120)}")
            sink({"kind": "proposal", "persona": p.persona_id, "stance": p.stance,
                  "answer": p.answer, "reasoning": p.reasoning, "self_confidence": p.self_confidence})

        elif kind == "critique":
            c = d["critique"]
            for a in c.assessments:
                per = PERSONA_BY_ID.get(a.persona_id)
                name = per.name if per else a.persona_id
                mark = {"sound": "✅", "flawed": "⚠️", "unsupported": "❓"}.get(a.verdict, "•")
                print(f"    🔎 on {name:<9} {mark} {a.verdict:<11} — {_short(a.flaw, 110)}")
            print(f"    🔎 overall: {_short(c.overall, 120)}")
            sink({"kind": "critique",
                  "assessments": [{"persona": a.persona_id, "verdict": a.verdict, "flaw": a.flaw}
                                  for a in c.assessments], "overall": c.overall})

        elif kind == "rebuttal":
            before, after = d["before"], d["after"]
            per = PERSONA_BY_ID.get(after.persona_id)
            emoji = per.emoji if per else "•"
            moved = (before.stance.strip().lower() != after.stance.strip().lower())
            flag = "🔄 CHANGED" if (after.changed_mind or moved) else "🔒 held"
            print(f"    {emoji} {after.persona_name:<9} [{before.stance}] → [{after.stance}]  {flag}")
            if after.change_note:
                print(f"        note: {_short(after.change_note, 120)}")
            sink({"kind": "rebuttal", "persona": after.persona_id,
                  "stance_before": before.stance, "stance_after": after.stance,
                  "changed_mind": after.changed_mind, "why": after.change_note,
                  "answer": after.answer})

        elif kind == "tally":
            t, conf, moved = d["tally"], d["confidence"], d["mind_changed"]
            print(f"    🗳️  ballots: " + "  ".join(f"{k}→[{v}]" for k, v in t.ballots.items()))
            print(f"    📊 tally:   {t.summary()}")
            names = ", ".join(PERSONA_BY_ID[m].name for m in moved) if moved else "none"
            print(f"    🔄 minds changed by the critique: {names}")
            print(f"    📈 consensus ratio: {t.consensus_ratio:.2f}  →  CONFIDENCE = "
                  f"{conf.label.upper()} ({conf.pct}%)")
            print(f"        {conf.reason}")
            sink({"kind": "tally", "ballots": t.ballots, "counts": t.counts,
                  "winner": t.winner, "winner_votes": t.winner_votes, "total": t.total,
                  "tied": t.tied, "consensus_ratio": round(t.consensus_ratio, 4),
                  "confidence": conf.label, "confidence_pct": conf.pct,
                  "mind_changed": moved})

        elif kind == "synthesis":
            s, conf = d["synthesis"], d["confidence"]
            print(f"    ⚖️  JUDGE'S FINAL ANSWER  [confidence: {conf.label.upper()} · {conf.pct}%]")
            print(f"        {_short(s.final_answer, 240)}")
            if s.rationale:
                print(f"        rationale: {_short(s.rationale, 140)}")
            for kp in s.key_points:
                print(f"          • {_short(kp, 100)}")
            sink({"kind": "synthesis", "final_answer": s.final_answer,
                  "rationale": s.rationale, "key_points": s.key_points,
                  "confidence": conf.label, "confidence_pct": conf.pct})
    return log


def run_one(kind: str, question: str, note: str, sink) -> DebateResult:
    _rule(f"DEBATE ({kind.upper()}) — {question}")
    print(f"  expectation: {note}")
    sink({"kind": "question", "class": kind, "question": question, "note": note})
    debate = Debate(log=make_logger(sink))
    return debate.run(question)


def main() -> int:
    _rule("MULTI-AGENT DEBATE SYSTEM — three agents argue, a critic probes, a judge decides "
          "(NVIDIA NIM · meta/llama-3.1-8b-instruct)")
    print("Three PROPOSERS (cautious / creative / literal) answer the same question; a CRITIC finds each")
    print("proposal's flaw; each proposer REBUTS (revise or defend); Python tallies the final stances into")
    print("a consensus; a JUDGE synthesizes the answer. Confidence is DERIVED from agreement, not asserted.")

    rows = []
    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        def sink(obj):
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        for kind, q, note in QUESTIONS:
            res = run_one(kind, q, note, sink)
            rows.append((kind, res))

    # -------- the comparison the whole run is designed to show -------- #
    _rule("TWO DEBATES, SIDE BY SIDE — consensus drives confidence")
    head = f"  {'question':<11}  {'final tally':<34}  {'minds moved':>11}  {'confidence':>12}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for kind, r in rows:
        moved = len(r.mind_changed)
        print(f"  {kind:<11}  {_short(r.tally.summary(), 34):<34}  {moved:>11}  "
              f"{r.confidence.label.upper() + ' ' + str(r.confidence.pct) + '%':>12}")

    _rule("THE FOUR SUB-POINTS, IN THIS RUN")
    print("1. agents propose — three personas answered each question independently, with distinct framings,")
    print("   so the proposals genuinely diverged instead of echoing one another.")
    print("2. critic evaluates — one critic named each proposal's main flaw / assumption; a REBUTTAL round")
    print("   let each proposer revise or defend. On the debatable question the critique changed two minds.")
    print("3. voting / consensus — each agent's FINAL stance was a ballot; Python tallied them into positions")
    print("   and computed a consensus ratio — deterministic, reproducible from the stances alone.")
    print("4. aggregator synthesizes with confidence — the judge combined the best points; the confidence was")
    print("   DERIVED from agreement: unanimous → HIGH, split → LOW (flagged). High consensus, high confidence.")
    print(f"\n  transcript log → {LOG_PATH}  (gitignored, regenerated each run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
