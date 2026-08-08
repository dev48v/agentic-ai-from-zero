"""Runnable demo for the Self-Reflective Agent (NVIDIA NIM).

    python 10-self-reflective/run.py

An agent that grades and improves its OWN work. For each task it: GENERATES a fast first-
pass draft; REFLECTS on it with an LLM-as-judge that scores the draft against an explicit
RUBRIC (each criterion 1-5, with evidence + actionable critique); then, if the answer
misses the quality gate, REFINES it under the judge's constraints and re-scores — looping,
bounded, until it passes or hits the iteration cap. Every per-iteration score, what changed,
the improvement delta, and the stop reason are logged.

It runs two tasks on purpose, both with a genuinely mediocre first draft:

  T1  write a precise DOCSTRING (with edge cases) for a remainder-splitting function — the
      quick draft is a one-liner with no Args/Returns/Raises and no edge cases; reflection
      has to add them.
  T2  answer a TRICKY QUESTION completely (two successive discounts, 20% then 10%) — the
      quick answer often falls for the additive "30%" trap or gives 28% with no working;
      reflection has to fix the number and/or complete the reasoning.

The MODEL drafts / judges / refines; the aggregation, the deterministic HARD CHECKS, the
quality GATE (rubric.py), and the loop control + improvement metrics (agent.py) are plain
Python. Everything printed below a phase header is a real NIM call.
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

from agent import ReflectionLoop, ReflectionResult  # noqa: E402
from rubric import TASKS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    stream=sys.stdout,
)

LOG_PATH = os.path.join(_PROJECT_DIR, "reflection-log.jsonl")
MAX_ITERS = 3


def _rule(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def _short(text: str, n: int = 150) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _score_bar(cid: str, name: str, score: int) -> str:
    filled = "█" * score + "·" * (5 - score)
    return f"        {name:<22} [{filled}] {score}/5"


# --------------------------------------------------------------------------- #
# The live logger — prints each phase/iteration as the loop runs, and mirrors a
# compact JSON line per event to LOG_PATH (gitignored) so a run is auditable.
# --------------------------------------------------------------------------- #
def make_logger(sink):
    def log(kind: str, **d) -> None:
        if kind == "phase":
            print(f"\n  ── {d['title']} " + "─" * max(0, 70 - len(d['title'])))
            sink({"kind": "phase", "title": d["title"]})

        elif kind == "iteration":
            it, task = d["it"], d["task"]
            ev = it.evaluation
            tag = "DRAFT" if it.kind == "draft" else f"REFINEMENT {it.n}"
            print(f"    ✍️  {tag} answer:")
            for line in _short(it.answer, 320).splitlines() or [""]:
                print(f"        {line}")
            print(f"    ⚖️  JUDGE scores against the rubric (LLM-as-judge on its own output):")
            crit_name = {c.id: c.name for c in task.rubric.criteria}
            for s in ev.scores:
                print(_score_bar(s.criterion_id, crit_name.get(s.criterion_id, s.criterion_id), s.score))
                if s.critique:
                    print(f"            ↳ {_short(s.critique, 104)}")
            hard = "  ".join(f"{cid}={'✅' if ok else '❌'}" for cid, ok in ev.hard_results)
            print(f"    🔒 hard checks (deterministic): {hard}")
            print(f"    📊 overall = {ev.overall:.2f}  ·  gate: {'✅ PASS' if ev.passed else '❌ FAIL'} — {ev.gate_reason}")
            if it.kind == "refine":
                moved = ("↑ " + ", ".join(it.improved)) if it.improved else "none up"
                fell = (" · ↓ " + ", ".join(it.regressed)) if it.regressed else ""
                print(f"    🔧 changed: {_short(it.change_note, 120)}")
                print(f"    🔁 criteria moved vs previous: {moved}{fell}")
            sink({"kind": "iteration", "n": it.n, "type": it.kind, "answer": it.answer,
                  "scores": [{"criterion": s.criterion_id, "score": s.score,
                              "evidence": s.evidence, "critique": s.critique} for s in ev.scores],
                  "overall": ev.overall,
                  "hard_checks": {cid: ok for cid, ok in ev.hard_results},
                  "passed": ev.passed, "gate_reason": ev.gate_reason,
                  "judge_comment": ev.judge_comment,
                  "change_note": it.change_note,
                  "improved": it.improved, "regressed": it.regressed})

        elif kind == "done":
            r, task = d["result"], d["task"]
            trail = " → ".join(f"{s:.2f}" for s in r.score_trail)
            hard_trail = " → ".join(r.hard_trail)
            print(f"\n    🏁 STOP — reason: {r.stop_reason.upper()}  ·  "
                  f"iterations: {len(r.iterations)} (1 draft + {len(r.iterations) - 1} refine)")
            print(f"    📈 improvement metric — rubric score: {trail}   (Δ {r.delta:+.2f})")
            print(f"    🔒 improvement metric — hard checks : {hard_trail}")
            sink({"kind": "done", "task": task.id, "stop_reason": r.stop_reason,
                  "score_trail": r.score_trail, "hard_trail": r.hard_trail, "delta": r.delta,
                  "iterations": len(r.iterations),
                  "final_passed": r.final.evaluation.passed})
    return log


def run_one(task, sink) -> ReflectionResult:
    _rule(f"TASK · {task.title}")
    print(f"  prompt      : {_short(task.prompt, 200)}")
    print(f"  rubric      : {task.rubric.name}  ·  threshold {task.rubric.threshold:.2f}  ·  "
          f"{len(task.rubric.criteria)} criteria  ·  {len(task.rubric.hard_checks)} hard checks")
    print(f"  expectation : {task.note}")
    sink({"kind": "task", "id": task.id, "title": task.title, "prompt": task.prompt,
          "threshold": task.rubric.threshold, "note": task.note})
    loop = ReflectionLoop(max_iters=MAX_ITERS, log=make_logger(sink))
    return loop.solve(task)


def main() -> int:
    _rule("SELF-REFLECTIVE AGENT — it grades and rewrites its own work until it passes a "
          "quality gate  (NVIDIA NIM · meta/llama-3.1-8b-instruct)")
    print("For each task the agent GENERATES a fast draft, REFLECTS on it as an LLM-as-judge scoring an")
    print("explicit rubric (1-5 per criterion + evidence + critique), then REFINES under those constraints")
    print("and re-scores — bounded loop, stops when it clears the gate or hits the cap. The model drafts /")
    print("judges / refines; the aggregation, hard checks, gate, loop control + metrics are deterministic Python.")

    rows = []
    with open(LOG_PATH, "w", encoding="utf-8") as fh:
        def sink(obj):
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        for task in TASKS:
            res = run_one(task, sink)
            rows.append((task, res))

    # -------- the improvement summary the whole run is designed to show -------- #
    _rule("IMPROVEMENT METRICS — the score climbing across revisions until the gate passes")
    head = (f"  {'task':<11}  {'rubric-score trail':<22}  {'Δ':>6}  "
            f"{'hard-check trail':<18}  {'iters':>5}  {'stop reason':>11}")
    print(head)
    print("  " + "-" * (len(head) - 2))
    for task, r in rows:
        trail = " → ".join(f"{s:.2f}" for s in r.score_trail)
        htrail = " → ".join(r.hard_trail)
        print(f"  {task.id:<11}  {trail:<22}  {r.delta:>+6.2f}  {htrail:<18}  "
              f"{len(r.iterations):>5}  {r.stop_reason.upper():>11}")

    # Honest, data-driven note: flag any task where the LLM score barely moved but the
    # deterministic hard checks did — i.e. the hard checks, not the judge, drove the fix.
    for task, r in rows:
        hp0, hpN = r.first.evaluation.hard_passed, r.final.evaluation.hard_passed
        if abs(r.delta) < 0.05 and hpN > hp0:
            print(f"\n  note: on '{task.id}' the LLM judge barely moved the rubric score (Δ {r.delta:+.2f} —")
            print(f"  over-generous, a known self-judge bias), yet the DETERMINISTIC hard-check trail rose")
            print(f"  {hp0}/{r.first.evaluation.hard_total} → {hpN}/{r.final.evaluation.hard_total}: the hard checks, not the judge, caught the gap and drove the")
            print(f"  gate from FAIL to PASS. This is exactly why the gate does not trust the LLM score alone.")

    _rule("THE FOUR SUB-POINTS, IN THIS RUN")
    print("1. execute + evaluate via LLM-as-judge — the agent produced an answer, then a JUDGE scored it")
    print("   against an explicit rubric (1-5 per criterion), returning evidence + a number per criterion.")
    print("2. critic reasoning — the judge returned specific, actionable critique (what's wrong / missing /")
    print("   how to fix), not just a score; that critique is what the refinement round consumes.")
    print("3. regenerate with constraints — below-gate answers were regenerated to keep the good and fix the")
    print("   named gaps, bounded by max_iters; the loop stopped on the deterministic quality gate.")
    print("4. log improvement metrics — per-iteration overall score, which criteria moved, the final delta,")
    print("   and the stop reason (passed / max-iters) were all recorded above.")
    print(f"\n  transcript log → {LOG_PATH}  (gitignored, regenerated each run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
