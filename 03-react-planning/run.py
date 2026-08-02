"""Runnable demo for the ReAct Planning Agent.

Two scenarios, both hitting NVIDIA NIM:

  DEMO A — a genuine multi-step task that needs the loop:
      look up two facts about a fictional company, then compute a result from
      them. Shows Thought -> Action -> Observation -> Reflect across several
      iterations, ending in a SOLVED final answer.

  DEMO B — graceful degradation under a tool outage:
      the same style of task but the answer needs a fact only obtainable via
      web_search, and we flip the (mock) search backend into an OUTAGE so every
      call raises. Shows the agent catching tool ERRORs, the self-critic flagging
      it off-track, and a clean degrade (DEGRADED partial answer, or the
      MAX_STEPS cap as the backstop) instead of a crash or a hallucinated success.

Run:  python 03-react-planning/run.py
"""

from __future__ import annotations

import logging
import os
import sys

# Make `common` (repo root) and this project's modules importable regardless of cwd.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _REPO_ROOT)

# Windows consoles default to cp1252 and choke on the ✓/→/⚠ glyphs below.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

from agent import DEGRADED, MAX_STEPS, SOLVED, ReActAgent, ReActResult  # noqa: E402
import tools  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    stream=sys.stdout,
)

_STATUS_BADGE = {SOLVED: "SOLVED ✓", DEGRADED: "DEGRADED ⚠", MAX_STEPS: "MAX_STEPS ⛔"}


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _show(res: ReActResult) -> None:
    print(f"\nGoal   : {res.goal}")
    print(f"model  : {res.model}   max_steps: {res.max_steps}")
    for s in res.steps:
        print(f"\n  --- Step {s.n} " + "-" * 60)
        print(f"  Thought     : {s.thought}")
        print(f"  Action      : {s.action}")
        print(f"  Action Input: {s.action_input}")
        ok = "ok" if s.observation_ok else "ERROR"
        print(f"  Observation : [{ok}] {s.observation}")
        print(f"  Reflect     : on_track={s.on_track} — {s.reflection}")
        if s.next_hint:
            print(f"                next: {s.next_hint}")
    print("\n  " + "=" * 72)
    print(f"  STATUS : {_STATUS_BADGE.get(res.status, res.status)}")
    print(f"  reason : {res.reason}")
    print(f"  ANSWER : {res.final_answer}")
    print(f"  steps  : {res.iterations}/{res.max_steps}    elapsed: {res.elapsed_s}s")


def main() -> int:
    _rule("ReAct PLANNING AGENT — demo run (NVIDIA NIM)")

    # ----------------------------------------------------------------- #
    # DEMO A — solvable multi-step task (look up x2, then compute).
    # ----------------------------------------------------------------- #
    tools.set_search_outage(False)
    _rule("DEMO A — multi-step task (expect the loop to SOLVE it)")
    agent = ReActAgent(max_steps=6)
    goal_a = (
        "For the fictional company Zephyr Labs, what was the TOTAL amount spent on "
        "launching satellites? Use the knowledge base to find how many satellites were "
        "launched and the cost per satellite (already given in millions of USD), then "
        "multiply them. Answer in millions of USD."
    )
    _show(agent.run(goal_a))

    # ----------------------------------------------------------------- #
    # DEMO B — tool outage -> graceful degradation.
    # The needed fact is NOT in the knowledge base; it would require web_search,
    # which we force into a simulated 503 outage. The agent must degrade cleanly.
    # ----------------------------------------------------------------- #
    tools.set_search_outage(True)
    _rule("DEMO B — search backend OUTAGE (expect graceful degradation, not a crash)")
    agent_b = ReActAgent(max_steps=5)
    goal_b = (
        "What is the current live market share of Zephyr Labs in the global satellite "
        "industry this quarter? Find the figure using web search, then state it."
    )
    _show(agent_b.run(goal_b))
    tools.set_search_outage(False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
